from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from data.market_data_quality import DECISION_ELIGIBLE_OHLCV_VIEW_SQL
from db.writer_lease import require_writer_lease


TARGET_SCHEMA_VERSION = 5
MIN_FREE_HEADROOM_BYTES = 64 * 1024 * 1024
DECISION_ELIGIBLE_OHLCV_REQUIRED_COLUMNS = {
    "symbol": "VARCHAR",
    "date": "DATE",
    "observed_at": "TIMESTAMP",
    "instrument": "VARCHAR",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "close": "DOUBLE",
    "volume": "BIGINT",
    "source": "VARCHAR",
    "quality": "VARCHAR",
    "derivation_type": "VARCHAR",
    "decision_eligible": "BOOLEAN",
}


class MigrationError(RuntimeError):
    pass


class DatabaseOwnedElsewhere(MigrationError):
    """Raised when the database file cannot even be opened read-only because
    another live process holds it exclusively (DuckDB, unlike SQLite, has no
    WAL-mode cross-process concurrent reader; on Windows this surfaces as a
    sharing-violation IOException). This is not a failure: it means another
    process is the live writer authority for this database right now, and
    per the collector's own already-established standalone-start behavior
    (`smart_collector.py` never runs `migrate_database` itself -- it trusts
    that an existing `terminal.db` is already at the schema its `WriterService`
    expects), the same trust applies here: whoever is actively running
    against this file got there through their own successful startup, so
    this database is not stale or unmigrated."""


MIGRATION_2_STATEMENTS = (
    "DROP INDEX IF EXISTS idx_instr_sym",
    "DROP INDEX IF EXISTS idx_instr_name",
    "DROP INDEX IF EXISTS idx_instr_type",
    "DROP INDEX IF EXISTS idx_fo_symbol_expiry",
    "DROP INDEX IF EXISTS idx_fo_sym_date",
    "DROP INDEX IF EXISTS idx_fo_expiry",
    "DROP INDEX IF EXISTS idx_backtest_runs_status",
    "ALTER TABLE instruments ADD COLUMN IF NOT EXISTS schema_version INTEGER DEFAULT 1",
    "ALTER TABLE fo_eod ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'UNKNOWN'",
    "ALTER TABLE fo_eod ADD COLUMN IF NOT EXISTS source_format TEXT DEFAULT 'unknown'",
    "ALTER TABLE fo_eod ADD COLUMN IF NOT EXISTS price_quality TEXT DEFAULT 'UNKNOWN'",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS chronological_segment_return DOUBLE",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS validity_status TEXT DEFAULT 'INVALID_LEGACY_PRICE_SEMANTICS'",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS price_policy_version TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS invalid_reason TEXT",
    "CREATE INDEX IF NOT EXISTS idx_instr_sym ON instruments(symbol, exchange)",
    "CREATE INDEX IF NOT EXISTS idx_instr_name ON instruments(name, exchange)",
    "CREATE INDEX IF NOT EXISTS idx_instr_type ON instruments(instrument_type, exchange)",
    "CREATE INDEX IF NOT EXISTS idx_fo_symbol_expiry ON fo_eod(symbol, expiry)",
    "CREATE INDEX IF NOT EXISTS idx_fo_sym_date ON fo_eod(symbol, date)",
    "CREATE INDEX IF NOT EXISTS idx_fo_expiry ON fo_eod(expiry, symbol)",
    "CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status, started_at)",
)

MIGRATION_3_STATEMENTS = (
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS symbol TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS data_as_of TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS model_version TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS artifact_hash TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS feature_schema_version TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS release_version TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS validation_status TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS fallback_status TEXT",
    "ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS outcome_error REAL",
)

MIGRATION_4_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS ingestion_manifests (
        source_id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        source_identity TEXT NOT NULL,
        expected_scope_json TEXT NOT NULL,
        observed_record_count BIGINT NOT NULL,
        ingestion_completed_at TIMESTAMP NOT NULL,
        validation_completed_at TIMESTAMP NOT NULL,
        completeness_status TEXT NOT NULL,
        downstream_commit_id TEXT NOT NULL,
        retention_decision TEXT NOT NULL DEFAULT 'RETAIN',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingestion_manifest_retention ON ingestion_manifests(retention_decision, validation_completed_at)",
)

# Backfills decision_eligible_ohlcv for databases that reached schema v4 before
# the view existed (schema_initializer only runs on the v0->v1 leg, so an
# already-upgraded database never got it). CREATE OR REPLACE VIEW is also the
# fresh-install/idempotent no-op path when the view already matches contract.
MIGRATION_5_STATEMENTS = (DECISION_ELIGIBLE_OHLCV_VIEW_SQL,)


def migrate_database(
    database: Path,
    *,
    schema_initializer: Callable[[Path], None],
    target_version: int = TARGET_SCHEMA_VERSION,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    copy_file: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any] = shutil.copy2,
    replace_file: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any] = os.replace,
    validate_database: Callable[[Path, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validator = validate_database or _validate_database
    database = database.resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    journal = database.with_suffix(database.suffix + ".migration.json")
    stage = database.with_suffix(database.suffix + ".migration-stage")

    # Exclusive writer access is only required when this call is actually
    # going to mutate the database (recovering a genuinely interrupted prior
    # migration, or performing a real version upgrade) -- reproduced live:
    # every backend startup calls this as a preflight, including when the
    # database is already at the current schema version and nothing will be
    # written, and unconditionally acquiring the lease here made the desktop
    # backend crash-loop outright (WriterLeaseError) whenever the collector
    # already legitimately held writer authority, even though no migration
    # was needed. A completed migration's journal file is kept on disk as an
    # audit trail (never deleted), so "a journal file exists" is not itself
    # evidence that recovery must run -- only `_recover_interrupted`'s own
    # parsed `state` can tell an actually-interrupted migration apart from a
    # long-committed one, so the lease is acquired from inside it, right
    # before its two mutating branches. require_writer_lease() is idempotent
    # for the same process, so calling it again below (once real migration
    # work is confirmed necessary) is a safe no-op if recovery already
    # acquired it.
    _recover_interrupted(database, stage, journal, replace_file)

    try:
        current = _current_version(database) if database.exists() else 0
    except DatabaseOwnedElsewhere:
        # Reproduced live: the desktop backend's startup preflight crashed
        # outright (collector-first -> desktop-second topology) because it
        # could not even open a read-only connection while the collector's
        # WriterService actively held the file -- DuckDB has no cross-process
        # concurrent-reader mode like SQLite's WAL. The active owner is only
        # running because its own earlier startup already resolved schema
        # state, so there is nothing to (re)migrate or validate here.
        return {"current_schema_version": target_version, "target_schema_version": target_version, "migrated": False}
    if current > target_version:
        raise MigrationError(f"database schema {current} is newer than supported target {target_version}")
    if current == target_version:
        validator(database, target_version)
        return {"current_schema_version": current, "target_schema_version": target_version, "migrated": False}

    require_writer_lease(database)
    source_size = database.stat().st_size if database.exists() else 0
    free = int(disk_usage(database.parent).free)
    required = source_size + MIN_FREE_HEADROOM_BYTES
    if free < required:
        raise MigrationError(f"insufficient disk for staged migration: need {required}, have {free}")

    if stage.exists():
        stage.unlink()
    backup = (
        database.with_suffix(database.suffix + f".pre-v{current}.bak")
        if database.exists()
        else None
    )
    if backup is not None and backup.exists():
        raise MigrationError(f"rollback backup already exists; operator review required: {backup}")
    source_fingerprint = _file_fingerprint(database) if database.exists() else None
    _write_journal(journal, "PREPARED", current, target_version, None, backup)

    try:
        if database.exists():
            copy_file(database, stage)
            _flush_file(stage)
            _flush_parent_metadata(stage.parent)
        _write_journal(journal, "STAGED", current, target_version, None, backup)
        version = current
        while version < target_version:
            next_version = version + 1
            if next_version == 1:
                schema_initializer(stage)
                _record_migration(stage, 1, "baseline_schema", "schema_initializer")
            elif next_version == 2:
                _apply_sql_migration(stage, 2, "quote_and_backtest_provenance", MIGRATION_2_STATEMENTS)
            elif next_version == 3:
                _apply_sql_migration(stage, 3, "ml_prediction_attribution", MIGRATION_3_STATEMENTS)
            elif next_version == 4:
                _apply_sql_migration(
                    stage,
                    4,
                    "durable_ingestion_completeness_manifest",
                    MIGRATION_4_STATEMENTS,
                )
            elif next_version == 5:
                _apply_decision_eligible_view_migration(
                    stage,
                    5,
                    "decision_eligible_ohlcv_backfill",
                    MIGRATION_5_STATEMENTS,
                )
            else:
                raise MigrationError(f"no migration path from schema {version} to {next_version}")
            version = next_version
        _flush_file(stage)
        staged_manifest = validator(stage, target_version)
        _flush_file(stage)
        _write_journal(
            journal,
            "VALIDATED",
            current,
            target_version,
            None,
            backup,
            staged_manifest=staged_manifest,
            source_fingerprint=source_fingerprint,
        )
    except Exception as error:
        _write_journal(journal, "ROLLBACK_REQUIRED", current, target_version, str(error), backup)
        if stage.exists():
            stage.unlink()
        _write_journal(journal, "ROLLED_BACK", current, target_version, str(error), backup)
        raise MigrationError(f"staged migration failed: {error}") from error

    try:
        if backup is not None:
            _durable_replace(database, backup, replace_file)
            _write_journal(
                journal,
                "BACKUP_SWAPPED",
                current,
                target_version,
                None,
                backup,
                staged_manifest=staged_manifest,
                source_fingerprint=source_fingerprint,
            )
        _durable_replace(stage, database, replace_file)
        _write_journal(
            journal,
            "SWAPPED",
            current,
            target_version,
            None,
            backup,
            staged_manifest=staged_manifest,
            source_fingerprint=source_fingerprint,
        )
        post_manifest = validator(database, target_version)
        if post_manifest != staged_manifest:
            raise MigrationError("post-swap database manifest differs from validated stage")
        _write_journal(
            journal,
            "POST_VALIDATED",
            current,
            target_version,
            None,
            backup,
            staged_manifest=staged_manifest,
            source_fingerprint=source_fingerprint,
        )
    except Exception as error:
        _write_journal(
            journal,
            "ROLLBACK_STARTED",
            current,
            target_version,
            str(error),
            backup,
            staged_manifest=staged_manifest,
            source_fingerprint=source_fingerprint,
        )
        try:
            _restore_known_good(
                database,
                backup,
                current=current,
                target=target_version,
                replace_file=replace_file,
                source_fingerprint=source_fingerprint,
            )
        except Exception as rollback_error:
            _write_journal(
                journal,
                "ROLLBACK_REQUIRED",
                current,
                target_version,
                f"migration error: {error}; rollback error: {rollback_error}",
                backup,
                staged_manifest=staged_manifest,
                source_fingerprint=source_fingerprint,
            )
            raise MigrationError(
                f"migration failed and rollback failed; operator recovery required: {rollback_error}"
            ) from rollback_error
        _write_journal(
            journal,
            "BACKUP_RESTORED",
            current,
            target_version,
            str(error),
            backup,
            staged_manifest=staged_manifest,
            source_fingerprint=source_fingerprint,
        )
        _write_journal(
            journal,
            "ROLLED_BACK",
            current,
            target_version,
            str(error),
            backup,
            staged_manifest=staged_manifest,
            source_fingerprint=source_fingerprint,
        )
        raise MigrationError(f"migration commit failed and was rolled back: {error}") from error

    _write_journal(
        journal,
        "COMMITTED",
        current,
        target_version,
        None,
        backup,
        staged_manifest=staged_manifest,
        source_fingerprint=source_fingerprint,
    )
    return {
        "current_schema_version": target_version,
        "target_schema_version": target_version,
        "migrated": True,
        "rollback_backup": str(backup) if backup is not None else None,
    }


def _recover_interrupted(
    database: Path,
    stage: Path,
    journal: Path,
    replace_file: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any],
) -> None:
    if not journal.exists():
        return
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MigrationError(f"migration journal is corrupt: {error}") from error
    state = payload.get("state")
    backup_text = payload.get("backup")
    backup = Path(backup_text) if backup_text else None
    source_fingerprint = payload.get("source_fingerprint")
    if state in {
        "SWAPPED",
        "POST_VALIDATED",
        "ROLLBACK_REQUIRED",
        "ROLLBACK_STARTED",
        "BACKUP_RESTORED",
        "BACKUP_SWAPPED",
    }:
        require_writer_lease(database)
        try:
            _restore_known_good(
                database,
                backup,
                current=int(payload.get("current_schema_version") or 0),
                target=int(payload.get("target_schema_version") or TARGET_SCHEMA_VERSION),
                replace_file=replace_file,
                source_fingerprint=source_fingerprint,
            )
        except Exception as error:
            raise MigrationError(f"interrupted migration rollback failed: {error}") from error
        _write_journal(
            journal,
            "ROLLED_BACK",
            int(payload.get("current_schema_version") or 0),
            int(payload.get("target_schema_version") or TARGET_SCHEMA_VERSION),
            "recovered interrupted uncommitted swap",
            backup,
            source_fingerprint=source_fingerprint,
        )
    elif state == "VALIDATED" and not database.exists() and backup is not None and backup.exists():
        require_writer_lease(database)
        _durable_replace(backup, database, replace_file)
    elif not database.exists() and state not in {"PREPARED", "STAGED", "ROLLED_BACK"}:
        raise MigrationError("interrupted migration left no recoverable database")
    if stage.exists():
        stage.unlink()


def _restore_known_good(
    database: Path,
    backup: Path | None,
    *,
    current: int,
    target: int,
    replace_file: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any],
    source_fingerprint: str | None,
) -> None:
    quarantine = database.with_suffix(database.suffix + f".failed-v{target}")
    # Recovery is journal-driven and idempotent. A diagnostic quarantine is
    # never treated as the authority that can veto restoration.
    if database.exists() and source_fingerprint is not None:
        if _file_fingerprint(database) == source_fingerprint:
            return
    if database.exists():
        destination = quarantine
        suffix = 1
        while destination.exists():
            destination = quarantine.with_name(f"{quarantine.name}.{suffix}")
            suffix += 1
        _durable_replace(database, destination, replace_file)
    if backup is None:
        if current != 0:
            raise MigrationError("known-good backup path missing")
        return
    if not backup.exists():
        if database.exists() and source_fingerprint is not None:
            if _file_fingerprint(database) == source_fingerprint:
                return
        raise MigrationError(f"known-good backup missing: {backup}")
    _durable_replace(backup, database, replace_file)
    if source_fingerprint is not None and _file_fingerprint(database) != source_fingerprint:
        raise MigrationError("restored database fingerprint differs from known-good source")


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_metadata(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _record_migration(path: Path, version: int, name: str, source: str) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute("BEGIN TRANSACTION")
        _ensure_metadata(connection)
        checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [version, name, checksum],
        )
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _apply_sql_migration(path: Path, version: int, name: str, statements: tuple[str, ...]) -> None:
    connection = duckdb.connect(str(path))
    metadata_transaction = False
    try:
        # DuckDB 1.5 cannot commit index recreation and ALTER TABLE deltas in
        # one transaction. The whole database is already a disposable staged
        # copy, so execute DDL there and publish only through the atomic file
        # swap after validation. A crash leaves the live database untouched and
        # the journal causes the partial stage to be discarded on retry.
        _ensure_metadata(connection)
        for statement in statements:
            connection.execute(statement)
        checksum = hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()
        connection.execute("BEGIN TRANSACTION")
        metadata_transaction = True
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [version, name, checksum],
        )
        connection.execute("COMMIT")
        metadata_transaction = False
        connection.execute("CHECKPOINT")
    except Exception:
        if metadata_transaction:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        raise
    finally:
        connection.close()


def _validate_decision_eligible_view(connection: duckdb.DuckDBPyConnection) -> None:
    object_rows = connection.execute(
        "SELECT table_type FROM information_schema.tables WHERE table_name = 'decision_eligible_ohlcv'"
    ).fetchall()
    if not object_rows:
        raise MigrationError("decision_eligible_ohlcv is missing after migration")
    if str(object_rows[0][0]).upper() != "VIEW":
        raise MigrationError(
            f"decision_eligible_ohlcv exists as incompatible object type: {object_rows[0][0]}"
        )
    columns = {
        row[0]: str(row[1]).upper()
        for row in connection.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'decision_eligible_ohlcv'"
        ).fetchall()
    }
    missing = sorted(set(DECISION_ELIGIBLE_OHLCV_REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise MigrationError(f"decision_eligible_ohlcv missing columns: {', '.join(missing)}")
    mismatched = sorted(
        name
        for name, expected in DECISION_ELIGIBLE_OHLCV_REQUIRED_COLUMNS.items()
        if columns.get(name) != expected
    )
    if mismatched:
        raise MigrationError(f"decision_eligible_ohlcv column type mismatch: {', '.join(mismatched)}")


def _apply_decision_eligible_view_migration(
    path: Path, version: int, name: str, statements: tuple[str, ...]
) -> None:
    connection = duckdb.connect(str(path))
    metadata_transaction = False
    try:
        _ensure_metadata(connection)
        existing = connection.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = 'decision_eligible_ohlcv'"
        ).fetchone()
        if existing is not None and str(existing[0]).upper() != "VIEW":
            raise MigrationError(
                f"decision_eligible_ohlcv already exists as incompatible object type "
                f"({existing[0]}); refusing to silently replace it"
            )
        for statement in statements:
            connection.execute(statement)
        _validate_decision_eligible_view(connection)
        checksum = hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()
        connection.execute("BEGIN TRANSACTION")
        metadata_transaction = True
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [version, name, checksum],
        )
        connection.execute("COMMIT")
        metadata_transaction = False
        connection.execute("CHECKPOINT")
    except Exception:
        if metadata_transaction:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        raise
    finally:
        connection.close()


def _current_version(path: Path) -> int:
    try:
        connection = duckdb.connect(str(path), read_only=True)
    except duckdb.IOException as error:
        if "being used by another process" in str(error) or "already open in" in str(error):
            raise DatabaseOwnedElsewhere(str(error)) from error
        raise MigrationError(f"database preflight failed: {error}") from error
    except Exception as error:
        raise MigrationError(f"database preflight failed: {error}") from error
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "schema_migrations" not in tables:
            return 0
        row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
        return int(row[0] if row else 0)
    except Exception as error:
        raise MigrationError(f"database schema preflight failed: {error}") from error
    finally:
        connection.close()


def _validate_database(path: Path, target_version: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise MigrationError("migrated database is missing or empty")
    try:
        connection = duckdb.connect(str(path), read_only=True)
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        required_tables = {
            "schema_migrations",
            "ohlcv",
            "fo_eod",
            "instruments",
            "backtest_runs",
            "ingestion_manifests",
        }
        if target_version >= 5:
            required_tables = required_tables | {"decision_eligible_ohlcv"}
        missing_tables = sorted(required_tables.difference(tables))
        if missing_tables:
            raise MigrationError(f"database validation missing tables: {', '.join(missing_tables)}")
        if target_version >= 5:
            _validate_decision_eligible_view(connection)
        version = int(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])
        if version != target_version:
            raise MigrationError(f"database validation expected schema {target_version}, got {version}")
        schema_rows = connection.execute(
            """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name IN (
                'schema_migrations', 'ohlcv', 'fo_eod', 'instruments',
                'backtest_runs', 'ingestion_manifests', 'decision_eligible_ohlcv'
            )
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        fo_columns = {row[1] for row in schema_rows if row[0] == "fo_eod"}
        if not {"source", "source_format", "price_quality"}.issubset(fo_columns):
            raise MigrationError("database validation missing F&O provenance columns")
        required_columns = {
            "ohlcv": {"symbol", "date", "open", "high", "low", "close", "volume", "source"},
            "instruments": {"token", "symbol", "exchange", "schema_version"},
            "backtest_runs": {"job_id", "status", "validity_status", "price_policy_version"},
            "ingestion_manifests": {
                "source_id",
                "source_sha256",
                "source_identity",
                "expected_scope_json",
                "ingestion_completed_at",
                "validation_completed_at",
                "completeness_status",
                "downstream_commit_id",
                "retention_decision",
            },
        }
        for table, expected in required_columns.items():
            actual = {row[1] for row in schema_rows if row[0] == table}
            missing = sorted(expected - actual)
            if missing:
                raise MigrationError(
                    f"database validation missing {table} columns: {', '.join(missing)}"
                )
        table_counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in sorted(required_tables)
        }
        connection.execute("SELECT 1").fetchone()
        schema_checksum = hashlib.sha256(
            json.dumps(schema_rows, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": version,
            "table_counts": table_counts,
            "schema_checksum": schema_checksum,
        }
    except MigrationError:
        raise
    except Exception as error:
        raise MigrationError(f"database validation failed: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()


def _write_journal(
    path: Path,
    state: str,
    current: int,
    target: int,
    error: str | None,
    backup: Path | None = None,
    *,
    staged_manifest: dict[str, Any] | None = None,
    source_fingerprint: str | None = None,
) -> None:
    payload = {
        "state": state,
        "current_schema_version": current,
        "target_schema_version": target,
        "backup": str(backup) if backup is not None else None,
        "error": error,
        "staged_manifest": staged_manifest,
        "source_fingerprint": source_fingerprint,
        "pid": os.getpid(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _durable_replace(temporary, path, os.replace)


def _durable_replace(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    replace_file: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any],
) -> None:
    """Publish an already-closed file and durably flush the strongest available metadata."""
    source_path = Path(source)
    destination_path = Path(destination)
    _flush_file(source_path)
    if os.name == "nt" and replace_file is os.replace:
        _windows_replace_write_through(source_path, destination_path)
    else:
        replace_file(source_path, destination_path)
    _flush_file(destination_path)
    _flush_parent_metadata(destination_path.parent)


def _flush_file(path: Path) -> None:
    """Flush file bytes; on Windows os.fsync delegates to FlushFileBuffers."""
    with Path(path).open("r+b") as handle:
        os.fsync(handle.fileno())


def _flush_parent_metadata(directory: Path) -> None:
    """Flush rename metadata where the platform exposes a supported primitive.

    Windows does not expose POSIX directory fsync. The Windows replace path uses
    MOVEFILE_WRITE_THROUGH; this function intentionally performs no fictitious
    directory flush there. POSIX platforms fsync the real parent directory.
    """
    if os.name == "nt":
        return
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _windows_replace_write_through(source: Path, destination: Path) -> None:
    import ctypes
    from ctypes import wintypes

    move_file_ex = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move_file_ex.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file_ex.restype = wintypes.BOOL
    movefile_replace_existing = 0x00000001
    movefile_write_through = 0x00000008
    if not move_file_ex(
        str(source),
        str(destination),
        movefile_replace_existing | movefile_write_through,
    ):
        raise OSError(ctypes.get_last_error(), "MoveFileExW(MOVEFILE_WRITE_THROUGH) failed")
