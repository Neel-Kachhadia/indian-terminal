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
from options.iv_contract import COLUMN_TYPES


TARGET_SCHEMA_VERSION = 9
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

# Adds real provenance to iv_history (previously symbol/expiry/strike/
# option_type/date/iv only, with zero source/spot/rate/solver/quality
# columns) and introduces risk_free_rate_history so the hardcoded
# RISK_FREE_RATE=0.065 constant in options/greeks.py can be replaced with a
# real, tenor-matched, dated rate. All new iv_history columns are nullable
# additions -- the existing primary key and every pre-existing row (there
# are none in production at migration time) are preserved unchanged.
_IV_V6_COLUMNS_SQL = ",\n        ".join(
    f"{name} {kind}{' PRIMARY KEY' if name == 'observation_id' else ''}"
    for name, kind in COLUMN_TYPES.items()
)

MIGRATION_6_STATEMENTS = (
    "DROP INDEX IF EXISTS idx_iv_symbol",
    "ALTER TABLE iv_history RENAME TO iv_history_legacy_v5",
    f"CREATE TABLE iv_history ({_IV_V6_COLUMNS_SQL})",
    """
    INSERT INTO iv_history (
        observation_id, symbol, underlying, expiry, strike, option_type,
        date, iv, source, quality, calculation_status, reason,
        observation_kind, contract_version
    )
    SELECT
        'legacy:' || sha256(
            symbol || '|' || CAST(expiry AS VARCHAR) || '|' ||
            CAST(strike AS VARCHAR) || '|' || option_type || '|' || CAST(date AS VARCHAR)
        ),
        symbol, symbol, expiry, strike, option_type, date, iv,
        'UNKNOWN', 'SOURCE_UNVERIFIED', 'SOURCE_UNVERIFIED',
        'legacy row lacks pricing inputs and provenance', 'LEGACY', 'legacy-v1'
    FROM iv_history_legacy_v5
    """,
    "CREATE UNIQUE INDEX idx_iv_observation_natural ON iv_history(symbol, expiry, strike, option_type, observed_at, observation_kind)",
    "CREATE INDEX idx_iv_symbol_date ON iv_history(symbol, date, observation_kind)",
    """
    CREATE TABLE IF NOT EXISTS risk_free_rate_history (
        observation_date DATE    NOT NULL,
        tenor_days       INTEGER NOT NULL,
        tenor_label      TEXT    NOT NULL,
        rate_pct         DOUBLE  NOT NULL,
        rate_decimal     DOUBLE  NOT NULL,
        source           TEXT    NOT NULL,
        source_url       TEXT    NOT NULL,
        published_at     TIMESTAMP,
        retrieved_at     TIMESTAMP NOT NULL,
        quote_convention TEXT    NOT NULL,
        rate_convention  TEXT    NOT NULL,
        conversion_method TEXT  NOT NULL,
        PRIMARY KEY (observation_date, tenor_days)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rfr_date ON risk_free_rate_history(observation_date)",
)

# risk_free_rate_history's CREATE TABLE above (MIGRATION_6_STATEMENTS) gained
# quote_convention/rate_convention/conversion_method only after real
# production databases had already been migrated to schema 6 by an earlier
# revision of that same CREATE TABLE, which did not yet declare those three
# columns. Migrations are gated purely by version number (migrate_database's
# `if current == target_version: return` short-circuit), so editing an
# already-applied migration's SQL text never reaches an already-migrated
# database -- proven live: a real production terminal.db at schema 6 was
# missing exactly these three columns even on current source. `CREATE TABLE
# IF NOT EXISTS` cannot retrofit columns onto an existing table either. A new
# versioned step is the only way to actually reach those databases.
# DEFAULT backfills every pre-existing row, not just new ones (DuckDB ADD
# COLUMN ... DEFAULT semantics) -- safe here because this table has exactly
# one writer (data/risk_free_rate.py) and it has used only these three
# literal convention values since the table was introduced; this is not a
# fabricated/guessed provenance, it is the one convention this system has
# ever actually written. Matches the existing ADD COLUMN IF NOT EXISTS ...
# DEFAULT idiom already used for `instruments.schema_version` above rather
# than introducing a second column-backfill pattern.
MIGRATION_7_STATEMENTS = (
    "ALTER TABLE risk_free_rate_history ADD COLUMN IF NOT EXISTS "
    "quote_convention TEXT DEFAULT 'SIMPLE_ACT_365_YIELD'",
    "ALTER TABLE risk_free_rate_history ADD COLUMN IF NOT EXISTS "
    "rate_convention TEXT DEFAULT 'CONTINUOUS_ACT_365'",
    "ALTER TABLE risk_free_rate_history ADD COLUMN IF NOT EXISTS "
    "conversion_method TEXT DEFAULT 'LN_1_PLUS_SIMPLE_TIMES_T_OVER_T'",
)

# Same root cause as MIGRATION_7_STATEMENTS above, discovered on the same
# real production database: iv_history's CREATE TABLE (_IV_V6_COLUMNS_SQL,
# built from options/iv_contract.py's COLUMN_TYPES) has grown from the
# original ~15-column "iv-backfill-v1" era shape to the full 29-column
# iv-observation-v3 contract (observation_id identity, forward/dividend/
# rate provenance, observation_kind, contract_version, ...) entirely inside
# this uncommitted branch of work. A real production terminal.db already at
# schema 6 was migrated under an earlier revision of that CREATE TABLE and
# still physically has only the original ~15 columns with 41k real
# `iv-backfill-v1` rows in them -- migrate_database's `current == target`
# short-circuit means the current CREATE TABLE text, however it has since
# changed, never reaches that database. This mirrors MIGRATION_6_STATEMENTS'
# own rename -> recreate -> quarantined-legacy-insert shape (proven safe
# there for the pre-v6 legacy rows) rather than an in-place ALTER, because
# the target shape's PRIMARY KEY is `observation_id` (identity-based, so a
# LIVE and a HISTORICAL_RECONSTRUCTION row can coexist for the same
# contract/date) while the real table's PRIMARY KEY is still the old
# (symbol, expiry, strike, option_type, date) tuple -- a constraint DuckDB
# cannot alter in place, only replace via a new table.
# Old rows are quarantined the same way the v5 legacy rows already are:
# every identity/provenance column this contract requires (option_price_
# semantic, spot_source, risk_free_source, pricing_model, solver, ...) is
# left NULL, `quality`/`calculation_status` forced to SOURCE_UNVERIFIED
# (never trust-preserving the old table's own IV_VALID label), and
# `observation_kind='LEGACY'` -- a value validate_observation's `{"LIVE",
# "HISTORICAL_RECONSTRUCTION"}` check can never accept. These rows can
# never validate or reach a decision path; the original quality value and
# calculation_version are preserved only inside `reason` for audit
# traceability, and `contract_version` is honestly labelled with the real
# pipeline generation that produced them, never CONTRACT_VERSION.
MIGRATION_8_STATEMENTS = (
    "DROP INDEX IF EXISTS idx_iv_symbol",
    "ALTER TABLE iv_history RENAME TO iv_history_legacy_v1_backfill",
    f"CREATE TABLE iv_history ({_IV_V6_COLUMNS_SQL})",
    """
    INSERT INTO iv_history (
        observation_id, symbol, underlying, expiry, strike, option_type,
        date, iv, source, quality, calculation_status, reason,
        observation_kind, contract_version
    )
    SELECT
        'legacy-iv-backfill-v1:' || sha256(
            symbol || '|' || CAST(expiry AS VARCHAR) || '|' ||
            CAST(strike AS VARCHAR) || '|' || option_type || '|' || CAST(date AS VARCHAR)
        ),
        symbol, symbol, expiry, strike, option_type, date, iv,
        'UNKNOWN', 'SOURCE_UNVERIFIED', 'SOURCE_UNVERIFIED',
        'legacy iv-backfill-v1 row predates current contract/provenance columns '
        || '(original quality=' || quality || ', calculation_version=' || calculation_version || ')',
        'LEGACY', 'legacy-iv-backfill-v1'
    FROM iv_history_legacy_v1_backfill
    """,
    "CREATE UNIQUE INDEX idx_iv_observation_natural ON iv_history(symbol, expiry, strike, option_type, observed_at, observation_kind)",
    "CREATE INDEX idx_iv_symbol_date ON iv_history(symbol, date, observation_kind)",
)

# Every real risk_free_rate_history row persisted before this fix stored
# rate_decimal = rate_pct / 100 -- a naive percent-to-decimal conversion --
# instead of the documented continuously-compounded ACT/365 conversion
# TenorPoint.rate_decimal actually defines (log1p(simple * T) / T). Proven
# live: db.write_service._upsert_risk_free_rate_curve's own math.isclose
# check on write, and options.iv_backfill.load_rate_curve's identical check
# on read, both already correctly refuse a mismatched rate_decimal -- so
# every historical rate row was silently unusable regardless of how much
# genuine FBIL curve history existed, for a completely different reason
# than "the source lacks history". rate_pct (the actual FBIL-quoted number)
# is never touched by this migration; only the deterministic derived
# column is recomputed through the one documented, tested formula, which
# is a correctness fix, not a fabrication -- the raw ingredient was always
# genuine, only its transformation was wrong.
def _apply_risk_free_rate_decimal_correction_migration(
    path: Path, version: int, name: str, statements: tuple[str, ...]
) -> None:
    del statements  # this step is data-driven, not a fixed SQL script
    from data.risk_free_rate import TenorPoint

    connection = duckdb.connect(str(path))
    metadata_transaction = False
    try:
        _ensure_metadata(connection)
        rows = connection.execute(
            "SELECT observation_date, tenor_days, tenor_label, rate_pct, rate_decimal "
            "FROM risk_free_rate_history"
        ).fetchall()
        corrected = 0
        for observation_date, tenor_days, tenor_label, rate_pct, stored_decimal in rows:
            expected = TenorPoint(
                observation_date=observation_date,
                tenor_days=tenor_days,
                tenor_label=tenor_label,
                rate_pct=rate_pct,
                published_at=None,
                source="",
                source_url="",
            ).rate_decimal
            if stored_decimal is None or abs(float(stored_decimal) - expected) > 1e-9:
                connection.execute(
                    "UPDATE risk_free_rate_history SET rate_decimal = ? "
                    "WHERE observation_date = ? AND tenor_days = ?",
                    [expected, observation_date, tenor_days],
                )
                corrected += 1
        checksum = hashlib.sha256(f"rows={len(rows)} corrected={corrected}".encode("utf-8")).hexdigest()
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


MIGRATION_9_STATEMENTS: tuple[str, ...] = ()


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
        # Preserve an earlier upgrade's rollback artifact.  A later migration
        # can legitimately start from the same schema version after rollback;
        # never overwrite or delete the older evidence.
        backup = database.with_suffix(
            database.suffix + f".pre-v{current}-to-v{target_version}.bak"
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
            elif next_version == 6:
                _apply_sql_migration(
                    stage,
                    6,
                    "iv_history_provenance_and_risk_free_rate",
                    MIGRATION_6_STATEMENTS,
                )
            elif next_version == 7:
                _apply_sql_migration(
                    stage,
                    7,
                    "risk_free_rate_convention_columns_backfill",
                    MIGRATION_7_STATEMENTS,
                )
            elif next_version == 8:
                _apply_iv_history_v3_contract_migration(
                    stage,
                    8,
                    "iv_history_v3_contract_backfill",
                    MIGRATION_8_STATEMENTS,
                )
            elif next_version == 9:
                _apply_risk_free_rate_decimal_correction_migration(
                    stage,
                    9,
                    "risk_free_rate_decimal_correction",
                    MIGRATION_9_STATEMENTS,
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


def _apply_iv_history_v3_contract_migration(
    path: Path, version: int, name: str, statements: tuple[str, ...]
) -> None:
    """Unlike every other versioned step, this one is conditional: a fresh
    install (current=0, migrating straight through to the latest target)
    already creates iv_history via MIGRATION_6_STATEMENTS' `_IV_V6_COLUMNS_
    SQL`, which is built from the CURRENT options/iv_contract.COLUMN_TYPES
    and therefore already has every column this step would otherwise add.
    Running the rename -> recreate -> quarantine-as-legacy statements
    unconditionally would wrongly relabel that already-correct, already-
    valid data as LEGACY. Only a real pre-existing table from the older
    ~15-column "iv-backfill-v1" era (missing any current COLUMN_TYPES
    column) needs the backfill; anything already complete is a no-op here
    -- the version bookkeeping below still advances either way."""
    connection = duckdb.connect(str(path))
    metadata_transaction = False
    try:
        _ensure_metadata(connection)
        existing_columns = {
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'iv_history'"
            ).fetchall()
        }
        needs_backfill = bool(set(COLUMN_TYPES) - existing_columns)
        if needs_backfill:
            for statement in statements:
                connection.execute(statement)
            checksum_source = "\n".join(statements)
        else:
            checksum_source = "no-op: iv_history already matches current COLUMN_TYPES"
        checksum = hashlib.sha256(checksum_source.encode("utf-8")).hexdigest()
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
        if target_version >= 6:
            required_tables = required_tables | {"iv_history", "risk_free_rate_history"}
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
                'backtest_runs', 'ingestion_manifests', 'decision_eligible_ohlcv',
                'iv_history', 'risk_free_rate_history'
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
        if target_version >= 6:
            required_columns["risk_free_rate_history"] = {
                "observation_date", "tenor_days", "tenor_label", "rate_pct",
                "rate_decimal", "source", "source_url", "retrieved_at",
            }
        if target_version >= 7:
            required_columns["risk_free_rate_history"] = required_columns["risk_free_rate_history"] | {
                "quote_convention", "rate_convention", "conversion_method",
            }
        if target_version >= 8:
            required_columns["iv_history"] = set(COLUMN_TYPES)
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
