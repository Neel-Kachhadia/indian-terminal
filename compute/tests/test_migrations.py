from __future__ import annotations

import json
import os
from collections import namedtuple
from pathlib import Path

import duckdb
import pytest

from db import migrations as migrations_module
from db.duckdb_client import _create_schema_objects_at
from db.migrations import (
    DatabaseOwnedElsewhere,
    MigrationError,
    TARGET_SCHEMA_VERSION,
    _current_version,
    _file_fingerprint,
    _recover_interrupted,
    _validate_database,
    _validate_decision_eligible_view,
    migrate_database,
)


def sparse_database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE ohlcv(symbol TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT, source TEXT, PRIMARY KEY(symbol, date))"
        )
        connection.execute("INSERT INTO ohlcv VALUES ('NIFTY', DATE '2025-01-01', 1, 2, 1, 2, 10, 'test')")
        connection.execute(
            "CREATE TABLE instruments(token TEXT, symbol TEXT, name TEXT, exchange TEXT, instrument_type TEXT, expiry DATE, strike DOUBLE, option_type TEXT, lot_size INTEGER, tick_size DOUBLE, isin TEXT, updated_at TIMESTAMP, PRIMARY KEY(token, exchange))"
        )
    finally:
        connection.close()


def version(path: Path) -> int:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        return int(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])
    finally:
        connection.close()


@pytest.mark.contract
def test_fresh_install_migrates_to_target(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    result = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert result["migrated"] is True
    assert version(database) == TARGET_SCHEMA_VERSION


@pytest.mark.contract
def test_migration_noop_does_not_require_writer_lease(tmp_path: Path, monkeypatch) -> None:
    """Reproduces a real live crash: the desktop backend calls
    `init_schema()` -> `migrate_database()` on every startup as a routine
    preflight, even when the database is already at the current schema
    version and nothing will be written. `require_writer_lease()` was
    called unconditionally before that no-op check, so the backend
    crash-looped with WriterLeaseError whenever the collector already
    legitimately held writer authority -- entirely blocking the
    collector-first -> desktop-second topology. A database already at the
    target version must never need exclusive writer access just to confirm
    that."""
    database = tmp_path / "terminal.db"
    migrate_database(database, schema_initializer=_create_schema_objects_at)

    def _boom(_db: Path) -> None:
        raise AssertionError("require_writer_lease must not be called for an already-current database")

    monkeypatch.setattr(migrations_module, "require_writer_lease", _boom)
    result = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert result["migrated"] is False
    assert result["current_schema_version"] == TARGET_SCHEMA_VERSION


@pytest.mark.contract
def test_migration_defers_when_another_process_owns_the_database(tmp_path: Path) -> None:
    """Reproduces a real live crash: with the real BharatTerminal collector
    holding writer authority on terminal.db, launching the real desktop
    backend crashed outright -- `_current_version()`'s read-only preflight
    connect failed with a genuine DuckDB IOException (Windows sharing
    violation; DuckDB has no SQLite-WAL-style cross-process concurrent
    reader), which `migrate_database` previously treated as a fatal
    MigrationError. The collector's own standalone startup already trusts
    an existing terminal.db without re-validating it
    (`smart_collector.py::_try_become_writer_authority` never calls
    `migrate_database` at all) -- the backend must extend that same trust
    when it discovers another live process already owns the file, instead
    of crash-looping."""
    import subprocess
    import sys
    import time

    database = tmp_path / "terminal.db"
    migrate_database(database, schema_initializer=_create_schema_objects_at)

    holder = subprocess.Popen(
        [sys.executable, "-c", f"import duckdb, time; c = duckdb.connect(r'{database}'); time.sleep(20)"]
    )
    try:
        # Confirm the busy-elsewhere condition is actually reached before
        # trusting migrate_database's result -- both the graceful-defer path
        # and the ordinary already-current path return the same shape.
        deadline = time.monotonic() + 10
        confirmed_busy = False
        while time.monotonic() < deadline:
            try:
                _current_version(database)
                time.sleep(0.2)
            except DatabaseOwnedElsewhere:
                confirmed_busy = True
                break
        assert confirmed_busy, "holder subprocess never produced the busy-elsewhere condition"

        result = migrate_database(database, schema_initializer=_create_schema_objects_at)
        assert result["migrated"] is False
        assert result["current_schema_version"] == TARGET_SCHEMA_VERSION
    finally:
        holder.kill()
        holder.wait(timeout=10)


@pytest.mark.contract
def test_sparse_legacy_upgrade_preserves_rows_and_adds_columns(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    migrate_database(database, schema_initializer=_create_schema_objects_at)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == 1
        columns = {row[0] for row in connection.execute("DESCRIBE instruments").fetchall()}
        assert "schema_version" in columns
    finally:
        connection.close()


@pytest.mark.recovery
def test_interrupted_swap_restores_then_retries(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    backup = database.with_suffix(".db.pre-v0.bak")
    stage = database.with_suffix(".db.migration-stage")
    os.replace(database, backup)
    stage.write_bytes(b"partial")
    journal = database.with_suffix(".db.migration.json")
    journal.write_text(
        json.dumps({"state": "BACKUP_SWAPPED", "backup": str(backup)}),
        encoding="utf-8",
    )
    result = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert result["migrated"] is True
    assert version(database) == TARGET_SCHEMA_VERSION


@pytest.mark.recovery
def test_migration_retry_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    first = migrate_database(database, schema_initializer=_create_schema_objects_at)
    second = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert first["migrated"] is True
    assert second == {
        "current_schema_version": TARGET_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "migrated": False,
    }


@pytest.mark.recovery
def test_corrupt_database_fails_preflight_without_replacement(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    database.write_bytes(b"not-a-duckdb")
    before = database.read_bytes()
    with pytest.raises(MigrationError, match="preflight"):
        migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert database.read_bytes() == before


@pytest.mark.recovery
def test_disk_full_preflight_does_not_copy_or_modify(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    before = database.read_bytes()
    Usage = namedtuple("Usage", "total used free")
    with pytest.raises(MigrationError, match="insufficient disk"):
        migrate_database(
            database,
            schema_initializer=_create_schema_objects_at,
            disk_usage=lambda _: Usage(100, 99, 1),
        )
    assert database.read_bytes() == before


@pytest.mark.recovery
def test_atomic_commit_failure_restores_original(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    calls = 0

    def fail_second_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated commit failure")
        os.replace(source, target)

    with pytest.raises(MigrationError, match="rolled back"):
        migrate_database(
            database,
            schema_initializer=_create_schema_objects_at,
            replace_file=fail_second_replace,
        )
    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == 1
        assert "schema_migrations" not in {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    finally:
        connection.close()


@pytest.mark.recovery
def test_post_swap_validation_failure_restores_known_good_backup(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    before = database.read_bytes()
    validation_calls = 0

    def fail_post_swap(path: Path, target: int) -> dict[str, object]:
        nonlocal validation_calls
        validation_calls += 1
        manifest = _validate_database(path, target)
        if validation_calls == 2:
            raise MigrationError("simulated post-swap validation failure")
        return manifest

    with pytest.raises(MigrationError, match="rolled back"):
        migrate_database(
            database,
            schema_initializer=_create_schema_objects_at,
            validate_database=fail_post_swap,
        )
    assert database.read_bytes() == before
    journal = json.loads(database.with_suffix(".db.migration.json").read_text(encoding="utf-8"))
    assert journal["state"] == "ROLLED_BACK"
    assert database.with_suffix(f".db.failed-v{TARGET_SCHEMA_VERSION}").is_file()


@pytest.mark.recovery
def test_restart_after_quarantine_before_restore_converges_twice(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    source_fingerprint = _file_fingerprint(database)
    backup = database.with_suffix(".db.pre-v0.bak")
    stage = database.with_suffix(".db.migration-stage")
    quarantine = database.with_suffix(f".db.failed-v{TARGET_SCHEMA_VERSION}")
    journal = database.with_suffix(".db.migration.json")

    os.replace(database, backup)
    database.write_bytes(b"invalid-upgraded-database")
    journal.write_text(
        json.dumps(
            {
                "state": "ROLLBACK_STARTED",
                "current_schema_version": 0,
                "target_schema_version": TARGET_SCHEMA_VERSION,
                "backup": str(backup),
                "source_fingerprint": source_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    # Exact crash point: invalid live DB was quarantined; backup not restored.
    os.replace(database, quarantine)
    assert not database.exists() and backup.exists() and quarantine.exists()

    _recover_interrupted(database, stage, journal, os.replace)
    assert database.exists()
    assert _file_fingerprint(database) == source_fingerprint
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "ROLLED_BACK"

    # Second restart must remain converged; quarantine is only evidence.
    _recover_interrupted(database, stage, journal, os.replace)
    assert _file_fingerprint(database) == source_fingerprint


@pytest.mark.recovery
def test_stage_validation_failure_never_swaps_live_database(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    before = database.read_bytes()

    def fail_stage(path: Path, target: int) -> dict[str, object]:
        raise MigrationError("simulated stage validation failure")

    with pytest.raises(MigrationError, match="staged migration failed"):
        migrate_database(
            database,
            schema_initializer=_create_schema_objects_at,
            validate_database=fail_stage,
        )
    assert database.read_bytes() == before
    journal = json.loads(database.with_suffix(".db.migration.json").read_text(encoding="utf-8"))
    assert journal["state"] == "ROLLED_BACK"


@pytest.mark.recovery
def test_rollback_failure_leaves_unambiguous_recovery_journal(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    validation_calls = 0
    replace_calls = 0

    def fail_post_swap(path: Path, target: int) -> dict[str, object]:
        nonlocal validation_calls
        validation_calls += 1
        manifest = _validate_database(path, target)
        if validation_calls == 2:
            raise MigrationError("simulated post-swap validation failure")
        return manifest

    def fail_rollback(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 3:
            raise OSError("simulated rollback failure")
        os.replace(source, target)

    with pytest.raises(MigrationError, match="rollback failed"):
        migrate_database(
            database,
            schema_initializer=_create_schema_objects_at,
            validate_database=fail_post_swap,
            replace_file=fail_rollback,
        )
    journal = json.loads(database.with_suffix(".db.migration.json").read_text(encoding="utf-8"))
    assert journal["state"] == "ROLLBACK_REQUIRED"
    assert "rollback error" in journal["error"]


@pytest.mark.recovery
def test_stage_flush_failure_occurs_before_live_backup_or_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "terminal.db"
    sparse_database(database)
    before = _file_fingerprint(database)
    real_flush = migrations_module._flush_file

    def fail_stage_flush(path: Path) -> None:
        if str(path).endswith(".migration-stage"):
            raise OSError("simulated stage flush failure")
        real_flush(path)

    monkeypatch.setattr(migrations_module, "_flush_file", fail_stage_flush)
    with pytest.raises(MigrationError, match="staged migration failed"):
        migrate_database(database, schema_initializer=_create_schema_objects_at)

    assert database.is_file()
    assert _file_fingerprint(database) == before
    assert not database.with_suffix(".db.pre-v0.bak").exists()


def _view_exists(path: Path) -> bool:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        row = connection.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = 'decision_eligible_ohlcv'"
        ).fetchone()
        return row is not None and str(row[0]).upper() == "VIEW"
    finally:
        connection.close()


@pytest.mark.contract
def test_case_a_fresh_database_gets_decision_eligible_view(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    result = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert result["migrated"] is True
    assert version(database) == TARGET_SCHEMA_VERSION == 5
    assert _view_exists(database)


@pytest.mark.contract
def test_case_b_v3_database_upgrades_and_gains_view(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    migrate_database(database, schema_initializer=_create_schema_objects_at, target_version=3)
    assert version(database) == 3
    result = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert result["migrated"] is True
    assert version(database) == TARGET_SCHEMA_VERSION
    assert _view_exists(database)


@pytest.mark.contract
def test_case_c_historical_v4_database_missing_view_migrates_to_v5(tmp_path: Path) -> None:
    """Reproduces the exact CRITICAL-1 field failure: a database already at
    schema v4 whose decision_eligible_ohlcv view is absent (the historical
    gap left by schema_initializer only running on the v0->v1 leg). This
    must fail if migration 5 (or its contract check) regresses."""
    database = tmp_path / "terminal.db"
    migrate_database(database, schema_initializer=_create_schema_objects_at, target_version=4)
    assert version(database) == 4
    connection = duckdb.connect(str(database))
    try:
        connection.execute("DROP VIEW decision_eligible_ohlcv")
    finally:
        connection.close()
    assert not _view_exists(database)

    result = migrate_database(database, schema_initializer=_create_schema_objects_at)

    assert result["migrated"] is True
    assert version(database) == TARGET_SCHEMA_VERSION == 5
    assert _view_exists(database)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        _validate_decision_eligible_view(connection)
        rows = connection.execute("SELECT COUNT(*) FROM decision_eligible_ohlcv").fetchone()
        assert rows[0] == 0
    finally:
        connection.close()


@pytest.mark.contract
def test_case_d_view_already_present_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    migrate_database(database, schema_initializer=_create_schema_objects_at)
    before = _file_fingerprint(database)
    result = migrate_database(database, schema_initializer=_create_schema_objects_at)
    assert result == {
        "current_schema_version": TARGET_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "migrated": False,
    }
    assert _file_fingerprint(database) == before
    assert _view_exists(database)


@pytest.mark.recovery
def test_case_e_incompatible_object_name_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    migrate_database(database, schema_initializer=_create_schema_objects_at, target_version=4)
    connection = duckdb.connect(str(database))
    try:
        connection.execute("DROP VIEW decision_eligible_ohlcv")
        connection.execute("CREATE TABLE decision_eligible_ohlcv (symbol TEXT)")
    finally:
        connection.close()
    before = database.read_bytes()

    with pytest.raises(MigrationError, match="incompatible object type"):
        migrate_database(database, schema_initializer=_create_schema_objects_at)

    # Live database must be untouched: migration failed closed, not silently repaired.
    assert database.read_bytes() == before
    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = 'decision_eligible_ohlcv'"
        ).fetchone()
        assert row is not None and str(row[0]).upper() == "BASE TABLE"
    finally:
        connection.close()


def test_durable_replace_flushes_source_then_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        migrations_module,
        "_flush_file",
        lambda path: events.append(("flush", Path(path).name)),
    )
    monkeypatch.setattr(
        migrations_module,
        "_flush_parent_metadata",
        lambda path: events.append(("parent", Path(path).name)),
    )

    def replace_file(from_path: Path, to_path: Path) -> None:
        events.append(("replace", f"{Path(from_path).name}->{Path(to_path).name}"))
        os.replace(from_path, to_path)

    migrations_module._durable_replace(source, destination, replace_file)

    assert destination.read_bytes() == b"new"
    assert events == [
        ("flush", "source"),
        ("replace", "source->destination"),
        ("flush", "destination"),
        ("parent", tmp_path.name),
    ]
