from __future__ import annotations

import importlib
from pathlib import Path

import duckdb
import pytest


@pytest.mark.contract
def test_backtest_concurrency_and_symbol_caches_are_bounded(
    isolated_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = importlib.import_module("main")
    monkeypatch.setattr(main, "_MAX_ACTIVE_BACKTEST_JOBS", 2)
    monkeypatch.setattr(main, "_MAX_SYMBOL_CACHE_ENTRIES", 2)
    main._backtest_jobs.clear()
    assert main._reserve_backtest_job("a", {"status": "queued"}) is True
    assert main._reserve_backtest_job("b", {"status": "running"}) is True
    assert main._reserve_backtest_job("c", {"status": "queued"}) is False
    main._update_backtest_job("a", status="completed")
    assert main._reserve_backtest_job("c", {"status": "queued"}) is True

    cache = {"a": 1, "b": 2, "c": 3}
    timestamps = {"a": 1.0, "b": 2.0, "c": 3.0}
    main._prune_symbol_cache(cache, timestamps)
    assert cache == {"b": 2, "c": 3}
    assert timestamps == {"b": 2.0, "c": 3.0}


@pytest.mark.contract
def test_startup_recovery_fails_only_interrupted_backtests(isolated_app_data: Path) -> None:
    duckdb_client = importlib.import_module("db.duckdb_client")
    engine = importlib.import_module("backtest.engine")
    duckdb_client.init_schema()

    connection = duckdb.connect(str(duckdb_client.DB_PATH))
    try:
        connection.execute(
            """
            INSERT INTO backtest_runs
                (job_id, strategy_name, symbol, from_date, to_date, initial_capital,
                 status, progress, stage, started_at)
            VALUES
                ('queued-job', 's', 'NIFTY', '2026-01-01', '2026-01-02', 1000,
                 'queued', 0, 'queued', CURRENT_TIMESTAMP),
                ('done-job', 's', 'NIFTY', '2026-01-01', '2026-01-02', 1000,
                 'completed', 1, 'complete', CURRENT_TIMESTAMP)
            """
        )
    finally:
        connection.close()

    write_service = importlib.import_module("db.write_service")
    queue = isolated_app_data / "data" / "writer.sqlite"
    service = write_service.WriterService(duckdb_client.DB_PATH, queue_path=queue)
    service.start()
    try:
        first = write_service.submit_write(
            write_service.WriteOperation.RECOVER_BACKTESTS,
            {},
            database=duckdb_client.DB_PATH,
            queue_path=queue,
        )
        second = write_service.submit_write(
            write_service.WriteOperation.RECOVER_BACKTESTS,
            {},
            database=duckdb_client.DB_PATH,
            queue_path=queue,
        )
        assert first == {"recovered": 1}
        assert second == {"recovered": 0}
    finally:
        service.stop()

    connection = duckdb.connect(str(duckdb_client.DB_PATH), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT job_id, status, stage, validity_status, completed_at IS NOT NULL
            FROM backtest_runs ORDER BY job_id
            """
        ).fetchall()
    finally:
        connection.close()

    assert rows == [
        ("done-job", "completed", "complete", "INVALID_LEGACY_PRICE_SEMANTICS", False),
        ("queued-job", "failed", "interrupted", "INVALID_INTERRUPTED_RUN", True),
    ]
