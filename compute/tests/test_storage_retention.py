from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import duckdb
import pytest

from db.write_service import WriteOperation, WriterService, submit_write
from operations import retention, storage


@pytest.mark.contract
def test_low_disk_guard_fails_before_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(storage, "free_bytes", lambda _: 100)
    with pytest.raises(storage.LowDiskSpaceError, match="insufficient disk"):
        storage.require_free_space(tmp_path / "artifact.bin", 80, "test write", headroom_bytes=30)


@pytest.mark.contract
def test_file_retention_only_removes_files_older_than_cutoff(tmp_path: Path) -> None:
    old = tmp_path / "old.log"
    recent = tmp_path / "recent.log"
    old.write_text("old", encoding="utf-8")
    recent.write_text("recent", encoding="utf-8")
    cutoff = dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
    old_time = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    recent_time = dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc).timestamp()
    os.utime(old, (old_time, old_time))
    os.utime(recent, (recent_time, recent_time))

    assert retention.prune_files_older_than(tmp_path, cutoff) == 1
    assert not old.exists()
    assert recent.exists()


@pytest.mark.contract
def test_database_retention_preserves_recent_and_bounded_backtests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "retention.db"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE signals_history(date DATE)")
        connection.execute("CREATE TABLE ml_predictions(predicted_at TIMESTAMP)")
        connection.execute("CREATE TABLE ml_training_log(trained_at TIMESTAMP)")
        connection.execute("CREATE TABLE ohlc_forecasts(forecast_date DATE)")
        connection.execute("CREATE TABLE backtest_runs(job_id TEXT, status TEXT, completed_at TIMESTAMP)")
        connection.execute("CREATE TABLE backtest_trades(job_id TEXT)")
        for table, column in (
            ("signals_history", "date"),
            ("ml_predictions", "predicted_at"),
            ("ml_training_log", "trained_at"),
            ("ohlc_forecasts", "forecast_date"),
        ):
            connection.execute(f"INSERT INTO {table} ({column}) VALUES ('2010-01-01'), ('2026-08-20')")
        connection.execute(
            "INSERT INTO backtest_runs VALUES ('old', 'completed', '2010-01-01'), ('new', 'completed', '2026-08-20')"
        )
        connection.execute("INSERT INTO backtest_trades VALUES ('old'), ('new')")

    monkeypatch.setattr(retention, "BACKTEST_MINIMUM_RUNS", 1)
    instant = dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)

    def dispatch(operation, _payload, active_database):
        assert operation is WriteOperation.APPLY_RETENTION
        return retention.apply_database_retention(active_database, now=instant)

    queue = tmp_path / "writer.sqlite"
    service = WriterService(database, queue_path=queue, dispatcher=dispatch)
    service.start()
    try:
        result = submit_write(
            WriteOperation.APPLY_RETENTION,
            {},
            database=database,
            queue_path=queue,
        )
    finally:
        service.stop()
    assert result["backtest_runs"] == 1
    with duckdb.connect(str(database), read_only=True) as connection:
        for table in ("signals_history", "ml_predictions", "ml_training_log", "ohlc_forecasts"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
        assert connection.execute("SELECT job_id FROM backtest_runs").fetchall() == [("new",)]
        assert connection.execute("SELECT job_id FROM backtest_trades").fetchall() == [("new",)]


@pytest.mark.contract
def test_automatic_raw_bhav_retention_is_disabled_even_when_database_has_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    equity = tmp_path / "equity"
    fo = tmp_path / "fo"
    equity.mkdir()
    fo.mkdir()
    loaded = equity / "20200102.csv"
    unprocessed = equity / "20200103.csv"
    recent = fo / "20260820.csv"
    for path in (loaded, unprocessed, recent):
        path.write_text("fixture", encoding="utf-8")
    database = tmp_path / "terminal.db"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE ohlcv(date DATE)")
        connection.execute("CREATE TABLE fo_eod(date DATE)")
        connection.execute("INSERT INTO ohlcv VALUES ('2020-01-02')")
        connection.execute("INSERT INTO fo_eod VALUES ('2026-08-20')")
    monkeypatch.setattr(retention, "EQUITY_BHAV_DIR", equity)
    monkeypatch.setattr(retention, "FO_BHAV_DIR", fo)

    result = retention.prune_loaded_bhavcopy_files(
        database,
        now=dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc),
    )
    assert result == {"equity": 0, "fo": 0}
    assert loaded.exists()
    assert unprocessed.exists()
    assert recent.exists()


@pytest.mark.contract
def test_runtime_retention_runs_at_most_daily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "runtime.db"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE signals_history(date DATE)")
        connection.execute("CREATE TABLE ml_predictions(predicted_at TIMESTAMP)")
        connection.execute("CREATE TABLE ml_training_log(trained_at TIMESTAMP)")
        connection.execute("CREATE TABLE ohlc_forecasts(forecast_date DATE)")
    monkeypatch.setattr(retention, "RETENTION_STATE_PATH", tmp_path / "retention-state.json")
    monkeypatch.setattr(retention, "EQUITY_BHAV_DIR", tmp_path / "equity")
    monkeypatch.setattr(retention, "FO_BHAV_DIR", tmp_path / "fo")
    monkeypatch.setattr(retention, "QUARANTINE_DIR", tmp_path / "quarantine")
    instant = dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)

    def dispatch(operation, payload, active_database):
        assert operation is WriteOperation.APPLY_RETENTION
        return retention.apply_runtime_retention(
            active_database,
            now=dt.datetime.fromisoformat(str(payload["now"])),
        )

    queue = tmp_path / "writer.sqlite"
    service = WriterService(database, queue_path=queue, dispatcher=dispatch)
    service.start()
    try:
        first = submit_write(
            WriteOperation.APPLY_RETENTION,
            {"now": instant.isoformat()},
            database=database,
            queue_path=queue,
        )
        second = submit_write(
            WriteOperation.APPLY_RETENTION,
            {"now": (instant + dt.timedelta(hours=1)).isoformat()},
            database=database,
            queue_path=queue,
        )
    finally:
        service.stop()
    assert first["skipped"] is False
    assert second == {"skipped": True, "reason": "retention already completed today"}
