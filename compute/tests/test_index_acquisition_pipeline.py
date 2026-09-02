from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import pytest

from auto_collector import smart_collector
from db.bhav_loader import _insert_ohlcv_rows, _parse_index_csv
from db.duckdb_client import _create_schema_objects_at, _writer_transaction_scope, open_read_connection
from db.migrations import migrate_database
from db.write_service import WriteOperation
from db.writer_lease import release_writer_lease, require_writer_lease


# Schema-accurate fixture matching NSE's ind_close_all_DDMMYYYY.csv columns as
# consumed by db/bhav_loader.py's INDEX_URL + _parse_index_row. No network is
# available in this environment, so this proves the parse -> insert ->
# decision_eligible_ohlcv mechanism is correct; it does not prove NSE's current
# live schema matches (that needs a real download, blocked here).
INDEX_CSV = (
    "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
    "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),"
    "P/E,P/B,Div Yield\n"
    "NIFTY 50,02-01-2024,21750.00,21850.00,21700.00,21800.00,50.00,0.23,250000000,15000.00,22.5,3.5,1.2\n"
    "NIFTY BANK,02-01-2024,47000.00,47300.00,46900.00,47150.00,150.00,0.32,50000000,8000.00,,,\n"
    "NIFTY IT,02-01-2024,35000.00,35200.00,34900.00,35100.00,100.00,0.29,10000000,2000.00,,,\n"
)


@pytest.fixture()
def index_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "terminal.db"
    require_writer_lease(db_path)
    migrate_database(db_path, schema_initializer=_create_schema_objects_at)
    try:
        yield db_path
    finally:
        release_writer_lease()


def test_index_csv_parses_only_recognised_indices() -> None:
    rows = _parse_index_csv(INDEX_CSV)
    symbols = sorted(r["symbol"] for r in rows)
    assert symbols == ["NIFTY 50", "NIFTY BANK"]  # NIFTY IT is not a tracked index symbol
    assert all(r["source"] == "NSE_INDEX_BHAVCOPY" for r in rows)


def test_index_rows_become_decision_eligible_via_real_pipeline(index_db: Path) -> None:
    rows = _parse_index_csv(INDEX_CSV)
    connection = duckdb.connect(str(index_db))
    try:
        with _writer_transaction_scope(connection):
            _insert_ohlcv_rows(rows)
    finally:
        connection.close()

    con = open_read_connection(index_db)
    try:
        eligible = con.execute(
            "SELECT symbol, source, quality, decision_eligible FROM decision_eligible_ohlcv "
            "ORDER BY symbol"
        ).fetchall()
    finally:
        con.close()

    assert eligible == [
        ("NIFTY 50", "NSE_INDEX_BHAVCOPY", "OBSERVED_VERIFIED", True),
        ("NIFTY BANK", "NSE_INDEX_BHAVCOPY", "OBSERVED_VERIFIED", True),
    ]


def test_real_index_row_supersedes_legacy_synthetic_squatter(index_db: Path) -> None:
    # A legacy FUTIDX-derived synthetic row already occupies (NIFTY 50, 2024-01-02)
    # -- the exact PK a genuine downloaded index row would also use. The real row
    # must win; it must never be silently dropped by an already-occupied key.
    connection = duckdb.connect(str(index_db))
    try:
        with _writer_transaction_scope(connection) as writer:
            writer.execute(
                "INSERT INTO ohlcv VALUES ('NIFTY 50', DATE '2024-01-02', 1,1,1,1,1,'fo_eod_fut')"
            )
    finally:
        connection.close()

    rows = _parse_index_csv(INDEX_CSV)
    connection = duckdb.connect(str(index_db))
    try:
        with _writer_transaction_scope(connection):
            _insert_ohlcv_rows(rows)
    finally:
        connection.close()

    con = open_read_connection(index_db)
    try:
        row = con.execute(
            "SELECT source, close FROM ohlcv WHERE symbol='NIFTY 50' AND date=DATE '2024-01-02'"
        ).fetchall()
    finally:
        con.close()
    assert row == [("NSE_INDEX_BHAVCOPY", 21800.00)]


def test_no_fo_rows_participate_in_index_ingestion(index_db: Path) -> None:
    rows = _parse_index_csv(INDEX_CSV)
    connection = duckdb.connect(str(index_db))
    try:
        with _writer_transaction_scope(connection):
            _insert_ohlcv_rows(rows)
    finally:
        connection.close()

    con = open_read_connection(index_db)
    try:
        fo_count = con.execute("SELECT count(*) FROM fo_eod").fetchall()[0][0]
    finally:
        con.close()
    assert fo_count == 0


# ---- collector scheduling regression --------------------------------------
#
# Root cause of the NIFTY 50 index-history freshness gap (audited 2026-09-02):
# db.bhav_loader.download_index_bhav/load_date_range existed and worked, but
# were wired ONLY into main.py's one-shot startup seed (`if count < 200`),
# which permanently stops firing once the table clears that threshold. There
# was no entry in smart_collector.TASK_RUNNERS and no pending-task detection
# for it at all, so the dataset silently stopped advancing forever while
# every other scheduled dataset (equity, F&O, FII/DII, signals, BIF) kept
# updating daily. These tests pin the fix: a real recurring "bhavcopy_index"
# task, discovered as pending once "ohlcv" WHERE symbol='NIFTY 50' falls
# behind, that reuses the existing parse function and the generic
# INGEST_OHLCV writer op (no new writer surface introduced).


def test_bhavcopy_index_task_is_registered() -> None:
    assert "bhavcopy_index" in smart_collector.TASK_RUNNERS
    assert smart_collector.TASK_RUNNERS["bhavcopy_index"] is smart_collector.run_bhavcopy_index


def test_run_bhavcopy_index_inserts_via_generic_ingest_writer_op(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rows = _parse_index_csv(INDEX_CSV)
    assert sample_rows  # fixture sanity: NIFTY 50 + NIFTY BANK rows parsed

    monkeypatch.setattr("db.bhav_loader.download_index_bhav", lambda day: sample_rows)

    captured: dict[str, object] = {}

    def fake_submit_write(operation, payload):
        captured["operation"] = operation
        captured["payload"] = payload
        return {"inserted": len(payload["rows"])}

    monkeypatch.setattr(smart_collector, "submit_write", fake_submit_write)
    monkeypatch.setattr(smart_collector, "row_count_for_date", lambda dataset, for_date: len(sample_rows))

    details = smart_collector.run_bhavcopy_index(datetime.date(2024, 1, 2))

    assert captured["operation"] is WriteOperation.INGEST_OHLCV
    assert captured["payload"] == {"rows": sample_rows}
    assert details == f"inserted={len(sample_rows)} rows_on_date={len(sample_rows)}"


def test_run_bhavcopy_index_holiday_is_quiet_wait_not_loud_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("db.bhav_loader.download_index_bhav", lambda day: None)

    details = smart_collector.run_bhavcopy_index(datetime.date(2024, 1, 26))
    assert "not_available_yet" in details

    success, returned_details = smart_collector.run_task("bhavcopy_index", datetime.date(2024, 1, 26))
    assert success is False
    assert "not_available_yet" in returned_details


def test_stale_nifty_index_history_is_scheduled_as_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact defect reproduced: equity/F&O are fresh (no pending task) but
    the index table ('ohlcv' filtered to symbol='NIFTY 50') is 10 trading days
    behind -- get_pending_tasks must surface a catch-up bhavcopy_index task
    instead of silently never noticing."""
    today = datetime.date(2026, 9, 2)
    monkeypatch.setattr(smart_collector, "ist_today", lambda: today)
    monkeypatch.setattr(smart_collector, "ist_hour", lambda: 17.0)
    monkeypatch.setattr(smart_collector, "instrument_master_age_hours", lambda: 0.0)
    monkeypatch.setattr(smart_collector, "task_done", lambda *_args: False)
    monkeypatch.setattr(smart_collector, "days_since_last_success", lambda *_args: 0)
    monkeypatch.setattr(
        smart_collector,
        "ensure_fii_dii_table",
        lambda: (_ for _ in ()).throw(RuntimeError("fii/dii not exercised in this test")),
    )

    def latest(dataset: str, symbol: str | None = None):
        values = {
            ("ohlcv", None): today,
            ("fo_eod", None): today,
            ("ohlcv", "NIFTY 50"): today - datetime.timedelta(days=12),
            ("signals_source_ohlcv", "NIFTY 50"): today,
            ("signals_history", "NIFTY"): today,
            ("bif_history", "NIFTY"): today,
        }
        return values[(dataset, symbol)]

    monkeypatch.setattr(smart_collector, "latest_date_in", latest)

    pending = smart_collector.get_pending_tasks()

    index_tasks = [item for item in pending if item["task"] == "bhavcopy_index"]
    assert index_tasks, "stale NIFTY 50 index history must produce a pending bhavcopy_index task"
    assert all(item["priority"] == 2 for item in index_tasks)
    assert all(item["date"] > today - datetime.timedelta(days=12) for item in index_tasks)
