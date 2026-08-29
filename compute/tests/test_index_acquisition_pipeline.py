from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db.bhav_loader import _insert_ohlcv_rows, _parse_index_csv
from db.duckdb_client import _create_schema_objects_at, _writer_transaction_scope, open_read_connection
from db.migrations import migrate_database
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
