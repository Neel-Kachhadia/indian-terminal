from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest

from data.market_data_quality import INDEX_SYMBOLS
from db.bhav_loader import get_missing_dates
from db.duckdb_client import _create_schema_objects_at, _writer_transaction_scope
from db.migrations import migrate_database
from db.writer_lease import release_writer_lease, require_writer_lease


TUE = dt.date(2024, 1, 2)  # a real trading day, avoids weekend filtering noise


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "terminal.db"
    require_writer_lease(db_path)
    migrate_database(db_path, schema_initializer=_create_schema_objects_at)

    connection = duckdb.connect(str(db_path))
    monkeypatch.setattr("db.write_service._lookup_writer_connection", lambda database: connection)

    def insert(symbol: str, day: dt.date, source: str = "NSE_INDEX_BHAVCOPY") -> None:
        with _writer_transaction_scope(connection) as writer:
            writer.execute(
                "INSERT INTO ohlcv VALUES (?, ?, 100, 110, 90, 105, 10, ?)",
                [symbol, day, source],
            )

    try:
        yield insert
    finally:
        connection.close()
        release_writer_lease()


def test_nifty_present_does_not_hide_banknifty_gap(db) -> None:
    db("NIFTY 50", TUE)
    missing = get_missing_dates(TUE, TUE, symbols=("NIFTY BANK",))
    assert TUE in missing


def test_banknifty_present_does_not_hide_nifty_gap(db) -> None:
    db("NIFTY BANK", TUE)
    missing = get_missing_dates(TUE, TUE, symbols=("NIFTY 50",))
    assert TUE in missing


def test_both_present_neither_missing(db) -> None:
    db("NIFTY 50", TUE)
    db("NIFTY BANK", TUE)
    assert get_missing_dates(TUE, TUE, symbols=("NIFTY 50",)) == []
    assert get_missing_dates(TUE, TUE, symbols=("NIFTY BANK",)) == []


def test_alias_normalization_banknifty_equals_nifty_bank(db) -> None:
    db("NIFTY BANK", TUE)
    assert get_missing_dates(TUE, TUE, symbols=("BANKNIFTY",)) == []


def test_non_canonical_index_symbol_rejected(db) -> None:
    with pytest.raises(ValueError):
        get_missing_dates(TUE, TUE, symbols=("NOT_AN_INDEX",))


def test_legacy_futidx_derived_nifty_row_does_not_count_as_real_index_coverage(db) -> None:
    # A synthetic FUTIDX-derived row under symbol 'NIFTY' must never suppress
    # real NIFTY 50 index acquisition -- only canonically-sourced rows count.
    db("NIFTY", TUE, source="FO_EOD_FUTIDX_DERIVED")
    missing = get_missing_dates(TUE, TUE, symbols=("NIFTY 50",))
    assert TUE in missing


def test_default_symbol_set_is_canonical_nifty_50() -> None:
    assert "NIFTY 50" in INDEX_SYMBOLS
