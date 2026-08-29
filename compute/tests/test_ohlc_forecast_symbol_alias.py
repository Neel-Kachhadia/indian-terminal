from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db import duckdb_client


@pytest.mark.contract
def test_nifty_symbol_alias_matches_decision_eligible_rows(
    isolated_app_data: Path,
) -> None:
    """Reproduces a live defect: /ml/ohlc_forecast/NIFTY returned HTTP 404 for
    every request because _load_ohlcv_full_frame special-cased "NIFTY" to the
    literal symbol list ["NIFTY"], while decision_eligible_ohlcv stores the
    index under "NIFTY 50" -- so the query never matched any row, even with a
    full, valid historical corpus present."""
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    with duckdb.connect(str(database)) as connection:
        for day in ("2026-08-18", "2026-08-19", "2026-08-20"):
            connection.execute(
                "INSERT INTO ohlcv VALUES ('NIFTY 50', ?, 24000, 24100, 23900, 24050, 0, 'NSE_INDEX_BHAVCOPY')",
                [day],
            )

    import main

    frame = main._load_ohlcv_full_frame("NIFTY")
    assert not frame.empty, "NIFTY must resolve through the same alias table as every other symbol"
    assert len(frame) == 3
