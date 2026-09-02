from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db import duckdb_client


@pytest.mark.contract
def test_stock_trainer_loads_nifty_ohlcv_through_the_canonical_alias_table(
    isolated_app_data: Path,
) -> None:
    """Reproduces a live defect: ml.stock_trainer.StockModelTrainer._load_ohlcv
    hand-duplicated its own index candidate table and special-cased "NIFTY" to
    the literal symbol list ["NIFTY"], while decision_eligible_ohlcv stores the
    index under "NIFTY 50" (verified NSE_INDEX_BHAVCOPY rows). Every per-symbol
    training run for NIFTY (m01, m02, m05, m06, m08, m12-m15, m17, m20, m21)
    therefore saw 0 eligible rows and failed "insufficient data: 0/100 rows"
    from the day the index-quality source filter tightened -- the exact same
    bug class main._load_ohlcv_full_frame had (see
    test_ohlc_forecast_symbol_alias.py), just re-introduced via a second,
    independent hand-written candidate table instead of the shared one in
    data/index_aliases.py."""
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    with duckdb.connect(str(database)) as connection:
        for day in ("2026-08-18", "2026-08-19", "2026-08-20"):
            connection.execute(
                "INSERT INTO ohlcv VALUES ('NIFTY 50', ?, 24000, 24100, 23900, 24050, 0, 'NSE_INDEX_BHAVCOPY')",
                [day],
            )
            connection.execute(
                "INSERT INTO ohlcv VALUES ('NIFTY BANK', ?, 51000, 51200, 50800, 51050, 0, 'NSE_INDEX_BHAVCOPY')",
                [day],
            )

    from ml.stock_trainer import StockModelTrainer

    trainer = StockModelTrainer()
    nifty_frame = trainer._load_ohlcv("NIFTY")
    assert not nifty_frame.empty, "NIFTY must resolve through the same alias table as every other symbol"
    assert len(nifty_frame) == 3

    banknifty_frame = trainer._load_ohlcv("BANKNIFTY")
    assert not banknifty_frame.empty
    assert len(banknifty_frame) == 3
