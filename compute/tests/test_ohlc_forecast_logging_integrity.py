from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from db import duckdb_client
from db.duckdb_client import _writer_transaction_scope
from db.writer_lease import release_writer_lease, require_writer_lease


def _synthetic_ohlcv(periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-01", periods=periods)
    return pd.DataFrame(
        {
            "date": dates.date,
            "open": 50000.0,
            "high": 50200.0,
            "low": 49800.0,
            "close": 50000.0,
            "volume": 1_000,
        }
    )


@pytest.mark.contract
def test_banknifty_forecast_actuals_resolve_through_the_canonical_alias_table(
    isolated_app_data: Path,
) -> None:
    """Reproduces a live defect: OHLCForecaster._update_actuals hand-duplicated
    its own partial index candidate table -- only NIFTY -> "NIFTY 50", never
    BANKNIFTY -> "NIFTY BANK" -- the same bug class as
    ml.stock_trainer._load_ohlcv (see test_stock_trainer_index_symbol_alias.py).
    Verified actual OHLCV for the index is stored under "NIFTY BANK", so
    forecast.actual_close could never be resolved for BANKNIFTY: every
    forecast stayed permanently unresolved (actual_close IS NULL forever) and
    kept re-triggering new duplicate rows on each subsequent collector run."""
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.models.m21_ohlc_forecast import OHLCForecaster

        ohlcv = _synthetic_ohlcv()
        forecaster = OHLCForecaster()
        preview = forecaster.predict("BANKNIFTY", ohlcv, 50000.0, log_forecast=False)
        forecast_date = preview["date"]

        # First call creates the (unresolved) forecast row for forecast_date.
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                forecaster.predict("BANKNIFTY", ohlcv, 50000.0, log_forecast=True)

        with duckdb.connect(str(database)) as connection:
            connection.execute(
                "INSERT INTO ohlcv VALUES ('NIFTY BANK', CAST(? AS DATE), 50100, 50300, 49900, 50150, 1000, 'NSE_INDEX_BHAVCOPY')",
                [forecast_date],
            )

        # log_forecast() resolves outstanding actuals for prior forecasts
        # before inserting a new one -- this second call is what should
        # backfill actual_close on the row created above.
        next_ohlcv = pd.concat(
            [ohlcv, pd.DataFrame([{"date": pd.to_datetime(forecast_date).date(), "open": 50100.0, "high": 50300.0, "low": 49900.0, "close": 50150.0, "volume": 1000}])],
            ignore_index=True,
        )
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                forecaster.predict("BANKNIFTY", next_ohlcv, 50150.0, log_forecast=True)

        with duckdb.connect(str(database), read_only=True) as connection:
            rows = connection.execute(
                "SELECT actual_close, close_mae FROM ohlc_forecasts WHERE symbol = 'BANKNIFTY' AND forecast_date = CAST(? AS DATE)",
                [forecast_date],
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 50150.0, "actual_close must resolve via the 'NIFTY BANK' verified row"
        assert rows[0][1] is not None
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_relogging_the_same_forecast_date_never_creates_a_duplicate_row(
    isolated_app_data: Path,
) -> None:
    """Reproduces a live defect: OHLCForecaster._insert_forecast always did a
    bare INSERT with a random row id and no (symbol, forecast_date) dedup
    check. Every scheduled forecast run for a symbol already covered for that
    target date silently added another full duplicate row -- observed live,
    54 symbols had 9+ duplicate rows each for the same forecast_date."""
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.models.m21_ohlc_forecast import OHLCForecaster

        ohlcv = _synthetic_ohlcv()
        forecaster = OHLCForecaster()

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                result_1 = forecaster.predict("RELIANCE", ohlcv, 50000.0, log_forecast=True)
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                result_2 = forecaster.predict("RELIANCE", ohlcv, 50010.0, log_forecast=True)
        assert result_1["date"] == result_2["date"]

        with duckdb.connect(str(database), read_only=True) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM ohlc_forecasts WHERE symbol = 'RELIANCE' AND forecast_date = CAST(? AS DATE)",
                [result_1["date"]],
            ).fetchone()[0]
        assert count == 1, "a second forecast run for an already-covered target date must not duplicate the row"
    finally:
        release_writer_lease()
