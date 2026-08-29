from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from backtest.options_pricer import HistoricalOptionsPricer
from backtest.price_quality import HistoricalPriceQuality, classify_historical_option_row
from backtest.engine import classify_result_validity


BASE_ROW = {
    "symbol": "NIFTY",
    "expiry": dt.date(2026, 8, 27),
    "strike": 24_000.0,
    "option_type": "CE",
    "date": dt.date(2026, 8, 24),
    "open": 100.0,
    "high": 115.0,
    "low": 95.0,
    "close": 110.0,
    "volume": 500,
    "source": "NSE_FO_BHAVCOPY",
    "price_quality": "VALID_EXECUTABLE_OBSERVATION",
}


def test_positive_close_alone_is_not_executable() -> None:
    row = {**BASE_ROW, "open": 0, "high": 0, "low": 0, "close": 9.5, "volume": 0}

    observation = classify_historical_option_row(row)

    assert observation.quality is HistoricalPriceQuality.UNKNOWN
    assert observation.executable is False


def test_valid_candle_volume_and_source_is_executable() -> None:
    observation = classify_historical_option_row(BASE_ROW)

    assert observation.quality is HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION
    assert observation.price == 110.0


def test_zero_volume_is_reference_not_fill() -> None:
    observation = classify_historical_option_row({**BASE_ROW, "volume": 0})

    assert observation.quality is HistoricalPriceQuality.VALID_NON_EXECUTABLE_REFERENCE


@pytest.fixture
def price_db() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE fo_eod (
            symbol TEXT, expiry DATE, strike DOUBLE, option_type TEXT, date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            oi BIGINT, oi_change BIGINT, source TEXT, price_quality TEXT
        )
        """
    )
    yield connection
    connection.close()


def _insert(connection: duckdb.DuckDBPyConnection, row: dict[str, object]) -> None:
    connection.execute(
        "INSERT INTO fo_eod VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1000, 10, ?, ?)",
        [
            row["symbol"], row["expiry"], row["strike"], row["option_type"], row["date"],
            row["open"], row["high"], row["low"], row["close"], row["volume"],
            row["source"], row["price_quality"],
        ],
    )


def test_pricer_accepts_only_executable_observation(price_db: duckdb.DuckDBPyConnection) -> None:
    _insert(price_db, BASE_ROW)

    price = HistoricalOptionsPricer(price_db).get_option_price(
        "NIFTY", "2026-08-27", 24_000, "CE", "2026-08-24"
    )

    assert price == 110.0


def test_pricer_rejects_unproven_positive_close(price_db: duckdb.DuckDBPyConnection) -> None:
    _insert(price_db, {**BASE_ROW, "open": 0, "high": 0, "low": 0, "close": 9.5, "volume": 0})

    price = HistoricalOptionsPricer(price_db).get_option_price(
        "NIFTY", "2026-08-27", 24_000, "CE", "2026-08-24"
    )

    assert price is None


def test_legacy_row_without_provenance_is_rejected() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE fo_eod (
            symbol TEXT, expiry DATE, strike DOUBLE, option_type TEXT, date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            oi BIGINT, oi_change BIGINT
        )
        """
    )
    connection.execute(
        "INSERT INTO fo_eod VALUES ('NIFTY', '2026-08-27', 24000, 'CE', '2026-08-24', 100, 115, 95, 110, 500, 1000, 10)"
    )
    try:
        assert HistoricalOptionsPricer(connection).get_option_price(
            "NIFTY", "2026-08-27", 24_000, "CE", "2026-08-24"
        ) is None
    finally:
        connection.close()


def test_legacy_backtest_result_is_quarantined() -> None:
    validity, reason = classify_result_validity({"net_return_pct": 12.5}, "completed")

    assert validity == "INVALID_LEGACY_PRICE_SEMANTICS"
    assert reason and "quarantined" in reason


def test_current_policy_result_is_valid() -> None:
    validity, reason = classify_result_validity(
        {
            "price_policy": {"version": "fo-executable-v1"},
            "total_trades": 1,
            "execution_evidence": {
                "timing_contract_satisfied": True,
                "invalid_spot_observation_count": 0,
                "invalid_price_observation_count": 0,
                "all_executed_option_fills_valid": True,
                "all_required_spot_observations_valid": True,
                "invalid_fallback_used": False,
            },
        },
        "completed",
    )

    assert validity == "VALIDATED_EXECUTABLE_PRICE_POLICY"
    assert reason is None


def test_zero_trade_run_is_not_validated() -> None:
    validity, _ = classify_result_validity(
        {
            "price_policy": {"version": "fo-executable-v1"},
            "total_trades": 0,
            "execution_evidence": {
                "timing_contract_satisfied": True,
                "invalid_spot_observation_count": 0,
                "invalid_price_observation_count": 0,
                "all_executed_option_fills_valid": False,
                "all_required_spot_observations_valid": True,
                "invalid_fallback_used": False,
            },
        },
        "completed",
    )
    assert validity == "NO_EXECUTIONS"


def test_one_invalid_observation_prevents_validated_status() -> None:
    validity, _ = classify_result_validity(
        {
            "price_policy": {"version": "fo-executable-v1"},
            "total_trades": 1,
            "execution_evidence": {
                "timing_contract_satisfied": True,
                "invalid_spot_observation_count": 0,
                "invalid_price_observation_count": 1,
                "all_executed_option_fills_valid": True,
                "all_required_spot_observations_valid": True,
                "invalid_fallback_used": False,
            },
        },
        "completed",
    )
    assert validity == "INVALID_PRICE_DATA"
