from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest

from backtest.options_pricer import HistoricalOptionsPricer
from backtest.spot_quality import (
    SpotFallbackPolicy,
    SpotQuality,
    classify_live_spot,
    classify_spot_row,
)
from ml import ohlcv_backfill


@pytest.fixture
def spot_db() -> duckdb.DuckDBPyConnection:
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
    connection.execute(
        """
        CREATE TABLE ohlcv (
            symbol TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, volume BIGINT, source TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE VIEW decision_eligible_ohlcv AS
        SELECT *, CAST(date AS TIMESTAMP) AS observed_at
        FROM ohlcv
        WHERE upper(source) IN ('NSE_INDEX_BHAVCOPY', 'NSE_EQUITY_BHAVCOPY')
        """
    )
    yield connection
    connection.close()


def _spot(connection: duckdb.DuckDBPyConnection, date: str, source: str, close: float = 24_000) -> None:
    connection.execute(
        "INSERT INTO ohlcv VALUES ('NIFTY 50', CAST(? AS DATE), 1, 1, 1, ?, 1, ?)",
        [date, close, source],
    )


def test_derived_futures_source_cannot_feed_spot(spot_db: duckdb.DuckDBPyConnection) -> None:
    _spot(spot_db, "2026-08-21", "fo_eod_futidx_derived")
    observation = HistoricalOptionsPricer(spot_db).get_spot_observation("NIFTY", "2026-08-21")
    assert observation.quality is SpotQuality.UNAVAILABLE
    assert observation.valid is False


def test_missing_verified_spot_fails_closed(spot_db: duckdb.DuckDBPyConnection) -> None:
    observation = HistoricalOptionsPricer(spot_db).get_spot_observation("NIFTY", "2026-08-21")
    assert observation.quality is SpotQuality.UNAVAILABLE
    assert observation.price is None


def test_spot_freshness_policy(spot_db: duckdb.DuckDBPyConnection) -> None:
    _spot(spot_db, "2026-08-20", "NSE_INDEX_BHAVCOPY")
    pricer = HistoricalOptionsPricer(spot_db)
    assert pricer.get_spot_observation("NIFTY", "2026-08-20").quality is SpotQuality.VERIFIED
    assert pricer.get_spot_observation("NIFTY", "2026-08-21").quality is SpotQuality.STALE
    assert pricer.get_spot_observation(
        "NIFTY",
        "2026-08-21",
        fallback_policy=SpotFallbackPolicy.PREVIOUS_SESSION,
        maximum_staleness_sessions=1,
    ).quality is SpotQuality.VERIFIED
    assert pricer.get_spot_observation(
        "NIFTY",
        "2026-08-24",
        fallback_policy=SpotFallbackPolicy.PREVIOUS_SESSION,
        maximum_staleness_sessions=1,
    ).quality is SpotQuality.STALE


def test_spot_observation_exposes_complete_fallback_provenance(
    spot_db: duckdb.DuckDBPyConnection,
) -> None:
    _spot(spot_db, "2026-01-14", "NSE_INDEX_BHAVCOPY")
    observation = HistoricalOptionsPricer(spot_db).get_spot_observation(
        "NIFTY",
        "2026-01-15",
        fallback_policy=SpotFallbackPolicy.PREVIOUS_SESSION,
        maximum_staleness_sessions=0,
    )
    assert observation.valid is True
    assert observation.requested_as_of == dt.date(2026, 1, 15)
    assert observation.actual_observed_at == dt.datetime(
        2026, 1, 14, tzinfo=dt.timezone.utc
    )
    assert observation.source == "NSE_INDEX_BHAVCOPY"
    assert observation.staleness_sessions == 0
    assert observation.fallback_used is True
    assert observation.fallback_policy is SpotFallbackPolicy.PREVIOUS_SESSION
    assert observation.as_provenance() == {
        "instrument": "NIFTY",
        "requested_as_of": "2026-01-15",
        "actual_observed_at": "2026-01-14T00:00:00+00:00",
        "source": "NSE_INDEX_BHAVCOPY",
        "quality": "VERIFIED",
        "reason": "verified_source_and_freshness",
        "staleness_sessions": 0,
        "maximum_staleness_sessions": 0,
        "fallback_used": True,
        "fallback_policy": "PREVIOUS_SESSION",
        "freshness": "POLICY_ALLOWED_FALLBACK",
        "decision_eligible": True,
    }


def test_previous_session_is_forbidden_by_default(spot_db: duckdb.DuckDBPyConnection) -> None:
    _spot(spot_db, "2026-08-20", "NSE_INDEX_BHAVCOPY")
    observation = HistoricalOptionsPricer(spot_db).get_spot_observation("NIFTY", "2026-08-21")
    assert observation.quality is SpotQuality.STALE
    assert observation.fallback_used is True
    assert observation.fallback_policy is SpotFallbackPolicy.NO_FALLBACK


@pytest.mark.parametrize("source", ["UNKNOWN", "fo_eod_futidx_derived"])
def test_unknown_spot_source_is_invalid(source: str) -> None:
    observation = classify_spot_row(
        {"date": "2026-08-21", "close": 24_000, "source": source},
        symbol="NIFTY",
        requested_as_of=dt.date(2026, 8, 21),
    )
    assert observation.quality is SpotQuality.INVALID_SOURCE


def test_future_spot_observation_is_invalid() -> None:
    observation = classify_spot_row(
        {"date": "2026-08-22", "close": 24_000, "source": "NSE_INDEX_BHAVCOPY"},
        symbol="NIFTY",
        requested_as_of=dt.date(2026, 8, 21),
    )
    assert observation.quality is SpotQuality.FUTURE_OBSERVATION


@pytest.mark.parametrize(
    ("override", "expected_quality"),
    [
        ({}, SpotQuality.VERIFIED),
        ({"symbol": "BANKNIFTY"}, SpotQuality.INVALID_SOURCE),
        ({"source_type": "UNKNOWN"}, SpotQuality.INVALID_SOURCE),
        ({"freshness": "STALE"}, SpotQuality.INVALID_SOURCE),
        ({"fallback": True}, SpotQuality.INVALID_SOURCE),
        ({"market_state": "DISCONNECTED"}, SpotQuality.INVALID_SOURCE),
        ({"observed_at_ms": 1_788_000_000_000}, SpotQuality.FUTURE_OBSERVATION),
    ],
)
def test_live_spot_requires_complete_fresh_matching_provenance(
    override: dict[str, object],
    expected_quality: SpotQuality,
) -> None:
    requested = dt.datetime(2026, 8, 21, 6, 0, tzinfo=dt.timezone.utc)
    envelope = {
        "symbol": "NIFTY",
        "ltp": 24_000,
        "observed_at_ms": int((requested - dt.timedelta(seconds=1)).timestamp() * 1000),
        "source_type": "ANGEL_STREAM",
        "quality": "EXECUTABLE_STREAM",
        "freshness": "FRESH",
        "market_state": "LIVE",
        "fallback": False,
        **override,
    }
    observation = classify_live_spot(envelope, symbol="NIFTY", requested_at=requested)
    assert observation.quality is expected_quality
    assert observation.decision_eligible is (expected_quality is SpotQuality.VERIFIED)


def _backfill_db(path: Path, rows: list[tuple[object, ...]]) -> None:
    connection = duckdb.connect(path)
    try:
        connection.execute(
            "CREATE TABLE ohlcv (symbol TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT, source TEXT)"
        )
        connection.execute(
            "CREATE TABLE fo_eod (symbol TEXT, expiry DATE, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT)"
        )
        connection.executemany("INSERT INTO fo_eod VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    finally:
        connection.close()


def test_unknown_zero_ohl_futures_rows_cannot_become_spot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "zero-ohl.db"
    _backfill_db(path, [("NIFTY", "2026-08-27", "2026-08-21", 0, 0, 0, 24_000, 10)])
    connection = duckdb.connect(path)
    monkeypatch.setattr("db.write_service._lookup_writer_connection", lambda database: connection)
    result = ohlcv_backfill.verified_nifty_spot_availability()
    connection.close()
    connection = duckdb.connect(path, read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == 0
    finally:
        connection.close()
    assert result["rows_added"] == 0


def test_different_expiries_cannot_be_combined_into_spot_candle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cross-expiry.db"
    _backfill_db(
        path,
        [
            ("NIFTY", "2026-08-27", "2026-08-21", 24_000, 24_100, 23_900, 24_050, 10),
            ("NIFTY", "2026-09-24", "2026-08-21", 24_200, 24_300, 24_100, 24_250, 20),
        ],
    )
    connection = duckdb.connect(path)
    monkeypatch.setattr("db.write_service._lookup_writer_connection", lambda database: connection)
    ohlcv_backfill.verified_nifty_spot_availability()
    connection.close()
    connection = duckdb.connect(path, read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == 0
    finally:
        connection.close()
