from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from db import duckdb_client
from technical.indicators import compute_indicators


def _seed_ohlcv(database: Path, days: int, *, start_close: float = 24000.0) -> None:
    """Seed `days` weekday bars ending 2 days before the real wall-clock
    date. decision_eligible_ohlcv hard-filters `date <= CURRENT_DATE`
    (data/market_data_quality.py), so rows must never land in the future
    regardless of what from_date/to_date a caller passes."""
    with duckdb.connect(str(database)) as connection:
        day = datetime.date.today() - datetime.timedelta(days=2)
        close = start_close
        inserted = 0
        while inserted < days:
            if day.weekday() < 5:
                connection.execute(
                    "INSERT INTO ohlcv VALUES ('NIFTY 50', ?, ?, ?, ?, ?, 0, 'NSE_INDEX_BHAVCOPY')",
                    [day.isoformat(), close, close + 50, close - 50, close],
                )
                close += 1.0
                inserted += 1
            day -= datetime.timedelta(days=1)


@pytest.fixture()
def seeded_db(isolated_app_data: Path):
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    return database


@pytest.mark.contract
def test_ema_succeeds_with_sufficient_canonical_history(seeded_db: Path) -> None:
    _seed_ohlcv(seeded_db, days=260)
    result = compute_indicators("NIFTY", "1d", None, None, [{"name": "EMA", "params": {"period": 200}}])
    values = result["indicators"]["EMA_200"]
    assert len(values) == len(result["timestamps"])
    assert values[-1] is not None, "EMA200 must produce a real value once 200+ bars exist"


@pytest.mark.contract
def test_rsi_succeeds_with_sufficient_canonical_history(seeded_db: Path) -> None:
    _seed_ohlcv(seeded_db, days=60)
    result = compute_indicators("NIFTY", "1d", None, None, [{"name": "RSI", "params": {"period": 14}}])
    assert result["indicators"]["RSI_14"][-1] is not None


@pytest.mark.contract
def test_macd_succeeds_with_sufficient_canonical_history(seeded_db: Path) -> None:
    _seed_ohlcv(seeded_db, days=60)
    result = compute_indicators(
        "NIFTY", "1d", None, None, [{"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}]
    )
    assert result["indicators"]["MACD_line"][-1] is not None
    assert result["indicators"]["MACD_signal"][-1] is not None
    assert result["indicators"]["MACD_hist"][-1] is not None


@pytest.mark.contract
def test_bbands_succeeds_with_sufficient_canonical_history(seeded_db: Path) -> None:
    _seed_ohlcv(seeded_db, days=40)
    result = compute_indicators(
        "NIFTY", "1d", None, None, [{"name": "BBANDS", "params": {"period": 20, "std": 2.0}}]
    )
    assert result["indicators"]["BB_upper"][-1] is not None
    assert result["indicators"]["BB_middle"][-1] is not None
    assert result["indicators"]["BB_lower"][-1] is not None


@pytest.mark.contract
def test_insufficient_history_reports_honest_unavailable_not_fabricated(seeded_db: Path) -> None:
    """5 bars cannot support EMA200; the response must be null, never a
    fabricated/interpolated numeric value."""
    _seed_ohlcv(seeded_db, days=5)
    result = compute_indicators("NIFTY", "1d", None, None, [{"name": "EMA", "params": {"period": 200}}])
    values = result["indicators"]["EMA_200"]
    assert values, "must still report the honest (empty/null) series, not raise or fabricate"
    assert all(value is None for value in values)


@pytest.mark.contract
def test_ineligible_ohlcv_never_reaches_indicator_math(seeded_db: Path) -> None:
    """decision_eligible_ohlcv is the only source _load_ohlcv reads from
    (db/write_service.py's LOAD_TECHNICAL_OHLCV_RANGE handler). A row with an
    untrusted/ineligible provenance must not silently become valid indicator
    input -- reproduces the fail-closed decision-eligibility contract already
    enforced elsewhere in this codebase, at the indicator boundary."""
    with duckdb.connect(str(seeded_db)) as connection:
        connection.execute(
            "INSERT INTO ohlcv VALUES ('NIFTY 50', '2026-06-01', 24000, 24100, 23900, 24050, 0, 'UNVERIFIED_TEST_SOURCE')"
        )
    result = compute_indicators("NIFTY", "1d", None, None, [{"name": "SMA", "params": {"period": 1}}])
    assert result["timestamps"] == [], "an ineligible-provenance row must not surface as tradable history"


@pytest.mark.contract
def test_load_ohlcv_uses_a_tight_interactive_read_timeout(seeded_db: Path) -> None:
    """Regression for the Chart-indicators defect: the Rust client's timeout
    (src-tauri/src/commands.rs::get_indicators) must stay above whatever
    budget this call requests. Pin the value here so a future edit to either
    side is forced to keep them in sync instead of silently drifting back
    into the failure mode where Rust aborts the connection before this
    backend's own bounded read had a chance to answer."""
    _seed_ohlcv(seeded_db, days=5)
    with patch("technical.indicators.resolve_read") as mock_resolve:
        mock_resolve.return_value = {"rows": []}
        compute_indicators("NIFTY", "1d", None, None, [{"name": "SMA", "params": {"period": 1}}])
    assert mock_resolve.call_args.kwargs.get("timeout") == 8.0
