from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest

from data.market_data_quality import (
    DECISION_ELIGIBLE_OHLCV_VIEW_SQL,
    DecisionContext,
    DerivationType,
    MarketDataQuality,
    classify_market_bar,
    evaluate_market_bar,
)
from bif import engine as bif_engine_module


ADVERSARIAL_BARS = (
    ("verified", "NIFTY 50", "2026-08-21", "NSE_INDEX_BHAVCOPY", True),
    ("unknown_cash", "NIFTY 50", "2026-08-20", "UNKNOWN", False),
    ("legacy_derived", "NIFTY 50", "2026-08-19", "fo_eod_futidx_derived", False),
    ("synthetic_cross_expiry", "NIFTY", "2026-08-18", "FO_EOD_FUTIDX_DERIVED", False),
    ("wrong_contract", "NIFTY 50", "2026-08-17", "NSE_EQUITY_BHAVCOPY", False),
)


@pytest.mark.contract
def test_one_canonical_view_quarantines_all_noneligible_ohlcv() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE ohlcv (
            symbol TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, volume BIGINT, source TEXT
        )
        """
    )
    for _name, symbol, day, source, _eligible in ADVERSARIAL_BARS:
        connection.execute(
            "INSERT INTO ohlcv VALUES (?, CAST(? AS DATE), 100, 110, 90, 105, 10, ?)",
            [symbol, day, source],
        )
    connection.execute(
        "INSERT INTO ohlcv VALUES ('NIFTY 50', CURRENT_DATE + INTERVAL 1 DAY, 100, 110, 90, 105, 10, 'NSE_INDEX_BHAVCOPY')"
    )
    connection.execute(DECISION_ELIGIBLE_OHLCV_VIEW_SQL)
    rows = connection.execute(
        "SELECT instrument, source, quality, derivation_type, decision_eligible FROM decision_eligible_ohlcv"
    ).fetchall()
    connection.close()
    assert rows == [("NIFTY 50", "NSE_INDEX_BHAVCOPY", "OBSERVED_VERIFIED", "NONE", True)]


@pytest.mark.parametrize(
    ("source", "derivation", "expected_quality", "eligible"),
    [
        ("NSE_INDEX_BHAVCOPY", "NONE", MarketDataQuality.OBSERVED_VERIFIED, True),
        ("UNKNOWN", "LEGACY_UNKNOWN", MarketDataQuality.UNKNOWN, False),
        ("FO_EOD_FUTIDX_DERIVED", "CROSS_EXPIRY_FUTURES", MarketDataQuality.UNKNOWN, False),
    ],
)
def test_typed_quality_contract_fails_closed(
    source: str,
    derivation: str,
    expected_quality: MarketDataQuality,
    eligible: bool,
) -> None:
    bar = classify_market_bar(
        {
            "instrument": "NIFTY 50",
            "date": "2026-08-21",
            "open": 100,
            "high": 110,
            "low": 90,
            "close": 105,
            "volume": 10,
            "source": source,
            "derivation_type": derivation,
        },
        requested_as_of=dt.date(2026, 8, 21),
    )
    assert bar.quality is expected_quality
    assert bar.decision_eligible is eligible
    if source == "FO_EOD_FUTIDX_DERIVED":
        assert bar.derivation_type is DerivationType.CROSS_EXPIRY_FUTURES


@pytest.mark.contract
def test_all_decision_consumers_use_canonical_quality_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    consumers = (
        # auto_collector/smart_collector.py no longer reads DuckDB directly at
        # all -- it goes through db/write_service.py's typed read queue
        # (ReadOperation.LOAD_OHLCV_FULL_FRAME), which is the module actually
        # enforcing this boundary on the collector's behalf now.
        "db/write_service.py",
        "backtest/engine.py",
        "backtest/options_pricer.py",
        "data/stock_snapshot.py",
        "main.py",
        "ml/models/m21_ohlc_forecast.py",
    )
    for relative in consumers:
        source = (root / relative).read_text(encoding="utf-8").lower()
        assert "decision_eligible_ohlcv" in source, relative
        # db/write_service.py also hosts a couple of typed reads against the
        # raw ohlcv table on behalf of callers with a genuinely different,
        # non-decision purpose: LOAD_RAW_OHLCV_FOR_VALIDATION (validating
        # raw bhavcopy ingestion against Angel data -- comparing sources IS
        # the point, so filtering to the decision-quality view would defeat
        # it) and LOAD_NIFTY_SPOT_AVAILABILITY_STATS (spot provenance
        # diagnostics counting verified vs quarantined raw rows). Neither
        # caller (data/nse_bhavcopy.py, ml/ohlcv_backfill.py) was ever a
        # decision consumer subject to this boundary.
        if relative not in ("main.py", "db/write_service.py"):
            assert "from ohlcv" not in source, relative

    # These modules no longer read DuckDB directly at all -- their
    # decision_eligible_ohlcv queries were relocated into
    # db/write_service.py::_dispatch_read (LOAD_DECISION_ELIGIBLE_OHLCV,
    # LOAD_TECHNICAL_OHLCV_RANGE, LOAD_SCREENER_UNIVERSE_HISTORY), which is
    # already in the consumers list above and is the module actually
    # enforcing this boundary on their behalf now via resolve_read().
    typed_read_consumers = (
        "bif/engine.py",
        "technical/indicators.py",
        "technical/screener.py",
        "signals/backfill.py",
        "ml/stock_trainer.py",
        "ml/feature_engineering.py",
        "intelligence/alpha.py",
        "risk/correlation.py",
        "futures/basis.py",
    )
    for relative in typed_read_consumers:
        source = (root / relative).read_text(encoding="utf-8").lower()
        assert "resolve_read" in source, relative
        assert "from ohlcv" not in source, relative
        assert "duckdb.connect" not in source, relative


@pytest.mark.parametrize(
    ("row", "eligible"),
    [
        ({"date": "2026-08-21", "source": "NSE_INDEX_BHAVCOPY"}, True),
        ({"date": "2026-08-20", "source": "NSE_INDEX_BHAVCOPY"}, False),
        ({"date": "2026-08-22", "source": "NSE_INDEX_BHAVCOPY"}, False),
        ({"date": "2026-08-21", "source": "UNKNOWN"}, False),
        ({"date": "2026-08-21", "source": "fo_eod_futidx_derived"}, False),
        ({"date": "2026-08-21", "source": "NSE_INDEX_BHAVCOPY", "high": 90}, False),
    ],
)
def test_current_decision_boundary_combines_semantic_and_temporal_proof(
    row: dict[str, object],
    eligible: bool,
) -> None:
    candidate = {
        "instrument": "NIFTY 50",
        "date": "2026-08-21",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 10,
        "derivation_type": "NONE",
        **row,
    }
    result = evaluate_market_bar(
        candidate,
        context=DecisionContext.CURRENT_DECISION,
        requested_as_of=dt.datetime(2026, 8, 21, 11, 0, tzinfo=dt.timezone.utc),
    )
    assert result.decision_eligible is eligible


def test_actual_bif_current_loader_rejects_stale_latest_verified_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = {
        "instrument": "NIFTY 50",
        "date": "2026-08-20",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 10,
        "source": "NSE_INDEX_BHAVCOPY",
        "derivation_type": "NONE",
    }
    monkeypatch.setattr(bif_engine_module, "_load_ohlcv_rows", lambda _symbol: [stale])
    rows, eligibility = bif_engine_module._load_current_ohlcv_rows(
        "NIFTY",
        dt.datetime(2026, 8, 21, 11, 0, tzinfo=dt.timezone.utc),
    )
    assert rows == [stale]
    assert eligibility.semantic_quality_eligible is True
    assert eligibility.temporal_eligible is False
    assert eligibility.decision_eligible is False
