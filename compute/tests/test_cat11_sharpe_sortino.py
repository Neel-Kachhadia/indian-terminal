from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

import signals.cat11_statistical as cat11
from data.risk_free_rate import RateSelection


def _fake_rate(rate_decimal: float) -> RateSelection:
    today = dt.date.today()
    return RateSelection(
        observation_date=today,
        requested_days=30,
        selected_tenor_days=30,
        selected_tenor_label="30 Days",
        rate_decimal=rate_decimal,
        rate_pct=rate_decimal * 100.0,
        gap_days=0,
        method="NEAREST_NEIGHBOR",
        source="FBIL_TBILL_CURVE",
        source_url="https://www.fbil.org.in/wasdm/tbill/fetch?authenticated=false",
    )


def _returns(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_sharpe_returns_none_without_a_rate() -> None:
    ret = _returns([0.01, -0.005, 0.003, 0.002, -0.001] * 3)
    assert cat11._sharpe(ret, None) is None


def test_sortino_returns_none_without_a_rate() -> None:
    ret = _returns([0.01, -0.005, 0.003, 0.002, -0.001] * 3)
    assert cat11._sortino(ret, None) is None


def test_sharpe_matches_hand_derived_formula() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(loc=0.001, scale=0.01, size=60).tolist()
    ret = _returns(values)
    rate = 0.065

    result = cat11._sharpe(ret, rate)

    excess = ret - rate / 252.0
    expected = float(excess.mean() / excess.std(ddof=0) * math.sqrt(252.0))
    assert result == pytest.approx(expected, rel=1e-12)


def test_sortino_matches_hand_derived_downside_deviation_formula() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(loc=0.001, scale=0.01, size=60).tolist()
    ret = _returns(values)
    rate = 0.065

    result = cat11._sortino(ret, rate)

    downside = ret[ret < 0]
    downside_std = downside.std(ddof=0)
    expected = float((ret.mean() - rate / 252.0) / downside_std * math.sqrt(252.0))
    assert result == pytest.approx(expected, rel=1e-12)


def test_sharpe_needs_at_least_ten_observations() -> None:
    ret = _returns([0.01, -0.005, 0.003])
    assert cat11._sharpe(ret, 0.065) is None


def test_current_risk_free_rate_uses_canonical_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cat11, "latest_available_rate", lambda t, as_of=None: _fake_rate(0.07))
    as_of = dt.datetime(2026, 9, 11, 10, tzinfo=dt.timezone.utc)
    assert cat11._current_risk_free_rate(as_of) == pytest.approx(0.07)


def test_current_risk_free_rate_stale_or_missing_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cat11, "latest_available_rate", lambda t, as_of=None: None)
    as_of = dt.datetime(2026, 9, 11, 10, tzinfo=dt.timezone.utc)
    assert cat11._current_risk_free_rate(as_of) is None


def test_current_risk_free_rate_never_raises_on_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(t, as_of=None):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(cat11, "latest_available_rate", _boom)
    as_of = dt.datetime(2026, 9, 11, 10, tzinfo=dt.timezone.utc)
    assert cat11._current_risk_free_rate(as_of) is None


def test_compute_values_sharpe_sortino_none_when_rate_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from signals.framework import SignalContext

    monkeypatch.setattr(cat11, "latest_available_rate", lambda t, as_of=None: None)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 80)))
    df = pd.DataFrame({"close": close})
    ctx = SignalContext(
        symbol="NIFTY", ohlcv_df=df, chain_legs=[], spot=float(close.iloc[-1]),
        expiry_str="", cross_asset_snapshot={}, sentiment_data={}, fii_dii_data={},
        india_vix=None,
    )
    values, _ = cat11._compute_values(ctx)
    assert values["sharpe_60d"] is None
    assert values["sortino_60d"] is None


def test_compute_values_sharpe_sortino_populated_when_rate_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from signals.framework import SignalContext

    monkeypatch.setattr(cat11, "latest_available_rate", lambda t, as_of=None: _fake_rate(0.065))
    rng = np.random.default_rng(5)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 80)))
    df = pd.DataFrame({"close": close})
    ctx = SignalContext(
        symbol="NIFTY", ohlcv_df=df, chain_legs=[], spot=float(close.iloc[-1]),
        expiry_str="", cross_asset_snapshot={}, sentiment_data={}, fii_dii_data={},
        india_vix=None,
    )
    values, _ = cat11._compute_values(ctx)
    assert values["sharpe_60d"] is not None
    assert values["sortino_60d"] is not None
    assert math.isfinite(values["sharpe_60d"])
    assert math.isfinite(values["sortino_60d"])
