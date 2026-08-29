"""Regression tests: signals-catalogue-wide null-fabrication audit follow-up.

The IV/OI/PCR fabrication fix (see test_unknown_neutral_semantics.py) was
followed by a systematic sweep of compute/signals/*.py for the same
`(x or <neutral>)` idiom coalescing genuinely-missing source data into a
financially meaningful default before it reaches a composite score or
directional signal output. Three real defects were found and fixed here;
the remainder of the swept sites use `or 0.0` / `or inf` guards that bias a
*binary* pattern flag toward "no" on missing data (a defensible, reviewed
product decision distinct from fabricating a confident directional/
composite output) and were left as-is.
"""
from __future__ import annotations

from signals.cat01_price_action import _trend_score
from signals.cat06_cross_asset import _risk_signal
from signals.cat15_risk import _compute_values as risk_compute_values
from signals.framework import SignalContext


def test_trend_score_excludes_missing_ema_alignment_instead_of_zeroing_it() -> None:
    # Old behavior: (None or 0) / 4 * 100 == 0.0 dragged the trend composite
    # toward a fabricated bearish reading whenever ema_alignment_score was
    # unavailable. New behavior: exclude the missing piece from the average.
    values_missing = {"price_above_all_emas": True, "adx_14": None}
    values_zero_real = {
        "ema_alignment_score": 0,
        "price_above_all_emas": True,
        "adx_14": None,
    }
    assert _trend_score(values_missing) == 100.0  # only the real piece counts
    assert _trend_score(values_zero_real) == 50.0  # a genuine 0 still counts


def test_trend_score_returns_none_when_every_input_is_missing() -> None:
    assert _trend_score({}) is None


def test_trend_score_excludes_adx_direction_when_spread_missing() -> None:
    # adx_14 present (so old code entered the branch) but adx_di_spread
    # missing must not fabricate a bullish direction=1.0 via `or 0 >= 0`.
    values = {"ema_alignment_score": 4, "price_above_all_emas": True, "adx_14": 10.0}
    with_spread = {**values, "adx_di_spread": 5.0}
    assert _trend_score(values) == 100.0  # adx piece excluded, not fabricated
    assert _trend_score(with_spread) == 80.0  # (100 + 100 + 40) / 3


def test_risk_off_composite_signal_is_none_when_market_risk_score_unavailable() -> None:
    # Old behavior: (None or 0.0) > 65.0 == False -> fabricated "0" (not
    # risk-off) whenever every constituent risk piece was unavailable.
    ctx = SignalContext(
        symbol="TEST",
        ohlcv_df=None,
        chain_legs=[],
        spot=0.0,
        expiry_str="",
        cross_asset_snapshot={},
        sentiment_data={},
        fii_dii_data={},
        india_vix=None,
    )
    values = risk_compute_values(ctx)
    assert values["market_risk_score_0_100"] is None
    assert values["risk_off_composite_signal"] is None


def test_risk_signal_does_not_fabricate_directional_call_from_missing_gold_change() -> None:
    # Old behavior: (None or 0.0) <= 0 == True -> a real us/asia score above
    # the 0.2 threshold fired a confident risk_on (+1) signal even though
    # gold_change was genuinely unavailable, not actually <= 0.
    assert _risk_signal(0.5, 0.5, None) == 0
    assert _risk_signal(0.5, 0.5, -0.1) == 1
    assert _risk_signal(None, None, None) is None
