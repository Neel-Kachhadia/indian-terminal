from __future__ import annotations

from typing import Any

from .common import current_price, historical_vol, last, numeric_series, safe_div, z_score
from .framework import (
    DataSource,
    SignalRegime,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "vix"

SIGNAL_NAMES = [
    "vix_current",
    "vix_regime",
    "vix_regime_score",
    "vix_vs_20d_avg_pct",
    "vix_vs_52w_avg_pct",
    "vix_spike_15pct",
    "vix_spike_30pct",
    "vix_trend_5d_pct",
    "vix_percentile_52w",
    "vix_mean_reversion_z",
    "vix_nifty_correlation_30d",
    "vix_nifty_divergence",
    "vix_realised_spread",
    "vix_term_structure",
    "vix_historical_percentile",
    "vix_complacency_signal",
    "vix_fear_signal",
    "vix_extreme_fear_signal",
    "vix_vol_of_vol_proxy",
    "vix_vs_atr_regime",
    "vix_expected_daily_move_pct",
    "vix_expected_weekly_move_pct",
    "vix_risk_adjusted_momentum",
    "vix_normalised_0_100",
    "vix_regime_shift_warning",
]

TYPE_OVERRIDES = {
    "vix_regime": SignalType.CATEGORICAL,
    "vix_spike_15pct": SignalType.BINARY,
    "vix_spike_30pct": SignalType.BINARY,
    "vix_nifty_divergence": SignalType.DIRECTIONAL,
    "vix_complacency_signal": SignalType.BINARY,
    "vix_fear_signal": SignalType.BINARY,
    "vix_extreme_fear_signal": SignalType.BINARY,
    "vix_regime_shift_warning": SignalType.BINARY,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.FII_DII,
    type_overrides=TYPE_OVERRIDES,
    regime_overrides={
        "vix_spike_30pct": SignalRegime.HIGH_VOL,
        "vix_fear_signal": SignalRegime.HIGH_VOL,
        "vix_complacency_signal": SignalRegime.LOW_VOL,
    },
    source_overrides={
        "vix_current": DataSource.FII_DII,
        "vix_regime": DataSource.FII_DII,
    },
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values, histories = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values, histories = {}, {}
    availability = source_availability(ctx)
    availability[DataSource.FII_DII] = ctx.india_vix is not None
    return build_signals(CATEGORY, SPECS, values, availability, histories=histories)


def _compute_values(ctx: SignalContext) -> tuple[dict[str, Any], dict[str, Any]]:
    vix = ctx.india_vix
    if vix is None:
        return {}, {}
    df = ctx.ohlcv_df
    close = numeric_series(df, "close")
    hv_20 = historical_vol(close, 20)
    spot = current_price(df, ctx.spot)
    realised = last(hv_20)
    normalised = _score(vix)
    values = {
        "vix_current": vix,
        "vix_regime": _regime(vix),
        "vix_regime_score": normalised,
        "vix_spike_15pct": None,
        "vix_spike_30pct": None,
        "vix_realised_spread": vix - realised if realised is not None else None,
        "vix_term_structure": None,
        "vix_historical_percentile": None,
        "vix_complacency_signal": int(vix < 13.0),
        "vix_fear_signal": int(vix > 20.0),
        "vix_extreme_fear_signal": int(vix > 30.0),
        "vix_vs_atr_regime": safe_div(vix, realised),
        "vix_expected_daily_move_pct": vix / (252.0 ** 0.5),
        "vix_expected_weekly_move_pct": vix * (5.0 / 252.0) ** 0.5,
        "vix_normalised_0_100": normalised,
        "vix_regime_shift_warning": int(vix > 20.0 and (realised or 0.0) < 12.0),
    }
    if spot > 0:
        values["vix_risk_adjusted_momentum"] = safe_div(last(close.pct_change(5) * 100.0), vix)
    histories = {"vix_realised_spread": hv_20.tolist(), "vix_vs_atr_regime": hv_20.tolist()}
    return values, histories


def _regime(vix: float) -> str:
    if vix < 13:
        return "complacency"
    if vix > 20:
        return "fear"
    return "normal"


def _score(vix: float) -> float:
    return max(0.0, min(100.0, (vix - 10.0) / 25.0 * 100.0))
