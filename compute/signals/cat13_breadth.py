from __future__ import annotations

from typing import Any

from .common import current_price, ema, last, numeric_series
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "breadth"

SIGNAL_NAMES = [
    "stocks_above_ema50",
    "stocks_above_ema200",
    "stocks_at_52w_high",
    "stocks_at_52w_low",
    "breadth_thrust",
    "advance_decline_ratio",
    "new_high_new_low_ratio",
    "bank_nifty_vs_nifty",
    "it_sector_relative",
    "fmcg_sector_relative",
    "auto_sector_relative",
    "pharma_sector_relative",
    "infra_sector_relative",
    "metal_sector_relative",
    "realty_sector_relative",
    "fii_sector_rotation_signal",
    "sector_momentum_top",
    "sector_breadth_score",
    "results_beat_rate_season",
    "guidance_sentiment",
    "analyst_revision_direction",
    "index_above_ema50_proxy",
    "index_above_ema200_proxy",
    "index_52w_high_proxy",
    "index_52w_low_proxy",
    "breadth_data_quality_score",
    "sector_rotation_available",
    "market_breadth_proxy_score",
    "nifty_participation_warning",
    "breadth_confirmation_signal",
]

TYPE_OVERRIDES = {
    "sector_momentum_top": SignalType.CATEGORICAL,
    "guidance_sentiment": SignalType.DIRECTIONAL,
    "analyst_revision_direction": SignalType.DIRECTIONAL,
    "index_above_ema50_proxy": SignalType.BINARY,
    "index_above_ema200_proxy": SignalType.BINARY,
    "index_52w_high_proxy": SignalType.BINARY,
    "index_52w_low_proxy": SignalType.BINARY,
    "sector_rotation_available": SignalType.BINARY,
    "nifty_participation_warning": SignalType.BINARY,
    "breadth_confirmation_signal": SignalType.DIRECTIONAL,
}

SOURCE_OVERRIDES = {
    name: DataSource.FO_EOD
    for name in SIGNAL_NAMES
    if not name.endswith("_proxy")
    and name not in {
        "breadth_data_quality_score",
        "sector_rotation_available",
        "market_breadth_proxy_score",
        "nifty_participation_warning",
        "breadth_confirmation_signal",
    }
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OHLCV_DAILY,
    type_overrides=TYPE_OVERRIDES,
    source_overrides=SOURCE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values = {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx))


def _compute_values(ctx: SignalContext) -> dict[str, Any]:
    df = ctx.ohlcv_df
    values: dict[str, Any] = {
        "breadth_data_quality_score": 0.0,
        "sector_rotation_available": 0,
    }
    if df is None or getattr(df, "empty", True):
        return values
    close = numeric_series(df, "close")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    price = current_price(df, ctx.spot)
    ema50 = last(ema(close, 50))
    ema200 = last(ema(close, 200))
    high_52 = last(high.rolling(252, min_periods=20).max())
    low_52 = last(low.rolling(252, min_periods=20).min())
    above50 = int(price > ema50) if ema50 is not None else None
    above200 = int(price > ema200) if ema200 is not None else None
    values.update({
        "index_above_ema50_proxy": above50,
        "index_above_ema200_proxy": above200,
        "index_52w_high_proxy": int(price >= high_52) if high_52 is not None else None,
        "index_52w_low_proxy": int(price <= low_52) if low_52 is not None else None,
        "breadth_data_quality_score": 10.0,
        "market_breadth_proxy_score": _proxy_score(above50, above200, price, high_52, low_52),
        "nifty_participation_warning": 1,
        "breadth_confirmation_signal": 1 if above50 and above200 else (-1 if above50 == 0 and above200 == 0 else 0),
    })
    return values


def _proxy_score(above50: int | None, above200: int | None, price: float, high_52: float | None, low_52: float | None) -> float | None:
    pieces = []
    if above50 is not None:
        pieces.append(65.0 if above50 else 35.0)
    if above200 is not None:
        pieces.append(70.0 if above200 else 30.0)
    if high_52 is not None and low_52 is not None and high_52 > low_52:
        pieces.append((price - low_52) / (high_52 - low_52) * 100.0)
    return sum(pieces) / len(pieces) if pieces else None
