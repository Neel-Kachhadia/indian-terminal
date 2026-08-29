from __future__ import annotations

from typing import Any

from .common import option_chain_summary, safe_div
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "microstructure"

SIGNAL_NAMES = [
    "atm_call_bid_ask_spread",
    "atm_call_bid_ask_pct",
    "atm_put_bid_ask_spread",
    "atm_put_bid_ask_pct",
    "chain_avg_bid_ask_pct",
    "liquidity_score",
    "order_flow_imbalance",
    "bid_volume_ratio",
    "trade_intensity",
    "price_impact",
    "adverse_selection_cost",
    "nifty_futures_basis",
    "nifty_roll_cost",
    "cost_of_carry_annualised",
    "near_far_spread",
    "vwap_distance_pct",
    "open_range_breakout",
    "morning_star_open",
    "closing_auction_premium",
    "microstructure_quality_score",
    "atm_market_width_score",
    "option_depth_proxy",
    "liquidity_stress_flag",
    "quote_quality_warning",
    "tradability_score",
]

TYPE_OVERRIDES = {
    "open_range_breakout": SignalType.DIRECTIONAL,
    "morning_star_open": SignalType.DIRECTIONAL,
    "liquidity_stress_flag": SignalType.BINARY,
    "quote_quality_warning": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    "order_flow_imbalance": DataSource.TICK_DATA,
    "bid_volume_ratio": DataSource.TICK_DATA,
    "trade_intensity": DataSource.TICK_DATA,
    "price_impact": DataSource.TICK_DATA,
    "adverse_selection_cost": DataSource.TICK_DATA,
    "nifty_futures_basis": DataSource.FO_EOD,
    "nifty_roll_cost": DataSource.FO_EOD,
    "cost_of_carry_annualised": DataSource.FO_EOD,
    "near_far_spread": DataSource.FO_EOD,
    "vwap_distance_pct": DataSource.TICK_DATA,
    "open_range_breakout": DataSource.TICK_DATA,
    "morning_star_open": DataSource.TICK_DATA,
    "closing_auction_premium": DataSource.TICK_DATA,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OPTIONS_CHAIN,
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
    summary = option_chain_summary(ctx.chain_legs, ctx.spot)
    if not summary["legs"]:
        return {}
    call = summary["atm_call"] or {}
    put = summary["atm_put"] or {}
    call_spread, call_pct = _spread(call)
    put_spread, put_pct = _spread(put)
    spreads = [_spread(leg)[1] for leg in summary["legs"] if _spread(leg)[1] is not None]
    avg_spread = sum(spreads) / len(spreads) if spreads else None
    liquidity = max(0.0, min(100.0, 100.0 - avg_spread)) if avg_spread is not None else None
    atm_width_score = max(0.0, min(100.0, 100.0 - ((call_pct or 0.0) + (put_pct or 0.0)) / 2.0)) if call_pct is not None or put_pct is not None else None
    values = {
        "atm_call_bid_ask_spread": call_spread,
        "atm_call_bid_ask_pct": call_pct,
        "atm_put_bid_ask_spread": put_spread,
        "atm_put_bid_ask_pct": put_pct,
        "chain_avg_bid_ask_pct": avg_spread,
        "liquidity_score": liquidity,
        "microstructure_quality_score": liquidity,
        "atm_market_width_score": atm_width_score,
        "option_depth_proxy": summary["total_oi"] + summary["total_call_volume"] + summary["total_put_volume"],
        "liquidity_stress_flag": int((avg_spread or 0.0) > 10.0) if avg_spread is not None else None,
        "quote_quality_warning": int(not spreads),
        "tradability_score": _tradability(liquidity, summary["total_oi"]),
    }
    return values


def _spread(leg: dict[str, Any]) -> tuple[float | None, float | None]:
    bid = float(leg.get("bid") or 0.0)
    ask = float(leg.get("ask") or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None, None
    spread = ask - bid
    mid = (ask + bid) / 2.0
    return spread, safe_div(spread, mid) * 100.0 if mid else None


def _tradability(liquidity: float | None, total_oi: int) -> float | None:
    if liquidity is None:
        return None
    oi_score = min(100.0, total_oi / 100000.0)
    return (liquidity * 0.7) + (oi_score * 0.3)
