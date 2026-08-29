from __future__ import annotations

from typing import Any

from .common import option_chain_summary, peak_key, safe_div
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "derivatives_flow"

SIGNAL_NAMES = [
    "fii_index_fut_long_contracts",
    "fii_index_fut_short_contracts",
    "fii_index_fut_net_contracts",
    "fii_index_fut_long_short_ratio",
    "fii_index_options_calls_long",
    "fii_index_options_puts_long",
    "fii_index_options_net_premium",
    "dii_index_fut_net",
    "client_index_fut_net",
    "prop_desk_index_fut_net",
    "options_volume_today",
    "options_volume_vs_20d_avg",
    "call_volume_today",
    "put_volume_today",
    "call_put_volume_ratio",
    "options_premium_today",
    "most_active_call_strike",
    "most_active_put_strike",
    "unusual_call_oi_strike",
    "unusual_put_oi_strike",
    "synthetic_long_detection",
    "large_block_options",
    "call_volume_share",
    "put_volume_share",
    "oi_change_premium_pressure",
    "derivatives_activity_score",
    "call_activity_intensity",
    "put_activity_intensity",
    "flow_skew_score",
    "options_turnover_proxy",
]

TYPE_OVERRIDES = {
    "synthetic_long_detection": SignalType.BINARY,
    "large_block_options": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    name: DataSource.FO_EOD
    for name in SIGNAL_NAMES[:10]
}
SOURCE_OVERRIDES.update({
    "synthetic_long_detection": DataSource.TICK_DATA,
    "large_block_options": DataSource.TICK_DATA,
})

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
    call_volume = summary["total_call_volume"]
    put_volume = summary["total_put_volume"]
    total_volume = call_volume + put_volume
    premium = sum(leg["ltp"] * leg["oi"] for leg in summary["legs"])
    oi_change_pressure = sum(leg["ltp"] * leg["oi_change"] for leg in summary["legs"])
    values = {
        "options_volume_today": total_volume,
        "call_volume_today": call_volume,
        "put_volume_today": put_volume,
        "call_put_volume_ratio": safe_div(call_volume, put_volume),
        "options_premium_today": premium,
        "most_active_call_strike": peak_key(summary["call_volume"]),
        "most_active_put_strike": peak_key(summary["put_volume"]),
        "unusual_call_oi_strike": peak_key(summary["call_oi_change"]),
        "unusual_put_oi_strike": peak_key(summary["put_oi_change"]),
        "call_volume_share": safe_div(call_volume, total_volume),
        "put_volume_share": safe_div(put_volume, total_volume),
        "oi_change_premium_pressure": oi_change_pressure,
        "derivatives_activity_score": min(100.0, total_volume / 100000.0),
        "call_activity_intensity": min(100.0, call_volume / 100000.0),
        "put_activity_intensity": min(100.0, put_volume / 100000.0),
        "flow_skew_score": _flow_skew(call_volume, put_volume),
        "options_turnover_proxy": premium * 75,
    }
    values["options_volume_vs_20d_avg"] = None
    return values


def _flow_skew(call_volume: int, put_volume: int) -> float | None:
    total = call_volume + put_volume
    if total <= 0:
        return None
    return (put_volume - call_volume) / total * 100.0
