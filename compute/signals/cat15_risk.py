from __future__ import annotations

from typing import Any

from .common import atr, current_price, last, numeric_series, option_chain_summary, safe_div
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "risk"

SIGNAL_NAMES = [
    "portfolio_net_delta",
    "portfolio_net_gamma",
    "portfolio_net_theta",
    "portfolio_net_vega",
    "portfolio_var_95",
    "portfolio_var_99",
    "portfolio_cvar_95",
    "stress_nifty_down_5pct",
    "stress_nifty_down_10pct",
    "stress_iv_up_30pct",
    "stress_combined",
    "market_var_implied",
    "tail_risk_indicator",
    "systemic_risk_indicator",
    "margin_safety_atm",
    "liquidity_risk_atm",
    "concentration_risk_oi",
    "overnight_gap_risk",
    "market_risk_score_0_100",
    "gap_risk_score",
    "volatility_risk_score",
    "options_concentration_score",
    "hedging_pressure_risk",
    "event_risk_overlay",
    "risk_off_composite_signal",
]

TYPE_OVERRIDES = {
    "concentration_risk_oi": SignalType.BINARY,
    "risk_off_composite_signal": SignalType.DIRECTIONAL,
}

SOURCE_OVERRIDES = {
    name: DataSource.POSITION_DATA
    for name in SIGNAL_NAMES[:11]
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
    df = ctx.ohlcv_df
    close = numeric_series(df, "close") if df is not None else None
    spot = current_price(df, ctx.spot) if df is not None else ctx.spot
    atr_value = last(atr(df, 14)) if df is not None and not getattr(df, "empty", True) else None
    atm_call = summary.get("atm_call") or {}
    atm_put = summary.get("atm_put") or {}
    call_ltp = atm_call.get("ltp")
    put_ltp = atm_put.get("ltp")
    straddle = (call_ltp or 0.0) + (put_ltp or 0.0) if call_ltp is not None or put_ltp is not None else None
    atm_iv = atm_call.get("iv")
    expected_move = spot * atm_iv * (1.0 / 365.0) ** 0.5 if spot > 0 and atm_iv is not None else None
    top_oi = sorted(list(summary["call_oi"].values()) + list(summary["put_oi"].values()), reverse=True)
    concentration = safe_div(sum(top_oi[:3]), summary["total_oi"])
    call_spread = _spread_pct(atm_call)
    put_spread = _spread_pct(atm_put)
    liquidity_risk = _mean([call_spread, put_spread])
    tail_risk = _tail_risk(ctx, atm_iv)
    systemic = _systemic(ctx, summary, atm_iv)
    values = {
        "market_var_implied": safe_div(expected_move, 1.65),
        "tail_risk_indicator": tail_risk,
        "systemic_risk_indicator": systemic,
        "margin_safety_atm": safe_div(straddle, spot) * 100.0 if straddle is not None and spot > 0 else None,
        "liquidity_risk_atm": liquidity_risk,
        "concentration_risk_oi": int((concentration or 0.0) > 0.5) if concentration is not None else None,
        "overnight_gap_risk": safe_div(atr_value, spot) * 100.0 if atr_value is not None and spot > 0 else None,
        "gap_risk_score": _score(safe_div(atr_value, spot) * 100.0 if atr_value is not None and spot > 0 else None, 2.0),
        "volatility_risk_score": _score((ctx.india_vix or 0.0), 25.0) if ctx.india_vix is not None else _score((atm_iv or 0.0) * 100.0 if atm_iv else None, 25.0),
        "options_concentration_score": concentration * 100.0 if concentration is not None else None,
        "hedging_pressure_risk": _hedging_pressure(summary, spot),
        "event_risk_overlay": 0.0,
    }
    pieces = [
        values.get("tail_risk_indicator"),
        values.get("systemic_risk_indicator"),
        values.get("gap_risk_score"),
        values.get("volatility_risk_score"),
        values.get("options_concentration_score"),
    ]
    values["market_risk_score_0_100"] = _mean(pieces)
    market_risk_score = values["market_risk_score_0_100"]
    values["risk_off_composite_signal"] = (
        (-1 if market_risk_score > 65.0 else 0) if market_risk_score is not None else None
    )
    return values


def _spread_pct(leg: dict[str, Any]) -> float | None:
    bid = float(leg.get("bid") or 0.0)
    ask = float(leg.get("ask") or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (ask + bid) / 2.0
    return safe_div(ask - bid, mid) * 100.0 if mid else None


def _tail_risk(ctx: SignalContext, atm_iv: float | None) -> float | None:
    if atm_iv is None:
        return None
    pieces = [min(100.0, atm_iv * 300.0)]
    if ctx.india_vix is not None:
        pieces.append(min(100.0, max(0.0, (ctx.india_vix - 10.0) / 25.0 * 100.0)))
    return sum(pieces) / len(pieces)


def _systemic(ctx: SignalContext, summary: dict[str, Any], atm_iv: float | None) -> float | None:
    pieces = []
    if ctx.india_vix is not None:
        pieces.append(min(100.0, max(0.0, (ctx.india_vix - 10.0) / 25.0 * 100.0)))
    pcr = summary.get("pcr")
    if pcr is not None:
        pieces.append(min(100.0, abs(float(pcr) - 1.0) * 100.0))
    if atm_iv is not None:
        pieces.append(min(100.0, atm_iv * 300.0))
    return _mean(pieces)


def _hedging_pressure(summary: dict[str, Any], spot: float) -> float | None:
    if spot <= 0:
        return None
    total = 0.0
    for leg in summary["legs"]:
        delta = leg.get("delta")
        if delta is None:
            continue
        total += abs(delta * leg["oi"])
    return min(100.0, total / 100000.0)


def _score(value: float | None, high: float) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, value / high * 100.0))


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None
