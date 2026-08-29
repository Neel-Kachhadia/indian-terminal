from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from .common import (
    dte,
    gex,
    max_pain,
    nearest_delta_leg,
    option_chain_summary,
    peak_key,
    safe_div,
)
from .framework import (
    DataSource,
    SignalRegime,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "options"
_PCR_HISTORY: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
_ATM_IV_HISTORY: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=30))
_GEX_HISTORY: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=30))

SIGNAL_NAMES = [
    "pcr_total",
    "pcr_atm_5strikes",
    "pcr_5d_ma",
    "pcr_20d_ma",
    "pcr_vs_20d_avg",
    "pcr_trend_direction",
    "pcr_extreme_bullish",
    "pcr_extreme_bearish",
    "pcr_divergence_price",
    "pcr_weekly_vs_monthly",
    "total_call_oi",
    "total_put_oi",
    "total_oi",
    "call_oi_peak_strike",
    "put_oi_peak_strike",
    "oi_concentration_ratio",
    "oi_range",
    "oi_range_pct",
    "spot_in_oi_range",
    "oi_buildup_net",
    "call_oi_change_pct",
    "put_oi_change_pct",
    "oi_unwinding",
    "oi_buildup",
    "call_oi_resistance_buildup",
    "put_oi_support_buildup",
    "oi_shift_to_puts",
    "oi_pain_distance_pct",
    "max_pain",
    "delta_adjusted_oi",
    "atm_iv",
    "atm_put_iv",
    "iv_rank",
    "iv_percentile_52w",
    "iv_mean_reversion_z",
    "vol_risk_premium",
    "iv_skew_25delta",
    "iv_skew_direction",
    "iv_skew_z_score",
    "term_structure_slope",
    "term_structure_inverted",
    "iv_surface_convexity",
    "atm_iv_trend_5poll",
    "expected_move_pts",
    "expected_move_pct",
    "realized_vs_implied_ratio",
    "iv_crush_proximity",
    "iv_nifty_correlation_30d",
    "put_call_parity_atm",
    "iv_surface_quality",
    "gex",
    "gex_flip",
    "net_chain_delta",
    "net_chain_gamma",
    "gamma_concentration_atm",
    "charm_daily",
    "dealer_delta_hedge_pressure",
    "vanna_exposure",
    "volga_exposure",
    "net_chain_vega",
    "net_chain_theta",
    "pin_risk_3dte",
    "gamma_squeeze_potential",
    "vanna_squeeze_potential",
    "shadow_delta",
    "straddle_breakeven_up",
    "straddle_breakeven_down",
    "straddle_premium_pct",
    "iron_condor_range",
    "call_spread_cost_atm_100",
    "put_spread_cost_atm_100",
    "calendar_spread_iv_edge",
    "synthetic_futures_basis",
    "risk_reversal_25d",
    "butterfly_cost_atm",
    "atm_strike",
    "atm_call_ltp",
    "atm_put_ltp",
    "atm_call_delta",
    "atm_put_delta",
    "call_volume_total",
    "put_volume_total",
    "volume_pcr",
    "top_5_oi_share",
    "distance_to_call_wall_pct",
    "distance_to_put_wall_pct",
    "expected_move_upper",
    "expected_move_lower",
    "gamma_wall_strike",
    "vega_weighted_iv",
]

TYPE_OVERRIDES = {
    "pcr_trend_direction": SignalType.DIRECTIONAL,
    "pcr_extreme_bullish": SignalType.BINARY,
    "pcr_extreme_bearish": SignalType.BINARY,
    "pcr_divergence_price": SignalType.DIRECTIONAL,
    "spot_in_oi_range": SignalType.BINARY,
    "oi_unwinding": SignalType.BINARY,
    "oi_buildup": SignalType.BINARY,
    "call_oi_resistance_buildup": SignalType.BINARY,
    "put_oi_support_buildup": SignalType.BINARY,
    "oi_shift_to_puts": SignalType.BINARY,
    "iv_skew_direction": SignalType.DIRECTIONAL,
    "term_structure_inverted": SignalType.BINARY,
    "pin_risk_3dte": SignalType.BINARY,
    "gamma_squeeze_potential": SignalType.BINARY,
    "vanna_squeeze_potential": SignalType.BINARY,
}

REGIME_OVERRIDES = {
    "pcr_extreme_bullish": SignalRegime.RANGING,
    "pcr_extreme_bearish": SignalRegime.RANGING,
    "iv_mean_reversion_z": SignalRegime.HIGH_VOL,
    "gamma_squeeze_potential": SignalRegime.HIGH_VOL,
    "vanna_squeeze_potential": SignalRegime.HIGH_VOL,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OPTIONS_CHAIN,
    type_overrides=TYPE_OVERRIDES,
    regime_overrides=REGIME_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values, histories = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values, histories = {}, {}
    return build_signals(
        CATEGORY,
        SPECS,
        values,
        source_availability(ctx),
        histories=histories,
    )


def _compute_values(ctx: SignalContext) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = option_chain_summary(ctx.chain_legs, ctx.spot)
    if not summary["legs"]:
        return {}, {}

    spot = ctx.spot
    expiry_dte = dte(ctx.expiry_str) or 1
    atm_strike = summary["atm_strike"]
    atm_call = summary["atm_call"] or {}
    atm_put = summary["atm_put"] or {}
    pcr = summary["pcr"]
    history_key = f"{ctx.symbol}:{ctx.expiry_str}"
    if pcr is not None:
        _PCR_HISTORY[history_key].append(float(pcr))
    pcr_hist = list(_PCR_HISTORY[history_key])

    call_peak = peak_key(summary["call_oi"])
    put_peak = peak_key(summary["put_oi"])
    total_call_oi = summary["total_call_oi"]
    total_put_oi = summary["total_put_oi"]
    total_oi = summary["total_oi"]
    max_pain_value = max_pain(summary, spot)
    gex_value, gex_flip = gex(summary, spot, ctx.lot_size)
    if gex_value is not None:
        _GEX_HISTORY[history_key].append(float(gex_value))

    atm_iv = atm_call.get("iv")
    atm_put_iv = atm_put.get("iv")
    if atm_iv is not None:
        _ATM_IV_HISTORY[history_key].append(float(atm_iv))
    atm_iv_hist = list(_ATM_IV_HISTORY[history_key])

    put_25 = nearest_delta_leg(summary["puts"], "PE", -0.25)
    call_25 = nearest_delta_leg(summary["calls"], "CE", 0.25)
    skew_25 = _diff(_leg_iv(put_25), _leg_iv(call_25))
    call_oi_change = sum(summary["call_oi_change"].values())
    put_oi_change = sum(summary["put_oi_change"].values())
    total_oi_change = call_oi_change + put_oi_change
    atm_gamma = _leg_value(atm_call, "gamma", 0.0) + _leg_value(atm_put, "gamma", 0.0)
    net_gamma = _sum_leg_product(summary["legs"], "gamma", ctx.lot_size, sign_put=1.0)
    net_delta = _sum_delta(summary["legs"], ctx.lot_size)
    net_vega = _sum_leg_product(summary["legs"], "vega", ctx.lot_size, sign_put=1.0)
    net_theta = _sum_leg_product(summary["legs"], "theta", ctx.lot_size, sign_put=1.0)
    call_ltp = _leg_value(atm_call, "ltp")
    put_ltp = _leg_value(atm_put, "ltp")
    straddle = _add(call_ltp, put_ltp)
    expected_move = spot * float(atm_iv) * math.sqrt(expiry_dte / 365.0) if spot > 0 and atm_iv else None
    top_oi = sorted(
        list(summary["call_oi"].values()) + list(summary["put_oi"].values()),
        reverse=True,
    )
    surface_quality = safe_div(
        sum(1 for leg in summary["legs"] if leg.get("iv") is not None),
        len(summary["legs"]),
    )
    iv_values = [leg["iv"] for leg in summary["legs"] if leg.get("iv") is not None]
    iv_convexity = _iv_convexity(summary["legs"], atm_strike)
    gamma_by_strike = defaultdict(float)
    for leg in summary["legs"]:
        gamma = _leg_value(leg, "gamma", 0.0) or 0.0
        gamma_by_strike[leg["strike"]] += abs(gamma) * leg["oi"]

    values = {
        "pcr_total": pcr,
        "pcr_atm_5strikes": _atm_pcr(summary, atm_strike, 5),
        "pcr_5d_ma": _mean(pcr_hist[-5:]),
        "pcr_20d_ma": _mean(pcr_hist[-20:]),
        "pcr_vs_20d_avg": _pct(pcr, _mean(pcr_hist[-20:])),
        "pcr_trend_direction": _trend_direction(pcr_hist),
        "pcr_extreme_bullish": int(pcr > 1.5) if pcr is not None else None,
        "pcr_extreme_bearish": int((pcr or 0.0) < 0.5) if pcr is not None else None,
        "pcr_divergence_price": 0,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_oi": total_oi,
        "call_oi_peak_strike": call_peak,
        "put_oi_peak_strike": put_peak,
        "oi_concentration_ratio": safe_div(sum(top_oi[:3]), total_oi),
        "oi_range": _diff(call_peak, put_peak),
        "oi_range_pct": _pct(_diff(call_peak, put_peak), spot),
        "spot_in_oi_range": _in_range(spot, put_peak, call_peak),
        "oi_buildup_net": total_oi_change,
        "call_oi_change_pct": safe_div(call_oi_change, max(total_call_oi - call_oi_change, 1)) * 100.0 if total_call_oi else None,
        "put_oi_change_pct": safe_div(put_oi_change, max(total_put_oi - put_oi_change, 1)) * 100.0 if total_put_oi else None,
        "oi_unwinding": int(total_oi_change < -0.05 * max(total_oi - total_oi_change, 1)),
        "oi_buildup": int(total_oi_change > 0.05 * max(total_oi - total_oi_change, 1)),
        "call_oi_resistance_buildup": _strike_change_flag(summary["call_oi_change"], call_peak, summary["call_oi"]),
        "put_oi_support_buildup": _strike_change_flag(summary["put_oi_change"], put_peak, summary["put_oi"]),
        "oi_shift_to_puts": int(put_oi_change > call_oi_change) if total_oi_change != 0 else 0,
        "oi_pain_distance_pct": _pct(_diff(max_pain_value, spot), spot),
        "max_pain": max_pain_value,
        "delta_adjusted_oi": _delta_adjusted_oi(summary["legs"]),
        "atm_iv": atm_iv,
        "atm_put_iv": atm_put_iv,
        "iv_rank": _iv_rank(atm_iv, atm_iv_hist),
        "iv_percentile_52w": _iv_percentile(atm_iv, atm_iv_hist),
        "iv_mean_reversion_z": _z(atm_iv, atm_iv_hist),
        "iv_skew_25delta": skew_25,
        "iv_skew_direction": 1 if (skew_25 or 0.0) > 0 else (-1 if skew_25 is not None else None),
        "iv_skew_z_score": None,
        "iv_surface_convexity": iv_convexity,
        "atm_iv_trend_5poll": _pct(atm_iv, atm_iv_hist[-5] if len(atm_iv_hist) >= 5 else None),
        "expected_move_pts": expected_move,
        "expected_move_pct": _pct(expected_move, spot),
        "realized_vs_implied_ratio": None,
        "iv_crush_proximity": _iv_crush_proximity(atm_iv_hist),
        "put_call_parity_atm": _put_call_parity(call_ltp, put_ltp, atm_strike, spot),
        "iv_surface_quality": surface_quality,
        "gex": gex_value,
        "gex_flip": gex_flip,
        "net_chain_delta": net_delta,
        "net_chain_gamma": net_gamma,
        "gamma_concentration_atm": safe_div(atm_gamma, net_gamma),
        "charm_daily": safe_div(-net_delta, max(expiry_dte, 1)),
        "dealer_delta_hedge_pressure": -net_delta if net_delta is not None else None,
        "net_chain_vega": net_vega,
        "net_chain_theta": net_theta,
        "pin_risk_3dte": int(expiry_dte < 3 and spot > 0 and _near_oi(summary, spot, 50.0) > 0.2 * total_oi),
        "gamma_squeeze_potential": int((gex_value or 0.0) > 2.0 * (_mean(list(_GEX_HISTORY[history_key])) or float("inf"))),
        "shadow_delta": _add(net_delta, net_gamma * spot * 0.01 if net_gamma is not None else None),
        "straddle_breakeven_up": spot + straddle if spot > 0 and straddle is not None else None,
        "straddle_breakeven_down": spot - straddle if spot > 0 and straddle is not None else None,
        "straddle_premium_pct": _pct(straddle, spot),
        "iron_condor_range": _diff(call_peak, put_peak),
        "call_spread_cost_atm_100": _spread_cost(summary, atm_strike, "CE", 100.0),
        "put_spread_cost_atm_100": _spread_cost(summary, atm_strike, "PE", -100.0),
        "synthetic_futures_basis": _put_call_parity(call_ltp, put_ltp, atm_strike, spot),
        "risk_reversal_25d": _diff(_leg_iv(call_25), _leg_iv(put_25)),
        "butterfly_cost_atm": _butterfly_cost(summary, atm_strike),
        "atm_strike": atm_strike,
        "atm_call_ltp": call_ltp,
        "atm_put_ltp": put_ltp,
        "atm_call_delta": _leg_value(atm_call, "delta"),
        "atm_put_delta": _leg_value(atm_put, "delta"),
        "call_volume_total": summary["total_call_volume"],
        "put_volume_total": summary["total_put_volume"],
        "volume_pcr": safe_div(summary["total_put_volume"], summary["total_call_volume"]),
        "top_5_oi_share": safe_div(sum(top_oi[:5]), total_oi),
        "distance_to_call_wall_pct": _pct(_diff(call_peak, spot), spot),
        "distance_to_put_wall_pct": _pct(_diff(spot, put_peak), spot),
        "expected_move_upper": spot + expected_move if expected_move is not None else None,
        "expected_move_lower": spot - expected_move if expected_move is not None else None,
        "gamma_wall_strike": peak_key(gamma_by_strike),
        "vega_weighted_iv": safe_div(
            sum(leg["iv"] * abs(leg["vega"]) * leg["oi"] for leg in summary["legs"] if leg.get("iv") is not None and leg.get("vega") is not None),
            sum(abs(leg["vega"]) * leg["oi"] for leg in summary["legs"] if leg.get("iv") is not None and leg.get("vega") is not None),
        ),
    }
    values["pcr_weekly_vs_monthly"] = None
    values["term_structure_slope"] = None
    values["term_structure_inverted"] = None
    values["vol_risk_premium"] = None
    values["iv_nifty_correlation_30d"] = None
    values["vanna_exposure"] = None
    values["volga_exposure"] = None
    values["vanna_squeeze_potential"] = None
    values["calendar_spread_iv_edge"] = None

    histories = {
        "pcr_total": pcr_hist,
        "atm_iv": atm_iv_hist,
        "gex": list(_GEX_HISTORY[history_key]),
        "iv_surface_quality": [surface_quality] if surface_quality is not None else [],
        "iv_mean_reversion_z": atm_iv_hist,
    }
    return values, histories


def _leg_value(leg: dict[str, Any] | None, key: str, default: float | None = None) -> float | None:
    if not leg:
        return default
    value = leg.get(key)
    if value is None:
        return default
    return float(value)


def _leg_iv(leg: dict[str, Any] | None) -> float | None:
    return _leg_value(leg, "iv")


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _add(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left + right


def _pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or abs(base) < 1e-12:
        return None
    return value / base * 100.0


def _z(value: float | None, history: list[float]) -> float | None:
    if value is None or len(history) < 5:
        return None
    mean = _mean(history)
    variance = _mean([(item - mean) ** 2 for item in history]) if mean is not None else None
    if variance is None or variance <= 1e-12:
        return None
    return (value - mean) / math.sqrt(variance)


def _trend_direction(history: list[float]) -> int:
    if len(history) < 2:
        return 0
    if history[-1] > history[-2]:
        return 1
    if history[-1] < history[-2]:
        return -1
    return 0


def _atm_pcr(summary: dict[str, Any], atm_strike: float | None, width: int) -> float | None:
    if atm_strike is None or not summary["strikes"]:
        return None
    strikes = sorted(summary["strikes"], key=lambda strike: abs(strike - atm_strike))[:width]
    call_oi = sum(summary["call_oi"].get(strike, 0) for strike in strikes)
    put_oi = sum(summary["put_oi"].get(strike, 0) for strike in strikes)
    return safe_div(put_oi, call_oi)


def _in_range(spot: float, lower: float | None, upper: float | None) -> int | None:
    if lower is None or upper is None or spot <= 0:
        return None
    low, high = sorted([lower, upper])
    return int(low <= spot <= high)


def _strike_change_flag(changes: dict[float, int], strike: float | None, oi: dict[float, int]) -> int | None:
    if strike is None:
        return None
    change = changes.get(strike, 0)
    base = max(oi.get(strike, 0) - change, 1)
    return int(change / base > 0.10)


def _delta_adjusted_oi(legs: list[dict[str, Any]]) -> float:
    return sum((leg.get("delta") or 0.0) * leg["oi"] for leg in legs)


def _sum_delta(legs: list[dict[str, Any]], lot_size: int | None) -> float | None:
    if lot_size is None or lot_size <= 0:
        return None
    total = 0.0
    for leg in legs:
        delta = leg.get("delta")
        if delta is None:
            continue
        sign = 1.0 if leg["option_type"] == "CE" else -1.0
        total += sign * delta * leg["oi"] * lot_size
    return total


def _sum_leg_product(
    legs: list[dict[str, Any]],
    field: str,
    lot_size: int | None,
    *,
    sign_put: float,
) -> float | None:
    if lot_size is None or lot_size <= 0:
        return None
    total = 0.0
    for leg in legs:
        value = leg.get(field)
        if value is None:
            continue
        sign = 1.0 if leg["option_type"] == "CE" else sign_put
        total += sign * value * leg["oi"] * lot_size
    return total


def _iv_rank(value: float | None, history: list[float]) -> float | None:
    """None (not 50.0) when unavailable. iv_rank/iv_percentile_52w flow
    unguarded into backtest strategy threshold rules (options.iv_rank > 35,
    < 30, etc), three ML models (m03 strategy selector, m04 IV reversion,
    m18 bandit), BIF momentum, and narrative generation -- a fabricated
    midpoint here would silently masquerade as a real "IV sitting at the
    middle of its range" reading and could trigger or suppress real signal
    thresholds purely because data was missing. Matches this file's own
    `_z()` convention and signals/options_signals.py's already-correct
    sibling implementation (see test_options_signal_lot_size.py)."""
    if value is None or len(history) < 5:
        return None
    low = min(history)
    high = max(history)
    if high - low <= 1e-12:
        return None
    return max(0.0, min(100.0, (value - low) / (high - low) * 100.0))


def _iv_percentile(value: float | None, history: list[float]) -> float | None:
    if value is None or len(history) < 5:
        return None
    return sum(1 for item in history if item <= value) / len(history) * 100.0


def _iv_crush_proximity(history: list[float]) -> int | None:
    if len(history) < 5:
        return None
    threshold = (_mean(history) or 0.0) + 2.0 * math.sqrt(_mean([(item - (_mean(history) or 0.0)) ** 2 for item in history]) or 0.0)
    for offset, value in enumerate(reversed(history), start=0):
        if value >= threshold:
            return offset
    return None


def _put_call_parity(call_ltp: float | None, put_ltp: float | None, strike: float | None, spot: float) -> float | None:
    if call_ltp is None or put_ltp is None or strike is None or spot <= 0:
        return None
    return (call_ltp - put_ltp) + strike - spot


def _near_oi(summary: dict[str, Any], spot: float, distance: float) -> int:
    total = 0
    for strike, oi in summary["call_oi"].items():
        if abs(strike - spot) <= distance:
            total += oi
    for strike, oi in summary["put_oi"].items():
        if abs(strike - spot) <= distance:
            total += oi
    return total


def _spread_cost(summary: dict[str, Any], atm_strike: float | None, option_type: str, offset: float) -> float | None:
    if atm_strike is None:
        return None
    near = _find_leg(summary["legs"], atm_strike, option_type)
    far = _find_leg(summary["legs"], atm_strike + offset, option_type)
    if not near or not far:
        return None
    return near["ltp"] - far["ltp"]


def _find_leg(legs: list[dict[str, Any]], strike: float, option_type: str) -> dict[str, Any] | None:
    candidates = [leg for leg in legs if leg["option_type"] == option_type]
    if not candidates:
        return None
    return min(candidates, key=lambda leg: abs(leg["strike"] - strike))


def _butterfly_cost(summary: dict[str, Any], atm_strike: float | None) -> float | None:
    if atm_strike is None:
        return None
    lower = _find_leg(summary["legs"], atm_strike - 100.0, "CE")
    middle = _find_leg(summary["legs"], atm_strike, "CE")
    upper = _find_leg(summary["legs"], atm_strike + 100.0, "CE")
    if not lower or not middle or not upper:
        return None
    return lower["ltp"] - 2.0 * middle["ltp"] + upper["ltp"]


def _iv_convexity(legs: list[dict[str, Any]], atm_strike: float | None) -> float | None:
    if atm_strike is None:
        return None
    points = sorted((leg["strike"], leg["iv"]) for leg in legs if leg.get("iv") is not None)
    if len(points) < 5:
        return None
    closest = sorted(points, key=lambda item: abs(item[0] - atm_strike))[:5]
    closest = sorted(closest)
    curvatures = []
    for index in range(1, len(closest) - 1):
        left = closest[index - 1][1]
        mid = closest[index][1]
        right = closest[index + 1][1]
        curvatures.append(abs(left - 2.0 * mid + right))
    return _mean(curvatures)
