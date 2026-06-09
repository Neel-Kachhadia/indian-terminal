from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
from py_vollib.black_scholes.implied_volatility import implied_volatility


RISK_FREE_RATE = 0.065
FALLBACK_IV = 0.15
POINT_COUNT = 200


def compute_payoff(
    legs: list[dict[str, Any]],
    spot: float,
    spot_range_pct: float = 0.20,
) -> dict[str, Any]:
    """
    Compute multi-leg options/futures payoff.

    Reference sanity check:
    long 23400 CE at 221 premium and lot size 75 has breakeven 23621
    and max loss -16575, matching textbook expiry payoff.
    """
    spot = _positive_float(spot, "spot")
    spot_range_pct = max(0.01, min(float(spot_range_pct), 1.0))
    normalised_legs = [_normalise_leg(leg) for leg in legs]

    low = max(0.01, spot * (1.0 - spot_range_pct))
    high = max(low + 1.0, spot * (1.0 + spot_range_pct))
    spot_range = np.linspace(low, high, POINT_COUNT, dtype=np.float64)

    pnl_at_expiry = np.zeros_like(spot_range)
    pnl_today = np.zeros_like(spot_range)
    net_greeks = {
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
    }

    for leg in normalised_legs:
        pnl_at_expiry += _expiry_leg_pnl(leg, spot_range)
        pnl_today += _today_leg_pnl(leg, spot_range, spot)
        _add_leg_greeks(net_greeks, leg, spot)

    breakevens = _breakevens(spot_range, pnl_at_expiry)
    max_profit, max_loss = _max_profit_loss(normalised_legs, pnl_at_expiry)

    return {
        "spot_range": _round_list(spot_range, 2),
        "pnl_at_expiry": _round_list(pnl_at_expiry, 2),
        "pnl_today": _round_list(pnl_today, 2),
        "breakevens": breakevens,
        "max_profit": None if max_profit is None else round(max_profit, 2),
        "max_loss": None if max_loss is None else round(max_loss, 2),
        "net_greeks": {
            key: round(value, 6)
            for key, value in net_greeks.items()
        },
    }


def _normalise_leg(raw_leg: dict[str, Any]) -> dict[str, Any]:
    option_type = str(raw_leg.get("option_type", "")).strip().upper()
    if option_type not in {"CE", "PE", "FUT"}:
        raise ValueError(f"unsupported option_type: {option_type}")

    action = str(raw_leg.get("action", "")).strip().upper()
    if action not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported action: {action}")

    quantity = int(raw_leg.get("quantity", 1))
    lot_size = int(raw_leg.get("lot_size", 75))
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")

    strike = 0.0 if option_type == "FUT" else _positive_float(raw_leg.get("strike"), "strike")
    entry_price = _non_negative_float(raw_leg.get("entry_price"), "entry_price")

    return {
        "symbol": str(raw_leg.get("symbol") or "NIFTY").strip().upper(),
        "strike": strike,
        "expiry": str(raw_leg.get("expiry") or "").strip(),
        "option_type": option_type,
        "action": action,
        "quantity": quantity,
        "entry_price": entry_price,
        "lot_size": lot_size,
    }


def _expiry_leg_pnl(leg: dict[str, Any], spot_range: np.ndarray) -> np.ndarray:
    multiplier = _signed_multiplier(leg)
    option_type = leg["option_type"]
    entry_price = leg["entry_price"]

    if option_type == "CE":
        payoff = np.maximum(spot_range - leg["strike"], 0.0)
        return (payoff - entry_price) * multiplier
    if option_type == "PE":
        payoff = np.maximum(leg["strike"] - spot_range, 0.0)
        return (payoff - entry_price) * multiplier

    return (spot_range - entry_price) * multiplier


def _today_leg_pnl(
    leg: dict[str, Any],
    spot_range: np.ndarray,
    current_spot: float,
) -> np.ndarray:
    if leg["option_type"] == "FUT":
        return _expiry_leg_pnl(leg, spot_range)

    multiplier = _signed_multiplier(leg)
    sigma = _leg_iv(leg, current_spot)
    t = _time_to_expiry(leg["expiry"])
    flag = _flag(leg["option_type"])

    values = np.array(
        [
            _bs_price(flag, float(next_spot), leg["strike"], t, sigma)
            for next_spot in spot_range
        ],
        dtype=np.float64,
    )
    return (values - leg["entry_price"]) * multiplier


def _add_leg_greeks(
    net_greeks: dict[str, float],
    leg: dict[str, Any],
    spot: float,
) -> None:
    multiplier = _signed_multiplier(leg)
    option_type = leg["option_type"]

    if option_type == "FUT":
        net_greeks["delta"] += multiplier
        return

    sigma = _leg_iv(leg, spot)
    t = _time_to_expiry(leg["expiry"])
    flag = _flag(option_type)
    strike = leg["strike"]

    try:
        net_greeks["delta"] += delta(flag, spot, strike, t, RISK_FREE_RATE, sigma) * multiplier
        net_greeks["gamma"] += gamma(flag, spot, strike, t, RISK_FREE_RATE, sigma) * multiplier
        net_greeks["theta"] += theta(flag, spot, strike, t, RISK_FREE_RATE, sigma) * multiplier
        net_greeks["vega"] += vega(flag, spot, strike, t, RISK_FREE_RATE, sigma) * multiplier
    except Exception:
        return


def _leg_iv(leg: dict[str, Any], spot: float) -> float:
    if leg["option_type"] == "FUT":
        return 0.0

    flag = _flag(leg["option_type"])
    t = _time_to_expiry(leg["expiry"])
    try:
        iv = implied_volatility(
            leg["entry_price"],
            spot,
            leg["strike"],
            t,
            RISK_FREE_RATE,
            flag,
        )
        if math.isfinite(iv) and 0.01 <= iv <= 5.0:
            return float(iv)
    except Exception:
        pass

    return FALLBACK_IV


def _bs_price(flag: str, spot: float, strike: float, t: float, sigma: float) -> float:
    try:
        value = black_scholes(flag, spot, strike, t, RISK_FREE_RATE, sigma)
    except Exception:
        return _intrinsic(flag, spot, strike)

    if not math.isfinite(value):
        return _intrinsic(flag, spot, strike)
    return float(value)


def _intrinsic(flag: str, spot: float, strike: float) -> float:
    if flag == "c":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _breakevens(spot_range: np.ndarray, pnl: np.ndarray) -> list[float]:
    points: list[float] = []
    for index in range(1, len(spot_range)):
        previous_pnl = float(pnl[index - 1])
        current_pnl = float(pnl[index])
        previous_spot = float(spot_range[index - 1])
        current_spot = float(spot_range[index])

        if abs(previous_pnl) < 1e-9:
            _append_unique(points, previous_spot)
            continue
        if previous_pnl * current_pnl > 0:
            continue

        denominator = current_pnl - previous_pnl
        if abs(denominator) < 1e-12:
            continue

        ratio = -previous_pnl / denominator
        crossing = previous_spot + ratio * (current_spot - previous_spot)
        _append_unique(points, crossing)

    if len(pnl) > 0 and abs(float(pnl[-1])) < 1e-9:
        _append_unique(points, float(spot_range[-1]))

    return [round(point, 2) for point in points]


def _append_unique(points: list[float], point: float) -> None:
    if not points or abs(points[-1] - point) > 0.01:
        points.append(point)


def _max_profit_loss(
    legs: list[dict[str, Any]],
    pnl_at_expiry: np.ndarray,
) -> tuple[float | None, float | None]:
    right_slope = 0.0
    for leg in legs:
        multiplier = _signed_multiplier(leg)
        if leg["option_type"] in {"CE", "FUT"}:
            right_slope += multiplier

    max_profit = None if right_slope > 0.0 else float(np.max(pnl_at_expiry))
    max_loss = None if right_slope < 0.0 else float(np.min(pnl_at_expiry))
    return max_profit, max_loss


def _signed_multiplier(leg: dict[str, Any]) -> float:
    sign = 1.0 if leg["action"] == "BUY" else -1.0
    return sign * float(leg["quantity"]) * float(leg["lot_size"])


def _time_to_expiry(expiry: str) -> float:
    parsed = _parse_expiry(expiry)
    if parsed is None:
        return 1.0 / 365.0
    dte = max((parsed - _dt.date.today()).days, 1)
    return dte / 365.0


def _parse_expiry(expiry: str) -> _dt.date | None:
    value = expiry.strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def _flag(option_type: str) -> str:
    return "c" if option_type == "CE" else "p"


def _positive_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _round_list(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]
