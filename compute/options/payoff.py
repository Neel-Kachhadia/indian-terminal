from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks.analytical import delta, gamma, theta, vega

from data.risk_free_rate import latest_available_rate
from options.iv_pricing import PricingModel, SolveStatus, solve_iv_with_provenance


POINT_COUNT = 200
CONFIGURED_ASSUMPTION = "CONFIGURED_ASSUMPTION"
OBSERVED_MARKET_RATE = "OBSERVED_MARKET_RATE"
ZERO_DIVIDEND_ASSUMPTION = "ZERO_DIVIDEND_ASSUMPTION"


def compute_payoff(
    legs: list[dict[str, Any]],
    spot: float,
    spot_range_pct: float = 0.20,
    *,
    risk_free_rate: float | None = None,
    risk_free_rate_source: str | None = None,
    dividend_yield: float | None = None,
    dividend_yield_source: str | None = None,
) -> dict[str, Any]:
    """
    Compute multi-leg options/futures payoff.

    Reference sanity check:
    long 23400 CE at 221 premium and lot size 75 has breakeven 23621
    and max loss -16575, matching textbook expiry payoff.

    Today's P&L / Greeks require, per leg, both a rate and a dividend yield.
    Rate: a genuinely observed current FBIL curve point for that leg's own
    time-to-expiry is tried first (`OBSERVED_MARKET_RATE`); an explicit
    caller-supplied `risk_free_rate` is used only as a disclosed fallback,
    and only when the caller also passes `risk_free_rate_source ==
    CONFIGURED_ASSUMPTION`. Dividend: no observed dividend-yield source
    exists anywhere in this system, so a nonzero value is accepted ONLY as
    an explicit, disclosed `dividend_yield_source == CONFIGURED_ASSUMPTION`
    value -- never silently defaulted. A value of exactly 0.0 may instead be
    disclosed as `dividend_yield_source == ZERO_DIVIDEND_ASSUMPTION`: an
    honest label for "nobody configured a dividend, so the model assumes
    zero", distinct from `CONFIGURED_ASSUMPTION`, which claims somebody
    actually supplied a value. Either family missing/mislabelled degrades
    pnl_today/net_greeks to None with an explicit analytical_status, never a
    fabricated substitute.
    """
    spot = _positive_float(spot, "spot")
    spot_range_pct = max(0.01, min(float(spot_range_pct), 1.0))
    normalised_legs = [_normalise_leg(leg) for leg in legs]

    low = max(0.01, spot * (1.0 - spot_range_pct))
    high = max(low + 1.0, spot * (1.0 + spot_range_pct))
    spot_range = np.linspace(low, high, POINT_COUNT, dtype=np.float64)

    pnl_at_expiry = np.zeros_like(spot_range)
    pnl_today: np.ndarray | None = np.zeros_like(spot_range)
    net_greeks: dict[str, float] | None = {
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
    }
    analytics_status = "VALID"
    rate_sources_used: set[str] = set()
    dividend_sources_used: set[str] = set()

    for leg in normalised_legs:
        pnl_at_expiry += _expiry_leg_pnl(leg, spot_range)

        if leg["option_type"] == "FUT":
            if pnl_today is not None:
                pnl_today += _expiry_leg_pnl(leg, spot_range)
                if net_greeks is not None:
                    net_greeks["delta"] += _signed_multiplier(leg)
            continue

        if pnl_today is None:
            # Already degraded by an earlier leg; pnl_at_expiry above still
            # needs every leg summed, but analytics stay None.
            continue

        t = _time_to_expiry(leg["expiry"])
        rate, rate_source = _resolve_rate(t, risk_free_rate, risk_free_rate_source)
        dividend, dividend_source = _resolve_dividend(dividend_yield, dividend_yield_source)
        if t is None or rate is None or dividend is None:
            pnl_today = None
            net_greeks = None
            analytics_status = (
                "EXPIRED_OR_INVALID_EXPIRY" if t is None
                else "MISSING_RATE" if rate is None
                else "MISSING_DIVIDEND_ASSUMPTION"
            )
            continue
        rate_sources_used.add(rate_source)
        dividend_sources_used.add(dividend_source)

        leg_pnl_today = _today_leg_pnl(leg, spot_range, spot, rate, dividend)
        if leg_pnl_today is None:
            pnl_today = None
            net_greeks = None
            analytics_status = "NO_CONVERGENCE"
            continue
        pnl_today += leg_pnl_today
        if net_greeks is not None and not _add_leg_greeks(net_greeks, leg, spot, rate, dividend):
            net_greeks = None
            pnl_today = None
            analytics_status = "NO_CONVERGENCE"

    breakevens = _breakevens(spot_range, pnl_at_expiry)
    max_profit, max_loss = _max_profit_loss(normalised_legs, pnl_at_expiry)

    return {
        "spot_range": _round_list(spot_range, 2),
        "pnl_at_expiry": _round_list(pnl_at_expiry, 2),
        "pnl_today": _round_list(pnl_today, 2) if pnl_today is not None else None,
        "analytical_status": analytics_status,
        "risk_free_rate_source": _describe_sources(rate_sources_used),
        "dividend_yield_source": _describe_sources(dividend_sources_used),
        "breakevens": breakevens,
        "max_profit": None if max_profit is None else round(max_profit, 2),
        "max_loss": None if max_loss is None else round(max_loss, 2),
        "net_greeks": (
            {key: round(value, 6) for key, value in net_greeks.items()}
            if net_greeks is not None else None
        ),
    }


def _resolve_rate(
    t: float | None,
    configured_rate: float | None,
    configured_rate_source: str | None,
) -> tuple[float | None, str | None]:
    """Prefer a genuinely observed current FBIL rate; otherwise accept only
    an explicitly disclosed configured assumption. Never a silent default."""
    if t is not None:
        try:
            selection = latest_available_rate(t)
        except Exception:
            selection = None
        if selection is not None:
            return selection.rate_decimal, OBSERVED_MARKET_RATE
    if (
        configured_rate is not None
        and math.isfinite(configured_rate)
        and configured_rate_source == CONFIGURED_ASSUMPTION
    ):
        return configured_rate, CONFIGURED_ASSUMPTION
    return None, None


def _resolve_dividend(
    configured_dividend: float | None,
    configured_dividend_source: str | None,
) -> tuple[float | None, str | None]:
    """No observed dividend-yield source exists in this system: only an
    explicitly disclosed configured assumption -- or an explicitly disclosed
    zero-dividend model assumption, which must genuinely be zero -- is ever
    accepted. Any other label (including no label at all) fails closed."""
    if (
        configured_dividend is not None
        and math.isfinite(configured_dividend)
        and configured_dividend_source == CONFIGURED_ASSUMPTION
    ):
        return configured_dividend, CONFIGURED_ASSUMPTION
    if (
        configured_dividend == 0.0
        and configured_dividend_source == ZERO_DIVIDEND_ASSUMPTION
    ):
        return configured_dividend, ZERO_DIVIDEND_ASSUMPTION
    return None, None


def _describe_sources(sources: set[str]) -> str | None:
    if not sources:
        return None
    if len(sources) == 1:
        return next(iter(sources))
    return "MIXED"


def _normalise_leg(raw_leg: dict[str, Any]) -> dict[str, Any]:
    option_type = str(raw_leg.get("option_type", "")).strip().upper()
    if option_type not in {"CE", "PE", "FUT"}:
        raise ValueError(f"unsupported option_type: {option_type}")

    action = str(raw_leg.get("action", "")).strip().upper()
    if action not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported action: {action}")

    quantity = int(raw_leg.get("quantity", 1))
    if "lot_size" not in raw_leg:
        raise ValueError("lot_size is required")
    lot_size = int(raw_leg["lot_size"])
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
    risk_free_rate: float | None,
    dividend_yield: float | None,
) -> np.ndarray | None:
    if leg["option_type"] == "FUT":
        return _expiry_leg_pnl(leg, spot_range)

    multiplier = _signed_multiplier(leg)
    sigma = _leg_iv(leg, current_spot, risk_free_rate, dividend_yield)
    t = _time_to_expiry(leg["expiry"])
    if sigma is None or t is None or risk_free_rate is None or dividend_yield is None:
        return None
    flag = _flag(leg["option_type"])

    values = np.array(
        [
            _bs_price(flag, float(next_spot), leg["strike"], t, risk_free_rate, sigma, dividend_yield)
            for next_spot in spot_range
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        return None
    return (values - leg["entry_price"]) * multiplier


def _add_leg_greeks(
    net_greeks: dict[str, float],
    leg: dict[str, Any],
    spot: float,
    risk_free_rate: float | None,
    dividend_yield: float | None,
) -> bool:
    multiplier = _signed_multiplier(leg)
    option_type = leg["option_type"]

    if option_type == "FUT":
        net_greeks["delta"] += multiplier
        return True

    sigma = _leg_iv(leg, spot, risk_free_rate, dividend_yield)
    t = _time_to_expiry(leg["expiry"])
    if sigma is None or t is None or risk_free_rate is None or dividend_yield is None:
        return False
    flag = _flag(option_type)
    strike = leg["strike"]

    try:
        net_greeks["delta"] += delta(flag, spot, strike, t, risk_free_rate, sigma, dividend_yield) * multiplier
        net_greeks["gamma"] += gamma(flag, spot, strike, t, risk_free_rate, sigma, dividend_yield) * multiplier
        net_greeks["theta"] += theta(flag, spot, strike, t, risk_free_rate, sigma, dividend_yield) * multiplier
        net_greeks["vega"] += vega(flag, spot, strike, t, risk_free_rate, sigma, dividend_yield) * multiplier
    except Exception:
        return False
    return True


def _leg_iv(
    leg: dict[str, Any],
    spot: float,
    risk_free_rate: float | None,
    dividend_yield: float | None,
) -> float | None:
    """Invert only with explicit configured analytical parameters."""
    if leg["option_type"] == "FUT":
        return 0.0

    flag = _flag(leg["option_type"])
    t = _time_to_expiry(leg["expiry"])
    if t is None or risk_free_rate is None or dividend_yield is None:
        return None
    solved = solve_iv_with_provenance(
        flag, leg["entry_price"], spot, leg["strike"], t, risk_free_rate,
        pricing_model=PricingModel.BLACK_SCHOLES_MERTON,
        dividend_yield=dividend_yield,
    )
    return float(solved.iv) if solved.status is SolveStatus.OK else None


def _bs_price(
    flag: str,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
) -> float:
    try:
        value = black_scholes_merton(flag, spot, strike, t, rate, sigma, dividend_yield)
    except Exception:
        return float("nan")

    if not math.isfinite(value):
        return float("nan")
    return float(value)


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


def _time_to_expiry(expiry: str) -> float | None:
    parsed = _parse_expiry(expiry)
    if parsed is None:
        return None
    now = _dt.datetime.now(_dt.timezone.utc)
    ist = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    expiry_at = _dt.datetime.combine(parsed, _dt.time(15, 30), ist).astimezone(_dt.timezone.utc)
    value = (expiry_at - now).total_seconds() / (365 * 86400)
    return value if value > 0 else None


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
