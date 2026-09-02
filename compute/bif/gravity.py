from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Any

from data.exchange_calendar import today_ist


def compute_ngi(
    chain_legs: list[dict[str, Any]],
    spot: float,
    expiry: str,
) -> dict[str, Any]:
    """
    Nifty Gravity Index.

    Force(s) = OI(s) / distance(spot, s)^2, signed by option-side mechanics.
    """
    normalised = [_normalise_leg(leg) for leg in chain_legs]
    normalised = [leg for leg in normalised if leg is not None]
    spot_value = _finite(spot)
    dte_factor = _dte_factor(expiry)

    missing_inputs: list[str] = []
    if spot_value is None or spot_value <= 0:
        missing_inputs.append("decision_eligible_spot")
    if not normalised:
        missing_inputs.append("option_chain_open_interest")
    if dte_factor is None:
        missing_inputs.append("valid_expiry")
    if missing_inputs:
        return {
            "ngi": None,
            "ngi_direction": None,
            "gravity_center": None,
            "gravity_strength": None,
            "escape_velocity_up": None,
            "escape_velocity_down": None,
            "pin_probability": None,
            "dte_factor": round(dte_factor, 3) if dte_factor is not None else None,
            "top_gravity_strikes": [],
            "chain_leg_count": len(normalised),
            "missing_inputs": missing_inputs,
            "reason_code": "MISSING_REQUIRED_INPUTS",
            "is_valid": False,
        }
    spot = float(spot_value)
    assert dte_factor is not None

    force_rows: list[dict[str, Any]] = []
    force_by_strike: defaultdict[float, float] = defaultdict(float)
    oi_by_strike: defaultdict[float, int] = defaultdict(int)
    total_force = 0.0
    total_oi = 0

    for leg in normalised:
        distance = max(abs(spot - leg["strike"]), 25.0)
        raw_force = leg["oi"] / (distance * distance)
        signed_force = raw_force * _force_sign(leg["option_type"], leg["strike"], spot)
        weighted_force = signed_force * dte_factor
        force_by_strike[leg["strike"]] += weighted_force
        oi_by_strike[leg["strike"]] += leg["oi"]
        total_force += weighted_force
        total_oi += leg["oi"]
        force_rows.append({
            "strike": leg["strike"],
            "option_type": leg["option_type"],
            "oi": leg["oi"],
            "distance": round(distance, 2),
            "force": weighted_force,
        })

    ngi = math.tanh(total_force / 1_000_000.0) * 100.0
    gravity_center = (
        sum(strike * oi for strike, oi in oi_by_strike.items()) / total_oi
        if total_oi > 0
        else None
    )
    escape_up, escape_down = _escape_velocities(force_by_strike, spot, abs(ngi))
    pin_probability = _pin_probability(spot, gravity_center, abs(ngi), dte_factor)
    top_rows = sorted(force_rows, key=lambda row: abs(row["force"]), reverse=True)[:10]

    return {
        "ngi": round(_clamp(ngi, -100.0, 100.0), 3),
        "ngi_direction": _direction(ngi),
        "gravity_center": _round_or_none(gravity_center),
        "gravity_strength": round(abs(ngi), 3),
        "escape_velocity_up": _round_or_none(escape_up),
        "escape_velocity_down": _round_or_none(escape_down),
        "pin_probability": round(pin_probability, 3),
        "dte_factor": round(dte_factor, 3),
        "top_gravity_strikes": [
            {
                "strike": row["strike"],
                "option_type": row["option_type"],
                "oi": row["oi"],
                "distance": row["distance"],
                "force": round(row["force"], 6),
            }
            for row in top_rows
        ],
        "chain_leg_count": len(normalised),
        "missing_inputs": [],
        "reason_code": None,
        "is_valid": True,
    }


def _normalise_leg(leg: dict[str, Any]) -> dict[str, Any] | None:
    strike = _finite(leg.get("strike"))
    oi = _finite(leg.get("oi"))
    option_type = str(leg.get("option_type") or "").strip().upper()
    if option_type in {"CALL", "C"}:
        option_type = "CE"
    if option_type in {"PUT", "P"}:
        option_type = "PE"
    if strike is None or strike <= 0 or oi is None or oi <= 0 or option_type not in {"CE", "PE"}:
        return None
    return {
        "strike": float(strike),
        "oi": int(oi),
        "option_type": option_type,
    }


def _force_sign(option_type: str, strike: float, spot: float) -> float:
    if option_type == "CE":
        return 1.0 if strike >= spot else -0.3
    return -1.0 if strike <= spot else 0.3


def _dte_factor(expiry: str) -> float | None:
    dte = _days_to_expiry(expiry)
    if dte is None:
        return None
    return 1.0 / (1.0 + math.exp(0.8 * (float(dte) - 5.0)))


def _days_to_expiry(expiry: str) -> int | None:
    parsed = _parse_expiry(expiry)
    if parsed is None:
        return None
    return max((parsed - today_ist()).days, 0)


def _parse_expiry(expiry: str) -> _dt.date | None:
    clean = str(expiry or "").strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(clean, fmt).date()
        except ValueError:
            pass
    return None


def _escape_velocities(
    force_by_strike: dict[float, float],
    spot: float,
    strength: float,
) -> tuple[float | None, float | None]:
    above = [(strike, force) for strike, force in force_by_strike.items() if strike > spot]
    below = [(strike, force) for strike, force in force_by_strike.items() if strike < spot]
    scale = 1.0 + min(strength, 100.0) / 100.0
    up = None
    down = None
    if above:
        strike = max(above, key=lambda item: abs(item[1]))[0]
        up = max(25.0, strike - spot) * scale
    if below:
        strike = max(below, key=lambda item: abs(item[1]))[0]
        down = max(25.0, spot - strike) * scale
    return up, down


def _pin_probability(
    spot: float,
    gravity_center: float | None,
    strength: float,
    dte_factor: float,
) -> float:
    if gravity_center is None or spot <= 0:
        return 0.0
    center_distance = abs(spot - gravity_center)
    center_bonus = max(0.0, 1.0 - center_distance / 250.0) * 30.0
    strength_part = min(70.0, strength * 0.70)
    return _clamp((center_bonus + strength_part) * dte_factor, 0.0, 100.0)


def _direction(value: float) -> str:
    if value > 5:
        return "UP"
    if value < -5:
        return "DOWN"
    return "NEUTRAL"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 3)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
