from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Any


def compute_iv_surface(chain_legs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build live IV surface inputs from option legs already enriched with IV.

    IV values are returned as decimals (0.18 = 18%). Sparse cells stay None so
    Plotly receives JSON nulls instead of NaN.
    """
    legs = _valid_legs(chain_legs)
    if not legs:
        return _empty_surface()

    strikes_by_expiry: dict[str, set[float]] = defaultdict(set)
    iv_by_key: dict[tuple[float, str, str], float] = {}

    for leg in legs:
        strikes_by_expiry[leg["expiry"]].add(leg["strike"])
        iv_by_key[(leg["strike"], leg["expiry"], leg["option_type"])] = leg["iv"]

    expiries = [
        expiry
        for expiry, strikes in strikes_by_expiry.items()
        if len(strikes) >= 3
    ]
    expiries.sort(key=_expiry_sort_key)

    if not expiries:
        return _empty_surface()

    expiry_set = set(expiries)
    strikes = sorted({
        leg["strike"]
        for leg in legs
        if leg["expiry"] in expiry_set
    })
    dtes = [_days_to_expiry(expiry) for expiry in expiries]

    call_iv_grid = _build_grid(strikes, expiries, "CE", iv_by_key)
    put_iv_grid = _build_grid(strikes, expiries, "PE", iv_by_key)
    spot = _derive_spot(legs, strikes)

    nearest_expiry = expiries[0]
    skew_strikes = sorted(strikes_by_expiry[nearest_expiry])
    call_ivs = [
        iv_by_key.get((strike, nearest_expiry, "CE"))
        for strike in skew_strikes
    ]
    put_ivs = [
        iv_by_key.get((strike, nearest_expiry, "PE"))
        for strike in skew_strikes
    ]

    atm_call_iv: list[float | None] = []
    atm_put_iv: list[float | None] = []
    atm_strikes: list[float | None] = []
    for expiry in expiries:
        expiry_strikes = sorted(strikes_by_expiry[expiry])
        atm_strike = _nearest_strike(spot, expiry_strikes)
        atm_strikes.append(atm_strike)
        if atm_strike is None:
            atm_call_iv.append(None)
            atm_put_iv.append(None)
            continue

        atm_call_iv.append(iv_by_key.get((atm_strike, expiry, "CE")))
        atm_put_iv.append(iv_by_key.get((atm_strike, expiry, "PE")))

    skew_atm = _nearest_strike(spot, skew_strikes)

    return {
        "surface": {
            "strikes": strikes,
            "expiries": expiries,
            "dtes": dtes,
            "call_iv_grid": call_iv_grid,
            "put_iv_grid": put_iv_grid,
        },
        "skew": {
            "expiry": nearest_expiry,
            "strikes": skew_strikes,
            "call_ivs": call_ivs,
            "put_ivs": put_ivs,
            "atm_strike": skew_atm,
        },
        "term_structure": {
            "expiries": expiries,
            "dtes": dtes,
            "atm_call_iv": atm_call_iv,
            "atm_put_iv": atm_put_iv,
            "atm_strikes": atm_strikes,
        },
    }


def _valid_legs(chain_legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for leg in chain_legs:
        strike = _coerce_float(leg.get("strike"))
        expiry = str(leg.get("expiry") or "").strip()
        option_type = _normalise_option_type(leg.get("option_type"))
        iv = _normalise_iv(leg.get("iv"))

        if strike is None or not expiry or option_type is None or iv is None:
            continue

        valid.append({
            "strike": strike,
            "expiry": expiry,
            "option_type": option_type,
            "iv": iv,
            "delta": _coerce_float(leg.get("delta")),
        })
    return valid


def _normalise_option_type(value: Any) -> str | None:
    option_type = str(value or "").strip().upper()
    if option_type in {"CE", "CALL", "C"}:
        return "CE"
    if option_type in {"PE", "PUT", "P"}:
        return "PE"
    return None


def _normalise_iv(value: Any) -> float | None:
    iv = _coerce_float(value)
    if iv is None or iv <= 0:
        return None

    # NSE sometimes exposes IV as 34.5 while py_vollib returns 0.345.
    if iv > 3.0:
        iv = iv / 100.0

    if iv <= 0 or iv > 5.0:
        return None
    return iv


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None
    return number


def _build_grid(
    strikes: list[float],
    expiries: list[str],
    option_type: str,
    iv_by_key: dict[tuple[float, str, str], float],
) -> list[list[float | None]]:
    return [
        [iv_by_key.get((strike, expiry, option_type)) for expiry in expiries]
        for strike in strikes
    ]


def _derive_spot(legs: list[dict[str, Any]], strikes: list[float]) -> float:
    best_strike: float | None = None
    best_distance: float | None = None

    for leg in legs:
        if leg["option_type"] != "CE" or leg["delta"] is None:
            continue
        distance = abs(float(leg["delta"]) - 0.5)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_strike = leg["strike"]

    if best_strike is not None:
        return best_strike

    if not strikes:
        return 0.0

    return (strikes[0] + strikes[-1]) / 2.0


def _nearest_strike(spot: float, strikes: list[float]) -> float | None:
    if not strikes:
        return None
    return min(strikes, key=lambda strike: abs(strike - spot))


def _expiry_sort_key(expiry: str) -> tuple[int, str]:
    parsed = _parse_expiry(expiry)
    if parsed is None:
        return (999_999, expiry.lower())
    return ((parsed - _dt.date.today()).days, expiry.lower())


def _days_to_expiry(expiry: str) -> int:
    parsed = _parse_expiry(expiry)
    if parsed is None:
        return 0
    return max((parsed - _dt.date.today()).days, 0)


def _parse_expiry(expiry: str) -> _dt.date | None:
    value = expiry.strip()
    for fmt in (
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ):
        try:
            return _dt.datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    compact = value.replace("-", "").replace(" ", "")
    for fmt in ("%d%b%Y", "%d%B%Y"):
        try:
            return _dt.datetime.strptime(compact, fmt).date()
        except ValueError:
            pass

    return None


def _empty_surface() -> dict[str, Any]:
    return {
        "surface": {
            "strikes": [],
            "expiries": [],
            "dtes": [],
            "call_iv_grid": [],
            "put_iv_grid": [],
        },
        "skew": {
            "expiry": "",
            "strikes": [],
            "call_ivs": [],
            "put_ivs": [],
            "atm_strike": None,
        },
        "term_structure": {
            "expiries": [],
            "dtes": [],
            "atm_call_iv": [],
            "atm_put_iv": [],
            "atm_strikes": [],
        },
    }
