import datetime
import math
from typing import Any

import numpy as np
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
from py_vollib.black_scholes.implied_volatility import implied_volatility

try:
    from py_vollib_vectorized import get_all_greeks, vectorized_implied_volatility

    _VECTOR_IMPORT_ERROR: str | None = None
except ImportError as error:
    get_all_greeks = None
    vectorized_implied_volatility = None
    _VECTOR_IMPORT_ERROR = str(error)
except Exception as error:
    get_all_greeks = None
    vectorized_implied_volatility = None
    _VECTOR_IMPORT_ERROR = str(error)


RISK_FREE_RATE = 0.065
TRADING_DAYS_PER_YEAR = 252
MIN_LIQUID_LTP = 0.5
MIN_IV = 0.01
MAX_IV = 5.0
_FALLBACK_NOTICE_PRINTED = False


def compute_chain_greeks(
    legs: list[dict[str, Any]],
    spot: float,
    expiry_str: str,
) -> list[dict[str, Any]]:
    """
    Compute Black-Scholes IV and Greeks for a full options chain.

    Outputs are validated per leg; failed or illiquid legs keep Greeks as None.
    Vega follows py_vollib/Data Contracts convention: option price move per 1%
    volatility move.
    """
    output = [_with_empty_greeks(leg) for leg in legs]
    if spot <= 0:
        return output

    t = _time_to_expiry(expiry_str)
    candidates = _candidate_indices(output)
    if not candidates:
        return output

    if vectorized_implied_volatility is not None and get_all_greeks is not None:
        _compute_vectorized(output, candidates, spot, t)
    else:
        global _FALLBACK_NOTICE_PRINTED
        if _VECTOR_IMPORT_ERROR and not _FALLBACK_NOTICE_PRINTED:
            print(
                "greeks: vectorized import failed, using py_vollib: "
                f"{_first_error_line(_VECTOR_IMPORT_ERROR)}"
            )
            _FALLBACK_NOTICE_PRINTED = True
        _compute_fallback(output, candidates, spot, t)

    _validate_atm_call(output, spot)
    return output


def _compute_vectorized(
    legs: list[dict[str, Any]],
    indices: list[int],
    spot: float,
    t: float,
) -> None:
    prices = np.array([float(legs[index]["ltp"]) for index in indices], dtype=np.float64)
    strikes = np.array([float(legs[index]["strike"]) for index in indices], dtype=np.float64)
    flags = np.array([_flag(legs[index]["option_type"]) for index in indices])
    spots = np.full(len(indices), spot, dtype=np.float64)
    times = np.full(len(indices), t, dtype=np.float64)
    rates = np.full(len(indices), RISK_FREE_RATE, dtype=np.float64)

    try:
        ivs = vectorized_implied_volatility(
            price=prices,
            S=spots,
            K=strikes,
            t=times,
            r=rates,
            flag=flags,
            model="black_scholes",
            on_error="ignore",
            return_as="numpy",
        )
        greeks = get_all_greeks(
            flags,
            spots,
            strikes,
            times,
            rates,
            ivs,
            model="black_scholes",
            return_as="dict",
        )
    except Exception as error:
        print(f"greeks: vectorized compute failed, using py_vollib: {error}")
        _compute_fallback(legs, indices, spot, t)
        return

    for offset, index in enumerate(indices):
        values = {
            "iv": _finite_float(ivs[offset]),
            "delta": _finite_float(greeks["delta"][offset]),
            "gamma": _finite_float(greeks["gamma"][offset]),
            "theta": _finite_float(greeks["theta"][offset]),
            "vega": _finite_float(greeks["vega"][offset]),
        }
        _apply_if_valid(legs[index], values)


def _compute_fallback(
    legs: list[dict[str, Any]],
    indices: list[int],
    spot: float,
    t: float,
) -> None:
    for index in indices:
        leg = legs[index]
        try:
            flag = _flag(leg["option_type"])
            strike = float(leg["strike"])
            price = float(leg["ltp"])
            sigma = implied_volatility(price, spot, strike, t, RISK_FREE_RATE, flag)
            values = {
                "iv": _finite_float(sigma),
                "delta": _finite_float(delta(flag, spot, strike, t, RISK_FREE_RATE, sigma)),
                "gamma": _finite_float(gamma(flag, spot, strike, t, RISK_FREE_RATE, sigma)),
                "theta": _finite_float(theta(flag, spot, strike, t, RISK_FREE_RATE, sigma)),
                "vega": _finite_float(vega(flag, spot, strike, t, RISK_FREE_RATE, sigma)),
            }
            _apply_if_valid(leg, values)
        except Exception:
            _clear_greeks(leg)


def _with_empty_greeks(leg: dict[str, Any]) -> dict[str, Any]:
    next_leg = dict(leg)
    next_leg["iv"] = None
    next_leg["delta"] = None
    next_leg["gamma"] = None
    next_leg["theta"] = None
    next_leg["vega"] = None
    return next_leg


def _candidate_indices(legs: list[dict[str, Any]]) -> list[int]:
    indices = []
    for index, leg in enumerate(legs):
        try:
            if float(leg.get("ltp") or 0.0) <= MIN_LIQUID_LTP:
                continue
            if float(leg.get("strike") or 0.0) <= 0.0:
                continue
            _flag(leg.get("option_type"))
        except (TypeError, ValueError):
            continue
        indices.append(index)
    return indices


def _time_to_expiry(expiry_str: str) -> float:
    expiry_date = datetime.datetime.strptime(expiry_str, "%d-%b-%Y").date()
    today = datetime.date.today()
    dte = max((expiry_date - today).days, 1)
    return dte / 365.0


def _flag(option_type: Any) -> str:
    normalized = str(option_type).upper()
    if normalized == "CE":
        return "c"
    if normalized == "PE":
        return "p"
    raise ValueError(f"unsupported option_type: {option_type}")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_error_line(message: str) -> str:
    return next((line for line in message.splitlines() if line.strip()), message)


def _apply_if_valid(leg: dict[str, Any], values: dict[str, float | None]) -> None:
    if not _valid_values(values, str(leg.get("option_type", ""))):
        _clear_greeks(leg)
        return
    for key, value in values.items():
        leg[key] = value


def _valid_values(values: dict[str, float | None], option_type: str) -> bool:
    iv = values["iv"]
    option_delta = values["delta"]
    option_gamma = values["gamma"]
    option_theta = values["theta"]
    option_vega = values["vega"]

    if iv is None or iv < MIN_IV or iv > MAX_IV:
        return False
    if option_delta is None:
        return False
    if option_type.upper() == "CE" and not 0.0 <= option_delta <= 1.0:
        return False
    if option_type.upper() == "PE" and not -1.0 <= option_delta <= 0.0:
        return False
    if option_gamma is None or option_gamma < 0.0:
        return False
    if option_theta is None or option_theta >= 0.0:
        return False
    if option_vega is None or option_vega <= 0.0:
        return False
    return True


def _validate_atm_call(legs: list[dict[str, Any]], spot: float) -> None:
    call_indices = [
        index
        for index, leg in enumerate(legs)
        if str(leg.get("option_type", "")).upper() == "CE"
    ]
    if not call_indices:
        return

    atm_index = min(call_indices, key=lambda index: abs(float(legs[index]["strike"]) - spot))
    atm_delta = legs[atm_index].get("delta")
    if atm_delta is None or not 0.40 <= float(atm_delta) <= 0.60:
        _clear_greeks(legs[atm_index])


def _clear_greeks(leg: dict[str, Any]) -> None:
    leg["iv"] = None
    leg["delta"] = None
    leg["gamma"] = None
    leg["theta"] = None
    leg["vega"] = None
