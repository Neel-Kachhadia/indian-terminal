from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict, deque
from typing import Any

from data.exchange_calendar import today_ist
from db.write_service import ReadOperation, resolve_read
from options.iv_contract import validated_live_observation


TRADING_DAYS = 252
MIN_IV_HISTORY_ROWS = 30
_PCR_HISTORY: defaultdict[str, deque[float]] = defaultdict(
    lambda: deque(maxlen=20)
)


def compute_options_signals(
    legs: list[dict[str, Any]],
    spot: float,
    expiry_str: str,
    lot_size: int,
) -> dict[str, Any]:
    """
    Compute deterministic options-derived signals from live chain legs.

    IV rank/percentile are None when iv_history has fewer than 30 rows (unavailable, never a fabricated neutral value).
    GEX uses the validated instrument-master lot size and returns crores.
    """
    if not isinstance(lot_size, int) or lot_size <= 0:
        raise ValueError("validated positive lot_size is required")
    normalised = [_normalise_leg(leg) for leg in legs]
    normalised = [leg for leg in normalised if leg is not None]
    spot = _finite_or_zero(spot)
    oi_complete = bool(normalised) and all(leg.get("oi") is not None for leg in normalised)

    call_oi_by_strike: defaultdict[float, int] = defaultdict(int)
    put_oi_by_strike: defaultdict[float, int] = defaultdict(int)

    for leg in normalised:
        if not oi_complete:
            break
        if leg["option_type"] == "CE":
            call_oi_by_strike[leg["strike"]] += leg["oi"]
        elif leg["option_type"] == "PE":
            put_oi_by_strike[leg["strike"]] += leg["oi"]

    total_call_oi = sum(call_oi_by_strike.values())
    total_put_oi = sum(put_oi_by_strike.values())
    pcr = total_put_oi / total_call_oi if oi_complete and total_call_oi > 0 else None
    pcr_trend = _update_pcr_history(expiry_str, pcr)

    strikes = sorted(set(call_oi_by_strike) | set(put_oi_by_strike))
    atm_strike = _nearest_strike(spot, strikes)
    atm_iv = _atm_call_iv(normalised, spot, atm_strike)
    symbol = _first_symbol(normalised)
    iv_rank, iv_percentile = _compute_iv_rank(symbol, atm_strike, atm_iv)

    expected_move = _expected_move(spot, atm_iv, expiry_str)
    gex, gex_flip = _compute_gex(normalised, spot, lot_size, oi_complete)

    return {
        "gex": gex,
        "gex_flip": gex_flip,
        "lot_size": lot_size,
        "max_pain": _compute_max_pain(call_oi_by_strike, put_oi_by_strike, spot),
        "iv_rank": iv_rank,
        "iv_percentile": iv_percentile,
        "atm_iv": atm_iv,
        "expected_move": expected_move,
        "expected_move_pct": (
            expected_move / spot * 100.0
            if expected_move is not None and spot > 0
            else None
        ),
        "pcr": pcr,
        "pcr_trend": pcr_trend,
        "call_oi_peak": _peak_strike(call_oi_by_strike),
        "put_oi_peak": _peak_strike(put_oi_by_strike),
        "total_call_oi": int(total_call_oi) if oi_complete else None,
        "total_put_oi": int(total_put_oi) if oi_complete else None,
        "oi_complete": oi_complete,
    }


def _normalise_leg(leg: dict[str, Any]) -> dict[str, Any] | None:
    strike = _coerce_float(leg.get("strike"))
    option_type = _normalise_option_type(leg.get("option_type"))
    if strike is None or option_type is None:
        return None

    observation = validated_live_observation(leg)
    return {
        "symbol": str(leg.get("symbol") or "NIFTY").strip().upper(),
        "strike": strike,
        "option_type": option_type,
        "oi": _coerce_nonnegative_int(leg.get("oi")),
        "iv": _normalise_iv(observation.get("iv")) if observation else None,
        "gamma": _coerce_float(leg.get("gamma")) if observation else None,
        "delta": _coerce_float(leg.get("delta")) if observation else None,
    }


def _compute_gex(
    legs: list[dict[str, Any]],
    spot: float,
    lot_size: int,
    oi_complete: bool,
) -> tuple[float | None, float | None]:
    if spot <= 0 or not oi_complete:
        return None, None

    exposure_by_strike: defaultdict[float, float] = defaultdict(float)
    total_exposure = 0.0

    for leg in legs:
        gamma = leg.get("gamma")
        oi = int(leg.get("oi") or 0)
        if gamma is None or oi <= 0:
            continue

        contribution = float(gamma) * oi * spot * spot * lot_size
        if leg["option_type"] == "PE":
            contribution *= -1.0

        exposure_by_strike[leg["strike"]] += contribution
        total_exposure += contribution

    if not exposure_by_strike:
        return None, None
    return total_exposure / 1e7, _compute_gex_flip(exposure_by_strike, spot)


def _compute_gex_flip(
    exposure_by_strike: dict[float, float],
    spot: float,
) -> float | None:
    if not exposure_by_strike:
        return None

    running = 0.0
    previous_strike: float | None = None
    previous_running: float | None = None
    best_strike = min(exposure_by_strike, key=lambda strike: abs(strike - spot))
    best_abs = math.inf

    for strike in sorted(exposure_by_strike):
        running += exposure_by_strike[strike]

        if abs(running) < best_abs:
            best_abs = abs(running)
            best_strike = strike

        if previous_running is None or previous_strike is None:
            previous_running = running
            previous_strike = strike
            continue

        if previous_running == 0:
            return previous_strike
        if running == 0:
            return strike
        if previous_running * running < 0:
            weight = abs(previous_running) / (
                abs(previous_running) + abs(running)
            )
            return previous_strike + (strike - previous_strike) * weight

        previous_running = running
        previous_strike = strike

    return best_strike


def _compute_max_pain(
    call_oi_by_strike: dict[float, int],
    put_oi_by_strike: dict[float, int],
    spot: float,
) -> float | None:
    candidates = sorted(set(call_oi_by_strike) | set(put_oi_by_strike))
    if not candidates:
        return None

    best_strike = candidates[0]
    best_loss: float | None = None

    for expiry_price in candidates:
        total_loss = 0.0
        for strike, oi in call_oi_by_strike.items():
            total_loss += max(expiry_price - strike, 0.0) * oi
        for strike, oi in put_oi_by_strike.items():
            total_loss += max(strike - expiry_price, 0.0) * oi

        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_strike = expiry_price
        elif total_loss == best_loss and abs(expiry_price - spot) < abs(
            best_strike - spot
        ):
            best_strike = expiry_price

    return best_strike


def _compute_iv_rank(
    symbol: str,
    atm_strike: float | None,
    current_iv: float | None,
) -> tuple[float | None, float | None]:
    if current_iv is None or current_iv <= 0 or atm_strike is None:
        return None, None

    from_day = today_ist() - _dt.timedelta(days=370)
    symbols = _symbol_variants(symbol)

    try:
        result = resolve_read(
            ReadOperation.LOAD_IV_RANK_HISTORY,
            {"symbols": symbols, "from_date": from_day.isoformat()},
        )
        rows = result["rows"]
    except Exception as error:
        print(f"signals: iv_history query failed: {error}")
        return None, None

    iv_by_day: dict[str, tuple[float, float]] = {}
    for row in rows:
        date_text = str(row[0])
        strike = _coerce_float(row[1])
        iv = _normalise_iv(row[2])
        if strike is None or iv is None:
            continue

        distance = abs(strike - atm_strike)
        existing = iv_by_day.get(date_text)
        if existing is None or distance < existing[0]:
            iv_by_day[date_text] = (distance, iv)

    history = [iv for _, iv in iv_by_day.values()]
    history = history[-TRADING_DAYS:]
    if len(history) < MIN_IV_HISTORY_ROWS:
        return None, None

    low_iv = min(history)
    high_iv = max(history)
    if high_iv - low_iv <= 1e-9:
        return None, None
    iv_rank = (current_iv - low_iv) / (high_iv - low_iv) * 100.0

    iv_percentile = (
        sum(1 for iv in history if iv < current_iv) / len(history) * 100.0
    )
    return _clamp(iv_rank, 0.0, 100.0), _clamp(iv_percentile, 0.0, 100.0)


def _atm_call_iv(
    legs: list[dict[str, Any]],
    spot: float,
    atm_strike: float | None,
) -> float | None:
    call_legs = [
        leg
        for leg in legs
        if leg["option_type"] == "CE" and leg.get("iv") is not None
    ]
    if not call_legs:
        return None

    target = atm_strike if atm_strike is not None else spot
    if target <= 0:
        target = call_legs[0]["strike"]

    atm_leg = min(call_legs, key=lambda leg: abs(leg["strike"] - target))
    return float(atm_leg["iv"])


def _expected_move(spot: float, atm_iv: float | None, expiry_str: str) -> float | None:
    if spot <= 0 or atm_iv is None or atm_iv <= 0:
        return None

    expiry = _parse_expiry(expiry_str)
    if expiry is None:
        return None
    dte = max((expiry - today_ist()).days, 1)

    return spot * atm_iv * math.sqrt(dte / 365.0)


def _update_pcr_history(expiry_str: str, pcr: float | None) -> float | None:
    key = expiry_str.strip().upper() or "DEFAULT"
    if pcr is None:
        return None
    if pcr >= 0 and math.isfinite(pcr):
        _PCR_HISTORY[key].append(pcr)

    history = list(_PCR_HISTORY[key])[-5:]
    if not history:
        return pcr

    return sum(history) / len(history)


def _peak_strike(oi_by_strike: dict[float, int]) -> float | None:
    if not oi_by_strike:
        return None
    return max(oi_by_strike, key=lambda strike: oi_by_strike[strike])


def _nearest_strike(spot: float, strikes: list[float]) -> float | None:
    if not strikes:
        return None
    if spot <= 0:
        return strikes[len(strikes) // 2]
    return min(strikes, key=lambda strike: abs(strike - spot))


def _first_symbol(legs: list[dict[str, Any]]) -> str:
    for leg in legs:
        symbol = str(leg.get("symbol") or "").strip().upper()
        if symbol:
            return symbol
    return "NIFTY"


def _symbol_variants(symbol: str) -> list[str]:
    normalized = symbol.strip().upper()
    if normalized in {"NIFTY", "NIFTY 50"}:
        return ["NIFTY", "NIFTY 50"]
    return [normalized]


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


def _coerce_nonnegative_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _finite_or_zero(value: Any) -> float:
    number = _coerce_float(value)
    return number if number is not None else 0.0


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
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
