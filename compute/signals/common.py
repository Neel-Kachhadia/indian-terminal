from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from data.exchange_calendar import exchange_calendar, today_ist
from options.iv_contract import validated_live_observation


TRADING_DAYS = 252


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def as_int(value: Any, default: int = 0) -> int:
    number = as_float(value)
    if number is None:
        return default
    return int(number)


def safe_div(
    numerator: Any,
    denominator: Any,
    default: float | None = None,
) -> float | None:
    top = as_float(numerator)
    bottom = as_float(denominator)
    if top is None or bottom is None or abs(bottom) < 1e-12:
        return default
    return top / bottom


def clamp(value: Any, low: float, high: float) -> float | None:
    number = as_float(value)
    if number is None:
        return None
    return max(low, min(high, number))


def last(series: pd.Series, default: float | None = None) -> float | None:
    if series is None or len(series) == 0:
        return default
    value = series.dropna()
    if value.empty:
        return default
    return as_float(value.iloc[-1], default)


def nth_from_end(
    series: pd.Series,
    n: int,
    default: float | None = None,
) -> float | None:
    clean = series.dropna()
    if len(clean) <= n:
        return default
    return as_float(clean.iloc[-1 - n], default)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or getattr(df, "empty", True) or column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def current_price(df: pd.DataFrame, spot: float | None) -> float:
    spot_value = as_float(spot, 0.0) or 0.0
    if spot_value > 0:
        return spot_value
    return last(numeric_series(df, "close"), 0.0) or 0.0


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change().replace([np.inf, -np.inf], np.nan)


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=max(2, period // 2)).mean()


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=max(2, period // 2)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    output = 100 - (100 / (1 + rs))
    return output.fillna(50.0)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, 12) - ema(close, 26)
    signal = line.ewm(span=9, adjust=False, min_periods=5).mean()
    hist = line - signal
    return line, signal, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    close = numeric_series(df, "close")
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(close, period)
    std = close.rolling(period, min_periods=max(2, period // 2)).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def historical_vol(close: pd.Series, period: int = 20) -> pd.Series:
    return returns(close).rolling(period, min_periods=max(5, period // 2)).std(ddof=0) * math.sqrt(TRADING_DAYS) * 100.0


def roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(periods=period) * 100.0


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = numeric_series(df, "close")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    highest = high.rolling(period, min_periods=max(2, period // 2)).max()
    lowest = low.rolling(period, min_periods=max(2, period // 2)).min()
    return (close - lowest) / (highest - lowest).replace(0, np.nan) * 100.0


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = numeric_series(df, "close")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    highest = high.rolling(period, min_periods=max(2, period // 2)).max()
    lowest = low.rolling(period, min_periods=max(2, period // 2)).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    close = numeric_series(df, "close")
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period, min_periods=max(2, period // 2)).mean()
    mad = typical.rolling(period, min_periods=max(2, period // 2)).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    return (typical - mean) / (0.015 * mad.replace(0, np.nan))


def trix(close: pd.Series, period: int = 14) -> pd.Series:
    triple = ema(ema(ema(close, period), period), period)
    return triple.pct_change() * 100.0


def linear_slope_r2(series: pd.Series, period: int) -> tuple[float | None, float | None]:
    clean = series.dropna()
    if len(clean) < period:
        return None, None
    y = clean.iloc[-period:].to_numpy(dtype=float)
    x = np.arange(period, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(r2)


def z_score(value: Any, history: Sequence[Any]) -> float | None:
    current = as_float(value)
    numbers = [as_float(item) for item in history]
    numbers = [item for item in numbers if item is not None]
    if current is None or len(numbers) < 2:
        return None
    std = float(np.std(numbers, ddof=0))
    if std <= 1e-12:
        return None
    return (current - float(np.mean(numbers))) / std


def percentile(value: Any, history: Sequence[Any]) -> float | None:
    current = as_float(value)
    numbers = [as_float(item) for item in history]
    numbers = [item for item in numbers if item is not None]
    if current is None or not numbers:
        return None
    return sum(1 for item in numbers if item <= current) / len(numbers) * 100.0


def cross_signal(short: pd.Series, long: pd.Series, lookback: int = 3) -> int:
    diff = (short - long).dropna()
    if len(diff) < 2:
        return 0
    recent = diff.iloc[-(lookback + 1):]
    for index in range(1, len(recent)):
        previous = recent.iloc[index - 1]
        current = recent.iloc[index]
        if previous <= 0 < current:
            return 1
        if previous >= 0 > current:
            return -1
    return 0


def days_since_cross(short: pd.Series, long: pd.Series) -> int | None:
    diff = (short - long).dropna()
    if len(diff) < 2:
        return None
    latest_sign = 1 if diff.iloc[-1] >= 0 else -1
    for offset in range(1, len(diff)):
        current = diff.iloc[-offset]
        previous = diff.iloc[-offset - 1]
        if previous <= 0 < current or previous >= 0 > current:
            return offset if latest_sign > 0 else -offset
    return len(diff) if latest_sign > 0 else -len(diff)


def parse_expiry(expiry: str) -> _dt.date | None:
    value = str(expiry or "").strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def dte(expiry: str, today: _dt.date | None = None) -> int | None:
    expiry_day = parse_expiry(expiry)
    if expiry_day is None:
        return None
    base = today or today_ist()
    return max((expiry_day - base).days, 0)


def next_tuesday(today: _dt.date | None = None) -> _dt.date:
    base = today or today_ist()
    calendar = exchange_calendar()
    instant = _dt.datetime.combine(
        base,
        _dt.time.min,
        tzinfo=calendar.timezone,
    )
    return calendar.next_weekly_expiries(1, instant)[0]


def last_tuesday_of_month(day: _dt.date) -> _dt.date:
    return exchange_calendar().monthly_expiry(day.year, day.month)


def next_budget_day(today: _dt.date | None = None) -> _dt.date:
    base = today or today_ist()
    candidate = _dt.date(base.year, 2, 1)
    if candidate < base:
        candidate = _dt.date(base.year + 1, 2, 1)
    return candidate


def results_season_week(today: _dt.date | None = None) -> int | None:
    base = today or today_ist()
    season_starts = [
        _dt.date(base.year, 1, 1),
        _dt.date(base.year, 4, 1),
        _dt.date(base.year, 7, 1),
        _dt.date(base.year, 10, 1),
    ]
    for start in season_starts:
        delta = (base - start).days
        if 0 <= delta < 42:
            return delta // 7 + 1
    return None


def normalise_leg(leg: dict[str, Any]) -> dict[str, Any] | None:
    strike = as_float(leg.get("strike"))
    if strike is None:
        return None
    option_type = str(leg.get("option_type") or "").strip().upper()
    if option_type in {"CALL", "C"}:
        option_type = "CE"
    if option_type in {"PUT", "P"}:
        option_type = "PE"
    if option_type not in {"CE", "PE"}:
        return None
    observation = validated_live_observation(leg)
    return {
        "symbol": str(leg.get("symbol") or "NIFTY").strip().upper(),
        "expiry": str(leg.get("expiry") or ""),
        "strike": strike,
        "option_type": option_type,
        "ltp": as_float(leg.get("ltp"), 0.0) or 0.0,
        "bid": as_float(leg.get("bid"), 0.0) or 0.0,
        "ask": as_float(leg.get("ask"), 0.0) or 0.0,
        "volume": as_int(leg.get("volume")),
        "oi": as_int(leg.get("oi")),
        "oi_change": as_int(leg.get("oi_change")),
        "iv": normalise_iv(observation.get("iv")) if observation else None,
        "delta": as_float(leg.get("delta")) if observation else None,
        "gamma": as_float(leg.get("gamma")) if observation else None,
        "theta": as_float(leg.get("theta")) if observation else None,
        "vega": as_float(leg.get("vega")) if observation else None,
    }


def normalise_legs(legs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in (normalise_leg(leg) for leg in legs) if item is not None]


def normalise_iv(value: Any) -> float | None:
    iv = as_float(value)
    if iv is None or iv <= 0:
        return None
    if iv <= 0 or iv > 5.0:
        return None
    return iv


def option_chain_summary(
    legs: Iterable[dict[str, Any]],
    spot: float,
) -> dict[str, Any]:
    normalised = normalise_legs(legs)
    calls = [leg for leg in normalised if leg["option_type"] == "CE"]
    puts = [leg for leg in normalised if leg["option_type"] == "PE"]
    strikes = sorted({leg["strike"] for leg in normalised})
    atm_strike = min(strikes, key=lambda strike: abs(strike - spot)) if strikes and spot > 0 else (strikes[len(strikes) // 2] if strikes else None)

    call_oi = defaultdict(int)
    put_oi = defaultdict(int)
    call_volume = defaultdict(int)
    put_volume = defaultdict(int)
    call_oi_change = defaultdict(int)
    put_oi_change = defaultdict(int)
    by_key: dict[tuple[float, str], dict[str, Any]] = {}

    for leg in normalised:
        strike = leg["strike"]
        by_key[(strike, leg["option_type"])] = leg
        if leg["option_type"] == "CE":
            call_oi[strike] += leg["oi"]
            call_volume[strike] += leg["volume"]
            call_oi_change[strike] += leg["oi_change"]
        else:
            put_oi[strike] += leg["oi"]
            put_volume[strike] += leg["volume"]
            put_oi_change[strike] += leg["oi_change"]

    total_call_oi = sum(call_oi.values())
    total_put_oi = sum(put_oi.values())
    total_call_volume = sum(call_volume.values())
    total_put_volume = sum(put_volume.values())
    total_oi = total_call_oi + total_put_oi

    return {
        "legs": normalised,
        "calls": calls,
        "puts": puts,
        "strikes": strikes,
        "atm_strike": atm_strike,
        "atm_call": by_key.get((atm_strike, "CE")) if atm_strike is not None else None,
        "atm_put": by_key.get((atm_strike, "PE")) if atm_strike is not None else None,
        "call_oi": dict(call_oi),
        "put_oi": dict(put_oi),
        "call_volume": dict(call_volume),
        "put_volume": dict(put_volume),
        "call_oi_change": dict(call_oi_change),
        "put_oi_change": dict(put_oi_change),
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_call_volume": total_call_volume,
        "total_put_volume": total_put_volume,
        "total_oi": total_oi,
        "pcr": safe_div(total_put_oi, total_call_oi),
    }


def peak_key(values: dict[float, int | float]) -> float | None:
    if not values:
        return None
    return max(values, key=lambda key: values[key])


def max_pain(summary: dict[str, Any], spot: float) -> float | None:
    candidates = sorted(set(summary["call_oi"]) | set(summary["put_oi"]))
    if not candidates:
        return None
    best = candidates[0]
    best_loss: float | None = None
    for price in candidates:
        loss = 0.0
        for strike, oi in summary["call_oi"].items():
            loss += max(price - strike, 0.0) * oi
        for strike, oi in summary["put_oi"].items():
            loss += max(strike - price, 0.0) * oi
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best = price
        elif loss == best_loss and spot > 0 and abs(price - spot) < abs(best - spot):
            best = price
    return best


def gex(
    summary: dict[str, Any],
    spot: float,
    lot_size: int | None,
) -> tuple[float | None, float | None]:
    if spot <= 0 or lot_size is None or lot_size <= 0:
        return None, None
    exposure_by_strike: defaultdict[float, float] = defaultdict(float)
    total = 0.0
    for leg in summary["legs"]:
        gamma = as_float(leg.get("gamma"))
        oi = as_int(leg.get("oi"))
        if gamma is None or oi <= 0:
            continue
        contribution = gamma * oi * spot * spot * lot_size
        if leg["option_type"] == "PE":
            contribution *= -1.0
        exposure_by_strike[leg["strike"]] += contribution
        total += contribution

    return total / 1e7, _gex_flip(exposure_by_strike, spot)


def _gex_flip(exposure_by_strike: dict[float, float], spot: float) -> float | None:
    if not exposure_by_strike:
        return None
    running = 0.0
    previous_strike: float | None = None
    previous_running: float | None = None
    best = min(exposure_by_strike, key=lambda strike: abs(strike - spot))
    best_abs = math.inf
    for strike in sorted(exposure_by_strike):
        running += exposure_by_strike[strike]
        if abs(running) < best_abs:
            best_abs = abs(running)
            best = strike
        if previous_running is not None and previous_strike is not None:
            if previous_running == 0:
                return previous_strike
            if running == 0:
                return strike
            if previous_running * running < 0:
                weight = abs(previous_running) / (abs(previous_running) + abs(running))
                return previous_strike + (strike - previous_strike) * weight
        previous_running = running
        previous_strike = strike
    return best


def nearest_delta_leg(
    legs: Sequence[dict[str, Any]],
    option_type: str,
    target_delta: float,
) -> dict[str, Any] | None:
    candidates = [
        leg for leg in legs
        if leg["option_type"] == option_type and as_float(leg.get("delta")) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda leg: abs((as_float(leg.get("delta")) or 0.0) - target_delta))


def slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    y = np.asarray(values, dtype=float)
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, y, 1)[0])
