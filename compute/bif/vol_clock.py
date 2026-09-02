from __future__ import annotations

import math
from typing import Any

import numpy as np


MINIMUM_IV_OBSERVATIONS = 30


def compute_vrc(
    iv_history: list[dict[str, Any]],
    option_signals: dict[str, Any],
    ohlcv_rows: list[dict[str, Any]],
    signal_values: dict[str, float],
) -> dict[str, Any]:
    """
    Volatility Regime Clock.

    IV is represented as decimal annualized volatility; VRP is returned in
    percentage points for readability.
    """
    current_iv = _normalise_iv(
        option_signals.get("atm_iv")
        or signal_values.get("options.atm_iv")
        or signal_values.get("atm_iv")
    )
    history = [_normalise_iv(row.get("atm_iv") or row.get("iv")) for row in iv_history]
    history = [value for value in history if value is not None and value > 0]
    if current_iv is not None and (not history or abs(history[-1] - current_iv) > 1e-9):
        history.append(current_iv)

    if current_iv is None or current_iv <= 0:
        return _unavailable_result("NO_CURRENT_IV", len(history))

    iv_percentile = _iv_percentile(current_iv, history)
    if iv_percentile is None:
        return _unavailable_result("INSUFFICIENT_HISTORY", len(history))

    slope_now = _slope(history[-5:])
    slope_prev = _slope(history[-10:-5])
    acceleration = slope_now - slope_prev
    trend = _trend_label(slope_now)
    clock_label, position = _clock_position(iv_percentile, trend)
    hv20 = _historical_vol_20(ohlcv_rows)
    vrp = (current_iv - hv20) * 100.0 if hv20 is not None else None
    mean_30 = sum(history[-30:]) / len(history[-30:]) if history else current_iv
    expected_iv_7d = current_iv + 0.15 * (mean_30 - current_iv) * 7.0

    return {
        "clock_position": float(position),
        "clock_label": clock_label,
        "iv_percentile": round(_clamp(iv_percentile, 0.0, 100.0), 3),
        "iv_trend": trend,
        "iv_acceleration": round(acceleration * 100.0, 6),
        "vrp": round(vrp, 3) if vrp is not None else None,
        "vrp_signal": _vrp_signal(vrp) if vrp is not None else None,
        "optimal_strategy": _optimal_strategy(clock_label, vrp),
        "expected_iv_7d": round(max(0.0, expected_iv_7d), 6),
        "is_valid": True,
        "status": "READY",
        "reason_code": None,
        "iv_observations": len(history),
        "required_iv_observations": MINIMUM_IV_OBSERVATIONS,
        "hv_observations": min(len(ohlcv_rows), 21),
    }


def _iv_percentile(current_iv: float, history: list[float]) -> float | None:
    if len(history) < MINIMUM_IV_OBSERVATIONS:
        return None
    clean = history[-252:]
    low = min(clean)
    high = max(clean)
    if high - low <= 1e-12:
        return None
    return (current_iv - low) / (high - low) * 100.0


def _unavailable_result(reason: str, observations: int) -> dict[str, Any]:
    """Explicit unavailable-IV-regime state. `clock_label` is deliberately a
    reason code outside VRC_SCORE's keys (compute/bif/composite.py) so a
    missing/insufficient-history regime is never mistaken for a real
    volatility-clock reading (e.g. it must never resolve to "COMPRESSION",
    which is also a genuine regime label) by any composite-score consumer.
    """
    return {
        "clock_position": None,
        "clock_label": reason,
        "iv_percentile": None,
        "iv_trend": None,
        "iv_acceleration": None,
        "vrp": None,
        "vrp_signal": None,
        "optimal_strategy": "Wait for cleaner volatility data",
        "expected_iv_7d": None,
        "is_valid": False,
        "status": "INSUFFICIENT_DATA" if reason == "INSUFFICIENT_HISTORY" else "UNAVAILABLE",
        "reason_code": reason,
        "iv_observations": observations,
        "required_iv_observations": MINIMUM_IV_OBSERVATIONS,
        "hv_observations": 0,
    }


def _slope(values: list[float]) -> float:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = np.asarray(clean, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _trend_label(slope: float) -> str:
    if slope > 0.0005:
        return "RISING"
    if slope < -0.0005:
        return "FALLING"
    return "FLAT"


def _clock_position(iv_percentile: float, trend: str) -> tuple[str, int]:
    if iv_percentile < 25 and trend != "RISING":
        return "COMPRESSION", 0
    if iv_percentile < 25 and trend == "RISING":
        return "EARLY EXPANSION", 45
    if 25 <= iv_percentile <= 60 and trend == "RISING":
        return "EXPANSION", 90
    if iv_percentile > 75 and trend == "RISING":
        return "LATE EXPANSION", 135
    if iv_percentile > 75 and trend != "RISING":
        return "PEAK", 180
    if iv_percentile > 50 and trend == "FALLING":
        return "CONTRACTION", 270
    if iv_percentile < 50 and trend == "FALLING":
        return "LATE CONTRACTION", 315
    return "COMPRESSION", 0


def _historical_vol_20(ohlcv_rows: list[dict[str, Any]]) -> float | None:
    closes = [_finite(row.get("close")) for row in ohlcv_rows]
    closes = [value for value in closes if value is not None and value > 0]
    if len(closes) < 21:
        return None
    returns = []
    for index in range(1, len(closes)):
        if closes[index - 1] > 0:
            returns.append(closes[index] / closes[index - 1] - 1.0)
    tail = returns[-20:]
    if len(tail) < 10:
        return None
    mean = sum(tail) / len(tail)
    variance = sum((item - mean) ** 2 for item in tail) / len(tail)
    return math.sqrt(variance) * math.sqrt(252.0)


def _vrp_signal(vrp: float) -> str:
    if vrp > 0.5:
        return "SELL_VOL"
    if vrp < -0.5:
        return "BUY_VOL"
    return "NEUTRAL"


def _optimal_strategy(clock_label: str, vrp: float | None) -> str:
    if clock_label == "COMPRESSION":
        return "Consider long straddle"
    if clock_label in {"EARLY EXPANSION", "EXPANSION"}:
        return "Consider directional long"
    if clock_label in {"PEAK", "CONTRACTION"}:
        return "Consider short volatility"
    if clock_label == "LATE EXPANSION":
        return "Reduce long volatility"
    if vrp is not None and vrp > 0.5:
        return "Consider premium collection"
    return "Wait for vol edge"


def _normalise_iv(value: Any) -> float | None:
    number = _finite(value)
    if number is None or number <= 0:
        return None
    if number > 3.0:
        number /= 100.0
    if number <= 0 or number > 5.0:
        return None
    return number


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
