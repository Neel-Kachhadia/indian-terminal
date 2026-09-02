from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ml.feature_engineering import MLContext


DIRECTION_CLASSES = ["UP", "FLAT", "DOWN"]


def train_unavailable(
    model_id: str,
    reason: str,
    context: MLContext,
    *,
    minimum_rows: int = 0,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "is_trained": False,
        "reason": reason,
        "status": "UNTRAINED",
        "reason_code": "TRAINING_PIPELINE_NOT_IMPLEMENTED",
        "available_rows": int(getattr(context, "available_rows", 0) or 0),
        "minimum_rows": minimum_rows,
    }


def untrained_prediction(
    model_id: str,
    reason: str,
    context: MLContext,
    *,
    minimum_rows: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_id": model_id,
        "is_trained": False,
        "prediction": None,
        "probabilities": {},
        "confidence": None,
        "reason": reason,
        "status": "INSUFFICIENT_DATA",
        "reason_code": "INSUFFICIENT_ELIGIBLE_HISTORY",
        "available_rows": int(getattr(context, "available_rows", 0) or 0),
        "minimum_rows": minimum_rows,
    }
    if extra:
        payload.update(extra)
    return payload


def close_series(context: MLContext) -> pd.Series:
    frame = context.ohlcv
    if frame.empty or "close" not in frame.columns:
        return pd.Series(dtype="float64")
    dates = pd.to_datetime(frame["date"]).dt.date
    values = pd.to_numeric(frame["close"], errors="coerce")
    return pd.Series(values.to_numpy(dtype=float), index=dates).dropna()


def return_series(context: MLContext) -> pd.Series:
    close = close_series(context)
    if close.empty:
        return pd.Series(dtype="float64")
    return close.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100.0


def latest_return(context: MLContext, periods: int = 1) -> float | None:
    close = close_series(context)
    if len(close) <= periods:
        return None
    value = (close.iloc[-1] / close.iloc[-periods - 1] - 1.0) * 100.0
    return finite_float(value)


def realised_vol_pct(context: MLContext, window: int = 20) -> float | None:
    returns = return_series(context).tail(window)
    if len(returns) < max(5, window // 3):
        return None
    return finite_float(float(returns.std(ddof=0)) * math.sqrt(252.0))


def direction_probabilities(
    expected_return_pct: float | None,
    uncertainty_pct: float | None,
    *,
    confidence_multiplier: float = 1.0,
) -> dict[str, float]:
    if expected_return_pct is None or uncertainty_pct is None:
        return {}
    scale = max(float(uncertainty_pct), 0.25)
    strength = max(-1.0, min(1.0, expected_return_pct / (scale * 1.5)))
    flat = max(0.18, 0.48 - abs(strength) * 0.28)
    directional = 1.0 - flat
    if strength >= 0:
        up = directional * (0.5 + 0.5 * abs(strength))
        down = directional - up
    else:
        down = directional * (0.5 + 0.5 * abs(strength))
        up = directional - down
    probs = normalise({"UP": up, "FLAT": flat, "DOWN": down})
    if confidence_multiplier < 1.0:
        probs = dampen_probabilities(probs, confidence_multiplier)
    return probs


def confidence(probabilities: dict[str, float]) -> float | None:
    if not probabilities:
        return None
    return round(max(probabilities.values()), 6)


def normalise(probabilities: dict[str, float]) -> dict[str, float]:
    clean = {
        key: max(0.0, float(value))
        for key, value in probabilities.items()
        if math.isfinite(float(value))
    }
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {key: round(value / total, 6) for key, value in clean.items()}


def dampen_probabilities(
    probabilities: dict[str, float],
    confidence_multiplier: float,
) -> dict[str, float]:
    if not probabilities:
        return probabilities
    labels = list(probabilities)
    base = 1.0 / len(labels)
    multiplier = max(0.0, min(1.0, confidence_multiplier))
    return normalise({
        label: base + (float(probabilities[label]) - base) * multiplier
        for label in labels
    })


def top_current_features(context: MLContext, n: int = 8) -> dict[str, float]:
    items: list[tuple[str, float]] = []
    for name, value in context.current_features.items():
        number = finite_float(value)
        if number is None:
            continue
        items.append((name, abs(number)))
    items.sort(key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in items[:n]) or 1.0
    return {name: round(value / total, 6) for name, value in items[:n]}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
