from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import close_series, return_series, train_unavailable, untrained_prediction


MODEL_ID = "m13_nbeats"
MINIMUM_ROWS = 80


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "neural basis training not run; interpretable basis fallback active",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    close = close_series(context)
    if len(close) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient univariate rows: {len(close)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "nbeats_trend_1d": None,
                "nbeats_seasonal_1d": None,
                "nbeats_total_1d": None,
                "trend_component": [],
                "seasonality_component": [],
                "agrees_with_prophet": False,
            },
        )

    window = close.tail(60).to_numpy(dtype=float)
    x = np.linspace(-1.0, 1.0, len(window))
    trend_coef = np.polyfit(x, window, 3)
    trend_component = np.polyval(trend_coef, x)
    detrended = window - trend_component
    phase = np.linspace(0.0, 2.0 * np.pi, len(window))
    seasonal_component = np.zeros_like(window)
    for harmonic in range(1, 6):
        seasonal_component += (
            np.sin(phase * harmonic) * np.mean(detrended * np.sin(phase * harmonic))
            + np.cos(phase * harmonic) * np.mean(detrended * np.cos(phase * harmonic))
        ) / harmonic
    trend_next = float(np.polyval(trend_coef, 1.0 + 2.0 / len(window)))
    seasonal_next = float(seasonal_component[-5:].mean())
    total_next = trend_next + seasonal_next
    last = float(window[-1])
    trend_return = (trend_next / last - 1.0) * 100.0
    seasonal_return = seasonal_next / last * 100.0
    total_return = (total_next / last - 1.0) * 100.0
    recent_return = float(return_series(context).tail(5).mean())
    agrees = (total_return >= 0 and recent_return >= 0) or (total_return < 0 and recent_return < 0)
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "model_used": "basis_decomposition_fallback",
        "prediction": "UP" if total_return > 0.15 else "DOWN" if total_return < -0.15 else "FLAT",
        "nbeats_trend_1d": round(float(trend_return), 6),
        "nbeats_seasonal_1d": round(float(seasonal_return), 6),
        "nbeats_total_1d": round(float(total_return), 6),
        "trend_component": [round(float(value), 6) for value in trend_component[-20:]],
        "seasonality_component": [round(float(value), 6) for value in seasonal_component[-20:]],
        "agrees_with_prophet": bool(agrees),
        "confidence": 0.55 if agrees else 0.38,
        "reason": "Pure basis fallback, no neural training artifact",
    }
