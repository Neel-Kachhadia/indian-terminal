from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import (
    confidence,
    direction_probabilities,
    return_series,
    top_current_features,
    train_unavailable,
    untrained_prediction,
)


MODEL_ID = "m11_tft"
MINIMUM_ROWS = 60


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "pytorch-forecasting training not run in live path; using statistical fallback",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    returns = return_series(context).tail(MINIMUM_ROWS)
    if len(returns) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient sequence rows: {len(returns)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "return_p10_1d": None,
                "return_p50_1d": None,
                "return_p90_1d": None,
                "return_p50_3d": None,
                "return_p50_5d": None,
                "attention_top_days": [],
                "feature_importance": {},
                "prediction_interval_width": None,
                "model_used": "unavailable",
            },
        )

    recent = float(returns.tail(5).mean())
    medium = float(returns.tail(20).mean())
    long = float(returns.mean())
    sigma = float(returns.tail(20).std(ddof=0) or returns.std(ddof=0) or 0.8)
    forecast_1d = 0.50 * recent + 0.35 * medium + 0.15 * long
    forecast_3d = forecast_1d * 2.2
    forecast_5d = forecast_1d * 3.2
    p10 = forecast_1d - 1.2816 * sigma
    p90 = forecast_1d + 1.2816 * sigma
    probabilities = direction_probabilities(
        forecast_1d,
        sigma,
        confidence_multiplier=confidence_multiplier,
    )
    top_days = (
        np.argsort(np.abs(returns.to_numpy(dtype=float)))[-5:][::-1] + 1
    ).astype(int)
    direction = max(probabilities, key=probabilities.get)
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "model_used": "statistical_sequence_fallback",
        "prediction": direction,
        "direction": direction,
        "probabilities": probabilities,
        "confidence": confidence(probabilities),
        "return_p10_1d": round(float(p10), 6),
        "return_p50_1d": round(float(forecast_1d), 6),
        "return_p90_1d": round(float(p90), 6),
        "return_p50_3d": round(float(forecast_3d), 6),
        "return_p50_5d": round(float(forecast_5d), 6),
        "attention_top_days": [int(day) for day in top_days.tolist()],
        "feature_importance": top_current_features(context),
        "prediction_interval_width": round(float(p90 - p10), 6),
        "reason": "TFT dependency/training optional; fallback uses rolling sequence statistics",
    }
