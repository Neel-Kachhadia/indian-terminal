from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import confidence, direction_probabilities, return_series, train_unavailable, untrained_prediction


MODEL_ID = "m14_wavelet"
MINIMUM_ROWS = 96


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "per-scale LightGBM training not run; multi-scale fallback active",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    returns = return_series(context).tail(MINIMUM_ROWS)
    if len(returns) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient return rows: {len(returns)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "wavelet_signal_1d": None,
                "dominant_frequency": None,
                "level_predictions": {},
                "level_weights": {},
                "noise_ratio": None,
            },
        )

    values = returns.to_numpy(dtype=float)
    level_predictions = _level_predictions(values)
    weights = _level_weights(level_predictions)
    signal = sum(level_predictions[level] * weights[level] for level in level_predictions)
    sigma = float(np.std(values[-20:]) or np.std(values) or 0.8)
    probs = direction_probabilities(signal, sigma, confidence_multiplier=confidence_multiplier)
    dominant = max(weights, key=weights.get)
    noise_ratio = abs(level_predictions.get("level_1_1_2d", 0.0)) / (
        sum(abs(value) for value in level_predictions.values()) + 1e-9
    )
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "model_used": "rolling_multiscale_fallback",
        "prediction": max(probs, key=probs.get),
        "probabilities": probs,
        "confidence": confidence(probs),
        "wavelet_signal_1d": round(float(signal), 6),
        "dominant_frequency": dominant,
        "level_predictions": {key: round(float(value), 6) for key, value in level_predictions.items()},
        "level_weights": {key: round(float(value), 6) for key, value in weights.items()},
        "noise_ratio": round(float(noise_ratio), 6),
        "reason": "PyWavelets optional; fallback uses rolling scale returns",
    }


def _level_predictions(values: np.ndarray) -> dict[str, float]:
    return {
        "level_1_1_2d": float(np.mean(values[-2:])),
        "level_2_2_4d": float(np.mean(values[-4:])),
        "level_3_4_8d": float(np.mean(values[-8:])),
        "level_4_8_16d": float(np.mean(values[-16:])),
        "level_5_16_32d": float(np.mean(values[-32:])),
    }


def _level_weights(predictions: dict[str, float]) -> dict[str, float]:
    raw = {key: abs(value) + 1e-6 for key, value in predictions.items()}
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}
