from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import return_series, train_unavailable, untrained_prediction


MODEL_ID = "m15_kalman"
MINIMUM_ROWS = 40


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "state-space parameters estimated online; no offline training artifact",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    returns = return_series(context).tail(120)
    if len(returns) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient rows for state estimate: {len(returns)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "dynamic_beta_dow": None,
                "dynamic_beta_usdinr": None,
                "dynamic_beta_brent": None,
                "alpha_today": None,
                "signal_noise_ratio": None,
                "kalman_state": None,
                "state_uncertainty": None,
                "smoothed_signals": {},
            },
        )

    values = returns.to_numpy(dtype=float)
    smoothed = _ewma(values, alpha=0.18)
    residual = values - smoothed
    signal_std = float(np.std(smoothed[-20:]) or 0.0)
    noise_std = float(np.std(residual[-20:]) or 1e-6)
    snr = signal_std / noise_std
    vol = float(np.std(values[-20:]))
    state = 2 if vol > 1.6 else 1 if vol > 0.9 else 0
    alpha_today = float(values[-1] - smoothed[-1])
    uncertainty = min(1.0, noise_std / max(vol + noise_std, 1e-6))
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "READY_ANALYTICAL",
        "reason_code": None,
        "model_used": "ewma_state_space_fallback",
        "prediction": state,
        "dynamic_beta_dow": None,
        "dynamic_beta_usdinr": None,
        "dynamic_beta_brent": None,
        "alpha_today": round(alpha_today, 6),
        "signal_noise_ratio": round(float(snr), 6),
        "kalman_state": int(state),
        "state_uncertainty": round(float(uncertainty), 6),
        "smoothed_signals": {
            "return_ewma_1d": round(float(smoothed[-1]), 6),
            "return_ewma_5d": round(float(np.mean(smoothed[-5:])), 6),
        },
        "confidence": None,
        "confidence_formula": None,
        "reason": "filterpy optional; EWMA state estimator active",
    }


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.zeros_like(values, dtype=float)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output
