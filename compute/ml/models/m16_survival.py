from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import latest_return, return_series, top_current_features, train_unavailable, untrained_prediction


MODEL_ID = "m16_survival"
MINIMUM_ROWS = 120


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "Cox model not fitted in live path; empirical hazard fallback active",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    returns = return_series(context)
    if len(returns) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient reversal history: {len(returns)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "survival_probability_5d": None,
                "expected_duration_days": None,
                "hazard_ratio": None,
                "most_dangerous_signals": [],
            },
        )

    signs = np.sign(returns.to_numpy(dtype=float))
    reversals = np.where(np.diff(signs) != 0)[0] + 1
    durations = np.diff(np.r_[0, reversals, len(signs)])
    current_duration = int(durations[-1])
    avg_duration = float(np.mean(durations[:-1])) if len(durations) > 1 else 5.0
    hazard = 1.0 / max(avg_duration, 1.0)
    recent_move = abs(latest_return(context, 5) or 0.0)
    hazard_ratio = min(5.0, max(0.2, (current_duration / max(avg_duration, 1.0)) * (1.0 + recent_move / 5.0)))
    survival_5d = float(np.exp(-hazard * hazard_ratio * 5.0))
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "model_used": "empirical_hazard_fallback",
        "prediction": "TREND_SURVIVES" if survival_5d >= 0.5 else "REVERSAL_RISK",
        "survival_probability_5d": round(survival_5d, 6),
        "expected_duration_days": round(avg_duration, 3),
        "hazard_ratio": round(float(hazard_ratio), 6),
        "most_dangerous_signals": list(top_current_features(context, 5).keys()),
        "confidence": round(abs(survival_5d - 0.5) * 2.0, 6),
        "reason": "lifelines optional; empirical trend-duration hazard used",
    }
