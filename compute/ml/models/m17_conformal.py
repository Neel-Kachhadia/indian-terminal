from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import return_series, train_unavailable, untrained_prediction


MODEL_ID = "m17_conformal"
MINIMUM_ROWS = 80


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "MAPIE wrapper not fitted; rolling residual conformal fallback active",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(
    context: MLContext,
    center_return_pct: float | None = None,
    confidence_multiplier: float = 1.0,
) -> dict[str, Any]:
    del confidence_multiplier
    returns = return_series(context)
    if len(returns) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient residual history: {len(returns)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "interval_80": [None, None],
                "interval_90": [None, None],
                "interval_95": [None, None],
                "coverage_achieved": None,
                "interval_width": None,
            },
        )

    historical = returns.tail(252)
    naive = historical.shift(1).rolling(5, min_periods=1).mean()
    residuals = (historical - naive).abs().dropna().to_numpy(dtype=float)
    center = float(center_return_pct if center_return_pct is not None else historical.tail(5).mean())
    q80, q90, q95 = np.quantile(residuals, [0.80, 0.90, 0.95])
    interval90 = [center - float(q90), center + float(q90)]
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "model_used": "rolling_conformal_fallback",
        "prediction": interval90,
        "interval_80": [round(center - float(q80), 6), round(center + float(q80), 6)],
        "interval_90": [round(interval90[0], 6), round(interval90[1], 6)],
        "interval_95": [round(center - float(q95), 6), round(center + float(q95), 6)],
        "coverage_achieved": 0.90,
        "interval_width": round(float(interval90[1] - interval90[0]), 6),
        "confidence": round(max(0.0, 1.0 - float(q90) / 4.0), 6),
        "reason": "Distribution-free rolling residual interval; no MAPIE artifact",
    }
