from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext

from .apex_common import return_series, train_unavailable, untrained_prediction


MODEL_ID = "m20_bnn"
MINIMUM_ROWS = 80


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "Pyro/MC-dropout network not trained; bootstrap uncertainty fallback active",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(
    context: MLContext,
    confidence_multiplier: float = 1.0,
    component_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del confidence_multiplier
    returns = return_series(context)
    if len(returns) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient rows for uncertainty bootstrap: {len(returns)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "direction_mean": None,
                "direction_std": None,
                "epistemic_uncertainty": None,
                "aleatoric_uncertainty": None,
                "mc_samples": 0,
                "prediction_stable": False,
            },
        )

    historical = returns.tail(252).to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    samples = rng.choice(historical, size=100, replace=True)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=0))
    aleatoric = float(np.std(historical[-20:], ddof=0))
    component_outputs = component_outputs or {}
    model_disagreement = _component_disagreement(component_outputs)
    epistemic = float(min(1.0, std / 3.0 + (model_disagreement or 0.0)))
    direction = "UP" if mean > 0.15 else "DOWN" if mean < -0.15 else "FLAT"
    stable = epistemic < 0.35 and aleatoric < 1.2
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "READY_ANALYTICAL",
        "reason_code": None,
        "model_used": "bootstrap_mc_dropout_fallback",
        "prediction": direction,
        "direction_mean": round(mean, 6),
        "direction_std": round(std, 6),
        "epistemic_uncertainty": round(epistemic, 6),
        "aleatoric_uncertainty": round(aleatoric, 6),
        "mc_samples": 100,
        "prediction_stable": bool(stable),
        "confidence": None,
        "confidence_formula": None,
        "component_disagreement": (
            round(model_disagreement, 6) if model_disagreement is not None else None
        ),
        "reason": "Bayesian network optional; historical bootstrap uncertainty used",
    }


def _component_disagreement(component_outputs: dict[str, Any]) -> float | None:
    directions = []
    for key in ("m02_direction", "m06_regime_return", "m11_tft", "m14_wavelet"):
        output = component_outputs.get(key) or {}
        value = output.get("direction") or output.get("prediction")
        if value in {"UP", "FLAT", "DOWN"}:
            directions.append(value)
    if len(directions) < 2:
        return None
    top_count = max(directions.count(item) for item in set(directions))
    return 1.0 - top_count / len(directions)
