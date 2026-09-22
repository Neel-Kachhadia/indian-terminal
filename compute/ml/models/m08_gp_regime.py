from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext


MODEL_ID = "m08_gp_regime"


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": not context.current_vector.empty,
        "status": "READY_ANALYTICAL" if not context.current_vector.empty else "UNAVAILABLE",
        "reason_code": None if not context.current_vector.empty else "CURRENT_FEATURES_UNAVAILABLE",
        "train_rows": context.available_rows,
    }


def predict(
    context: MLContext,
    confidence_multiplier: float = 1.0,
    component_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del confidence_multiplier
    quality = _data_quality_score(context)
    uncertainty = _regime_uncertainty(component_outputs)
    available = not context.current_vector.empty
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": available,
        "status": "READY_ANALYTICAL" if available else "UNAVAILABLE",
        "reason_code": None if available else "CURRENT_FEATURES_UNAVAILABLE",
        "prediction": (
            "NORMAL" if quality >= 50 and uncertainty < 0.45 else "UNCERTAIN"
        ) if available else None,
        "iv_surface_gp": {
            "mean": [],
            "std": [],
            "quality": None,
            "status": "UNAVAILABLE",
            "reason_code": "IV_SURFACE_NOT_FITTED",
        },
        "regime_uncertainty": round(uncertainty, 6),
        "data_quality_score": round(quality, 3) if available else None,
        "reason": "data-quality diagnostic; no Gaussian-process artifact is registered",
        "confidence": None,
        "confidence_formula": None,
    }


def _data_quality_score(context: MLContext) -> float:
    vector = context.current_vector
    if vector.empty:
        return 0.0
    total = len(vector)
    present = int(vector.notna().sum())
    row_score = min(1.0, context.available_rows / 252.0)
    return (present / max(1, total) * 70.0) + (row_score * 30.0)


def _regime_uncertainty(component_outputs: dict[str, Any] | None) -> float:
    probs = ((component_outputs or {}).get("m01_vol_regime") or {}).get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return 0.5
    return max(0.0, min(1.0, 1.0 - max(float(value) for value in probs.values())))
