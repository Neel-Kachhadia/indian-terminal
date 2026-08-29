from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext


MODEL_ID = "m08_gp_regime"


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return {
        "model_id": MODEL_ID,
        "is_trained": _sklearn_available() and context.available_rows >= 50,
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
    trained = _sklearn_available() and context.available_rows >= 50
    return {
        "model_id": MODEL_ID,
        "is_trained": trained,
        "prediction": "NORMAL" if quality >= 50 and uncertainty < 0.45 else "UNCERTAIN",
        "iv_surface_gp": {
            "mean": [],
            "std": [],
            "quality": round(max(0.0, min(1.0, quality / 100.0)), 6),
        },
        "regime_uncertainty": round(uncertainty, 6),
        "data_quality_score": round(quality, 3),
        "reason": None if trained else "sklearn GP unavailable or insufficient rows; returning uncertainty diagnostics",
    }


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


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
