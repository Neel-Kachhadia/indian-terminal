from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext, current_value, iv_reversion_target
from ml.monitor import accuracy_for_model

from .common import predict_classifier_artifact, train_classifier_artifact, untrained_output


MODEL_ID = "m04_iv_reversion"
CLASSES = ["EXPANDING", "CONTRACTING", "STABLE"]
MINIMUM_ROWS = 100


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    artifact, reason, available_rows = _train(context, force=force)
    if artifact is None:
        return untrained_output(MODEL_ID, reason or "training failed", available_rows=available_rows, minimum_rows=MINIMUM_ROWS)
    return {"model_id": MODEL_ID, "is_trained": True, "train_rows": artifact["train_rows"]}


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    artifact, reason, available_rows = _train(context)
    if artifact is None:
        fallback = _heuristic_iv_direction(context)
        return untrained_output(
            MODEL_ID,
            reason or "model not trained",
            available_rows=available_rows,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "iv_direction": fallback,
                "probabilities": _fallback_probabilities(fallback),
                "confidence": 0.0,
                "feature_importance": {},
                "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
            },
        )
    output = predict_classifier_artifact(artifact, context, confidence_multiplier=confidence_multiplier)
    output.update(
        {
            "model_id": MODEL_ID,
            "iv_direction": output["prediction"],
            "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
        }
    )
    return output


def _train(context: MLContext, force: bool = False) -> tuple[dict[str, Any] | None, str | None, int]:
    if context.iv_history.empty:
        return None, "iv_history empty; cannot build 3-day IV target", 0
    return train_classifier_artifact(
        context,
        model_id=MODEL_ID,
        classes=CLASSES,
        target_builder=iv_reversion_target,
        estimator_factory=_estimator,
        minimum_rows=MINIMUM_ROWS,
        n_splits=5,
        gap=3,
        calibration_method="sigmoid",
        force=force,
    )


def _estimator() -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.04,
        num_leaves=21,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


def _heuristic_iv_direction(context: MLContext) -> str | None:
    iv_rank = current_value(context, ["iv_rank", "options.iv_rank"])
    if iv_rank is None:
        return None
    if iv_rank > 75:
        return "CONTRACTING"
    if iv_rank < 25:
        return "EXPANDING"
    return "STABLE"


def _fallback_probabilities(direction: str | None) -> dict[str, float]:
    if direction is None:
        return {}
    return {label: (0.58 if label == direction else 0.21) for label in CLASSES}
