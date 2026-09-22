from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext, direction_target
from ml.monitor import accuracy_for_model

from .common import predict_classifier_artifact, train_classifier_artifact, untrained_output


MODEL_ID = "m02_direction"
CLASSES = ["UP", "FLAT", "DOWN"]
MINIMUM_ROWS = 200


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    artifact, reason, available_rows = _train(context, force=force)
    if artifact is None:
        return untrained_output(MODEL_ID, reason or "training failed", available_rows=available_rows, minimum_rows=MINIMUM_ROWS)
    return {
        "model_id": MODEL_ID,
        "is_trained": True,
        "train_rows": artifact["train_rows"],
        "n_features": artifact.get("n_features"),
        "train_score": artifact.get("train_score"),
        "val_score": artifact.get("val_score"),
        "feature_importance": artifact.get("feature_importance", {}),
        "feature_importance_top5": artifact.get("feature_importance_top5", {}),
        "bif_rows_used": artifact.get("bif_rows_used"),
        "ohlcv_rows_used": artifact.get("ohlcv_rows_used"),
    }


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    artifact, reason, available_rows = _train(context, train_if_missing=False)
    if artifact is None:
        return untrained_output(
            MODEL_ID,
            reason or "model not trained",
            available_rows=available_rows,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "direction": None,
                "probabilities": {},
                "top_features": [],
                "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
                "accuracy_by_confidence": {
                    "high_confidence_calls": None,
                    "all_calls": accuracy_for_model(MODEL_ID)["accuracy"],
                },
            },
        )
    output = predict_classifier_artifact(artifact, context, confidence_multiplier=confidence_multiplier)
    output.update(
        {
            "model_id": MODEL_ID,
            "direction": output["prediction"],
            "top_features": [
                {"name": name, "importance": importance}
                for name, importance in output.get("feature_importance", {}).items()
            ],
            "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
            "accuracy_by_confidence": {
                "high_confidence_calls": None,
                "all_calls": accuracy_for_model(MODEL_ID)["accuracy"],
            },
        }
    )
    return output


def _train(
    context: MLContext,
    force: bool = False,
    train_if_missing: bool = True,
) -> tuple[dict[str, Any] | None, str | None, int]:
    return train_classifier_artifact(
        context,
        model_id=MODEL_ID,
        classes=CLASSES,
        target_builder=direction_target,
        estimator_factory=_estimator,
        minimum_rows=MINIMUM_ROWS,
        n_splits=5,
        gap=1,
        calibration_method="isotonic",
        force=force,
        train_if_missing=train_if_missing,
        allow_proxy=not train_if_missing,
    )


def _estimator() -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
