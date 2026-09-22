from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext, vol_regime_target
from ml.monitor import accuracy_for_model

from .common import (
    predict_classifier_artifact,
    train_classifier_artifact,
    untrained_output,
)


MODEL_ID = "m01_vol_regime"
CLASSES = ["LOW", "MEDIUM", "HIGH"]
MINIMUM_ROWS = 200


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    artifact, reason, available_rows = train_classifier_artifact(
        context,
        model_id=MODEL_ID,
        classes=CLASSES,
        target_builder=vol_regime_target,
        estimator_factory=_estimator,
        minimum_rows=MINIMUM_ROWS,
        n_splits=5,
        gap=5,
        calibration_method="isotonic",
        force=force,
        allow_proxy=False,
    )
    if artifact is None:
        return untrained_output(
            MODEL_ID,
            reason or "training failed",
            available_rows=available_rows,
            minimum_rows=MINIMUM_ROWS,
            extra={"regime": None, "out_of_regime_warning": False},
        )
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
    artifact, reason, available_rows = train_classifier_artifact(
        context,
        model_id=MODEL_ID,
        classes=CLASSES,
        target_builder=vol_regime_target,
        estimator_factory=_estimator,
        minimum_rows=MINIMUM_ROWS,
        n_splits=5,
        gap=5,
        calibration_method="isotonic",
        train_if_missing=False,
    )
    if artifact is None:
        return untrained_output(
            MODEL_ID,
            reason or "model not trained",
            available_rows=available_rows,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "regime": None,
                "probabilities": {},
                "feature_importance": {},
                "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
                "out_of_regime_warning": context.available_rows < MINIMUM_ROWS,
            },
        )

    output = predict_classifier_artifact(
        artifact,
        context,
        confidence_multiplier=confidence_multiplier,
    )
    output.update(
        {
            "model_id": MODEL_ID,
            "regime": output["prediction"],
            "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
            "out_of_regime_warning": _out_of_regime_warning(context),
        }
    )
    return output


def _estimator() -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )


def _out_of_regime_warning(context: MLContext) -> bool:
    vector = context.current_vector
    if vector.empty or context.feature_frame.empty:
        return False
    numeric = context.feature_frame.drop(columns=["date"], errors="ignore")
    numeric = numeric.select_dtypes(include=["number"])
    if numeric.empty:
        return False
    med = numeric.median(numeric_only=True)
    std = numeric.std(numeric_only=True, ddof=0).replace(0.0, 1.0)
    aligned = vector.reindex(numeric.columns).fillna(med)
    z = ((aligned - med) / std).abs()
    return bool((z > 4.0).sum() >= 3)
