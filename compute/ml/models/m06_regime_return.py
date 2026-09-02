from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext, direction_target
from ml.monitor import accuracy_for_model

from .common import predict_classifier_artifact, train_classifier_artifact, untrained_output


MODEL_ID = "m06_regime_return"
CLASSES = ["UP", "FLAT", "DOWN"]
MINIMUM_ROWS = 200


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    artifact, reason, available_rows = _train(context, force=force)
    if artifact is None:
        return untrained_output(MODEL_ID, reason or "training failed", available_rows=available_rows, minimum_rows=MINIMUM_ROWS)
    return {"model_id": MODEL_ID, "is_trained": True, "train_rows": artifact["train_rows"]}


def predict(
    context: MLContext,
    confidence_multiplier: float = 1.0,
    regime_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact, reason, available_rows = _train(context, train_if_missing=False)
    weights = _regime_weights(regime_output)
    active = max(weights, key=weights.get) if weights else None
    if artifact is None:
        return untrained_output(
            MODEL_ID,
            reason or "model not trained",
            available_rows=available_rows,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "direction": None,
                "active_regime_model": active,
                "regime_model_weights": weights,
                "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
                "accuracy_by_regime": {"LOW": None, "MEDIUM": None, "HIGH": None},
            },
        )
    output = predict_classifier_artifact(artifact, context, confidence_multiplier=confidence_multiplier)
    output.update(
        {
            "model_id": MODEL_ID,
            "direction": output["prediction"],
            "active_regime_model": active,
            "regime_model_weights": weights,
            "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
            "accuracy_by_regime": {"LOW": None, "MEDIUM": None, "HIGH": None},
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
        # _train() is the shared wrapper for both train_model()
        # (train_if_missing=True) and predict() (train_if_missing=False), so
        # this can't be a flat False the way m01_vol_regime's separate
        # train_model()-only call site uses -- that would also kill this
        # model's intended proxy fallback at inference time. Without this at
        # all (the prior state), train_classifier_artifact's default
        # allow_proxy=True let its own "already trained?" pre-check
        # (`load_latest_artifact(model_id, context.symbol, allow_proxy=...)`)
        # silently accept PROXY_SYMBOL's (NIFTY's) artifact as "nothing to
        # do" during REAL training for every other symbol -- is_trained=True
        # was then reported with NIFTY's train_rows/artifact, no
        # fallback_status disclosure, and no artifact was ever registered
        # for the requested symbol. Reproduced live: the moment NIFTY got
        # its first real m06 artifact, the very next symbol's (FINNIFTY's)
        # "training" silently no-opped this way. allow_proxy=not
        # train_if_missing mirrors m02_direction's identical shared-wrapper
        # fix: proxy allowed only at inference (train_if_missing=False),
        # never as a substitute for real per-symbol training.
        allow_proxy=not train_if_missing,
    )


def _estimator() -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=120,
        depth=4,
        learning_rate=0.04,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _regime_weights(regime_output: dict[str, Any] | None) -> dict[str, float]:
    probs = (regime_output or {}).get("probabilities")
    if isinstance(probs, dict) and probs:
        return {key: float(probs.get(key, 0.0)) for key in ["LOW", "MEDIUM", "HIGH"]}
    return {}
