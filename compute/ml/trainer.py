from __future__ import annotations

import datetime as dt
from typing import Any

from ml.feature_engineering import build_context
from ml.monitor import accuracy_for_model, accuracy_report
from ml.models.common import SAVED_MODELS_DIR, model_status
from ml.readiness import discover_model_modules
from ml.training_log import log_training_result


def training_modules() -> list[tuple[str, Any]]:
    return [
        (model_id, module)
        for model_id, module in discover_model_modules().items()
        if callable(getattr(module, "train_model", None))
    ]


# Backward-compatible exported catalog; its contents are discovered from the
# source package at import time rather than maintained as a second hardcoded list.
MODEL_MODULES = training_modules()


def train_all_models(force: bool = False, symbol: str = "NIFTY") -> dict[str, Any]:
    context = build_context(symbol)
    report: dict[str, Any] = {
        "symbol": context.symbol,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "force": force,
        "feature_rows": context.available_rows,
        "models": {},
    }
    for model_id, module in training_modules():
        try:
            result = module.train_model(context, force=force)
        except Exception as error:
            result = {
                "model_id": model_id,
                "is_trained": False,
                "reason": str(error),
            }
        artifact_status = model_status(model_id, context.symbol)
        result["artifact_ready"] = bool(artifact_status.get("is_trained"))
        if result["artifact_ready"]:
            result["status"] = "READY"
            result["reason_code"] = None
        elif not result.get("status"):
            reason = str(result.get("reason") or "")
            result["status"] = (
                "INSUFFICIENT_DATA"
                if "insufficient" in reason.lower() or "empty" in reason.lower()
                else "FAILED"
            )
            result["reason_code"] = (
                "INSUFFICIENT_ELIGIBLE_HISTORY"
                if result["status"] == "INSUFFICIENT_DATA"
                else "TRAINING_DID_NOT_PRODUCE_ARTIFACT"
            )
        report["models"][model_id] = result
        log_training_result(context.symbol, model_id, result, context=context)
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return report


def load_all_models() -> dict[str, Any]:
    return {
        model_id: model_status(model_id, "NIFTY")
        for model_id in discover_model_modules()
    }


def evaluate_model(model_id: str, n_days: int = 30) -> dict[str, Any]:
    return accuracy_for_model(model_id, n_days)


def status_report() -> dict[str, Any]:
    return {
        "saved_models_dir": str(SAVED_MODELS_DIR),
        "models": load_all_models(),
        "accuracy": accuracy_report(30),
    }
