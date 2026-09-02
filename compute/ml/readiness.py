from __future__ import annotations

import importlib
import math
import pkgutil
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

import ml.models
from ml.models.common import load_latest_artifact, safe_symbol
from ml.registry import latest_validated_record, resolve_verified_path
from ml.models.common import SAVED_MODELS_DIR


def discover_model_modules() -> dict[str, ModuleType]:
    """Discover the canonical model catalog from source, not a copied list."""
    discovered: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(ml.models.__path__):
        if not info.name.startswith("m") or not info.name[1:3].isdigit():
            continue
        module = importlib.import_module(f"ml.models.{info.name}")
        model_id = str(getattr(module, "MODEL_ID", "")).strip()
        if model_id:
            discovered[model_id] = module
    return dict(sorted(discovered.items()))


def build_readiness_matrix(
    context: Any,
    outputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    vector = context.current_vector.replace([np.inf, -np.inf], np.nan)
    actual_features = [str(column) for column, value in vector.items() if _finite(value) is not None]
    feature_date = _latest_date(getattr(context, "feature_frame", pd.DataFrame()))
    ohlcv_date = _latest_date(getattr(context, "ohlcv", pd.DataFrame()))
    matrix: list[dict[str, Any]] = []
    for model_id, module in discover_model_modules().items():
        output = outputs.get(model_id) or {}
        record = latest_validated_record(model_id, safe_symbol(context.symbol))
        path = resolve_verified_path(record, SAVED_MODELS_DIR) if record else None
        artifact = (
            load_latest_artifact(model_id, context.symbol, allow_proxy=False)
            if path is not None
            else None
        )
        expected = [str(item) for item in (artifact or {}).get("feature_columns", [])]
        missing = [column for column in expected if _finite(vector.get(column)) is None]
        dataset = str(getattr(module, "INPUT_DATASET", "feature_frame"))
        available_rows = _available_rows(context, output, dataset)
        required = _minimum_rows(module)
        eligible_rows = _int_or_none(output.get("available_rows"))
        if eligible_rows is None:
            eligible_rows = available_rows
        validation_metric = (
            {
                "name": record.get("validation_metric_name"),
                "value": record.get("validation_metric_value"),
            }
            if record
            else None
        )
        confidence = _confidence(output)
        status = _status(output, artifact, required, eligible_rows)
        current_eligible = _current_inference_eligible(output, model_id)
        matrix.append(
            {
                "model": model_id,
                "artifact_exists": record is not None and path is not None,
                "artifact_hash_valid": path is not None,
                "schema_valid": _schema_valid(record, artifact),
                "model_loaded": artifact is not None,
                "required_inputs": list(getattr(module, "REQUIRED_INPUTS", (dataset,))),
                "input_rows_available": available_rows,
                "required_lookback": required,
                "eligible_rows": eligible_rows,
                "latest_eligible_date": ohlcv_date if dataset == "ohlcv" else feature_date,
                "expected_feature_count": len(expected) if expected else None,
                "actual_feature_count": len(actual_features),
                "missing_features": missing,
                "trained": bool(artifact is not None),
                "last_trained": record.get("training_timestamp") if record else None,
                "validated": bool(record and record.get("validation_status") == "VALIDATED"),
                "validation_sample_count": (
                    int(record["validation_sample_count"]) if record else None
                ),
                "out_of_sample_metric": validation_metric,
                "current_inference_eligible": current_eligible,
                "current_confidence": confidence,
                "confidence_formula": output.get("confidence_formula"),
                "gates_applied": _gates(output),
                "status": status,
                "reason_code": output.get("reason_code"),
                "reason": output.get("reason"),
            }
        )
    return matrix


def _available_rows(context: Any, output: dict[str, Any], dataset: str) -> int:
    explicit = _int_or_none(output.get("available_rows"))
    if explicit is not None:
        return explicit
    if dataset == "iv_history":
        return int(len(getattr(context, "iv_history", [])))
    if dataset == "ohlcv":
        return int(len(getattr(context, "ohlcv", [])))
    if dataset == "fo_eod":
        return int(getattr(context, "fo_eod_rows", 0))
    return int(getattr(context, "available_rows", 0))


def _minimum_rows(module: ModuleType) -> int | None:
    for name in ("MINIMUM_ROWS", "MINIMUM_REFERENCE_ROWS", "MIN_ROWS"):
        value = _int_or_none(getattr(module, name, None))
        if value is not None:
            return value
    return None


def _status(
    output: dict[str, Any],
    artifact: dict[str, Any] | None,
    required: int | None,
    eligible_rows: int,
) -> str:
    explicit = str(output.get("status") or "").strip().upper()
    if explicit:
        return explicit
    if artifact is not None and _current_inference_eligible(output, str(output.get("model_id") or "")):
        return "READY"
    if required is not None and eligible_rows < required:
        return "INSUFFICIENT_DATA"
    return "UNTRAINED" if artifact is None else "UNAVAILABLE"


def _current_inference_eligible(output: dict[str, Any], model_id: str) -> bool:
    if output.get("is_valid") is False:
        return False
    if model_id == "m21_ohlc":
        return output.get("forecast_close") is not None and output.get("is_trained") is True
    prediction = output.get("prediction")
    if prediction is None:
        prediction = output.get("composite_direction")
    if prediction is None:
        return False
    if output.get("is_trained") is True:
        return output.get("feature_schema_compatible") is not False
    return output.get("is_valid") is True


def _schema_valid(
    record: dict[str, Any] | None,
    artifact: dict[str, Any] | None,
) -> bool | None:
    if record is None or artifact is None:
        return None
    return str(record.get("feature_schema_version")) == str(
        artifact.get("feature_schema_version")
    )


def _confidence(output: dict[str, Any]) -> float | str | None:
    value = output.get("confidence")
    if value is None:
        value = output.get("composite_confidence")
    if isinstance(value, str):
        return value
    return _finite(value)


def _gates(output: dict[str, Any]) -> list[str]:
    gates: list[str] = []
    if output.get("feature_schema_compatible") is False:
        gates.append("FEATURE_SCHEMA_INCOMPATIBLE")
    if output.get("computed_with_confidence_multiplier") not in (None, 1, 1.0):
        gates.append("ANOMALY_CONFIDENCE_MULTIPLIER")
    if output.get("is_trained") is not True:
        gates.append("ARTIFACT_NOT_TRAINED")
    if output.get("calibrated") is False:
        gates.append("CALIBRATION_UNAVAILABLE")
    return gates


def _latest_date(frame: pd.DataFrame) -> str | None:
    if frame.empty or "date" not in frame.columns:
        return None
    value = pd.to_datetime(frame["date"], errors="coerce").max()
    return value.date().isoformat() if pd.notna(value) else None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
