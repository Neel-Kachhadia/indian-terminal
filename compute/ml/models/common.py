from __future__ import annotations

import datetime as dt
import io
import importlib.metadata
import logging
import math
import platform
import re
import uuid
import zipfile
from pathlib import Path
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable

import numpy as np
import pandas as pd
import skops.io as sio

from data.paths import DATA_DIR
from ml.calibrator import clipped_confidence, fit_calibrated_classifier, normalise_probabilities
from ml.feature_engineering import MLContext, training_frame_with_target
from ml.registry import (
    feature_schema_version,
    latest_validated_record,
    read_verified_artifact_bytes,
    register_artifact,
    resolve_verified_path,
    retire_missing_versions,
    validated_records,
)
from release_identity import load_release_identity
from operations.storage import require_free_space


SAVED_MODELS_DIR = DATA_DIR / "ml" / "saved_models"
PROXY_SYMBOL = "NIFTY"
SERIALIZATION_FORMAT = "SKOPS"
SERIALIZATION_VERSION = 1
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 20_000
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
TRUSTED_SKOPS_TYPES = frozenset(
    {
        "catboost.core.CatBoostClassifier",
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
        "sklearn.model_selection._split.TimeSeriesSplit",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
    }
)
MAX_CACHED_ARTIFACTS = 16
_ARTIFACT_CACHE: OrderedDict[str, tuple[Path, int, int, dict[str, Any]]] = OrderedDict()
_ARTIFACT_CACHE_LOCK = Lock()
_ARTIFACT_LOAD_FAILURES: set[tuple[str, str, str]] = set()
_ARTIFACT_LOAD_FAILURES_LOCK = Lock()
_RUNTIME_LOG = logging.getLogger("ml.runtime")


def untrained_output(
    model_id: str,
    reason: str,
    *,
    available_rows: int,
    minimum_rows: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status, reason_code = _unavailable_status(reason)
    output = {
        "model_id": model_id,
        "is_trained": False,
        "prediction": None,
        "probabilities": {},
        "confidence": None,
        "accuracy_30d": None,
        "available_rows": available_rows,
        "minimum_rows": minimum_rows,
        "reason": reason,
        "status": status,
        "reason_code": reason_code,
    }
    if extra:
        output.update(extra)
    return output


def _unavailable_status(reason: str) -> tuple[str, str]:
    text = reason.lower()
    if "insufficient" in text or "below minimum" in text or "empty" in text:
        return "INSUFFICIENT_DATA", "INSUFFICIENT_ELIGIBLE_HISTORY"
    if "quarantin" in text or "corrupt" in text or "hash" in text:
        return "QUARANTINED", "ARTIFACT_VERIFICATION_FAILED"
    if "stale" in text:
        return "STALE", "ARTIFACT_STALE"
    if "failed" in text or "error" in text or "dependency unavailable" in text:
        return "FAILED", "TRAINING_OR_DEPENDENCY_FAILED"
    return "UNTRAINED", "ARTIFACT_NOT_REGISTERED"


def train_classifier_artifact(
    context: MLContext,
    *,
    model_id: str,
    classes: list[str],
    target_builder: Callable[[MLContext], pd.Series],
    estimator_factory: Callable[[], Any],
    minimum_rows: int,
    n_splits: int,
    gap: int,
    calibration_method: str,
    force: bool = False,
    train_if_missing: bool = True,
    allow_proxy: bool = True,
) -> tuple[dict[str, Any] | None, str | None, int]:
    if not force:
        artifact = load_latest_artifact(model_id, context.symbol, allow_proxy=allow_proxy)
        if artifact is not None:
            return artifact, None, int(artifact.get("train_rows", 0))

    if not train_if_missing:
        return None, "model not trained for symbol", context.available_rows

    frame = training_frame_with_target(context, target_builder)
    available_rows = int(len(frame))
    if available_rows < minimum_rows:
        return None, f"insufficient data: {available_rows}/{minimum_rows} rows", available_rows

    try:
        estimator = estimator_factory()
    except Exception as error:
        return None, f"dependency unavailable: {error}", available_rows

    feature_cols = [
        col
        for col in frame.columns
        if col not in {"date", "target"} and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not feature_cols:
        return None, "no numeric features available", available_rows

    frame = frame.sort_values("date").reset_index(drop=True)
    x_frame = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_labels = frame["target"].astype(str)
    class_to_index = {label: index for index, label in enumerate(classes)}
    mask = y_labels.isin(class_to_index)
    x_frame = x_frame.loc[mask]
    y_labels = y_labels.loc[mask]
    if len(x_frame) < minimum_rows:
        return None, f"usable labelled rows below minimum: {len(x_frame)}/{minimum_rows}", len(x_frame)
    if y_labels.nunique() < 2:
        return None, "target has fewer than two classes", len(x_frame)

    filtered = frame.loc[mask].reset_index(drop=True)
    x_frame = x_frame.loc[mask].reset_index(drop=True)
    y_labels = y_labels.loc[mask].reset_index(drop=True)
    try:
        split = temporal_train_validation_test_split(filtered["date"], gap=gap)
    except ValueError as error:
        return None, str(error), len(x_frame)
    train_slice, validation_slice, test_slice = split["slices"]
    train_features = x_frame.iloc[train_slice]
    usable_columns = [column for column in feature_cols if train_features[column].notna().any()]
    excluded_features = sorted(set(feature_cols) - set(usable_columns))
    if not usable_columns:
        return None, "no finite features in chronological training split", len(x_frame)
    feature_cols = usable_columns
    x_frame = x_frame[feature_cols]
    medians = x_frame.iloc[train_slice].median(numeric_only=True).reindex(feature_cols)
    if medians.isna().any():
        missing = ", ".join(str(item) for item in medians[medians.isna()].index[:5])
        return None, f"training medians unavailable for features: {missing}", len(x_frame)
    x_frame = x_frame.fillna(medians)
    if not np.isfinite(x_frame.to_numpy(dtype=float)).all():
        return None, "non-finite values remain after training-split median imputation", len(x_frame)
    y_all = y_labels.map(class_to_index).to_numpy(dtype=int)
    x_all = x_frame.to_numpy(dtype=float)
    x_train, y_train = x_all[train_slice], y_all[train_slice]
    x_validation, y_validation = x_all[validation_slice], y_all[validation_slice]
    x_test, y_test = x_all[test_slice], y_all[test_slice]
    if len(set(y_train.tolist())) < 2:
        return None, "training split has fewer than two classes", len(x_frame)
    model, calibrated, calibration_warning = fit_calibrated_classifier(
        estimator,
        x_train,
        y_train,
        method=calibration_method,
        n_splits=n_splits,
        gap=gap,
    )
    train_score = _score(model, x_train, y_train)
    validation_score = _score(model, x_validation, y_validation)
    test_score = _score(model, x_test, y_test)
    calibration = _classification_calibration(model, x_test, y_test)
    if validation_score is None or test_score is None or calibration["brier_score"] is None:
        return None, "temporal validation metrics unavailable", len(x_frame)
    skill_floor = _classification_skill_floor(len(classes), len(y_validation))
    if validation_score < skill_floor:
        return (
            None,
            f"validation accuracy {validation_score:.4f} does not clear the "
            f"chance-baseline skill floor {skill_floor:.4f} for {len(classes)} "
            f"classes at {len(y_validation)} validation samples -- rejected as "
            "statistically indistinguishable from random",
            len(x_frame),
        )
    feature_importance = _feature_importance(model, feature_cols)
    now = dt.datetime.now(dt.timezone.utc)
    release = load_release_identity()
    model_version = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    artifact = {
        "model_id": model_id,
        "model_version": model_version,
        "symbol": safe_symbol(context.symbol),
        "universe": context.symbol,
        "model": model,
        "classes": classes,
        "feature_columns": feature_cols,
        "feature_medians": medians.to_dict(),
        "trained_at": now.isoformat(),
        "training_timestamp": now.isoformat(),
        "training_data_horizon": f"{filtered['date'].iloc[0]}..{filtered['date'].iloc[-1]}",
        "feature_schema_version": feature_schema_version(feature_cols),
        "code_version": str(release["python_api_version"]),
        "release_version": str(release["release_version"]),
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "validation_start": split["validation_start"],
        "validation_end": split["validation_end"],
        "test_start": split["test_start"],
        "test_end": split["test_end"],
        "train_rows": int(len(x_train)),
        "validation_sample_count": int(len(y_validation)),
        "test_sample_count": int(len(y_test)),
        "validation_metric_name": "accuracy",
        "validation_metric_value": validation_score,
        "calibration_metric_name": "multiclass_brier_score",
        "calibration_metric_value": calibration["brier_score"],
        "calibration_sample_count": int(len(y_test)),
        "n_features": int(len(feature_cols)),
        "expected_feature_count": int(len(feature_cols)),
        "dataset_fingerprint": dataset_fingerprint(
            pd.concat(
                [filtered[["date", "target"]].reset_index(drop=True), x_frame],
                axis=1,
            )
        ),
        "runtime_versions": runtime_versions(model),
        "excluded_all_null_training_features": excluded_features,
        "train_score": train_score,
        "val_score": validation_score,
        "test_score": test_score,
        "bif_rows_used": (
            int(pd.to_numeric(filtered.iloc[train_slice]["bif_available"], errors="coerce").fillna(0).sum())
            if "bif_available" in filtered.columns
            else 0
        ),
        "ohlcv_rows_used": int(len(context.ohlcv)),
        "calibrated": calibrated,
        "calibration_warning": calibration_warning,
        "feature_importance": feature_importance,
        "feature_importance_top5": dict(list(feature_importance.items())[:5]),
    }
    _, record = save_artifact(model_id, artifact, context.symbol)
    artifact["artifact_hash"] = record["artifact_hash"]
    artifact["registry_record"] = record
    return artifact, None, int(len(x_frame))


def predict_classifier_artifact(
    artifact: dict[str, Any],
    context: MLContext,
    *,
    confidence_multiplier: float = 1.0,
) -> dict[str, Any]:
    columns = list(artifact["feature_columns"])
    vector = context.current_vector.reindex(columns).replace([np.inf, -np.inf], np.nan)
    missing_features = [str(column) for column in columns if pd.isna(vector.get(column))]
    schema_compatible = bool(
        vector.notna().all() and np.isfinite(vector.to_numpy(dtype=float)).all()
    )
    medians = pd.Series(artifact.get("feature_medians", {}), dtype=float).reindex(columns)
    vector = vector.fillna(medians)
    unresolved = [str(column) for column in columns if pd.isna(vector.get(column))]
    if unresolved or not np.isfinite(vector.to_numpy(dtype=float)).all():
        return {
            "prediction": None,
            "probabilities": {},
            "confidence": None,
            "is_trained": True,
            "calibrated": bool(artifact.get("calibrated")),
            "feature_schema_compatible": False,
            "missing_features": missing_features,
            "unresolved_features": unresolved,
            "reason_code": "CURRENT_FEATURE_VECTOR_UNAVAILABLE",
            "reason": (
                "current feature vector cannot be completed from stored training medians: "
                + ", ".join(unresolved[:10])
            ),
            "model_attribution": model_attribution(
                artifact,
                context.symbol,
                data_as_of=_data_as_of(context),
                feature_schema_compatible=False,
            ),
        }
    x_current = vector.to_numpy(dtype=float).reshape(1, -1)
    model = artifact["model"]
    classes = list(artifact["classes"])
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x_current)[0]
        model_classes = getattr(model, "classes_", np.arange(len(raw)))
        mapped = np.zeros(len(classes), dtype=float)
        for index, class_id in enumerate(model_classes):
            class_index = int(class_id)
            if 0 <= class_index < len(classes):
                mapped[class_index] = float(raw[index])
    else:
        prediction = int(model.predict(x_current)[0])
        mapped = np.zeros(len(classes), dtype=float)
        mapped[prediction] = 1.0
    probabilities = normalise_probabilities(classes, mapped)
    prediction = max(probabilities, key=probabilities.get)
    confidence = clipped_confidence(probabilities, confidence_multiplier)
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "confidence": confidence,
        "is_trained": True,
        "trained_at": artifact.get("trained_at"),
        "train_rows": artifact.get("train_rows"),
        "calibrated": bool(artifact.get("calibrated")),
        "calibration_warning": artifact.get("calibration_warning"),
        "feature_importance": artifact.get("feature_importance", {}),
        "feature_schema_compatible": schema_compatible,
        "missing_features": missing_features,
        "imputed_feature_count": len(missing_features),
        "confidence_formula": "maximum calibrated class probability, capped at 0.85",
        "model_attribution": model_attribution(
            artifact,
            context.symbol,
            data_as_of=_data_as_of(context),
            feature_schema_compatible=schema_compatible,
        ),
    }


def save_artifact(
    model_id: str,
    artifact: dict[str, Any],
    symbol: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    target_dir = artifact_directory(symbol)
    target_dir.mkdir(parents=True, exist_ok=True)
    version = str(artifact.get("model_version") or "").strip()
    if not version:
        raise RuntimeError("model version missing")
    artifact["serialization_format"] = SERIALIZATION_FORMAT
    artifact["artifact_format_version"] = SERIALIZATION_VERSION
    path = target_dir / f"{model_id}_{version}.skops"
    temporary = path.with_suffix(".skops.tmp")
    serialized = sio.dumps(artifact, compression=zipfile.ZIP_STORED)
    _validate_skops_payload(serialized)
    require_free_space(path, len(serialized) * 2, "model artifact save")
    with temporary.open("wb") as handle:
        handle.write(serialized)
        handle.flush()
    temporary.replace(path)
    try:
        record = register_artifact(path, artifact, SAVED_MODELS_DIR)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    _keep_last_versions(model_id, target_dir, keep=3, symbol=str(artifact["symbol"]))
    _cache_artifact(path, record, artifact)
    return path, record


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over the exact ordered training dataset values."""
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("datetime64[ns]").astype(str)
        elif normalized[column].dtype == "object":
            normalized[column] = normalized[column].astype(str)
    hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return "sha256:" + __import__("hashlib").sha256(hashes.tobytes()).hexdigest()


def runtime_versions(model: Any) -> dict[str, str]:
    packages = {"numpy", "pandas", "scikit-learn", "skops"}
    root = str(type(model).__module__).split(".", 1)[0]
    package_alias = {"sklearn": "scikit-learn"}.get(root, root)
    packages.add(package_alias)
    versions = {"python": platform.python_version()}
    for package in sorted(packages):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "UNAVAILABLE"
    return versions


def load_latest_artifact(
    model_id: str,
    symbol: str | None = None,
    *,
    allow_proxy: bool = True,
) -> dict[str, Any] | None:
    if not SAVED_MODELS_DIR.exists():
        return None
    requested = safe_symbol(symbol or "GLOBAL")
    candidates = [(requested, "PRIMARY")]
    if allow_proxy and requested != safe_symbol(PROXY_SYMBOL):
        candidates.append((safe_symbol(PROXY_SYMBOL), "PROXY"))
    if allow_proxy and requested != "GLOBAL":
        candidates.append(("GLOBAL", "FALLBACK"))
    for candidate_symbol, fallback_status in candidates:
        record = latest_validated_record(model_id, candidate_symbol)
        if record is None:
            continue
        cached = _cached_artifact(record, SAVED_MODELS_DIR)
        if cached is not None:
            cached["fallback_status"] = fallback_status
            cached["requested_symbol"] = symbol or "GLOBAL"
            return cached
        try:
            path, artifact = load_verified_artifact(record, SAVED_MODELS_DIR)
            artifact["artifact_hash"] = record["artifact_hash"]
            artifact["registry_record"] = record
            artifact["fallback_status"] = fallback_status
            artifact["requested_symbol"] = symbol or "GLOBAL"
            _cache_artifact(path, record, artifact)
            return artifact
        except Exception as error:
            _report_artifact_load_failure(model_id, record, error)
    return None


def load_verified_artifact(
    record: dict[str, Any],
    saved_root: Path = SAVED_MODELS_DIR,
) -> tuple[Path, dict[str, Any]]:
    """Hash-check then deserialize one registry record; raise on any defect."""
    verified = read_verified_artifact_bytes(
        record,
        saved_root,
        maximum_bytes=MAX_ARTIFACT_BYTES,
    )
    if verified is None:
        raise RuntimeError("artifact hash, size, path, or file-type verification failed")
    path, serialized = verified
    _validate_skops_payload(serialized)
    trusted = sio.get_untrusted_types(data=serialized)
    artifact = sio.loads(serialized, trusted=trusted)
    if not isinstance(artifact, dict) or artifact.get("model_version") != record["model_version"]:
        raise RuntimeError("artifact metadata does not match registry")
    if (
        artifact.get("serialization_format") != SERIALIZATION_FORMAT
        or artifact.get("artifact_format_version") != SERIALIZATION_VERSION
    ):
        raise RuntimeError("artifact serialization metadata is unsupported")
    return path, artifact


def _report_artifact_load_failure(
    model_id: str,
    record: dict[str, Any],
    error: Exception,
) -> None:
    text = str(error).strip()
    cause = next((line.strip() for line in text.splitlines() if line.strip()), type(error).__name__)
    cause = cause[:240]
    key = (model_id, type(error).__name__, cause)
    with _ARTIFACT_LOAD_FAILURES_LOCK:
        if key in _ARTIFACT_LOAD_FAILURES:
            return
        _ARTIFACT_LOAD_FAILURES.add(key)
    artifact = Path(str(record.get("artifact_path") or "UNKNOWN")).name
    _RUNTIME_LOG.exception(
        "model=%s artifact=%s exception=%s cause=%s state=FAIL_CLOSED",
        model_id,
        artifact,
        type(error).__name__,
        cause,
        extra={
            "console_meta": {
                "tag": "ERROR",
                "category": "ML",
                "detail": "model unavailable; actionability fail-closed",
            }
        },
    )


def model_status(model_id: str, symbol: str | None = None) -> dict[str, Any]:
    artifact = load_latest_artifact(model_id, symbol, allow_proxy=False)
    if artifact is None:
        return {"model_id": model_id, "is_trained": False, "latest_version": None}
    return {
        "model_id": model_id,
        "is_trained": True,
        "latest_version": artifact.get("trained_at"),
        "train_rows": artifact.get("train_rows"),
        "calibrated": bool(artifact.get("calibrated")),
        "model_version": artifact.get("model_version"),
        "artifact_hash": artifact.get("artifact_hash"),
        "validation_status": artifact.get("validation_status"),
    }


def artifact_directory(symbol: str | None = None) -> Path:
    if symbol is None:
        return SAVED_MODELS_DIR
    return SAVED_MODELS_DIR / safe_symbol(symbol)


def safe_symbol(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", symbol.strip().upper())
    return cleaned.strip("_") or "UNKNOWN"


def direct_artifact_path(model_id: str, symbol: str) -> Path | None:
    record = latest_validated_record(model_id, safe_symbol(symbol))
    return resolve_verified_path(record, SAVED_MODELS_DIR) if record else None


def artifact_is_fresh(model_id: str, symbol: str, max_age_hours: int = 24) -> bool:
    record = latest_validated_record(model_id, safe_symbol(symbol))
    if record is None or resolve_verified_path(record, SAVED_MODELS_DIR) is None:
        return False
    try:
        trained_at = dt.datetime.fromisoformat(str(record["training_timestamp"]))
    except ValueError:
        return False
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - trained_at.astimezone(dt.timezone.utc)
    return age.total_seconds() <= max_age_hours * 3600


def symbol_has_artifacts(symbol: str, model_ids: list[str] | None = None) -> bool:
    ids = model_ids or ["m01_vol_regime", "m02_direction", "m21_ohlc"]
    return any(latest_validated_record(model_id, safe_symbol(symbol)) is not None for model_id in ids)


def trained_model_symbols(model_ids: list[str] | None = None) -> list[str]:
    ids = model_ids or ["m01_vol_regime", "m02_direction", "m21_ohlc"]
    return sorted({str(record["symbol"]) for record in validated_records(ids)})


def safe_probability(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def temporal_train_validation_test_split(dates: pd.Series, *, gap: int) -> dict[str, Any]:
    ordered = pd.to_datetime(dates, errors="raise").reset_index(drop=True)
    if not ordered.is_monotonic_increasing or ordered.duplicated().any():
        raise ValueError("training dates must be unique and chronological")
    rows = len(ordered)
    holdout = max(20, min(rows // 5, 100))
    train_stop = rows - (2 * holdout) - (2 * gap)
    if train_stop < 30:
        raise ValueError("insufficient rows for purged train/validation/test splits")
    validation_start = train_stop + gap
    validation_stop = validation_start + holdout
    test_start = validation_stop + gap
    slices = (slice(0, train_stop), slice(validation_start, validation_stop), slice(test_start, rows))
    boundaries = {
        "train_start": ordered.iloc[0].date().isoformat(),
        "train_end": ordered.iloc[train_stop - 1].date().isoformat(),
        "validation_start": ordered.iloc[validation_start].date().isoformat(),
        "validation_end": ordered.iloc[validation_stop - 1].date().isoformat(),
        "test_start": ordered.iloc[test_start].date().isoformat(),
        "test_end": ordered.iloc[-1].date().isoformat(),
        "slices": slices,
    }
    if not (
        boundaries["train_end"] < boundaries["validation_start"]
        and boundaries["validation_end"] < boundaries["test_start"]
    ):
        raise ValueError("temporal split boundaries overlap")
    return boundaries


def model_attribution(
    artifact: dict[str, Any],
    requested_symbol: str,
    *,
    data_as_of: str | None,
    feature_schema_compatible: bool = True,
) -> dict[str, Any]:
    return {
        "model_id": artifact.get("model_id"),
        "model_version": artifact.get("model_version"),
        "trained_symbol": artifact.get("symbol"),
        "requested_symbol": requested_symbol,
        "universe": artifact.get("universe"),
        "training_data_horizon": artifact.get("training_data_horizon"),
        "feature_schema_version": artifact.get("feature_schema_version"),
        "training_timestamp": artifact.get("training_timestamp"),
        "artifact_hash": artifact.get("artifact_hash"),
        "code_version": artifact.get("code_version"),
        "release_version": artifact.get("release_version"),
        "validation_status": artifact.get("validation_status"),
        "artifact_verified": True,
        "registry_status": "VALIDATED",
        "feature_schema_compatible": feature_schema_compatible,
        "fallback_status": artifact.get("fallback_status", "PRIMARY"),
        "data_as_of": data_as_of,
    }


def _classification_skill_floor(class_count: int, validation_n: int) -> float:
    """Minimum validation accuracy a classifier must clear to be registered
    VALIDATED, instead of merely completing the temporal-split pipeline.

    A model that never beats random guessing must not become
    `registry_status == "VALIDATED"` -- recommendation_eligibility() and
    ml/service.py's confidence gate both treat VALIDATED as a proxy for
    "this model has genuine skill", not "this code path ran without error".
    The bound is a one-sided 95% binomial test against the chance baseline
    (1/class_count): the further validation_n is below what's needed to
    distinguish skill from luck, the higher the bar, so thin validation
    windows are held to a *stricter* standard rather than waved through.
    """
    chance = 1.0 / max(2, class_count)
    if validation_n <= 0:
        return 1.0
    standard_error = math.sqrt(chance * (1.0 - chance) / validation_n)
    return min(1.0, chance + 1.645 * standard_error)


def _score(model: Any, features: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        value = float(model.score(features, labels))
    except Exception:
        return None
    return round(value, 6) if math.isfinite(value) else None


def _classification_calibration(model: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    if not hasattr(model, "predict_proba") or not len(labels):
        return {"brier_score": None}
    try:
        raw = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(getattr(model, "classes_", []), dtype=int)
        aligned = np.zeros((len(labels), int(max(classes.max(initial=0), labels.max(initial=0))) + 1))
        for raw_index, class_id in enumerate(classes):
            aligned[:, int(class_id)] = raw[:, raw_index]
        truth = np.zeros_like(aligned)
        truth[np.arange(len(labels)), labels] = 1.0
        value = float(np.mean(np.sum((aligned - truth) ** 2, axis=1)))
        return {"brier_score": round(value, 6) if math.isfinite(value) else None}
    except Exception:
        return {"brier_score": None}


def _data_as_of(context: MLContext) -> str | None:
    if context.feature_frame.empty or "date" not in context.feature_frame.columns:
        return None
    value = pd.to_datetime(context.feature_frame["date"], errors="coerce").max()
    return value.date().isoformat() if pd.notna(value) else None


def _feature_importance(estimator: Any, feature_cols: list[str]) -> dict[str, float]:
    try:
        values = getattr(estimator, "feature_importances_", None)
    except Exception:
        return {}
    if values is None:
        values = _calibrated_feature_importances(estimator)
    if values is None:
        return {}
    try:
        values_array = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return {}
    if values_array.size != len(feature_cols):
        return {}
    pairs = sorted(
        zip(feature_cols, [float(value) for value in values_array]),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    total = sum(value for _, value in pairs) or 1.0
    return {name: round(value / total, 6) for name, value in pairs}


def _calibrated_feature_importances(estimator: Any) -> Any:
    calibrated_classifiers = getattr(estimator, "calibrated_classifiers_", None)
    if not calibrated_classifiers:
        return None
    arrays = []
    for calibrated in calibrated_classifiers:
        base = getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
        values = getattr(base, "feature_importances_", None)
        if values is not None:
            try:
                arrays.append(np.asarray(values, dtype=float).reshape(-1))
            except Exception:
                continue
    if not arrays:
        return None
    return np.mean(arrays, axis=0)


def _keep_last_versions(model_id: str, target_dir: Path, keep: int, *, symbol: str) -> None:
    matches = sorted(target_dir.glob(f"{model_id}_*.skops"), reverse=True)
    for stale in matches[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass
    retained = {
        path.resolve().relative_to(SAVED_MODELS_DIR.resolve()).as_posix()
        for path in matches[:keep]
        if path.is_file()
    }
    retire_missing_versions(model_id, safe_symbol(symbol), retained)


def _validate_skops_payload(data: bytes) -> None:
    if not data or len(data) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("model artifact size is outside the allowed bounds")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise RuntimeError("model artifact contains too many archive entries")
            if any(entry.compress_type != zipfile.ZIP_STORED for entry in entries):
                raise RuntimeError("compressed model archives are not accepted")
            if sum(entry.file_size for entry in entries) > MAX_UNCOMPRESSED_BYTES:
                raise RuntimeError("model artifact expands beyond the allowed bound")
    except zipfile.BadZipFile as error:
        raise RuntimeError("model artifact is not a valid skops archive") from error
    untrusted = set(sio.get_untrusted_types(data=data))
    unexpected = sorted(untrusted - TRUSTED_SKOPS_TYPES)
    if unexpected:
        raise RuntimeError(f"model artifact contains unsupported types: {', '.join(unexpected)}")


def _cache_artifact(path: Path, record: dict[str, Any], artifact: dict[str, Any]) -> None:
    stat = path.stat()
    key = str(record["artifact_hash"])
    cached = dict(artifact)
    cached["artifact_hash"] = key
    cached["registry_record"] = dict(record)
    with _ARTIFACT_CACHE_LOCK:
        _ARTIFACT_CACHE[key] = (path.resolve(), stat.st_size, stat.st_mtime_ns, cached)
        _ARTIFACT_CACHE.move_to_end(key)
        while len(_ARTIFACT_CACHE) > MAX_CACHED_ARTIFACTS:
            _ARTIFACT_CACHE.popitem(last=False)


def _cached_artifact(record: dict[str, Any], saved_root: Path) -> dict[str, Any] | None:
    key = str(record["artifact_hash"])
    with _ARTIFACT_CACHE_LOCK:
        entry = _ARTIFACT_CACHE.get(key)
        if entry is None:
            return None
        path, size, modified_ns, artifact = entry
        expected = (saved_root.resolve() / str(record["artifact_path"])).resolve()
        try:
            expected.relative_to(saved_root.resolve())
            stat = expected.stat()
        except (ValueError, OSError):
            _ARTIFACT_CACHE.pop(key, None)
            return None
        if expected != path or stat.st_size != size or stat.st_mtime_ns != modified_ns:
            _ARTIFACT_CACHE.pop(key, None)
            return None
        _ARTIFACT_CACHE.move_to_end(key)
        return dict(artifact)
