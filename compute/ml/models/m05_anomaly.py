from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from ml.feature_engineering import MLContext


MODEL_ID = "m05_anomaly"
MINIMUM_REFERENCE_ROWS = 30
ANOMALY_THRESHOLD = 0.75


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    """Report readiness for the online detector; no persistent artifact is trained."""
    del force
    reference_rows = max(0, context.available_rows - 1)
    ready = reference_rows >= MINIMUM_REFERENCE_ROWS
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": ready,
        "status": "READY_ANALYTICAL" if ready else "INSUFFICIENT_DATA",
        "reason_code": None if ready else "INSUFFICIENT_REFERENCE_HISTORY",
        "available_rows": reference_rows,
        "minimum_rows": MINIMUM_REFERENCE_ROWS,
        "model_kind": "ONLINE_REFERENCE_DETECTOR",
    }


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    reference_rows = max(0, context.available_rows - 1)
    if reference_rows < MINIMUM_REFERENCE_ROWS:
        return _unavailable(
            "INSUFFICIENT_REFERENCE_HISTORY",
            f"reference history {reference_rows}/{MINIMUM_REFERENCE_ROWS} rows",
            reference_rows,
        )
    market = getattr(context, "market_data_eligibility", None)
    if market is None or getattr(market, "decision_eligible", False) is not True:
        return _unavailable(
            "CURRENT_MARKET_DATA_INELIGIBLE",
            "current anomaly observation is stale, partial, or lacks verified market provenance",
            reference_rows,
        )
    try:
        prepared = _prepare_matrix(context)
    except ValueError as error:
        return _unavailable(
            "FEATURE_PREPROCESSING_UNAVAILABLE",
            str(error),
            reference_rows,
        )
    if _sklearn_available():
        return _isolation_forest_score(context, prepared)
    return _robust_zscore_score(context, prepared)


def _unavailable(
    reason_code: str,
    reason: str,
    reference_rows: int,
) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": False,
        "status": "INSUFFICIENT_DATA" if reason_code.startswith("INSUFFICIENT") else "UNAVAILABLE",
        "reason_code": reason_code,
        "reason": reason,
        "prediction": None,
        "is_anomalous": None,
        "anomaly_score": None,
        "anomaly_type": None,
        "confidence_multiplier": None,
        "similar_historical_dates": None,
        "reference_rows": reference_rows,
        "minimum_rows": MINIMUM_REFERENCE_ROWS,
        "model_kind": "ONLINE_REFERENCE_DETECTOR",
    }


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _isolation_forest_score(
    context: MLContext,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest

    matrix = prepared["matrix"]
    current = prepared["current"]
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(matrix)
    raw = -float(model.score_samples(current.reshape(1, -1))[0])
    reference_scores = -model.score_samples(matrix)
    low = float(np.percentile(reference_scores, 5))
    high = float(np.percentile(reference_scores, 95))
    score = _scale_score(raw, low, high)
    return _result(context, prepared, score, "isolation_forest")


def _robust_zscore_score(
    context: MLContext,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    matrix = prepared["matrix"]
    current = prepared["current"]
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    mad[mad < 1e-9] = 1.0
    robust_z = np.abs((current - median) / (1.4826 * mad))
    top = np.sort(robust_z)[-min(10, robust_z.size):]
    score = min(1.0, float(np.mean(top)) / 6.0)
    return _result(context, prepared, score, "robust_zscore")


def _result(
    context: MLContext,
    prepared: dict[str, Any],
    score: float,
    method: str,
) -> dict[str, Any]:
    anomalous = score >= ANOMALY_THRESHOLD
    columns = prepared["columns"]
    matrix = prepared["matrix"]
    current = prepared["current"]
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "ANOMALOUS" if anomalous else "READY",
        "reason_code": "ANOMALY_THRESHOLD_EXCEEDED" if anomalous else None,
        "prediction": "ANOMALOUS" if anomalous else "NORMAL",
        "is_anomalous": anomalous,
        "anomaly_score": round(score, 6),
        "anomaly_type": _anomaly_type(columns, current, matrix),
        "confidence_multiplier": 0.2 if anomalous else 1.0,
        "similar_historical_dates": _similar_dates(
            context,
            prepared["reference_indexes"],
            matrix,
            current,
        ),
        "model_used": method,
        "model_kind": "ONLINE_REFERENCE_DETECTOR",
        "reference_rows": int(matrix.shape[0]),
        "minimum_rows": MINIMUM_REFERENCE_ROWS,
        "feature_columns": columns,
        "feature_count": len(columns),
        "feature_schema_version": _schema_version(columns),
        "excluded_features": prepared["excluded_features"],
        "current_imputed_features": prepared["current_imputed_features"],
        "reference_imputed_cells": prepared["reference_imputed_cells"],
        "threshold": ANOMALY_THRESHOLD,
        "score_formula": "current score scaled to reference 5th..95th score percentiles",
        "preprocessing": "reference-only medians; current row excluded from fit and thresholds",
    }


def _prepare_matrix(context: MLContext) -> dict[str, Any]:
    numeric = (
        context.feature_frame.drop(columns=["date"], errors="ignore")
        .select_dtypes(include=["number"])
        .replace([np.inf, -np.inf], np.nan)
    )
    if len(numeric) < 2:
        raise ValueError("current observation and reference history are required")

    reference = numeric.iloc[:-1].copy()
    current = numeric.iloc[-1].copy()
    finite_reference = reference.notna().sum()
    columns = [str(column) for column in reference.columns if finite_reference[column] > 0]
    excluded = sorted(str(column) for column in reference.columns if finite_reference[column] == 0)
    if not columns:
        raise ValueError("no features have finite values in reference history")

    reference = reference[columns]
    current = current.reindex(columns)
    medians = reference.median(numeric_only=True).reindex(columns)
    unresolved = [str(column) for column in columns if pd.isna(medians.get(column))]
    if unresolved:
        raise ValueError("reference medians unavailable: " + ", ".join(unresolved[:10]))

    current_missing = [str(column) for column in columns if pd.isna(current.get(column))]
    reference_missing = int(reference.isna().sum().sum())
    reference = reference.fillna(medians)
    current = current.fillna(medians)
    matrix = reference.to_numpy(dtype=float)
    current_array = current.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(current_array).all():
        raise ValueError("non-finite values remain after reference-only median imputation")
    if matrix.shape[0] < MINIMUM_REFERENCE_ROWS:
        raise ValueError(
            f"reference history {matrix.shape[0]}/{MINIMUM_REFERENCE_ROWS} rows"
        )
    return {
        "matrix": matrix,
        "current": current_array,
        "columns": columns,
        "excluded_features": excluded,
        "current_imputed_features": current_missing,
        "reference_imputed_cells": reference_missing,
        "reference_indexes": list(reference.index),
    }


def _scale_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _anomaly_type(
    columns: list[str],
    current: np.ndarray,
    matrix: np.ndarray,
) -> str | None:
    if not columns:
        return None
    median = np.median(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std < 1e-9] = 1.0
    index = int(np.argmax(np.abs((current - median) / std)))
    name = columns[index].lower()
    if any(term in name for term in ["iv", "vix", "hv", "atr", "vol"]):
        return "vol_spike"
    if any(term in name for term in ["gex", "oi", "pcr"]):
        return "oi_extreme"
    if any(term in name for term in ["corr", "cross", "global", "usdinr", "brent"]):
        return "correlation_breakdown"
    return "unknown"


def _similar_dates(
    context: MLContext,
    reference_indexes: list[Any],
    matrix: np.ndarray,
    current: np.ndarray,
) -> list[str] | None:
    if matrix.shape[0] < 5 or "date" not in context.feature_frame.columns:
        return None
    distances = np.linalg.norm(matrix - current.reshape(1, -1), axis=1)
    nearest = np.argsort(distances)[:3]
    indexes = [reference_indexes[int(index)] for index in nearest]
    return context.feature_frame.loc[indexes, "date"].astype(str).tolist()


def _schema_version(columns: list[str]) -> str:
    encoded = json.dumps(columns, separators=(",", ":")).encode("utf-8")
    return "anomaly-features-v1:" + hashlib.sha256(encoded).hexdigest()[:16]
