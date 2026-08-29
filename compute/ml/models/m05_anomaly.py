from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext


MODEL_ID = "m05_anomaly"
MINIMUM_ROWS = 30


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    rows = context.available_rows
    return {
        "model_id": MODEL_ID,
        "is_trained": rows >= MINIMUM_ROWS,
        "train_rows": rows,
        "model_used": "isolation_forest" if _sklearn_available() else "robust_zscore_fallback",
    }


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    if context.available_rows < MINIMUM_ROWS:
        return {
            "model_id": MODEL_ID,
            "is_trained": False,
            "is_anomalous": False,
            "prediction": "NORMAL",
            "anomaly_score": 0.1,
            "anomaly_type": None,
            "confidence_multiplier": 1.0,
            "similar_historical_dates": None,
            "reason": f"insufficient data: {context.available_rows}/{MINIMUM_ROWS} rows",
        }
    if _sklearn_available():
        return _isolation_forest_score(context)
    return _robust_zscore_score(context)


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _isolation_forest_score(context: MLContext) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest

    matrix, columns, current = _matrix(context)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(matrix)
    raw = -float(model.score_samples(current.reshape(1, -1))[0])
    train_scores = -model.score_samples(matrix)
    low = float(np.percentile(train_scores, 5))
    high = float(np.percentile(train_scores, 95))
    score = _scale_score(raw, low, high)
    anomalous = score >= 0.75
    return {
        "model_id": MODEL_ID,
        "is_trained": True,
        "prediction": "ANOMALOUS" if anomalous else "NORMAL",
        "is_anomalous": anomalous,
        "anomaly_score": round(score, 6),
        "anomaly_type": _anomaly_type(context, columns, current, matrix),
        "confidence_multiplier": 0.2 if anomalous else 1.0,
        "similar_historical_dates": _similar_dates(context, matrix, current),
        "model_used": "isolation_forest",
    }


def _robust_zscore_score(context: MLContext) -> dict[str, Any]:
    matrix, columns, current = _matrix(context)
    median = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - median), axis=0)
    mad[mad < 1e-9] = 1.0
    robust_z = np.abs((current - median) / (1.4826 * mad))
    top = np.sort(robust_z)[-10:]
    score = min(1.0, float(np.nanmean(top)) / 6.0)
    anomalous = score >= 0.75
    return {
        "model_id": MODEL_ID,
        "is_trained": True,
        "prediction": "ANOMALOUS" if anomalous else "NORMAL",
        "is_anomalous": anomalous,
        "anomaly_score": round(score, 6),
        "anomaly_type": _anomaly_type(context, columns, current, matrix),
        "confidence_multiplier": 0.2 if anomalous else 1.0,
        "similar_historical_dates": _similar_dates(context, matrix, current),
        "model_used": "robust_zscore_fallback",
    }


def _matrix(context: MLContext) -> tuple[np.ndarray, list[str], np.ndarray]:
    frame = context.feature_frame.drop(columns=["date"], errors="ignore").select_dtypes(include=["number"])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    medians = frame.median(numeric_only=True).fillna(0.0)
    frame = frame.fillna(medians)
    columns = list(frame.columns)
    matrix = frame.to_numpy(dtype=float)
    current = context.current_vector.reindex(columns).fillna(medians).fillna(0.0).to_numpy(dtype=float)
    return matrix, columns, current


def _scale_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.1
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _anomaly_type(
    context: MLContext,
    columns: list[str],
    current: np.ndarray,
    matrix: np.ndarray,
) -> str | None:
    del context
    if not columns:
        return None
    median = np.nanmedian(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std[std < 1e-9] = 1.0
    index = int(np.nanargmax(np.abs((current - median) / std)))
    name = columns[index].lower()
    if any(term in name for term in ["iv", "vix", "hv", "atr", "vol"]):
        return "vol_spike"
    if any(term in name for term in ["gex", "oi", "pcr"]):
        return "oi_extreme"
    if any(term in name for term in ["corr", "cross", "global", "usdinr", "brent"]):
        return "correlation_breakdown"
    return "unknown"


def _similar_dates(context: MLContext, matrix: np.ndarray, current: np.ndarray) -> list[str] | None:
    if matrix.shape[0] < 5 or "date" not in context.feature_frame.columns:
        return None
    distances = np.linalg.norm(matrix - current.reshape(1, -1), axis=1)
    indexes = np.argsort(distances)[:3]
    dates = context.feature_frame.iloc[indexes]["date"].astype(str).tolist()
    return dates
