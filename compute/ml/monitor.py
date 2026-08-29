from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from typing import Any

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read


def ensure_ml_tables() -> None:
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id              TEXT        PRIMARY KEY,
                    model_id        TEXT        NOT NULL,
                    predicted_at    TIMESTAMP   NOT NULL,
                    features_hash   TEXT,
                    prediction      TEXT        NOT NULL,
                    probabilities   TEXT        NOT NULL,
                    confidence      REAL        NOT NULL,
                    outcome         TEXT,
                    outcome_at      TIMESTAMP,
                    correct         INTEGER,
                    brier_score     REAL
                )
                """
            )
            for name, data_type in (
                ("symbol", "TEXT"),
                ("data_as_of", "TEXT"),
                ("model_version", "TEXT"),
                ("artifact_hash", "TEXT"),
                ("feature_schema_version", "TEXT"),
                ("release_version", "TEXT"),
                ("validation_status", "TEXT"),
                ("fallback_status", "TEXT"),
                ("outcome_error", "REAL"),
            ):
                con.execute(f"ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS {name} {data_type}")
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ml_model
                ON ml_predictions(model_id, predicted_at)
                """
            )
        finally:
            con.close()


def log_prediction(
    model_id: str,
    prediction: str | None,
    probabilities: dict[str, float] | None,
    confidence: float | None,
    features: dict[str, Any],
    *,
    attribution: dict[str, Any],
) -> str:
    ensure_ml_tables()
    clean_probs = probabilities or {}
    payload = json.dumps(clean_probs, sort_keys=True)
    feature_hash = hashlib.sha256(
        json.dumps(features, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]
    prediction_id = str(uuid.uuid4())
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                INSERT INTO ml_predictions
                    (id, model_id, predicted_at, features_hash, prediction,
                     probabilities, confidence, symbol, data_as_of, model_version,
                     artifact_hash, feature_schema_version, release_version,
                     validation_status, fallback_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id,
                    model_id,
                    dt.datetime.now(dt.timezone.utc),
                    feature_hash,
                    prediction or "UNTRAINED",
                    payload,
                    float(confidence or 0.0),
                    attribution.get("requested_symbol"),
                    attribution.get("data_as_of"),
                    attribution.get("model_version"),
                    attribution.get("artifact_hash"),
                    attribution.get("feature_schema_version"),
                    attribution.get("release_version"),
                    attribution.get("validation_status"),
                    attribution.get("fallback_status"),
                ],
            )
        finally:
            con.close()
    return prediction_id


def record_prediction_outcome(prediction_id: str, actual: str) -> dict[str, Any]:
    ensure_ml_tables()
    normalized = actual.strip().upper()
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            row = con.execute(
                "SELECT prediction, probabilities, outcome FROM ml_predictions WHERE id = ?",
                [prediction_id],
            ).fetchone()
            if row is None:
                raise KeyError("prediction not found")
            if row[2] is not None:
                raise RuntimeError("prediction outcome already recorded")
            probabilities = json.loads(str(row[1]))
            if not isinstance(probabilities, dict):
                probabilities = {}
            clean = {
                str(label): float(value)
                for label, value in probabilities.items()
                if isinstance(value, (int, float))
            }
            correct = int(str(row[0]).upper() == normalized)
            error = float(1 - correct)
            brier = _multiclass_brier(clean, normalized)
            con.execute(
                """
                UPDATE ml_predictions
                SET outcome = ?, outcome_at = ?, correct = ?, outcome_error = ?, brier_score = ?
                WHERE id = ? AND outcome IS NULL
                """,
                [
                    normalized,
                    dt.datetime.now(dt.timezone.utc),
                    correct,
                    error,
                    brier,
                    prediction_id,
                ],
            )
        finally:
            con.close()
    return {
        "prediction_id": prediction_id,
        "actual": normalized,
        "error": error,
        "correct": bool(correct),
        "calibration_metric": {"name": "multiclass_brier_score", "value": brier},
    }


def accuracy_for_model(model_id: str, n_days: int = 30) -> dict[str, Any]:
    result = resolve_read(
        ReadOperation.LOAD_MODEL_ACCURACY_STATS,
        {"model_id": model_id, "n_days": int(n_days)},
    )
    rows = result["row"]
    total = int(rows[0] or 0)
    correct = int(rows[1] or 0)
    accuracy = (correct / total) if total else None
    brier = round(float(rows[3]), 6) if rows[3] is not None else None
    return {
        "model_id": model_id,
        "window_days": n_days,
        "resolved_predictions": total,
        "sample_count": total,
        "measured": total > 0,
        "accuracy": round(accuracy, 6) if accuracy is not None else None,
        "classification_error_rate": round(float(rows[2]), 6) if rows[2] is not None else None,
        "brier_score": brier,
        "calibration_metric": {
            "name": "multiclass_brier_score",
            "value": brier,
            "sample_count": total,
            "measured": total > 0 and brier is not None,
        },
    }


def accuracy_report(n_days: int = 30) -> dict[str, Any]:
    model_ids = [
        "m01_vol_regime",
        "m02_direction",
        "m03_strategy",
        "m04_iv_reversion",
        "m05_anomaly",
        "m06_regime_return",
        "m07_iv_forecast",
        "m08_gp_regime",
        "m09_rl_sizer",
        "m10_ensemble",
        "m11_tft",
        "m12_prophet",
        "m13_nbeats",
        "m14_wavelet",
        "m15_kalman",
        "m16_survival",
        "m17_conformal",
        "m18_bandit",
        "m19_dqn",
        "m20_bnn",
        "m21_ohlc",
    ]
    return {
        "window_days": n_days,
        "models": {model_id: accuracy_for_model(model_id, n_days) for model_id in model_ids},
    }


def _multiclass_brier(probabilities: dict[str, float], actual: str) -> float | None:
    if not probabilities:
        return None
    labels = set(probabilities)
    labels.add(actual)
    value = sum((float(probabilities.get(label, 0.0)) - float(label == actual)) ** 2 for label in labels)
    return round(value, 6)
