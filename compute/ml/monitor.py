from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from typing import Any

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read

# model_id -> (target_builder, minimum future trading days needed before a
# prediction can be resolved). Only classifiers with a target definition
# that is (a) genuinely forward-looking and (b) defined independently of any
# trained artifact are eligible -- reusing the exact training-time target
# functions guarantees the resolved "actual" label can never drift from what
# the model was actually trained/evaluated against, and cannot be lookahead
# (it only ever reads OHLCV rows dated at or after the prediction's own
# data_as_of, which by construction have not yet occurred at prediction
# time and are only available now because real time has since passed).
_RESOLVABLE_TARGETS: dict[str, tuple[str, int]] = {
    "m02_direction": ("direction_target", 1),
    "m06_regime_return": ("direction_target", 1),
    "m01_vol_regime": ("vol_regime_target", 5),
}


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
                ("trained_symbol", "TEXT"),
            ):
                con.execute(f"ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS {name} {data_type}")
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ml_model
                ON ml_predictions(model_id, predicted_at)
                """
            )
            # Same duplicate-row class already found and fixed once in
            # ohlc_forecasts (m21_ohlc_forecast.py's _insert_forecast): a
            # bare INSERT with no natural-key guard means every re-run of a
            # scheduled snapshot for a (model_id, symbol, data_as_of) already
            # logged would silently add another full duplicate observation.
            # This index doesn't enforce uniqueness itself (DuckDB has no
            # partial/expression uniqueness here across nullable columns),
            # so log_prediction below does the existing-row check explicitly
            # -- this index just makes that check cheap.
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ml_natural_key
                ON ml_predictions(model_id, symbol, data_as_of)
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
) -> str | None:
    """Persist one attributed prediction observation. Idempotent on
    (model_id, symbol, data_as_of): a scheduled snapshot re-run for a
    combination already logged returns the existing row's id rather than
    inserting a duplicate -- see idx_ml_natural_key's comment for the prior
    incident (ohlc_forecasts) this mirrors. Returns None if a required
    attribution field is missing for the natural key, so a snapshot caller
    can skip counting it as newly logged."""
    if prediction is None:
        raise ValueError("unavailable prediction must not be logged")
    if confidence is None:
        raise ValueError("prediction confidence is required for OOS tracking")
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("prediction confidence must be finite") from error
    if not (0.0 <= numeric_confidence <= 1.0):
        raise ValueError("prediction confidence must be within [0, 1]")
    if not probabilities:
        raise ValueError("prediction probabilities are required for OOS tracking")
    symbol = attribution.get("requested_symbol")
    data_as_of = attribution.get("data_as_of")
    if not symbol or not data_as_of:
        raise ValueError("attribution.requested_symbol and data_as_of are required for OOS tracking")
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
            existing = con.execute(
                "SELECT id FROM ml_predictions WHERE model_id = ? AND symbol = ? AND data_as_of = ?",
                [model_id, symbol, data_as_of],
            ).fetchone()
            if existing:
                return str(existing[0])
            con.execute(
                """
                INSERT INTO ml_predictions
                    (id, model_id, predicted_at, features_hash, prediction,
                     probabilities, confidence, symbol, data_as_of, model_version,
                     artifact_hash, feature_schema_version, release_version,
                     validation_status, fallback_status, trained_symbol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id,
                    model_id,
                    dt.datetime.now(dt.timezone.utc),
                    feature_hash,
                    prediction,
                    payload,
                    numeric_confidence,
                    symbol,
                    data_as_of,
                    attribution.get("model_version"),
                    attribution.get("artifact_hash"),
                    attribution.get("feature_schema_version"),
                    attribution.get("release_version"),
                    attribution.get("validation_status"),
                    attribution.get("fallback_status"),
                    attribution.get("trained_symbol"),
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
        "status": "READY" if total >= 30 else "INSUFFICIENT_OOS_HISTORY",
        "required_sample_count": 30,
        "actual_sample_count": total,
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


class _OhlcvOnlyContext:
    """Minimal stand-in for MLContext -- direction_target/vol_regime_target
    only ever read `.ohlcv`, never any other MLContext field."""

    def __init__(self, ohlcv: Any) -> None:
        self.ohlcv = ohlcv


def _load_symbol_close_frame(symbol: str) -> Any:
    import pandas as pd

    from backtest.spot_quality import verified_spot_sources
    from data.index_aliases import index_symbol_candidates

    candidates = index_symbol_candidates(symbol) or [symbol]
    sources = sorted(verified_spot_sources(symbol))
    result = resolve_read(
        ReadOperation.LOAD_FEATURE_OHLCV,
        {"candidates": candidates, "sources": sources},
    )
    return pd.DataFrame(result["rows"], columns=result["columns"])


def resolve_pending_prediction_outcomes(
    model_ids: list[str] | None = None,
    *,
    limit: int = 5000,
) -> dict[str, Any]:
    """Canonical outcome resolver for classifier predictions logged via
    log_prediction()/PREDICT_AND_LOG. There has never been a scheduled task
    that calls record_prediction_outcome() -- it exists only as a manual
    POST /ml/predictions/{id}/outcome endpoint with no automated caller, so
    every classifier prediction ever logged has sat with outcome=NULL
    forever (verified live: 2,790 rows, 0 resolved, across every model_id).

    This resolves only the 3 model_ids with a target definition that is both
    genuinely forward-looking and defined independently of any trained
    artifact (direction_target for m02/m06, vol_regime_target for m01 -- see
    _RESOLVABLE_TARGETS), by re-deriving the actual label with the exact
    same training-time target function against OHLCV that has, in real
    time, since become available. It never rewrites `prediction`,
    `probabilities`, `confidence`, or any other prediction-time field --
    only ever fills the previously-NULL outcome/outcome_at/correct/
    brier_score columns, and only once, per record_prediction_outcome's own
    idempotency guard (raises "already recorded" on a second attempt, caught
    and skipped here).

    No lookahead: a row is only resolved once real time has actually passed
    its target date and that date's OHLCV is present, and the resolved
    label is derived from data no earlier than the prediction's own
    data_as_of -- never from data that predates it.
    """
    from ml.feature_engineering import direction_target, vol_regime_target

    target_builders = {"direction_target": direction_target, "vol_regime_target": vol_regime_target}
    eligible = [m for m in (model_ids or list(_RESOLVABLE_TARGETS)) if m in _RESOLVABLE_TARGETS]
    summary: dict[str, Any] = {
        "eligible_model_ids": eligible,
        "candidates": 0,
        "resolved": 0,
        "not_yet_resolvable": 0,
        "already_resolved_skipped": 0,
        "no_symbol_or_data_as_of": 0,
        "errors": 0,
        "by_model": {m: {"resolved": 0, "not_yet_resolvable": 0} for m in eligible},
    }
    if not eligible:
        return summary

    placeholders = ", ".join("?" for _ in eligible)
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            rows = con.execute(
                f"""
                SELECT id, model_id, symbol, data_as_of
                FROM ml_predictions
                WHERE model_id IN ({placeholders}) AND outcome IS NULL
                ORDER BY data_as_of ASC
                LIMIT ?
                """,
                [*eligible, int(limit)],
            ).fetchall()
        finally:
            con.close()
    summary["candidates"] = len(rows)

    labels_cache: dict[tuple[str, str], Any] = {}
    for prediction_id, model_id, symbol, data_as_of in rows:
        if not symbol or not data_as_of:
            summary["no_symbol_or_data_as_of"] += 1
            continue
        builder_name, _horizon = _RESOLVABLE_TARGETS[model_id]
        cache_key = (builder_name, str(symbol))
        labels = labels_cache.get(cache_key)
        if labels is None:
            try:
                frame = _load_symbol_close_frame(str(symbol))
                labels = target_builders[builder_name](_OhlcvOnlyContext(frame))
            except Exception:
                summary["errors"] += 1
                continue
            labels_cache[cache_key] = labels
        try:
            anchor = dt.date.fromisoformat(str(data_as_of)[:10])
        except ValueError:
            summary["no_symbol_or_data_as_of"] += 1
            continue
        actual = labels.get(anchor) if hasattr(labels, "get") else None
        if actual is None or (isinstance(actual, float) and actual != actual):
            summary["not_yet_resolvable"] += 1
            summary["by_model"][model_id]["not_yet_resolvable"] += 1
            continue
        try:
            record_prediction_outcome(str(prediction_id), str(actual))
            summary["resolved"] += 1
            summary["by_model"][model_id]["resolved"] += 1
        except RuntimeError:
            summary["already_resolved_skipped"] += 1
        except Exception:
            summary["errors"] += 1
    return summary


def _multiclass_brier(probabilities: dict[str, float], actual: str) -> float | None:
    if not probabilities:
        return None
    labels = set(probabilities)
    labels.add(actual)
    value = sum((float(probabilities.get(label, 0.0)) - float(label == actual)) ** 2 for label in labels)
    return round(value, 6)
