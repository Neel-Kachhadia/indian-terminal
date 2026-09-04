"""Independent oracle for m10_ensemble's reliability-weighted aggregation.

This file re-derives the composite direction/confidence from the exact
formula documented in compute/ml/models/m10_ensemble.py (traced from source,
2026-09-03: weight = max(0.25, 30d-OOS-accuracy) if sample_count >= 30 else
1.0; probabilities = normalised weighted average; confidence = clipped max
probability (cap 0.85) x coverage x anomaly multiplier; agreement = weighted
vote share for the winning direction) WITHOUT calling m10_ensemble.predict
or any of its private helpers. It only calls the shared, already-tested
accuracy reader (ml.monitor.accuracy_for_model) to get real measured
accuracy from genuinely resolved isolated-DB predictions -- not a guess.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest

from db import duckdb_client
from db.duckdb_client import _writer_transaction_scope
from db.writer_lease import release_writer_lease, require_writer_lease

MAX_CONFIDENCE = 0.85
CLASSES = ["UP", "FLAT", "DOWN"]
EXPECTED_DIRECTION_MODELS = ("m02_direction", "m06_regime_return")
MINIMUM_OOS_WEIGHT_SAMPLES = 30


def _seed_resolved_predictions(
    connection: duckdb.DuckDBPyConnection, model_id: str, *, n_correct: int, n_wrong: int
) -> None:
    import uuid

    now = dt.datetime.now(dt.timezone.utc)
    for i in range(n_correct + n_wrong):
        correct = 1 if i < n_correct else 0
        connection.execute(
            """
            INSERT INTO ml_predictions
                (id, model_id, predicted_at, features_hash, prediction, probabilities,
                 confidence, symbol, data_as_of, outcome, outcome_at, correct, brier_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(uuid.uuid4()), model_id, now - dt.timedelta(days=i % 20 + 1), "hash",
                "UP", '{"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}', 0.6,
                "TESTSYM", (now - dt.timedelta(days=i % 20 + 1)).date().isoformat(),
                "UP" if correct else "DOWN", now, correct, 0.15,
            ],
        )


def _direction_output(prediction: str, probs: dict[str, float]) -> dict:
    return {
        "is_trained": True,
        "calibrated": True,
        "feature_schema_compatible": True,
        "model_attribution": {"artifact_verified": True, "registry_status": "VALIDATED"},
        "prediction": prediction,
        "probabilities": probs,
    }


def _normalise(raw: list[float]) -> dict[str, float]:
    total = sum(raw)
    if total <= 0:
        return {label: 0.0 for label in CLASSES}
    return {label: value / total for label, value in zip(CLASSES, raw)}


def _oracle(
    component_outputs: dict,
    accuracy_by_model: dict[str, dict],
    *,
    anomaly_multiplier: float = 1.0,
) -> dict:
    """Independent re-derivation -- see module docstring for the traced formula."""
    vectors = []
    for model_id in EXPECTED_DIRECTION_MODELS:
        output = component_outputs.get(model_id)
        if not output or output.get("is_trained") is not True:
            continue
        stats = accuracy_by_model[model_id]
        measured = stats["sample_count"] >= MINIMUM_OOS_WEIGHT_SAMPLES
        accuracy = stats["accuracy"] if measured else None
        weight = max(0.25, accuracy) if accuracy is not None else 1.0
        vectors.append({"model_id": model_id, "weight": weight, **output})

    if not vectors:
        return {"status": "UNAVAILABLE"}

    weight_sum = sum(v["weight"] for v in vectors)
    totals = {label: 0.0 for label in CLASSES}
    for v in vectors:
        for label in CLASSES:
            totals[label] += v["probabilities"][label] * v["weight"]
    probabilities = _normalise([totals[label] for label in CLASSES])
    probabilities = dict(zip(CLASSES, [totals[label] / weight_sum for label in CLASSES]))
    direction = max(probabilities, key=probabilities.get)
    coverage = len(vectors) / len(EXPECTED_DIRECTION_MODELS)
    raw_confidence = round(max(0.0, min(MAX_CONFIDENCE, max(probabilities.values()) * 1.0)), 6)
    confidence = round(raw_confidence * coverage * anomaly_multiplier, 6)
    supporting = sum(v["weight"] for v in vectors if v["prediction"] == direction)
    agreement = round(max(0.0, min(1.0, supporting / weight_sum)), 6)
    return {
        "status": "READY_ANALYTICAL",
        "direction": direction,
        "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
        "confidence": confidence,
        "agreement": agreement,
        "coverage": round(coverage, 6),
        "weight_sum": round(weight_sum, 6),
        "eligible_count": len(vectors),
    }


@pytest.mark.contract
def test_oracle_matches_production_ensemble_with_measured_unequal_accuracy(
    isolated_app_data: Path,
) -> None:
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.models.m10_ensemble import predict as m10_predict
        from ml.monitor import accuracy_for_model, ensure_ml_tables

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                ensure_ml_tables()
                # m02: 24/30 correct (0.8 accuracy, >= floor 0.25 unchanged).
                _seed_resolved_predictions(connection, "m02_direction", n_correct=24, n_wrong=6)
                # m06: 6/30 correct (0.2 accuracy, clamped to weight floor 0.25).
                _seed_resolved_predictions(connection, "m06_regime_return", n_correct=6, n_wrong=24)

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                accuracy_by_model = {
                    model_id: accuracy_for_model(model_id) for model_id in EXPECTED_DIRECTION_MODELS
                }
                for model_id in EXPECTED_DIRECTION_MODELS:
                    assert accuracy_by_model[model_id]["sample_count"] == 30
                assert accuracy_by_model["m02_direction"]["accuracy"] == pytest.approx(0.8)
                assert accuracy_by_model["m06_regime_return"]["accuracy"] == pytest.approx(0.2)

                component_outputs = {
                    "m02_direction": _direction_output("UP", {"UP": 0.7, "FLAT": 0.2, "DOWN": 0.1}),
                    "m06_regime_return": _direction_output("DOWN", {"UP": 0.2, "FLAT": 0.2, "DOWN": 0.6}),
                    "m05_anomaly": {"is_valid": True, "confidence_multiplier": 1.0, "is_anomalous": False},
                    "m09_rl_sizer": {"recommended_size": 0.0},
                    "m03_strategy": {},
                }
                production = m10_predict(None, component_outputs)
                oracle = _oracle(component_outputs, accuracy_by_model)

                assert oracle["status"] == "READY_ANALYTICAL"
                assert production["status"] == "READY_ANALYTICAL"
                assert production["composite_direction"] == oracle["direction"]
                assert production["eligible_component_count"] == oracle["eligible_count"]
                assert production["evidence_coverage"] == pytest.approx(oracle["coverage"], abs=1e-6)
                assert production["weight_sum"] == pytest.approx(oracle["weight_sum"], abs=1e-6)
                assert production["model_agreement"] == pytest.approx(oracle["agreement"], abs=1e-6)
                assert production["composite_confidence"] == pytest.approx(oracle["confidence"], abs=1e-6)
                for label in CLASSES:
                    assert production["composite_probabilities"][label] == pytest.approx(
                        oracle["probabilities"][label], abs=1e-6
                    )
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_oracle_matches_production_when_only_one_model_eligible(isolated_app_data: Path) -> None:
    """Real invariant this proves: an unavailable model is EXCLUDED from
    numerator and denominator, never treated as a neutral/0.5 vote."""
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.models.m10_ensemble import predict as m10_predict
        from ml.monitor import accuracy_for_model, ensure_ml_tables

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                ensure_ml_tables()
                _seed_resolved_predictions(connection, "m02_direction", n_correct=15, n_wrong=15)

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                accuracy_by_model = {
                    model_id: accuracy_for_model(model_id) for model_id in EXPECTED_DIRECTION_MODELS
                }
                component_outputs = {
                    "m02_direction": _direction_output("UP", {"UP": 0.7, "FLAT": 0.2, "DOWN": 0.1}),
                    "m06_regime_return": {"is_trained": False, "reason_code": "INSUFFICIENT_DATA"},
                    "m05_anomaly": {"is_valid": True, "confidence_multiplier": 1.0, "is_anomalous": False},
                    "m09_rl_sizer": {"recommended_size": 0.0},
                    "m03_strategy": {},
                }
                production = m10_predict(None, component_outputs)
                oracle = _oracle(component_outputs, accuracy_by_model)

                assert oracle["eligible_count"] == 1
                assert production["eligible_component_count"] == 1
                assert production["evidence_coverage"] == pytest.approx(0.5, abs=1e-6)
                assert production["composite_direction"] == "UP"
                # Excluded model contributes nothing -- not a 0.5-weighted vote.
                assert production["weight_sum"] == pytest.approx(oracle["weight_sum"], abs=1e-6)
                assert production["composite_confidence"] == pytest.approx(oracle["confidence"], abs=1e-6)
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_oracle_matches_production_when_zero_models_eligible(isolated_app_data: Path) -> None:
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.models.m10_ensemble import predict as m10_predict

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                component_outputs = {
                    "m02_direction": {"is_trained": False, "reason_code": "INSUFFICIENT_DATA"},
                    "m06_regime_return": {"is_trained": False, "reason_code": "INSUFFICIENT_DATA"},
                    "m05_anomaly": {},
                    "m09_rl_sizer": {},
                    "m03_strategy": {},
                }
                production = m10_predict(None, component_outputs)
                assert production["status"] == "UNAVAILABLE"
                assert production["reason_code"] == "NO_ELIGIBLE_DIRECTION_MODELS"
                assert production["composite_direction"] is None
                assert production["eligible_component_count"] == 0
    finally:
        release_writer_lease()
