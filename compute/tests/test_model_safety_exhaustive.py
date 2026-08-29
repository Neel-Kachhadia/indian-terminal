from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ml.feature_engineering import MLContext
from ml.models.common import predict_classifier_artifact


class _StubClassifier:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, x_current: np.ndarray) -> np.ndarray:
        return np.array([[0.2, 0.5, 0.3]])


def _context(feature_frame: pd.DataFrame) -> MLContext:
    return MLContext(
        symbol="RELIANCE",
        signal_payload={},
        current_features={},
        feature_frame=feature_frame,
        ohlcv=pd.DataFrame(),
        iv_history=pd.DataFrame(),
        bif_history=pd.DataFrame(),
        fo_eod_rows=0,
        current_spot=100.0,
        spot_provenance={},
        market_data_eligibility=None,  # unused by predict_classifier_artifact
        decision_requested_at=dt.datetime.now(dt.timezone.utc),
    )


def _artifact() -> dict:
    return {
        "model_id": "m02_direction",
        "model_version": "20260101T000000Z-abc123",
        "symbol": "RELIANCE",
        "universe": "RELIANCE",
        "model": _StubClassifier(),
        "classes": ["UP", "FLAT", "DOWN"],
        "feature_columns": ["feature_a", "feature_b", "feature_c"],
        "feature_medians": {"feature_a": 1.0, "feature_b": 2.0, "feature_c": 3.0},
        "trained_at": "2026-01-01T00:00:00+00:00",
        "training_data_horizon": "2025-01-01..2025-12-31",
        "feature_schema_version": "features-v1:deadbeef",
        "code_version": "1",
        "release_version": "1",
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
        "train_rows": 500,
        "calibrated": True,
        "artifact_hash": "hash",
    }


def test_feature_schema_compatible_true_when_all_trained_features_present() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "feature_a": [1.5],
            "feature_b": [2.5],
            "feature_c": [3.5],
        }
    )
    output = predict_classifier_artifact(_artifact(), _context(frame))
    assert output["feature_schema_compatible"] is True
    assert output["model_attribution"]["feature_schema_compatible"] is True


def test_feature_schema_compatible_false_when_a_trained_feature_is_missing_live() -> None:
    """Discriminating case: before this fix, model_attribution() hardcoded
    feature_schema_compatible=True unconditionally, so recommendation_policy's
    fail-closed actionability gate could never actually catch a live feature
    vector that has drifted from what the model was trained on -- the model
    would silently predict off a median-imputed value for the missing column
    and still be reported as schema-compatible."""
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "feature_a": [1.5],
            "feature_b": [2.5],
            # feature_c intentionally absent -- simulates a live feature-engineering
            # drift or an indicator that failed to compute for this bar.
        }
    )
    output = predict_classifier_artifact(_artifact(), _context(frame))
    assert output["feature_schema_compatible"] is False
    assert output["model_attribution"]["feature_schema_compatible"] is False
    # The model still produces a diagnostic prediction (imputed via median) --
    # it is the downstream actionability gate's job to refuse ACTIONABLE status,
    # not this function's job to hide the prediction entirely.
    assert output["prediction"] in {"UP", "FLAT", "DOWN"}


def test_feature_schema_compatible_false_when_live_feature_is_non_finite() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "feature_a": [1.5],
            "feature_b": [np.inf],
            "feature_c": [3.5],
        }
    )
    output = predict_classifier_artifact(_artifact(), _context(frame))
    assert output["feature_schema_compatible"] is False


def test_recommendation_gate_rejects_actionability_on_schema_incompatibility() -> None:
    from ml.recommendation_policy import RecommendationEvidence, recommendation_eligibility

    frame_missing = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "feature_a": [1.5],
            "feature_b": [2.5],
        }
    )
    m02_output = predict_classifier_artifact(_artifact(), _context(frame_missing))
    ensemble = {
        "model_attribution": {
            "artifact_hash": "hash",
            "model_version": "v1",
            "training_data_horizon": "h",
            "feature_schema_version": "v1",
            "artifact_verified": True,
            "registry_status": "VALIDATED",
            "feature_schema_compatible": True,
        },
        "is_trained": True,
        "calibrated": True,
    }
    components = {"m02_direction": m02_output}
    evidence = RecommendationEvidence(
        market_data_quality="OBSERVED_VERIFIED",
        market_data_freshness="FRESH",
        market_data_provenance="NSE_EQUITY_BHAVCOPY",
        prediction_as_of="2026-01-01T00:00:00Z",
        requested_decision_at="2026-01-01T00:01:00Z",
        fallback_state="NONE",
        input_feature_schema="v1",
    )
    eligibility = recommendation_eligibility(ensemble, components, evidence=evidence)
    assert eligibility.eligible is False
    assert any("feature schema incompatible" in reason for reason in eligibility.reasons)
