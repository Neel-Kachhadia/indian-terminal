"""Phase 9 determinism (source-freeze-closure pass).

Tests inference-time determinism at the same layer production actually
executes at prediction time: predict_classifier_artifact(artifact, context)
for m01/m02/m06 (each using that model's real production estimator class
and hyperparameters from its own _estimator() factory -- not a stub), and
OHLCForecaster.predict for m21 (analytical, no trained artifact).

This deliberately tests INFERENCE determinism (identical artifact + identical
input -> identical output), not whether a from-scratch tiny synthetic fixture
would independently clear the production skill-floor training gate -- that
gate is a legitimate, separate safety check (see test_model_safety_exhaustive
and ml/models/common.py's chance-baseline rejection) and is intentionally not
being fought here.

Writes acceptance-evidence/source-freeze-closure/determinism.json.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import MLContext
from ml.models.common import predict_classifier_artifact
from ml.models.m21_ohlc_forecast import OHLCForecaster

EVIDENCE_PATH = Path(r"D:\BharatTerminal\acceptance-evidence\source-freeze-closure\determinism.json")
_RESULTS: list[dict] = []
CLASSES = ["UP", "FLAT", "DOWN"]
FEATURES = ["feature_a", "feature_b", "feature_c", "feature_d"]


def _context() -> MLContext:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-21")],
            "feature_a": [1.53],
            "feature_b": [-0.42],
            "feature_c": [2.71],
            "feature_d": [0.05],
        }
    )
    return MLContext(
        symbol="RELIANCE",
        signal_payload={},
        current_features={},
        feature_frame=frame,
        ohlcv=pd.DataFrame(),
        iv_history=pd.DataFrame(),
        bif_history=pd.DataFrame(),
        fo_eod_rows=0,
        current_spot=100.0,
        spot_provenance={},
        market_data_eligibility=None,  # unused by predict_classifier_artifact
        decision_requested_at=dt.datetime(2026, 8, 21, 11, tzinfo=dt.timezone.utc),
    )


def _synthetic_training_data(seed: int, rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, len(FEATURES)))
    # deterministic-enough decision boundary so .fit() converges to a fixed
    # tree structure; the point under test is inference reproducibility of
    # THAT fixed structure, not training accuracy.
    score = x[:, 0] * 1.5 - x[:, 1] * 0.8 + x[:, 2] * 0.3
    y = np.where(score > 0.5, 0, np.where(score < -0.5, 2, 1))
    return x, y


def _artifact(model_id: str, estimator) -> dict:
    return {
        "model_id": model_id,
        "model_version": "20260101T000000Z-detcheck",
        "symbol": "RELIANCE",
        "universe": "RELIANCE",
        "model": estimator,
        "classes": CLASSES,
        "feature_columns": FEATURES,
        "feature_medians": {name: 0.0 for name in FEATURES},
        "trained_at": "2026-01-01T00:00:00+00:00",
        "training_data_horizon": "2025-01-01..2025-12-31",
        "feature_schema_version": "features-v1:deadbeef",
        "code_version": "1",
        "release_version": "1",
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
        "train_rows": 400,
        "calibrated": True,
        "artifact_hash": "hash",
    }


def _deep_equal(a: object, b: object, path: str, mismatches: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            _deep_equal(a.get(key), b.get(key), f"{path}.{key}", mismatches)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            mismatches.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _deep_equal(x, y, f"{path}[{i}]", mismatches)
    else:
        if a != b:
            mismatches.append(f"{path}: {a!r} != {b!r}")


def _assert_identical(label: str, first: dict, second: dict) -> None:
    mismatches: list[str] = []
    _deep_equal(first, second, label, mismatches)
    passed = len(mismatches) == 0
    _RESULTS.append({"model": label, "runs": 5, "byte_float_identical": passed, "mismatches": mismatches[:10]})
    assert passed, f"{label} non-deterministic across identical repeated inference calls: {mismatches[:5]}"


def _run_inference_determinism(label: str, estimator) -> None:
    context = _context()
    artifact = _artifact(label, estimator)
    outputs = [predict_classifier_artifact(artifact, context) for _ in range(5)]
    for run in outputs[1:]:
        _assert_identical(label, outputs[0], run)


@pytest.mark.contract
def test_m01_vol_regime_inference_deterministic() -> None:
    from xgboost import XGBClassifier

    x, y = _synthetic_training_data(seed=1)
    model = XGBClassifier(
        n_estimators=90, max_depth=3, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, random_state=42, n_jobs=1,
    )
    model.fit(x, y)
    _run_inference_determinism("m01_vol_regime", model)


@pytest.mark.contract
def test_m02_direction_inference_deterministic() -> None:
    from lightgbm import LGBMClassifier

    x, y = _synthetic_training_data(seed=2)
    model = LGBMClassifier(
        n_estimators=120, learning_rate=0.035, num_leaves=31, subsample=0.85,
        colsample_bytree=0.85, random_state=42, n_jobs=1, verbosity=-1,
    )
    model.fit(x, y)
    _run_inference_determinism("m02_direction", model)


@pytest.mark.contract
def test_m06_regime_return_inference_deterministic() -> None:
    from catboost import CatBoostClassifier

    x, y = _synthetic_training_data(seed=3)
    model = CatBoostClassifier(
        iterations=120, depth=4, learning_rate=0.04, loss_function="MultiClass",
        random_seed=42, verbose=False, thread_count=1,
    )
    model.fit(x, y)
    _run_inference_determinism("m06_regime_return", model)


@pytest.mark.contract
def test_m21_ohlc_forecast_deterministic() -> None:
    rows = 120
    dates = pd.bdate_range("2026-01-02", periods=rows)
    rng = np.random.default_rng(99)
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=rows))
    open_ = close - rng.normal(0, 0.3, size=rows)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.5, 0.2, size=rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.5, 0.2, size=rows))
    ohlcv = pd.DataFrame(
        {"date": dates.date, "open": open_, "high": high, "low": low, "close": close, "volume": 1_000}
    )
    forecaster = OHLCForecaster()
    outputs = [
        forecaster.predict(
            "NIFTY", ohlcv, float(close[-1]), atm_iv=14.5, vix=13.2,
            log_forecast=False, include_accuracy=False,
        )
        for _ in range(5)
    ]
    for run in outputs[1:]:
        _assert_identical("m21_ohlc", outputs[0], run)


def test_zz_write_determinism_evidence() -> None:
    assert len(_RESULTS) >= 4, f"expected at least 4 recorded determinism results, got {len(_RESULTS)}"
    failures = sum(1 for r in _RESULTS if not r["byte_float_identical"])
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps({"cases": len(_RESULTS), "failures": failures, "details": _RESULTS}, indent=2),
        encoding="utf-8",
    )
