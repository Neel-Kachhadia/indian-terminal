"""Phase 4 ensemble oracle stress expansion (source-freeze-closure pass).

Expands the independent oracle in test_ensemble_independent_oracle.py to
adversarial/boundary cases. Reuses the SAME independently-derived formula
(never calls m10_ensemble.predict internals, never reuses its private
helpers) documented in that file's module docstring. accuracy_for_model is
monkeypatched per-case so each case can hit precise numeric boundaries
(sample_count exactly 30, weight floor 0.25, etc.) without needing to seed
thousands of DB rows per case.

Writes acceptance-evidence/source-freeze-closure/ensemble_stress_oracle.csv
as a side effect of the session so the closure pass has a durable artifact.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from ml.models import m10_ensemble as production

CLASSES = ["UP", "FLAT", "DOWN"]
EXPECTED_DIRECTION_MODELS = ("m02_direction", "m06_regime_return")
MAX_CONFIDENCE = 0.85
MINIMUM_OOS_WEIGHT_SAMPLES = 30
EVIDENCE_PATH = Path(
    r"D:\BharatTerminal\acceptance-evidence\source-freeze-closure\ensemble_stress_oracle.csv"
)
_ROWS: list[dict] = []


def _direction_output(
    prediction: str,
    probs: dict[str, float],
    *,
    is_trained: bool = True,
    calibrated: bool = True,
    schema_ok: bool = True,
    artifact_verified: bool = True,
    registry_status: str = "VALIDATED",
    reason_code: str | None = None,
) -> dict:
    out = {
        "is_trained": is_trained,
        "calibrated": calibrated,
        "feature_schema_compatible": schema_ok,
        "model_attribution": {
            "artifact_verified": artifact_verified,
            "registry_status": registry_status,
        },
        "prediction": prediction,
        "probabilities": probs,
    }
    if reason_code is not None:
        out["reason_code"] = reason_code
    return out


def _oracle(component_outputs: dict, accuracy_by_model: dict, *, anomaly_multiplier: float = 1.0) -> dict:
    vectors = []
    for model_id in EXPECTED_DIRECTION_MODELS:
        output = component_outputs.get(model_id)
        if not output or output.get("is_trained") is not True:
            continue
        if output.get("calibrated") is not True:
            continue
        if output.get("feature_schema_compatible") is not True:
            continue
        attribution = output.get("model_attribution") or {}
        if attribution.get("artifact_verified") is not True:
            continue
        if attribution.get("registry_status") != "VALIDATED":
            continue
        stats = accuracy_by_model.get(model_id, {"accuracy": None, "sample_count": 0})
        measured = stats["sample_count"] >= MINIMUM_OOS_WEIGHT_SAMPLES
        accuracy = stats["accuracy"] if measured else None
        weight = max(0.25, accuracy) if accuracy is not None else 1.0
        # Production (_direction_vectors in m10_ensemble.py) recomputes each
        # component's vote from argmax(probabilities); it never trusts a
        # component's self-reported "prediction" field for the vote used in
        # agreement. The oracle must mirror that exactly, not copy **output.
        recomputed_prediction = max(output["probabilities"], key=output["probabilities"].get)
        vectors.append(
            {
                "model_id": model_id,
                "weight": round(weight, 6),
                "probabilities": output["probabilities"],
                "prediction": recomputed_prediction,
            }
        )

    if not vectors:
        return {"status": "UNAVAILABLE", "eligible_count": 0}

    weight_sum = sum(v["weight"] for v in vectors)
    totals = {label: 0.0 for label in CLASSES}
    for v in vectors:
        for label in CLASSES:
            totals[label] += v["probabilities"].get(label, 0.0) * v["weight"]
    probabilities = {label: totals[label] / weight_sum for label in CLASSES}
    direction = max(probabilities, key=probabilities.get)
    coverage = len(vectors) / len(EXPECTED_DIRECTION_MODELS)
    raw_confidence = max(0.0, min(MAX_CONFIDENCE, max(probabilities.values())))
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


def _run_case(
    case_id: str,
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
    component_outputs: dict,
    accuracy_by_model: dict,
) -> tuple[dict, dict]:
    def fake_accuracy_for_model(model_id: str, days: int = 30) -> dict:
        return accuracy_by_model.get(model_id, {"accuracy": None, "sample_count": 0})

    monkeypatch.setattr(production, "accuracy_for_model", fake_accuracy_for_model)
    prod = production.predict(None, dict(component_outputs))
    oracle = _oracle(component_outputs, accuracy_by_model)
    return prod, oracle


def _record(case_id: str, scenario: str, prod: dict, oracle: dict, note: str = "") -> None:
    match = True
    err = 0.0
    if oracle.get("status") != prod.get("status"):
        match = False
    elif oracle["status"] == "READY_ANALYTICAL":
        if prod["composite_direction"] != oracle["direction"]:
            match = False
        for label in CLASSES:
            e = abs(prod["composite_probabilities"][label] - oracle["probabilities"][label])
            err = max(err, e)
            if e > 1e-6:
                match = False
        e = abs(prod["composite_confidence"] - oracle["confidence"])
        err = max(err, e)
        if e > 1e-6:
            match = False
        e = abs(prod["model_agreement"] - oracle["agreement"])
        err = max(err, e)
        if e > 1e-6:
            match = False
        e = abs(prod["evidence_coverage"] - oracle["coverage"])
        err = max(err, e)
        if e > 1e-6:
            match = False
        e = abs(prod["weight_sum"] - oracle["weight_sum"])
        err = max(err, e)
        if e > 1e-6:
            match = False
    _ROWS.append(
        {
            "case_id": case_id,
            "scenario": scenario,
            "oracle_status": oracle.get("status"),
            "production_status": prod.get("status"),
            "match": match,
            "max_error": round(err, 8),
            "note": note,
        }
    )
    assert match, f"{case_id} mismatch: prod={prod} oracle={oracle}"


@pytest.mark.contract
def test_case01_three_eligible_models_not_reachable() -> None:
    """Only 2 direction models exist in production (EXPECTED_DIRECTION_MODELS
    has length 2, hardcoded in m10_ensemble.py). A 3rd eligible direction
    model cannot exist under the current contract."""
    assert len(production.EXPECTED_DIRECTION_MODELS) == 2
    _ROWS.append(
        {
            "case_id": "case01",
            "scenario": "3 eligible models equal accuracy",
            "oracle_status": "N/A",
            "production_status": "N/A",
            "match": "NOT_REACHABLE_BY_CONTRACT",
            "max_error": 0.0,
            "note": "EXPECTED_DIRECTION_MODELS has exactly 2 entries in production source",
        }
    )


@pytest.mark.contract
def test_case02_unequal_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.7, "FLAT": 0.2, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("DOWN", {"UP": 0.2, "FLAT": 0.2, "DOWN": 0.6}),
        "m05_anomaly": {"is_valid": True, "confidence_multiplier": 1.0, "is_anomalous": False},
    }
    acc = {
        "m02_direction": {"accuracy": 0.8, "sample_count": 30},
        "m06_regime_return": {"accuracy": 0.2, "sample_count": 30},
    }
    prod, oracle = _run_case("case02", "unequal accuracy", monkeypatch, comp, acc)
    _record("case02", "unequal accuracy", prod, oracle)


@pytest.mark.contract
def test_case03_one_below_min_oos(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("UP", {"UP": 0.55, "FLAT": 0.25, "DOWN": 0.2}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.9, "sample_count": 30},
        "m06_regime_return": {"accuracy": 0.9, "sample_count": 29},
    }
    prod, oracle = _run_case("case03", "one model below minimum OOS sample (29<30)", monkeypatch, comp, acc)
    assert oracle["eligible_count"] == 2
    # below-threshold model must fall back to equal weight 1.0, not use its measured accuracy
    _record("case03", "one model below minimum OOS sample", prod, oracle)


@pytest.mark.contract
def test_case04_one_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": {"is_trained": False, "reason_code": "MODEL_UNAVAILABLE"},
    }
    acc = {"m02_direction": {"accuracy": 0.7, "sample_count": 40}}
    prod, oracle = _run_case("case04", "one model unavailable", monkeypatch, comp, acc)
    assert oracle["eligible_count"] == 1
    _record("case04", "one model unavailable", prod, oracle)


@pytest.mark.contract
def test_case05_one_model_ready_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """READY_PROXY does not clear is_trained/calibrated/schema flags in
    production's eligibility helper (_ineligibility_reason never inspects a
    'status' field) -- a proxy-trained model is still weighted like a
    directly-trained one. This is documented current behaviour, not a new
    claim; the oracle must reproduce it exactly for the gate to be honest."""
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": {
            **_direction_output("DOWN", {"UP": 0.3, "FLAT": 0.2, "DOWN": 0.5}),
            "status": "READY_PROXY",
            "trained_symbol": "NIFTY",
            "fallback_status": "PROXY",
        },
    }
    acc = {
        "m02_direction": {"accuracy": 0.7, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.6, "sample_count": 40},
    }
    prod, oracle = _run_case("case05", "one model READY_PROXY", monkeypatch, comp, acc)
    assert oracle["eligible_count"] == 2
    _record(
        "case05",
        "one model READY_PROXY",
        prod,
        oracle,
        note="proxy status does not gate ensemble eligibility in production (by design, unchanged)",
    )


@pytest.mark.contract
def test_case06_schema_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output(
            "DOWN", {"UP": 0.3, "FLAT": 0.2, "DOWN": 0.5}, schema_ok=False
        ),
    }
    acc = {"m02_direction": {"accuracy": 0.7, "sample_count": 40}}
    prod, oracle = _run_case("case06", "one model schema incompatible", monkeypatch, comp, acc)
    assert oracle["eligible_count"] == 1
    _record("case06", "one model schema incompatible", prod, oracle)


@pytest.mark.contract
def test_case07_one_model_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """No 'stale' concept exists in _ineligibility_reason; the closest
    production analogue is registry_status != VALIDATED (e.g. STALE)."""
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output(
            "DOWN", {"UP": 0.3, "FLAT": 0.2, "DOWN": 0.5}, registry_status="STALE"
        ),
    }
    acc = {"m02_direction": {"accuracy": 0.7, "sample_count": 40}}
    prod, oracle = _run_case("case07", "one model stale (registry_status=STALE)", monkeypatch, comp, acc)
    assert oracle["eligible_count"] == 1
    _record("case07", "one model stale", prod, oracle)


@pytest.mark.contract
def test_case08_anomaly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("UP", {"UP": 0.55, "FLAT": 0.25, "DOWN": 0.2}),
        "m05_anomaly": {"is_valid": True, "confidence_multiplier": 0.3, "is_anomalous": True},
    }
    acc = {
        "m02_direction": {"accuracy": 0.7, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.6, "sample_count": 40},
    }
    prod, _unused = _run_case("case08", "one model anomaly-gated", monkeypatch, comp, acc)
    oracle = _oracle(comp, acc, anomaly_multiplier=0.3)
    _record("case08", "one model anomaly-gated (multiplier=0.3)", prod, oracle)
    assert prod["analytical_trade_candidate"] is False


@pytest.mark.contract
def test_case09_all_models_same_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.7, "FLAT": 0.2, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("UP", {"UP": 0.65, "FLAT": 0.2, "DOWN": 0.15}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.7, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.6, "sample_count": 40},
    }
    prod, oracle = _run_case("case09", "all models same direction", monkeypatch, comp, acc)
    assert oracle["agreement"] == 1.0
    _record("case09", "all models same direction", prod, oracle)


@pytest.mark.contract
def test_case10_split_vote(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.2, "DOWN": 0.2}),
        "m06_regime_return": _direction_output("DOWN", {"UP": 0.2, "FLAT": 0.2, "DOWN": 0.6}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.7, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.3, "sample_count": 40},
    }
    prod, oracle = _run_case("case10", "split vote unequal weight", monkeypatch, comp, acc)
    _record("case10", "split vote unequal weight", prod, oracle)


@pytest.mark.contract
def test_case11_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal-weight, opposed-direction models whose aggregated mass ties
    exactly 50/50 at the composite level (each model internally consistent
    -- own prediction matches own argmax(probabilities), so this is a real
    reachable tie, not a self-contradictory fixture). Documents that argmax
    tie-break is Python max() first-max-wins in CLASSES iteration order
    (UP before FLAT before DOWN), deterministically -- not randomised."""
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.1, "DOWN": 0.3}),
        "m06_regime_return": _direction_output("DOWN", {"UP": 0.3, "FLAT": 0.1, "DOWN": 0.6}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.6, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.6, "sample_count": 40},
    }
    prod, oracle = _run_case("case11", "exact tie", monkeypatch, comp, acc)
    assert oracle["probabilities"]["UP"] == pytest.approx(oracle["probabilities"]["DOWN"])
    assert oracle["direction"] == "UP"  # first key with max value, CLASSES=[UP,FLAT,DOWN]
    assert prod["composite_direction"] == "UP"
    _record("case11", "exact tie (deterministic first-max tiebreak)", prod, oracle)


@pytest.mark.contract
def test_case12_no_eligible_models(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": {"is_trained": False, "reason_code": "INSUFFICIENT_DATA"},
        "m06_regime_return": {"is_trained": False, "reason_code": "INSUFFICIENT_DATA"},
    }
    prod, oracle = _run_case("case12", "no eligible models", monkeypatch, comp, {})
    assert oracle["status"] == "UNAVAILABLE"
    assert prod["status"] == "UNAVAILABLE"
    assert prod["reason_code"] == "NO_ELIGIBLE_DIRECTION_MODELS"
    _record("case12", "no eligible models", prod, oracle)


@pytest.mark.contract
def test_case13_confidence_clipping_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw max-class probability of 1.0 must be clipped to MAX_CONFIDENCE=0.85."""
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}),
        "m06_regime_return": _direction_output("UP", {"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.9, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.9, "sample_count": 40},
    }
    prod, oracle = _run_case("case13", "confidence clipping boundary (raw=1.0)", monkeypatch, comp, acc)
    assert oracle["confidence"] == MAX_CONFIDENCE  # coverage=1.0, multiplier=1.0
    assert prod["composite_confidence"] == MAX_CONFIDENCE
    _record("case13", "confidence clipping boundary", prod, oracle)


@pytest.mark.contract
def test_case14_probability_exactly_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("DOWN", {"UP": 0.0, "FLAT": 0.4, "DOWN": 0.6}),
        "m06_regime_return": _direction_output("DOWN", {"UP": 0.0, "FLAT": 0.3, "DOWN": 0.7}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.6, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.6, "sample_count": 40},
    }
    prod, oracle = _run_case("case14", "probability exactly 0 for a class", monkeypatch, comp, acc)
    assert prod["composite_probabilities"]["UP"] == 0.0
    _record("case14", "probability exactly 0", prod, oracle)


@pytest.mark.contract
def test_case15_probability_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}),
        "m06_regime_return": {"is_trained": False, "reason_code": "INSUFFICIENT_DATA"},
    }
    acc = {"m02_direction": {"accuracy": 0.6, "sample_count": 40}}
    prod, oracle = _run_case("case15", "probability exactly 1 for a class", monkeypatch, comp, acc)
    assert prod["composite_probabilities"]["UP"] == 1.0
    _record("case15", "probability exactly 1", prod, oracle)


@pytest.mark.contract
def test_case16_weight_floor_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """accuracy exactly 0.25 == the floor: max(0.25, 0.25) == 0.25, not clamped up."""
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("DOWN", {"UP": 0.3, "FLAT": 0.2, "DOWN": 0.5}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.25, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.1, "sample_count": 40},  # below floor -> clamped to 0.25
    }
    prod, oracle = _run_case("case16", "weight floor boundary (accuracy=0.25 and accuracy<0.25)", monkeypatch, comp, acc)
    assert oracle["weight_sum"] == pytest.approx(0.5)  # 0.25 + 0.25(clamped)
    _record("case16", "weight floor boundary", prod, oracle)


@pytest.mark.contract
def test_case17_minimum_sample_exactly_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("UP", {"UP": 0.55, "FLAT": 0.25, "DOWN": 0.2}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.9, "sample_count": 30},  # exactly threshold -> measured
        "m06_regime_return": {"accuracy": 0.9, "sample_count": 40},
    }
    prod, oracle = _run_case("case17", "minimum sample exactly threshold (30)", monkeypatch, comp, acc)
    assert oracle["weight_sum"] == pytest.approx(1.8)  # both use measured accuracy 0.9
    _record("case17", "minimum sample exactly threshold", prod, oracle)


@pytest.mark.contract
def test_case18_minimum_sample_threshold_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("UP", {"UP": 0.55, "FLAT": 0.25, "DOWN": 0.2}),
    }
    acc = {
        "m02_direction": {"accuracy": 0.9, "sample_count": 29},  # one below threshold -> equal weight 1.0
        "m06_regime_return": {"accuracy": 0.9, "sample_count": 40},
    }
    prod, oracle = _run_case("case18", "minimum sample threshold-1 (29)", monkeypatch, comp, acc)
    assert oracle["weight_sum"] == pytest.approx(1.0 + 0.9)  # m02 falls back to 1.0, m06 uses 0.9
    _record("case18", "minimum sample threshold minus 1", prod, oracle)


@pytest.mark.contract
def test_case19_anomaly_multiplier_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": _direction_output("UP", {"UP": 0.55, "FLAT": 0.25, "DOWN": 0.2}),
        "m05_anomaly": {"is_valid": True, "confidence_multiplier": 0.0, "is_anomalous": True},
    }
    acc = {
        "m02_direction": {"accuracy": 0.7, "sample_count": 40},
        "m06_regime_return": {"accuracy": 0.6, "sample_count": 40},
    }
    prod, _unused = _run_case("case19", "anomaly multiplier boundary (0.0)", monkeypatch, comp, acc)
    oracle = _oracle(comp, acc, anomaly_multiplier=0.0)
    _record("case19", "anomaly multiplier boundary=0.0", prod, oracle)
    assert prod["composite_confidence"] == 0.0


@pytest.mark.contract
def test_case20_coverage_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = {
        "m02_direction": _direction_output("UP", {"UP": 0.6, "FLAT": 0.3, "DOWN": 0.1}),
        "m06_regime_return": {"is_trained": False, "reason_code": "MODEL_UNAVAILABLE"},
    }
    acc = {"m02_direction": {"accuracy": 0.7, "sample_count": 40}}
    prod, oracle = _run_case("case20", "coverage reduction (1/2 eligible)", monkeypatch, comp, acc)
    assert oracle["coverage"] == 0.5
    assert prod["evidence_coverage"] == 0.5
    _record("case20", "coverage reduction", prod, oracle)


def test_zz_write_evidence_csv() -> None:
    """Runs last alphabetically-ish (zz) so all case rows are collected first.
    pytest collects/executes in file order by default so this depends on
    being placed after the case tests; that ordering is verified by the
    non-empty assertion below rather than assumed silently."""
    assert len(_ROWS) >= 20, f"expected >=20 recorded stress cases, got {len(_ROWS)}"
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "scenario", "oracle_status", "production_status", "match", "max_error", "note"],
        )
        writer.writeheader()
        for row in _ROWS:
            writer.writerow(row)
