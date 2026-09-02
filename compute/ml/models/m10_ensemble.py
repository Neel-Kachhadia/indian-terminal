from __future__ import annotations

import math
from typing import Any

from ml.calibrator import clipped_confidence, normalise_probabilities
from ml.feature_engineering import MLContext
from ml.monitor import accuracy_for_model


MODEL_ID = "m10_ensemble"
CLASSES = ["UP", "FLAT", "DOWN"]
EXPECTED_DIRECTION_MODELS = ("m02_direction", "m06_regime_return")
MINIMUM_OOS_WEIGHT_SAMPLES = 30


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del context, force
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "status": "UNTRAINED",
        "reason_code": "STACKING_HISTORY_UNAVAILABLE",
        "ensemble_method": "reliability_weighted_average",
        "reason": "stacking waits for resolved out-of-sample prediction history",
    }


def predict(context: MLContext, component_outputs: dict[str, Any]) -> dict[str, Any]:
    del context
    vectors, excluded = _direction_vectors(component_outputs)
    anomaly = component_outputs.get("m05_anomaly") or {}
    anomaly_valid = anomaly.get("is_valid") is True
    multiplier = _finite(anomaly.get("confidence_multiplier")) if anomaly_valid else 1.0
    multiplier = multiplier if multiplier is not None else 1.0

    if not vectors:
        output = _unavailable(component_outputs, excluded)
        output["computed_with_confidence_multiplier"] = multiplier
        output["accuracy_tracking"] = _accuracy_tracking()
        output.update(_legacy_contract(output, component_outputs))
        return output

    totals = {label: 0.0 for label in CLASSES}
    weight_sum = sum(float(item["weight"]) for item in vectors)
    contributions: list[dict[str, Any]] = []
    for item in vectors:
        weight = float(item["weight"])
        weighted = {
            label: float(item["probabilities"][label]) * weight
            for label in CLASSES
        }
        for label in CLASSES:
            totals[label] += weighted[label]
        contributions.append({**item, "weighted_contribution": weighted})

    probabilities = normalise_probabilities(
        CLASSES,
        [totals[label] / weight_sum for label in CLASSES],
    )
    direction = max(probabilities, key=probabilities.get)
    coverage = len(vectors) / len(EXPECTED_DIRECTION_MODELS)
    raw_confidence = clipped_confidence(probabilities, 1.0)
    confidence = round(raw_confidence * coverage * multiplier, 6)
    agreement = _agreement_score(direction, vectors)
    sizer = component_outputs.get("m09_rl_sizer") or {}
    strategy = (component_outputs.get("m03_strategy") or {}).get("strategy")
    size = _finite(sizer.get("recommended_size"))
    analytical_trade_candidate = bool(
        not (anomaly_valid and anomaly.get("is_anomalous") is True)
        and direction != "FLAT"
        and agreement >= 0.5
        and size is not None
        and size > 0.0
    )
    accuracy = _accuracy_tracking()
    output = {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "READY_ANALYTICAL",
        "reason_code": None,
        "composite_direction": direction,
        "prediction": direction,
        "composite_probabilities": probabilities,
        "probabilities": probabilities,
        "composite_confidence": confidence,
        "confidence": confidence,
        "model_agreement": round(agreement, 6),
        "ensemble_method": "reliability_weighted_average",
        "component_outputs": component_outputs,
        "eligible_component_count": len(vectors),
        "expected_component_count": len(EXPECTED_DIRECTION_MODELS),
        "evidence_coverage": round(coverage, 6),
        "weight_sum": round(weight_sum, 6),
        "component_contributions": contributions,
        "excluded_components": excluded,
        "agreement_formula": "weighted hard-vote share supporting the composite direction",
        "confidence_formula": (
            "max aggregated calibrated class probability capped at 0.85 "
            "x eligible-model coverage x valid anomaly multiplier"
        ),
        "analytical_trade_candidate": analytical_trade_candidate,
        "analytical_strategy_candidate": strategy,
        "analytical_reasoning": _reasoning(
            direction,
            probabilities,
            component_outputs,
            analytical_trade_candidate,
        ),
        "accuracy_tracking": accuracy,
        "computed_with_confidence_multiplier": multiplier,
    }
    output.update(_legacy_contract(output, component_outputs))
    return output


def _unavailable(
    component_outputs: dict[str, Any],
    excluded: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": False,
        "status": "UNAVAILABLE",
        "reason_code": "NO_ELIGIBLE_DIRECTION_MODELS",
        "reason": "0/2 calibrated direction models are inference-eligible",
        "composite_direction": None,
        "prediction": None,
        "composite_probabilities": {},
        "probabilities": {},
        "composite_confidence": None,
        "confidence": None,
        "model_agreement": None,
        "ensemble_method": "reliability_weighted_average",
        "component_outputs": component_outputs,
        "eligible_component_count": 0,
        "expected_component_count": len(EXPECTED_DIRECTION_MODELS),
        "evidence_coverage": 0.0,
        "weight_sum": 0.0,
        "component_contributions": [],
        "excluded_components": excluded,
        "agreement_formula": "weighted hard-vote share supporting the composite direction",
        "confidence_formula": (
            "max aggregated calibrated class probability capped at 0.85 "
            "x eligible-model coverage x valid anomaly multiplier"
        ),
        "analytical_trade_candidate": False,
        "analytical_strategy_candidate": None,
        "analytical_reasoning": [
            "No calibrated, schema-compatible direction model is available."
        ],
    }


def _direction_vectors(
    component_outputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    vectors: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for model_id in EXPECTED_DIRECTION_MODELS:
        output = component_outputs.get(model_id)
        reason = _ineligibility_reason(output)
        if reason is not None:
            excluded.append({"model_id": model_id, "reason": reason})
            continue
        assert isinstance(output, dict)
        raw = output["probabilities"]
        probabilities = normalise_probabilities(
            CLASSES,
            [float(raw.get(label, 0.0)) for label in CLASSES],
        )
        stats = accuracy_for_model(model_id)
        measured = stats.get("sample_count", 0) >= MINIMUM_OOS_WEIGHT_SAMPLES
        accuracy = _finite(stats.get("accuracy")) if measured else None
        weight = max(0.25, accuracy) if accuracy is not None else 1.0
        vectors.append(
            {
                "model_id": model_id,
                "prediction": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
                "weight": round(weight, 6),
                "weight_source": (
                    f"30d_oos_accuracy_n={stats.get('sample_count', 0)}"
                    if accuracy is not None
                    else "equal_weight_insufficient_oos_history"
                ),
            }
        )
    return vectors, excluded


def _ineligibility_reason(output: Any) -> str | None:
    if not isinstance(output, dict) or not output:
        return "output missing"
    if output.get("is_trained") is not True:
        return str(output.get("reason_code") or output.get("reason") or "model not trained")
    if output.get("calibrated") is not True:
        return "calibration unavailable"
    if output.get("feature_schema_compatible") is not True:
        return "feature schema incompatible"
    attribution = output.get("model_attribution")
    if isinstance(attribution, dict):
        if attribution.get("artifact_verified") is not True:
            return "artifact hash not verified"
        if attribution.get("registry_status") != "VALIDATED":
            return "artifact registry status not validated"
    probabilities = output.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return "probabilities unavailable"
    values = [_finite(probabilities.get(label)) for label in CLASSES]
    if any(value is None or value < 0 for value in values):
        return "probabilities non-finite or negative"
    if sum(float(value) for value in values if value is not None) <= 0:
        return "probability mass unavailable"
    return None


def _agreement_score(direction: str, vectors: list[dict[str, Any]]) -> float:
    weight_sum = sum(float(item["weight"]) for item in vectors)
    if weight_sum <= 0:
        return 0.0
    supporting = sum(
        float(item["weight"])
        for item in vectors
        if item["prediction"] == direction
    )
    return max(0.0, min(1.0, supporting / weight_sum))


def _reasoning(
    direction: str,
    probabilities: dict[str, float],
    component_outputs: dict[str, Any],
    trade: bool,
) -> list[str]:
    reasons: list[str] = []
    anomaly = component_outputs.get("m05_anomaly") or {}
    if anomaly.get("is_valid") is True and anomaly.get("is_anomalous") is True:
        reasons.append("Anomaly circuit breaker active; suppressing confidence.")
    reasons.append(
        f"Composite direction {direction} with {probabilities.get(direction, 0.0) * 100:.0f}% aggregated probability."
    )
    if not trade:
        reasons.append("Analytical trade candidate fails agreement, sizing, or anomaly criteria.")
    return reasons[:3]


def _accuracy_tracking() -> dict[str, Any]:
    model_ids = [
        "m01_vol_regime",
        "m02_direction",
        "m04_iv_reversion",
        "m06_regime_return",
        "m10_ensemble",
    ]
    stats = {model_id: accuracy_for_model(model_id) for model_id in model_ids}
    resolved = {
        model_id: item["accuracy"]
        for model_id, item in stats.items()
        if item["accuracy"] is not None
    }
    best = max(resolved, key=resolved.get) if resolved else None
    worst = min(resolved, key=resolved.get) if resolved else None
    ensemble_7d = accuracy_for_model("m10_ensemble", 7)
    return {
        "ensemble_accuracy_30d": stats["m10_ensemble"]["accuracy"],
        "ensemble_accuracy_7d": ensemble_7d["accuracy"],
        "best_model_30d": best,
        "worst_model_30d": worst,
        "models": {model_id: item["accuracy"] for model_id, item in stats.items()},
        "model_details": stats,
        "status": (
            "READY"
            if stats["m10_ensemble"]["sample_count"] >= MINIMUM_OOS_WEIGHT_SAMPLES
            else "INSUFFICIENT_OOS_HISTORY"
        ),
        "required_sample_count": MINIMUM_OOS_WEIGHT_SAMPLES,
        "actual_sample_count": stats["m10_ensemble"]["sample_count"],
    }


def _legacy_contract(output: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    m1 = components.get("m01_vol_regime", {})
    m2 = components.get("m02_direction", {})
    m3 = components.get("m03_strategy", {})
    m4 = components.get("m04_iv_reversion", {})
    probs = output.get("composite_probabilities") or {}
    return {
        "vol_regime": {
            "prediction": (m1.get("regime") or m1.get("prediction") or None),
            "probabilities": m1.get("probabilities", {}),
            "confidence": m1.get("confidence"),
        },
        "direction": {
            "prediction": (m2.get("direction") or m2.get("prediction") or None),
            "probabilities": m2.get("probabilities", {}),
            "confidence": m2.get("confidence"),
        },
        "strategy_selector": {
            "prediction": m3.get("strategy"),
            "confidence": m3.get("confidence"),
        },
        "iv_direction": {
            "prediction": m4.get("iv_direction"),
            "confidence": m4.get("confidence"),
        },
        "ensemble_composite": {
            "bearish_score": (
                round(float(probs["DOWN"]) * 100.0, 3)
                if "DOWN" in probs
                else None
            ),
            "regime_warning": bool(
                (components.get("m05_anomaly") or {}).get("is_valid") is True
                and (components.get("m05_anomaly") or {}).get("is_anomalous") is True
            ),
            "top_signals": [],
        },
        "model_accuracy_30d": output.get("accuracy_tracking", {}).get("models", {}),
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
