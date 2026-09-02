from __future__ import annotations

import math
from typing import Any

from ml.recommendation_policy import (
    RecommendationEvidence,
    actionable_state,
    model_output_eligibility,
)


VRC_SCORE = {
    "COMPRESSION": 70.0,
    "EARLY EXPANSION": 65.0,
    "EXPANSION": 55.0,
    "LATE EXPANSION": 45.0,
    "PEAK": 30.0,
    "CONTRACTION": 35.0,
    "LATE CONTRACTION": 50.0,
}

NFVE_SCORE = {
    "UNDERVALUED": 70.0,
    "FAIRLY_VALUED": 50.0,
    "OVERVALUED": 30.0,
}

WEIGHTS = {
    "bms": 0.25,
    "ngi": 0.15,
    "iad": 0.20,
    "vrc": 0.15,
    "nfve": 0.10,
    "ml": 0.15,
}
MINIMUM_COMPONENTS = 2
MINIMUM_WEIGHT_COVERAGE = 0.35


def compute_cds(
    bms: dict[str, Any],
    ngi: dict[str, Any],
    iad: dict[str, Any],
    vrc: dict[str, Any],
    nfve: dict[str, Any],
    ml_prediction: dict[str, Any],
    *,
    decision_evidence: RecommendationEvidence | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Composite score over observed, valid components only.

    Missing components are excluded and weights are renormalized.  Confidence
    is dispersion agreement multiplied by original-weight coverage, so sparse
    neutral inputs cannot produce a high-confidence CDS.
    """
    model_eligibility = model_output_eligibility(
        ml_prediction,
        evidence=decision_evidence,
    )
    candidates = {
        "bms": _signed_component(bms, "bms"),
        "ngi": _signed_component(ngi, "ngi"),
        "iad": _signed_component(iad, "iad_score"),
        "vrc": _vrc_component(vrc),
        "nfve": _nfve_component(nfve),
        "ml": _ml_component(ml_prediction, model_eligibility.eligible),
    }
    available = {
        name: score for name, score in candidates.items() if score is not None
    }
    coverage = sum(WEIGHTS[name] for name in available) / sum(WEIGHTS.values())
    analytical_valid = (
        len(available) >= MINIMUM_COMPONENTS
        and coverage >= MINIMUM_WEIGHT_COVERAGE
    )

    cds = None
    confidence = None
    label = None
    proposed_action = "FLAT"
    strength = "NONE"
    if analytical_valid:
        weight_total = sum(WEIGHTS[name] for name in available)
        cds = sum(available[name] * WEIGHTS[name] for name in available) / weight_total
        anomaly_score, anomaly_flag, anomaly_valid = _anomaly(ml_prediction)
        anomaly_gate = anomaly_valid and (
            anomaly_flag is True
            or (anomaly_score is not None and anomaly_score > 0.7)
        )
        if anomaly_gate:
            cds = 50.0 + (cds - 50.0) * 0.2
        confidence = _confidence(list(available.values()), coverage)
        label, proposed_action, strength = _decision(cds)
    else:
        anomaly_gate = False

    recommendation_state = actionable_state(model_eligibility, proposed_action)
    if recommendation_state != "ACTIONABLE":
        proposed_action = "UNVALIDATED" if recommendation_state == "UNVALIDATED" else "FLAT"
        strength = "NONE"

    eligibility_payload = model_eligibility.to_dict()
    reasons = {
        "bms": _component_reason(bms, candidates["bms"]),
        "ngi": _component_reason(ngi, candidates["ngi"]),
        "iad": _component_reason(iad, candidates["iad"]),
        "vrc": _component_reason(vrc, candidates["vrc"]),
        "nfve": _component_reason(nfve, candidates["nfve"]),
        "ml": None if candidates["ml"] is not None else "MODEL_OUTPUT_INELIGIBLE",
    }
    return {
        "cds": round(_clamp(cds, 0.0, 100.0), 3) if cds is not None else None,
        "cds_label": label,
        "cds_confidence": round(confidence, 4) if confidence is not None else None,
        "confidence_formula": "component dispersion agreement x original-weight coverage",
        "action": proposed_action,
        "action_strength": strength,
        "recommendation_state": recommendation_state,
        "recommendation_eligibility": eligibility_payload,
        "source_validation_state": (
            "VALIDATED" if eligibility_payload["input_provenance_valid"] else "INVALID"
        ),
        "model_validation_state": eligibility_payload["model_validation"],
        "data_quality_state": (
            "DECISION_ELIGIBLE" if model_eligibility.eligible else "INVALID"
        ),
        "anomaly_gate_active": anomaly_gate,
        "component_scores": {
            key: (round(value, 3) if value is not None else None)
            for key, value in candidates.items()
        },
        "component_reasons": reasons,
        "available_component_count": len(available),
        "required_component_count": MINIMUM_COMPONENTS,
        "component_weight_coverage": round(coverage, 4),
        "minimum_weight_coverage": MINIMUM_WEIGHT_COVERAGE,
        "weight_normalization": "renormalized_available_components",
        "reason_code": None if analytical_valid else "INSUFFICIENT_VALID_COMPONENTS",
        "is_valid": recommendation_state == "ACTIONABLE" and analytical_valid,
        "analytical_valid": analytical_valid,
    }


def _signed_component(payload: dict[str, Any], key: str) -> float | None:
    if payload.get("is_valid") is not True:
        return None
    value = _finite(payload.get(key))
    return _clamp((value + 100.0) / 2.0, 0.0, 100.0) if value is not None else None


def _vrc_component(payload: dict[str, Any]) -> float | None:
    if payload.get("is_valid") is not True:
        return None
    return VRC_SCORE.get(str(payload.get("clock_label") or "").upper())


def _nfve_component(payload: dict[str, Any]) -> float | None:
    if payload.get("is_valid") is not True:
        return None
    return NFVE_SCORE.get(str(payload.get("valuation") or "").upper())


def _ml_component(payload: dict[str, Any], eligible: bool) -> float | None:
    if not eligible:
        return None
    probabilities = payload.get("composite_probabilities")
    if not isinstance(probabilities, dict):
        return None
    for key in ("UP", "bullish", "BULLISH"):
        value = _finite(probabilities.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value * 100.0
    return None


def _component_reason(payload: dict[str, Any], score: float | None) -> str | None:
    if score is not None:
        return None
    return str(payload.get("reason_code") or "COMPONENT_UNAVAILABLE")


def _anomaly(
    ml_prediction: dict[str, Any],
) -> tuple[float | None, bool | None, bool]:
    components = ml_prediction.get("component_outputs")
    anomaly = components.get("m05_anomaly") if isinstance(components, dict) else None
    if not isinstance(anomaly, dict) or anomaly.get("is_valid") is not True:
        return None, None, False
    return _finite(anomaly.get("anomaly_score")), anomaly.get("is_anomalous"), True


def _confidence(scores: list[float], coverage: float) -> float:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    agreement = _clamp(1.0 - math.sqrt(variance) / 50.0, 0.0, 1.0)
    return agreement * _clamp(coverage, 0.0, 1.0)


def _decision(cds: float) -> tuple[str, str, str]:
    if cds > 80:
        return "STRONG BULL", "CONSIDER LONG", "STRONG"
    if cds > 65:
        return "BULL", "CONSIDER LONG", "MODERATE"
    if cds > 55:
        return "MILD BULL", "CONSIDER LONG", "WEAK"
    if cds > 45:
        return "NEUTRAL", "FLAT", "NONE"
    if cds > 35:
        return "MILD BEAR", "CONSIDER SHORT", "WEAK"
    if cds > 20:
        return "BEAR", "CONSIDER SHORT", "MODERATE"
    return "STRONG BEAR", "CONSIDER SHORT", "STRONG"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
