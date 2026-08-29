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
    """
    Composite Decision Score from five BIF indicators plus ML ensemble.
    """
    bms_norm = (_number(bms.get("bms"), 0.0) + 100.0) / 2.0
    ngi_norm = (_number(ngi.get("ngi"), 0.0) + 100.0) / 2.0
    iad_norm = (_number(iad.get("iad_score"), 0.0) + 100.0) / 2.0
    # vrc is only usable when the volatility clock actually resolved a regime
    # (clock_label in VRC_SCORE). An unavailable-IV/insufficient-history read
    # (vol_clock.py's "NO_DATA"/"INSUFFICIENT_HISTORY" reason codes, is_valid
    # False) must not silently inject a fabricated neutral-or-real score into
    # the composite decision -- it is excluded and the remaining weights are
    # renormalized instead.
    vrc_label = str(vrc.get("clock_label") or "").upper()
    vrc_norm = VRC_SCORE.get(vrc_label) if vrc.get("is_valid") else None
    nfve_norm = NFVE_SCORE.get(str(nfve.get("valuation") or "").upper(), 50.0)
    ml_norm = _ml_up_probability(ml_prediction) * 100.0
    model_eligibility = model_output_eligibility(
        ml_prediction,
        evidence=decision_evidence,
    )
    if not model_eligibility.eligible:
        ml_norm = 50.0

    weights = {
        "bms": 0.25,
        "ngi": 0.15,
        "iad": 0.20,
        "vrc": 0.15,
        "nfve": 0.10,
        "ml": 0.15,
    }
    available = {
        "bms": _clamp(bms_norm, 0.0, 100.0),
        "ngi": _clamp(ngi_norm, 0.0, 100.0),
        "iad": _clamp(iad_norm, 0.0, 100.0),
        "nfve": _clamp(nfve_norm, 0.0, 100.0),
        "ml": _clamp(ml_norm, 0.0, 100.0),
    }
    if vrc_norm is not None:
        available["vrc"] = _clamp(vrc_norm, 0.0, 100.0)

    scores = {key: available.get(key) for key in weights}
    weight_total = sum(weights[key] for key in available)
    cds = sum(available[key] * weights[key] for key in available) / weight_total

    anomaly_score, anomaly_flag = _anomaly(ml_prediction)
    anomaly_gate = anomaly_flag or anomaly_score > 0.7
    if anomaly_gate:
        cds = 50.0 + (cds - 50.0) * 0.2

    confidence = _confidence(list(available.values()))
    label, action, strength = _decision(cds)
    recommendation_state = actionable_state(model_eligibility, action)
    if recommendation_state != "ACTIONABLE":
        action = "UNVALIDATED" if recommendation_state == "UNVALIDATED" else "FLAT"
        strength = "NONE"

    eligibility_payload = model_eligibility.to_dict()
    return {
        "cds": round(_clamp(cds, 0.0, 100.0), 3),
        "cds_label": label,
        "cds_confidence": round(confidence, 4),
        "action": action,
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
            for key, value in scores.items()
        },
        "is_valid": recommendation_state == "ACTIONABLE",
        "analytical_valid": True,
    }


def _ml_up_probability(ml_prediction: dict[str, Any]) -> float:
    probabilities = ml_prediction.get("composite_probabilities")
    if isinstance(probabilities, dict):
        for key in ("UP", "bullish", "BULLISH"):
            value = _finite(probabilities.get(key))
            if value is not None:
                return _clamp(value, 0.0, 1.0)

    direction = str(ml_prediction.get("composite_direction") or "").upper()
    confidence = _finite(ml_prediction.get("composite_confidence"))
    confidence = _clamp(confidence if confidence is not None else 0.5, 0.0, 1.0)
    if direction == "UP":
        return 0.5 + confidence / 2.0
    if direction == "DOWN":
        return 0.5 - confidence / 2.0
    return 0.5


def _anomaly(ml_prediction: dict[str, Any]) -> tuple[float, bool]:
    components = ml_prediction.get("component_outputs")
    if not isinstance(components, dict):
        return 0.0, False
    anomaly = components.get("m05_anomaly")
    if not isinstance(anomaly, dict):
        return 0.0, False
    return _number(anomaly.get("anomaly_score"), 0.0), bool(anomaly.get("is_anomalous"))


def _confidence(scores: list[float]) -> float:
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std = math.sqrt(variance)
    return _clamp(1.0 - std / 50.0, 0.0, 1.0)


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


def _number(value: Any, default: float) -> float:
    number = _finite(value)
    return number if number is not None else default


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
