from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class RecommendationState(StrEnum):
    ANALYTICAL_OUTPUT = "ANALYTICAL_OUTPUT"
    INFORMATIONAL = "INFORMATIONAL"
    UNVALIDATED = "UNVALIDATED"
    VALIDATED_NON_ACTIONABLE = "VALIDATED_NON_ACTIONABLE"
    ACTIONABLE = "ACTIONABLE"


REQUIRED_ACTION_MODELS = (
    "m10_ensemble", "m02_direction", "m17_conformal",
    "m18_bandit", "m19_dqn", "m20_bnn",
)
VALID_MARKET_QUALITY = frozenset({"OBSERVED_VERIFIED", "EXECUTABLE_STREAM"})
VALID_FRESHNESS = frozenset({"CURRENT_SESSION_ELIGIBLE", "FRESH"})
INVALID_PROVENANCE = frozenset(
    {"", "NONE", "UNKNOWN", "FO_EOD_FUTIDX_DERIVED", "LEGACY_UNVERIFIED"}
)
VALID_PROVENANCE_SOURCES = frozenset(
    {"NSE_INDEX_BHAVCOPY", "NSE_EQUITY_BHAVCOPY", "ANGEL_STREAM"}
)


@dataclass(frozen=True)
class RecommendationEvidence:
    market_data_quality: str | None
    market_data_freshness: str | None
    market_data_provenance: str | None
    prediction_as_of: str | None
    requested_decision_at: str | None
    fallback_state: str | None
    input_feature_schema: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RecommendationEvidence":
        data = value if isinstance(value, Mapping) else {}
        return cls(
            _text_or_none(data.get("market_data_quality")),
            _text_or_none(data.get("market_data_freshness")),
            _text_or_none(data.get("market_data_provenance")),
            _text_or_none(data.get("prediction_as_of")),
            _text_or_none(data.get("requested_decision_at")),
            _text_or_none(data.get("fallback_state")),
            _text_or_none(data.get("input_feature_schema")),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "market_data_quality": self.market_data_quality,
            "market_data_freshness": self.market_data_freshness,
            "market_data_provenance": self.market_data_provenance,
            "prediction_as_of": self.prediction_as_of,
            "requested_decision_at": self.requested_decision_at,
            "fallback_state": self.fallback_state,
            "input_feature_schema": self.input_feature_schema,
        }


@dataclass(frozen=True)
class RecommendationEligibility:
    eligible: bool
    status: RecommendationState
    reasons: tuple[str, ...]
    evidence: RecommendationEvidence
    model_validation: str

    def to_dict(self) -> dict[str, Any]:
        evidence = self.evidence.to_dict()
        return {
            "eligible": self.eligible,
            "state": self.status.value,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "market_data_quality": evidence["market_data_quality"],
            "market_data_freshness": evidence["market_data_freshness"],
            "provenance": evidence["market_data_provenance"],
            "as_of": evidence["prediction_as_of"],
            "requested_decision_at": evidence["requested_decision_at"],
            "fallback_state": evidence["fallback_state"],
            "input_feature_schema": evidence["input_feature_schema"],
            "model_validation": self.model_validation,
            "input_provenance_valid": _valid_market_provenance(
                evidence["market_data_provenance"]
            ),
            "output_freshness_acceptable": (
                evidence["market_data_freshness"] in VALID_FRESHNESS
            ),
            "decision_input_evidence": evidence,
        }


def recommendation_eligibility(
    ensemble: dict[str, Any],
    components: dict[str, Any] | None,
    *,
    evidence: RecommendationEvidence | Mapping[str, Any] | None,
) -> RecommendationEligibility:
    """Single fail-closed actionability policy; absent proof is invalid proof."""
    decision_evidence = (
        evidence
        if isinstance(evidence, RecommendationEvidence)
        else RecommendationEvidence.from_mapping(evidence)
    )
    component_outputs = components
    if not isinstance(component_outputs, dict):
        candidate = ensemble.get("component_outputs")
        component_outputs = candidate if isinstance(candidate, dict) else {}
    required = {"m10_ensemble": ensemble}
    required.update(
        {model_id: component_outputs.get(model_id, {}) for model_id in REQUIRED_ACTION_MODELS[1:]}
    )
    reasons: list[str] = []
    for model_id, output in required.items():
        if not isinstance(output, dict) or not output:
            reasons.append(f"{model_id}: output missing")
            continue
        attribution = output.get("model_attribution")
        if not isinstance(attribution, dict):
            reasons.append(f"{model_id}: attribution missing")
            continue
        for field in (
            "artifact_hash", "model_version", "training_data_horizon", "feature_schema_version",
        ):
            if not attribution.get(field):
                reasons.append(f"{model_id}: {field} missing")
        if attribution.get("artifact_verified") is not True:
            reasons.append(f"{model_id}: artifact not verified")
        if attribution.get("registry_status") != "VALIDATED":
            reasons.append(f"{model_id}: registry not validated")
        if attribution.get("feature_schema_compatible") is not True:
            reasons.append(f"{model_id}: feature schema incompatible")
        if attribution.get("validation_status") != "VALIDATED":
            reasons.append(f"{model_id}: model validation unavailable")
        if output.get("is_trained") is not True:
            reasons.append(f"{model_id}: not trained")
        if output.get("calibrated") is not True:
            reasons.append(f"{model_id}: calibration unavailable")

    if decision_evidence.market_data_quality not in VALID_MARKET_QUALITY:
        reasons.append("market-data quality missing or invalid")
    if decision_evidence.market_data_freshness not in VALID_FRESHNESS:
        reasons.append("market-data freshness missing or invalid")
    if not _valid_market_provenance(decision_evidence.market_data_provenance):
        reasons.append("market-data provenance missing or invalid")
    if not decision_evidence.input_feature_schema:
        reasons.append("input feature schema missing")
    if not decision_evidence.prediction_as_of:
        reasons.append("prediction as-of missing")
    if not decision_evidence.requested_decision_at:
        reasons.append("requested decision timestamp missing")
    prediction_at = _timestamp_or_none(decision_evidence.prediction_as_of)
    requested_at = _timestamp_or_none(decision_evidence.requested_decision_at)
    if decision_evidence.prediction_as_of and prediction_at is None:
        reasons.append("prediction as-of invalid")
    if decision_evidence.requested_decision_at and requested_at is None:
        reasons.append("requested decision timestamp invalid")
    if prediction_at is not None and requested_at is not None and prediction_at > requested_at:
        reasons.append("prediction uses future information")
    if decision_evidence.fallback_state != "NONE":
        reasons.append("fallback state missing or unauthorized")

    unique = tuple(sorted(set(reasons)))
    eligible = not unique
    model_reasons = [reason for reason in unique if reason.startswith(REQUIRED_ACTION_MODELS)]
    return RecommendationEligibility(
        eligible,
        RecommendationState.VALIDATED_NON_ACTIONABLE if eligible else RecommendationState.UNVALIDATED,
        unique,
        decision_evidence,
        "VALIDATED" if not model_reasons else "UNVALIDATED",
    )


def model_output_eligibility(
    output: dict[str, Any],
    *,
    evidence: RecommendationEvidence | Mapping[str, Any] | None = None,
) -> RecommendationEligibility:
    supplied = evidence
    if supplied is None:
        embedded = output.get("decision_input_evidence")
        supplied = embedded if isinstance(embedded, Mapping) else None
    return recommendation_eligibility(output, None, evidence=supplied)


def actionable_state(eligibility: RecommendationEligibility, proposed_action: str) -> str:
    if not eligibility.eligible:
        return RecommendationState.UNVALIDATED.value
    if proposed_action in {"CONSIDER LONG", "CONSIDER SHORT"}:
        return RecommendationState.ACTIONABLE.value
    return RecommendationState.VALIDATED_NON_ACTIONABLE.value


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _timestamp_or_none(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = dt.datetime.combine(dt.date.fromisoformat(value), dt.time.min, dt.timezone.utc)
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _valid_market_provenance(value: str | None) -> bool:
    text = str(value or "").strip().upper()
    if not text or text in INVALID_PROVENANCE:
        return False
    sources = [item.strip() for item in text.split("+")]
    return bool(sources) and all(source in VALID_PROVENANCE_SOURCES for source in sources)
