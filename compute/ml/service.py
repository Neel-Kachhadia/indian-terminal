from __future__ import annotations

import copy
import time
from typing import Any

from ml.feature_engineering import build_context, current_value
from ml.monitor import ensure_ml_tables, log_prediction
from ml.models.common import symbol_has_artifacts
from ml.models.m21_ohlc_forecast import OHLCForecaster
from ml.models import (
    m01_vol_regime,
    m02_direction,
    m03_strategy_selector,
    m04_iv_reversion,
    m05_anomaly,
    m06_regime_return,
    m07_iv_forecast,
    m08_gp_regime,
    m09_rl_sizer,
    m10_ensemble,
    m11_tft,
    m12_prophet,
    m13_nbeats,
    m14_wavelet,
    m15_kalman,
    m16_survival,
    m17_conformal,
    m18_bandit,
    m19_dqn,
    m20_bnn,
)
from ml.registry import feature_schema_version
from ml.recommendation_policy import RecommendationEvidence, recommendation_eligibility
from release_identity import load_release_identity


def predict_all_models(
    symbol: str,
    signal_payload: dict[str, Any] | None = None,
    *,
    log: bool = False,
) -> dict[str, Any]:
    if log:
        ensure_ml_tables()
    context = build_context(symbol, signal_payload)
    feature_snapshot = _feature_snapshot(context.current_features)

    components: dict[str, Any] = {}
    components["m05_anomaly"] = _safe_predict(
        "m05_anomaly",
        lambda: m05_anomaly.predict(context),
    )
    multiplier = float(components["m05_anomaly"].get("confidence_multiplier", 1.0) or 1.0)

    components["m01_vol_regime"] = _safe_predict(
        "m01_vol_regime",
        lambda: m01_vol_regime.predict(context, multiplier),
    )
    components["m02_direction"] = _safe_predict(
        "m02_direction",
        lambda: m02_direction.predict(context, multiplier),
    )
    components["m03_strategy"] = _safe_predict(
        "m03_strategy",
        lambda: m03_strategy_selector.predict(context, multiplier),
    )
    components["m04_iv_reversion"] = _safe_predict(
        "m04_iv_reversion",
        lambda: m04_iv_reversion.predict(context, multiplier),
    )
    components["m06_regime_return"] = _safe_predict(
        "m06_regime_return",
        lambda: m06_regime_return.predict(context, multiplier, components["m01_vol_regime"]),
    )
    components["m07_iv_forecast"] = _safe_predict(
        "m07_iv_forecast",
        lambda: m07_iv_forecast.predict(context, multiplier),
    )
    components["m08_gp_regime"] = _safe_predict(
        "m08_gp_regime",
        lambda: m08_gp_regime.predict(context, multiplier, components),
    )
    components["m09_rl_sizer"] = _safe_predict(
        "m09_rl_sizer",
        lambda: m09_rl_sizer.predict(context, multiplier, components),
    )
    components["m11_tft"] = _safe_predict(
        "m11_tft",
        lambda: m11_tft.predict(context, multiplier),
    )
    components["m12_prophet"] = _safe_predict(
        "m12_prophet",
        lambda: m12_prophet.predict(context, multiplier),
    )
    components["m13_nbeats"] = _safe_predict(
        "m13_nbeats",
        lambda: m13_nbeats.predict(context, multiplier),
    )
    components["m14_wavelet"] = _safe_predict(
        "m14_wavelet",
        lambda: m14_wavelet.predict(context, multiplier),
    )
    components["m15_kalman"] = _safe_predict(
        "m15_kalman",
        lambda: m15_kalman.predict(context, multiplier),
    )
    components["m16_survival"] = _safe_predict(
        "m16_survival",
        lambda: m16_survival.predict(context, multiplier),
    )
    tft_center = components["m11_tft"].get("return_p50_1d")
    components["m17_conformal"] = _safe_predict(
        "m17_conformal",
        lambda: m17_conformal.predict(context, tft_center, multiplier),
    )
    components["m18_bandit"] = _safe_predict(
        "m18_bandit",
        lambda: m18_bandit.predict(context, multiplier),
    )
    ensemble = _safe_predict(
        "m10_ensemble",
        lambda: m10_ensemble.predict(context, components),
    )
    components["m20_bnn"] = _safe_predict(
        "m20_bnn",
        lambda: m20_bnn.predict(context, multiplier, components),
    )
    components["m19_dqn"] = _safe_predict(
        "m19_dqn",
        lambda: m19_dqn.predict(context, multiplier, {**components, "m10_ensemble": ensemble}),
    )
    ensemble["component_outputs"] = components
    ensemble["symbol"] = context.symbol
    ensemble["computed_at"] = time.time()
    ensemble["feature_rows"] = context.available_rows
    ensemble["fo_eod_rows"] = context.fo_eod_rows
    ensemble["has_symbol_model"] = symbol_has_artifacts(
        context.symbol,
        ["m01_vol_regime", "m02_direction", "m21_ohlc"],
    )
    ensemble["model_source"] = (
        "symbol_specific"
        if context.symbol == "NIFTY" or ensemble["has_symbol_model"]
        else "nifty_proxy"
    )
    ensemble["ohlc_forecast"] = _ohlc_forecast(context)
    _extend_apex_output(ensemble, context, components, signal_payload or {})
    data_as_of = _context_data_as_of(context)
    for model_id, output in components.items():
        output.setdefault(
            "model_attribution",
            _runtime_attribution(model_id, context.symbol, data_as_of=data_as_of),
        )
    ensemble.setdefault(
        "model_attribution",
        _runtime_attribution(
            "m10_ensemble",
            context.symbol,
            data_as_of=data_as_of,
            component_versions={
                model_id: output.get("model_attribution", {}).get("model_version")
                for model_id, output in components.items()
            },
        ),
    )
    decision_evidence = _decision_evidence(context, data_as_of=data_as_of)
    ensemble["decision_input_evidence"] = decision_evidence.to_dict()
    eligibility = _action_eligibility(
        ensemble,
        components,
        evidence=decision_evidence,
    )
    ensemble["action_recommendation"] = _apex_action(
        ensemble,
        components.get("m18_bandit", {}),
        components.get("m19_dqn", {}),
        components.get("m17_conformal", {}),
        components.get("m20_bnn", {}),
        eligibility=eligibility.to_dict(),
    )

    if log:
        for model_id, output in components.items():
            _log_output(model_id, output, feature_snapshot)
        _log_output("m10_ensemble", ensemble, feature_snapshot)

    return ensemble


def _ohlc_forecast(context: Any) -> dict[str, Any] | None:
    try:
        atm_iv = current_value(context, ["atm_iv", "options.atm_iv"])
        vix = current_value(context, ["india_vix", "vix.india_vix"])
        return OHLCForecaster().predict(
            context.symbol,
            context.ohlcv,
            context.current_spot,
            atm_iv=atm_iv,
            vix=vix,
            include_accuracy=True,
            log_forecast=False,
        )
    except Exception as error:
        print(f"ml: ohlc forecast failed: {error}")
        return None


def _extend_apex_output(
    ensemble: dict[str, Any],
    context: Any,
    components: dict[str, Any],
    signal_payload: dict[str, Any],
) -> None:
    try:
        from intelligence.narrative import prediction_narrative
        from intelligence.regime_detector import detect_regime

        regime = detect_regime(context.symbol, context.ohlcv, signal_payload, persist=False)
    except Exception as error:
        print(f"ml: regime/narrative extension failed: {error}")
        regime = {}

    tft = components.get("m11_tft", {})
    prophet = components.get("m12_prophet", {})
    wavelet = components.get("m14_wavelet", {})
    kalman = components.get("m15_kalman", {})
    conformal = components.get("m17_conformal", {})
    bandit = components.get("m18_bandit", {})
    dqn = components.get("m19_dqn", {})
    bnn = components.get("m20_bnn", {})

    ensemble["spot"] = context.current_spot
    ensemble["tft_return_p50_1d"] = tft.get("return_p50_1d")
    ensemble["prophet_regime"] = prophet.get("regime")
    ensemble["wavelet_signal_1d"] = wavelet.get("wavelet_signal_1d")
    ensemble["kalman_state"] = kalman.get("kalman_state")
    ensemble["hmm_state"] = regime.get("state_label")
    ensemble["hmm_state_probabilities"] = regime.get("state_probabilities", {})
    ensemble["conformal_interval_90"] = conformal.get("interval_90")
    ensemble["bayesian_uncertainty"] = bnn.get("epistemic_uncertainty")
    ensemble["calibrated_confidence"] = ensemble.get("composite_confidence")
    ensemble["uncertainty_level"] = _uncertainty_level(conformal, bnn)
    try:
        ensemble["narrative"] = prediction_narrative(context.symbol, signal_payload, ensemble)
    except Exception:
        ensemble["narrative"] = "Market intelligence available in narrative endpoint."
    ensemble["trained_status"] = {
        key: bool(value.get("is_trained"))
        for key, value in components.items()
        if isinstance(value, dict)
    }


def _uncertainty_level(conformal: dict[str, Any], bnn: dict[str, Any]) -> str:
    width = _as_float((conformal or {}).get("interval_width"))
    epistemic = _as_float((bnn or {}).get("epistemic_uncertainty"))
    score = 0.0
    if width is not None:
        score += min(1.0, width / 3.0) * 0.55
    if epistemic is not None:
        score += min(1.0, epistemic) * 0.45
    if score >= 0.66:
        return "HIGH"
    if score >= 0.33:
        return "MEDIUM"
    return "LOW"


def _apex_action(
    ensemble: dict[str, Any],
    bandit: dict[str, Any],
    dqn: dict[str, Any],
    conformal: dict[str, Any],
    bnn: dict[str, Any],
    *,
    eligibility: dict[str, Any],
) -> dict[str, Any]:
    direction = ensemble.get("composite_direction")
    confidence = _as_float(ensemble.get("composite_confidence")) or 0.0
    uncertainty = ensemble.get("uncertainty_level")
    strategy = bandit.get("recommended_strategy") or "STAY_FLAT"
    size = _as_float(dqn.get("recommended_size")) or 0.0
    trade = bool(
        eligibility["eligible"]
        and direction in {"UP", "DOWN"}
        and confidence >= 0.58
        and uncertainty != "HIGH"
        and size > 0
    )
    action = "STAY FLAT"
    if trade and direction == "UP":
        action = "CONSIDER LONG"
    elif trade and direction == "DOWN":
        action = "CONSIDER SHORT"
    interval = conformal.get("interval_90")
    max_loss = None
    if isinstance(interval, list) and len(interval) == 2:
        lower = _as_float(interval[0])
        if lower is not None:
            max_loss = round(abs(lower) * 0.01 * float(ensemble.get("spot") or 0.0) * 75.0 * max(size, 0.25), 2)
    if not eligibility["eligible"]:
        action = "UNVALIDATED"
    return {
        "trade": trade,
        "action": action,
        "eligibility": eligibility,
        "strategy": strategy if trade else "STAY_FLAT",
        "size": size if trade else 0.0,
        "entry_level": ensemble.get("spot"),
        "stop_loss_level": None,
        "target_level": None,
        "risk_reward_ratio": None,
        "maximum_notional_loss": max_loss,
        "kelly_fraction": size,
        "recommended_fraction": size if trade else 0.0,
        "reasoning": [
            f"Composite direction {direction} at {confidence * 100:.0f}% confidence.",
            f"Uncertainty level {uncertainty}.",
            "Action language is decision support only; no automated execution.",
        ],
    }


def _action_eligibility(
    ensemble: dict[str, Any],
    components: dict[str, Any],
    *,
    evidence: RecommendationEvidence,
) -> Any:
    return recommendation_eligibility(ensemble, components, evidence=evidence)


def _decision_evidence(context: Any, *, data_as_of: str | None) -> RecommendationEvidence:
    market = context.market_data_eligibility
    spot = context.spot_provenance if isinstance(context.spot_provenance, dict) else {}
    spot_eligible = spot.get("decision_eligible") is True
    market_and_spot_eligible = market.decision_eligible and spot_eligible
    spot_source = str(spot.get("source") or "NONE").strip().upper()
    combined_provenance = (
        f"{market.source}+{spot_source}"
        if market.source not in {"", "NONE", "UNKNOWN"}
        and spot_source not in {"", "NONE", "UNKNOWN"}
        else None
    )
    feature_columns = sorted(str(column) for column in context.feature_frame.columns if column != "date")
    return RecommendationEvidence(
        market_data_quality=("OBSERVED_VERIFIED" if market_and_spot_eligible else "INVALID_UNVERIFIED"),
        market_data_freshness=(
            "CURRENT_SESSION_ELIGIBLE" if market_and_spot_eligible else "STALE"
        ),
        market_data_provenance=combined_provenance,
        prediction_as_of=data_as_of,
        requested_decision_at=context.decision_requested_at.isoformat(),
        fallback_state=(
            "NONE"
            if market_and_spot_eligible and spot.get("fallback_used") is False
            else "MISSING_OR_UNAUTHORIZED_SPOT"
        ),
        input_feature_schema=(
            feature_schema_version(feature_columns) if feature_columns else None
        ),
    )


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _safe_predict(model_id: str, fn: Any) -> dict[str, Any]:
    try:
        output = fn()
        if not isinstance(output, dict):
            raise TypeError("model output must be dict")
        output.setdefault("model_id", model_id)
        return output
    except Exception as error:
        return {
            "model_id": model_id,
            "is_trained": False,
            "prediction": None,
            "probabilities": {},
            "confidence": None,
            "reason": f"{model_id} crashed: {error}",
        }


def _log_output(model_id: str, output: dict[str, Any], features: dict[str, Any]) -> None:
    probabilities = output.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = output.get("composite_probabilities")
    if not isinstance(probabilities, dict):
        probabilities = _pseudo_probabilities(output)
    prediction = output.get("prediction")
    if prediction is None:
        prediction = (
            output.get("regime")
            or output.get("direction")
            or output.get("strategy")
            or output.get("iv_direction")
        )
    confidence = output.get("confidence")
    if confidence is None:
        confidence = output.get("composite_confidence")
    try:
        attribution = output.get("model_attribution")
        if not isinstance(attribution, dict):
            raise RuntimeError("prediction attribution missing")
        output["prediction_tracking_id"] = log_prediction(
            model_id,
            str(prediction) if prediction is not None else None,
            probabilities,
            confidence,
            features,
            attribution=attribution,
        )
    except Exception as error:
        print(f"ml: prediction log failed for {model_id}: {error}")


def _pseudo_probabilities(output: dict[str, Any]) -> dict[str, float]:
    if "is_anomalous" in output:
        return {"ANOMALOUS": 1.0 if output.get("is_anomalous") else 0.0, "NORMAL": 0.0 if output.get("is_anomalous") else 1.0}
    if "recommended_size" in output:
        return {"SIZED": float(output.get("recommended_size") or 0.0)}
    if "iv_forecast_3d" in output and output.get("iv_forecast_3d") is not None:
        return {"FORECAST_AVAILABLE": 1.0}
    return {}


def _feature_snapshot(features: dict[str, Any]) -> dict[str, Any]:
    snapshot = copy.deepcopy(features)
    if len(snapshot) > 80:
        keys = sorted(snapshot)[:80]
        snapshot = {key: snapshot[key] for key in keys}
    return snapshot


def _context_data_as_of(context: Any) -> str | None:
    frame = getattr(context, "feature_frame", None)
    if frame is None or frame.empty or "date" not in frame.columns:
        return None
    import pandas as pd

    value = pd.to_datetime(frame["date"], errors="coerce").max()
    return value.date().isoformat() if pd.notna(value) else None


def _runtime_attribution(
    model_id: str,
    symbol: str,
    *,
    data_as_of: str | None,
    component_versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    release = load_release_identity()
    attribution = {
        "model_id": model_id,
        "model_version": f"HEURISTIC-v1@{release['release_version']}",
        "trained_symbol": None,
        "requested_symbol": symbol,
        "universe": symbol,
        "training_data_horizon": None,
        "feature_schema_version": feature_schema_version([model_id, "runtime-features-v1"]),
        "training_timestamp": None,
        "artifact_hash": None,
        "code_version": str(release["python_api_version"]),
        "release_version": str(release["release_version"]),
        "validation_status": "UNVALIDATED",
        "artifact_verified": False,
        "registry_status": "UNVALIDATED",
        "feature_schema_compatible": False,
        "fallback_status": "NONE",
        "data_as_of": data_as_of,
    }
    if component_versions is not None:
        attribution["component_versions"] = component_versions
    return attribution
