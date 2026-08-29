from __future__ import annotations

from typing import Any

from ml.calibrator import clipped_confidence, normalise_probabilities
from ml.feature_engineering import MLContext, current_value
from ml.monitor import accuracy_for_model


MODEL_ID = "m10_ensemble"
CLASSES = ["UP", "FLAT", "DOWN"]


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del context, force
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "ensemble_method": "simple_avg",
        "reason": "stacking waits for resolved live prediction history",
    }


def predict(context: MLContext, component_outputs: dict[str, Any]) -> dict[str, Any]:
    vectors = _direction_vectors(context, component_outputs)
    method = "bayesian_avg" if any(item["weight"] != 1.0 for item in vectors) else "simple_avg"
    if not vectors:
        probabilities = {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}
    else:
        totals = {label: 0.0 for label in CLASSES}
        weight_sum = 0.0
        for item in vectors:
            weight = item["weight"]
            weight_sum += weight
            for label in CLASSES:
                totals[label] += item["probabilities"].get(label, 0.0) * weight
        probabilities = normalise_probabilities(
            CLASSES,
            [totals[label] / max(weight_sum, 1e-9) for label in CLASSES],
        )

    anomaly = component_outputs.get("m05_anomaly", {})
    multiplier = float(anomaly.get("confidence_multiplier", 1.0) or 1.0)
    direction = max(probabilities, key=probabilities.get)
    confidence = clipped_confidence(probabilities, multiplier)
    agreement = _agreement_score(probabilities)
    sizer = component_outputs.get("m09_rl_sizer", {})
    strategy = (component_outputs.get("m03_strategy") or {}).get("strategy")
    analytical_trade_candidate = bool(
        not anomaly.get("is_anomalous")
        and direction != "FLAT"
        and agreement >= 0.42
        and float(sizer.get("recommended_size") or 0.0) > 0.0
    )
    accuracy = _accuracy_tracking()
    output = {
        "model_id": MODEL_ID,
        "is_trained": False,
        "composite_direction": direction,
        "prediction": direction,
        "composite_probabilities": probabilities,
        "probabilities": probabilities,
        "composite_confidence": confidence,
        "confidence": confidence,
        "model_agreement": round(agreement, 6),
        "ensemble_method": method,
        "component_outputs": component_outputs,
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


def _direction_vectors(context: MLContext, component_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    for key in ["m02_direction", "m06_regime_return"]:
        probs = (component_outputs.get(key) or {}).get("probabilities")
        if isinstance(probs, dict) and probs:
            acc = accuracy_for_model(key)["accuracy"]
            vectors.append(
                {
                    "probabilities": {label: float(probs.get(label, 0.0)) for label in CLASSES},
                    "weight": max(0.25, float(acc)) if acc is not None else 1.0,
                }
            )
    score = current_value(context, ["overall_score", "composite.overall_score"])
    if score is not None:
        up = max(0.05, min(0.75, score / 100.0))
        down = max(0.05, min(0.75, (100.0 - score) / 100.0))
        flat = max(0.10, 1.0 - abs(score - 50.0) / 50.0)
        probs = normalise_probabilities(CLASSES, [up, flat, down])
        vectors.append({"probabilities": probs, "weight": 0.5})
    return vectors


def _agreement_score(probabilities: dict[str, float]) -> float:
    top = max(probabilities.values()) if probabilities else 0.0
    return max(0.0, min(1.0, (top - (1.0 / 3.0)) / (2.0 / 3.0)))


def _reasoning(
    direction: str,
    probabilities: dict[str, float],
    component_outputs: dict[str, Any],
    trade: bool,
) -> list[str]:
    reasons: list[str] = []
    anomaly = component_outputs.get("m05_anomaly", {})
    if anomaly.get("is_anomalous"):
        reasons.append("Anomaly circuit breaker active; suppressing trade.")
    reasons.append(f"Composite direction {direction} with {probabilities.get(direction, 0.0) * 100:.0f}% probability.")
    m3_reasoning = (component_outputs.get("m03_strategy") or {}).get("reasoning") or []
    reasons.extend(str(item) for item in m3_reasoning[:2])
    if not trade:
        reasons.append("Recommendation stays flat until model agreement and sizing clear threshold.")
    return reasons[:3]


def _accuracy_tracking() -> dict[str, Any]:
    model_ids = [
        "m01_vol_regime",
        "m02_direction",
        "m04_iv_reversion",
        "m06_regime_return",
        "m10_ensemble",
    ]
    scores = {model_id: accuracy_for_model(model_id)["accuracy"] for model_id in model_ids}
    resolved = {key: value for key, value in scores.items() if value is not None}
    best = max(resolved, key=resolved.get) if resolved else None
    worst = min(resolved, key=resolved.get) if resolved else None
    return {
        "ensemble_accuracy_30d": scores.get("m10_ensemble"),
        "ensemble_accuracy_7d": accuracy_for_model("m10_ensemble", 7)["accuracy"],
        "best_model_30d": best,
        "worst_model_30d": worst,
        "models": scores,
    }


def _legacy_contract(output: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    m1 = components.get("m01_vol_regime", {})
    m2 = components.get("m02_direction", {})
    m3 = components.get("m03_strategy", {})
    m4 = components.get("m04_iv_reversion", {})
    probs = output["composite_probabilities"]
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
            "bearish_score": round(float(probs.get("DOWN", 0.0)) * 100.0, 3),
            "regime_warning": bool((components.get("m05_anomaly") or {}).get("is_anomalous")),
            "top_signals": [],
        },
        "model_accuracy_30d": output["accuracy_tracking"].get("models", {}),
    }
