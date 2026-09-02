from __future__ import annotations

from typing import Any

MODEL_ID = "m09_rl_sizer"


def train_model(context: Any, force: bool = False) -> dict[str, Any]:
    del context, force
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "model_used": "kelly_fallback",
        "reason": "stable-baselines3 PPO unavailable; quarter-Kelly fallback active",
    }

def predict(
    context: Any,
    confidence_multiplier: float = 1.0,
    component_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del context
    direction_probs = _direction_probabilities(component_outputs or {})
    if direction_probs is None:
        return {
            "model_id": MODEL_ID,
            "is_trained": False,
            "is_valid": False,
            "status": "UNAVAILABLE",
            "reason_code": "NO_ELIGIBLE_DIRECTION_PROBABILITIES",
            "reason": "0/2 trained direction models provide eligible probabilities",
            "prediction": None,
            "recommended_size": None,
            "recommended_direction": None,
            "kelly_fraction": None,
            "quarter_kelly": None,
            "model_used": "quarter_kelly_analytical",
            "confidence": None,
        }
    up = direction_probs["UP"]
    down = direction_probs["DOWN"]
    flat = direction_probs["FLAT"]
    edge_direction = up - down
    win_probability = max(up, down)
    payoff_ratio = 1.15
    loss_probability = 1.0 - win_probability
    kelly = max(0.0, (win_probability * payoff_ratio - loss_probability) / payoff_ratio)
    quarter_kelly = min(0.25, kelly * 0.25)
    if flat >= max(up, down):
        recommended_direction = 0.0
        recommended_size = 0.0
    else:
        recommended_direction = max(-1.0, min(1.0, edge_direction * 2.0))
        recommended_size = quarter_kelly
    if confidence_multiplier < 1.0:
        recommended_size *= confidence_multiplier
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "READY_ANALYTICAL",
        "reason_code": None,
        "prediction": round(recommended_direction, 6),
        "recommended_size": round(recommended_size, 6),
        "recommended_direction": round(recommended_direction, 6),
        "kelly_fraction": round(kelly, 6),
        "quarter_kelly": round(quarter_kelly, 6),
        "model_used": "quarter_kelly_analytical",
        "confidence": None,
        "confidence_formula": None,
        "payoff_ratio_assumption": payoff_ratio,
        "reason": "analytical quarter-Kelly sizing; payoff ratio is an explicit assumption, not measured",
    }


def _direction_probabilities(
    component_outputs: dict[str, Any],
) -> dict[str, float] | None:
    candidates: list[dict[str, float]] = []
    for key in ["m02_direction", "m06_regime_return"]:
        output = component_outputs.get(key) or {}
        probs = output.get("probabilities")
        if output.get("is_trained") is True and isinstance(probs, dict) and probs:
            candidates.append({label: float(probs.get(label, 0.0)) for label in ["UP", "FLAT", "DOWN"]})
    if not candidates:
        return None
    return {
        label: sum(item[label] for item in candidates) / len(candidates)
        for label in ["UP", "FLAT", "DOWN"]
    }
