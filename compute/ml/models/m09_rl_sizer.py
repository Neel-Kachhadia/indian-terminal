from __future__ import annotations

from typing import Any

from ml.calibrator import clipped_confidence


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
    up = direction_probs.get("UP", 0.33)
    down = direction_probs.get("DOWN", 0.33)
    flat = direction_probs.get("FLAT", 0.34)
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
        "prediction": round(recommended_direction, 6),
        "recommended_size": round(recommended_size, 6),
        "recommended_direction": round(recommended_direction, 6),
        "kelly_fraction": round(kelly, 6),
        "quarter_kelly": round(quarter_kelly, 6),
        "model_used": "kelly_fallback",
        "confidence": clipped_confidence({"edge": win_probability}, confidence_multiplier),
    }


def _direction_probabilities(component_outputs: dict[str, Any]) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for key in ["m02_direction", "m06_regime_return"]:
        probs = (component_outputs.get(key) or {}).get("probabilities")
        if isinstance(probs, dict) and probs:
            candidates.append({label: float(probs.get(label, 0.0)) for label in ["UP", "FLAT", "DOWN"]})
    if not candidates:
        return {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}
    return {
        label: sum(item[label] for item in candidates) / len(candidates)
        for label in ["UP", "FLAT", "DOWN"]
    }
