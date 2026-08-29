from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext

from .apex_common import clamp, train_unavailable


MODEL_ID = "m19_dqn"
ACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "stable-baselines3 DQN training not run; quarter-Kelly fallback active",
        context,
        minimum_rows=500,
    )


def predict(
    context: MLContext,
    confidence_multiplier: float = 1.0,
    component_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del context
    components = component_outputs or {}
    ensemble_probs = _probabilities_from_components(components)
    top_prob = max(ensemble_probs.values()) if ensemble_probs else 0.34
    edge = max(0.0, top_prob - 0.5)
    uncertainty = _uncertainty(components)
    size = clamp(edge * 2.0 * confidence_multiplier * (1.0 - uncertainty) * 0.25, 0.0, 1.0)
    q_values = {
        str(action): round(float(top_prob - uncertainty - abs(action - size)), 6)
        for action in ACTIONS
    }
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "prediction": size,
        "recommended_size": round(float(size), 6),
        "q_values": q_values,
        "model_used": "kelly_fallback",
        "confidence": round(max(0.0, top_prob - uncertainty), 6),
        "reason": "DQN policy unavailable; conservative quarter-Kelly fallback used",
    }


def _probabilities_from_components(components: dict[str, Any]) -> dict[str, float]:
    for key in ("m10_ensemble", "ensemble"):
        probs = (components.get(key) or {}).get("composite_probabilities")
        if isinstance(probs, dict) and probs:
            return {str(name): float(value) for name, value in probs.items()}
    for key in ("m02_direction", "m11_tft", "m14_wavelet"):
        probs = (components.get(key) or {}).get("probabilities")
        if isinstance(probs, dict) and probs:
            return {str(name): float(value) for name, value in probs.items()}
    return {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}


def _uncertainty(components: dict[str, Any]) -> float:
    conformal_width = (components.get("m17_conformal") or {}).get("interval_width")
    bnn = (components.get("m20_bnn") or {}).get("epistemic_uncertainty")
    values = []
    for value in (conformal_width, bnn):
        try:
            values.append(max(0.0, min(1.0, float(value) / 4.0)))
        except (TypeError, ValueError):
            pass
    return sum(values) / len(values) if values else 0.35
