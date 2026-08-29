from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext, current_value

from .apex_common import normalise, train_unavailable


MODEL_ID = "m18_bandit"
STRATEGIES = [
    "STAY_FLAT",
    "LONG_CALL",
    "LONG_PUT",
    "ATM_STRADDLE",
    "IRON_CONDOR",
    "BULL_SPREAD",
    "BEAR_SPREAD",
    "SHORT_STRANGLE",
]


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "live outcome priors not accumulated yet",
        context,
        minimum_rows=1,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    iv_rank = current_value(context, ["iv_rank", "options.iv_rank"])
    pcr = current_value(context, ["pcr", "options.pcr_total", "options.pcr"])
    overall = current_value(context, ["overall_score", "composite.overall_score"])
    inputs_available = iv_rank is not None and pcr is not None and overall is not None
    raw = {strategy: 1.0 for strategy in STRATEGIES}
    raw["STAY_FLAT"] += 2.0
    if inputs_available:
        if iv_rank > 70:
            raw["IRON_CONDOR"] += 1.8
            raw["SHORT_STRANGLE"] += 1.2
        if iv_rank < 30:
            raw["ATM_STRADDLE"] += 1.4
        if overall > 58 and pcr > 1.0:
            raw["BULL_SPREAD"] += 1.8
            raw["LONG_CALL"] += 0.8
        if overall < 42 and pcr < 0.9:
            raw["BEAR_SPREAD"] += 1.8
            raw["LONG_PUT"] += 0.8
    probabilities = normalise(raw)
    if confidence_multiplier < 1.0:
        probabilities["STAY_FLAT"] = min(0.95, probabilities["STAY_FLAT"] + (1.0 - confidence_multiplier) * 0.35)
        probabilities = normalise(probabilities)
    recommended = max(probabilities, key=probabilities.get)
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "prediction": recommended,
        "strategy_probabilities": probabilities,
        "probabilities": probabilities,
        "recommended_strategy": recommended,
        "strategy": recommended,
        "n_pulls": {strategy: 0 for strategy in STRATEGIES},
        "n_wins": {strategy: 0 for strategy in STRATEGIES},
        "exploration_rate": 1.0,
        "confidence": round(probabilities[recommended], 6),
        "reason": (
            "Thompson priors only; update after live outcomes"
            if inputs_available
            else "Thompson priors only; iv_rank/pcr/overall_score unavailable, defaulting toward no-action"
        ),
    }
