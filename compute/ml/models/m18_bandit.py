from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext

from .apex_common import train_unavailable, untrained_prediction


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
    del confidence_multiplier
    return untrained_prediction(
        MODEL_ID,
        "no resolved live strategy outcomes; bandit posterior is unavailable",
        context,
        minimum_rows=1,
        extra={
        "strategy_probabilities": {},
        "recommended_strategy": None,
        "strategy": None,
        "n_pulls": {strategy: 0 for strategy in STRATEGIES},
        "n_wins": {strategy: 0 for strategy in STRATEGIES},
        "exploration_rate": None,
        "observed_outcomes": 0,
        },
    )
