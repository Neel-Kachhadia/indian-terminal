from __future__ import annotations

from typing import Any

from ml.feature_engineering import MLContext, current_value
from ml.monitor import accuracy_for_model

from .common import untrained_output


MODEL_ID = "m03_strategy"
CLASSES = [
    "BUY_STRADDLE",
    "SELL_STRADDLE",
    "BULL_CALL_SPREAD",
    "BEAR_PUT_SPREAD",
    "IRON_CONDOR",
    "CALENDAR_SPREAD",
    "DIRECTIONAL_BUY",
    "STAY_FLAT",
]
MINIMUM_ROWS = 50


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    if context.fo_eod_rows < MINIMUM_ROWS:
        return untrained_output(
            MODEL_ID,
            "fo_eod data unavailable; strategy labels require historical option P&L",
            available_rows=context.fo_eod_rows,
            minimum_rows=MINIMUM_ROWS,
            extra={"data_available": False},
        )
    return untrained_output(
        MODEL_ID,
        "strategy backtest label builder not available in current milestone",
        available_rows=context.fo_eod_rows,
        minimum_rows=MINIMUM_ROWS,
        extra={"data_available": True},
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    strategy, reasoning = _deterministic_strategy(context)
    ready = strategy is not None
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "strategy": strategy,
        "prediction": strategy,
        "probabilities": {},
        "confidence": None,
        "is_valid": ready,
        "status": "READY_ANALYTICAL" if ready else "UNAVAILABLE",
        "reason_code": None if ready else "REQUIRED_OPTIONS_INPUTS_UNAVAILABLE",
        "confidence_formula": None,
        "reasoning": reasoning,
        "accuracy_30d": accuracy_for_model(MODEL_ID)["accuracy"],
        "data_available": context.fo_eod_rows >= MINIMUM_ROWS,
        "available_rows": context.fo_eod_rows,
        "minimum_rows": MINIMUM_ROWS,
        "reason": None if context.fo_eod_rows >= MINIMUM_ROWS else "fo_eod empty; using deterministic safety selector",
    }


def _deterministic_strategy(context: MLContext) -> tuple[str | None, list[str]]:
    iv_rank = current_value(context, ["iv_rank", "options.iv_rank"])
    gex = current_value(context, ["gex", "options.gex"])
    pcr = current_value(context, ["pcr", "options.pcr", "options.pcr_total"])
    dte = current_value(context, ["dte", "expiry.dte"])
    reasoning: list[str] = []

    if iv_rank is None or gex is None or pcr is None or dte is None:
        reasoning.append(
            "Required options inputs (iv_rank/gex/pcr/dte) unavailable; no option-edge signal issued."
        )
        return None, reasoning

    if iv_rank >= 70 and gex >= 0:
        reasoning.append(f"IV rank {iv_rank:.0f}: options expensive, favour defined short-vol.")
        reasoning.append("Positive GEX: mean-reversion regime.")
        return "IRON_CONDOR" if dte > 2 else "SELL_STRADDLE", reasoning
    if iv_rank <= 30 and gex < 0:
        reasoning.append(f"IV rank {iv_rank:.0f}: options cheap.")
        reasoning.append("Negative GEX: move amplification risk.")
        return "BUY_STRADDLE", reasoning
    if pcr > 1.25:
        reasoning.append(f"PCR {pcr:.2f}: put-heavy market, contrarian bullish setup.")
        return "BULL_CALL_SPREAD", reasoning
    if pcr < 0.65:
        reasoning.append(f"PCR {pcr:.2f}: call-heavy market, bearish hedge favoured.")
        return "BEAR_PUT_SPREAD", reasoning
    reasoning.append("No durable option edge detected from current signals.")
    return "STAY_FLAT", reasoning
