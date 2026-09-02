"""Regression tests: UNKNOWN/unavailable options inputs must never be
silently coalesced into a financially meaningful neutral value (IV rank 50,
PCR 1.0, GEX/overall_score 0/50, etc.) before feeding an actionable
strategy-selection model. See P17 handoff: unknown IV != 50, unknown PCR != 1.0.
"""
from __future__ import annotations

import pandas as pd

from ml.models import m03_strategy_selector, m18_bandit


class _FakeContext:
    """Duck-typed stand-in for MLContext: current_value() only reads
    .current_features (dict) and .current_vector (pd.Series)."""

    def __init__(self, features: dict[str, float] | None = None) -> None:
        self.current_features = features or {}
        self.current_vector = pd.Series(dtype="float64")


def test_m03_strategy_selector_is_unavailable_when_options_inputs_missing() -> None:
    strategy, reasoning = m03_strategy_selector._deterministic_strategy(_FakeContext())
    assert strategy is None
    assert any("unavailable" in line for line in reasoning)


def test_m03_strategy_selector_does_not_act_on_partial_fabricated_inputs() -> None:
    # Old behavior: iv_rank=80 (real) with gex missing defaulted gex to 0.0,
    # which satisfied `iv_rank >= 70 and gex >= 0` and issued a confident
    # IRON_CONDOR/SELL_STRADDLE recommendation from data that was actually
    # unavailable. gex/pcr/dte unknown must block the recommendation instead.
    ctx = _FakeContext({"iv_rank": 80.0})  # gex/pcr/dte still missing
    strategy, reasoning = m03_strategy_selector._deterministic_strategy(ctx)
    assert strategy is None
    assert any("unavailable" in line for line in reasoning)


def test_m03_strategy_selector_uses_real_inputs_when_present() -> None:
    ctx = _FakeContext({"iv_rank": 80.0, "gex": 1.0, "pcr": 1.0, "dte": 5.0})
    strategy, reasoning = m03_strategy_selector._deterministic_strategy(ctx)
    assert strategy in {"IRON_CONDOR", "SELL_STRADDLE"}
    assert reasoning


def test_m18_bandit_discloses_missing_inputs_instead_of_hiding_fabrication() -> None:
    # Old behavior: iv_rank/pcr/overall_score silently defaulted to 50/1.0/50
    # and the response always claimed "Thompson priors only; update after
    # live outcomes" even when every input was fabricated. The reason string
    # must now disclose that the priors were not backed by real data.
    result = m18_bandit.predict(_FakeContext())
    assert "unavailable" in result["reason"]
    assert result["reason"] != "Thompson priors only; update after live outcomes"


def test_m18_bandit_does_not_recommend_without_resolved_outcomes() -> None:
    ctx = _FakeContext({"iv_rank": 20.0, "pcr": 1.0, "overall_score": 70.0})
    result = m18_bandit.predict(ctx)
    assert result["recommended_strategy"] is None
    assert result["strategy_probabilities"] == {}
    assert result["status"] == "INSUFFICIENT_DATA"
    assert "outcomes" in result["reason"]
