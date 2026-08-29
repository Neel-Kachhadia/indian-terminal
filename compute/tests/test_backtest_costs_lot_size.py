from __future__ import annotations

from dataclasses import dataclass

import pytest

from backtest.costs import IndianMarketCosts


@dataclass
class _FakeLeg:
    option_type: str
    strike: float
    lots: int
    lot_size: object


def test_compute_leg_cost_scales_from_leg_lot_size_not_hardcoded_one() -> None:
    engine = IndianMarketCosts()
    leg_65 = _FakeLeg(option_type="CE", strike=25_000, lots=1, lot_size=65)
    leg_75 = _FakeLeg(option_type="CE", strike=25_000, lots=1, lot_size=75)

    cost_65 = engine.compute_leg_cost(leg_65, 100.0, "BUY", 25_000)
    cost_75 = engine.compute_leg_cost(leg_75, 100.0, "BUY", 25_000)

    # brokerage is a flat per-order fee, so only the turnover-derived
    # components (which scale with lot_size) are compared directly.
    assert cost_65.stamp == pytest.approx(cost_75.stamp * 65 / 75)
    assert cost_65.slippage == pytest.approx(cost_75.slippage * 65 / 75)
    assert cost_65.exchange == pytest.approx(cost_75.exchange * 65 / 75)


def test_compute_leg_cost_rejects_missing_lot_size_instead_of_defaulting_to_one() -> None:
    """Regression: compute_leg_cost previously used getattr(leg, "lot_size", 1),
    which would silently cost a position as a single-unit lot instead of
    failing when a leg object had no validated lot_size."""
    engine = IndianMarketCosts()
    leg_without_lot_size = _FakeLeg(option_type="CE", strike=25_000, lots=1, lot_size=None)

    with pytest.raises(ValueError, match="lot_size"):
        engine.compute_leg_cost(leg_without_lot_size, 100.0, "BUY", 25_000)


def test_compute_leg_cost_rejects_zero_or_negative_lot_size() -> None:
    engine = IndianMarketCosts()
    leg = _FakeLeg(option_type="CE", strike=25_000, lots=1, lot_size=0)

    with pytest.raises(ValueError, match="lot_size"):
        engine.compute_leg_cost(leg, 100.0, "BUY", 25_000)
