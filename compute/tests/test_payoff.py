from __future__ import annotations

import pytest

from options.payoff import compute_payoff


def test_long_call_reference_payoff() -> None:
    result = compute_payoff(
        [
            {
                "symbol": "NIFTY",
                "strike": 23_400,
                "expiry": "2026-08-27",
                "option_type": "CE",
                "action": "BUY",
                "quantity": 1,
                "entry_price": 221,
                "lot_size": 75,
            }
        ],
        23_400,
    )

    assert result["breakevens"] == [23_621.0]
    assert result["max_loss"] == -16_575.0
    assert result["max_profit"] is None


@pytest.mark.parametrize("quantity", [0, -1])
def test_payoff_rejects_non_positive_quantity(quantity: int) -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        compute_payoff(
            [
                {
                    "strike": 23_400,
                    "expiry": "2026-08-27",
                    "option_type": "CE",
                    "action": "BUY",
                    "quantity": quantity,
                    "entry_price": 221,
                    "lot_size": 75,
                }
            ],
            23_400,
        )


def test_solvable_leg_reports_not_degraded() -> None:
    result = compute_payoff(
        [
            {
                "symbol": "NIFTY",
                "strike": 23_400,
                "expiry": "2026-08-27",
                "option_type": "CE",
                "action": "BUY",
                "quantity": 1,
                "entry_price": 221,
                "lot_size": 75,
            }
        ],
        23_400,
    )

    assert result["pnl_today_degraded"] is False


def test_unsolvable_iv_reports_degraded_not_silently_fabricated() -> None:
    """A near-zero entry_price cannot be inverted to a sane implied vol --
    options/payoff.py::_leg_iv falls back to a hardcoded FALLBACK_IV in that
    case. The caller must be told the "today" curve and Greeks used a
    generic assumption rather than one solved from the leg's own price,
    instead of silently presenting a Black-Scholes curve as if it reflected
    the position described."""
    result = compute_payoff(
        [
            {
                "symbol": "NIFTY",
                "strike": 23_400,
                "expiry": "2026-08-27",
                "option_type": "CE",
                "action": "BUY",
                "quantity": 1,
                "entry_price": 0.01,
                "lot_size": 75,
            }
        ],
        23_400,
    )

    assert result["pnl_today_degraded"] is True
    # pure intrinsic-value math (no IV involved) must remain unaffected by
    # the IV fallback used only for the secondary pnl_today/Greeks curves
    assert result["max_loss"] == pytest.approx(-0.75, abs=0.01)


def test_missing_lot_size_is_never_silently_defaulted() -> None:
    with pytest.raises(ValueError, match="lot_size is required"):
        compute_payoff(
            [
                {
                    "symbol": "NIFTY",
                    "strike": 23_400,
                    "expiry": "2026-08-27",
                    "option_type": "CE",
                    "action": "BUY",
                    "quantity": 1,
                    "entry_price": 221,
                }
            ],
            23_400,
        )
