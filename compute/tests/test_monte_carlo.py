from __future__ import annotations

from backtest.monte_carlo import monte_carlo_summary


def test_monte_carlo_is_deterministic_for_fixed_seed() -> None:
    trades = [{"net_pnl": pnl} for pnl in (100.0, -50.0, 75.0, -25.0)]

    first = monte_carlo_summary(trades, 100_000.0, n_simulations=500, seed=7)
    second = monte_carlo_summary(trades, 100_000.0, n_simulations=500, seed=7)

    assert first == second
    assert first["assumptions"]["serial_dependence_preserved"] is False
