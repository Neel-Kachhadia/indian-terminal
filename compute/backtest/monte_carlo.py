from __future__ import annotations

from typing import Any

import numpy as np


def monte_carlo_summary(
    trades: list[dict[str, Any]],
    initial_capital: float,
    n_simulations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    pnls = np.array([float(trade.get("net_pnl") or 0.0) for trade in trades], dtype=np.float64)
    if pnls.size == 0:
        return {
            "simulations": 0,
            "return_p5": 0.0,
            "return_p50": 0.0,
            "return_p95": 0.0,
            "max_drawdown_p95": 0.0,
            "probability_of_ruin": 0.0,
            "verdict": "INSUFFICIENT_TRADES",
            "assumptions": _assumptions(seed),
        }

    n_simulations = max(100, min(int(n_simulations), 50000))
    rng = np.random.default_rng(seed)
    paths = rng.choice(pnls, size=(n_simulations, pnls.size), replace=True)
    equity_paths = initial_capital + np.cumsum(paths, axis=1)
    final_returns = (equity_paths[:, -1] / initial_capital - 1.0) * 100.0 if initial_capital else np.zeros(n_simulations)
    running_high = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = np.divide(equity_paths - running_high, running_high, out=np.zeros_like(equity_paths), where=running_high != 0)
    max_dd = np.min(drawdowns, axis=1) * 100.0
    ruin = np.mean(np.min(equity_paths, axis=1) < initial_capital * 0.5) * 100.0

    return {
        "simulations": n_simulations,
        "return_p5": round(float(np.percentile(final_returns, 5)), 3),
        "return_p50": round(float(np.percentile(final_returns, 50)), 3),
        "return_p95": round(float(np.percentile(final_returns, 95)), 3),
        "max_drawdown_p50": round(float(np.percentile(max_dd, 50)), 3),
        "max_drawdown_p95": round(float(np.percentile(max_dd, 5)), 3),
        "probability_of_ruin": round(float(ruin), 3),
        "verdict": "ACCEPTABLE_RISK" if ruin < 5.0 else "RUIN_RISK_HIGH",
        "assumptions": _assumptions(seed),
    }


def _assumptions(seed: int) -> dict[str, Any]:
    return {
        "method": "iid_trade_pnl_resampling_with_replacement",
        "seed": int(seed),
        "serial_dependence_preserved": False,
        "regime_dependence_preserved": False,
        "interpretation": "distributional sensitivity only; not a forward-performance claim",
    }
