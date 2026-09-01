from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_var(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    spot: float | None = None,
    lot_size: int | None = None,
) -> dict[str, Any]:
    if lot_size is None or lot_size <= 0:
        return {"symbol": symbol, "methods": [], "error": "validated lot size unavailable"}
    if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
        return {"symbol": symbol, "methods": [], "error": "verified live spot unavailable"}
    close = pd.to_numeric(ohlcv.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(close) < 30:
        return {"symbol": symbol, "methods": [], "error": "insufficient OHLCV history"}
    returns = close.pct_change().dropna().tail(252).to_numpy(dtype=float)
    current_spot = float(spot)
    notional = current_spot * lot_size
    methods = [
        _historical_var(returns, notional),
        _parametric_var(returns, notional),
        _monte_carlo_var(returns, notional),
    ]
    return {
        "symbol": symbol,
        "notional_per_lot": round(notional, 2),
        "lot_size": lot_size,
        "methods": methods,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _historical_var(returns: np.ndarray, notional: float) -> dict[str, Any]:
    pnl = returns * notional
    return _var_payload("historical", pnl)


def _parametric_var(returns: np.ndarray, notional: float) -> dict[str, Any]:
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    simulated = np.array([mean - 1.64485 * std, mean - 2.32635 * std]) * notional
    return {
        "method": "parametric",
        "var_95_1d": round(abs(float(simulated[0])), 2),
        "var_99_1d": round(abs(float(simulated[1])), 2),
        "cvar_95_1d": round(abs(float((mean - 2.0627 * std) * notional)), 2),
        "var_95_5d": round(abs(float(simulated[0])) * np.sqrt(5.0), 2),
    }


def _monte_carlo_var(returns: np.ndarray, notional: float) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    paths = rng.normal(mean, std, size=10_000) * notional
    return _var_payload("monte_carlo", paths)


def _var_payload(method: str, pnl: np.ndarray) -> dict[str, Any]:
    var95 = abs(float(np.quantile(pnl, 0.05)))
    var99 = abs(float(np.quantile(pnl, 0.01)))
    tail = pnl[pnl <= np.quantile(pnl, 0.05)]
    cvar = abs(float(np.mean(tail))) if len(tail) else var95
    return {
        "method": method,
        "var_95_1d": round(var95, 2),
        "var_99_1d": round(var99, 2),
        "cvar_95_1d": round(cvar, 2),
        "var_95_5d": round(var95 * np.sqrt(5.0), 2),
    }
