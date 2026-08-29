from __future__ import annotations

from typing import Any

import pandas as pd


def compute_drawdown(symbol: str, ohlcv: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(ohlcv.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    if close.empty:
        return {"symbol": symbol, "error": "insufficient OHLCV history"}
    peaks = close.cummax()
    drawdowns = (close / peaks - 1.0) * 100.0
    durations = _drawdown_durations(drawdowns)
    return {
        "symbol": symbol,
        "max_drawdown_pct": round(float(drawdowns.min()), 3),
        "current_drawdown_pct": round(float(drawdowns.iloc[-1]), 3),
        "average_drawdown_duration_days": round(float(sum(durations) / len(durations)), 3) if durations else 0.0,
        "recovery_time_distribution": {
            "p50_days": _quantile(durations, 0.5),
            "p90_days": _quantile(durations, 0.9),
        },
        "drawdown_probability_current_regime": round(min(1.0, abs(float(drawdowns.iloc[-1])) / 12.0), 6),
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _drawdown_durations(drawdowns: pd.Series) -> list[int]:
    durations = []
    current = 0
    for value in drawdowns:
        if value < 0:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)
    return durations


def _quantile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    series = pd.Series(values)
    return int(series.quantile(q))
