from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regime_attribution(
    daily_records: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    equity = np.array([float(row.get("portfolio_value") or 0.0) for row in daily_records], dtype=np.float64)
    dates = [str(row.get("date")) for row in daily_records]
    returns = _returns(equity)
    regimes = _vol_regimes(returns, dates[1:])
    trade_rows = _trade_regime_rows(trades, regimes)

    by_vol = _aggregate(trade_rows, "vol_regime")
    by_bif = _aggregate(trade_rows, "bif_regime")
    best = _best_regime(by_vol)
    worst = _worst_regime(by_vol)
    return {
        "by_vol_regime": by_vol,
        "by_bif_regime": by_bif,
        "best_regime": best,
        "worst_regime": worst,
        "trade_regimes": trade_rows[:200],
    }


def _returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([], dtype=np.float64)
    prev = equity[:-1]
    cur = equity[1:]
    return np.divide(cur - prev, prev, out=np.zeros_like(cur), where=prev != 0)


def _vol_regimes(returns: np.ndarray, dates: list[str]) -> dict[str, str]:
    if returns.size == 0:
        return {}
    series = pd.Series(returns)
    rolling = series.rolling(20, min_periods=5).std().fillna(series.std() or 0.0) * np.sqrt(252.0)
    output: dict[str, str] = {}
    for date, vol in zip(dates, rolling, strict=False):
        if vol < 0.12:
            label = "LOW_VOL"
        elif vol > 0.22:
            label = "HIGH_VOL"
        else:
            label = "MED_VOL"
        output[str(date)] = label
    return output


def _trade_regime_rows(
    trades: list[dict[str, Any]],
    regimes: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for trade in trades:
        entry_date = str(trade.get("entry_date") or "")
        cds = _number(trade.get("bif_cds_entry"))
        rows.append(
            {
                "trade_id": trade.get("trade_id"),
                "entry_date": entry_date,
                "exit_date": trade.get("exit_date"),
                "net_pnl": _number(trade.get("net_pnl")) or 0.0,
                "vol_regime": regimes.get(entry_date) or trade.get("regime_at_entry") or "UNKNOWN",
                "bif_regime": _bif_label(cds),
                "iv_at_entry": _number(trade.get("iv_at_entry")),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in rows:
        label = str(row.get(key) or "UNKNOWN")
        bucket = output.setdefault(
            label,
            {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "avg_pnl": 0.0, "win_rate": 0.0},
        )
        pnl = float(row.get("net_pnl") or 0.0)
        bucket["trades"] += 1
        bucket["wins"] += int(pnl > 0)
        bucket["losses"] += int(pnl < 0)
        bucket["total_pnl"] += pnl
    for bucket in output.values():
        trades = int(bucket["trades"])
        bucket["avg_pnl"] = round(bucket["total_pnl"] / trades, 2) if trades else 0.0
        bucket["total_pnl"] = round(bucket["total_pnl"], 2)
        bucket["win_rate"] = round(bucket["wins"] / trades, 4) if trades else 0.0
    return output


def _best_regime(buckets: dict[str, Any]) -> str | None:
    if not buckets:
        return None
    return max(buckets.items(), key=lambda item: float(item[1].get("avg_pnl") or 0.0))[0]


def _worst_regime(buckets: dict[str, Any]) -> str | None:
    if not buckets:
        return None
    return min(buckets.items(), key=lambda item: float(item[1].get("avg_pnl") or 0.0))[0]


def _bif_label(cds: float | None) -> str:
    if cds is None:
        return "UNKNOWN"
    if cds > 65:
        return "BULL"
    if cds < 35:
        return "BEAR"
    return "NEUTRAL"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
