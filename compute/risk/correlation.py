from __future__ import annotations

from typing import Any

import pandas as pd

from db.write_service import ReadOperation, resolve_read


ASSETS = ["NIFTY", "NIFTY 50", "DOW", "SPX", "NIKKEI", "BRENT", "GOLD", "USDINR", "INDIA VIX"]


def compute_correlations(symbol: str) -> dict[str, Any]:
    frame = _load_asset_returns()
    if frame.empty or symbol not in frame.columns:
        return {
            "symbol": symbol,
            "matrix_30d": {},
            "matrix_90d": {},
            "contagion_signal": False,
            "reason": "cross-asset history unavailable in DuckDB",
        }
    corr30 = frame.tail(30).corr().round(4).to_dict()
    corr90 = frame.tail(90).corr().round(4).to_dict()
    latest = frame.tail(30).corr()
    high_corr = int((latest.abs() > 0.8).sum().sum() - len(latest.columns))
    return {
        "symbol": symbol,
        "matrix_30d": corr30,
        "matrix_90d": corr90,
        "correlation_regime_change": _regime_change(frame),
        "contagion_signal": high_corr >= 5,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _load_asset_returns() -> pd.DataFrame:
    result = resolve_read(ReadOperation.LOAD_CROSS_ASSET_RETURNS, {})
    rows = pd.DataFrame(result["rows"], columns=result["columns"])
    if rows.empty:
        return pd.DataFrame()
    pivot = rows.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    returns = pivot.pct_change() * 100.0
    if "NIFTY 50" in returns.columns and "NIFTY" not in returns.columns:
        returns["NIFTY"] = returns["NIFTY 50"]
    return returns.dropna(how="all")


def _regime_change(frame: pd.DataFrame) -> bool:
    if len(frame) < 120 or "NIFTY" not in frame.columns:
        return False
    corr30 = frame.tail(30).corr().get("NIFTY", pd.Series(dtype=float)).mean()
    corr90 = frame.tail(90).corr().get("NIFTY", pd.Series(dtype=float)).mean()
    try:
        return abs(float(corr30 - corr90)) > 0.25
    except Exception:
        return False
