from __future__ import annotations

from typing import Any

import pandas as pd

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read


def compute_alpha_signals(symbol: str, limit: int = 20, *, persist: bool = False) -> dict[str, Any]:
    frame = _load_signal_return_frame(symbol)
    if frame.empty:
        return {
            "symbol": symbol,
            "top_signals": [],
            "factors": [],
            "computed_at": pd.Timestamp.utcnow().isoformat(),
            "reason": "signals_history or OHLCV unavailable",
        }
    rows: list[dict[str, Any]] = []
    signal_cols = [col for col in frame.columns if col not in {"date", "future_return_1d"}]
    for col in signal_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        ic_30 = _corr(series.tail(30), frame["future_return_1d"].tail(30))
        ic_90 = _corr(series.tail(90), frame["future_return_1d"].tail(90))
        decay = {
            "1d": ic_30,
            "2d": _corr(series.tail(90), frame["future_return_1d"].shift(-1).tail(90)),
            "3d": _corr(series.tail(90), frame["future_return_1d"].shift(-2).tail(90)),
            "5d": _corr(series.tail(90), frame["future_return_1d"].shift(-4).tail(90)),
            "10d": _corr(series.tail(90), frame["future_return_1d"].shift(-9).tail(90)),
        }
        stability = None if ic_30 is None or ic_90 in (None, 0) else ic_30 / ic_90
        score = abs(ic_30 or 0.0) * (1.0 / (1.0 + abs((stability or 1.0) - 1.0)))
        rows.append({
            "signal_name": col,
            "ic_30d": ic_30,
            "ic_90d": ic_90,
            "ic_stability": stability,
            "decay_curve": decay,
            "score": score,
            "pca_factor": _factor_name(col),
        })
    rows.sort(key=lambda item: item["score"], reverse=True)
    top = rows[:limit]
    if persist:
        _write_alpha(symbol, top)
    return {
        "symbol": symbol,
        "top_signals": top,
        "factors": _factor_summary(top),
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _load_signal_return_frame(symbol: str) -> pd.DataFrame:
    ohlcv_symbol = "NIFTY 50" if symbol == "NIFTY" else symbol
    result = resolve_read(
        ReadOperation.LOAD_ALPHA_SIGNAL_RETURN_FRAME,
        {"symbol": symbol, "ohlcv_symbol": ohlcv_symbol},
    )
    signals = pd.DataFrame(result["signal_rows"], columns=["date", "signal_name", "value"])
    ohlcv = pd.DataFrame(result["ohlcv_rows"], columns=["date", "close"])
    if signals.empty or ohlcv.empty:
        return pd.DataFrame()
    pivot = signals.pivot_table(index="date", columns="signal_name", values="value", aggfunc="last")
    ohlcv["future_return_1d"] = pd.to_numeric(ohlcv["close"], errors="coerce").pct_change().shift(-1) * 100.0
    merged = pivot.reset_index().merge(ohlcv[["date", "future_return_1d"]], on="date", how="inner")
    return merged.dropna(subset=["future_return_1d"])


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 10 or frame.iloc[:, 0].nunique() < 2:
        return None
    return round(float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method="spearman")), 6)


def _factor_name(signal_name: str) -> str:
    text = signal_name.lower()
    if any(token in text for token in ("iv", "vix", "vol")):
        return "vol_factor"
    if any(token in text for token in ("fii", "dii", "pcr", "oi", "gex")):
        return "flow_factor"
    if any(token in text for token in ("rsi", "ema", "return", "trend")):
        return "momentum_factor"
    if any(token in text for token in ("risk", "drawdown", "var")):
        return "risk_factor"
    return "residual_factor"


def _factor_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[str, float] = {}
    for row in rows:
        factor = row["pca_factor"]
        summary[factor] = summary.get(factor, 0.0) + float(row["score"])
    total = sum(summary.values()) or 1.0
    return [
        {"factor": name, "exposure": round(value / total, 6)}
        for name, value in sorted(summary.items(), key=lambda item: item[1], reverse=True)
    ]


def _write_alpha(symbol: str, rows: list[dict[str, Any]]) -> None:
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS alpha_signals (
                    signal_name     TEXT        NOT NULL,
                    symbol          TEXT        NOT NULL,
                    computed_at     TIMESTAMP   NOT NULL,
                    ic_30d          DOUBLE,
                    ic_90d          DOUBLE,
                    ic_stability    DOUBLE,
                    ic_decay_3d     DOUBLE,
                    ic_decay_5d     DOUBLE,
                    pca_factor      TEXT,
                    PRIMARY KEY (signal_name, symbol)
                )
                """
            )
            for row in rows:
                con.execute(
                    """
                    INSERT OR REPLACE INTO alpha_signals
                        (signal_name, symbol, computed_at, ic_30d, ic_90d,
                         ic_stability, ic_decay_3d, ic_decay_5d, pca_factor)
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        row["signal_name"],
                        symbol,
                        row["ic_30d"],
                        row["ic_90d"],
                        row["ic_stability"],
                        row["decay_curve"].get("3d"),
                        row["decay_curve"].get("5d"),
                        row["pca_factor"],
                    ],
                )
        finally:
            con.close()
