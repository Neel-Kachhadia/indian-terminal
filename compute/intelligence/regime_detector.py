from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from db.duckdb_client import DB_WRITE_LOCK, writer_connection


STATE_LABELS = {
    0: "CALM",
    1: "GREED",
    2: "FEAR",
    3: "PANIC",
}


def detect_regime(
    symbol: str,
    ohlcv: pd.DataFrame,
    signals: dict[str, Any] | None = None,
    *,
    persist: bool = False,
) -> dict[str, Any]:
    as_of_date = _as_of_date(ohlcv)
    close = pd.to_numeric(ohlcv.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(close) < 30:
        payload = _fallback_payload(symbol, "insufficient OHLCV history")
        payload["as_of_date"] = as_of_date
        if persist:
            _write_regime(symbol, payload)
        return payload

    returns = close.pct_change().dropna() * 100.0
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0 if len(close) > 6 else 0.0
    vol_20 = float(returns.tail(20).std(ddof=0) * np.sqrt(252.0))
    vix = _num(((signals or {}).get("market") or {}).get("india_vix"))
    state = _state_from_metrics(ret_5d, vol_20, vix)
    probs = _state_probabilities(state, ret_5d, vol_20, vix)
    change_point, days_ago = _change_point(returns)
    days_in_state = _days_in_current_regime(close, state)
    payload = {
        "symbol": symbol,
        "current_state": state,
        "state_label": STATE_LABELS[state],
        "state_probabilities": probs,
        "regime_transition_warning": bool(change_point and (days_ago or 99) <= 5),
        "change_point_detected": bool(change_point),
        "change_point_days_ago": days_ago,
        "days_in_current_regime": days_in_state,
        "model_used": "rule_based_hmm_fallback",
        "is_trained": False,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "as_of_date": as_of_date,
    }
    if persist:
        _write_regime(symbol, payload)
    return payload


def _fallback_payload(symbol: str, reason: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "current_state": 0,
        "state_label": "CALM",
        "state_probabilities": {"CALM": 1.0, "GREED": 0.0, "FEAR": 0.0, "PANIC": 0.0},
        "regime_transition_warning": False,
        "change_point_detected": False,
        "change_point_days_ago": None,
        "days_in_current_regime": 0,
        "model_used": "rule_based_hmm_fallback",
        "is_trained": False,
        "reason": reason,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _as_of_date(ohlcv: pd.DataFrame) -> str | None:
    if ohlcv.empty or "date" not in ohlcv.columns:
        return None
    values = pd.to_datetime(ohlcv["date"], errors="coerce").dropna()
    return values.max().date().isoformat() if not values.empty else None


def _state_from_metrics(ret_5d: float, vol_20: float, vix: float | None) -> int:
    stress_vol = max(vol_20, float(vix or 0.0))
    if stress_vol > 24 or ret_5d < -5:
        return 3
    if stress_vol > 16 or ret_5d < -2:
        return 2
    if ret_5d > 3 and stress_vol < 16:
        return 1
    return 0


def _state_probabilities(state: int, ret_5d: float, vol_20: float, vix: float | None) -> dict[str, float]:
    stress = min(1.0, max(vol_20, float(vix or 0.0)) / 30.0)
    trend = max(-1.0, min(1.0, ret_5d / 5.0))
    raw = {
        "CALM": max(0.05, 1.0 - stress - max(trend, 0.0) * 0.2),
        "GREED": max(0.05, (1.0 - stress) * max(trend, 0.0)),
        "FEAR": max(0.05, stress * max(-trend, 0.2)),
        "PANIC": max(0.05, stress * max(-trend, 0.0) * 1.4),
    }
    raw[STATE_LABELS[state]] += 0.35
    total = sum(raw.values()) or 1.0
    return {key: round(value / total, 6) for key, value in raw.items()}


def _change_point(returns: pd.Series) -> tuple[bool, int | None]:
    if len(returns) < 60:
        return False, None
    rolling = returns.rolling(20).std(ddof=0)
    baseline = rolling.iloc[-60:-20].mean()
    recent = rolling.iloc[-5:].mean()
    if baseline and recent > baseline * 1.6:
        return True, 2
    return False, None


def _days_in_current_regime(close: pd.Series, state: int) -> int:
    returns = close.pct_change().dropna() * 100.0
    days = 0
    for index in range(len(returns) - 1, 20, -1):
        window = returns.iloc[index - 20:index]
        ret_5d = float(close.iloc[index] / close.iloc[index - 5] - 1.0) * 100.0
        vol_20 = float(window.std(ddof=0) * np.sqrt(252.0))
        if _state_from_metrics(ret_5d, vol_20, None) != state:
            break
        days += 1
    return days


def _write_regime(symbol: str, payload: dict[str, Any]) -> None:
    as_of_date = payload.get("as_of_date")
    if not as_of_date:
        raise ValueError("persistent regime computation requires a market-data as-of date")
    probs = payload.get("state_probabilities", {})
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS regime_history (
                    date            DATE        NOT NULL,
                    symbol          TEXT        NOT NULL,
                    hmm_state       INTEGER,
                    state_label     TEXT,
                    state_prob_0    DOUBLE,
                    state_prob_1    DOUBLE,
                    state_prob_2    DOUBLE,
                    state_prob_3    DOUBLE,
                    change_point    INTEGER     DEFAULT 0,
                    PRIMARY KEY (date, symbol)
                )
                """
            )
            con.execute(
                """
                INSERT OR REPLACE INTO regime_history
                    (date, symbol, hmm_state, state_label, state_prob_0, state_prob_1,
                     state_prob_2, state_prob_3, change_point)
                VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    as_of_date,
                    symbol,
                    payload.get("current_state"),
                    payload.get("state_label"),
                    probs.get("CALM"),
                    probs.get("GREED"),
                    probs.get("FEAR"),
                    probs.get("PANIC"),
                    1 if payload.get("change_point_detected") else 0,
                ],
            )
        finally:
            con.close()


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
