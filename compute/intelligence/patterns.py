from __future__ import annotations

from typing import Any

import pandas as pd

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read


def detect_patterns(
    symbol: str,
    signals: dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    persist: bool = False,
) -> dict[str, Any]:
    options = signals.get("options", {}) if isinstance(signals, dict) else {}
    market = signals.get("market", {}) if isinstance(signals, dict) else {}
    fii = (market.get("fii_dii") or {}) if isinstance(market, dict) else {}
    scores = [
        _pattern("Vol Compression Setup", _score([
            _below(options.get("gex"), 0.0, invert=True),
            _below(market.get("india_vix"), 14.0),
            _above(_bb_squeeze(ohlcv), 0.5),
        ]), "ATM_STRADDLE"),
        _pattern("FII Accumulation Signal", _score([
            _above(fii.get("fii_net_cash"), 2000.0),
            _above(options.get("pcr"), 1.0),
            _below(options.get("iv_rank"), 30.0),
        ]), "BULL_SPREAD"),
        _pattern("Max Pain Magnetism", _score([
            _distance(options.get("max_pain"), signals.get("spot"), 50.0),
            _above(options.get("total_call_oi"), 1.0),
        ]), "IRON_CONDOR"),
        _pattern("Gamma Squeeze Setup", _score([
            _below(options.get("gex"), 0.0),
            _below(_rsi(ohlcv), 35.0),
            _above(fii.get("fii_net_cash"), -2000.0, invert=True),
        ]), "BEAR_SPREAD"),
        _pattern("IV Crush Opportunity", _score([
            _above(options.get("iv_rank"), 70.0),
            _above(options.get("atm_iv"), 0.12),
        ]), "IRON_CONDOR"),
        _pattern("Triple Confirmation Bullish", _score([
            _above(_ema_alignment(ohlcv), 0.75),
            _above(fii.get("fii_net_cash"), 0.0),
            _above(options.get("pcr"), 1.2),
        ]), "BULL_SPREAD"),
        _pattern("Distribution Signal", _score([
            _above(_near_high(ohlcv), 0.85),
            _above(fii.get("fii_net_cash"), -1.0, invert=True),
            _above(options.get("iv_rank"), 50.0),
        ]), "STAY_FLAT"),
        _pattern("Expiry Pinning", _score([
            _distance(options.get("max_pain"), signals.get("spot"), 100.0),
            _above(options.get("total_call_oi"), 1.0),
        ]), "SHORT_STRANGLE"),
    ]
    history = _pattern_history(symbol)
    for item in scores:
        item.update(history.get(item["pattern_name"], {}))
    as_of_values = pd.to_datetime(ohlcv.get("date", pd.Series(dtype=str)), errors="coerce").dropna()
    as_of_date = as_of_values.max().date().isoformat() if not as_of_values.empty else None
    if persist:
        if not as_of_date:
            raise ValueError("persistent pattern computation requires a market-data as-of date")
        _write_pattern_snapshot(symbol, scores, as_of_date)
    active = [item for item in scores if item["match_score"] >= 60.0]
    return {
        "symbol": symbol,
        "patterns": scores,
        "active_patterns": active,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "as_of_date": as_of_date,
    }


def _pattern(name: str, score: float, strategy: str) -> dict[str, Any]:
    return {
        "pattern_name": name,
        "match_score": round(score, 3),
        "historical_win_rate": None,
        "average_move_5d": None,
        "optimal_strategy": strategy,
    }


def _score(values: list[float | None]) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return 0.0
    return max(0.0, min(100.0, sum(clean) / len(clean) * 100.0))


def _above(value: Any, threshold: float, invert: bool = False) -> float | None:
    number = _num(value)
    if number is None:
        return None
    passed = number > threshold
    if invert:
        passed = not passed
    return 1.0 if passed else max(0.0, min(1.0, number / threshold if threshold else 0.0))


def _below(value: Any, threshold: float, invert: bool = False) -> float | None:
    number = _num(value)
    if number is None:
        return None
    passed = number < threshold
    if invert:
        passed = not passed
    return 1.0 if passed else 0.0


def _distance(a: Any, b: Any, threshold: float) -> float | None:
    left = _num(a)
    right = _num(b)
    if left is None or right is None:
        return None
    return max(0.0, 1.0 - abs(left - right) / threshold)


def _rsi(ohlcv: pd.DataFrame) -> float | None:
    if ohlcv.empty or "close" not in ohlcv.columns:
        return None
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0.0, pd.NA)
    value = 100.0 - 100.0 / (1.0 + rs.iloc[-1])
    return _num(value)


def _bb_squeeze(ohlcv: pd.DataFrame) -> float | None:
    if len(ohlcv) < 40:
        return None
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    width = close.rolling(20).std(ddof=0) / close.rolling(20).mean()
    current = width.iloc[-1]
    rank = (width.tail(120) <= current).mean()
    return 1.0 - float(rank)


def _ema_alignment(ohlcv: pd.DataFrame) -> float | None:
    if len(ohlcv) < 60:
        return None
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    emas = [close.ewm(span=span, adjust=False).mean().iloc[-1] for span in (9, 21, 50, 100)]
    score = sum(1 for left, right in zip(emas, emas[1:]) if left >= right) / 3.0
    return float(score)


def _near_high(ohlcv: pd.DataFrame) -> float | None:
    if len(ohlcv) < 60:
        return None
    close = pd.to_numeric(ohlcv["close"], errors="coerce").tail(252)
    low, high = close.min(), close.max()
    if high <= low:
        return None
    return float((close.iloc[-1] - low) / (high - low))


def _pattern_history(symbol: str) -> dict[str, dict[str, Any]]:
    try:
        result = resolve_read(ReadOperation.LOAD_PATTERN_HISTORY_STATS, {"symbol": symbol})
        rows = result["rows"]
    except Exception:
        rows = []
    return {
        str(name): {
            "historical_win_rate": None if win is None else round(float(win), 6),
            "average_move_5d": None if move is None else round(float(move), 6),
        }
        for name, win, move in rows
    }


def _write_pattern_snapshot(symbol: str, patterns: list[dict[str, Any]], as_of_date: str) -> None:
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_history (
                    date            DATE        NOT NULL,
                    pattern_name    TEXT        NOT NULL,
                    symbol          TEXT        NOT NULL,
                    match_score     DOUBLE,
                    outcome_5d      DOUBLE,
                    PRIMARY KEY (date, pattern_name, symbol)
                )
                """
            )
            for item in patterns:
                con.execute(
                    """
                    INSERT OR REPLACE INTO pattern_history
                        (date, pattern_name, symbol, match_score, outcome_5d)
                    VALUES (CAST(? AS DATE), ?, ?, ?, NULL)
                    """,
                    [as_of_date, item["pattern_name"], symbol, item["match_score"]],
                )
        finally:
            con.close()


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number
