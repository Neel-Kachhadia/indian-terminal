import math
import time
from typing import Any

import pandas as pd
import pandas_ta as ta

from db.write_service import ReadOperation, resolve_read


NIFTY50_SYMBOLS = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDIGO",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SHRIRAMFIN",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]

SYMBOL_ALIASES = {
    "ETERNAL": ["ETERNAL", "ZOMATO"],
}

SUPPORTED_FIELDS = {
    "close",
    "open",
    "high",
    "low",
    "volume",
    "rsi_14",
    "ema_20",
    "ema_50",
    "above_ema_20",
    "above_ema_50",
    "volume_spike",
}


def run_screener(
    conditions: list[dict[str, Any]],
    logic: str,
    universe: str,
    limit: int,
) -> dict[str, Any]:
    """
    Run Milestone 6 daily screener against DuckDB OHLCV rows.
    Only NIFTY50 universe is supported in this milestone.
    """
    normalized_universe = universe.strip().upper()
    if normalized_universe != "NIFTY50":
        raise ValueError("only NIFTY50 universe is supported in Milestone 6")

    normalized_logic = logic.strip().upper()
    if normalized_logic not in {"AND", "OR"}:
        raise ValueError("logic must be AND or OR")

    cleaned_conditions = [_clean_condition(item) for item in conditions]
    limit = max(1, min(int(limit or 50), 200))
    history = _load_universe_history(NIFTY50_SYMBOLS)

    results: list[dict[str, Any]] = []
    for symbol in NIFTY50_SYMBOLS:
        df = history.get(symbol)
        if df is None or df.empty:
            continue

        fields = _compute_fields(df)
        matches = [
            _evaluate_condition(condition, fields)
            for condition in cleaned_conditions
        ]

        if not cleaned_conditions:
            is_match = True
        elif normalized_logic == "AND":
            is_match = all(match is not None for match in matches)
        else:
            is_match = any(match is not None for match in matches)

        if not is_match:
            continue

        matched_conditions = [match for match in matches if match is not None]
        score = (
            1.0
            if not cleaned_conditions
            else len(matched_conditions) / len(cleaned_conditions)
        )
        results.append(
            {
                "symbol": symbol,
                "ltp": float(fields["close"]),
                "matched_conditions": matched_conditions,
                "score": float(score),
            }
        )

    results.sort(key=lambda item: (item["score"], item["ltp"]), reverse=True)
    return {
        "results": results[:limit],
        "total_scanned": len(NIFTY50_SYMBOLS),
        "computed_at": time.time(),
    }


def _load_universe_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    lookup_symbols: list[str] = []
    reverse_alias: dict[str, str] = {}
    for symbol in symbols:
        aliases = SYMBOL_ALIASES.get(symbol, [symbol])
        for alias in aliases:
            lookup_symbols.append(alias)
            reverse_alias[alias] = symbol

    result = resolve_read(
        ReadOperation.LOAD_SCREENER_UNIVERSE_HISTORY,
        {"symbols": lookup_symbols},
    )
    rows = result["rows"]

    records_by_symbol: dict[str, list[dict[str, Any]]] = {
        symbol: [] for symbol in symbols
    }
    for row in rows:
        canonical = reverse_alias.get(str(row[0]).upper())
        if canonical is None:
            continue
        records_by_symbol[canonical].append(
            {
                "date": str(row[1]),
                "open": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "close": float(row[5]),
                "volume": int(row[6]),
            }
        )

    frames: dict[str, pd.DataFrame] = {}
    for symbol, records in records_by_symbol.items():
        if records:
            frames[symbol] = pd.DataFrame(records).sort_values("date")

    return frames


def _compute_fields(df: pd.DataFrame) -> dict[str, float | bool]:
    close = df["close"]
    volume = df["volume"]
    latest = df.iloc[-1]
    ema_20 = _last_number(_ema(close, 20))
    ema_50 = _last_number(_ema(close, 50))
    rsi_14 = _last_number(_rsi(close, 14))
    average_volume = volume.rolling(20, min_periods=20).mean().iloc[-1]

    fields: dict[str, float | bool] = {
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "close": float(latest["close"]),
        "volume": float(latest["volume"]),
        "rsi_14": rsi_14,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "above_ema_20": _valid(ema_20) and float(latest["close"]) > ema_20,
        "above_ema_50": _valid(ema_50) and float(latest["close"]) > ema_50,
        "volume_spike": (
            _valid(float(average_volume))
            and float(average_volume) > 0
            and float(latest["volume"]) > 2 * float(average_volume)
        ),
    }
    return fields


def _evaluate_condition(
    condition: dict[str, Any],
    fields: dict[str, float | bool],
) -> str | None:
    field = str(condition["field"])
    value = fields.get(field)
    if value is None:
        return None

    if isinstance(value, bool):
        return _describe_condition(condition, value) if value else None

    if not _valid(value):
        return None

    target = float(condition["value"])
    operator = str(condition["operator"])
    matched = _compare(float(value), operator, target)
    return _describe_condition(condition, value) if matched else None


def _clean_condition(condition: dict[str, Any]) -> dict[str, Any]:
    field = str(condition.get("field", "")).strip().lower()
    operator = str(condition.get("operator", ">")).strip()
    timeframe = str(condition.get("timeframe", "1d")).strip().lower()

    if field not in SUPPORTED_FIELDS:
        raise ValueError(f"unsupported field: {field}")
    if operator not in {">", "<", ">=", "<=", "=="}:
        raise ValueError(f"unsupported operator: {operator}")
    if timeframe != "1d":
        raise ValueError("only 1d timeframe is supported in Milestone 6")

    return {
        "field": field,
        "operator": operator,
        "value": float(condition.get("value", 0.0)),
        "timeframe": timeframe,
    }


def _compare(actual: float, operator: str, expected: float) -> bool:
    if operator == ">":
        return actual > expected
    if operator == "<":
        return actual < expected
    if operator == ">=":
        return actual >= expected
    if operator == "<=":
        return actual <= expected
    if operator == "==":
        return actual == expected
    raise ValueError(f"unsupported operator: {operator}")


def _describe_condition(condition: dict[str, Any], value: float | bool) -> str:
    field = str(condition["field"])
    if isinstance(value, bool):
        return field.replace("_", " ")

    return (
        f"{field.replace('_', ' ')} {condition['operator']} "
        f"{condition['value']:g} ({float(value):.2f})"
    )


def _ema(series: pd.Series, period: int) -> pd.Series:
    result = ta.ema(series, length=period)
    if result is None:
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    return result


def _rsi(series: pd.Series, period: int) -> pd.Series:
    result = ta.rsi(series, length=period)
    if result is not None:
        return result

    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _last_number(series: pd.Series) -> float:
    if series.empty:
        return math.nan
    value = series.iloc[-1]
    if value is None or pd.isna(value):
        return math.nan
    parsed = float(value)
    return parsed if math.isfinite(parsed) else math.nan


def _valid(value: float) -> bool:
    return math.isfinite(float(value))
