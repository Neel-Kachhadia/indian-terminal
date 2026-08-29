import math
import time
from threading import Lock
from typing import Any

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
    "NIFTY": ["NIFTY 50", "NIFTY"],
    "NIFTY 50": ["NIFTY 50", "NIFTY"],
    "ETERNAL": ["ETERNAL", "ZOMATO"],
}

SNAPSHOT_FIELDS = {
    "close",
    "open",
    "high",
    "low",
    "volume",
    "change_pct",
    "volume_ratio",
    "week_52_high_pct",
    "week_52_low_pct",
}

OPERATORS = {
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "=": "=",
    "==": "=",
    "!=": "!=",
}

_UNIVERSE_CACHE: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}
_CACHE_LOCK = Lock()
_CACHE_TTL_SECONDS = 300


def get_stock_snapshot(symbol: str) -> dict[str, Any] | None:
    """
    Fast latest EOD snapshot from DuckDB ohlcv for one symbol.
    NIFTY maps to the stored NIFTY 50 OHLCV rows.
    """
    symbols = _query_symbols(symbol)
    result = resolve_read(ReadOperation.LOAD_STOCK_SNAPSHOT, {"symbols": symbols})
    rows = result["rows"]

    if not rows:
        return None
    return _row_to_snapshot(rows[0])


def get_universe_snapshot(
    symbols: list[str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Latest EOD snapshot for many symbols in one DuckDB query.
    If symbols is None, all OHLCV symbols are eligible and sorted by activity.
    """
    limit = _clean_limit(limit, 5000)
    cleaned = _expand_symbols(symbols) if symbols else None
    result = resolve_read(
        ReadOperation.LOAD_UNIVERSE_SNAPSHOT,
        {"symbols": cleaned, "limit": limit},
    )
    rows = result["rows"]

    return [_row_to_snapshot(row) for row in rows]


def get_stock_universe(universe: str = "FNO", limit: int = 250) -> list[dict[str, Any]]:
    normalized = universe.strip().upper()
    limit = _clean_limit(limit, 5000)
    cache_key = (normalized, limit)
    now = time.time()

    with _CACHE_LOCK:
        cached = _UNIVERSE_CACHE.get(cache_key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return [dict(item) for item in cached[1]]

    symbols = _universe_symbols(normalized)
    results = get_universe_snapshot(symbols, limit)

    with _CACHE_LOCK:
        _UNIVERSE_CACHE[cache_key] = (now, [dict(item) for item in results])
    return results


def run_screener_query(
    conditions: list[dict[str, Any]],
    logic: str = "AND",
    universe: str = "FNO",
    limit: int = 50,
) -> dict[str, Any]:
    """
    Run screener conditions against computed EOD snapshot fields.
    Condition filtering is pushed into DuckDB SQL.
    """
    normalized_logic = logic.strip().upper()
    if normalized_logic not in {"AND", "OR"}:
        raise ValueError("logic must be AND or OR")

    normalized_universe = universe.strip().upper()
    symbols = _universe_symbols(normalized_universe)
    limit = _clean_limit(limit, 500)

    scanned_symbols = _expand_symbols(symbols) if symbols else symbols
    result = resolve_read(
        ReadOperation.RUN_SCREENER_QUERY_ROWS,
        {
            "symbols": scanned_symbols,
            "conditions": conditions,
            "logic": normalized_logic,
            "limit": limit,
        },
    )
    rows = result["rows"]
    labels = result["labels"]

    results = [_row_to_snapshot(row) for row in rows]
    for item in results:
        item["ltp"] = item["close"]
        item["matched_conditions"] = labels
        item["score"] = float(len(labels) or 1)

    return {
        "results": results,
        "total_scanned": int(result["total_scanned"] or 0),
        "computed_at": time.time(),
    }


def _universe_symbols(universe: str) -> list[str] | None:
    if universe == "ALL":
        return None
    if universe == "NIFTY50":
        return NIFTY50_SYMBOLS
    if universe != "FNO":
        raise ValueError("universe must be FNO, NIFTY50, or ALL")

    result = resolve_read(ReadOperation.LOAD_FNO_UNIVERSE_SYMBOLS, {})
    return result["symbols"]


def _conditions_sql(
    conditions: list[dict[str, Any]],
    logic: str,
) -> tuple[str, list[Any], list[str]]:
    if not conditions:
        return "", [], []

    clauses: list[str] = []
    params: list[Any] = []
    labels: list[str] = []
    for item in conditions:
        field = str(item.get("field", "")).strip().lower()
        operator = str(item.get("operator", ">")).strip()
        if field not in SNAPSHOT_FIELDS:
            raise ValueError(f"unsupported field: {field}")
        if operator not in OPERATORS:
            raise ValueError(f"unsupported operator: {operator}")
        value = float(item.get("value", 0.0))
        clauses.append(f"{field} {OPERATORS[operator]} ?")
        params.append(value)
        labels.append(f"{field.replace('_', ' ')} {operator} {value:g}")

    return f"WHERE {f' {logic} '.join(clauses)}", params, labels


def _total_scanned_sql(symbols: list[str] | None) -> str:
    if not symbols:
        return "SELECT COUNT(DISTINCT symbol) FROM decision_eligible_ohlcv"
    placeholders = ", ".join("?" for _ in symbols)
    return f"SELECT COUNT(DISTINCT symbol) FROM decision_eligible_ohlcv WHERE symbol IN ({placeholders})"


def _query_symbols(symbol: str) -> list[str]:
    normalized = _canonical_symbol(symbol)
    return SYMBOL_ALIASES.get(normalized, [normalized])


def _expand_symbols(symbols: list[str]) -> list[str]:
    expanded: list[str] = []
    for symbol in symbols:
        expanded.extend(_query_symbols(symbol))
    return _dedupe(expanded)


def _canonical_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _dedupe(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def _clean_limit(limit: int, maximum: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        parsed = 100
    return max(1, min(parsed, maximum))


def _row_to_snapshot(row: Any) -> dict[str, Any]:
    symbol = str(row[0])
    date = str(row[1])
    open_ = _finite(row[2])
    high = _finite(row[3])
    low = _finite(row[4])
    close = _finite(row[5])
    volume = int(row[6] or 0)
    prev_close = _finite(row[7])
    week_52_high = _finite(row[8])
    week_52_low = _finite(row[9])
    avg_volume_20d = _finite(row[10])
    close_20d_ago = _finite(row[11]) if len(row) > 11 else None
    nifty_return_20d = _finite(row[12]) if len(row) > 12 else None

    change = None if close is None or prev_close is None else close - prev_close
    change_pct = _percent(change, prev_close)
    volume_ratio = None if not avg_volume_20d else volume / avg_volume_20d
    week_52_high_pct = _percent(_diff(close, week_52_high), week_52_high)
    week_52_low_pct = _percent(_diff(close, week_52_low), week_52_low)
    return_20d_pct = _percent(_diff(close, close_20d_ago), close_20d_ago)
    relative_strength = (
        None
        if return_20d_pct is None or nifty_return_20d is None
        else return_20d_pct - nifty_return_20d
    )

    return {
        "symbol": "NIFTY" if symbol == "NIFTY 50" else symbol,
        "date": date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        "avg_volume_20d": int(avg_volume_20d or 0),
        "volume_ratio": volume_ratio,
        "week_52_high_pct": week_52_high_pct,
        "week_52_low_pct": week_52_low_pct,
        "return_20d_pct": return_20d_pct,
        "relative_strength": relative_strength,
        "signal": _signal_label(change_pct, volume_ratio, relative_strength),
    }


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _percent(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * 100


def _signal_label(
    change_pct: float | None,
    volume_ratio: float | None,
    relative_strength: float | None,
) -> str:
    if volume_ratio is not None and volume_ratio >= 2.0:
        return "VOL SPIKE"
    if change_pct is not None and change_pct >= 1.0 and (
        relative_strength is None or relative_strength >= 0
    ):
        return "BULLISH"
    if change_pct is not None and change_pct <= -1.0 and (
        relative_strength is None or relative_strength <= 0
    ):
        return "BEARISH"
    return "NEUTRAL"
