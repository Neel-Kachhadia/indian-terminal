import datetime
import math
from typing import Any

import pandas as pd
import pandas_ta as ta

from db.duckdb_client import get_connection


def compute_indicators(
    symbol: str,
    timeframe: str,
    from_date: str | None,
    to_date: str | None,
    indicators: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Fetch daily OHLCV from DuckDB and return indicator arrays aligned to dates.
    Intraday timeframes arrive in Milestone 7; VWAP returns nulls for daily.
    """
    if timeframe != "1d":
        raise ValueError("only 1d timeframe is supported in Milestone 6")

    from_day, to_day = _date_range(from_date, to_date)
    df = _load_ohlcv(symbol, from_day, to_day)
    timestamps = [str(value) for value in df["date"].tolist()]
    output: dict[str, list[float | None]] = {}

    for item in indicators:
        name = str(item.get("name", "")).strip().upper()
        params = item.get("params") if isinstance(item.get("params"), dict) else {}

        if name == "EMA":
            period = _int_param(params, "period", 20)
            output[f"EMA_{period}"] = _to_json_values(_ema(df["close"], period))
        elif name == "SMA":
            period = _int_param(params, "period", 20)
            output[f"SMA_{period}"] = _to_json_values(_sma(df["close"], period))
        elif name == "RSI":
            period = _int_param(params, "period", 14)
            output[f"RSI_{period}"] = _to_json_values(_rsi(df["close"], period))
        elif name == "MACD":
            fast = _int_param(params, "fast", 12)
            slow = _int_param(params, "slow", 26)
            signal = _int_param(params, "signal", 9)
            macd_line, macd_signal, macd_hist = _macd(
                df["close"], fast, slow, signal
            )
            output["MACD_line"] = _to_json_values(macd_line)
            output["MACD_signal"] = _to_json_values(macd_signal)
            output["MACD_hist"] = _to_json_values(macd_hist)
        elif name == "BBANDS":
            period = _int_param(params, "period", 20)
            std = _float_param(params, "std", 2.0)
            upper, middle, lower = _bbands(df["close"], period, std)
            output["BB_upper"] = _to_json_values(upper)
            output["BB_middle"] = _to_json_values(middle)
            output["BB_lower"] = _to_json_values(lower)
        elif name == "VWAP":
            output["VWAP"] = [None for _ in timestamps]
        else:
            raise ValueError(f"unsupported indicator: {name}")

    return {
        "timestamps": timestamps,
        "indicators": output,
    }


def _load_ohlcv(
    symbol: str,
    from_day: datetime.date,
    to_day: datetime.date,
) -> pd.DataFrame:
    symbols = _query_symbols(symbol)
    placeholders = ", ".join("?" for _ in symbols)
    con = get_connection()
    try:
        rows = con.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), open, high, low, close, volume
            FROM ohlcv
            WHERE symbol IN ({placeholders})
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY date ASC
            """,
            [*symbols, from_day.isoformat(), to_day.isoformat()],
        ).fetchall()
    finally:
        con.close()

    by_date: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        row_symbol = str(row[0])
        date = str(row[1])
        if date in by_date and row_symbol != "NIFTY 50":
            continue
        by_date[date] = row

    records = [
        {
            "symbol": str(row[0]),
            "date": str(row[1]),
            "open": float(row[2]),
            "high": float(row[3]),
            "low": float(row[4]),
            "close": float(row[5]),
            "volume": int(row[6]),
        }
        for row in (by_date[date] for date in sorted(by_date))
    ]

    return pd.DataFrame(
        records,
        columns=["symbol", "date", "open", "high", "low", "close", "volume"],
    )


def _ema(series: pd.Series, period: int) -> pd.Series:
    result = ta.ema(series, length=period)
    if result is None:
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    return result


def _sma(series: pd.Series, period: int) -> pd.Series:
    result = ta.sma(series, length=period)
    if result is None:
        return series.rolling(period, min_periods=period).mean()
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


def _macd(
    series: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    result = ta.macd(series, fast=fast, slow=slow, signal=signal)
    if isinstance(result, pd.DataFrame):
        line = _first_column(result, "MACD_")
        hist = _first_column(result, "MACDh_")
        signal_line = _first_column(result, "MACDs_")
        if line is not None and signal_line is not None and hist is not None:
            return line, signal_line, hist

    line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def _bbands(
    series: pd.Series,
    period: int,
    std: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    result = ta.bbands(series, length=period, std=std)
    if isinstance(result, pd.DataFrame):
        upper = _first_column(result, "BBU_")
        middle = _first_column(result, "BBM_")
        lower = _first_column(result, "BBL_")
        if upper is not None and middle is not None and lower is not None:
            return upper, middle, lower

    middle = _sma(series, period)
    deviation = series.rolling(period, min_periods=period).std()
    upper = middle + std * deviation
    lower = middle - std * deviation
    return upper, middle, lower


def _first_column(frame: pd.DataFrame, prefix: str) -> pd.Series | None:
    for column in frame.columns:
        if str(column).startswith(prefix):
            return frame[column]
    return None


def _to_json_values(series: pd.Series) -> list[float | None]:
    values: list[float | None] = []
    for value in series.tolist():
        if value is None or pd.isna(value):
            values.append(None)
            continue
        parsed = float(value)
        values.append(parsed if math.isfinite(parsed) else None)
    return values


def _date_range(
    from_date: str | None,
    to_date: str | None,
) -> tuple[datetime.date, datetime.date]:
    today = datetime.date.today()
    to_day = _parse_date(to_date) if to_date else today
    from_day = (
        _parse_date(from_date)
        if from_date
        else today - datetime.timedelta(days=365)
    )
    if from_day > to_day:
        raise ValueError("from_date must be before to_date")
    return from_day, to_day


def _parse_date(value: str | None) -> datetime.date:
    if value is None:
        raise ValueError("date is required")
    return datetime.date.fromisoformat(value)


def _query_symbols(symbol: str) -> list[str]:
    normalized = symbol.strip().upper()
    if normalized in {"NIFTY", "NIFTY 50"}:
        return ["NIFTY 50", "NIFTY"]
    return [normalized]


def _int_param(params: dict[str, Any], key: str, default: int) -> int:
    value = int(params.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _float_param(params: dict[str, Any], key: str, default: float) -> float:
    value = float(params.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value
