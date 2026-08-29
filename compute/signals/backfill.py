from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pandas_ta as ta

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read


MINIMUM_LOOKBACK_ROWS = 200
WRITE_SYMBOL = "NIFTY"


def backfill_price_action_signals() -> dict[str, Any]:
    return backfill_symbol_signals(WRITE_SYMBOL)


def backfill_symbol_signals(symbol: str) -> dict[str, Any]:
    target_symbol = symbol.strip().upper()
    frame, source_symbol = _load_symbol_ohlcv(target_symbol)
    if frame.empty:
        return {
            "source_symbol": None,
            "target_symbol": target_symbol,
            "ohlcv_rows": 0,
            "dates_considered": 0,
            "dates_skipped_existing": 0,
            "dates_written": 0,
            "rows_written": 0,
            "reason": f"no {target_symbol} OHLCV rows found",
        }

    indicators = _compute_price_action_indicators(frame)
    existing_dates = _existing_signal_dates(target_symbol)
    rows: list[tuple[str, str, str, float]] = []
    dates_written: set[str] = set()
    dates_skipped = 0

    for index, item in indicators.iterrows():
        if index < MINIMUM_LOOKBACK_ROWS - 1:
            continue
        date_text = str(item["date"])
        if date_text in existing_dates:
            dates_skipped += 1
            continue
        date_rows = _rows_for_date(item)
        if not date_rows:
            continue
        rows.extend((date_text, signal_name, target_symbol, value) for signal_name, value in date_rows)
        dates_written.add(date_text)

    rows_written = _write_signal_rows(rows)
    return {
        "source_symbol": source_symbol,
        "target_symbol": target_symbol,
        "ohlcv_rows": int(len(frame)),
        "dates_considered": max(0, int(len(frame)) - MINIMUM_LOOKBACK_ROWS + 1),
        "dates_skipped_existing": dates_skipped,
        "dates_written": len(dates_written),
        "rows_written": rows_written,
    }


def _load_nifty_ohlcv() -> tuple[pd.DataFrame, str | None]:
    return _load_symbol_ohlcv(WRITE_SYMBOL)


def _load_symbol_ohlcv(symbol: str) -> tuple[pd.DataFrame, str | None]:
    candidates = ("NIFTY", "NIFTY 50") if symbol in {"NIFTY", "NIFTY 50"} else (symbol,)
    for candidate in candidates:
        result = resolve_read(ReadOperation.LOAD_SYMBOL_OHLCV_FOR_BACKFILL, {"symbol": candidate})
        frame = pd.DataFrame(
            result["rows"],
            columns=["date", "open", "high", "low", "close", "volume"],
        )
        if not frame.empty:
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            return frame, candidate
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), None


def _compute_price_action_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    output = pd.DataFrame({"date": frame["date"]})

    for length in (9, 21, 50, 200):
        output[f"price_action.ema_{length}"] = ta.ema(close, length=length)

    output["price_action.rsi_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        output["price_action.macd_line"] = _first_column(macd, "MACD_")
        output["price_action.macd_hist"] = _first_column(macd, "MACDh_")
        output["price_action.macd_signal"] = _first_column(macd, "MACDs_")

    bbands = ta.bbands(close, length=20, std=2.0)
    if isinstance(bbands, pd.DataFrame):
        output["price_action.bb_upper"] = _first_column(bbands, "BBU_")
        output["price_action.bb_middle"] = _first_column(bbands, "BBM_")
        output["price_action.bb_lower"] = _first_column(bbands, "BBL_")

    output["price_action.atr_14"] = ta.atr(high=high, low=low, close=close, length=14)

    returns = close.pct_change()
    for length in (10, 20, 30):
        output[f"price_action.hv_{length}"] = (
            ta.stdev(returns, length=length) * math.sqrt(252.0) * 100.0
        )

    return output.replace([float("inf"), float("-inf")], pd.NA)


def _first_column(frame: pd.DataFrame, prefix: str) -> pd.Series:
    for column in frame.columns:
        if str(column).startswith(prefix):
            return frame[column]
    return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="float64")


def _existing_signal_dates(symbol: str) -> set[str]:
    result = resolve_read(ReadOperation.LOAD_EXISTING_SIGNAL_DATES, {"symbol": symbol})
    return set(result["dates"])


def _rows_for_date(row: pd.Series) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for name, value in row.items():
        if name == "date" or value is None or pd.isna(value):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append((str(name), number))
    return values


def _write_signal_rows(rows: list[tuple[str, str, str, float]]) -> int:
    """Set-based bulk insert against `signals_history`.

    The prior implementation ran `executemany(... ON CONFLICT DO NOTHING)`,
    checking each candidate row for a conflict individually against the full
    (multi-million-row) target table. Reproduced on a disposable 3.2M-row
    table: that per-row conflict check does not scale -- DuckDB's PK/unique
    constraint validation for a large batch of INSERT..ON CONFLICT statements
    inside one transaction is not a simple O(log n) index probe per row once
    the batch itself is large, and wall time grew from seconds to tens of
    minutes as target cardinality grew, matching the live 20+ minute hang.

    This instead registers the candidate rows as a queryable relation (no
    physical staging table -- nothing to leak on crash/retry) and performs
    ONE set-based `INSERT ... SELECT` with an anti-join against the target's
    business key, so DuckDB's optimizer picks a single bulk join strategy
    instead of one conflict probe per row. Same business key, same conflict
    behavior (existing rows are left untouched, never overwritten), same
    idempotent-rerun guarantee (a second call with identical rows inserts
    zero rows).
    """
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=["date", "signal_name", "symbol", "value"])
    # Defend against duplicate business keys *within this batch* (the caller
    # already de-duplicates by construction, but the set-based INSERT has no
    # per-row ON CONFLICT to fall back on, so a batch-internal duplicate would
    # otherwise violate the PRIMARY KEY inside the single INSERT statement).
    frame = frame.drop_duplicates(subset=["date", "signal_name", "symbol"], keep="last")
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute("BEGIN TRANSACTION")
            con.register("_signal_candidate_rows", frame)
            try:
                result = con.execute(
                    """
                    INSERT INTO signals_history (date, signal_name, symbol, value)
                    SELECT CAST(c.date AS DATE), c.signal_name, c.symbol, c.value
                    FROM _signal_candidate_rows c
                    ANTI JOIN signals_history t
                      ON t.date = CAST(c.date AS DATE)
                     AND t.signal_name = c.signal_name
                     AND t.symbol = c.symbol
                    """
                ).fetchall()
            finally:
                con.unregister("_signal_candidate_rows")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()
    inserted = int(result[0][0]) if result and result[0] and result[0][0] is not None else 0
    return inserted
