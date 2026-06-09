from __future__ import annotations

import datetime as _dt
import json
import math
from collections import defaultdict
from typing import Any

import redis

from db.duckdb_client import get_connection


def compute_futures_basis(symbol: str = "NIFTY") -> dict[str, Any]:
    """
    Compute near-month futures basis history from DuckDB.

    Empty `fo_eod` returns no-data arrays instead of raising so UI can render.
    """
    normalized = _normalise_symbol(symbol)
    empty = _empty_result()

    try:
        futures_rows, spot_rows = _read_historical_rows(normalized)
    except Exception as error:
        print(f"futures: basis query failed: {error}")
        return {**empty, "error": "no futures data"}

    if not futures_rows:
        return {**empty, "error": "no futures data"}

    spot_by_date = _spot_by_date(spot_rows)
    rows = _basis_rows(futures_rows, spot_by_date)
    if not rows:
        return {**empty, "error": "no futures data"}

    near_expiry = _current_near_expiry(futures_rows)
    current_basis, current_carry = _current_basis_from_redis(
        normalized,
        near_expiry,
        rows[-1]["spot"],
    )

    return {
        "dates": [row["date"] for row in rows],
        "spot": [row["spot"] for row in rows],
        "futures": [row["futures"] for row in rows],
        "basis": [row["basis"] for row in rows],
        "basis_pct": [row["basis_pct"] for row in rows],
        "annualised_carry": [row["annualised_carry"] for row in rows],
        "near_expiry": near_expiry,
        "current_basis": current_basis,
        "current_carry": current_carry,
    }


def _read_historical_rows(
    symbol: str,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    con = get_connection()
    try:
        futures_rows = con.execute(
            """
            SELECT CAST(date AS VARCHAR), CAST(expiry AS VARCHAR), close
            FROM fo_eod
            WHERE UPPER(symbol) IN (?, ?)
              AND UPPER(option_type) = 'FUT'
              AND close IS NOT NULL
            ORDER BY date ASC, expiry ASC
            """,
            [symbol, _spot_symbol(symbol)],
        ).fetchall()
        spot_rows = con.execute(
            """
            SELECT CAST(date AS VARCHAR), UPPER(symbol), close
            FROM ohlcv
            WHERE UPPER(symbol) IN (?, ?)
              AND close IS NOT NULL
            ORDER BY date ASC,
              CASE WHEN UPPER(symbol) = ? THEN 0 ELSE 1 END
            """,
            [_spot_symbol(symbol), symbol, _spot_symbol(symbol)],
        ).fetchall()
    finally:
        con.close()

    return futures_rows, spot_rows


def _basis_rows(
    futures_rows: list[tuple[str, str, float]],
    spot_by_date: dict[str, float],
) -> list[dict[str, float | str]]:
    by_date: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for date_text, expiry_text, futures_close in futures_rows:
        by_date[str(date_text)].append((str(expiry_text), float(futures_close)))

    output: list[dict[str, float | str]] = []
    for date_text in sorted(by_date):
        spot_close = spot_by_date.get(date_text)
        if spot_close is None or spot_close <= 0:
            continue

        date_value = _parse_date(date_text)
        if date_value is None:
            continue

        near = _near_future_for_date(date_value, by_date[date_text])
        if near is None:
            continue

        expiry_text, futures_close = near
        expiry_value = _parse_date(expiry_text)
        if expiry_value is None:
            continue

        dte = max((expiry_value - date_value).days, 1)
        basis = futures_close - spot_close
        basis_pct = (basis / spot_close) * 100.0
        annualised_carry = basis_pct * (365.0 / dte)

        output.append({
            "date": date_text,
            "spot": round(spot_close, 2),
            "futures": round(futures_close, 2),
            "basis": round(basis, 2),
            "basis_pct": round(basis_pct, 4),
            "annualised_carry": round(annualised_carry, 4),
        })

    return output


def _near_future_for_date(
    date_value: _dt.date,
    futures: list[tuple[str, float]],
) -> tuple[str, float] | None:
    candidates: list[tuple[_dt.date, str, float]] = []
    for expiry_text, futures_close in futures:
        expiry_value = _parse_date(expiry_text)
        if expiry_value is None or expiry_value < date_value:
            continue
        candidates.append((expiry_value, expiry_text, futures_close))

    if not candidates:
        return None

    _, expiry_text, futures_close = min(candidates, key=lambda item: item[0])
    return expiry_text, futures_close


def _spot_by_date(spot_rows: list[tuple[str, str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for date_text, _symbol, close in spot_rows:
        output.setdefault(str(date_text), float(close))
    return output


def _current_near_expiry(futures_rows: list[tuple[str, str, float]]) -> str:
    today = _dt.date.today()
    expiries = sorted({
        expiry_text
        for _date_text, expiry_text, _close in futures_rows
        if (_parse_date(str(expiry_text)) or _dt.date.min) >= today
    }, key=lambda value: _parse_date(str(value)) or _dt.date.max)

    if expiries:
        return str(expiries[0])

    all_expiries = sorted({
        str(expiry_text)
        for _date_text, expiry_text, _close in futures_rows
    }, key=lambda value: _parse_date(value) or _dt.date.min)
    return all_expiries[-1] if all_expiries else ""


def _current_basis_from_redis(
    symbol: str,
    near_expiry: str,
    fallback_spot: float,
) -> tuple[float | None, float | None]:
    if not near_expiry:
        return None, None

    try:
        client = redis.Redis(host="127.0.0.1", port=6379, db=0)
        spot = _read_spot(client, symbol) or fallback_spot
        futures_ltp = _read_futures_ltp(client, symbol, near_expiry)
    except redis.RedisError:
        return None, None

    if spot <= 0 or futures_ltp is None:
        return None, None

    basis = futures_ltp - spot
    expiry = _parse_date(near_expiry)
    if expiry is None:
        return round(basis, 2), None

    dte = max((expiry - _dt.date.today()).days, 1)
    carry = (basis / spot) * (365.0 / dte) * 100.0
    return round(basis, 2), round(carry, 4)


def _read_spot(client: redis.Redis, symbol: str) -> float | None:
    for key in (f"tick:{symbol}", f"tick:{_spot_symbol(symbol)}"):
        payload = client.get(key)
        if payload is None:
            continue
        data = _decode_payload(payload)
        if isinstance(data, dict):
            spot = _finite_float(data.get("ltp"))
            if spot is not None and spot > 0:
                return spot
    return None


def _read_futures_ltp(
    client: redis.Redis,
    symbol: str,
    near_expiry: str,
) -> float | None:
    keys = [
        f"chain:{symbol}:{near_expiry}",
        f"chain:{_spot_symbol(symbol)}:{near_expiry}",
    ]
    for key in keys:
        payload = client.get(key)
        if payload is None:
            continue
        data = _decode_payload(payload)
        legs = _payload_legs(data)
        for leg in legs:
            if str(leg.get("option_type", "")).strip().upper() != "FUT":
                continue
            ltp = _finite_float(leg.get("ltp") or leg.get("close"))
            if ltp is not None and ltp > 0:
                return ltp
    return None


def _payload_legs(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        nested = data.get("legs") or data.get("data")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def _decode_payload(payload: Any) -> Any:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        return json.loads(text)
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


def _parse_date(value: str) -> _dt.date | None:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def _normalise_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if value in {"NIFTY", "NIFTY 50"}:
        return "NIFTY"
    return value


def _spot_symbol(symbol: str) -> str:
    if symbol == "NIFTY":
        return "NIFTY 50"
    return symbol


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _empty_result() -> dict[str, Any]:
    return {
        "dates": [],
        "spot": [],
        "futures": [],
        "basis": [],
        "basis_pct": [],
        "annualised_carry": [],
        "near_expiry": "",
        "current_basis": None,
        "current_carry": None,
    }
