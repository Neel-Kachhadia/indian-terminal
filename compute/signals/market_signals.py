from __future__ import annotations

import datetime as _dt
import math
import time as _time
from typing import Any

import requests as _requests


NSE_HOME = "https://www.nseindia.com"
ALL_INDICES_URL = "https://www.nseindia.com/api/allIndices"
FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def fetch_india_vix() -> float | None:
    """
    Fetch India VIX from NSE allIndices.

    Returns None when NSE gives HTML, non-JSON, or no India VIX row.
    """
    try:
        session = _nse_session()
        _warm_session(session)
        response = session.get(
            ALL_INDICES_URL,
            headers={
                "Accept": "application/json",
                "Referer": NSE_HOME,
            },
            timeout=10,
        )
        if not _json_response(response):
            return None

        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        for item in rows:
            if not isinstance(item, dict):
                continue
            if str(item.get("index") or "").strip().upper() == "INDIA VIX":
                return _number(item.get("last"))
    except (_requests.RequestException, ValueError) as error:
        print(f"signals: India VIX fetch failed: {error}")

    return None


def fetch_fii_dii() -> dict[str, Any]:
    """
    Fetch FII/DII net cash flows from NSE.

    Returns an error dict when unavailable. Missing futures data is returned as
    0.0 because fiidiiTradeReact normally exposes cash rows only.
    """
    try:
        session = _nse_session()
        _warm_session(session)
        response = session.get(
            FII_DII_URL,
            headers={
                "Accept": "application/json",
                "Referer": NSE_HOME,
            },
            timeout=10,
        )
        if not _json_response(response):
            return {"error": "NSE returned non-JSON FII/DII response"}

        payload = response.json()
        rows = _extract_rows(payload)
        if not rows:
            return {"error": "no FII/DII rows"}

        return _parse_fii_dii_rows(rows)
    except (_requests.RequestException, ValueError) as error:
        print(f"signals: FII/DII fetch failed: {error}")
        return {"error": str(error)}


def _nse_session() -> _requests.Session:
    session = _requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    })
    return session


def _warm_session(session: _requests.Session) -> None:
    session.get(NSE_HOME, timeout=10)
    _time.sleep(0.5)


def _json_response(response: _requests.Response) -> bool:
    if response.status_code != 200:
        return False
    text = response.text.lstrip()
    if not text or text.startswith("<"):
        return False
    content_type = response.headers.get("content-type", "")
    return "json" in content_type.lower() or text[0] in "[{"


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("data", "rows", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return [payload]


def _parse_fii_dii_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_date = _latest_date(rows)
    latest_rows = [
        row
        for row in rows
        if _date_text(row) == latest_date
    ] or rows

    direct = _parse_direct_fields(latest_rows[0])
    if direct is not None:
        return direct

    fii_row = _find_category_row(latest_rows, ("FII", "FPI"))
    dii_row = _find_category_row(latest_rows, ("DII",))
    if fii_row is None and dii_row is None:
        return {"error": "no FII/DII category rows"}

    fii_buy = _row_buy(fii_row)
    fii_sell = _row_sell(fii_row)
    dii_buy = _row_buy(dii_row)
    dii_sell = _row_sell(dii_row)

    return {
        "date": latest_date,
        "fii_net_cash": _row_net(fii_row, fii_buy, fii_sell),
        "dii_net_cash": _row_net(dii_row, dii_buy, dii_sell),
        "fii_net_futures": _first_number(
            fii_row,
            (
                "fii_net_futures",
                "fiiNetFutures",
                "indexFuturesNet",
                "futuresNet",
            ),
            default=0.0,
        ),
        "fii_buy_cash": fii_buy,
        "fii_sell_cash": fii_sell,
        "dii_buy_cash": dii_buy,
        "dii_sell_cash": dii_sell,
    }


def _parse_direct_fields(row: dict[str, Any]) -> dict[str, Any] | None:
    fii_net = _first_number(
        row,
        ("fii_net_cash", "fiiNetCash", "fii_net", "fiiNet", "fii"),
        default=None,
    )
    dii_net = _first_number(
        row,
        ("dii_net_cash", "diiNetCash", "dii_net", "diiNet", "dii"),
        default=None,
    )
    if fii_net is None and dii_net is None:
        return None

    fii_buy = _first_number(
        row,
        ("fii_buy_cash", "fiiBuyCash", "fii_buy", "fiiBuy"),
        default=0.0,
    )
    fii_sell = _first_number(
        row,
        ("fii_sell_cash", "fiiSellCash", "fii_sell", "fiiSell"),
        default=0.0,
    )
    dii_buy = _first_number(
        row,
        ("dii_buy_cash", "diiBuyCash", "dii_buy", "diiBuy"),
        default=0.0,
    )
    dii_sell = _first_number(
        row,
        ("dii_sell_cash", "diiSellCash", "dii_sell", "diiSell"),
        default=0.0,
    )

    return {
        "date": _date_text(row),
        "fii_net_cash": fii_net or 0.0,
        "dii_net_cash": dii_net or 0.0,
        "fii_net_futures": _first_number(
            row,
            ("fii_net_futures", "fiiNetFutures", "indexFuturesNet"),
            default=0.0,
        ),
        "fii_buy_cash": fii_buy,
        "fii_sell_cash": fii_sell,
        "dii_buy_cash": dii_buy,
        "dii_sell_cash": dii_sell,
    }


def _find_category_row(
    rows: list[dict[str, Any]],
    tokens: tuple[str, ...],
) -> dict[str, Any] | None:
    for row in rows:
        category = " ".join(
            str(row.get(key) or "")
            for key in ("category", "name", "type", "participant")
        ).upper()
        if any(token in category for token in tokens):
            return row
    return None


def _latest_date(rows: list[dict[str, Any]]) -> str:
    best_text = _date_text(rows[0]) if rows else ""
    best_date = _parse_date(best_text)

    for row in rows:
        date_text = _date_text(row)
        parsed = _parse_date(date_text)
        if parsed is None:
            continue
        if best_date is None or parsed > best_date:
            best_date = parsed
            best_text = date_text

    return best_text


def _date_text(row: dict[str, Any]) -> str:
    for key in ("date", "Date", "tradeDate", "tradedate", "createdDate"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _parse_date(value: str) -> _dt.date | None:
    clean = value.strip()
    for fmt in (
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d-%B-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ):
        try:
            return _dt.datetime.strptime(clean, fmt).date()
        except ValueError:
            pass
    return None


def _row_buy(row: dict[str, Any] | None) -> float:
    return _first_number(
        row,
        (
            "buyValue",
            "buy_value",
            "buy",
            "grossBuyValue",
            "grossBuy",
            "fii_buy_cash",
            "dii_buy_cash",
        ),
        default=0.0,
    )


def _row_sell(row: dict[str, Any] | None) -> float:
    return _first_number(
        row,
        (
            "sellValue",
            "sell_value",
            "sell",
            "grossSellValue",
            "grossSell",
            "fii_sell_cash",
            "dii_sell_cash",
        ),
        default=0.0,
    )


def _row_net(row: dict[str, Any] | None, buy: float, sell: float) -> float:
    direct = _first_number(
        row,
        (
            "netValue",
            "net_value",
            "net",
            "netAmount",
            "cashNet",
            "fii_net_cash",
            "dii_net_cash",
        ),
        default=None,
    )
    if direct is not None:
        return direct
    return buy - sell


def _first_number(
    row: dict[str, Any] | None,
    keys: tuple[str, ...],
    default: float | None,
) -> float | None:
    if row is None:
        return default
    lower_lookup = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = lower_lookup.get(key.lower())
        number = _number(value)
        if number is not None:
            return number
    return default


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text or text in {"-", "--"}:
            return None
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()").replace(",", "")
        try:
            number = float(text)
        except ValueError:
            return None
        if negative:
            number *= -1

    if not math.isfinite(number):
        return None
    return number
