from contextlib import asynccontextmanager
import asyncio
import datetime
import json
from threading import Lock
from time import time
import time as _time
from typing import Any

import redis as _redis
import requests as _requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db.bhav_loader import load_date_range
from db.duckdb_client import DB_WRITE_LOCK, get_connection, init_schema


class OHLCVRow(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str = "bhav"


class HistoricalDownloadRequest(BaseModel):
    symbol: str
    from_date: str
    to_date: str


class GreeksLegRequest(BaseModel):
    strike: float
    option_type: str
    ltp: float
    oi: int = 0
    volume: int = 0


class GreeksRequest(BaseModel):
    symbol: str
    expiry: str
    spot: float
    legs: list[GreeksLegRequest]


class IndicatorRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    from_date: str | None = None
    to_date: str | None = None
    indicators: list[dict[str, Any]] = Field(default_factory=list)


class ScreenerRequest(BaseModel):
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    logic: str = "AND"
    universe: str = "NIFTY50"
    limit: int = 50


class PayoffLeg(BaseModel):
    symbol: str
    strike: float
    expiry: str
    option_type: str
    action: str
    quantity: int = 1
    entry_price: float
    lot_size: int = 75


class PayoffRequest(BaseModel):
    legs: list[PayoffLeg]
    spot: float
    spot_range_pct: float = 0.20


_download_status: dict[str, Any] = {
    "running": False,
    "downloaded": 0,
    "skipped": 0,
    "errors": 0,
}
_download_status_lock = Lock()
_redis_client = _redis.Redis(host="127.0.0.1", port=6379, db=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    await seed_nifty_if_empty()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, float | str]:
    return {"status": "ok", "timestamp": time()}


@app.post("/greeks")
def compute_greeks(request: GreeksRequest) -> Any:
    from options.greeks import compute_chain_greeks

    start = _time.time()
    legs = [
        {
            "strike": leg.strike,
            "option_type": leg.option_type,
            "ltp": leg.ltp,
            "oi": leg.oi,
            "volume": leg.volume,
        }
        for leg in request.legs
    ]

    try:
        result = compute_chain_greeks(legs, request.spot, request.expiry)
        elapsed_ms = int((_time.time() - start) * 1000)
        print(f"greeks: computed {len(result)} strikes in {elapsed_ms}ms")
        return {
            "legs": result,
            "computed_at": _time.time(),
        }
    except Exception as e:
        print(f"greeks: computation error: {e}")
        return {"error": str(e), "legs": [], "computed_at": _time.time()}


@app.post("/indicators")
def get_indicators(request: IndicatorRequest) -> Any:
    from technical.indicators import compute_indicators

    try:
        return compute_indicators(
            request.symbol,
            request.timeframe,
            request.from_date,
            request.to_date,
            request.indicators,
        )
    except Exception as e:
        print(f"indicators: error: {e}")
        return {"error": str(e)}


@app.post("/screener")
def run_screener_endpoint(request: ScreenerRequest) -> Any:
    from technical.screener import run_screener

    try:
        return run_screener(
            request.conditions,
            request.logic,
            request.universe,
            request.limit,
        )
    except Exception as e:
        print(f"screener: error: {e}")
        return {"error": str(e)}


@app.post("/payoff")
def compute_payoff_endpoint(request: PayoffRequest) -> Any:
    from options.payoff import compute_payoff

    try:
        legs = [_model_dump(leg) for leg in request.legs]
        return compute_payoff(legs, request.spot, request.spot_range_pct)
    except Exception as e:
        print(f"payoff: error: {e}")
        return {"error": str(e)}


@app.get("/futures/basis/{symbol}")
def get_futures_basis(symbol: str) -> Any:
    from futures.basis import compute_futures_basis

    try:
        return compute_futures_basis(symbol)
    except Exception as e:
        print(f"futures: basis error: {e}")
        return {"error": str(e)}


@app.get("/chain/{symbol}")
def get_chain_from_nse(symbol: str) -> Any:
    try:
        session = _requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.5",
        })
        session.get("https://www.nseindia.com", timeout=10)
        _time.sleep(0.5)
        session.get("https://www.nseindia.com/option-chain", timeout=10)
        _time.sleep(0.5)

        url = f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolDerivativesData&symbol={symbol.upper()}"
        r = session.get(url, headers={
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/option-chain",
        }, timeout=15)

        if r.status_code != 200:
            return {"error": f"NSE returned {r.status_code}"}

        text = r.text
        if text.strip().startswith("<"):
            return {"error": "session_expired"}

        data = r.json()

        if "data" not in data:
            return {"error": "no data field", "raw": text[:200]}

        spot = 0.0
        rows_by_strike: dict = {}
        for item in data["data"]:
            strike = float(str(item.get("strikePrice", "0")).strip())
            expiry = item.get("expiryDate", "")
            opt_type = item.get("optionType", "")
            key = (strike, expiry)
            if key not in rows_by_strike:
                rows_by_strike[key] = {"strikePrice": strike, "expiryDate": expiry}
            leg = {
                "lastPrice": item.get("lastPrice", 0),
                "bidprice": 0,
                "askPrice": 0,
                "openInterest": item.get("openInterest", 0),
                "changeinOpenInterest": item.get("changeinOpenInterest", 0),
                "totalTradedVolume": item.get("totalTradedVolume", 0),
                "impliedVolatility": item.get("impliedVolatility", 0),
            }
            if opt_type == "CE":
                rows_by_strike[key]["CE"] = leg
            elif opt_type == "PE":
                rows_by_strike[key]["PE"] = leg

        records_data = list(rows_by_strike.values())

        try:
            spot_url = "https://www.nseindia.com/api/allIndices"
            sr = session.get(
                spot_url,
                headers={
                    "Accept": "application/json",
                    "Referer": "https://www.nseindia.com",
                },
                timeout=10,
            )
            if sr.status_code == 200:
                indices = sr.json().get("data", [])
                for idx in indices:
                    if idx.get("index") == "NIFTY 50":
                        spot = float(idx.get("last", 0))
                        break
        except Exception:
            pass

        return {
            "records": {
                "data": records_data,
                "underlyingValue": spot,
            }
        }

    except Exception as e:
        print(f"nse: chain fetch error: {e}")
        return {"error": str(e)}


@app.get("/expiries/{symbol}")
def get_expiries(symbol: str) -> list[str]:
    _query_symbols(symbol)
    return get_next_expiries()


@app.get("/iv_surface/{symbol}")
def get_iv_surface(symbol: str) -> Any:
    from options.iv_surface import compute_iv_surface

    expiries = get_next_expiries()
    legs = _read_chain_legs_from_redis(symbol, expiries)
    legs_with_iv = [leg for leg in legs if leg.get("iv") is not None]

    if not legs_with_iv:
        return {"error": "no chain data in Redis"}

    return compute_iv_surface(legs_with_iv)


@app.get("/ohlcv/{symbol}")
def get_ohlcv(
    symbol: str,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict[str, Any]]:
    from_day, to_day = _date_range(from_date, to_date)
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

    by_date: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_symbol = str(row[0])
        date = str(row[1])
        if date in by_date and row_symbol != "NIFTY 50":
            continue

        by_date[date] = {
            "date": date,
            "open": float(row[2]),
            "high": float(row[3]),
            "low": float(row[4]),
            "close": float(row[5]),
            "volume": int(row[6]),
        }

    return [by_date[date] for date in sorted(by_date)]


@app.post("/ingest/bhav")
def ingest_bhav(rows: list[OHLCVRow]) -> dict[str, int]:
    inserted = _insert_rows([_model_dump(row) for row in rows])
    return {"inserted": inserted}


@app.post("/download/historical")
async def download_historical(
    request: HistoricalDownloadRequest,
) -> dict[str, str]:
    from_day = _parse_date(request.from_date)
    to_day = _parse_date(request.to_date)

    if _status_snapshot()["running"]:
        return {
            "status": "already_running",
            "from": from_day.isoformat(),
            "to": to_day.isoformat(),
        }

    _start_download(from_day, to_day)
    return {
        "status": "started",
        "from": from_day.isoformat(),
        "to": to_day.isoformat(),
    }


@app.get("/download/status")
def download_status() -> dict[str, Any]:
    return _status_snapshot()


async def seed_nifty_if_empty() -> None:
    con = get_connection()
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = 'NIFTY 50'"
        ).fetchone()[0]
    finally:
        con.close()

    if count < 200:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=365)
        print(
            "bhav: seeding NIFTY data "
            f"{from_date.isoformat()} to {today.isoformat()}"
        )
        _start_download(from_date, today)


def _start_download(
    from_date: datetime.date, to_date: datetime.date
) -> None:
    _set_status(running=True, downloaded=0, skipped=0, errors=0)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_download, from_date, to_date)


def _run_download(
    from_date: datetime.date, to_date: datetime.date
) -> None:
    try:
        result = load_date_range(from_date, to_date, _update_download_progress)
        _set_status(running=False, **result)
    except Exception as error:
        snapshot = _status_snapshot()
        print(f"bhav: background download failed: {error}")
        _set_status(running=False, errors=int(snapshot["errors"]) + 1)


def _insert_rows(rows: list[dict[str, Any]]) -> int:
    inserted = 0

    with DB_WRITE_LOCK:
        con = get_connection()
        try:
            con.execute("BEGIN TRANSACTION")
            for row in rows:
                symbol = str(row["symbol"]).strip().upper()
                date = _parse_date(str(row["date"])).isoformat()
                exists = con.execute(
                    """
                    SELECT 1
                    FROM ohlcv
                    WHERE symbol = ? AND date = CAST(? AS DATE)
                    """,
                    [symbol, date],
                ).fetchone()
                if exists:
                    continue

                con.execute(
                    """
                    INSERT INTO ohlcv
                        (symbol, date, open, high, low, close, volume, source)
                    VALUES (?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        symbol,
                        date,
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        int(row["volume"]),
                        str(row.get("source") or "bhav"),
                    ],
                )
                inserted += 1

            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

    return inserted


def _status_snapshot() -> dict[str, Any]:
    with _download_status_lock:
        return dict(_download_status)


def _set_status(**updates: Any) -> None:
    with _download_status_lock:
        _download_status.update(updates)


def _update_download_progress(result: dict[str, int]) -> None:
    _set_status(running=True, **result)


def _model_dump(row: OHLCVRow) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return row.dict()


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
        raise HTTPException(
            status_code=400, detail="from_date must be before to_date"
        )
    return from_day, to_day


def _parse_date(value: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as error:
        raise HTTPException(
            status_code=400, detail=f"invalid date: {value}"
        ) from error


def _query_symbols(symbol: str) -> list[str]:
    normalized = symbol.strip().upper()
    if normalized in {"NIFTY", "NIFTY 50"}:
        return ["NIFTY 50", "NIFTY"]
    return [normalized]


def _redis_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized in {"NIFTY", "NIFTY 50"}:
        return "NIFTY"
    return normalized


def _read_chain_legs_from_redis(
    symbol: str,
    expiries: list[str],
) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    for key in _matching_chain_keys(symbol, expiries):
        try:
            payload = _redis_client.get(key)
        except _redis.RedisError as error:
            print(f"iv_surface: redis GET {key} failed: {error}")
            continue

        if payload is None:
            continue

        legs.extend(_decode_chain_payload(payload, key))

    return legs


def _matching_chain_keys(symbol: str, expiries: list[str]) -> list[str]:
    redis_symbol = _redis_symbol(symbol)
    expiry_lookup = {_normalise_expiry_text(expiry) for expiry in expiries}
    keys = [f"chain:{redis_symbol}:{expiry}" for expiry in expiries]

    try:
        for raw_key in _redis_client.scan_iter(match="chain:*"):
            key = _redis_text(raw_key)
            parts = key.split(":", 2)
            if len(parts) != 3:
                continue
            _, key_symbol, key_expiry = parts
            if key_symbol.strip().upper() != redis_symbol:
                continue
            if _normalise_expiry_text(key_expiry) in expiry_lookup:
                keys.append(key)
    except _redis.RedisError as error:
        print(f"iv_surface: redis key scan failed: {error}")

    seen: set[str] = set()
    unique_keys: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)

    return unique_keys


def _decode_chain_payload(payload: Any, key: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(_redis_text(payload))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        print(f"iv_surface: JSON parse failed for {key}: {error}")
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        nested = data.get("legs") or data.get("data")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]

    return []


def _redis_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalise_expiry_text(value: str) -> str:
    return value.strip().lower()


def get_next_expiries(n: int = 4) -> list[str]:
    """
    Returns next n NIFTY expiry dates.
    NSE changed NIFTY weekly expiry from Thursday to Tuesday in 2024.
    """
    today = datetime.date.today()
    days_until_tuesday = (1 - today.weekday()) % 7
    if days_until_tuesday == 0:
        now_ist = datetime.datetime.utcnow() + datetime.timedelta(
            hours=5, minutes=30
        )
        if now_ist.hour >= 15 and now_ist.minute >= 30:
            days_until_tuesday = 7
    first_expiry = today + datetime.timedelta(days=days_until_tuesday)
    return [
        (first_expiry + datetime.timedelta(days=7 * index)).strftime("%d-%b-%Y")
        for index in range(n)
    ]


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8765)
