from contextlib import asynccontextmanager, suppress
import asyncio
from collections import deque
import copy
import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import secrets
import sys
import tempfile
from threading import Lock, Thread
from time import time
import time as _time
from typing import Any
from urllib.parse import quote, urljoin, urlsplit
import uuid

from defusedxml import ElementTree as ET
import redis as _redis
import requests as _requests
import pandas as pd
import uvicorn
import websockets
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from data.error_logger import _logger as elog
from data.exchange_calendar import exchange_calendar, is_trading_day_candidate, next_expiry_labels, today_ist
from data.index_aliases import INDEX_CANONICAL_NAME, index_symbol_candidates
from data.instrument_master import master as instrument_master
from data.instrument_schema import InstrumentSchemaError, read_instrument_frame
from data.market_data_quality import DecisionContext, evaluate_market_bar
from data.nse_session import nse
from data.paths import BASE_DIR, DATA_DIR, DUCKDB_PATH, EQUITY_BHAV_DIR, INSTRUMENTS_CACHE, ensure_dirs
from db.bhav_loader import load_date_range
from db.duckdb_client import DB_WRITE_LOCK, init_schema, open_read_connection, writer_connection
from db.write_service import (
    ReadOperation,
    WriteOperation,
    WriterService,
    WriterServiceError,
    resolve_read,
    submit_write,
)
from db.writer_lease import WriterLeaseError
from live.market_state import MarketDataState, evaluate_market_state, parse_quote_envelope
from security import credential_manager
from security.integrity import integrity_check
from security.redis_config import authenticated_redis_client
from security.log_scrubber import install_log_scrubber, scrub_payload
from security.validators import http_date, http_symbol, validate_symbol
from release_identity import load_release_identity
from auto_collector import job_supervisor, setup_autostart
from auto_collector.smart_collector import main as run_smart_collector_once
from auto_collector.supervision import status_snapshot
from operations.logging_config import configure_structured_logging


_inherited_process_token = os.environ.pop("BHARATTERMINAL_PROCESS_TOKEN", "")
if _inherited_process_token and not re.fullmatch(r"[0-9a-fA-F]{64,}", _inherited_process_token):
    raise RuntimeError("invalid inherited process-token format")
PROCESS_TOKEN = _inherited_process_token or secrets.token_hex(32)
RELEASE_IDENTITY = load_release_identity()
install_log_scrubber()
logger = logging.getLogger(__name__)


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
    oi: int | None = None
    volume: int | None = None


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


class StockScreenerRequest(BaseModel):
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    logic: str = "AND"
    universe: str = "FNO"
    limit: int = 50


class PayoffLeg(BaseModel):
    symbol: str
    strike: float
    expiry: str
    option_type: str
    action: str
    quantity: int = 1
    entry_price: float
    lot_size: int


class PayoffRequest(BaseModel):
    legs: list[PayoffLeg]
    spot: float
    spot_range_pct: float = 0.20


class MLPredictRequest(BaseModel):
    symbol: str = "NIFTY"
    signals: dict[str, Any] | None = None


class MLOutcomeRequest(BaseModel):
    actual: str = Field(min_length=1, max_length=64)


class BackfillRequest(BaseModel):
    scope: str = "full"
    years_equity: int = 5
    years_fo: int = 5
    months_intraday: int = 6


class BacktestRunRequest(BaseModel):
    strategy: str = "sell_weekly_atm_straddle"
    symbol: str = "NIFTY"
    from_date: str = "2024-01-01"
    to_date: str = "2026-06-19"
    initial_capital: float = 500000
    chronological_segments: bool = True
    monte_carlo: bool = True
    n_mc_simulations: int = 10000
    cost_model: str = "full"
    slippage_model: str = "realistic"
    position_sizer: str = "quarter_kelly"


_download_status: dict[str, Any] = {
    "running": False,
    "downloaded": 0,
    "skipped": 0,
    "errors": 0,
}
_download_status_lock = Lock()
_redis_client = authenticated_redis_client()
_market_signals_lock = Lock()
_market_signals_cache: dict[str, Any] = {
    "timestamp": 0.0,
    "data": {
        "india_vix": None,
        "fii_dii": {},
    },
}
_sentiment_cache: dict[str, Any] = {}
_sentiment_cache_time: float = 0.0
_sentiment_cache_lock = Lock()
_sentiment_fetch_in_progress = False
_SENTIMENT_CACHE_TTL = 120
_sentiment_score_history: deque[tuple[float, float]] = deque(maxlen=500)
_sentiment_article_history: deque[dict[str, Any]] = deque(maxlen=500)
_cross_asset_cache: dict[str, Any] = {}
_cross_asset_cache_time: float = 0.0
_cross_asset_cache_lock = Lock()
_CROSS_ASSET_CACHE_TTL = 60
_all_signals_cache: dict[str, Any] = {}
_all_signals_cache_time: dict[str, float] = {}
_all_signals_cache_lock = Lock()
_ALL_SIGNALS_CACHE_TTL = 60
_MAX_SYMBOL_CACHE_ENTRIES = 512
_apex_cache: dict[str, dict[str, Any]] = {}
_apex_cache_lock = Lock()
_APEX_CACHE_TTL = 60
TEST_MATRIX: dict[str, list[str]] = {
    "nifty50": ["RELIANCE", "TCS"],
    "banking": ["HDFCBANK", "SBIN"],
    "it": ["INFY", "WIPRO"],
    "pharma": ["SUNPHARMA", "DRREDDY"],
    "auto": ["MARUTI", "TATAMOTORS"],
    "metal": ["TATASTEEL", "HINDALCO"],
    "fmcg": ["HINDUNILVR", "BRITANNIA"],
    "energy": ["ONGC", "BPCL"],
    "index_derivatives": ["NIFTY", "BANKNIFTY"],
    "midcap_fno": ["INDUSINDBK", "DIVISLAB"],
}
ALL_TEST_SYMBOLS = [
    symbol
    for symbols in TEST_MATRIX.values()
    for symbol in symbols
]
INDEX_DERIVATIVE_SYMBOLS = set(TEST_MATRIX["index_derivatives"])
_backfill_state: dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "current_symbol": None,
    "current_date": None,
    "equity_downloaded": 0,
    "equity_total": 0,
    "fo_downloaded": 0,
    "fo_total": 0,
    "equity_loaded": 0,
    "fo_loaded": 0,
    "intraday_rows": 0,
    "errors": [],
    "started_at": None,
    "elapsed_minutes": 0,
}
_backfill_lock = Lock()
_parser_test_report: dict[str, Any] = {}
_parser_test_report_lock = Lock()
_ml_training_state: dict[str, Any] = {
    "running": False,
    "last_report": None,
    "last_error": None,
}
_ml_training_lock = Lock()
_stock_training_state: dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "current_symbol": None,
    "done": 0,
    "total": 0,
    "trained": 0,
    "errors": 0,
    "started_at": None,
    "elapsed_minutes": 0,
    "last_report": None,
}
_stock_training_lock = Lock()
_ohlc_forecast_cache: dict[str, dict[str, Any]] = {}
_ohlc_forecast_cache_time: dict[str, float] = {}
_ohlc_forecast_cache_lock = Lock()
_OHLC_FORECAST_CACHE_TTL = 60
_bif_engine: Any | None = None
_bif_engine_lock = Lock()
_backtest_jobs: dict[str, dict[str, Any]] = {}
_backtest_jobs_lock = Lock()
_MAX_IN_MEMORY_BACKTEST_JOBS = 100
_MAX_ACTIVE_BACKTEST_JOBS = 2
_broker_auth_status: dict[str, str] = {
    "state": "DISCONNECTED",
    "message": "Angel authentication not connected.",
}
_broker_auth_status_lock = Lock()
_integrity_status: dict[str, Any] = {
    "status": "DEGRADED",
    "trusted": False,
    "checked": [],
    "missing": [],
    "additions": [],
    "modifications": [],
    "reasons": ["integrity verification not run"],
}
_TEMP_DIR = Path(tempfile.gettempdir()) / "BharatTerminal"
_writer_service: Any | None = None


def _load_symbol_to_name() -> dict[str, str]:
    try:
        frame = read_instrument_frame(INSTRUMENTS_CACHE)
        nse_rows = frame[frame["exchange"] == "NSE"]
        mapping: dict[str, str] = {}
        for _, row in nse_rows.iterrows():
            symbol = str(row.get("symbol") or "").strip().upper()
            name = str(row.get("name") or "").strip()
            if not symbol or not name:
                continue
            mapping[symbol] = name
            if symbol.endswith("-EQ"):
                mapping.setdefault(symbol[:-3], name)
        logger.info("Loaded %s company names", len(mapping))
        return mapping
    except Exception as error:
        logger.warning("Could not load instruments master: %s", error)
        return {}


SYMBOL_TO_NAME: dict[str, str] = {}

# Stays "FINAL" for every real launch path (Task Scheduler, the Tauri parent
# process, a developer running this file directly). The ONLY place that ever
# overwrites it is the `release-openapi-probe` branch in `__main__` below,
# which release_gate.py's build-time OpenAPI smoke test is the sole caller
# of -- see `security.integrity.integrity_check` for why that phase exists
# and exactly what it still cryptographically enforces.
_INTEGRITY_EXPECTED_PHASE = "FINAL"

# Bounded local startup budget for each writer-queue call made during
# lifespan startup (see the RECOVER_BACKTESTS/APPLY_RETENTION calls below).
# Deliberately well under write_service's own internal 120s poll timeout so
# a stalled or unreachable writer authority degrades this one step instead
# of stalling the whole FastAPI readiness sequence.
_STARTUP_WRITER_BUDGET_SECONDS = 10.0


async def _startup_submit_write(
    operation: WriteOperation, payload: dict[str, Any], *, budget: float = _STARTUP_WRITER_BUDGET_SECONDS
) -> Any:
    """Run a writer-queue submit during lifespan startup without ever
    blocking the event loop or stalling startup past `budget` seconds.

    submit_write() is a synchronous, thread-blocking polling loop (see
    db/write_service.py: `time.sleep(POLL_INTERVAL_SECONDS)` inside a `while`
    bounded by its own internal timeout of up to 120s). Calling it directly
    from an async function blocks the whole process for however long the
    writer authority takes to respond -- `asyncio.to_thread` keeps that I/O
    off the event loop, and `asyncio.wait_for` enforces this process's own,
    much shorter, startup budget independent of write_service's internal
    timeout. Raises on timeout/failure; callers decide how to degrade.
    """
    return await asyncio.wait_for(asyncio.to_thread(submit_write, operation, payload), timeout=budget)


async def _run_writer_owner_startup_maintenance(*, owns_writer: bool) -> bool:
    """Run database-mutating startup maintenance only in the process that
    actually owns the DuckDB WriterService.

    A collector-first desktop (or an acceptance sidecar) is a remote queue
    client, not another startup authority.  Enqueuing recovery/retention from
    every remote client made a nominally read-only process launch mutate the
    production database and repeated global housekeeping once per process.
    The collector already owns and maintains that database; a standalone API
    still runs these tasks because it becomes the local writer owner.
    """
    if not owns_writer:
        logger.info("writer-owner startup maintenance skipped in remote-client process")
        return False
    try:
        interrupted_backtests = (
            await _startup_submit_write(WriteOperation.RECOVER_BACKTESTS, {})
        ).get("recovered", 0)
        if interrupted_backtests:
            logger.warning(
                "marked %s interrupted backtest run(s) failed after restart",
                interrupted_backtests,
            )
    except Exception as error:
        logger.warning("interrupted-backtest recovery skipped: %s", error)
    try:
        retention = await _startup_submit_write(WriteOperation.APPLY_RETENTION, {})
        logger.info("runtime retention complete: %s", retention)
    except Exception as error:
        logger.warning("runtime retention failed: %s", error)
    return True


def _start_instrument_refresh_if_owner(*, owns_writer: bool) -> bool:
    if not owns_writer:
        logger.info("instrument refresh skipped in remote-client process")
        return False
    Thread(target=_refresh_instrument_master_safely, daemon=True).start()
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _writer_service, _integrity_status, SYMBOL_TO_NAME
    runtime_data_dir = Path(os.environ.get("BHARATTERMINAL_APP_DATA", str(BASE_DIR))) / "data"
    _integrity_status = integrity_check(expected_phase=_INTEGRITY_EXPECTED_PHASE)
    allow_untrusted_dev = (
        not getattr(__import__("sys"), "frozen", False)
        and os.environ.get("BHARATTERMINAL_ALLOW_UNTRUSTED_DEV") == "1"
    )
    if not _integrity_status.get("trusted") and not allow_untrusted_dev:
        raise RuntimeError("protected release integrity verification failed")
    configure_structured_logging(runtime_data_dir / "logs" / "compute.jsonl")
    if not _integrity_status.get("trusted"):
        logger.warning("untrusted development runtime explicitly permitted")
    _clear_temp_dir()
    ensure_dirs()
    elog.initialize()
    init_schema()
    _writer_service = WriterService(DUCKDB_PATH)
    try:
        _writer_service.start()
    except (WriterServiceError, WriterLeaseError) as error:
        # Reproduced live (collector-first -> desktop-second topology): the
        # collector's own standalone startup already becomes the writer
        # authority via this exact same WriterService/writer_lease pair
        # (see smart_collector.py::_try_become_writer_authority, which
        # already documents falling through to submit_write()/submit_read()
        # against a remote owner when it can't become one). The desktop
        # backend previously had no matching fallback: `.start()` raising
        # here was uncaught, crash-looping the whole application on every
        # startup attempt. submit_write()/submit_read() talk to whichever
        # process actually owns the file via the shared queue file, not
        # through `_writer_service`, so this process works correctly as a
        # non-owning client once `_writer_service` is left None.
        logger.warning("DuckDB writer authority owned elsewhere; running as a remote client: %s", error)
        _writer_service = None
    owns_writer = _writer_service is not None
    await _run_writer_owner_startup_maintenance(owns_writer=owns_writer)
    SYMBOL_TO_NAME = _load_symbol_to_name()
    await seed_nifty_if_empty()
    _start_instrument_refresh_if_owner(owns_writer=owns_writer)
    _start_sentiment_refresh()
    try:
        yield
    finally:
        if _writer_service is not None:
            _writer_service.stop()
            _writer_service = None
        _clear_temp_dir()


def _refresh_instrument_master_safely() -> None:
    try:
        submit_write(WriteOperation.REFRESH_INSTRUMENT_MASTER, {"force": False})
    except Exception:
        # Full traceback to the internal structured log -- an
        # error-message-only log line here previously hid the real cause
        # (a Windows fsync-on-read-only-handle bug, see
        # instrument_master.py's _write_cache) for a long time. This is an
        # internal diagnostic log, never a user-facing response, so a full
        # traceback here is not a leakage concern.
        logger.exception("Instrument master refresh failed")


app = FastAPI(
    title="BharatTerminal Compute API",
    version=str(RELEASE_IDENTITY["python_api_version"]),
    lifespan=lifespan,
)

MAX_REQUEST_BODY_BYTES = 1024 * 1024
ALLOWED_BROWSER_ORIGINS = {
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        messages: list[dict[str, Any]] = []
        total = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                messages.append(message)
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                response = JSONResponse({"error": "request body too large"}, status_code=413)
                await response(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay() -> dict[str, Any]:
            return next(iterator, {"type": "http.request", "body": b"", "more_body": False})

        await self.app(scope, replay, send)


app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    return JSONResponse({"error": "invalid request"}, status_code=422)


@app.exception_handler(InstrumentSchemaError)
async def safe_instrument_master_error(_: Request, error: InstrumentSchemaError) -> JSONResponse:
    # Global handler so every route that can hit "no instrument master
    # loaded yet" / an unresolvable symbol gets the same stable, typed
    # precondition response -- not just the routes that happen to have their
    # own local try/except InstrumentSchemaError block. This is a genuine
    # "not ready yet" condition, not an unexpected server fault, so it
    # returns 503 with a caller-distinguishable error code rather than
    # falling through to the generic 500 handler below.
    return JSONResponse(
        {"error": "instrument_master_unavailable", "detail": str(error)},
        status_code=503,
    )


@app.exception_handler(Exception)
async def safe_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    # Catch-all for genuinely unexpected errors (does NOT intercept
    # HTTPException/StarletteHTTPException -- FastAPI's own more-specific
    # handler for that type is matched first, so existing `raise
    # HTTPException(status_code=..., detail=...)` call sites are unaffected).
    # Never swallows the failure: it still surfaces as a real 500 to the
    # caller (no fake success) and the full exception is logged server-side
    # with a traceback -- only the CLIENT-facing body is reduced to a
    # stable, bounded contract with no stack trace, file path, or secret
    # ever included.
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"error": "internal_error", "detail": "an unexpected error occurred"},
        status_code=500,
    )


@app.middleware("http")
async def verify_process_token(request: Request, call_next):
    if request.url.hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}:
        return JSONResponse({"error": "invalid host"}, status_code=400)
    origin = request.headers.get("Origin")
    if origin:
        parsed_origin = urlsplit(origin)
        normalized_origin = f"{parsed_origin.scheme}://{parsed_origin.netloc}"
        if normalized_origin not in ALLOWED_BROWSER_ORIGINS:
            return JSONResponse({"error": "origin forbidden"}, status_code=403)
    content_length = request.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) < 0 or int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse({"error": "request body too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"error": "invalid content length"}, status_code=400)
    lowered_path = request.url.path.lower()
    raw_path = request.scope.get("raw_path", b"")
    if isinstance(raw_path, bytes):
        raw_path_text = raw_path.decode("utf-8", errors="ignore").lower()
    else:
        raw_path_text = str(raw_path).lower()
    if (
        ".." in lowered_path
        or "%2e%2e" in lowered_path
        or "%2f" in lowered_path
        or ".." in raw_path_text
        or "%2e%2e" in raw_path_text
        or "%2f" in raw_path_text
    ):
        return JSONResponse({"error": "path traversal attempt blocked"}, status_code=400)
    if request.url.path == "/health/live":
        return await call_next(request)
    token = request.headers.get("X-Process-Token")
    if not secrets.compare_digest(token or "", PROCESS_TOKEN):
        return JSONResponse({"error": "unauthorized"}, status_code=403)
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, Any]:
    return _health_snapshot()


@app.get("/health/live")
async def health_live() -> dict[str, Any]:
    snapshot = _health_snapshot()
    return {"status": snapshot["liveness"]["status"]}


@app.get("/health/ready")
async def health_ready() -> Any:
    snapshot = _health_snapshot()
    status_code = 200 if snapshot["readiness"]["ready"] else 503
    return JSONResponse(snapshot, status_code=status_code)


def _health_snapshot() -> dict[str, Any]:
    from db import duckdb_client
    from live import multi_feed
    from operations.health import health_snapshot

    return health_snapshot(
        database_path=duckdb_client.DB_PATH,
        collector_db=DATA_DIR / "collector_log.db",
        redis_client=_redis_client,
        feed_status=multi_feed.get_status,
        release=RELEASE_IDENTITY,
        integrity=_integrity_status,
        external_success_times={
            "cross_asset": _cross_asset_cache_time,
            "sentiment": _sentiment_cache_time,
        },
    )


@app.get("/live/quote/{symbol}")
def live_quote(symbol: str) -> Any:
    redis_symbol = _live_redis_symbol(symbol)
    key = f"tick:{redis_symbol}"
    try:
        payload = _redis_client.get(key)
    except _redis.RedisError as error:
        raise HTTPException(status_code=503, detail=f"redis GET {key} failed") from error

    if payload is None:
        return {"error": "no live data", "symbol": redis_symbol}

    tick = _decode_live_tick(payload, redis_symbol, key)
    if tick is None:
        return {"error": "no live data", "symbol": redis_symbol}
    return tick


@app.get("/live/quotes")
def live_quotes() -> Any:
    quotes: dict[str, dict[str, Any]] = {}
    try:
        for raw_key in _redis_client.scan_iter(match="tick:*", count=500):
            if len(quotes) >= 500:
                break
            key = _redis_text(raw_key)
            symbol = key.split(":", 1)[1] if ":" in key else key
            payload = _redis_client.get(key)
            if payload is None:
                continue
            tick = _decode_live_tick(payload, symbol, key)
            if tick is None:
                continue
            quotes[symbol] = tick
    except _redis.RedisError as error:
        raise HTTPException(status_code=503, detail="redis tick scan failed") from error

    return quotes


@app.get("/live/feed_status")
def live_feed_status() -> Any:
    try:
        from live import multi_feed

        return multi_feed.get_status()
    except Exception as error:
        return {
            "connected": False,
            "symbols_subscribed": 0,
            "symbols_active": 0,
            "last_tick_ms_ago": -1,
            "uptime_seconds": 0,
            "token_count": 0,
            "error": str(error),
        }


@app.get("/broker/auth/status")
def broker_auth_status() -> dict[str, str]:
    with _broker_auth_status_lock:
        return dict(_broker_auth_status)


@app.get("/market/session")
def get_market_session() -> Any:
    return exchange_calendar().session_snapshot()


@app.get("/instruments/feed/{symbol}")
def resolve_feed_instrument(symbol: str) -> Any:
    normalized = http_symbol(symbol)
    try:
        return instrument_master.resolve_index_feed(normalized)
    except InstrumentSchemaError as error:
        raise HTTPException(
            status_code=503,
            detail=f"CONFIGURATION / INSTRUMENT RESOLUTION ERROR: {error}",
        ) from error


@app.get("/instruments/lot/{symbol}")
def resolve_instrument_lot_size(symbol: str, expiry: str | None = None) -> Any:
    normalized = http_symbol(symbol)
    try:
        lot_size = instrument_master.get_lot_size(normalized, expiry)
    except InstrumentSchemaError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"symbol": normalized, "expiry": expiry, "lot_size": lot_size}


@app.websocket("/internal/angel/feed")
async def internal_angel_feed(websocket: WebSocket) -> None:
    client_host = websocket.client.host if websocket.client else ""
    token = websocket.headers.get("X-Process-Token")
    if client_host not in {"127.0.0.1", "::1", "testclient"} or not secrets.compare_digest(
        token or "", PROCESS_TOKEN
    ):
        await websocket.close(code=1008)
        return
    await _bridge_angel_feed(websocket)


@app.get("/app/integrity")
def app_integrity() -> Any:
    return _integrity_status


@app.get("/collector/status")
def collector_status() -> Any:
    import sqlite3

    log_db = DATA_DIR / "collector_log.db"
    log_file = DATA_DIR / "collector.log"
    heartbeat_file = DATA_DIR / "collector_heartbeat.json"
    log_fresh = False
    if log_file.exists():
        age_s = datetime.datetime.now().timestamp() - log_file.stat().st_mtime
        log_fresh = age_s <= 45 * 60
    heartbeat: dict[str, Any] = {}
    if heartbeat_file.exists():
        try:
            heartbeat = json.loads(heartbeat_file.read_text(encoding="utf-8"))
        except Exception:
            heartbeat = {}
    supervision = status_snapshot(log_db)

    if not log_db.exists():
        return {
            **supervision,
            "collector_recent_log": log_fresh,
            "heartbeat_diagnostic": heartbeat,
            "last_success": None,
            "recent_tasks": [],
            "error": "collector log DB not found",
        }

    try:
        with sqlite3.connect(str(log_db)) as con:
            con.row_factory = sqlite3.Row
            recent = con.execute(
                """
                SELECT task, date, status, details, started_at, finished_at, duration_s
                FROM collector_runs
                ORDER BY started_at DESC LIMIT 10
                """
            ).fetchall()
            last_success = con.execute(
                """
                SELECT MAX(started_at)
                FROM collector_runs
                WHERE status = 'SUCCEEDED'
                """
            ).fetchone()[0]
        return {
            **supervision,
            "collector_recent_log": log_fresh,
            "heartbeat_diagnostic": heartbeat,
            "last_success": last_success,
            "recent_tasks": [dict(row) for row in recent],
        }
    except Exception as error:
        return {
            **supervision,
            "collector_recent_log": log_fresh,
            "heartbeat_diagnostic": heartbeat,
            "last_success": None,
            "recent_tasks": [],
            "error": str(error),
        }


@app.post("/greeks")
def compute_greeks(request: GreeksRequest) -> Any:
    from options.greeks import compute_chain_greeks

    request.symbol = http_symbol(request.symbol)
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
        logger.debug("greeks computed for %s strikes in %sms", len(result), elapsed_ms)
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

    request.symbol = http_symbol(request.symbol)
    request.from_date = http_date(request.from_date)
    request.to_date = http_date(request.to_date)
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


@app.get("/stocks/snapshot/{symbol}")
def stock_snapshot(symbol: str) -> Any:
    from data.stock_snapshot import get_stock_snapshot

    symbol = http_symbol(symbol)
    try:
        snapshot = get_stock_snapshot(symbol)
    except Exception as error:
        print(f"stocks: snapshot error: {error}")
        return {"error": str(error)}

    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"symbol not found: {symbol}")
    return snapshot


@app.get("/stocks/universe")
def stock_universe(limit: int = 250, universe: str = "FNO") -> Any:
    from data.stock_snapshot import get_stock_universe

    try:
        return get_stock_universe(universe, limit)
    except Exception as error:
        print(f"stocks: universe error: {error}")
        return {"error": str(error), "results": []}


@app.post("/screener/run")
def run_stock_screener(request: StockScreenerRequest) -> Any:
    from data.stock_snapshot import run_screener_query

    try:
        return run_screener_query(
            request.conditions,
            request.logic,
            request.universe,
            request.limit,
        )
    except Exception as error:
        print(f"stocks: screener error: {error}")
        return {"error": str(error), "results": []}


@app.get("/equity/news/{symbol}")
def equity_news(symbol: str) -> Any:
    normalized = http_symbol(symbol)
    cache_key = f"news:{normalized}"
    cached = _equity_cache_get(cache_key)
    if cached is not None:
        return cached

    news = _fetch_equity_news(normalized)
    payload = {"symbol": normalized, "news": news}
    return payload


@app.get("/equity/financials/{symbol}")
def equity_financials(symbol: str) -> Any:
    normalized = http_symbol(symbol)
    cache_key = f"financials:{normalized}"
    cached = _equity_cache_get(cache_key)
    if cached is not None:
        return cached

    payload = _fetch_equity_financials(normalized)
    return payload


@app.get("/equity/filings/{symbol}")
def equity_filings(symbol: str) -> Any:
    normalized = http_symbol(symbol)
    cache_key = f"filings:{normalized}"
    cached = _equity_cache_get(cache_key)
    if cached is not None:
        return cached

    payload = _fetch_equity_filings(normalized)
    return payload


@app.get("/equity/summary/{symbol}")
def equity_summary(symbol: str) -> Any:
    normalized = http_symbol(symbol)
    cache_key = f"summary:{normalized}"
    cached = _equity_cache_get(cache_key)
    if cached is not None:
        return cached

    news_payload = equity_news(normalized)
    financials_payload = equity_financials(normalized)
    filings_payload = equity_filings(normalized)
    financials = copy.deepcopy(financials_payload)
    if isinstance(financials.get("quarters"), list):
        financials["quarters"] = financials["quarters"][:4]
    payload = {
        "symbol": normalized,
        "company_name": _company_name(normalized),
        "news": list(news_payload.get("news") or [])[:5],
        "financials": financials,
        "filings": list(filings_payload.get("filings") or [])[:5],
    }
    return payload


@app.get("/equity/company/{symbol}")
def equity_company(symbol: str) -> Any:
    normalized = http_symbol(symbol)
    return {"symbol": normalized, "name": _company_name(normalized)}


@app.post("/payoff")
def compute_payoff_endpoint(request: PayoffRequest) -> Any:
    from options.payoff import compute_payoff

    try:
        legs = [_model_dump(leg) for leg in request.legs]
        for leg in legs:
            canonical_lot_size = instrument_master.get_lot_size(
                str(leg["symbol"]),
                str(leg["expiry"]),
            )
            if int(leg["lot_size"]) != canonical_lot_size:
                raise ValueError(
                    f"lot_size mismatch for {leg['symbol']} {leg['expiry']}: "
                    f"expected {canonical_lot_size}, received {leg['lot_size']}"
                )
        return compute_payoff(legs, request.spot, request.spot_range_pct)
    except (InstrumentSchemaError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/futures/basis/{symbol}")
def get_futures_basis(symbol: str) -> Any:
    from futures.basis import compute_futures_basis

    symbol = http_symbol(symbol)
    try:
        return compute_futures_basis(symbol)
    except Exception as e:
        print(f"futures: basis error: {e}")
        return {"error": str(e)}


@app.get("/signals/{symbol}")
def get_signals(symbol: str) -> Any:
    from signals.options_signals import compute_options_signals

    symbol = http_symbol(symbol)
    expiries = get_next_expiries(1)
    expiry = expiries[0] if expiries else ""
    legs = get_chain_from_redis(symbol, expiry)
    if not legs:
        legs = _read_chain_legs_from_redis(symbol, [expiry])

    spot_observation = _read_spot_observation_from_redis(symbol)
    spot = float(spot_observation.price) if spot_observation.decision_eligible else None
    try:
        lot_size = instrument_master.get_lot_size(symbol, expiry)
    except InstrumentSchemaError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    if spot is None:
        options = {"error": "verified live spot unavailable", "available": False}
    else:
        try:
            options = compute_options_signals(legs, spot, expiry, lot_size)
        except Exception as e:
            print(f"signals: options computation error: {e}")
            options = {"error": str(e), "available": False}

    return {
        "options": options,
        "market": _get_cached_market_signals(),
        "spot_provenance": spot_observation.as_provenance(),
        "decision_eligible": spot_observation.decision_eligible,
        "expiry": expiry,
        "computed_at": _time.time(),
    }


@app.get("/signals/all/{symbol}")
def get_all_signals(symbol: str) -> Any:
    from signals.aggregator import compute_all_signals

    normalized_symbol = http_symbol(symbol)
    now = _time.time()
    with _all_signals_cache_lock:
        cached = _all_signals_cache.get(normalized_symbol)
        cached_at = _all_signals_cache_time.get(normalized_symbol, 0.0)
        if cached and now - cached_at < _ALL_SIGNALS_CACHE_TTL:
            return copy.deepcopy(cached)

    expiry = get_next_expiries(1)[0]
    legs = get_chain_from_redis(normalized_symbol, expiry)
    if not legs:
        legs = _read_chain_legs_from_redis(normalized_symbol, [expiry])

    spot_observation = _read_spot_observation_from_redis(normalized_symbol)
    spot = float(spot_observation.price) if spot_observation.decision_eligible else None
    try:
        lot_size = instrument_master.get_lot_size(normalized_symbol, expiry)
    except InstrumentSchemaError:
        lot_size = None
    if not legs and spot is not None:
        fallback_legs, _fallback_spot, fallback_expiry = _fetch_chain_legs_with_greeks(
            normalized_symbol,
            expiry,
            spot,
        )
        if fallback_legs:
            legs = fallback_legs
            expiry = fallback_expiry or expiry

    ohlcv_df = _load_ohlcv_frame(normalized_symbol)
    market_data_eligibility = _current_market_data_eligibility(ohlcv_df)
    analytical_ohlcv = (
        ohlcv_df
        if market_data_eligibility.decision_eligible
        else ohlcv_df.iloc[0:0].copy()
    )
    market = _get_cached_market_signals()
    cross_asset = _safe_cross_asset_snapshot()
    sentiment = _safe_current_sentiment()

    try:
        payload = compute_all_signals(
            normalized_symbol,
            analytical_ohlcv,
            legs,
            spot,
            expiry,
            cross_asset,
            sentiment,
            market.get("fii_dii", {}),
            market.get("india_vix"),
            lot_size=lot_size,
        )
    except Exception as error:
        print(f"signals: all-signals aggregation failed: {error}")
        payload = {
            "symbol": normalized_symbol,
            "spot": spot,
            "computed_at": _time.time(),
            "error": str(error),
            "signal_counts": {
                "total": 0,
                "non_null": 0,
                "null": 0,
                "by_category": {},
            },
            "categories": {},
            "composite_scores": {},
        }

    payload["spot_provenance"] = spot_observation.as_provenance()
    payload["market_data_eligibility"] = market_data_eligibility.to_dict()
    payload["decision_eligible"] = (
        spot_observation.decision_eligible and market_data_eligibility.decision_eligible
    )

    with _all_signals_cache_lock:
        _all_signals_cache[normalized_symbol] = copy.deepcopy(payload)
        _all_signals_cache_time[normalized_symbol] = now
        _prune_symbol_cache(_all_signals_cache, _all_signals_cache_time)

    return payload


@app.post("/signals/backfill")
def backfill_signals() -> Any:
    try:
        return submit_write(WriteOperation.BACKFILL_PRICE_ACTION, {})
    except Exception as error:
        print(f"signals: price_action backfill failed: {error}")
        return {"rows_written": 0, "error": str(error)}


@app.get("/ml/predict/{symbol}")
def get_ml_prediction(symbol: str) -> Any:
    from ml.service import predict_all_models

    normalized_symbol = http_symbol(symbol)
    signal_payload = get_all_signals(normalized_symbol)
    try:
        return predict_all_models(normalized_symbol, signal_payload, log=False)
    except Exception as error:
        print(f"ml: prediction failed: {error}")
        return {
            "symbol": normalized_symbol,
            "computed_at": _time.time(),
            "composite_direction": None,
            "composite_confidence": None,
            "component_outputs": {},
            "error": str(error),
        }


@app.post("/ml/predict")
def post_ml_prediction(request: MLPredictRequest) -> Any:
    normalized_symbol = http_symbol(request.symbol)
    signal_payload = request.signals if request.signals is not None else get_all_signals(normalized_symbol)
    try:
        return submit_write(
            WriteOperation.PREDICT_AND_LOG,
            {"symbol": normalized_symbol, "signals": signal_payload},
        )
    except Exception as error:
        print(f"ml: prediction failed: {error}")
        return {
            "symbol": normalized_symbol,
            "computed_at": _time.time(),
            "composite_direction": None,
            "composite_confidence": None,
            "component_outputs": {},
            "error": str(error),
        }


@app.get("/ml/accuracy")
def get_ml_accuracy() -> Any:
    from ml.monitor import accuracy_report

    return accuracy_report(30)


@app.post("/ml/predictions/{prediction_id}/outcome")
def post_ml_prediction_outcome(prediction_id: str, request: MLOutcomeRequest) -> Any:
    return submit_write(
        WriteOperation.RECORD_PREDICTION_OUTCOME,
        {"prediction_id": prediction_id, "actual": request.actual},
    )


@app.get("/bif/all/{symbol}")
def get_bif_all(symbol: str) -> Any:
    symbol = http_symbol(symbol)
    return _get_bif_engine().compute_all(symbol)


@app.get("/bif/cds/{symbol}")
def get_bif_cds(symbol: str) -> Any:
    symbol = http_symbol(symbol)
    return _get_bif_engine().get_cds_only(symbol)


@app.get("/bif/history/{symbol}")
def get_bif_history(symbol: str, days: int = 30) -> Any:
    symbol = http_symbol(symbol)
    return _get_bif_engine().get_history(symbol, days)


@app.post("/backtest/run")
async def run_backtest(
    request: BacktestRunRequest,
    background_tasks: BackgroundTasks,
) -> Any:
    params = _normalise_backtest_params(_model_dump(request))
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "params": params,
        "result": None,
        "error": None,
    }
    if not _reserve_backtest_job(job_id, payload):
        raise HTTPException(status_code=429, detail="backtest capacity reached")
    try:
        submit_write(
            WriteOperation.SAVE_BACKTEST_STARTED,
            {"job_id": job_id, "params": params},
            request_id=f"backtest-start:{job_id}",
        )
    except Exception:
        with _backtest_jobs_lock:
            _backtest_jobs.pop(job_id, None)
        raise
    background_tasks.add_task(_submit_backtest_job, job_id, params)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/backtest/status/{job_id}",
        "result_url": f"/backtest/result/{job_id}",
    }


@app.get("/backtest/status/{job_id}")
def backtest_status(job_id: str) -> Any:
    from backtest.engine import load_run

    payload = load_run(job_id)
    cached = _get_backtest_job(job_id)
    if cached:
        payload.update({key: cached.get(key) for key in ("status", "progress", "stage", "error")})
        if cached.get("result") is not None:
            payload["result"] = cached["result"]
    return payload


@app.get("/backtest/result/{job_id}")
def backtest_result(job_id: str) -> Any:
    from backtest.engine import load_run

    payload = load_run(job_id)
    cached = _get_backtest_job(job_id)
    if cached and cached.get("result") is not None:
        payload["result"] = cached["result"]
    return payload


@app.get("/backtest/strategies")
def list_backtest_strategies() -> Any:
    from backtest.strategies import list_strategies

    strategies = list_strategies()
    return {"strategies": strategies, "count": len(strategies)}


@app.get("/backtest/history")
def backtest_history(limit: int = 25) -> Any:
    from backtest.engine import load_history

    runs = load_history(limit)
    return {"runs": runs, "count": len(runs)}


@app.post("/ml/backfill_nifty_ohlcv")
def backfill_nifty_ohlcv() -> Any:
    raise HTTPException(
        status_code=410,
        detail=(
            "futures-derived spot backfill was removed; ingest verified "
            "NSE_INDEX_BHAVCOPY observations"
        ),
    )


@app.post("/ml/retrain")
def retrain_ml(background_tasks: BackgroundTasks, force: bool = False) -> Any:
    return _queue_ml_training(background_tasks, force)


@app.post("/ml/retrain/{symbol}")
def retrain_ml_symbol(
    symbol: str,
    background_tasks: BackgroundTasks,
    force: bool = True,
) -> Any:
    normalized_symbol = symbol.strip().upper()
    if normalized_symbol in {"ALL", "FNO", "STOCKS"}:
        with _stock_training_lock:
            if _stock_training_state["running"]:
                return {"status": "already_running", "check_at": "/ml/training_status"}
            _stock_training_state.update(
                {
                    "running": True,
                    "stage": "queued",
                    "current_symbol": None,
                    "done": 0,
                    "total": 0,
                    "trained": 0,
                    "errors": 0,
                    "started_at": datetime.datetime.now().isoformat(),
                    "elapsed_minutes": 0,
                    "last_report": None,
                }
            )
        background_tasks.add_task(_run_stock_training, force)
        return {
            "status": "started",
            "symbol": normalized_symbol,
            "force": force,
            "monitor_at": "/ml/training_status",
        }

    with _stock_training_lock:
        normalized_symbol = http_symbol(normalized_symbol)
        if _stock_training_state["running"]:
            return {"status": "already_running", "check_at": "/ml/training_status"}
        _stock_training_state.update(
            {
                "running": True,
                "stage": "single_symbol",
                "current_symbol": normalized_symbol,
                "done": 0,
                "total": 1,
                "trained": 0,
                "errors": 0,
                "started_at": datetime.datetime.now().isoformat(),
                "elapsed_minutes": 0,
                "last_report": None,
            }
        )
    background_tasks.add_task(_run_single_stock_training, normalized_symbol, force)
    return {
        "status": "started",
        "symbol": normalized_symbol,
        "force": force,
        "monitor_at": "/ml/training_status",
    }


def _queue_ml_training(background_tasks: BackgroundTasks, force: bool = False) -> dict[str, Any]:
    with _ml_training_lock:
        if _ml_training_state["running"]:
            return {
                "status": "already_running",
                "last_report": _ml_training_state["last_report"],
            }
        _ml_training_state["running"] = True
        _ml_training_state["last_error"] = None

    background_tasks.add_task(_run_ml_training, force)
    return {"status": "queued", "force": force}


@app.get("/ml/status")
def get_ml_status() -> Any:
    from ml.trainer import status_report

    with _ml_training_lock:
        state = copy.deepcopy(_ml_training_state)
    report = status_report()
    report["training"] = state
    return report


@app.get("/ml/ohlc_forecast/{symbol}")
def ohlc_forecast(symbol: str) -> Any:
    normalized_symbol = http_symbol(symbol)
    cache_key = normalized_symbol
    now = _time.time()
    with _ohlc_forecast_cache_lock:
        cached = _ohlc_forecast_cache.get(cache_key)
        cached_at = _ohlc_forecast_cache_time.get(cache_key, 0.0)
        if cached and now - cached_at < _OHLC_FORECAST_CACHE_TTL:
            return copy.deepcopy(cached)

    from ml.models.m21_ohlc_forecast import OHLCForecaster

    frame = _load_ohlcv_full_frame(normalized_symbol)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"symbol not found: {normalized_symbol}")

    market_eligibility = _current_market_data_eligibility(frame)
    if not market_eligibility.decision_eligible:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "current-decision OHLCV unavailable",
                "market_data_eligibility": market_eligibility.to_dict(),
            },
        )
    spot_observation = _read_spot_observation_from_redis(normalized_symbol)
    current_close = (
        float(spot_observation.price)
        if spot_observation.decision_eligible and spot_observation.price is not None
        else _to_float(frame["close"].tail(1).iloc[0])
    )
    atm_iv = _read_atm_iv_from_redis(normalized_symbol, current_close)
    market = _market_cache_snapshot()
    vix = _to_float(market.get("india_vix")) if isinstance(market, dict) else 0.0
    forecaster = OHLCForecaster()
    payload = forecaster.predict(
        normalized_symbol,
        frame,
        current_close,
        atm_iv=atm_iv if atm_iv > 0 else None,
        vix=vix if vix > 0 else None,
        include_accuracy=True,
        log_forecast=False,
    )
    payload["market_data_eligibility"] = market_eligibility.to_dict()
    payload["spot_provenance"] = spot_observation.as_provenance()
    payload["current_close_fallback"] = (
        "NONE" if spot_observation.decision_eligible else "EXPECTED_COMPLETED_SESSION_OHLCV"
    )
    with _ohlc_forecast_cache_lock:
        _ohlc_forecast_cache[cache_key] = copy.deepcopy(payload)
        _ohlc_forecast_cache_time[cache_key] = now
        _prune_symbol_cache(_ohlc_forecast_cache, _ohlc_forecast_cache_time)
    return payload


@app.post("/ml/train_all_stocks")
async def train_all_stocks(background_tasks: BackgroundTasks, force: bool = False) -> Any:
    with _stock_training_lock:
        if _stock_training_state["running"]:
            return {"status": "already_running", "check_at": "/ml/training_status"}
        _stock_training_state.update(
            {
                "running": True,
                "stage": "queued",
                "current_symbol": None,
                "done": 0,
                "total": 0,
                "trained": 0,
                "errors": 0,
                "started_at": datetime.datetime.now().isoformat(),
                "elapsed_minutes": 0,
                "last_report": None,
            }
        )

    background_tasks.add_task(_run_stock_training, force)
    return {
        "status": "started",
        "message": "Training ML models for all FNO stocks",
        "estimated_minutes": "30-60",
        "monitor_at": "/ml/training_status",
    }


@app.get("/ml/training_status")
def training_status() -> Any:
    with _stock_training_lock:
        state = copy.deepcopy(_stock_training_state)
    if state["started_at"]:
        started_at = datetime.datetime.fromisoformat(state["started_at"])
        state["elapsed_minutes"] = round(
            (datetime.datetime.now() - started_at).total_seconds() / 60.0,
            2,
        )
    return state


@app.get("/ml/trained_symbols")
def trained_symbols() -> Any:
    from ml.stock_trainer import trained_symbols_summary

    return trained_symbols_summary()


@app.post("/ml/train_symbol/{symbol}")
async def train_single_symbol(
    symbol: str,
    background_tasks: BackgroundTasks,
    force: bool = False,
) -> Any:
    normalized_symbol = http_symbol(symbol)
    with _stock_training_lock:
        if _stock_training_state["running"]:
            return {"status": "already_running", "check_at": "/ml/training_status"}
        _stock_training_state.update(
            {
                "running": True,
                "stage": "single_symbol",
                "current_symbol": normalized_symbol,
                "done": 0,
                "total": 1,
                "trained": 0,
                "errors": 0,
                "started_at": datetime.datetime.now().isoformat(),
                "elapsed_minutes": 0,
                "last_report": None,
            }
        )

    background_tasks.add_task(_run_single_stock_training, normalized_symbol, force)
    return {
        "status": "started",
        "symbol": normalized_symbol,
        "estimated_seconds": "30-120",
        "monitor_at": "/ml/training_status",
    }


@app.get("/decision/intelligence")
def get_decision_intelligence_default(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"decision:{symbol.strip().upper()}",
        lambda: _build_decision_intelligence(symbol),
    )


@app.get("/decision/intelligence/{symbol}")
def get_decision_intelligence(symbol: str) -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"decision:{symbol.strip().upper()}",
        lambda: _build_decision_intelligence(symbol),
    )


@app.get("/narrative/market_brief")
def get_market_brief(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"brief:{symbol.strip().upper()}",
        lambda: _build_market_brief(symbol),
    )


@app.get("/intelligence/patterns")
def get_intelligence_patterns(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"patterns:{symbol.strip().upper()}",
        lambda: _build_patterns(symbol),
    )


@app.get("/intelligence/alpha")
def get_intelligence_alpha(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"alpha:{symbol.strip().upper()}",
        lambda: _build_alpha(symbol),
    )


@app.get("/intelligence/regime")
def get_intelligence_regime(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"regime:{symbol.strip().upper()}",
        lambda: _build_regime(symbol),
    )


@app.get("/intelligence/options_flow")
def get_intelligence_options_flow(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"options_flow:{symbol.strip().upper()}",
        lambda: _build_options_flow(symbol),
    )


@app.get("/risk/var")
def get_risk_var(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"risk_var:{symbol.strip().upper()}",
        lambda: _build_var(symbol),
    )


@app.get("/risk/greeks/{strategy}")
def get_risk_greeks(strategy: str, symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"risk_greeks:{symbol.strip().upper()}:{strategy}",
        lambda: _build_greeks_dashboard(symbol, strategy),
    )


@app.get("/risk/drawdown")
def get_risk_drawdown(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"risk_drawdown:{symbol.strip().upper()}",
        lambda: _build_drawdown(symbol),
    )


@app.get("/risk/stress_test")
def get_risk_stress_test(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"risk_stress:{symbol.strip().upper()}",
        lambda: _build_stress_test(symbol),
    )


@app.get("/risk/correlations")
def get_risk_correlations(symbol: str = "NIFTY") -> Any:
    symbol = http_symbol(symbol)
    return _cached_apex(
        f"risk_corr:{symbol.strip().upper()}",
        lambda: _build_correlations(symbol),
    )


def _run_ml_training(force: bool = False) -> None:
    try:
        from ml.trainer import train_all_models

        report = train_all_models(force=force, symbol="NIFTY")
        with _ml_training_lock:
            _ml_training_state["last_report"] = report
            _ml_training_state["last_error"] = None
    except Exception as error:
        with _ml_training_lock:
            _ml_training_state["last_error"] = str(error)
    finally:
        with _ml_training_lock:
            _ml_training_state["running"] = False


def _run_stock_training(force: bool = False) -> None:
    try:
        from ml.stock_trainer import StockModelTrainer

        summary = StockModelTrainer().train_all_fno_stocks(force=force)
        with _stock_training_lock:
            _stock_training_state["last_report"] = summary
            _stock_training_state["stage"] = "complete"
    except Exception as error:
        with _stock_training_lock:
            _stock_training_state["stage"] = f"error: {error}"
            _stock_training_state["errors"] = int(_stock_training_state.get("errors", 0)) + 1
            _stock_training_state["last_report"] = {"error": str(error)}
    finally:
        with _stock_training_lock:
            _stock_training_state["running"] = False
            _stock_training_state["current_symbol"] = None


def _run_single_stock_training(symbol: str, force: bool = False) -> None:
    try:
        from ml.stock_trainer import StockModelTrainer

        result = StockModelTrainer().train_symbol(symbol, force=force)
        with _stock_training_lock:
            _stock_training_state["done"] = 1
            _stock_training_state["trained"] = 1 if result.get("models_trained") else 0
            _stock_training_state["errors"] = 1 if result.get("error") else 0
            _stock_training_state["stage"] = "complete" if not result.get("error") else "error"
            _stock_training_state["last_report"] = result
    except Exception as error:
        with _stock_training_lock:
            _stock_training_state["done"] = 1
            _stock_training_state["errors"] = 1
            _stock_training_state["stage"] = f"error: {error}"
            _stock_training_state["last_report"] = {"symbol": symbol, "error": str(error)}
    finally:
        with _stock_training_lock:
            _stock_training_state["running"] = False
            _stock_training_state["current_symbol"] = None


def _get_bif_engine() -> Any:
    global _bif_engine
    with _bif_engine_lock:
        if _bif_engine is None:
            from bif.engine import BIFEngine

            _bif_engine = BIFEngine(
                providers={
                    "chain": _bif_chain_provider,
                    "spot": _bif_spot_provider,
                    "market": _get_cached_market_signals,
                    "cross_asset": _safe_cross_asset_snapshot,
                    "signals": get_all_signals,
                    "ml": _bif_ml_provider,
                }
            )
        return _bif_engine


def _bif_chain_provider(symbol: str) -> dict[str, Any]:
    expiries = get_next_expiries(4)
    expiry = expiries[0]
    legs = get_chain_from_redis(symbol, expiry)
    if not legs:
        legs = _read_chain_legs_from_redis(symbol, expiries)
    return {"legs": legs, "expiry": expiry}


def _bif_spot_provider(symbol: str) -> Any:
    """BIF accepts canonical provenance only; no strike or stale-OHLC rescue."""
    return _read_spot_observation_from_redis(symbol)


def _bif_ml_provider(symbol: str, signals_payload: dict[str, Any]) -> dict[str, Any]:
    from ml.service import predict_all_models

    return predict_all_models(symbol.strip().upper(), signals_payload, log=False)


def _normalise_backtest_params(params: dict[str, Any]) -> dict[str, Any]:
    today = today_ist()
    from_day = _parse_date(str(params.get("from_date") or "2024-01-01"))
    to_day = _parse_date(str(params.get("to_date") or today.isoformat()))
    if from_day > to_day:
        raise HTTPException(status_code=400, detail="from_date must be before to_date")
    capital = float(params.get("initial_capital") or 500000)
    if capital <= 0:
        raise HTTPException(status_code=400, detail="initial_capital must be positive")
    return {
        "strategy": str(params.get("strategy") or "sell_weekly_atm_straddle"),
        "symbol": str(params.get("symbol") or "NIFTY").strip().upper(),
        "from_date": from_day.isoformat(),
        "to_date": to_day.isoformat(),
        "initial_capital": capital,
        "chronological_segments": bool(params.get("chronological_segments", True)),
        "monte_carlo": bool(params.get("monte_carlo", True)),
        "n_mc_simulations": int(params.get("n_mc_simulations") or 10000),
        "cost_model": str(params.get("cost_model") or "full"),
        "slippage_model": str(params.get("slippage_model") or "realistic"),
        "position_sizer": str(params.get("position_sizer") or "quarter_kelly"),
    }


def _submit_backtest_job(job_id: str, params: dict[str, Any]) -> None:
    try:
        submit_write(
            WriteOperation.RUN_BACKTEST_JOB,
            {"job_id": job_id, "params": params},
            request_id=f"backtest-run:{job_id}",
            timeout=60 * 60,
        )
    except Exception as error:
        _update_backtest_job(job_id, status="failed", stage="failed", error=str(error))


def _run_backtest_job(job_id: str, params: dict[str, Any]) -> None:
    from backtest.engine import (
        BacktestEngine,
        save_run_completed,
        save_run_failed,
        save_run_progress,
    )

    def progress(percent: float, stage: str) -> None:
        safe_percent = max(0.0, min(1.0, float(percent)))
        _update_backtest_job(
            job_id,
            status="running",
            progress=safe_percent,
            stage=stage,
        )
        try:
            save_run_progress(job_id, safe_percent, stage)
        except Exception as error:
            print(f"backtest: progress save failed: {error}")

    try:
        progress(0.01, "starting")
        engine = BacktestEngine(**params)
        result = engine.run(progress)
        save_run_completed(job_id, result)
        _update_backtest_job(
            job_id,
            status="completed",
            progress=1.0,
            stage="complete",
            result=result,
            error=None,
        )
    except Exception as error:
        print(f"backtest: job {job_id} failed: {error}")
        try:
            save_run_failed(job_id, error)
        except Exception as save_error:
            print(f"backtest: failure save failed: {save_error}")
        _update_backtest_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(error),
        )


def _reserve_backtest_job(job_id: str, payload: dict[str, Any]) -> bool:
    with _backtest_jobs_lock:
        active = sum(
            str(job.get("status", "")).lower() in {"queued", "running"}
            for job in _backtest_jobs.values()
        )
        if active >= _MAX_ACTIVE_BACKTEST_JOBS:
            return False
        _backtest_jobs[job_id] = copy.deepcopy(payload)
        _prune_backtest_jobs_locked()
        return True


def _update_backtest_job(job_id: str, **updates: Any) -> None:
    with _backtest_jobs_lock:
        job = _backtest_jobs.setdefault(job_id, {"job_id": job_id})
        job.update(updates)
        _prune_backtest_jobs_locked()


def _get_backtest_job(job_id: str) -> dict[str, Any] | None:
    with _backtest_jobs_lock:
        job = _backtest_jobs.get(job_id)
        return copy.deepcopy(job) if job else None


def _prune_backtest_jobs_locked() -> None:
    terminal = {"completed", "failed", "cancelled"}
    for stale_id in list(_backtest_jobs):
        if len(_backtest_jobs) <= _MAX_IN_MEMORY_BACKTEST_JOBS:
            break
        if str(_backtest_jobs[stale_id].get("status", "")).lower() in terminal:
            _backtest_jobs.pop(stale_id, None)


def _prune_symbol_cache(data: dict[str, Any], timestamps: dict[str, float]) -> None:
    while len(data) > _MAX_SYMBOL_CACHE_ENTRIES:
        stale = min(data, key=lambda key: timestamps.get(key, 0.0))
        data.pop(stale, None)
        timestamps.pop(stale, None)


def _cached_apex(key: str, builder: Any, ttl: int = _APEX_CACHE_TTL) -> Any:
    now = _time.time()
    with _apex_cache_lock:
        cached = _apex_cache.get(key)
        if cached and now - float(cached["timestamp"]) < ttl:
            return copy.deepcopy(cached["data"])

    try:
        data = builder()
    except Exception as error:
        data = {"error": str(error), "computed_at": _time.time()}

    with _apex_cache_lock:
        _apex_cache[key] = {"timestamp": now, "data": copy.deepcopy(data)}
    return data


def _build_decision_intelligence(symbol: str) -> dict[str, Any]:
    normalized = symbol.strip().upper()
    inputs = _apex_inputs(normalized)
    prediction = get_ml_prediction(normalized)
    regime = _build_regime(normalized, inputs)
    patterns = _build_patterns(normalized, inputs)
    options_flow = _build_options_flow(normalized, inputs)
    var = _build_var(normalized, inputs)
    greeks = _build_greeks_dashboard(normalized, "all", inputs)
    drawdown = _build_drawdown(normalized, inputs)
    stress = _build_stress_test(normalized, inputs)
    correlations = _build_correlations(normalized)
    brief = _build_market_brief(normalized, inputs, prediction, regime)
    model_grid = _model_grid(prediction)
    action = prediction.get("action_recommendation", {})
    components = prediction.get("component_outputs", {})
    tft = components.get("m11_tft", {})
    prophet = components.get("m12_prophet", {})
    wavelet = components.get("m14_wavelet", {})
    bnn = components.get("m20_bnn", {})
    conformal = components.get("m17_conformal", {})
    options = inputs["signals"].get("options", {})
    return {
        "symbol": normalized,
        "spot": inputs["spot"],
        "expiry": inputs["expiry"],
        "computed_at": _time.time(),
        "apex_strip": {
            "spot": inputs["spot"],
            "regime": regime.get("state_label"),
            "regime_probability": max((regime.get("state_probabilities") or {"": 0}).values()),
            "anomaly": "ANOMALY" if (components.get("m05_anomaly") or {}).get("is_anomalous") else "NORMAL",
            "ml_direction": prediction.get("composite_direction"),
            "ml_confidence": prediction.get("composite_confidence"),
            "tft_1d": tft.get("return_p50_1d"),
            "var_per_lot": _first_var(var),
            "next_event": prophet.get("next_event_name"),
            "next_event_days": prophet.get("next_event_days"),
            "observation": _strip_observation(options, prediction),
        },
        "ml_consensus": {
            "direction": prediction.get("composite_direction"),
            "confidence": prediction.get("composite_confidence"),
            "uncertainty_level": prediction.get("uncertainty_level"),
            "model_agreement": prediction.get("model_agreement"),
            "conformal_interval_90": conformal.get("interval_90"),
            "tft": {
                "return_p50_1d": tft.get("return_p50_1d"),
                "return_p50_3d": tft.get("return_p50_3d"),
                "return_p50_5d": tft.get("return_p50_5d"),
                "return_p10_1d": tft.get("return_p10_1d"),
                "return_p90_1d": tft.get("return_p90_1d"),
            },
            "prophet_regime": prophet.get("regime"),
            "prophet_trend_strength": prophet.get("trend_strength"),
            "wavelet_dominant_frequency": wavelet.get("dominant_frequency"),
            "bayesian_uncertainty": {
                "epistemic": bnn.get("epistemic_uncertainty"),
                "aleatoric": bnn.get("aleatoric_uncertainty"),
            },
        },
        "patterns": patterns,
        "action_intelligence": action,
        "market_brief": brief,
        "risk_dashboard": {
            "var": var,
            "greeks": greeks,
            "drawdown": drawdown,
            "correlations": correlations,
            "stress_test": stress,
        },
        "options_flow": options_flow,
        "model_grid": model_grid,
        "trained_status": prediction.get("trained_status", {}),
    }


def _build_market_brief(
    symbol: str,
    inputs: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from intelligence.narrative import generate_market_brief

    normalized = symbol.strip().upper()
    inputs = inputs or _apex_inputs(normalized)
    prediction = prediction or get_ml_prediction(normalized)
    regime = regime or _build_regime(normalized, inputs)
    return generate_market_brief(normalized, inputs["signals"], prediction, regime)


def _build_patterns(symbol: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    from intelligence.patterns import detect_patterns

    inputs = inputs or _apex_inputs(symbol)
    return detect_patterns(symbol.strip().upper(), inputs["signals"], inputs["ohlcv"])


def _build_alpha(symbol: str) -> dict[str, Any]:
    from intelligence.alpha import compute_alpha_signals

    return compute_alpha_signals(symbol.strip().upper())


def _build_regime(symbol: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    from intelligence.regime_detector import detect_regime

    inputs = inputs or _apex_inputs(symbol)
    return detect_regime(symbol.strip().upper(), inputs["ohlcv"], inputs["signals"])


def _build_options_flow(symbol: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    from intelligence.options_flow import compute_options_flow

    inputs = inputs or _apex_inputs(symbol)
    return compute_options_flow(symbol.strip().upper(), inputs["legs"], inputs["spot"])


def _build_var(symbol: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    from risk.var_engine import compute_var

    inputs = inputs or _apex_inputs(symbol)
    return compute_var(
        symbol.strip().upper(),
        inputs["ohlcv"],
        spot=inputs["spot"],
        lot_size=inputs["lot_size"],
    )


def _build_greeks_dashboard(
    symbol: str,
    strategy: str,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from risk.greeks_dashboard import compute_greeks_dashboard

    inputs = inputs or _apex_inputs(symbol)
    dashboard = compute_greeks_dashboard(symbol.strip().upper(), inputs["legs"], inputs["spot"])
    if strategy and strategy.lower() != "all" and "templates" in dashboard:
        key = strategy.strip().lower()
        return {
            "symbol": dashboard.get("symbol"),
            "strategy": key,
            "template": dashboard.get("templates", {}).get(key),
            "computed_at": dashboard.get("computed_at"),
        }
    return dashboard


def _build_drawdown(symbol: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    from risk.drawdown import compute_drawdown

    inputs = inputs or _apex_inputs(symbol)
    return compute_drawdown(symbol.strip().upper(), inputs["ohlcv"])


def _build_stress_test(symbol: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    from risk.stress_test import compute_stress_tests

    inputs = inputs or _apex_inputs(symbol)
    return compute_stress_tests(
        symbol.strip().upper(),
        inputs["ohlcv"],
        spot=inputs["spot"],
        lot_size=inputs["lot_size"],
    )


def _build_correlations(symbol: str) -> dict[str, Any]:
    from risk.correlation import compute_correlations

    return compute_correlations(symbol.strip().upper())


def _apex_inputs(symbol: str) -> dict[str, Any]:
    normalized = symbol.strip().upper()
    expiry = get_next_expiries(1)[0]
    legs = get_chain_from_redis(normalized, expiry)
    if not legs:
        legs = _read_chain_legs_from_redis(normalized, [expiry])
    spot_observation = _read_spot_observation_from_redis(normalized)
    spot = float(spot_observation.price) if spot_observation.decision_eligible else None
    try:
        lot_size = instrument_master.get_lot_size(normalized, expiry)
    except InstrumentSchemaError:
        lot_size = None
    ohlcv = _load_ohlcv_frame(normalized)
    market_data_eligibility = _current_market_data_eligibility(ohlcv)
    if not market_data_eligibility.decision_eligible:
        ohlcv = ohlcv.iloc[0:0].copy()
    signals = get_signals(normalized)
    if isinstance(signals, dict):
        signals["spot"] = spot
        try:
            signals["_all_signals"] = get_all_signals(normalized)
        except Exception as error:
            print(f"narrative: /signals/all fallback unavailable: {error!r}")
    return {
        "symbol": normalized,
        "expiry": expiry,
        "legs": legs,
        "spot": spot,
        "lot_size": lot_size,
        "spot_provenance": spot_observation.as_provenance(),
        "market_data_eligibility": market_data_eligibility.to_dict(),
        "ohlcv": ohlcv,
        "signals": signals if isinstance(signals, dict) else {},
    }


def _model_grid(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    names = {
        "m01_vol_regime": "Vol Regime",
        "m02_direction": "Direction",
        "m03_strategy": "Strategy",
        "m04_iv_reversion": "IV Reversion",
        "m05_anomaly": "Anomaly",
        "m06_regime_return": "Regime Return",
        "m07_iv_forecast": "IV Forecast",
        "m08_gp_regime": "GP Regime",
        "m09_rl_sizer": "RL Sizer",
        "m10_ensemble": "Ensemble",
        "m11_tft": "TFT",
        "m12_prophet": "Prophet",
        "m13_nbeats": "N-BEATS",
        "m14_wavelet": "Wavelet",
        "m15_kalman": "Kalman",
        "m16_survival": "Survival",
        "m17_conformal": "Conformal",
        "m18_bandit": "Bandit",
        "m19_dqn": "DQN Sizer",
        "m20_bnn": "Bayesian NN",
    }
    components = prediction.get("component_outputs") or {}
    rows: list[dict[str, Any]] = []
    for model_id, name in names.items():
        output = prediction if model_id == "m10_ensemble" else components.get(model_id, {})
        rows.append({
            "model_id": model_id,
            "model": name,
            "prediction": _model_prediction_text(output),
            "confidence": output.get("confidence") or output.get("composite_confidence"),
            "accuracy_7d": None,
            "accuracy_30d": output.get("accuracy_30d"),
            "status": "Trained" if output.get("is_trained") else "Fallback",
            "reason": output.get("reason"),
        })
    return rows


def _model_prediction_text(output: dict[str, Any]) -> str | None:
    for key in ("prediction", "direction", "regime", "strategy", "iv_direction", "recommended_strategy"):
        value = output.get(key)
        if value is not None:
            return str(value)
    return None


def _first_var(var_payload: dict[str, Any]) -> float | None:
    methods = var_payload.get("methods")
    if isinstance(methods, list) and methods:
        return methods[0].get("var_95_1d")
    return None


def _strip_observation(options: dict[str, Any], prediction: dict[str, Any]) -> str:
    flip = options.get("gex_flip")
    if flip:
        return f"Vol compression near GEX flip {flip}"
    direction = prediction.get("composite_direction")
    return f"Watch model agreement before acting on {direction or 'FLAT'} bias"


@app.get("/sentiment/current")
def get_current_sentiment() -> Any:
    cached = _fresh_sentiment_cache()
    if cached is not None:
        return cached

    payload = _sentiment_cache_snapshot()
    if payload is None:
        payload = _empty_sentiment_payload()
    return payload


@app.get("/cross_asset/snapshot")
def get_cross_asset_snapshot() -> Any:
    from cross_asset.fetcher import compute_premarket_bias, fetch_cross_asset_snapshot

    global _cross_asset_cache, _cross_asset_cache_time

    now = _time.time()
    with _cross_asset_cache_lock:
        if (
            _cross_asset_cache
            and _cross_asset_cache_time > 0
            and now - _cross_asset_cache_time < _CROSS_ASSET_CACHE_TTL
        ):
            return copy.deepcopy(_cross_asset_cache)

    try:
        nifty_spot = _read_spot_from_redis("NIFTY")
        snapshot = fetch_cross_asset_snapshot(nifty_spot)
        bias = compute_premarket_bias(snapshot, nifty_spot)
        result = {
            "snapshot": snapshot,
            "bias": bias,
            "computed_at": _time.time(),
        }

        with _cross_asset_cache_lock:
            _cross_asset_cache = copy.deepcopy(result)
            _cross_asset_cache_time = float(result["computed_at"])

        return result
    except Exception as e:
        print(f"cross_asset: error: {e}")
        with _cross_asset_cache_lock:
            if _cross_asset_cache:
                cached = copy.deepcopy(_cross_asset_cache)
                cached["error"] = str(e)
                return cached
        return {"error": str(e), "snapshot": {}, "bias": {}}


@app.get("/sentiment/history")
def get_sentiment_history() -> Any:
    now = _time.time()
    since = now - 24 * 60 * 60

    with _sentiment_cache_lock:
        points = [
            (timestamp, score)
            for timestamp, score in _sentiment_score_history
            if timestamp >= since
        ]
        if not points and _sentiment_cache:
            cached_score = _sentiment_score_value(_sentiment_cache)
            if cached_score is not None:
                points = [(now, cached_score)]

    hourly: dict[datetime.datetime, list[float]] = {}
    for timestamp, score in points:
        bucket = datetime.datetime.fromtimestamp(
            timestamp,
            tz=datetime.timezone.utc,
        ).replace(minute=0, second=0, microsecond=0)
        hourly.setdefault(bucket, []).append(score)

    buckets = sorted(hourly)
    return {
        "timestamps": [bucket.isoformat() for bucket in buckets],
        "scores": [
            sum(hourly[bucket]) / len(hourly[bucket])
            for bucket in buckets
        ],
    }


@app.get("/chain/{symbol}")
def get_chain_from_nse(symbol: str) -> Any:
    symbol = http_symbol(symbol)
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

        url = (
            "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
            f"?functionName=getSymbolDerivativesData&symbol={quote(symbol)}"
        )
        r = session.get(url, headers={
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/option-chain",
        }, timeout=15)

        if r.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail=f"NSE option-chain source returned HTTP {r.status_code}",
            )

        text = r.text
        if text.strip().startswith("<"):
            raise HTTPException(status_code=503, detail="NSE option-chain session expired")

        data = r.json()

        source_rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(source_rows, list) or not source_rows:
            raise HTTPException(status_code=503, detail="NSE option-chain payload has no rows")

        observed_at = _parse_nse_chain_timestamp(data.get("timestamp"))
        if observed_at is None:
            raise HTTPException(
                status_code=503,
                detail="NSE option-chain payload has no valid source timestamp",
            )

        calendar = exchange_calendar()
        now = datetime.datetime.now(calendar.timezone)
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds < -300:
            raise HTTPException(
                status_code=503,
                detail="NSE option-chain source timestamp is implausibly in the future",
            )
        age_seconds = max(0.0, age_seconds)
        session_snapshot = calendar.session_snapshot(now)
        is_live = session_snapshot.get("current_state") == "OPEN" and age_seconds <= 180

        expected_underlying = _normalise_nse_underlying(symbol)
        spot_values: list[float] = []
        rows_by_strike: dict[tuple[float, str], dict[str, Any]] = {}
        for item in source_rows:
            if not isinstance(item, dict):
                continue
            source_underlying = _normalise_nse_underlying(item.get("underlying"))
            if source_underlying and source_underlying != expected_underlying:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "NSE option-chain underlying mismatch: "
                        f"expected {symbol}, received {item.get('underlying')}"
                    ),
                )

            row_spot = _parse_number(item.get("underlyingValue"))
            if row_spot is not None and row_spot > 0:
                spot_values.append(row_spot)

            strike = _parse_number(item.get("strikePrice"))
            expiry = str(item.get("expiryDate") or "").strip()
            opt_type = str(item.get("optionType") or "").strip().upper()
            last_price = _nonnegative_source_number(item, "lastPrice")
            if strike is None or strike <= 0 or not expiry or last_price is None:
                continue
            key = (strike, expiry)
            if key not in rows_by_strike:
                rows_by_strike[key] = {"strikePrice": strike, "expiryDate": expiry}
            leg = {
                "lastPrice": last_price,
                "bidprice": _nonnegative_source_number(item, "bidprice", "bidPrice", "bid"),
                "askPrice": _nonnegative_source_number(item, "askPrice", "askprice", "ask"),
                "openInterest": _nonnegative_source_integer(item, "openInterest"),
                "changeinOpenInterest": _source_integer(item, "changeinOpenInterest"),
                "totalTradedVolume": _nonnegative_source_integer(item, "totalTradedVolume"),
                "impliedVolatility": _positive_source_number(item, "impliedVolatility"),
            }
            if opt_type == "CE":
                rows_by_strike[key]["CE"] = leg
            elif opt_type == "PE":
                rows_by_strike[key]["PE"] = leg

        records_data = list(rows_by_strike.values())
        if not records_data:
            raise HTTPException(status_code=503, detail="NSE option-chain payload has no valid legs")
        if not spot_values:
            raise HTTPException(
                status_code=503,
                detail=f"NSE option-chain payload has no valid {symbol} underlying value",
            )
        if max(spot_values) - min(spot_values) > 0.01:
            raise HTTPException(
                status_code=503,
                detail=f"NSE option-chain payload has inconsistent {symbol} underlying values",
            )
        spot = spot_values[0]

        return {
            "records": {
                "data": records_data,
                "underlyingValue": spot,
            },
            "provenance": {
                "source": "NSE_GET_SYMBOL_DERIVATIVES_DATA",
                "symbol": symbol,
                "observed_at": observed_at.isoformat(),
                "received_at": now.isoformat(),
                "age_seconds": round(age_seconds, 3),
                "freshness": "FRESH" if is_live else "STALE",
                "decision_eligible": is_live,
                "market_state": session_snapshot.get("current_state"),
            },
        }

    except HTTPException:
        raise
    except (_requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as error:
        logger.warning("nse: chain fetch unavailable: %s", error)
        raise HTTPException(status_code=503, detail="NSE option-chain source unavailable") from error


def _parse_nse_chain_timestamp(value: Any) -> datetime.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.datetime.strptime(text, "%d-%b-%Y %H:%M:%S")
    except ValueError:
        try:
            parsed = datetime.datetime.fromisoformat(text)
        except ValueError:
            return None
    timezone = exchange_calendar().timezone
    return parsed.replace(tzinfo=timezone) if parsed.tzinfo is None else parsed.astimezone(timezone)


def _normalise_nse_underlying(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return "NIFTY" if compact in {"NIFTY", "NIFTY50"} else compact


def _source_number(record: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in record:
            return _parse_number(record.get(name))
    return None


def _nonnegative_source_number(record: dict[str, Any], *names: str) -> float | None:
    value = _source_number(record, *names)
    return value if value is not None and value >= 0 else None


def _positive_source_number(record: dict[str, Any], *names: str) -> float | None:
    value = _source_number(record, *names)
    return value if value is not None and value > 0 else None


def _source_integer(record: dict[str, Any], *names: str) -> int | None:
    value = _source_number(record, *names)
    return int(value) if value is not None and value.is_integer() else None


def _nonnegative_source_integer(record: dict[str, Any], *names: str) -> int | None:
    value = _source_integer(record, *names)
    return value if value is not None and value >= 0 else None


@app.get("/expiries/{symbol}")
def get_expiries(symbol: str) -> list[str]:
    symbol = http_symbol(symbol)
    _query_symbols(symbol)
    return get_next_expiries()


@app.get("/iv_surface/{symbol}")
def get_iv_surface(symbol: str) -> Any:
    from options.iv_surface import compute_iv_surface

    symbol = http_symbol(symbol)
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
    normalized = http_symbol(symbol)
    from_date = http_date(from_date)
    to_date = http_date(to_date)
    from backtest.spot_quality import verified_spot_sources

    from_day, to_day = _date_range(from_date, to_date)
    symbols = _query_symbols(normalized)
    sources = sorted(verified_spot_sources(normalized))

    result = resolve_read(
        ReadOperation.LOAD_OHLCV_CHART_RANGE,
        {
            "symbols": symbols,
            "sources": sources,
            "from_date": from_day.isoformat(),
            "to_date": to_day.isoformat(),
        },
    )
    rows = result["rows"]

    by_date: dict[str, dict[str, Any]] = {}
    preferred_symbol = "NIFTY" if normalized == "NIFTY" else "NIFTY 50" if normalized == "NIFTY 50" else normalized
    for row in rows:
        row_symbol = str(row[0])
        date = str(row[1])
        if date in by_date and by_date[date].get("_source_symbol") == preferred_symbol:
            continue
        if date in by_date and row_symbol != preferred_symbol:
            continue

        by_date[date] = {
            "_source_symbol": row_symbol,
            "instrument": row_symbol,
            "date": date,
            "observed_at": row[2],
            "open": float(row[3]),
            "high": float(row[4]),
            "low": float(row[5]),
            "close": float(row[6]),
            "volume": int(row[7]),
            "source": str(row[8]),
            "quality": str(row[9]),
            "derivation_type": str(row[10]),
        }

    return [
        {key: value for key, value in by_date[date].items() if key != "_source_symbol"}
        for date in sorted(by_date)
    ]


@app.post("/ingest/bhav")
def ingest_bhav(rows: list[OHLCVRow]) -> dict[str, int]:
    for row in rows:
        row.symbol = http_symbol(row.symbol)
        row.date = http_date(row.date) or row.date
    payload = [_model_dump(row) for row in rows]
    return submit_write(WriteOperation.INGEST_OHLCV, {"rows": payload})


@app.post("/download/historical")
async def download_historical(
    request: HistoricalDownloadRequest,
) -> dict[str, str]:
    request.symbol = http_symbol(request.symbol)
    request.from_date = http_date(request.from_date) or request.from_date
    request.to_date = http_date(request.to_date) or request.to_date
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


@app.get("/data/preflight")
def data_preflight() -> Any:
    free_gb = shutil.disk_usage(BASE_DIR).free / (1024**3)
    with _broker_auth_status_lock:
        angel_auth_state = _broker_auth_status["state"]
    checks: dict[str, Any] = {
        "disk_free_gb": round(free_gb, 2),
        "disk_sufficient": free_gb >= 15,
        "data_dir": str(DATA_DIR),
        "duckdb_path": str(DUCKDB_PATH),
        "duckdb_exists": DUCKDB_PATH.exists(),
        "bhavcopy_dirs_exist": EQUITY_BHAV_DIR.exists(),
        "angel_auth_state": angel_auth_state,
        "jugaad_installed": False,
        "smartapi_installed": False,
        "pyotp_installed": False,
    }

    try:
        import jugaad_data  # noqa: F401

        checks["jugaad_installed"] = True
    except ImportError:
        pass

    try:
        import SmartApi  # noqa: F401

        checks["smartapi_installed"] = True
    except ImportError:
        pass

    try:
        import pyotp  # noqa: F401

        checks["pyotp_installed"] = True
    except ImportError:
        pass

    checks["packages_installed"] = all(
        [
            checks["jugaad_installed"],
            checks["smartapi_installed"],
            checks["pyotp_installed"],
        ]
    )
    checks["ready"] = all(
        [
            checks["disk_sufficient"],
            checks["angel_auth_state"] == "CONNECTED",
            checks["packages_installed"],
        ]
    )
    return checks


@app.post("/data/test_parsers")
async def test_parsers(background_tasks: BackgroundTasks) -> Any:
    if _backfill_snapshot()["running"]:
        return {"status": "already_running"}

    background_tasks.add_task(_submit_parser_tests)
    return {
        "status": "started",
        "categories": TEST_MATRIX,
        "symbols": ALL_TEST_SYMBOLS,
        "symbol_count": len(ALL_TEST_SYMBOLS),
        "check_results_at": "/data/test_parsers/report",
        "check_status_at": "/data/backfill/status",
    }


@app.post("/data/test_5stocks")
async def test_5_stocks(background_tasks: BackgroundTasks) -> Any:
    return await test_parsers(background_tasks)


@app.get("/data/test_parsers/report")
def parser_test_report() -> Any:
    return _get_parser_report()


@app.get("/data/test_5stocks/report")
def test_report() -> Any:
    return parser_test_report()


@app.post("/data/backfill")
async def trigger_backfill(
    background_tasks: BackgroundTasks,
    request: BackfillRequest | None = None,
) -> Any:
    if _backfill_snapshot()["running"]:
        return {"status": "already_running", "check_at": "/data/backfill/status"}

    payload = request or BackfillRequest()
    allowed_scopes = {"full", "fno_only", "nifty50_only"}
    if payload.scope not in allowed_scopes:
        raise HTTPException(status_code=400, detail=f"scope must be one of {sorted(allowed_scopes)}")

    background_tasks.add_task(
        _submit_full_backfill,
        payload.scope,
        payload.years_equity,
        payload.years_fo,
        payload.months_intraday,
    )
    return {
        "status": "started",
        "scope": payload.scope,
        "estimated_hours": {
            "full": "8-12",
            "fno_only": "4-6",
            "nifty50_only": "1-2",
        }.get(payload.scope, "unknown"),
        "monitor_at": "/data/backfill/status",
        "results_at": "/data/summary",
        "errors_at": "/data/errors",
    }


@app.get("/data/backfill/status")
def backfill_status() -> Any:
    state = _backfill_snapshot()
    state["disk_free_gb"] = round(shutil.disk_usage(BASE_DIR).free / (1024**3), 2)
    if state["started_at"]:
        started_at = datetime.datetime.fromisoformat(state["started_at"])
        state["elapsed_minutes"] = int(
            (datetime.datetime.now() - started_at).total_seconds() // 60
        )
    return state


@app.get("/data/summary")
def data_summary() -> Any:
    result = resolve_read(ReadOperation.LOAD_DATA_SUMMARY_STATS, {})
    eq = result["eq"]
    fo = result["fo"]
    intra = result["intra"]
    sig = result["sig"]
    ohlc_fc = result["ohlc_fc"]
    instr = result["instr"]
    gift = result["gift"]

    db_size = DUCKDB_PATH.stat().st_size / (1024**3) if DUCKDB_PATH.exists() else 0
    return {
        "ohlcv": {
            "rows": int(eq[0] or 0),
            "symbols": int(eq[3] or 0),
            "date_range": [str(eq[1]), str(eq[2])],
        },
        "fo_eod": {
            "rows": int(fo[0] or 0),
            "symbols": int(fo[3] or 0),
            "date_range": [str(fo[1]), str(fo[2])],
        },
        "intraday_5min": {
            "rows": int(intra[0] or 0),
            "datetime_range": [str(intra[1]), str(intra[2])],
        },
        "signals_history": {"rows": int(sig[0] or 0)},
        "ohlc_forecasts": {"rows": int(ohlc_fc[0] or 0)},
        "instruments": {
            "rows": int(instr[0] or 0),
            "exchanges": int(instr[1] or 0),
        },
        "gift_nifty_eod": {
            "rows": int(gift[0] or 0),
            "date_range": [str(gift[1]), str(gift[2])],
        },
        "storage": {
            "duckdb_size_gb": round(db_size, 3),
            "disk_free_gb": round(shutil.disk_usage(BASE_DIR).free / (1024**3), 2),
        },
    }


@app.get("/data/errors")
def data_errors(limit: int = 20, unresolved_only: bool = False) -> Any:
    return {
        "errors": elog.get_recent_errors(limit, unresolved_only),
        "summary": elog.summary(),
    }


@app.post("/data/retry_failed")
async def retry_failed(background_tasks: BackgroundTasks) -> Any:
    failed_eq = elog.get_failed("equity_download")
    failed_fo = elog.get_failed("fo_download")
    background_tasks.add_task(_retry_failed_dates, failed_eq, failed_fo)
    return {
        "equity_failed_dates": len(failed_eq),
        "fo_failed_dates": len(failed_fo),
        "status": "retry_started",
    }


@app.get("/data/fo_universe")
def get_fo_universe() -> Any:
    from data.instrument_master import master

    symbols = master.get_fno_stocks()
    return {"count": len(symbols), "symbols": symbols}


async def seed_nifty_if_empty() -> None:
    from db.write_service import is_database_owned_by_another_process

    if is_database_owned_by_another_process():
        # Reproduced live (collector-first -> desktop-second topology): a
        # DIFFERENT process already holds terminal.db as writer authority,
        # checked proactively (writer-queue heartbeat), not by attempting a
        # raw read and catching the DuckDB exception it would raise. A
        # collector that's already running and maintaining the database is
        # definitionally not an empty fresh-install case this seeding exists
        # for, so it's correct to skip rather than crash backend startup.
        logger.warning("NIFTY seed-check skipped; database owned by another live process")
        return

    con = open_read_connection()
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM decision_eligible_ohlcv WHERE symbol = 'NIFTY 50'"
        ).fetchone()[0]
    finally:
        con.close()

    if count < 200:
        today = today_ist()
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
    loop.run_in_executor(None, _submit_historical_download, from_date, to_date)


def _submit_historical_download(
    from_date: datetime.date,
    to_date: datetime.date,
) -> None:
    try:
        submit_write(
            WriteOperation.DOWNLOAD_HISTORICAL,
            {"from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
            timeout=60 * 60,
        )
    except Exception as error:
        snapshot = _status_snapshot()
        logger.error("historical writer request failed: %s", error)
        _set_status(running=False, errors=int(snapshot["errors"]) + 1)


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
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                "symbol": str(row["symbol"]).strip().upper(),
                "date": _parse_date(str(row["date"])).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "source": str(row.get("source") or "MANUAL_UNVERIFIED"),
            }
        )

    if not normalized_rows:
        return 0

    frame = pd.DataFrame(normalized_rows)

    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute("BEGIN TRANSACTION")
            con.register("df_ingest_ohlcv", frame)
            con.execute(
                """
                INSERT OR IGNORE INTO ohlcv
                    (symbol, date, open, high, low, close, volume, source)
                SELECT symbol, CAST(date AS DATE), open, high, low, close,
                       CAST(volume AS BIGINT), source
                FROM df_ingest_ohlcv
                """
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

    return len(normalized_rows)


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


def _clear_temp_dir() -> None:
    if not _TEMP_DIR.exists():
        return
    for path in _TEMP_DIR.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as error:
            print(f"temp cleanup skipped {path.name}: {error}")


def _set_broker_auth_status(state: str, message: str) -> None:
    if state not in {"CONNECTED", "AUTHENTICATING", "DISCONNECTED", "ERROR"}:
        raise ValueError("invalid bounded broker authentication state")
    with _broker_auth_status_lock:
        _broker_auth_status["state"] = state
        _broker_auth_status["message"] = message


async def _bridge_angel_feed(websocket: WebSocket) -> None:
    credentials: dict[str, str] = {}
    session: dict[str, str] = {}
    headers: dict[str, str] = {}
    _set_broker_auth_status("AUTHENTICATING", "Authenticating with Angel One.")
    try:
        try:
            credentials = credential_manager.get_angel_credentials()
        except ValueError:
            _set_broker_auth_status(
                "DISCONNECTED",
                "Angel credentials unavailable. Run secure credential provisioning.",
            )
            await websocket.close(code=1008)
            return
        except Exception:
            _set_broker_auth_status("ERROR", "Secure credential store unavailable.")
            await websocket.close(code=1011)
            return

        session = _create_angel_session(credentials)
        credentials.clear()
        headers = {
            "Authorization": _bearer_angel_token(session["jwt_token"]),
            "x-api-key": session["api_key"],
            "x-client-code": session["client_id"],
            "x-feed-token": session["feed_token"],
            "x-client-type": "WEB",
        }
        session.clear()

        async with websockets.connect(
            "wss://smartapisocket.angelone.in/smart-stream",
            additional_headers=headers,
            open_timeout=15,
            close_timeout=5,
            ping_interval=20,
            # Angel's Smart Stream accepts RFC ping frames but does not
            # reliably answer them with RFC pong frames.  `websockets`
            # otherwise starts a 20-second pong deadline after its first
            # ping and closes an otherwise healthy, data-producing feed at
            # almost exactly 40 seconds.  Keep sending transport pings, as
            # the official SDK does, without treating a missing pong as a
            # dead quote stream; application data freshness is enforced by
            # the Rust ingester from exchange timestamps.
            ping_timeout=None,
            max_size=2 * 1024 * 1024,
        ) as remote:
            headers.clear()
            await websocket.accept()
            _set_broker_auth_status("CONNECTED", "Angel authentication connected.")
            local_task = asyncio.create_task(_forward_local_feed(websocket, remote))
            remote_task = asyncio.create_task(_forward_remote_feed(remote, websocket))
            done, pending = await asyncio.wait(
                {local_task, remote_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            results = await asyncio.gather(*done, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) and not isinstance(result, WebSocketDisconnect):
                    raise result
        _set_broker_auth_status("DISCONNECTED", "Angel feed disconnected.")
    except Exception as error:
        logger.warning("Angel secure feed bridge failed: %s", type(error).__name__)
        _set_broker_auth_status("ERROR", "Angel authentication or feed connection failed.")
        with suppress(Exception):
            await websocket.close(code=1011)
    finally:
        credentials.clear()
        session.clear()
        headers.clear()


async def _forward_local_feed(websocket: WebSocket, remote: Any) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        if message.get("text") is not None:
            await remote.send(message["text"])
        elif message.get("bytes") is not None:
            await remote.send(message["bytes"])


async def _forward_remote_feed(remote: Any, websocket: WebSocket) -> None:
    async for message in remote:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_text(str(message))


def _create_angel_session(credentials: dict[str, str]) -> dict[str, Any]:
    try:
        import pyotp
        from SmartApi import SmartConnect
    except ImportError as error:
        raise RuntimeError("Angel One dependencies unavailable") from error

    smart = SmartConnect(credentials["ANGEL_API_KEY"])
    totp = pyotp.TOTP(credentials["ANGEL_TOTP_SECRET"]).now()
    session = smart.generateSession(
        credentials["ANGEL_CLIENT_ID"],
        credentials["ANGEL_PASSWORD"],
        totp,
    )
    if not session or not session.get("status"):
        raise RuntimeError("Angel One rejected credentials")
    data = session.get("data") or {}
    expires_in = data.get("expiresIn") or data.get("expiry") or "unknown"
    print(f"auth: token received, expires in {expires_in}")
    jwt_token = data.get("jwtToken") or data.get("jwt")
    feed_token = data.get("feedToken") or smart.getfeedToken()
    if not jwt_token or not feed_token:
        raise RuntimeError("Angel One session missing tokens")
    return {
        "client_id": credentials["ANGEL_CLIENT_ID"],
        "api_key": credentials["ANGEL_API_KEY"],
        "jwt_token": jwt_token,
        "feed_token": feed_token,
        "expires_in": expires_in,
    }


def _bearer_angel_token(token: str) -> str:
    return token if token[:7].lower() == "bearer " else f"Bearer {token}"


def _date_range(
    from_date: str | None,
    to_date: str | None,
) -> tuple[datetime.date, datetime.date]:
    today = today_ist()
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


def _load_ohlcv_frame(symbol: str) -> pd.DataFrame:
    today = today_ist()
    from_day = today - datetime.timedelta(days=430)
    try:
        rows = get_ohlcv(symbol, from_day.isoformat(), today.isoformat())
    except Exception as error:
        print(f"signals: ohlcv load failed: {error}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(rows)
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_ohlcv_full_frame(symbol: str) -> pd.DataFrame:
    symbols = _query_symbols(symbol)
    result = resolve_read(ReadOperation.LOAD_ML_FORECAST_INPUT_FRAME, {"symbols": symbols})
    frame = pd.DataFrame(result["rows"], columns=result["columns"])

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "instrument", "date", "observed_at", "open", "high", "low", "close",
                "volume", "source", "quality", "derivation_type",
            ]
        )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _current_market_data_eligibility(
    frame: pd.DataFrame,
    requested_at: datetime.datetime | None = None,
) -> Any:
    latest = frame.iloc[-1].to_dict() if not frame.empty else None
    return evaluate_market_bar(
        latest,
        context=DecisionContext.CURRENT_DECISION,
        requested_as_of=requested_at or datetime.datetime.now(datetime.timezone.utc),
    )


def _equity_cache_get(key: str) -> Any | None:
    try:
        payload = _redis_client.get(key)
    except _redis.RedisError as error:
        print(f"equity: redis GET {key} failed: {error}")
        return None
    if payload is None:
        return None
    try:
        return json.loads(_redis_text(payload))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        print(f"equity: cache JSON parse failed for {key}: {error}")
        return None


def _equity_cache_set(key: str, payload: Any, ttl_seconds: int) -> None:
    try:
        _redis_client.setex(key, ttl_seconds, json.dumps(payload, default=str))
    except (_redis.RedisError, TypeError, ValueError) as error:
        print(f"equity: redis SETEX {key} failed: {error}")


def _fetch_equity_news(symbol: str) -> list[dict[str, Any]]:
    announcements = _fetch_nse_announcements(symbol)
    et_items = _fetch_et_news(symbol)
    return _sort_dedupe_news([*announcements, *et_items])[:30]


def _fetch_nse_announcements(symbol: str) -> list[dict[str, Any]]:
    url = (
        "https://www.nseindia.com/api/corporate-announcements"
        f"?index=equities&symbol={quote(symbol)}"
    )
    payload = _nse_json(url)
    items: list[dict[str, Any]] = []
    for record in _extract_records(payload):
        title = _text_field(
            record,
            "subject",
            "desc",
            "description",
            "announcement",
            "title",
            "sm_desc",
        )
        body = _text_field(
            record,
            "attchmntText",
            "attachmentText",
            "details",
            "body",
            "summary",
            "longDesc",
            "description",
        )
        if not title:
            title = _first_words(body, 90)
        if not title:
            continue
        date_text = _iso_datetime(
            _field(
                record,
                "an_dt",
                "sort_date",
                "date",
                "announcementDate",
                "disseminationDate",
                "exchdisstime",
                "createdAt",
                "submissionDate",
            )
        )
        pdf_url = _normalise_url(
            _text_field(
                record,
                "attchmntFile",
                "attachment",
                "attachmentUrl",
                "fileUrl",
                "pdf_url",
            )
        )
        items.append(
            _news_item(
                title=title,
                source="NSE",
                date=date_text,
                summary=_summary(body),
                url=None,
                pdf_url=pdf_url,
                body=body or None,
                item_type="announcement",
            )
        )
    return _sort_dedupe_news(items)[:20]


def _fetch_et_news(symbol: str) -> list[dict[str, Any]]:
    company_name = _company_name(symbol)
    url = "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms"
    try:
        response = _requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            print(f"equity: ET RSS returned {response.status_code}")
            return []
        root = ET.fromstring(response.content)
    except (ET.ParseError, _requests.RequestException) as error:
        print(f"equity: ET RSS fetch failed: {error}")
        return []

    items: list[dict[str, Any]] = []
    for node in root.findall(".//item"):
        title = _xml_text(node, "title")
        description = _strip_html(_xml_text(node, "description"))
        if not _matches_company(symbol, company_name, title, description):
            continue
        date_text = _iso_datetime(_xml_text(node, "pubDate"))
        items.append(
            _news_item(
                title=title,
                source="ET",
                date=date_text,
                summary=_summary(description),
                url=_xml_text(node, "link") or None,
                pdf_url=None,
                body=None,
                item_type="news",
            )
        )
    return _sort_dedupe_news(items)[:10]


def _fetch_equity_financials(symbol: str) -> dict[str, Any]:
    url = (
        "https://www.nseindia.com/api/corporates-financial-results"
        f"?index=equities&symbol={quote(symbol)}&period=Quarterly"
    )
    payload = _nse_json(url)
    quarters = [
        item
        for item in (
            _normalise_financial_record(symbol, record)
            for record in _extract_records(payload)
        )
        if item is not None
    ]
    quarters.sort(
        key=lambda item: _sort_datetime(item.get("report_date")),
        reverse=True,
    )
    quarters = quarters[:8]
    _fill_yoy_growth(quarters)
    if not quarters:
        return {"symbol": symbol, "quarters": [], "error": "no data"}
    return {
        "symbol": symbol,
        "quarters": quarters,
        "annual": [],
        "ratios": _extract_ratios(payload),
    }


def _normalise_financial_record(
    symbol: str,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    report_date = _iso_date(
        _field(
            record,
            "report_date",
            "reportDate",
            "date",
            "toDate",
            "periodEnded",
            "periodEndDate",
            "broadCastDate",
            "broadcastDate",
            "xbrlFilingDate",
        )
    )
    revenue = _number_field(
        record,
        "revenue",
        "totalRevenue",
        "revenueFromOperations",
        "revenueFromOperation",
        "totalIncome",
        "income",
        "sales",
    )
    pat = _number_field(
        record,
        "pat",
        "profitAfterTax",
        "profitLossForPeriod",
        "netProfit",
        "netProfitLoss",
        "profitAfterTaxAfterMI",
    )
    eps = _number_field(
        record,
        "eps",
        "basicEPS",
        "dilutedEPS",
        "earningsPerShare",
        "basicEps",
        "dilutedEps",
    )
    if report_date is None and revenue is None and pat is None and eps is None:
        return None
    period = _text_field(
        record,
        "period",
        "quarter",
        "quarterEnded",
        "periodEnded",
        "financialYear",
    )
    if not period:
        period = _quarter_label(report_date)
    return {
        "period": period or symbol,
        "revenue": revenue,
        "pat": pat,
        "eps": eps,
        "yoy_growth_pct": _number_field(
            record,
            "yoy_growth_pct",
            "yoyGrowth",
            "yoy",
            "growthPct",
        ),
        "report_date": report_date,
    }


def _fetch_equity_filings(symbol: str) -> dict[str, Any]:
    filings = _fetch_nse_filings(symbol)
    annual_reports = _fetch_nse_annual_reports(symbol)
    rows = _sort_dedupe_filings([*filings, *annual_reports])
    return {"symbol": symbol, "filings": rows}


def _fetch_nse_filings(symbol: str) -> list[dict[str, Any]]:
    # NSE retired the dedicated corpType=filings endpoint; the current
    # corporate-filings UI serves both the Announcements and Filings tabs
    # from this same feed (verified against nseindia.com's own bundled JS,
    # 2026-08-25 -- there is no separate filings-only API left).
    url = (
        "https://www.nseindia.com/api/corporate-announcements"
        f"?index=equities&symbol={quote(symbol)}"
    )
    payload = _nse_json(url)
    rows: list[dict[str, Any]] = []
    for record in _extract_records(payload):
        title = _text_field(
            record,
            "subject",
            "desc",
            "description",
            "filingName",
            "name",
            "title",
            "purpose",
        )
        if not title:
            continue
        pdf_url = _normalise_url(
            _text_field(record, "attchmntFile", "attachment", "fileUrl", "pdf_url", "url")
        )
        rows.append(
            {
                "title": title,
                "date": _iso_date(
                    _field(
                        record,
                        "an_dt",
                        "sort_date",
                        "date",
                        "filingDate",
                        "submissionDate",
                        "exchdisstime",
                    )
                ),
                "type": _filing_type(title),
                "pdf_url": pdf_url,
                "size_kb": _size_kb(record),
            }
        )
    return rows


def _fetch_nse_annual_reports(symbol: str) -> list[dict[str, Any]]:
    url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={quote(symbol)}"
    payload = _nse_json(url)
    rows: list[dict[str, Any]] = []
    for record in _extract_records(payload):
        year = _text_field(record, "year", "toYr", "financialYear", "fy", "period")
        title = _text_field(record, "title", "reportName", "name")
        if not title:
            title = f"Annual Report {year}".strip()
        pdf_url = _normalise_url(
            _text_field(
                record, "attchmntFile", "attachment", "fileUrl", "pdf_url", "url", "fileName"
            )
        )
        if not title and not pdf_url:
            continue
        rows.append(
            {
                "title": title,
                "date": _iso_date(
                    _field(record, "date", "reportDate", "broadcast_dttm", "toDate", "toYr", "year")
                ),
                "type": "annual_report",
                "pdf_url": pdf_url,
                "size_kb": _size_kb(record),
            }
        )
    return rows


def _nse_json(url: str) -> Any:
    response = nse.get(url)
    if response is None:
        return None
    if response.status_code != 200:
        print(f"equity: NSE returned {response.status_code} for {url}")
        return None
    text = response.text.strip()
    if text.startswith("<"):
        print(f"equity: NSE HTML response for {url}")
        return None
    try:
        return response.json()
    except ValueError as error:
        print(f"equity: NSE JSON parse failed: {error}")
        return None


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "data",
        "results",
        "result",
        "rows",
        "financialResults",
        "quarterly",
        "annualReports",
        "announcements",
        "filings",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _extract_records(value)
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value
    return []


def _field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    lookup = {_normalise_field_name(key): value for key, value in record.items()}
    for name in names:
        value = lookup.get(_normalise_field_name(name))
        if value not in (None, ""):
            return value
    return None


def _normalise_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _text_field(record: dict[str, Any], *names: str) -> str:
    value = _field(record, *names)
    if value is None:
        return ""
    return _strip_html(str(value)).strip()


def _number_field(record: dict[str, Any], *names: str) -> float | None:
    value = _field(record, *names)
    return _parse_number(value)


def _parse_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "--", "NA", "N.A.", "null"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return -abs(number) if negative else number


def _company_name(symbol: str) -> str:
    normalized = symbol.strip().upper()
    return (
        SYMBOL_TO_NAME.get(normalized)
        or SYMBOL_TO_NAME.get(f"{normalized}-EQ")
        or normalized
    )


def _matches_company(symbol: str, company_name: str, *texts: str) -> bool:
    haystack = " ".join(texts).casefold()
    terms = {symbol.casefold(), company_name.casefold()}
    for token in re.split(r"[^A-Za-z0-9&]+", company_name):
        if len(token) >= 4:
            terms.add(token.casefold())
    return any(term and term in haystack for term in terms)


def _xml_text(node: Any, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _summary(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = _strip_html(text)
    return cleaned[:200] if cleaned else None


def _first_words(text: str, limit: int) -> str:
    cleaned = _strip_html(text)
    return cleaned[:limit].strip()


def _normalise_url(value: str) -> str | None:
    url = value.strip()
    if not url:
        return None
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin("https://www.nseindia.com/", url)


def _news_item(
    *,
    title: str,
    source: str,
    date: str | None,
    summary: str | None,
    url: str | None,
    pdf_url: str | None,
    body: str | None,
    item_type: str,
) -> dict[str, Any]:
    safe_date = date or datetime.datetime.now(datetime.timezone.utc).isoformat()
    digest = hashlib.sha256(f"{title}|{safe_date}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": digest,
        "title": title,
        "source": source,
        "date": safe_date,
        "summary": summary,
        "url": url,
        "pdf_url": pdf_url,
        "body": body,
        "type": item_type,
    }


def _sort_dedupe_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        title_key = re.sub(r"\s+", " ", str(item.get("title") or "").strip().casefold())
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        unique.append(item)
    unique.sort(key=lambda item: _sort_datetime(item.get("date")), reverse=True)
    return unique


def _sort_dedupe_filings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        title = re.sub(r"\s+", " ", str(item.get("title") or "").strip().casefold())
        key = f"{title}|{item.get('pdf_url') or ''}"
        if not title or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda item: _sort_datetime(item.get("date")), reverse=True)
    return unique


def _iso_datetime(value: Any) -> str | None:
    parsed = _parse_equity_datetime(value)
    return parsed.isoformat() if parsed else None


def _iso_date(value: Any) -> str | None:
    parsed = _parse_equity_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _parse_equity_datetime(value: Any) -> datetime.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime.datetime):
        parsed = value
    elif isinstance(value, datetime.date):
        parsed = datetime.datetime.combine(value, datetime.time.min)
    elif isinstance(value, (int, float)):
        number = float(value)
        if number <= 0:
            return None
        if number > 10_000_000_000:
            number /= 1000
        parsed = datetime.datetime.fromtimestamp(number, tz=datetime.timezone.utc)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"na", "n/a", "null", "-"}:
            return None
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            parsed = None
        if parsed is None:
            iso_text = text.replace("Z", "+00:00")
            try:
                parsed = datetime.datetime.fromisoformat(iso_text)
            except ValueError:
                parsed = None
        if parsed is None:
            parsed = _parse_equity_datetime_formats(text)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        )
    return parsed


def _parse_equity_datetime_formats(text: str) -> datetime.datetime | None:
    formats = (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y %I:%M %p",
        "%d-%b-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y",
    )
    for date_format in formats:
        try:
            return datetime.datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def _sort_datetime(value: Any) -> datetime.datetime:
    return _parse_equity_datetime(value) or datetime.datetime.min.replace(
        tzinfo=datetime.timezone.utc
    )


def _quarter_label(report_date: str | None) -> str:
    if not report_date:
        return ""
    try:
        day = datetime.date.fromisoformat(report_date)
    except ValueError:
        return report_date
    quarter_by_month = {3: "Q4", 6: "Q1", 9: "Q2", 12: "Q3"}
    quarter = quarter_by_month.get(day.month)
    if not quarter:
        quarter = f"Q{((day.month - 1) // 3) + 1}"
    fy_year = day.year if day.month <= 3 else day.year + 1
    return f"{quarter} FY{str(fy_year)[-2:]}"


def _fill_yoy_growth(quarters: list[dict[str, Any]]) -> None:
    for index, quarter in enumerate(quarters):
        if quarter.get("yoy_growth_pct") is not None:
            continue
        revenue = quarter.get("revenue")
        previous = quarters[index + 4].get("revenue") if index + 4 < len(quarters) else None
        if (
            isinstance(revenue, (int, float))
            and isinstance(previous, (int, float))
            and previous != 0
        ):
            quarter["yoy_growth_pct"] = round(((revenue - previous) / abs(previous)) * 100, 2)


def _extract_ratios(payload: Any) -> dict[str, float | None]:
    source = payload if isinstance(payload, dict) else {}
    return {
        "pe": _number_field(source, "pe", "p_e", "priceEarnings", "priceToEarnings"),
        "pb": _number_field(source, "pb", "p_b", "priceBook", "priceToBook"),
        "roe": _number_field(source, "roe", "returnOnEquity"),
        "debt_to_equity": _number_field(
            source,
            "debt_to_equity",
            "debtEquity",
            "debtToEquity",
        ),
    }


def _filing_type(title: str) -> str:
    text = title.casefold()
    if "annual" in text and "report" in text:
        return "annual_report"
    if any(term in text for term in ("quarter", "financial result", "audited result", "unaudited result", "results")):
        return "quarterly_results"
    if "shareholding" in text or "share holding" in text:
        return "shareholding"
    return "other"


def _size_kb(record: dict[str, Any]) -> int | None:
    value = _parse_number(_field(record, "size_kb", "sizeKb", "fileSize", "attchmntFileSize"))
    if value is None:
        return None
    if value > 100_000:
        return int(value / 1024)
    return int(value)


def _read_atm_iv_from_redis(symbol: str, spot: float) -> float:
    if spot <= 0:
        return 0.0
    legs = _read_chain_legs_from_redis(symbol, get_next_expiries(2))
    best_iv = 0.0
    best_distance: float | None = None
    for leg in legs:
        iv = _to_float(leg.get("iv"))
        strike = _to_float(leg.get("strike"))
        if iv <= 0 or strike <= 0:
            continue
        distance = abs(strike - spot)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_iv = iv
    return best_iv


def _safe_cross_asset_snapshot() -> dict[str, Any]:
    try:
        return get_cross_asset_snapshot()
    except Exception as error:
        print(f"signals: cross_asset snapshot unavailable: {error}")
        return {}


def _safe_current_sentiment() -> dict[str, Any]:
    try:
        return get_current_sentiment()
    except Exception as error:
        print(f"signals: sentiment snapshot unavailable: {error}")
        return _empty_sentiment_payload()


def _fetch_chain_legs_with_greeks(
    symbol: str,
    expiry: str,
    spot: float,
) -> tuple[list[dict[str, Any]], float, str]:
    try:
        payload = get_chain_from_nse(symbol)
        if not isinstance(payload, dict) or payload.get("error"):
            return [], spot, expiry

        records = payload.get("records")
        if not isinstance(records, dict):
            return [], spot, expiry

        nse_spot = _to_float(records.get("underlyingValue"))
        if nse_spot > 0:
            spot = nse_spot

        legs, leg_expiry = _chain_records_to_legs(
            records.get("data", []),
            expiry,
            symbol,
        )
        if not legs or spot <= 0:
            return legs, spot, leg_expiry or expiry

        from options.greeks import compute_chain_greeks

        return compute_chain_greeks(legs, spot, leg_expiry or expiry), spot, leg_expiry or expiry
    except Exception as error:
        print(f"signals: fallback chain fetch failed: {error}")
        return [], spot, expiry


def _chain_records_to_legs(
    rows: Any,
    target_expiry: str,
    symbol: str = "NIFTY",
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(rows, list):
        return [], target_expiry

    target_norm = _normalise_expiry_text(target_expiry)
    exact_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and _normalise_expiry_text(str(row.get("expiryDate") or "")) == target_norm
    ]
    selected_rows = exact_rows
    selected_expiry = target_expiry

    if not selected_rows:
        expiry_order: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            expiry_text = str(row.get("expiryDate") or "").strip()
            if expiry_text and expiry_text not in expiry_order:
                expiry_order.append(expiry_text)
        selected_expiry = expiry_order[0] if expiry_order else target_expiry
        selected_rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("expiryDate") or "").strip() == selected_expiry
        ]

    legs: list[dict[str, Any]] = []
    for row in selected_rows:
        strike = _to_float(row.get("strikePrice"))
        if strike <= 0:
            continue
        for option_type in ("CE", "PE"):
            leg = row.get(option_type)
            if not isinstance(leg, dict):
                continue
            last_price = _nonnegative_source_number(leg, "lastPrice")
            if last_price is None:
                continue
            legs.append({
                "symbol": symbol,
                "expiry": selected_expiry,
                "strike": strike,
                "option_type": option_type,
                "ltp": last_price,
                "bid": _nonnegative_source_number(leg, "bidprice", "bidPrice", "bid"),
                "ask": _nonnegative_source_number(leg, "askPrice", "askprice", "ask"),
                "oi": _nonnegative_source_integer(leg, "openInterest"),
                "oi_change": _source_integer(leg, "changeinOpenInterest"),
                "volume": _nonnegative_source_integer(leg, "totalTradedVolume"),
            })

    return legs, selected_expiry


def _parse_date(value: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as error:
        raise HTTPException(
            status_code=400, detail=f"invalid date: {value}"
        ) from error


def _query_symbols(symbol: str) -> list[str]:
    try:
        normalized = validate_symbol(symbol)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    # Single source of truth for index long-form/short-form aliases (see
    # data/index_aliases.py). This used to hand-duplicate its own partial
    # candidate table covering only NIFTY <-> "NIFTY 50" -- BANKNIFTY,
    # FINNIFTY and MIDCPNIFTY had no equivalent, so /ml/ohlc_forecast/
    # BANKNIFTY and the other callers of this function (equity/OHLCV chart
    # range, forecast input frame) silently saw 0 rows for those symbols the
    # same way NIFTY did before that specific fix.
    return index_symbol_candidates(normalized) or [normalized]


def _redis_symbol(symbol: str) -> str:
    try:
        normalized = validate_symbol(symbol)
    except ValueError:
        normalized = "NIFTY"
    if normalized in {"NIFTY", "NIFTY 50"}:
        return "NIFTY"
    return normalized


def _live_redis_symbol(symbol: str) -> str:
    try:
        normalized = validate_symbol(symbol)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    # Reverse of data.index_aliases.INDEX_CANONICAL_NAME (the one canonical
    # source for index name<->symbol mappings) -- maps a real instrument
    # master display name like "NIFTY BANK" back to the short symbol
    # ("BANKNIFTY") the live feed's Redis keys are keyed by.
    aliases = {name: symbol for symbol, name in INDEX_CANONICAL_NAME.items()}
    return aliases.get(normalized, normalized)


def _decode_live_tick(payload: Any, symbol: str, key: str) -> dict[str, Any] | None:
    try:
        data = json.loads(_redis_text(payload))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        print(f"live: tick JSON parse failed for {key}: {error}")
        return None

    if not isinstance(data, dict):
        return None
    try:
        quote = parse_quote_envelope(data)
        if quote.symbol != symbol.strip().upper():
            raise ValueError(f"symbol mismatch: expected {symbol}, received {quote.symbol}")
        phase = _live_feed_phase()
        quote = evaluate_market_state(
            quote,
            expected_feed_live=phase == "LIVE",
            expected_feed_starting=phase in {
                "STARTING_BACKEND", "WAITING_FOR_HEALTH", "AUTH_READY", "CONNECTING_FEED",
            },
        )
        return quote.to_dict()
    except (TypeError, ValueError) as error:
        print(f"live: invalid quote contract for {key}: {error}")
        return None


def _live_feed_phase() -> str:
    try:
        from live import multi_feed

        return str(multi_feed.get_status().get("state") or "DISCONNECTED")
    except Exception:
        return "DISCONNECTED"


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


def get_chain_from_redis(symbol: str, expiry: str) -> list[dict[str, Any]]:
    key = f"chain:{_redis_symbol(symbol)}:{expiry}"
    try:
        payload = _redis_client.get(key)
    except _redis.RedisError as error:
        print(f"signals: redis GET {key} failed: {error}")
        return []

    if payload is None:
        return []

    return _decode_chain_payload(payload, key)


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


def _read_spot_from_redis(symbol: str) -> float:
    observation = _read_spot_observation_from_redis(symbol)
    if not observation.decision_eligible or observation.price is None:
        return 0.0
    return float(observation.price)


def _read_spot_observation_from_redis(symbol: str) -> Any:
    from backtest.spot_quality import classify_live_spot, unavailable_spot

    requested_at = datetime.datetime.now(datetime.timezone.utc)
    key = f"tick:{_redis_symbol(symbol)}"
    try:
        payload = _redis_client.get(key)
    except _redis.RedisError as error:
        print(f"signals: redis GET {key} failed: {error}")
        return unavailable_spot(symbol, requested_at, "authenticated_redis_unavailable")

    if payload is None:
        return unavailable_spot(symbol, requested_at, "live_spot_missing_from_redis")

    tick = _decode_live_tick(payload, _redis_symbol(symbol), key)
    if tick is None:
        return unavailable_spot(symbol, requested_at, "live_spot_envelope_invalid")
    return classify_live_spot(
        tick,
        symbol=_redis_symbol(symbol),
        requested_at=requested_at,
    )


def _to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _get_cached_market_signals() -> dict[str, Any]:
    now = _time.time()
    with _market_signals_lock:
        cached_at = float(_market_signals_cache["timestamp"])
        if cached_at > 0 and now - cached_at < 60:
            return _market_cache_snapshot()

    from signals.market_signals import fetch_fii_dii, fetch_india_vix

    data = {
        "india_vix": fetch_india_vix(),
        "fii_dii": fetch_fii_dii(),
    }

    with _market_signals_lock:
        _market_signals_cache["timestamp"] = now
        _market_signals_cache["data"] = data
        return _market_cache_snapshot()


def _market_cache_snapshot() -> dict[str, Any]:
    data = _market_signals_cache.get("data", {})
    if not isinstance(data, dict):
        return {"india_vix": None, "fii_dii": {}}

    fii_dii = data.get("fii_dii")
    if not isinstance(fii_dii, dict):
        fii_dii = {}

    return {
        "india_vix": data.get("india_vix"),
        "fii_dii": dict(fii_dii),
    }


def _fresh_sentiment_cache() -> dict[str, Any] | None:
    now = _time.time()
    with _sentiment_cache_lock:
        if (
            _sentiment_cache
            and _sentiment_cache_time > 0
            and now - _sentiment_cache_time < _SENTIMENT_CACHE_TTL
        ):
            return copy.deepcopy(_sentiment_cache)
    return None


def _sentiment_cache_snapshot() -> dict[str, Any] | None:
    with _sentiment_cache_lock:
        if not _sentiment_cache:
            return None
        return copy.deepcopy(_sentiment_cache)


def _has_sentiment_cache() -> bool:
    with _sentiment_cache_lock:
        return bool(_sentiment_cache)


def _start_sentiment_refresh() -> Thread | None:
    global _sentiment_fetch_in_progress

    with _sentiment_cache_lock:
        if _sentiment_fetch_in_progress:
            return None
        _sentiment_fetch_in_progress = True

    thread = Thread(target=_refresh_sentiment_cache, daemon=True)
    thread.start()
    return thread


def _refresh_sentiment_cache() -> None:
    global _sentiment_cache, _sentiment_cache_time, _sentiment_fetch_in_progress

    try:
        payload, articles = _build_sentiment_payload()
        with _sentiment_cache_lock:
            _sentiment_cache = payload
            _sentiment_cache_time = float(payload["computed_at"])
            score_value = _sentiment_score_value(payload)
            if score_value is not None:
                # Only record a real observed score. A window with zero
                # classified articles must leave a gap, not a fabricated
                # point, so the history series can't be mistaken for signal.
                _sentiment_score_history.append((_sentiment_cache_time, score_value))
            for article in articles:
                _sentiment_article_history.append(article)
    except Exception as error:
        print(f"sentiment: refresh failed: {error}")
        with _sentiment_cache_lock:
            if not _sentiment_cache:
                _sentiment_cache = _empty_sentiment_payload()
                _sentiment_cache_time = float(_sentiment_cache["computed_at"])
    finally:
        with _sentiment_cache_lock:
            _sentiment_fetch_in_progress = False


def _build_sentiment_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sentiment.classifier import classify_headline, compute_sentiment_score
    from sentiment.rss_fetcher import fetch_all_feeds, fetch_nse_announcements

    articles = fetch_all_feeds(max_age_hours=24)
    classified_articles: list[dict[str, Any]] = []
    for article in articles:
        classification = classify_headline(
            str(article.get("title") or ""),
            str(article.get("summary") or ""),
            str(article.get("source") or ""),
        )
        classified_articles.append({**article, **classification})

    score = compute_sentiment_score(classified_articles, window_hours=4)
    headlines = [
        {
            "source": article["source"],
            "title": article["title"],
            "sentiment": article["sentiment"],
            "confidence": article["confidence"],
            "impact": article["impact"],
            "age_minutes": article["age_minutes"],
            "url": article["url"],
            "published_at": article["published_at"],
        }
        for article in classified_articles[:20]
    ]

    return (
        {
            "score": score,
            "headlines": headlines,
            "announcements": fetch_nse_announcements(),
            "computed_at": _time.time(),
        },
        classified_articles,
    )


def _sentiment_score_value(payload: dict[str, Any] | None) -> float | None:
    """Extract the numeric sentiment score, or None if genuinely unavailable.

    `payload["score"]` may itself be None (nothing computed yet) or a dict
    whose own "score" field is None (articles present but none classified
    inside the scoring window) -- both are "no data", never a number.
    """
    if not payload:
        return None
    score = payload.get("score")
    if not isinstance(score, dict):
        return None
    value = score.get("score")
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _empty_sentiment_payload() -> dict[str, Any]:
    # No fabricated neutral score: unavailable sentiment must surface as an
    # explicit null, not a fake 50.0 that downstream signals/UI would treat
    # as a real (if boring) market reading.
    return {
        "score": None,
        "headlines": [],
        "announcements": [],
        "computed_at": _time.time(),
    }


def _normalise_expiry_text(value: str) -> str:
    return value.strip().lower()


def get_next_expiries(n: int = 4) -> list[str]:
    return next_expiry_labels(n)


def _backfill_snapshot() -> dict[str, Any]:
    with _backfill_lock:
        return copy.deepcopy(_backfill_state)


def _update_backfill_state(**updates: Any) -> None:
    with _backfill_lock:
        _backfill_state.update(updates)


def _append_backfill_error(message: str) -> None:
    with _backfill_lock:
        errors = _backfill_state.setdefault("errors", [])
        if isinstance(errors, list):
            errors.append(message)


def _new_parser_report(run_status: str = "idle") -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category, symbols in TEST_MATRIX.items():
        categories[category] = {
            "status": "FAIL",
            "symbols": {
                symbol: _new_symbol_report(symbol, category)
                for symbol in symbols
            },
        }

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_status": run_status,
        "overall_status": "FAIL",
        "summary": {
            "categories_tested": len(TEST_MATRIX),
            "categories_passed": 0,
            "symbols_tested": len(ALL_TEST_SYMBOLS),
            "symbols_passed": 0,
            "equity_rows_total": 0,
            "fo_rows_total": 0,
            "parser_errors": 0,
            "validation_mismatches": 0,
        },
        "categories": categories,
        "parser_coverage": {
            "equity_formats_tested": [],
            "fo_formats_tested": [],
            "instrument_types_tested": [],
            "notes": "pre_2023 format only visible in older archived files",
        },
        "recommendation": "TEST NOT RUN. Trigger POST /data/test_parsers.",
    }


def _new_symbol_report(symbol: str, category: str) -> dict[str, Any]:
    if symbol in INDEX_DERIVATIVE_SYMBOLS:
        equity = {
            "status": "skipped",
            "reason": "Index not in equity Bhavcopy",
        }
    else:
        equity = {
            "status": "failed",
            "source": "unavailable",
            "rows_downloaded": 0,
            "rows_loaded": 0,
            "parse_error": None,
        }

    return {
        "equity_bhavcopy": equity,
        "fo_bhavcopy": {
            "status": "failed",
            "source": "unavailable",
            "rows_downloaded": 0,
            "rows_loaded": 0,
            "instrument_types_found": [],
            "parse_error": None,
        },
        "angel_eod": {
            "status": "failed",
            "rows_returned": 0,
            "error": "not run",
        },
        "validation": {
            "status": "skipped",
            "max_diff_pct": None,
            "mismatches": 0,
        },
        "overall": "FAIL",
    }


def _set_parser_report(report: dict[str, Any]) -> None:
    with _parser_test_report_lock:
        _parser_test_report.clear()
        _parser_test_report.update(copy.deepcopy(report))


def _get_parser_report() -> dict[str, Any]:
    with _parser_test_report_lock:
        if _parser_test_report:
            return copy.deepcopy(_parser_test_report)
    return _new_parser_report()


def _submit_parser_tests() -> None:
    try:
        submit_write(WriteOperation.RUN_PARSER_TESTS, {}, timeout=60 * 60)
    except Exception as error:
        _update_backfill_state(running=False, stage=f"failed: {error}")


def _run_test_parsers() -> None:
    parser_report = _new_parser_report(run_status="running")
    _set_parser_report(parser_report)
    _update_backfill_state(
        running=True,
        stage="test_parsers",
        current_symbol=None,
        current_date=None,
        equity_downloaded=0,
        equity_total=0,
        fo_downloaded=0,
        fo_total=0,
        equity_loaded=0,
        fo_loaded=0,
        intraday_rows=0,
        errors=[],
        started_at=datetime.datetime.now().isoformat(),
        elapsed_minutes=0,
    )

    try:
        from data.instrument_master import master
        from data.nse_bhavcopy import NSEBhavcopy

        _update_backfill_state(stage="instrument_master")
        try:
            rows = master.refresh()
            print(f"Instrument master: {rows} rows")
        except Exception as error:
            message = f"instrument_master: {error}"
            print(f"Instrument master failed: {error}")
            _append_backfill_error(message)
            elog.log("backfill_test", "instrument_master_failed", str(error))

        bhav = NSEBhavcopy()
        to_date = today_ist()
        from_date = to_date - datetime.timedelta(days=30)
        total_days = (to_date - from_date).days

        _update_backfill_state(stage="equity_download_parser_test", equity_total=total_days)
        eq_result = _download_bhav_range_with_sources(
            bhav,
            "equity",
            from_date,
            to_date,
            on_progress=lambda day, result: _update_backfill_state(
                current_date=str(day),
                equity_downloaded=result["downloaded"] + result["skipped"],
            ),
        )
        print(f"Equity download: {eq_result}")

        _update_backfill_state(stage="fo_download_parser_test", fo_total=total_days)
        fo_result = _download_bhav_range_with_sources(
            bhav,
            "fo",
            from_date,
            to_date,
            on_progress=lambda day, result: _update_backfill_state(
                current_date=str(day),
                fo_downloaded=result["downloaded"] + result["skipped"],
            ),
        )
        print(f"F&O download: {fo_result}")

        stock_symbols = [
            symbol for symbol in ALL_TEST_SYMBOLS
            if symbol not in INDEX_DERIVATIVE_SYMBOLS
        ]

        _update_backfill_state(stage="equity_load_parser_test")
        eq_loaded = bhav.load_equity_to_duckdb(symbols=stock_symbols, from_date=from_date)
        _update_backfill_state(equity_loaded=eq_loaded)
        print(f"Equity loaded: {eq_loaded} rows")

        _update_backfill_state(stage="fo_load_parser_test")
        fo_loaded = bhav.load_fo_to_duckdb(symbols=ALL_TEST_SYMBOLS, from_date=from_date)
        _update_backfill_state(fo_loaded=fo_loaded)
        print(f"F&O loaded: {fo_loaded} rows")

        fetcher = None
        try:
            from data.angel_fetcher import AngelFetcher

            fetcher = AngelFetcher()
        except Exception as error:
            message = f"angel_eod: {error}"
            print(f"Angel EOD init failed: {error}")
            _append_backfill_error(message)
            elog.log("parser_test", "angel_init_failed", str(error))

        parser_report["parser_coverage"] = bhav.parser_coverage(from_date)
        _update_backfill_state(stage="symbol_parser_validation")
        for category, symbols in TEST_MATRIX.items():
            for symbol in symbols:
                _update_backfill_state(current_symbol=symbol)
                parser_report["categories"][category]["symbols"][symbol] = (
                    _build_parser_symbol_report(
                        bhav,
                        symbol,
                        category,
                        from_date,
                        to_date,
                        eq_result,
                        fo_result,
                        fetcher,
                    )
                )
            parser_report["categories"][category]["status"] = _category_parser_status(
                category,
                parser_report["categories"][category]["symbols"],
            )
            _refresh_parser_report_summary(parser_report)
            _set_parser_report(parser_report)

        _refresh_parser_report_summary(parser_report)
        parser_report["run_status"] = "complete"
        parser_report["timestamp"] = datetime.datetime.now().isoformat()
        _set_parser_report(parser_report)
        _update_backfill_state(stage="complete")
        print("Parser test complete. Check /data/test_parsers/report")
    except Exception as error:
        _append_backfill_error(str(error))
        parser_report["run_status"] = "failed"
        parser_report["overall_status"] = "FAIL"
        parser_report["recommendation"] = (
            f"TEST FAILED. Parser test crashed: {error}"
        )
        _set_parser_report(parser_report)
        _update_backfill_state(stage=f"error: {error}")
        elog.log("parser_test", "run_failed", str(error))
    finally:
        _update_backfill_state(running=False)


def _run_test_5stocks() -> None:
    _run_test_parsers()


def _download_bhav_range_with_sources(
    bhav: Any,
    kind: str,
    from_date: datetime.date,
    to_date: datetime.date,
    on_progress=None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "holiday": 0,
        "sources": [],
    }
    current = from_date
    consecutive_failures = 0

    while current <= to_date:
        if not is_trading_day_candidate(current):
            current += datetime.timedelta(days=1)
            continue

        if kind == "equity":
            success, source = bhav.download_equity_day(current)
        else:
            success, source = bhav.download_fo_day(current)
            if source in ("unified_equity_file", "use_unified"):
                success = True

        result["sources"].append(source)
        if source == "holiday":
            result["holiday"] += 1
        elif source in {"cached", "pre_2024_skip"}:
            result["skipped"] += 1
        elif success:
            result["downloaded"] += 1
            consecutive_failures = 0
        else:
            result["failed"] += 1
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("parser_test: 5 consecutive failures, pausing 5 minutes")
                _time.sleep(300)
                consecutive_failures = 0

        if on_progress:
            on_progress(current, dict(result))

        current += datetime.timedelta(days=1)
        _time.sleep(0.5)

    result["sources"] = sorted(set(result["sources"]))
    return result


def _build_parser_symbol_report(
    bhav: Any,
    symbol: str,
    category: str,
    from_date: datetime.date,
    to_date: datetime.date,
    eq_download: dict[str, Any],
    fo_download: dict[str, Any],
    fetcher: Any | None,
) -> dict[str, Any]:
    is_index = symbol in INDEX_DERIVATIVE_SYMBOLS
    report = _new_symbol_report(symbol, category)

    if not is_index:
        eq_diag = bhav.inspect_equity_files(symbol, from_date)
        eq_rows_loaded = _count_ohlcv_rows(symbol, from_date, to_date)
        eq_parse_errors = eq_diag.get("parse_errors", [])
        _log_parse_errors("equity", symbol, eq_parse_errors)
        report["equity_bhavcopy"] = {
            "status": "ok" if eq_rows_loaded > 0 and eq_diag["rows"] > 0 else "failed",
            "source": _download_source(eq_download),
            "rows_downloaded": int(eq_diag["rows"]),
            "rows_loaded": eq_rows_loaded,
            "parse_error": _parse_error_text(eq_parse_errors),
            "parse_error_count": len(eq_parse_errors),
        }

    fo_diag = bhav.inspect_fo_files(symbol, from_date)
    fo_rows_loaded = _count_fo_rows(symbol, from_date, to_date)
    fo_parse_errors = fo_diag.get("parse_errors", [])
    _log_parse_errors("fo", symbol, fo_parse_errors)
    report["fo_bhavcopy"] = {
        "status": (
            "ok"
            if fo_rows_loaded > 0
            and fo_diag["rows"] > 0
            and fo_diag["instrument_types"]
            else "failed"
        ),
        "source": _download_source(fo_download),
        "rows_downloaded": int(fo_diag["rows"]),
        "rows_loaded": fo_rows_loaded,
        "instrument_types_found": fo_diag["instrument_types"],
        "parse_error": _parse_error_text(fo_parse_errors),
        "parse_error_count": len(fo_parse_errors),
    }

    angel_status, angel_df = _fetch_angel_eod(symbol, fetcher)
    report["angel_eod"] = angel_status
    if is_index:
        report["validation"] = {
            "status": "skipped",
            "max_diff_pct": None,
            "mismatches": 0,
            "reason": "Index derivative parser test skips equity close validation",
        }
    elif angel_df is None or angel_df.empty:
        report["validation"] = {
            "status": "skipped",
            "max_diff_pct": None,
            "mismatches": 0,
            "reason": "Angel EOD unavailable",
        }
    else:
        validation = bhav.validate_vs_angel(symbol, angel_df, n_days=5)
        report["validation"] = {
            "status": validation.get("recommendation", "skipped"),
            "max_diff_pct": validation.get("max_diff_pct"),
            "mismatches": validation.get("mismatches", 0),
        }

    report["overall"] = _symbol_parser_overall(symbol, report)
    return report


def _fetch_angel_eod(
    symbol: str,
    fetcher: Any | None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    if fetcher is None:
        return {
            "status": "failed",
            "rows_returned": 0,
            "error": "Angel fetcher unavailable",
        }, None

    try:
        start_dt = datetime.datetime.now() - datetime.timedelta(days=5)
        end_dt = datetime.datetime.now()
        if symbol in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
            instrument = instrument_master.resolve_index_feed(symbol)
            frame = fetcher.fetch(
                instrument["exchange"],
                instrument["token"],
                "ONE_DAY",
                start_dt,
                end_dt,
            )
        else:
            frame = fetcher.fetch_by_symbol(symbol, "ONE_DAY", start_dt, end_dt)
        return {
            "status": "ok" if frame is not None and not frame.empty else "failed",
            "rows_returned": 0 if frame is None else int(len(frame)),
            "error": None if frame is not None and not frame.empty else "no rows returned",
        }, frame
    except Exception as error:
        elog.log("parser_test", "angel_eod_failed", str(error), symbol=symbol)
        return {
            "status": "failed",
            "rows_returned": 0,
            "error": str(error),
        }, None


def _count_ohlcv_rows(
    symbol: str,
    from_date: datetime.date,
    to_date: datetime.date,
) -> int:
    # Uses the raw open_read_connection() fallback directly rather than
    # resolve_read(): safe because this is only ever reached from
    # _build_parser_symbol_report(), only ever called from _run_test_parsers(),
    # which only ever runs dispatched through
    # submit_write(WriteOperation.RUN_PARSER_TESTS, ...) -- i.e. always on
    # whichever process's writer thread actually owns the connection, with
    # _ACTIVE_WRITER_CONNECTION already set, so open_read_connection()'s own
    # first branch (not its unsafe raw-open fallback) is what always fires
    # here. Verify this invariant before reusing this pattern elsewhere.
    con = open_read_connection()
    try:
        row = con.execute(
            """
            SELECT COUNT(*) FROM ohlcv
            WHERE symbol = ?
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            """,
            [symbol, from_date.isoformat(), to_date.isoformat()],
        ).fetchone()
        return int(row[0] or 0)
    finally:
        con.close()


def _count_fo_rows(
    symbol: str,
    from_date: datetime.date,
    to_date: datetime.date,
) -> int:
    # Same invariant as _count_ohlcv_rows() above: only ever reached inside
    # the RUN_PARSER_TESTS write dispatch, on the owning writer's own thread.
    con = open_read_connection()
    try:
        row = con.execute(
            """
            SELECT COUNT(*) FROM fo_eod
            WHERE symbol = ?
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            """,
            [symbol, from_date.isoformat(), to_date.isoformat()],
        ).fetchone()
        return int(row[0] or 0)
    finally:
        con.close()


def _download_source(download_result: dict[str, Any]) -> str:
    sources = set(download_result.get("sources", []))
    if "unified_equity_file" in sources:
        return "unified_equity_file"
    if "use_unified" in sources:
        return "use_unified"
    if "jugaad" in sources:
        return "jugaad"
    if "nse_direct" in sources:
        return "nse_direct"
    if "cached" in sources:
        return "nse_direct"
    return "unavailable"


def _parse_error_text(errors: list[str]) -> str | None:
    if not errors:
        return None
    return "; ".join(errors)[:1000]


def _log_parse_errors(kind: str, symbol: str, errors: list[str]) -> None:
    for error in errors:
        elog.log(
            "parser_test",
            f"{kind}_parse_error",
            error,
            symbol=symbol,
        )


def _symbol_parser_overall(symbol: str, report: dict[str, Any]) -> str:
    fo_ok = report["fo_bhavcopy"]["status"] == "ok"
    if symbol in INDEX_DERIVATIVE_SYMBOLS:
        return "PASS" if fo_ok else "FAIL"
    equity_ok = report["equity_bhavcopy"]["status"] == "ok"
    return "PASS" if equity_ok and fo_ok else "FAIL"


def _category_parser_status(
    category: str,
    symbols: dict[str, Any],
) -> str:
    if category == "index_derivatives":
        nifty = symbols.get("NIFTY", {})
        banknifty = symbols.get("BANKNIFTY", {})
        nifty_types = set(nifty.get("fo_bhavcopy", {}).get("instrument_types_found", []))
        return (
            "PASS"
            if nifty.get("fo_bhavcopy", {}).get("status") == "ok"
            and "OPTIDX" in nifty_types
            and banknifty.get("fo_bhavcopy", {}).get("status") == "ok"
            else "FAIL"
        )

    equity_any = any(
        item.get("equity_bhavcopy", {}).get("status") == "ok"
        for item in symbols.values()
    )
    fo_any = any(
        item.get("fo_bhavcopy", {}).get("status") == "ok"
        and item.get("fo_bhavcopy", {}).get("instrument_types_found")
        for item in symbols.values()
    )
    return "PASS" if equity_any and fo_any else "FAIL"


def _refresh_parser_report_summary(report: dict[str, Any]) -> None:
    categories = report["categories"]
    categories_passed = sum(
        1 for category in categories.values()
        if category.get("status") == "PASS"
    )
    symbols_passed = 0
    equity_rows_total = 0
    fo_rows_total = 0
    parser_errors = 0
    validation_mismatches = 0
    instrument_types: set[str] = set(report["parser_coverage"].get("instrument_types_tested", []))

    for category in categories.values():
        for symbol_report in category["symbols"].values():
            if symbol_report.get("overall") == "PASS":
                symbols_passed += 1
            equity = symbol_report.get("equity_bhavcopy", {})
            fo = symbol_report.get("fo_bhavcopy", {})
            equity_rows_total += int(equity.get("rows_loaded", 0) or 0)
            fo_rows_total += int(fo.get("rows_loaded", 0) or 0)
            parser_errors += int(equity.get("parse_error_count", 0) or 0)
            parser_errors += int(fo.get("parse_error_count", 0) or 0)
            instrument_types.update(fo.get("instrument_types_found", []))
            validation = symbol_report.get("validation", {})
            validation_mismatches += int(validation.get("mismatches", 0) or 0)

    report["parser_coverage"]["instrument_types_tested"] = sorted(instrument_types)
    report["summary"] = {
        "categories_tested": len(TEST_MATRIX),
        "categories_passed": categories_passed,
        "symbols_tested": len(ALL_TEST_SYMBOLS),
        "symbols_passed": symbols_passed,
        "equity_rows_total": equity_rows_total,
        "fo_rows_total": fo_rows_total,
        "parser_errors": parser_errors,
        "validation_mismatches": validation_mismatches,
    }

    required_types = {"OPTSTK", "FUTSTK", "OPTIDX", "FUTIDX"}
    index_failed = categories["index_derivatives"].get("status") != "PASS"
    if (
        categories_passed == len(TEST_MATRIX)
        and parser_errors == 0
        and required_types.issubset(instrument_types)
    ):
        report["overall_status"] = "PASS"
        report["recommendation"] = (
            "TEST PASSED. All parsers working. Safe to trigger full backfill."
        )
    elif categories_passed <= 6 or index_failed or parser_errors > 5:
        report["overall_status"] = "FAIL"
        report["recommendation"] = (
            "TEST FAILED. Critical parsers broken. Do not run full backfill."
        )
    else:
        report["overall_status"] = "PARTIAL"
        failed_count = len(TEST_MATRIX) - categories_passed
        report["recommendation"] = (
            f"PARTIAL PASS. {failed_count} categories failed. Check category details."
        )


def _submit_full_backfill(
    scope: str,
    years_equity: int,
    years_fo: int,
    months_intraday: int,
) -> None:
    try:
        submit_write(
            WriteOperation.RUN_FULL_BACKFILL,
            {
                "scope": scope,
                "years_equity": years_equity,
                "years_fo": years_fo,
                "months_intraday": months_intraday,
            },
            timeout=24 * 60 * 60,
        )
    except Exception as error:
        _append_backfill_error(str(error))
        _update_backfill_state(running=False, stage=f"failed: {error}")


def _run_full_backfill(
    scope: str,
    years_equity: int,
    years_fo: int,
    months_intraday: int,
) -> None:
    nifty50 = [
        "ADANIENT",
        "ADANIPORTS",
        "APOLLOHOSP",
        "ASIANPAINT",
        "AXISBANK",
        "BAJAJ-AUTO",
        "BAJFINANCE",
        "BAJAJFINSV",
        "BPCL",
        "BHARTIARTL",
        "BRITANNIA",
        "CIPLA",
        "COALINDIA",
        "DIVISLAB",
        "DRREDDY",
        "EICHERMOT",
        "GRASIM",
        "HCLTECH",
        "HDFCBANK",
        "HDFCLIFE",
        "HEROMOTOCO",
        "HINDALCO",
        "HINDUNILVR",
        "ICICIBANK",
        "ITC",
        "INDUSINDBK",
        "INFY",
        "JSWSTEEL",
        "KOTAKBANK",
        "LTIM",
        "LT",
        "M&M",
        "MARUTI",
        "NESTLEIND",
        "NTPC",
        "ONGC",
        "POWERGRID",
        "RELIANCE",
        "SBILIFE",
        "SBIN",
        "SUNPHARMA",
        "TCS",
        "TATACONSUM",
        "TATAMOTORS",
        "TATASTEEL",
        "TECHM",
        "TITAN",
        "TRENT",
        "ULTRACEMCO",
        "WIPRO",
    ]

    _update_backfill_state(
        running=True,
        stage="starting",
        current_symbol=None,
        current_date=None,
        equity_downloaded=0,
        equity_total=0,
        fo_downloaded=0,
        fo_total=0,
        equity_loaded=0,
        fo_loaded=0,
        intraday_rows=0,
        errors=[],
        started_at=datetime.datetime.now().isoformat(),
        elapsed_minutes=0,
    )

    to_date = today_ist()
    bhavcopy_cutoff = datetime.date(2024, 1, 1)
    eq_from = max(
        to_date - datetime.timedelta(days=365 * years_equity),
        bhavcopy_cutoff,
    )
    fo_from = max(
        to_date - datetime.timedelta(days=365 * years_fo),
        bhavcopy_cutoff,
    )

    try:
        from data.instrument_master import master
        from data.nse_bhavcopy import NSEBhavcopy

        bhav = NSEBhavcopy()
        _update_backfill_state(stage="instrument_master")
        try:
            master.refresh()
        except Exception as error:
            _append_backfill_error(f"instrument_master: {error}")
            elog.log("backfill", "instrument_master_failed", str(error))

        if scope == "nifty50_only":
            load_symbols: list[str] | str = nifty50
        elif scope == "fno_only":
            load_symbols = master.get_fno_stocks()
        else:
            load_symbols = "ALL"

        _update_backfill_state(
            stage="fo_bhavcopy_download",
            fo_total=(to_date - fo_from).days,
        )
        bhav.download_fo_range(
            fo_from,
            to_date,
            on_progress=lambda day, result: _update_backfill_state(
                current_date=str(day),
                fo_downloaded=result["downloaded"] + result["skipped"],
            ),
        )

        _update_backfill_state(
            stage="equity_bhavcopy_download",
            equity_total=(to_date - eq_from).days,
        )
        bhav.download_equity_range(
            eq_from,
            to_date,
            on_progress=lambda day, result: _update_backfill_state(
                current_date=str(day),
                equity_downloaded=result["downloaded"] + result["skipped"],
            ),
        )

        _update_backfill_state(stage="fo_load_duckdb")
        fo_loaded = bhav.load_fo_to_duckdb(symbols=load_symbols, from_date=fo_from)
        _update_backfill_state(fo_loaded=fo_loaded)

        _update_backfill_state(stage="equity_load_duckdb")
        eq_loaded = bhav.load_equity_to_duckdb(symbols=load_symbols, from_date=eq_from)
        _update_backfill_state(equity_loaded=eq_loaded)

        _update_backfill_state(stage="intraday_backfill")
        try:
            from data.angel_fetcher import AngelFetcher

            fetcher = AngelFetcher()
            instrument = instrument_master.resolve_index_feed("NIFTY")
            frame = fetcher.fetch(
                instrument["exchange"],
                instrument["token"],
                "FIVE_MINUTE",
                datetime.datetime.now() - datetime.timedelta(days=30 * months_intraday),
                datetime.datetime.now(),
            )
            if not frame.empty:
                _save_intraday_to_duckdb("NIFTY", frame)
                _update_backfill_state(intraday_rows=len(frame))
        except Exception as error:
            _append_backfill_error(f"intraday: {error}")
            elog.log("backfill", "intraday_failed", str(error))

        _update_backfill_state(stage="complete")
    except Exception as error:
        _append_backfill_error(str(error))
        _update_backfill_state(stage=f"failed: {error}")
        elog.log("backfill", "run_failed", str(error))
    finally:
        _update_backfill_state(running=False)


def _save_intraday_to_duckdb(symbol: str, df: pd.DataFrame) -> None:
    frame = df.copy().reset_index().iloc[:, :6]
    frame.columns = ["datetime", "open", "high", "low", "close", "volume"]
    frame["symbol"] = symbol
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    cumulative_volume = frame["volume"].cumsum().replace(0, pd.NA)
    frame["vwap"] = (typical * frame["volume"]).cumsum() / cumulative_volume

    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute("BEGIN")
            con.register("df_intra", frame)
            con.execute(
                """
                INSERT OR REPLACE INTO intraday_5min
                (symbol, datetime, open, high, low, close, volume, vwap)
                SELECT symbol, datetime, open, high, low, close,
                       CAST(volume AS BIGINT), vwap
                FROM df_intra
                """
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()


def _retry_failed_dates(failed_eq: list[str], failed_fo: list[str]) -> None:
    from data.nse_bhavcopy import NSEBhavcopy

    bhav = NSEBhavcopy()
    for date_str in failed_eq:
        try:
            bhav.download_equity_day(_parse_compact_date(date_str))
        except Exception as error:
            elog.log("retry_failed", "equity_retry_failed", str(error), date_str=date_str)
    for date_str in failed_fo:
        try:
            bhav.download_fo_day(_parse_compact_date(date_str))
        except Exception as error:
            elog.log("retry_failed", "fo_retry_failed", str(error), date_str=date_str)


def _parse_compact_date(date_str: str) -> datetime.date:
    cleaned = str(date_str).strip()
    if "-" in cleaned:
        return datetime.date.fromisoformat(cleaned)
    return datetime.date(int(cleaned[:4]), int(cleaned[4:6]), int(cleaned[6:8]))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "credentials":
        raise SystemExit(credential_manager.cli(sys.argv[2:]))
    # Task Scheduler's installed-runtime Action invokes this frozen executable
    # with "collector-supervisor" (see auto_collector.setup_autostart.runtime_layout
    # for `sys.frozen`): the Job Object process-tree owner runs in this process,
    # then spawns a "collector-run" child of this SAME executable (see
    # auto_collector.job_supervisor._spawn_collector) instead of a repo-relative
    # `uv run .../smart_collector.py`, so the installed product never depends on
    # the development checkout, compute/.venv, or uv.
    if len(sys.argv) >= 2 and sys.argv[1] == "collector-supervisor":
        raise SystemExit(job_supervisor.run())
    if len(sys.argv) >= 2 and sys.argv[1] == "collector-run":
        raise SystemExit(run_smart_collector_once(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "collector-watchdog":
        # Task Scheduler's own RestartOnFailure cannot recover the
        # collector-supervisor process itself dying on Windows -- see
        # auto_collector.supervision's "Watchdog" section for the reproduced
        # live evidence (SCHED_S_TASK_TERMINATED). This is the entry point
        # for a second, independent, time-triggered Task Scheduler task.
        raise SystemExit(job_supervisor.run_watchdog_check())
    if len(sys.argv) >= 2 and sys.argv[1] == "collector-task":
        raise SystemExit(setup_autostart.cli(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "frozen-runtime-probe":
        from ml.frozen_runtime import print_probe

        integrity = integrity_check(expected_phase="PROBE")
        if integrity.get("trusted") is not True:
            raise RuntimeError(f"frozen-runtime probe integrity failed: {integrity.get('reasons')}")
        raise SystemExit(print_probe())
    if len(sys.argv) >= 2 and sys.argv[1] == "frozen-model-sweep":
        from ml.frozen_runtime import print_sweep

        integrity = integrity_check(expected_phase="FINAL")
        if integrity.get("trusted") is not True:
            raise RuntimeError(f"frozen model sweep integrity failed: {integrity.get('reasons')}")
        symbol = sys.argv[2] if len(sys.argv) >= 3 else None
        raise SystemExit(print_sweep(symbol))
    if len(sys.argv) >= 2 and sys.argv[1] == "release-openapi-probe":
        # Build-time only: release_gate.py's packaged-openapi step is the
        # ONLY caller of this branch (it launches the sidecar with exactly
        # this argv token). No real launch path -- Task Scheduler's
        # "collector-supervisor"/"collector-run", or the Tauri parent's bare
        # `Command::new(sidecar).spawn()` in src-tauri/src/main.rs -- ever
        # passes this argument, so `_INTEGRITY_EXPECTED_PHASE` stays "FINAL"
        # (the real, full-strength check) for every real install.
        _INTEGRITY_EXPECTED_PHASE = "PROBE"
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("BHARATTERMINAL_PORT", "8765")))
