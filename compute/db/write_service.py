from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import duckdb

from data.index_aliases import INDEX_FEED_ALIASES as _INDEX_FEED_ALIASES
from data.paths import DUCKDB_PATH
import db.duckdb_client as duckdb_client
from db.duckdb_client import (
    _ACTIVE_WRITER_CONNECTION,
    _WRITER_CONNECTION_LOCK,
    _database_access_lock,
    _lookup_writer_connection,
    _writer_transaction_scope,
    open_read_connection,
    register_writer_connection,
    unregister_writer_connection,
    writer_connection,
)
from db.writer_lease import require_writer_lease


# fo_eod underlying symbols include index/currency/commodity derivative
# underlyings (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYFPI, NIFTYNXT50,
# ...) alongside real equity underlyings. The equity stock screener/universe
# and the per-symbol equity ML trainer must only ever see real NSE cash-equity
# underlyings -- an index has no per-share screener metrics and must not be
# ranked/trained as if it were a stock (it already has its own dedicated
# index-level pipeline; see ml/trainer.py::train_all_models()). Rather than a
# hardcoded symbol blocklist (which silently misses new index/derivative
# underlyings as NSE adds them -- exactly how this was found: FINNIFTY and
# MIDCPNIFTY were excluded by name but NIFTY, BANKNIFTY, NIFTYFPI and
# NIFTYNXT50 were not), this checks real NSE-EQ instrument-master membership.
_FNO_EQUITY_UNDERLYING_FILTER_SQL = """
    EXISTS (
        SELECT 1 FROM instruments i
        WHERE i.exchange = 'NSE' AND i.instrument_type = ''
          AND i.symbol = fo_eod.symbol || '-EQ'
    )
"""

QUEUE_CAPACITY = 256
QUEUE_COMPACT_THRESHOLD_BYTES = 64 * 1024 * 1024
COMPLETED_READ_RETENTION_SECONDS = 10 * 60
HEARTBEAT_INTERVAL_SECONDS = 1.0
HEARTBEAT_STALE_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.05
_SERVICE_REGISTRY_LOCK = threading.Lock()
_ACTIVE_DATABASE_SERVICES: set[Path] = set()

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _own_process_start_time() -> int | None:
    """Windows process creation time as a single monotonic-for-this-boot integer.

    Combined with a PID this proves *which* process holds the lease, not just
    which PID. Returns None off Windows or if the platform call fails, which
    callers treat as "identity cannot be established" (conservative fallback).
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    creation, exit_t, kernel_t, user_t = (wintypes.FILETIME() for _ in range(4))
    handle = kernel32.GetCurrentProcess()
    if not kernel32.GetProcessTimes(
        handle, ctypes.byref(creation), ctypes.byref(exit_t), ctypes.byref(kernel_t), ctypes.byref(user_t)
    ):
        return None
    return (creation.dwHighDateTime << 32) | creation.dwLowDateTime


def _owner_liveness(pid: int | None, start_time: int | None) -> bool | None:
    """Prove whether the recorded lease owner process is still the same live process.

    Returns True if the exact process (PID + creation time) is alive, False if
    it is provably dead or the PID was reused by an unrelated process, and
    None if liveness cannot be established (caller must fall back to the
    conservative heartbeat-staleness policy).
    """
    if pid is None or start_time is None or os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == _ERROR_ACCESS_DENIED:
            return None
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        if exit_code.value != _STILL_ACTIVE:
            return False
        creation, exit_t, kernel_t, user_t = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_t), ctypes.byref(kernel_t), ctypes.byref(user_t)
        ):
            return None
        current_start = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return current_start == int(start_time)
    finally:
        kernel32.CloseHandle(handle)


class WriteOperation(StrEnum):
    LOAD_EQUITY_BHAVCOPY = "LOAD_EQUITY_BHAVCOPY"
    LOAD_FO_BHAVCOPY = "LOAD_FO_BHAVCOPY"
    REFRESH_INSTRUMENT_MASTER = "REFRESH_INSTRUMENT_MASTER"
    UPSERT_FII_DII = "UPSERT_FII_DII"
    UPSERT_EQUITY_SECTORS = "UPSERT_EQUITY_SECTORS"
    UPSERT_IV_OBSERVATIONS = "UPSERT_IV_OBSERVATIONS"
    UPSERT_RISK_FREE_RATE_CURVE = "UPSERT_RISK_FREE_RATE_CURVE"
    BACKFILL_SIGNALS = "BACKFILL_SIGNALS"
    COMPUTE_BIF = "COMPUTE_BIF"
    RUN_OHLC_FORECASTS = "RUN_OHLC_FORECASTS"
    # Model training is deliberately executed outside the DuckDB writer
    # thread.  These legacy values remain only so an old durable queue entry
    # can be failed explicitly after an upgrade instead of being replayed.
    TRAIN_NIFTY_MODELS = "TRAIN_NIFTY_MODELS"
    TRAIN_STOCK_MODELS = "TRAIN_STOCK_MODELS"
    TRAIN_SINGLE_STOCK = "TRAIN_SINGLE_STOCK"
    ENSURE_ML_TRAINING_LOG = "ENSURE_ML_TRAINING_LOG"
    RECORD_ML_TRAINING_LOG = "RECORD_ML_TRAINING_LOG"
    RECORD_INGESTION_MANIFEST = "RECORD_INGESTION_MANIFEST"
    APPLY_RETENTION = "APPLY_RETENTION"
    RECOVER_BACKTESTS = "RECOVER_BACKTESTS"
    INGEST_OHLCV = "INGEST_OHLCV"
    DOWNLOAD_HISTORICAL = "DOWNLOAD_HISTORICAL"
    BACKFILL_PRICE_ACTION = "BACKFILL_PRICE_ACTION"
    LOG_PREDICTION = "LOG_PREDICTION"
    PREDICT_AND_LOG = "PREDICT_AND_LOG"
    RECORD_PREDICTION_OUTCOME = "RECORD_PREDICTION_OUTCOME"
    RESOLVE_PENDING_PREDICTION_OUTCOMES = "RESOLVE_PENDING_PREDICTION_OUTCOMES"
    SAVE_BACKTEST_STARTED = "SAVE_BACKTEST_STARTED"
    RUN_BACKTEST_JOB = "RUN_BACKTEST_JOB"
    RUN_FULL_BACKFILL = "RUN_FULL_BACKFILL"
    RUN_PARSER_TESTS = "RUN_PARSER_TESTS"


class ReadOperation(StrEnum):
    """Bounded, typed reads a caller with no writer authority of its own (the
    standalone collector process, when a different process already owns the
    DuckDB connection) can request through the writer queue. No arbitrary SQL
    is accepted -- each member maps to one fixed, parameterized query below."""

    LATEST_DATE_IN = "LATEST_DATE_IN"
    ROW_COUNT_FOR_DATE = "ROW_COUNT_FOR_DATE"
    FORECAST_SYMBOLS = "FORECAST_SYMBOLS"
    LOAD_OHLCV_FULL_FRAME = "LOAD_OHLCV_FULL_FRAME"
    RESOLVE_INDEX_INSTRUMENT = "RESOLVE_INDEX_INSTRUMENT"
    RESOLVE_LIVE_FEED_UNIVERSE = "RESOLVE_LIVE_FEED_UNIVERSE"
    GET_FNO_UNDERLYINGS = "GET_FNO_UNDERLYINGS"
    GET_INSTRUMENT_TOKEN = "GET_INSTRUMENT_TOKEN"
    GET_LOT_SIZE = "GET_LOT_SIZE"
    LOAD_BIF_HISTORY = "LOAD_BIF_HISTORY"
    LOAD_DECISION_ELIGIBLE_OHLCV = "LOAD_DECISION_ELIGIBLE_OHLCV"
    LOAD_IV_HISTORY_AVERAGE = "LOAD_IV_HISTORY_AVERAGE"
    LOAD_SIGNAL_HISTORY_VALUES = "LOAD_SIGNAL_HISTORY_VALUES"
    LOAD_TECHNICAL_OHLCV_RANGE = "LOAD_TECHNICAL_OHLCV_RANGE"
    LOAD_SCREENER_UNIVERSE_HISTORY = "LOAD_SCREENER_UNIVERSE_HISTORY"
    LOAD_STOCK_SNAPSHOT = "LOAD_STOCK_SNAPSHOT"
    LOAD_UNIVERSE_SNAPSHOT = "LOAD_UNIVERSE_SNAPSHOT"
    RUN_SCREENER_QUERY_ROWS = "RUN_SCREENER_QUERY_ROWS"
    LOAD_FNO_UNIVERSE_SYMBOLS = "LOAD_FNO_UNIVERSE_SYMBOLS"
    LOAD_VALUATION_UNIVERSE = "LOAD_VALUATION_UNIVERSE"
    LOAD_IV_RANK_HISTORY = "LOAD_IV_RANK_HISTORY"
    LOAD_SYMBOL_OHLCV_FOR_BACKFILL = "LOAD_SYMBOL_OHLCV_FOR_BACKFILL"
    LOAD_EXISTING_SIGNAL_DATES = "LOAD_EXISTING_SIGNAL_DATES"
    LOAD_MODEL_ACCURACY_STATS = "LOAD_MODEL_ACCURACY_STATS"
    LOAD_ML_TRAINING_LOG = "LOAD_ML_TRAINING_LOG"
    LOAD_BACKTEST_RUN = "LOAD_BACKTEST_RUN"
    LOAD_BACKTEST_HISTORY = "LOAD_BACKTEST_HISTORY"
    LOAD_STOCK_TRAINER_FNO_SYMBOLS = "LOAD_STOCK_TRAINER_FNO_SYMBOLS"
    LOAD_VERIFIED_SOURCE_OHLCV = "LOAD_VERIFIED_SOURCE_OHLCV"
    LOAD_FEATURE_OHLCV = "LOAD_FEATURE_OHLCV"
    LOAD_FEATURE_BIF_HISTORY = "LOAD_FEATURE_BIF_HISTORY"
    LOAD_FEATURE_SIGNAL_HISTORY = "LOAD_FEATURE_SIGNAL_HISTORY"
    LOAD_FEATURE_IV_HISTORY = "LOAD_FEATURE_IV_HISTORY"
    COUNT_FO_EOD_ROWS = "COUNT_FO_EOD_ROWS"
    LOAD_LATEST_SIGNAL_VALUES = "LOAD_LATEST_SIGNAL_VALUES"
    LOAD_PATTERN_HISTORY_STATS = "LOAD_PATTERN_HISTORY_STATS"
    LOAD_ALPHA_SIGNAL_RETURN_FRAME = "LOAD_ALPHA_SIGNAL_RETURN_FRAME"
    FIND_NIFTY_FRONT_FUTURE_TOKEN = "FIND_NIFTY_FRONT_FUTURE_TOKEN"
    LOAD_FO_EOD_RANGE = "LOAD_FO_EOD_RANGE"
    LOAD_GIFT_NIFTY_LAST_CLOSE = "LOAD_GIFT_NIFTY_LAST_CLOSE"
    LOAD_RAW_OHLCV_FOR_VALIDATION = "LOAD_RAW_OHLCV_FOR_VALIDATION"
    LOAD_FUTURES_BASIS_HISTORY = "LOAD_FUTURES_BASIS_HISTORY"
    LOAD_NIFTY_SPOT_AVAILABILITY_STATS = "LOAD_NIFTY_SPOT_AVAILABILITY_STATS"
    LOAD_CROSS_ASSET_RETURNS = "LOAD_CROSS_ASSET_RETURNS"
    LOAD_MACRO_TABLE_LATEST_VALUES = "LOAD_MACRO_TABLE_LATEST_VALUES"
    LOAD_INDEX_BHAVCOPY_PRESENT_DATES = "LOAD_INDEX_BHAVCOPY_PRESENT_DATES"
    LOAD_OHLCV_CHART_RANGE = "LOAD_OHLCV_CHART_RANGE"
    LOAD_ML_FORECAST_INPUT_FRAME = "LOAD_ML_FORECAST_INPUT_FRAME"
    LOAD_DATA_SUMMARY_STATS = "LOAD_DATA_SUMMARY_STATS"
    LOAD_HEALTH_SNAPSHOT = "LOAD_HEALTH_SNAPSHOT"
    LOAD_RISK_FREE_RATE_CURVE = "LOAD_RISK_FREE_RATE_CURVE"
    LOAD_IV_HISTORY_ROWS_FOR_SYMBOL = "LOAD_IV_HISTORY_ROWS_FOR_SYMBOL"
    COUNT_IV_HISTORY_ROWS = "COUNT_IV_HISTORY_ROWS"


# (table, date_column) for each dataset name a READ request may name. Mirrors
# auto_collector.smart_collector's own client-side allowlist; duplicated
# deliberately so the server side never trusts the caller's validation.
_READ_DATASETS: dict[str, tuple[str, str]] = {
    "bif_history": ("bif_history", "date"),
    "fii_dii_daily": ("fii_dii_daily", "date"),
    "fo_eod": ("fo_eod", "date"),
    "ohlcv": ("ohlcv", "date"),
    "signals_source_ohlcv": ("decision_eligible_ohlcv", "date"),
    "signals_history": ("signals_history", "date"),
}


class WriterServiceError(RuntimeError):
    pass


class WriterUnavailableError(WriterServiceError):
    pass


class WriterBackpressureError(WriterServiceError):
    pass


class WriterRequestFailed(WriterServiceError):
    pass


def queue_path_for(database: Path = DUCKDB_PATH) -> Path:
    return database.with_suffix(database.suffix + ".writer-queue.sqlite")


class WriterService:
    def __init__(
        self,
        database: Path = DUCKDB_PATH,
        *,
        queue_path: Path | None = None,
        capacity: int = QUEUE_CAPACITY,
        dispatcher: Callable[[WriteOperation, dict[str, Any], Path], Any] | None = None,
    ) -> None:
        self.database = Path(database).resolve()
        self.queue_path = Path(queue_path or queue_path_for(self.database)).resolve()
        self.capacity = max(1, int(capacity))
        self.owner_id = f"{os.getpid()}:{uuid.uuid4()}"
        self._dispatcher = dispatcher or _dispatch_operation
        self._stop = threading.Event()
        self._heartbeat_stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._consecutive_reads = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with _SERVICE_REGISTRY_LOCK:
            if self.database in _ACTIVE_DATABASE_SERVICES:
                raise WriterServiceError(f"DuckDB writer already active: {self.database}")
            _ACTIVE_DATABASE_SERVICES.add(self.database)
        try:
            require_writer_lease(self.database)
            # Opened and registered synchronously here, before start() returns
            # control to the caller -- not on the background thread. A caller
            # (e.g. a concurrent GUI request thread) that observes start()
            # having returned must always find the connection already
            # registered; opening it inside _run() instead left a real race
            # where an early concurrent read could run before registration
            # completed and fall through to the old per-call read_only=True
            # path, colliding with this now-open read-write connection
            # (reproduced directly with 4 threads hammering reads immediately
            # after start()).
            with _database_access_lock(self.database):
                self._connection = duckdb.connect(str(self.database))
            register_writer_connection(self.database, self._connection)
            self._ensure_applied_request_ledger(self._connection)
            _ensure_queue_schema(self.queue_path)
            own_start_time = _own_process_start_time()
            with _queue_connection(self.queue_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT owner_id, state, heartbeat_at, pid, process_start_time "
                    "FROM writer_runtime WHERE singleton = 1"
                ).fetchone()
                if existing is not None and str(existing[1]) in {"RUNNING", "DRAINING"}:
                    owner_pid = int(existing[3]) if existing[3] is not None else None
                    owner_start_time = existing[4]
                    liveness = _owner_liveness(owner_pid, owner_start_time)
                    if liveness is True:
                        connection.rollback()
                        raise WriterServiceError(f"DuckDB writer already active: {existing[0]}")
                    if liveness is None:
                        heartbeat = _parse_utc(str(existing[2]))
                        if (
                            dt.datetime.now(dt.timezone.utc) - heartbeat
                        ).total_seconds() <= HEARTBEAT_STALE_SECONDS:
                            connection.rollback()
                            raise WriterServiceError(f"DuckDB writer already active: {existing[0]}")
                    # liveness is False: owner process is provably dead (or the
                    # PID was reused by an unrelated process) -- reclaim now
                    # instead of waiting out the heartbeat TTL.
                recovered_at = _utc_text()
                # READ callers are synchronous and bounded.  If the old
                # writer died, every in-flight caller died or timed out with
                # it; replaying those reads has no consumer and can leave a
                # large stale FIFO ahead of live GUI traffic.  Writes retain
                # their durable replay semantics.
                connection.execute(
                    "UPDATE writer_requests SET state = 'FAILED', owner_id = NULL, "
                    "error = ?, updated_at = ? "
                    "WHERE kind = 'READ' AND state IN ('PENDING', 'RUNNING')",
                    ["reader abandoned during writer handoff", recovered_at],
                )
                connection.execute(
                    "UPDATE writer_requests SET state = 'PENDING', owner_id = NULL, updated_at = ? "
                    "WHERE kind <> 'READ' AND state = 'RUNNING'",
                    [recovered_at],
                )
                connection.execute(
                    "INSERT OR REPLACE INTO writer_runtime "
                    "(singleton, owner_id, pid, process_start_time, state, heartbeat_at) "
                    "VALUES (1, ?, ?, ?, 'RUNNING', ?)",
                    [self.owner_id, os.getpid(), own_start_time, _utc_text()],
                )
                connection.commit()
            _compact_queue_if_needed(self.queue_path)
        except BaseException:
            with _SERVICE_REGISTRY_LOCK:
                _ACTIVE_DATABASE_SERVICES.discard(self.database)
            if self._connection is not None:
                unregister_writer_connection(self.database)
                self._connection.close()
                self._connection = None
            raise
        self._stop.clear()
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="duckdb-domain-writer-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        self._thread = threading.Thread(
            target=self._run,
            name="duckdb-domain-writer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with _queue_connection(self.queue_path) as connection:
            connection.execute(
                "UPDATE writer_runtime SET state = 'DRAINING', heartbeat_at = ? "
                "WHERE singleton = 1 AND owner_id = ? AND state = 'RUNNING'",
                [_utc_text(), self.owner_id],
            )
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(timeout, 0.1))
            if self._thread.is_alive():
                raise WriterServiceError("DuckDB writer service did not stop")
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(timeout, 0.1))
            if self._heartbeat_thread.is_alive():
                raise WriterServiceError("DuckDB writer heartbeat did not stop")
        with _queue_connection(self.queue_path) as connection:
            connection.execute(
                "UPDATE writer_runtime SET state = 'STOPPED', heartbeat_at = ? "
                "WHERE singleton = 1 AND owner_id = ?",
                [_utc_text(), self.owner_id],
            )
        with _SERVICE_REGISTRY_LOCK:
            _ACTIVE_DATABASE_SERVICES.discard(self.database)
        if self._connection is not None:
            unregister_writer_connection(self.database)
            self._connection.close()
            self._connection = None

    def _run(self) -> None:
        # Uses self._connection, opened and registered synchronously in
        # start() before any request-processing thread runs -- not opened
        # here per-request. DuckDB's file-level exclusivity only exists while
        # a connection object is open, so a per-request open/close would
        # leave the file genuinely unlocked between requests, during which an
        # external, non-cooperating `duckdb.connect(path, read_only=False)`
        # succeeds unconditionally (reproduced live: 20/20 bypass attempts
        # succeeded against a running, healthy per-request-connection
        # writer). Safe to hold continuously now that every cross-process
        # reader goes through submit_read() into this same loop instead of
        # opening its own raw connection to this file.
        assert self._connection is not None
        connection = self._connection
        active_request: str | None = None
        try:
            while True:
                request = self._claim_next()
                if request is None:
                    if self._stop.is_set():
                        break
                    self._stop.wait(POLL_INTERVAL_SECONDS)
                    continue
                request_id, kind, operation_text, payload_text = request
                active_request = request_id
                try:
                    if kind == "READ":
                        result = _dispatch_read(operation_text, json.loads(payload_text), connection)
                    else:
                        result = self._apply_atomically(
                            connection,
                            request_id,
                            operation_text,
                            payload_text,
                        )
                except Exception as error:
                    self._ack(
                        request_id,
                        "FAILED",
                        error=f"{type(error).__name__}: {error}",
                    )
                    active_request = None
                    continue
                try:
                    self._ack(request_id, "SUCCEEDED", result=result)
                except Exception:
                    # DuckDB commit may already be durable. Leave RUNNING so a
                    # restart replays the stable request id through its marker.
                    raise
                active_request = None
        except BaseException as error:
            self._fail_runtime(error, active_request=active_request)
        finally:
            self._heartbeat_stop.set()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                self._heartbeat()
            except sqlite3.OperationalError as error:
                # Queue writers can briefly hold SQLite's single-writer lock.
                # Missing one heartbeat is not a terminal DuckDB failure; a
                # genuinely wedged queue will age past HEARTBEAT_STALE_SECONDS
                # and the watchdog will fail it closed.
                if "locked" in str(error).lower():
                    continue
                self._stop.set()
                self._fail_runtime(error, active_request=None)
                return
            except BaseException as error:
                self._stop.set()
                self._fail_runtime(error, active_request=None)
                return

    def _ensure_applied_request_ledger(self, connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS writer_applied_requests (
                request_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                result_json TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )

    def _apply_atomically(
        self,
        connection: duckdb.DuckDBPyConnection,
        request_id: str,
        operation_text: str,
        payload_text: str,
    ) -> Any:
        payload_sha256 = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        connection.execute("BEGIN TRANSACTION")
        try:
            applied = connection.execute(
                "SELECT operation, payload_sha256, result_json "
                "FROM writer_applied_requests WHERE request_id = ?",
                [request_id],
            ).fetchone()
            if applied is not None:
                if str(applied[0]) != operation_text or str(applied[1]) != payload_sha256:
                    raise WriterRequestFailed("idempotency key reused with different request")
                result = json.loads(str(applied[2]))
                # A durable request replay proves this attempt performed no
                # new IV inserts. Returning the original cached `upserted`
                # count made callers report an already-applied batch as new
                # rows even though row-level persistence was correctly a
                # no-op. Preserve generic cached-result semantics, but make
                # this insertion-count API truthful for every submission.
                if (
                    operation_text == WriteOperation.UPSERT_IV_OBSERVATIONS.value
                    and isinstance(result, dict)
                    and "upserted" in result
                ):
                    result = {**result, "upserted": 0}
                connection.execute("COMMIT")
                return result

            operation = WriteOperation(operation_text)
            payload = json.loads(payload_text)
            if not isinstance(payload, dict):
                raise ValueError("request payload must be an object")
            with _writer_transaction_scope(connection):
                result = self._dispatcher(operation, payload, self.database)
            result_json = json.dumps(result, default=str, separators=(",", ":"))
            connection.execute(
                "INSERT INTO writer_applied_requests "
                "(request_id, operation, payload_sha256, result_json, applied_at) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [request_id, operation.value, payload_sha256, result_json],
            )
            connection.execute("COMMIT")
            return json.loads(result_json)
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def _heartbeat(self) -> None:
        with _queue_connection(self.queue_path) as connection:
            changed = connection.execute(
                "UPDATE writer_runtime SET heartbeat_at = ? "
                "WHERE singleton = 1 AND owner_id = ? AND state IN ('RUNNING','DRAINING')",
                [_utc_text(), self.owner_id],
            ).rowcount
        if changed != 1:
            self._stop.set()

    def _fail_runtime(self, error: BaseException, *, active_request: str | None) -> None:
        detail = f"{type(error).__name__}: {error}"
        with _queue_connection(self.queue_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE writer_runtime SET state = 'FAILED', heartbeat_at = ? "
                "WHERE singleton = 1 AND owner_id = ?",
                [_utc_text(), self.owner_id],
            )
            if active_request is None:
                connection.execute(
                    "UPDATE writer_requests SET state = 'FAILED', error = ?, updated_at = ? "
                    "WHERE state = 'PENDING'",
                    [f"writer terminated: {detail}", _utc_text()],
                )
            connection.commit()

    def _claim_next(self) -> tuple[str, str, str, str] | None:
        with _queue_connection(self.queue_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            prefer_reads = self._consecutive_reads < 8
            preferred_kind = "READ" if prefer_reads else "WRITE"
            row = connection.execute(
                "SELECT request_id, kind, operation, payload_json FROM writer_requests "
                "WHERE state = 'PENDING' "
                "ORDER BY CASE WHEN kind = ? THEN 0 ELSE 1 END, created_at, request_id LIMIT 1",
                [preferred_kind],
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            changed = connection.execute(
                "UPDATE writer_requests SET state = 'RUNNING', owner_id = ?, updated_at = ? "
                "WHERE request_id = ? AND state = 'PENDING'",
                [self.owner_id, _utc_text(), row[0]],
            ).rowcount
            connection.commit()
            if changed == 1:
                if str(row[1]) == "READ":
                    self._consecutive_reads += 1
                else:
                    self._consecutive_reads = 0
            return (
                (str(row[0]), str(row[1]), str(row[2]), str(row[3])) if changed == 1 else None
            )

    def _ack(
        self,
        request_id: str,
        state: str,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        result_json = json.dumps(result, default=str, separators=(",", ":"))
        with _queue_connection(self.queue_path) as connection:
            changed = connection.execute(
                "UPDATE writer_requests SET state = ?, result_json = ?, error = ?, updated_at = ? "
                "WHERE request_id = ? AND state = 'RUNNING' AND owner_id = ?",
                [state, result_json, error, _utc_text(), request_id, self.owner_id],
            ).rowcount
        if changed != 1:
            self._stop.set()


def submit_write(
    operation: WriteOperation,
    payload: dict[str, Any],
    *,
    database: Path = DUCKDB_PATH,
    queue_path: Path | None = None,
    timeout: float = 120.0,
    capacity: int = QUEUE_CAPACITY,
    request_id: str | None = None,
) -> Any:
    if not isinstance(operation, WriteOperation):
        raise TypeError("operation must be WriteOperation")
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    encoded = json.dumps(payload, default=str, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
        raise WriterBackpressureError("writer request exceeds 16 MiB payload limit")
    path = Path(queue_path or queue_path_for(database)).resolve()
    _assert_writer_live(path, accept_new=True)
    stable_request_id = str(request_id or uuid.uuid4()).strip()
    if not stable_request_id or len(stable_request_id) > 200:
        raise ValueError("request_id must contain 1..200 characters")
    # Completed idempotency replays are the common backfill-resume path. Read
    # them without taking SQLite's global write lock so sustained replays
    # cannot starve the independent writer heartbeat.
    with _queue_connection(path, read_only=True) as connection:
        existing = connection.execute(
            "SELECT operation, payload_json FROM writer_requests WHERE request_id = ?",
            [stable_request_id],
        ).fetchone()

    replayed_request = existing is not None
    if existing is not None:
        if str(existing[0]) != operation.value or str(existing[1]) != encoded:
            raise WriterRequestFailed("idempotency key reused with different request")
    else:
        with _queue_connection(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Recheck under the write lock: another submitter may have inserted
            # the same stable request id after our read-only fast path.
            existing = connection.execute(
                "SELECT operation, payload_json FROM writer_requests WHERE request_id = ?",
                [stable_request_id],
            ).fetchone()
            if existing is not None:
                replayed_request = True
                if str(existing[0]) != operation.value or str(existing[1]) != encoded:
                    connection.rollback()
                    raise WriterRequestFailed("idempotency key reused with different request")
                connection.commit()
            else:
                queued = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM writer_requests WHERE state IN ('PENDING', 'RUNNING')"
                    ).fetchone()[0]
                )
                if queued >= max(1, int(capacity)):
                    connection.rollback()
                    raise WriterBackpressureError("DuckDB writer queue capacity reached")
                connection.execute(
                    "INSERT INTO writer_requests "
                    "(request_id, operation, payload_json, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'PENDING', ?, ?)",
                    [stable_request_id, operation.value, encoded, _utc_text(), _utc_text()],
                )
                connection.commit()

    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline:
        with _queue_connection(path, read_only=True) as connection:
            row = connection.execute(
                "SELECT state, result_json, error FROM writer_requests WHERE request_id = ?",
                [stable_request_id],
            ).fetchone()
        if row is None:
            raise WriterServiceError("writer request disappeared")
        state = str(row[0])
        if state == "SUCCEEDED":
            result = json.loads(str(row[1]))
            if (
                replayed_request
                and operation is WriteOperation.UPSERT_IV_OBSERVATIONS
                and isinstance(result, dict)
                and "upserted" in result
            ):
                return {**result, "upserted": 0}
            return result
        if state == "FAILED":
            raise WriterRequestFailed(str(row[2] or "writer request failed"))
        _assert_writer_live(path, accept_new=False)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise WriterUnavailableError(f"writer request timed out: {operation.value}")


def submit_read(
    operation: ReadOperation,
    payload: dict[str, Any],
    *,
    database: Path = DUCKDB_PATH,
    queue_path: Path | None = None,
    timeout: float = 30.0,
    capacity: int = QUEUE_CAPACITY,
) -> Any:
    """Request one of a fixed set of typed, parameterized reads from whichever
    process currently owns the live DuckDB connection for `database`. Used by
    the standalone collector process instead of opening its own cross-process
    DuckDB handle, which DuckDB's file-level exclusivity would otherwise
    refuse (or, if allowed, would leave the write-side idle-window bypass
    open) whenever a different process already owns the connection. Always
    freshly dispatched (no idempotency replay -- a read has no durable effect
    to deduplicate)."""
    if not isinstance(operation, ReadOperation):
        raise TypeError("operation must be ReadOperation")
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    encoded = json.dumps(payload, default=str, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise WriterBackpressureError("reader request exceeds 16 KiB payload limit")
    path = Path(queue_path or queue_path_for(database)).resolve()
    _assert_writer_live(path, accept_new=True)
    request_id = str(uuid.uuid4())
    with _queue_connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        queued = int(
            connection.execute(
                "SELECT COUNT(*) FROM writer_requests WHERE state IN ('PENDING', 'RUNNING')"
            ).fetchone()[0]
        )
        if queued >= max(1, int(capacity)):
            connection.rollback()
            raise WriterBackpressureError("DuckDB writer queue capacity reached")
        connection.execute(
            "INSERT INTO writer_requests "
            "(request_id, operation, payload_json, state, kind, created_at, updated_at) "
            "VALUES (?, ?, ?, 'PENDING', 'READ', ?, ?)",
            [request_id, operation.value, encoded, _utc_text(), _utc_text()],
        )
        connection.commit()

    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline:
        with _queue_connection(path, read_only=True) as connection:
            row = connection.execute(
                "SELECT state, result_json, error FROM writer_requests WHERE request_id = ?",
                [request_id],
            ).fetchone()
        if row is None:
            raise WriterServiceError("reader request disappeared")
        state = str(row[0])
        if state == "SUCCEEDED":
            result = json.loads(str(row[1]))
            _delete_completed_read(path, request_id, "SUCCEEDED")
            return result
        if state == "FAILED":
            error = str(row[2] or "reader request failed")
            _delete_completed_read(path, request_id, "FAILED")
            raise WriterRequestFailed(error)
        _assert_writer_live(path, accept_new=False)
        time.sleep(POLL_INTERVAL_SECONDS)
    # A timed-out synchronous reader has no consumer left.  Cancel it while
    # still pending so it cannot accumulate ahead of later live requests.
    # If the writer already claimed it, let that bounded read finish/ack.
    with _queue_connection(path) as connection:
        connection.execute(
            "UPDATE writer_requests SET state = 'FAILED', error = ?, updated_at = ? "
            "WHERE request_id = ? AND state = 'PENDING' AND kind = 'READ'",
            ["reader request timed out", _utc_text(), request_id],
        )
    raise WriterUnavailableError(f"reader request timed out: {operation.value}")


def resolve_read(
    operation: ReadOperation,
    payload: dict[str, Any],
    *,
    database: Path | None = None,
    timeout: float = 30.0,
) -> Any:
    """Single decision boundary for every bounded typed read in the codebase.

    Proactively checks writer ownership state first (mirrors
    ObservationalConnection.execute()'s own local-connection branches), never
    merely by catching a DuckDB exception after blindly opening the file.
    Three cases:
      1. This process holds the live connection (an active writer
         transaction on this thread, or a WriterService registered in this
         same process) -- dispatch against it directly, never a second raw
         connection.
      2. A DIFFERENT process holds it (proven via a fresh heartbeat in the
         writer-queue runtime table, not by attempting the open) -- route
         through the typed cross-process submit_read() queue.
      3. No process holds it at all (no WriterService has ever started, or
         its heartbeat is stale/stopped) -- a short-lived direct read-only
         open is safe and correct; nothing to collide with. Exception
         handling remains available at call sites as defense-in-depth only,
         never as the primary local-vs-remote decision."""
    if not isinstance(operation, ReadOperation):
        raise TypeError("operation must be ReadOperation")
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    # Resolved the same way, and at the same point (call time, not import
    # time), as open_read_connection()'s own `database or DB_PATH` lookup --
    # so a test (or a legitimate runtime reconfiguration) that repoints
    # duckdb_client.DB_PATH is honored here too, not just by direct
    # open_read_connection() callers.
    resolved_database = Path(database) if database is not None else duckdb_client.DB_PATH
    active_writer = _ACTIVE_WRITER_CONNECTION.get()
    if active_writer is not None:
        # Same thread, nested inside this process's own active writer
        # transaction -- must see its own uncommitted writes (see
        # ObservationalConnection.execute()'s identical branch for why a
        # cursor of this connection would not).
        return _dispatch_read(operation.value, payload, active_writer)
    local_writer = _lookup_writer_connection(resolved_database)
    if local_writer is not None:
        # This process owns a live WriterService for `database` but the
        # calling thread is not the writer's own -- serialize a fresh cursor
        # against it, same as ObservationalConnection.execute().
        with _WRITER_CONNECTION_LOCK:
            cursor = local_writer.cursor()
            try:
                return _dispatch_read(operation.value, payload, cursor)
            finally:
                cursor.close()
    status = writer_service_status(queue_path_for(resolved_database))
    if status.get("live"):
        return submit_read(operation, payload, database=resolved_database, timeout=timeout)
    connection = open_read_connection(resolved_database)
    try:
        return _dispatch_read(operation.value, payload, connection)
    finally:
        connection.close()


def is_database_owned_by_another_process(database: Path | None = None) -> bool:
    """True only when a DIFFERENT process currently holds `database` as live
    writer authority: this process has no local writer connection for it
    (neither an active transaction nor a registered WriterService) and the
    writer-queue runtime table shows a live, fresh-heartbeat owner. For
    callers that must skip an operation entirely when they cannot safely
    perform even a read against a file another process holds exclusively
    (e.g. desktop-startup seeding) -- proactive, not a guess made by
    attempting a raw open and catching the resulting DuckDB exception."""
    resolved_database = Path(database) if database is not None else duckdb_client.DB_PATH
    if _ACTIVE_WRITER_CONNECTION.get() is not None:
        return False
    if _lookup_writer_connection(resolved_database) is not None:
        return False
    return bool(writer_service_status(queue_path_for(resolved_database)).get("live"))


def writer_service_status(queue_path: Path) -> dict[str, Any]:
    path = Path(queue_path)
    if not path.is_file():
        return {"live": False, "state": "MISSING", "owner_id": None}
    with _queue_connection(path, read_only=True) as connection:
        row = connection.execute(
            "SELECT owner_id, pid, process_start_time, state, heartbeat_at "
            "FROM writer_runtime WHERE singleton = 1"
        ).fetchone()
        queued = int(
            connection.execute(
                "SELECT COUNT(*) FROM writer_requests WHERE state IN ('PENDING', 'RUNNING')"
            ).fetchone()[0]
        )
    if row is None:
        return {"live": False, "state": "MISSING", "owner_id": None, "queued": queued}
    heartbeat = _parse_utc(str(row[4]))
    heartbeat_fresh = (
        dt.datetime.now(dt.timezone.utc) - heartbeat
    ).total_seconds() <= HEARTBEAT_STALE_SECONDS
    owner_live = _owner_liveness(
        int(row[1]) if row[1] is not None else None,
        int(row[2]) if row[2] is not None else None,
    )
    # Process existence prevents unsafe lease takeover, but does not prove the
    # writer thread is responsive. A stale heartbeat must be unhealthy even
    # while the owning process remains alive so supervision can recover it.
    owner_responsive = heartbeat_fresh and owner_live is not False
    live = str(row[3]) == "RUNNING" and owner_responsive
    responsive = str(row[3]) in {"RUNNING", "DRAINING"} and owner_responsive
    return {
        "live": live,
        "responsive": responsive,
        "owner_id": row[0],
        "pid": row[1],
        "process_start_time": row[2],
        "owner_process_live": owner_live,
        "state": row[3],
        "heartbeat_at": row[4],
        "queued": queued,
    }


def _assert_writer_live(path: Path, *, accept_new: bool) -> None:
    status = writer_service_status(path)
    allowed = bool(status.get("live")) if accept_new else bool(status.get("responsive"))
    if not allowed:
        raise WriterUnavailableError(f"DuckDB writer service unavailable: {status.get('state')}")


def _ensure_queue_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _queue_connection(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS writer_requests (
                request_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('PENDING','RUNNING','SUCCEEDED','FAILED')),
                result_json TEXT,
                error TEXT,
                owner_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        existing_request_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(writer_requests)").fetchall()
        }
        if "kind" not in existing_request_columns:
            connection.execute(
                "ALTER TABLE writer_requests ADD COLUMN kind TEXT NOT NULL DEFAULT 'WRITE'"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS writer_runtime (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                owner_id TEXT NOT NULL,
                pid INTEGER NOT NULL,
                state TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL
            )
            """
        )
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(writer_runtime)").fetchall()
        }
        if "process_start_time" not in existing_columns:
            connection.execute("ALTER TABLE writer_runtime ADD COLUMN process_start_time INTEGER")


def _queue_connection(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
    else:
        connection = sqlite3.connect(str(path), timeout=5)
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _delete_completed_read(path: Path, request_id: str, state: str) -> None:
    with _queue_connection(path) as connection:
        connection.execute(
            "DELETE FROM writer_requests "
            "WHERE request_id = ? AND kind = 'READ' AND state = ?",
            [request_id, state],
        )


def _compact_queue_if_needed(path: Path) -> None:
    cutoff = (
        dt.datetime.now(dt.timezone.utc)
        - dt.timedelta(seconds=COMPLETED_READ_RETENTION_SECONDS)
    ).isoformat(timespec="microseconds")
    with _queue_connection(path) as connection:
        connection.execute(
            "DELETE FROM writer_requests "
            "WHERE kind = 'READ' AND state = 'SUCCEEDED' AND updated_at < ?",
            [cutoff],
        )
    try:
        oversized = path.stat().st_size >= QUEUE_COMPACT_THRESHOLD_BYTES
    except OSError:
        return
    if not oversized:
        return
    with _queue_connection(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")


def _read_dataset(dataset: str) -> tuple[str, str]:
    try:
        return _READ_DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"read dataset is not allowlisted: {dataset!r}") from exc


def _dispatch_read(
    operation_text: str,
    payload: dict[str, Any],
    connection: duckdb.DuckDBPyConnection,
) -> Any:
    operation = ReadOperation(operation_text)
    if operation is ReadOperation.LATEST_DATE_IN:
        table, column = _read_dataset(str(payload["dataset"]))
        symbol = payload.get("symbol")
        if symbol is None:
            row = connection.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
        else:
            row = connection.execute(
                f"SELECT MAX({column}) FROM {table} WHERE symbol = ?", [str(symbol)]
            ).fetchone()
        return {"date": str(row[0]) if row and row[0] is not None else None}
    if operation is ReadOperation.ROW_COUNT_FOR_DATE:
        table, column = _read_dataset(str(payload["dataset"]))
        row = connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = CAST(? AS DATE)",
            [str(payload["date"])],
        ).fetchone()
        return {"count": int(row[0]) if row else 0}
    if operation is ReadOperation.FORECAST_SYMBOLS:
        rows = connection.execute(
            """
            SELECT DISTINCT symbol
            FROM fo_eod
            WHERE date = (SELECT MAX(date) FROM fo_eod)
            ORDER BY symbol
            LIMIT 80
            """
        ).fetchall()
        return {"symbols": [str(row[0]) for row in rows if row and row[0]]}
    if operation is ReadOperation.LOAD_OHLCV_FULL_FRAME:
        candidates = [str(item) for item in payload["symbols"]]
        placeholders = ", ".join("?" for _ in candidates)
        rows = connection.execute(
            f"""
            SELECT CAST(date AS VARCHAR) AS date, open, high, low, close, volume
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
            ORDER BY date ASC
            """,
            candidates,
        ).fetchall()
        return {
            "columns": ["date", "open", "high", "low", "close", "volume"],
            "rows": [list(row) for row in rows],
        }
    if operation is ReadOperation.RESOLVE_INDEX_INSTRUMENT:
        from data.instrument_schema import INSTRUMENT_SCHEMA_VERSION

        normalized = "".join(str(payload["symbol"]).strip().upper().split())
        aliases = _INDEX_FEED_ALIASES.get(normalized)
        if aliases is None:
            raise ValueError("unsupported index feed symbol")
        alias_list = sorted(aliases)
        rows = connection.execute(
            """
            SELECT schema_version, token, symbol, name, exchange, instrument_type
            FROM instruments
            WHERE schema_version = ?
              AND exchange = 'NSE'
              AND instrument_type = 'AMXIDX'
              AND (
                list_contains(?, upper(replace(symbol, ' ', '')))
                OR list_contains(?, upper(replace(name, ' ', '')))
              )
            ORDER BY token
            """,
            [INSTRUMENT_SCHEMA_VERSION, alias_list, alias_list],
        ).fetchall()
        return {
            "normalized": normalized,
            "rows": [
                {
                    "schema_version": int(row[0]),
                    "token": str(row[1]),
                    "symbol": str(row[2]),
                    "name": str(row[3] or ""),
                    "exchange": str(row[4]),
                    "instrument_type": str(row[5]),
                }
                for row in rows
            ],
        }
    if operation is ReadOperation.GET_FNO_UNDERLYINGS:
        rows = connection.execute(
            """
            SELECT DISTINCT COALESCE(NULLIF(nfo.name, ''), nfo.symbol) AS underlying
            FROM instruments nfo
            WHERE nfo.exchange = 'NFO'
              AND nfo.instrument_type IN ('FUTSTK', 'OPTSTK')
              AND UPPER(COALESCE(NULLIF(nfo.name, ''), nfo.symbol)) NOT LIKE '%NSETEST%'
              AND EXISTS (
                  SELECT 1 FROM instruments cash
                  WHERE cash.exchange = 'NSE'
                    AND cash.instrument_type = ''
                    AND cash.symbol = COALESCE(NULLIF(nfo.name, ''), nfo.symbol) || '-EQ'
              )
            ORDER BY underlying
            """
        ).fetchall()
        return {"underlyings": [str(row[0]) for row in rows if row[0]]}
    if operation is ReadOperation.RESOLVE_LIVE_FEED_UNIVERSE:
        from data.index_aliases import CANONICAL_INDEX_SYMBOLS, INDEX_FEED_ALIASES
        from data.instrument_schema import INSTRUMENT_SCHEMA_VERSION

        equity_rows = connection.execute(
            """
            WITH underlyings AS (
                SELECT DISTINCT COALESCE(NULLIF(nfo.name, ''), nfo.symbol) AS underlying
                FROM instruments nfo
                WHERE nfo.exchange = 'NFO'
                  AND nfo.instrument_type IN ('FUTSTK', 'OPTSTK')
                  AND UPPER(COALESCE(NULLIF(nfo.name, ''), nfo.symbol)) NOT LIKE '%NSETEST%'
            ), candidates AS (
                SELECT
                    cash.schema_version,
                    cash.token,
                    underlyings.underlying AS symbol,
                    cash.symbol AS canonical_symbol,
                    cash.name,
                    cash.exchange,
                    cash.instrument_type,
                    ROW_NUMBER() OVER (
                        PARTITION BY underlyings.underlying
                        ORDER BY
                            CASE
                                WHEN cash.symbol = underlyings.underlying || '-EQ' THEN 0
                                WHEN cash.symbol = underlyings.underlying THEN 1
                                WHEN cash.name = underlyings.underlying THEN 2
                                ELSE 3
                            END,
                            cash.token
                    ) AS preference
                FROM underlyings
                JOIN instruments cash
                  ON cash.exchange = 'NSE'
                 AND cash.instrument_type = ''
                 AND (
                     cash.symbol = underlyings.underlying || '-EQ'
                     OR cash.symbol = underlyings.underlying
                     OR cash.name = underlyings.underlying
                 )
            )
            SELECT schema_version, token, symbol, canonical_symbol, name, exchange,
                   instrument_type
            FROM candidates
            WHERE preference = 1
            ORDER BY symbol
            """
        ).fetchall()
        rows = [
            {
                "schema_version": int(row[0]),
                "token": str(row[1]),
                "symbol": str(row[2]).strip().upper(),
                "canonical_symbol": str(row[3]).strip().upper(),
                "name": str(row[4] or ""),
                "exchange": str(row[5]),
                "instrument_type": str(row[6]),
                "asset_class": "EQUITY",
            }
            for row in equity_rows
        ]
        for canonical in CANONICAL_INDEX_SYMBOLS:
            aliases = sorted(INDEX_FEED_ALIASES[canonical])
            index_rows = connection.execute(
                """
                SELECT schema_version, token, symbol, name, exchange, instrument_type
                FROM instruments
                WHERE schema_version = ?
                  AND exchange = 'NSE'
                  AND instrument_type = 'AMXIDX'
                  AND (
                      list_contains(?, upper(replace(symbol, ' ', '')))
                      OR list_contains(?, upper(replace(name, ' ', '')))
                  )
                ORDER BY token
                """,
                [INSTRUMENT_SCHEMA_VERSION, aliases, aliases],
            ).fetchall()
            tokens = {str(row[1]).strip() for row in index_rows if str(row[1]).strip()}
            if len(tokens) != 1:
                raise ValueError(f"{canonical} index token is missing or ambiguous")
            row = index_rows[0]
            rows.append(
                {
                    "schema_version": int(row[0]),
                    "token": str(row[1]),
                    "symbol": canonical,
                    "canonical_symbol": str(row[2]).strip().upper(),
                    "name": str(row[3] or ""),
                    "exchange": str(row[4]),
                    "instrument_type": str(row[5]),
                    "asset_class": "INDEX",
                }
            )
        rows.sort(key=lambda item: (item["asset_class"], item["symbol"]))
        return {"instruments": rows}
    if operation is ReadOperation.GET_INSTRUMENT_TOKEN:
        normalized = str(payload["symbol"]).strip().upper()
        normalized_exchange = str(payload.get("exchange", "NSE")).strip().upper()
        normalized_type = str(payload.get("instrument_type", "EQ")).strip().upper()
        expiry = payload.get("expiry")
        strike = payload.get("strike")
        option_type = payload.get("option_type")
        if normalized_type == "EQ":
            query = """
                SELECT token FROM instruments
                WHERE exchange = ?
                  AND (name = ? OR symbol = ? OR symbol = ?)
                ORDER BY
                  CASE WHEN symbol = ? THEN 0
                       WHEN symbol = ? THEN 1
                       ELSE 2 END
                LIMIT 1
            """
            params: list[Any] = [
                normalized_exchange,
                normalized,
                normalized,
                f"{normalized}-EQ",
                normalized,
                f"{normalized}-EQ",
            ]
        else:
            query = """
                SELECT token FROM instruments
                WHERE exchange = ?
                  AND instrument_type = ?
                  AND (name = ? OR symbol = ?)
            """
            params = [normalized_exchange, normalized_type, normalized, normalized]
            if expiry:
                query += " AND expiry = CAST(? AS DATE)"
                params.append(str(expiry))
            if strike is not None:
                query += " AND ABS(strike - ?) < 0.01"
                params.append(strike)
            if option_type:
                query += " AND option_type = ?"
                params.append(str(option_type).strip().upper())
            query += " ORDER BY expiry NULLS LAST LIMIT 1"
        row = connection.execute(query, params).fetchone()
        return {"token": str(row[0]) if row else None}
    if operation is ReadOperation.GET_LOT_SIZE:
        normalized = str(payload["symbol"]).strip().upper()
        if not normalized or len(normalized) > 64:
            raise ValueError("invalid lot-size symbol")
        expiry = payload.get("expiry")
        params: list[Any] = [normalized, normalized]
        expiry_filter = ""
        if expiry is not None:
            parsed_expiry = _required_date(payload, "expiry")
            expiry_filter = " AND expiry = CAST(? AS DATE)"
            params.append(parsed_expiry.isoformat())
        row = connection.execute(
            f"""
            SELECT MIN(lot_size), MAX(lot_size) FROM instruments
            WHERE exchange = 'NFO'
              AND instrument_type IN ('FUTIDX', 'FUTSTK', 'OPTIDX', 'OPTSTK')
              AND (name = ? OR symbol = ?)
              AND lot_size > 0
              {expiry_filter}
            """,
            params,
        ).fetchone()
        if not row or row[0] is None:
            return {"lot_size": None}
        if int(row[0]) != int(row[1]):
            raise ValueError(f"ambiguous lot size for {normalized}")
        return {"lot_size": int(row[0])}
    if operation is ReadOperation.LOAD_BIF_HISTORY:
        symbol = str(payload["symbol"])
        cutoff_date = str(payload["cutoff_date"])
        rows = connection.execute(
            """
            SELECT
                CAST(date AS VARCHAR), symbol, bms, bms_label, ngi,
                ngi_direction, gravity_center, iad, iad_label,
                wyckoff_stage, vrc_position, vrc_label, vrp,
                nfve_fair, nfve_diff, nfve_valuation, cds,
                cds_label, cds_confidence, action, anomaly_active,
                CAST(computed_at AS VARCHAR)
            FROM bif_history
            WHERE symbol = ? AND date >= CAST(? AS DATE)
            ORDER BY date DESC
            """,
            [symbol, cutoff_date],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_DECISION_ELIGIBLE_OHLCV:
        symbols = [str(item) for item in payload["symbols"]]
        from_date = str(payload["from_date"])
        placeholders = ", ".join("?" for _ in symbols)
        rows = connection.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), observed_at, open, high, low, close, volume,
                   source, quality, derivation_type
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND date >= CAST(? AS DATE)
            ORDER BY date ASC
            """,
            [*symbols, from_date],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_IV_HISTORY_AVERAGE:
        rows = _validated_iv_daily(connection, payload['symbols'], payload)
        return {"rows": [[row['date'], row['iv']] for row in rows]}
    if operation is ReadOperation.LOAD_SIGNAL_HISTORY_VALUES:
        symbols = [str(item) for item in payload["symbols"]]
        from_date = str(payload["from_date"])
        placeholders = ", ".join("?" for _ in symbols)
        rows = connection.execute(
            f"""
            SELECT signal_name, value
            FROM signals_history
            WHERE symbol IN ({placeholders})
              AND date >= CAST(? AS DATE)
            ORDER BY date ASC
            """,
            [*symbols, from_date],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_TECHNICAL_OHLCV_RANGE:
        symbols = [str(item) for item in payload["symbols"]]
        from_date = str(payload["from_date"])
        to_date = str(payload["to_date"])
        placeholders = ", ".join("?" for _ in symbols)
        rows = connection.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), open, high, low, close, volume,
                   CAST(observed_at AS VARCHAR), source, quality, derivation_type
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY date ASC
            """,
            [*symbols, from_date, to_date],
        ).fetchall()
        return {
            "columns": ["symbol", "date", "open", "high", "low", "close", "volume",
                        "observed_at", "source", "quality", "derivation_type"],
            "rows": [list(row) for row in rows],
        }
    if operation is ReadOperation.LOAD_SCREENER_UNIVERSE_HISTORY:
        lookup_symbols = [str(item) for item in payload["symbols"]]
        placeholders = ", ".join("?" for _ in lookup_symbols)
        rows = connection.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), open, high, low, close, volume
            FROM (
                SELECT
                    symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS row_number
                FROM decision_eligible_ohlcv
                WHERE symbol IN ({placeholders})
            )
            WHERE row_number <= 260
            ORDER BY symbol ASC, date ASC
            """,
            lookup_symbols,
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_STOCK_SNAPSHOT:
        symbols = [str(item) for item in payload["symbols"]]
        placeholders = ", ".join("?" for _ in symbols)
        rows = connection.execute(
            f"""
            WITH ranked AS (
                SELECT
                    symbol,
                    CAST(date AS VARCHAR) AS date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    LEAD(close) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS prev_close,
                    LEAD(close, 20) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS close_20d_ago,
                    MAX(high) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 251 FOLLOWING
                    ) AS week_52_high,
                    MIN(low) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 251 FOLLOWING
                    ) AS week_52_low,
                    AVG(volume) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 19 FOLLOWING
                    ) AS avg_volume_20d,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS row_number
                FROM decision_eligible_ohlcv
                WHERE symbol IN ({placeholders})
            )
            SELECT
                symbol,
                date,
                open,
                high,
                low,
                close,
                volume,
                prev_close,
                week_52_high,
                week_52_low,
                avg_volume_20d,
                close_20d_ago
            FROM ranked
            WHERE row_number = 1
            ORDER BY CASE WHEN symbol = ? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            [*symbols, symbols[0]],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_UNIVERSE_SNAPSHOT:
        symbols = payload.get("symbols")
        limit = int(payload["limit"])
        params: list[Any] = []
        symbol_filter = ""
        if symbols:
            cleaned = [str(item) for item in symbols]
            placeholders = ", ".join("?" for _ in cleaned)
            symbol_filter = f"AND symbol IN ({placeholders})"
            params.extend(cleaned)
        rows = connection.execute(
            f"""
            WITH latest_day AS (
                SELECT MAX(date) AS max_date FROM decision_eligible_ohlcv
            ),
            base AS (
                SELECT
                    symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM decision_eligible_ohlcv, latest_day
                WHERE date >= max_date - INTERVAL 430 DAY
                  {symbol_filter}
            ),
            ranked AS (
                SELECT
                    symbol,
                    CAST(date AS VARCHAR) AS date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    LEAD(close) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS prev_close,
                    LEAD(close, 20) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS close_20d_ago,
                    MAX(high) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 251 FOLLOWING
                    ) AS week_52_high,
                    MIN(low) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 251 FOLLOWING
                    ) AS week_52_low,
                    AVG(volume) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 19 FOLLOWING
                    ) AS avg_volume_20d,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS row_number
                FROM base
            ),
            nifty AS (
                SELECT
                    CASE
                        WHEN close_20d_ago IS NULL OR close_20d_ago = 0
                        THEN NULL
                        ELSE ((close - close_20d_ago) / close_20d_ago) * 100
                    END AS nifty_return_20d_pct
                FROM ranked
                WHERE row_number = 1
                  AND symbol IN ('NIFTY 50', 'NIFTY')
                ORDER BY CASE WHEN symbol = 'NIFTY 50' THEN 0 ELSE 1 END
                LIMIT 1
            )
            SELECT
                ranked.symbol,
                ranked.date,
                ranked.open,
                ranked.high,
                ranked.low,
                ranked.close,
                ranked.volume,
                ranked.prev_close,
                ranked.week_52_high,
                ranked.week_52_low,
                ranked.avg_volume_20d,
                ranked.close_20d_ago,
                nifty.nifty_return_20d_pct
            FROM ranked
            LEFT JOIN nifty ON TRUE
            WHERE ranked.row_number = 1
            ORDER BY
                CASE
                    WHEN ranked.avg_volume_20d IS NULL OR ranked.avg_volume_20d = 0
                    THEN NULL
                    ELSE ranked.volume / ranked.avg_volume_20d
                END DESC NULLS LAST,
                ranked.volume DESC NULLS LAST
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.RUN_SCREENER_QUERY_ROWS:
        from data.stock_snapshot import _conditions_sql, _total_scanned_sql

        symbols = payload.get("symbols")
        conditions = payload["conditions"]
        logic = str(payload["logic"])
        limit = int(payload["limit"])
        where_sql, condition_params, labels = _conditions_sql(conditions, logic)

        params: list[Any] = []
        symbol_filter = ""
        scanned_symbols = symbols
        if symbols:
            scanned_symbols = [str(item) for item in symbols]
            placeholders = ", ".join("?" for _ in scanned_symbols)
            symbol_filter = f"AND symbol IN ({placeholders})"
            params.extend(scanned_symbols)

        rows = connection.execute(
            f"""
            WITH latest_day AS (
                SELECT MAX(date) AS max_date FROM decision_eligible_ohlcv
            ),
            base AS (
                SELECT
                    symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM decision_eligible_ohlcv, latest_day
                WHERE date >= max_date - INTERVAL 430 DAY
                  {symbol_filter}
            ),
            ranked AS (
                SELECT
                    symbol,
                    CAST(date AS VARCHAR) AS date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    LEAD(close) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS prev_close,
                    LEAD(close, 20) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS close_20d_ago,
                    MAX(high) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 251 FOLLOWING
                    ) AS week_52_high,
                    MIN(low) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 251 FOLLOWING
                    ) AS week_52_low,
                    AVG(volume) OVER (
                        PARTITION BY symbol ORDER BY date DESC
                        ROWS BETWEEN CURRENT ROW AND 19 FOLLOWING
                    ) AS avg_volume_20d,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY date DESC
                    ) AS row_number
                FROM base
            ),
            snapshot AS (
                SELECT
                    symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    prev_close,
                    close - prev_close AS change,
                    CASE
                        WHEN prev_close IS NULL OR prev_close = 0
                        THEN NULL
                        ELSE ((close - prev_close) / prev_close) * 100
                    END AS change_pct,
                    week_52_high,
                    week_52_low,
                    avg_volume_20d,
                    close_20d_ago,
                    CASE
                        WHEN avg_volume_20d IS NULL OR avg_volume_20d = 0
                        THEN NULL
                        ELSE volume / avg_volume_20d
                    END AS volume_ratio,
                    CASE
                        WHEN week_52_high IS NULL OR week_52_high = 0
                        THEN NULL
                        ELSE ((close - week_52_high) / week_52_high) * 100
                    END AS week_52_high_pct,
                    CASE
                        WHEN week_52_low IS NULL OR week_52_low = 0
                        THEN NULL
                        ELSE ((close - week_52_low) / week_52_low) * 100
                    END AS week_52_low_pct,
                    CASE
                        WHEN close_20d_ago IS NULL OR close_20d_ago = 0
                        THEN NULL
                        ELSE ((close - close_20d_ago) / close_20d_ago) * 100
                    END AS return_20d_pct
                FROM ranked
                WHERE row_number = 1
            ),
            nifty_return AS (
                SELECT return_20d_pct AS nifty_return_20d_pct
                FROM snapshot
                WHERE symbol IN ('NIFTY 50', 'NIFTY')
                ORDER BY CASE WHEN symbol = 'NIFTY 50' THEN 0 ELSE 1 END
                LIMIT 1
            )
            SELECT
                snapshot.symbol,
                snapshot.date,
                snapshot.open,
                snapshot.high,
                snapshot.low,
                snapshot.close,
                snapshot.volume,
                snapshot.prev_close,
                snapshot.week_52_high,
                snapshot.week_52_low,
                snapshot.avg_volume_20d,
                snapshot.close_20d_ago,
                nifty_return.nifty_return_20d_pct
            FROM snapshot
            LEFT JOIN nifty_return ON TRUE
            {where_sql}
            ORDER BY volume_ratio DESC NULLS LAST, volume DESC NULLS LAST
            LIMIT ?
            """,
            [*params, *condition_params, limit],
        ).fetchall()
        total_scanned = connection.execute(
            _total_scanned_sql(scanned_symbols),
            scanned_symbols or [],
        ).fetchone()[0]
        return {
            "rows": [list(row) for row in rows],
            "total_scanned": int(total_scanned or 0),
            "labels": labels,
        }
    if operation is ReadOperation.LOAD_FNO_UNIVERSE_SYMBOLS:
        rows = connection.execute(
            f"""
            SELECT DISTINCT symbol
            FROM fo_eod
            WHERE date = (SELECT MAX(date) FROM fo_eod)
              AND {_FNO_EQUITY_UNDERLYING_FILTER_SQL}
            ORDER BY symbol
            """
        ).fetchall()
        return {"symbols": [str(row[0]).upper() for row in rows]}
    if operation is ReadOperation.LOAD_VALUATION_UNIVERSE:
        rows = connection.execute(
            f"""
            WITH universe AS (
                SELECT DISTINCT symbol
                FROM fo_eod
                WHERE date = (SELECT MAX(date) FROM fo_eod)
                  AND {_FNO_EQUITY_UNDERLYING_FILTER_SQL}
            ), latest AS (
                SELECT symbol, CAST(date AS VARCHAR) AS price_date, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS recency
                FROM decision_eligible_ohlcv
                WHERE symbol IN (SELECT symbol FROM universe)
                  AND close IS NOT NULL
            )
            SELECT universe.symbol, NULLIF(TRIM(fo_universe.sector), ''),
                   latest.price_date, latest.close
            FROM universe
            LEFT JOIN fo_universe ON upper(fo_universe.symbol) = upper(universe.symbol)
            LEFT JOIN latest ON latest.symbol = universe.symbol AND latest.recency = 1
            ORDER BY universe.symbol
            """
        ).fetchall()
        return {
            "rows": [
                {
                    "symbol": str(row[0]).strip().upper(),
                    "sector": str(row[1]).strip() if row[1] is not None else None,
                    "price_date": str(row[2]) if row[2] is not None else None,
                    "previous_close": float(row[3]) if row[3] is not None else None,
                }
                for row in rows
            ]
        }
    if operation is ReadOperation.LOAD_IV_RANK_HISTORY:
        rows = _validated_iv_daily(connection, payload['symbols'], payload)
        return {"rows": [[row['date'], row['strike'], row['iv']] for row in rows]}
    if operation is ReadOperation.LOAD_SYMBOL_OHLCV_FOR_BACKFILL:
        symbol = str(payload["symbol"])
        rows = connection.execute(
            """
            SELECT CAST(date AS VARCHAR) AS date, open, high, low, close, volume
            FROM decision_eligible_ohlcv
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            [symbol],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_EXISTING_SIGNAL_DATES:
        symbol = str(payload["symbol"])
        rows = connection.execute(
            """
            SELECT DISTINCT CAST(date AS VARCHAR)
            FROM signals_history
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchall()
        return {"dates": [str(row[0]) for row in rows]}
    if operation is ReadOperation.LOAD_MODEL_ACCURACY_STATS:
        model_id = str(payload["model_id"])
        n_days = int(payload["n_days"])
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_rows,
                   AVG(outcome_error) AS error_rate,
                   AVG(brier_score) AS brier_score
            FROM ml_predictions
            WHERE model_id = ?
              AND predicted_at >= NOW() - (? || ' days')::INTERVAL
              AND correct IS NOT NULL
            """,
            [model_id, n_days],
        ).fetchone()
        return {"row": list(row)}
    if operation is ReadOperation.LOAD_ML_TRAINING_LOG:
        # Bounded, typed observability read over the training-rejection log
        # (Phase 15): NOT arbitrary SQL -- fixed columns, fixed ordering,
        # optional equality filters on model_id/symbol/status only, hard
        # LIMIT. model_registry.db never stores a rejected attempt at all
        # (only VALIDATED/RETIRED rows), so this table is the only place
        # rejection/insufficient-data training history exists.
        model_id = payload.get("model_id")
        symbol = payload.get("symbol")
        status = payload.get("status")
        raw_limit = payload.get("limit")
        limit = 500 if raw_limit is None else int(raw_limit)
        if limit <= 0 or limit > 5000:
            raise ValueError("limit must be between 1 and 5000")
        # A read must never create schema (that's _ensure_ml_training_log's/
        # _record_ml_training_log's job, both write-side) -- if no training
        # has ever run yet, the table simply doesn't exist, and that is
        # reported as zero rows, not silently swallowed with a bare except.
        clauses: list[str] = []
        params: list[Any] = []
        if model_id:
            clauses.append("model_id = ?")
            params.append(str(model_id))
        if symbol:
            clauses.append("symbol = ?")
            params.append(str(symbol).strip().upper())
        if status:
            clauses.append("status = ?")
            params.append(str(status).strip().upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            rows = connection.execute(
                f"""
                SELECT symbol, model_id, CAST(trained_at AS VARCHAR), n_rows, n_features,
                       train_score, val_score, status, error_msg
                FROM ml_training_log
                {where}
                ORDER BY trained_at DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        except duckdb.CatalogException:
            rows = []
        columns = [
            "symbol", "model_id", "trained_at", "n_rows", "n_features",
            "train_score", "val_score", "status", "error_msg",
        ]
        return {"columns": columns, "rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_BACKTEST_RUN:
        job_id = str(payload["job_id"])
        row = connection.execute(
            """
            SELECT job_id, strategy_name, symbol, CAST(from_date AS VARCHAR),
                   CAST(to_date AS VARCHAR), initial_capital, status, progress,
                   stage, CAST(started_at AS VARCHAR), CAST(completed_at AS VARCHAR),
                   error, result_json
            FROM backtest_runs
            WHERE job_id = ?
            """,
            [job_id],
        ).fetchone()
        return {"row": list(row) if row else None}
    if operation is ReadOperation.LOAD_BACKTEST_HISTORY:
        limit = int(payload["limit"])
        rows = connection.execute(
            """
            SELECT job_id, strategy_name, symbol, CAST(from_date AS VARCHAR),
                   CAST(to_date AS VARCHAR), status, net_return_pct, sharpe_ratio,
                   max_drawdown, win_rate, total_trades, CAST(started_at AS VARCHAR),
                   result_json
            FROM backtest_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_STOCK_TRAINER_FNO_SYMBOLS:
        rows = connection.execute(
            f"""
            SELECT DISTINCT symbol
            FROM fo_eod
            WHERE date = (SELECT MAX(date) FROM fo_eod)
              AND {_FNO_EQUITY_UNDERLYING_FILTER_SQL}
            ORDER BY symbol
            """
        ).fetchall()
        return {"symbols": [str(row[0]).strip().upper() for row in rows if str(row[0]).strip()]}
    if operation is ReadOperation.LOAD_VERIFIED_SOURCE_OHLCV:
        candidates = [str(item) for item in payload["candidates"]]
        sources = [str(item) for item in payload["sources"]]
        placeholders = ", ".join("?" for _ in candidates)
        source_placeholders = ", ".join("?" for _ in sources)
        rows = connection.execute(
            f"""
            SELECT CAST(date AS VARCHAR) AS date, open, high, low, close, volume
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND UPPER(COALESCE(source, '')) IN ({source_placeholders})
            ORDER BY date ASC
            """,
            [*candidates, *sources],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_FEATURE_OHLCV:
        candidates = [str(item) for item in payload["candidates"]]
        sources = [str(item) for item in payload["sources"]]
        placeholders = ", ".join(["?"] * len(candidates))
        source_placeholders = ", ".join(["?"] * len(sources))
        rows = connection.execute(
            f"""
            SELECT symbol AS instrument, date, observed_at, open, high, low, close, volume,
                   source, quality, derivation_type
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND UPPER(COALESCE(source, '')) IN ({source_placeholders})
            ORDER BY date
            """,
            [*candidates, *sources],
        ).fetchall()
        columns = [
            "instrument", "date", "observed_at", "open", "high", "low", "close",
            "volume", "source", "quality", "derivation_type",
        ]
        return {"columns": columns, "rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_FEATURE_BIF_HISTORY:
        candidates = [str(item) for item in payload["candidates"]]
        placeholders = ", ".join(["?"] * len(candidates))
        rows = connection.execute(
            f"""
            SELECT date,
                   cds,
                   bms,
                   iad,
                   ngi,
                   vrc_position AS vrc,
                   nfve_diff AS nfve
            FROM bif_history
            WHERE symbol IN ({placeholders})
            ORDER BY date
            """,
            candidates,
        ).fetchall()
        columns = ["date", "cds", "bms", "iad", "ngi", "vrc", "nfve"]
        return {"columns": columns, "rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_FEATURE_SIGNAL_HISTORY:
        candidates = [str(item) for item in payload["candidates"]]
        placeholders = ", ".join(["?"] * len(candidates))
        rows = connection.execute(
            f"""
            SELECT date, signal_name, value
            FROM signals_history
            WHERE symbol IN ({placeholders})
            ORDER BY date
            """,
            candidates,
        ).fetchall()
        return {"columns": ["date", "signal_name", "value"], "rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_FEATURE_IV_HISTORY:
        rows = _validated_iv_daily(connection, payload['candidates'], payload)
        return {"columns": ["date", "atm_iv"], "rows": [[row['date'], row['iv']] for row in rows]}
    if operation is ReadOperation.COUNT_FO_EOD_ROWS:
        candidates = [str(item) for item in payload["candidates"]]
        placeholders = ", ".join(["?"] * len(candidates))
        row = connection.execute(
            f"SELECT COUNT(*) FROM fo_eod WHERE symbol IN ({placeholders})",
            candidates,
        ).fetchone()
        return {"count": int(row[0] or 0)}
    if operation is ReadOperation.LOAD_LATEST_SIGNAL_VALUES:
        symbol = str(payload["symbol"])
        rows = connection.execute(
            """
            SELECT signal_name, value
            FROM signals_history
            WHERE symbol = ?
              AND date = (
                SELECT MAX(date)
                FROM signals_history
                WHERE symbol = ?
              )
            """,
            [symbol, symbol],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_PATTERN_HISTORY_STATS:
        symbol = str(payload["symbol"])
        rows = connection.execute(
            """
            SELECT pattern_name,
                   AVG(CASE WHEN outcome_5d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                   AVG(ABS(outcome_5d)) AS avg_move
            FROM pattern_history
            WHERE symbol = ?
            GROUP BY pattern_name
            """,
            [symbol],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_ALPHA_SIGNAL_RETURN_FRAME:
        symbol = str(payload["symbol"])
        ohlcv_symbol = str(payload["ohlcv_symbol"])
        signal_rows = connection.execute(
            """
            SELECT CAST(date AS VARCHAR) AS date, signal_name, value
            FROM signals_history
            WHERE symbol = ?
            ORDER BY date
            """,
            [symbol],
        ).fetchall()
        ohlcv_rows = connection.execute(
            """
            SELECT CAST(date AS VARCHAR) AS date, close
            FROM decision_eligible_ohlcv
            WHERE symbol IN (?, ?)
            ORDER BY date
            """,
            [symbol, ohlcv_symbol],
        ).fetchall()
        return {
            "signal_rows": [list(row) for row in signal_rows],
            "ohlcv_rows": [list(row) for row in ohlcv_rows],
        }
    if operation is ReadOperation.FIND_NIFTY_FRONT_FUTURE_TOKEN:
        row = connection.execute(
            """
            SELECT token FROM instruments
            WHERE (name = 'NIFTY' OR symbol LIKE 'NIFTY%')
              AND exchange = 'NFO'
              AND instrument_type = 'FUTIDX'
              AND expiry >= CURRENT_DATE
            ORDER BY expiry ASC
            LIMIT 1
            """
        ).fetchone()
        return {"token": str(row[0]) if row else None}
    if operation is ReadOperation.LOAD_FO_EOD_RANGE:
        symbol = str(payload["symbol"])
        from_date = str(payload["from_date"])
        to_date = str(payload["to_date"])
        rows = connection.execute(
            """
            SELECT symbol, expiry, strike, option_type, date, open, high, low,
                   close, volume, oi, oi_change, source, source_format, price_quality
            FROM fo_eod
            WHERE symbol = ?
              AND date BETWEEN ? AND ?
            ORDER BY date, expiry, strike, option_type
            """,
            [symbol, from_date, to_date],
        ).fetchall()
        return {
            "columns": ["symbol", "expiry", "strike", "option_type", "date", "open",
                        "high", "low", "close", "volume", "oi", "oi_change", "source",
                        "source_format", "price_quality"],
            "rows": [list(row) for row in rows],
        }
    if operation is ReadOperation.LOAD_GIFT_NIFTY_LAST_CLOSE:
        row = connection.execute(
            """
            SELECT close FROM gift_nifty_eod
            ORDER BY date DESC LIMIT 1
            """
        ).fetchone()
        return {"close": float(row[0]) if row else None}
    if operation is ReadOperation.LOAD_RAW_OHLCV_FOR_VALIDATION:
        symbol = str(payload["symbol"])
        n_days = int(payload["n_days"])
        rows = connection.execute(
            """
            SELECT date, close FROM ohlcv
            WHERE symbol = ?
            ORDER BY date DESC LIMIT ?
            """,
            [symbol, n_days],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_FUTURES_BASIS_HISTORY:
        symbol = str(payload["symbol"])
        spot_symbol = str(payload["spot_symbol"])
        futures_rows = connection.execute(
            """
            SELECT CAST(date AS VARCHAR), CAST(expiry AS VARCHAR), close
            FROM fo_eod
            WHERE UPPER(symbol) IN (?, ?)
              AND UPPER(option_type) = 'FUT'
              AND close IS NOT NULL
            ORDER BY date ASC, expiry ASC
            """,
            [symbol, spot_symbol],
        ).fetchall()
        spot_rows = connection.execute(
            """
            SELECT CAST(date AS VARCHAR), UPPER(symbol), close
            FROM decision_eligible_ohlcv
            WHERE UPPER(symbol) IN (?, ?)
              AND close IS NOT NULL
            ORDER BY date ASC,
              CASE WHEN UPPER(symbol) = ? THEN 0 ELSE 1 END
            """,
            [spot_symbol, symbol, spot_symbol],
        ).fetchall()
        return {
            "futures_rows": [list(row) for row in futures_rows],
            "spot_rows": [list(row) for row in spot_rows],
        }
    if operation is ReadOperation.LOAD_NIFTY_SPOT_AVAILABILITY_STATS:
        verified_sources = [str(item) for item in payload["verified_sources"]]
        derived_source = str(payload["derived_source"])
        verified = connection.execute(
            """
            SELECT COUNT(*), MIN(date), MAX(date)
            FROM ohlcv
            WHERE symbol IN ('NIFTY', 'NIFTY 50')
              AND list_contains(?, UPPER(COALESCE(source, '')))
            """,
            [verified_sources],
        ).fetchone()
        quarantined = connection.execute(
            """
            SELECT COUNT(*)
            FROM ohlcv
            WHERE symbol IN ('NIFTY', 'NIFTY 50')
              AND LOWER(COALESCE(source, '')) = ?
            """,
            [derived_source],
        ).fetchone()
        return {"verified": list(verified), "quarantined": list(quarantined)}
    if operation is ReadOperation.LOAD_CROSS_ASSET_RETURNS:
        rows = connection.execute(
            """
            SELECT symbol, CAST(date AS VARCHAR), close
            FROM decision_eligible_ohlcv
            WHERE symbol IN ('NIFTY', 'NIFTY 50', 'DOW', 'SPX', 'NIKKEI', 'BRENT', 'GOLD', 'USDINR', 'INDIA VIX')
            ORDER BY date
            """
        ).fetchall()
        return {"columns": ["symbol", "date", "close"], "rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_MACRO_TABLE_LATEST_VALUES:
        exists = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'macro'"
        ).fetchone()[0]
        if not exists:
            return {"rows": []}
        rows = connection.execute(
            """
            SELECT metric, value
            FROM macro
            QUALIFY row_number() OVER (
                PARTITION BY metric ORDER BY date DESC
            ) = 1
            """
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_INDEX_BHAVCOPY_PRESENT_DATES:
        aliases = [str(item) for item in payload["aliases"]]
        from_date = str(payload["from_date"])
        to_date = str(payload["to_date"])
        placeholders = ", ".join(["?"] * len(aliases))
        rows = connection.execute(
            f"""
            SELECT DISTINCT date
            FROM ohlcv
            WHERE symbol IN ({placeholders})
              AND upper(source) = 'NSE_INDEX_BHAVCOPY'
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            """,
            [*aliases, from_date, to_date],
        ).fetchall()
        return {"dates": [str(row[0]) for row in rows]}
    if operation is ReadOperation.LOAD_OHLCV_CHART_RANGE:
        symbols = [str(item) for item in payload["symbols"]]
        sources = [str(item) for item in payload["sources"]]
        from_date = str(payload["from_date"])
        to_date = str(payload["to_date"])
        placeholders = ", ".join("?" for _ in symbols)
        source_placeholders = ", ".join("?" for _ in sources)
        rows = connection.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), observed_at, open, high, low, close,
                   volume, source, quality, derivation_type
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND UPPER(COALESCE(source, '')) IN ({source_placeholders})
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY date ASC
            """,
            [*symbols, *sources, from_date, to_date],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_ML_FORECAST_INPUT_FRAME:
        symbols = [str(item) for item in payload["symbols"]]
        placeholders = ", ".join("?" for _ in symbols)
        rows = connection.execute(
            f"""
            SELECT symbol AS instrument, CAST(date AS VARCHAR) AS date, observed_at,
                   open, high, low, close, volume, source, quality, derivation_type
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
            ORDER BY date ASC
            """,
            symbols,
        ).fetchall()
        columns = [
            "instrument", "date", "observed_at", "open", "high", "low", "close",
            "volume", "source", "quality", "derivation_type",
        ]
        return {"columns": columns, "rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_DATA_SUMMARY_STATS:
        def optional_count_latest(table: str, column: str) -> tuple[int, Any]:
            exists = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name = ?",
                [table],
            ).fetchone()
            if not exists or int(exists[0]) != 1:
                return (0, None)
            # Both identifiers are fixed server-side literals below; caller
            # input never reaches SQL identifier construction.
            row = connection.execute(
                f"SELECT COUNT(*), MAX({column}) FROM {table}"
            ).fetchone()
            return (int(row[0] or 0), row[1])

        eq = connection.execute(
            "SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT symbol) FROM ohlcv"
        ).fetchone()
        fo = connection.execute(
            "SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT symbol) FROM fo_eod"
        ).fetchone()
        intra = connection.execute(
            "SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM intraday_5min"
        ).fetchone()
        sig = connection.execute(
            "SELECT COUNT(*), MAX(date) FROM signals_history"
        ).fetchone()
        bif = optional_count_latest("bif_history", "date")
        fii = optional_count_latest("fii_dii_daily", "date")
        ohlc_fc = optional_count_latest("ohlc_forecasts", "forecast_date")
        instr = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT exchange) FROM instruments"
        ).fetchone()
        gift = connection.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM gift_nifty_eod"
        ).fetchone()
        return {
            "eq": list(eq),
            "fo": list(fo),
            "intra": list(intra),
            "sig": list(sig),
            "bif": list(bif),
            "fii": list(fii),
            "ohlc_fc": list(ohlc_fc),
            "instr": list(instr),
            "gift": list(gift),
        }
    if operation is ReadOperation.LOAD_HEALTH_SNAPSHOT:
        schema_row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        rows = {
            "ohlcv": connection.execute("SELECT MAX(date) FROM ohlcv").fetchone(),
            "fo_eod": connection.execute("SELECT MAX(date) FROM fo_eod").fetchone(),
            "iv_history": connection.execute("SELECT MAX(date) FROM iv_history").fetchone(),
            "signals_history": connection.execute(
                "SELECT MAX(date) FROM signals_history"
            ).fetchone(),
        }
        latest = {
            table: str(row[0]) if row and row[0] is not None else None
            for table, row in rows.items()
        }
        return {
            "schema_version": int(schema_row[0] or 0) if schema_row else 0,
            "latest": latest,
        }
    if operation is ReadOperation.LOAD_RISK_FREE_RATE_CURVE:
        from_date = str(payload["from_date"])
        to_date = str(payload["to_date"])
        # published_at/retrieved_at are DuckDB TIMESTAMP (no timezone
        # concept) but every writer only ever stores a genuinely UTC
        # instant into them (data/risk_free_rate.py sets both from
        # dt.datetime.now(timezone.utc) / an already-UTC-converted FBIL
        # publish time). A bare `CAST(... AS VARCHAR)` therefore produces a
        # naive-looking string ('2026-09-04 17:19:59.495656', no offset) --
        # options/iv_contract.timestamp() and iv_backfill._timestamp() both
        # correctly refuse to guess a timezone for an unmarked string and
        # return None, which silently made every single rate row
        # unusable (retrieved_at=None fails `select_rate_for_tenor`'s own
        # provenance check) regardless of how much genuine FBIL curve
        # history existed. Appending the UTC marker here is not a policy
        # change -- it corrects a lossy TIMESTAMP->VARCHAR round trip back
        # to what was always genuinely true.
        rows = connection.execute(
            """
            SELECT CAST(observation_date AS VARCHAR), tenor_days, tenor_label,
                   rate_pct, rate_decimal, source, source_url,
                   CASE WHEN published_at IS NULL THEN NULL
                        ELSE CAST(published_at AS VARCHAR) || '+00:00' END,
                   CASE WHEN retrieved_at IS NULL THEN NULL
                        ELSE CAST(retrieved_at AS VARCHAR) || '+00:00' END,
                   quote_convention, rate_convention, conversion_method
            FROM risk_free_rate_history
            WHERE observation_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY observation_date ASC, tenor_days ASC
            """,
            [from_date, to_date],
        ).fetchall()
        return {"rows": [list(row) for row in rows]}
    if operation is ReadOperation.LOAD_IV_HISTORY_ROWS_FOR_SYMBOL:
        symbol = str(payload["symbol"])
        rows = connection.execute(
            f"SELECT {', '.join(_IV_OBSERVATION_COLUMNS)} FROM iv_history WHERE symbol = ? "
            "ORDER BY date, expiry, strike, option_type",
            [symbol],
        ).fetchall()
        return {"columns": list(_IV_OBSERVATION_COLUMNS), "rows": [list(row) for row in rows]}
    if operation is ReadOperation.COUNT_IV_HISTORY_ROWS:
        row = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) FROM iv_history"
        ).fetchone()
        return {"rows": int(row[0] or 0), "symbols": int(row[1] or 0), "first_date": str(row[2]) if row[2] else None, "last_date": str(row[3]) if row[3] else None}
    raise ValueError(f"unhandled read operation: {operation_text}")


def _dispatch_operation(operation: WriteOperation, payload: dict[str, Any], database: Path) -> Any:
    if operation is WriteOperation.UPSERT_FII_DII:
        return _upsert_fii_dii(database, payload)
    if operation is WriteOperation.UPSERT_EQUITY_SECTORS:
        return _upsert_equity_sectors(database, payload)
    if operation is WriteOperation.UPSERT_IV_OBSERVATIONS:
        return _upsert_iv_observations(database, payload)
    if operation is WriteOperation.UPSERT_RISK_FREE_RATE_CURVE:
        return _upsert_risk_free_rate_curve(database, payload)
    if operation is WriteOperation.RECORD_INGESTION_MANIFEST:
        return _record_ingestion_manifest(database, payload)
    if operation is WriteOperation.LOAD_EQUITY_BHAVCOPY:
        from data.nse_bhavcopy import NSEBhavcopy

        day = _required_date(payload, "from_date")
        return {"rows": NSEBhavcopy().load_equity_to_duckdb(symbols="ALL", from_date=day)}
    if operation is WriteOperation.LOAD_FO_BHAVCOPY:
        from data.nse_bhavcopy import NSEBhavcopy

        day = _required_date(payload, "from_date")
        return {"rows": NSEBhavcopy().load_fo_to_duckdb(symbols="ALL", from_date=day)}
    if operation is WriteOperation.REFRESH_INSTRUMENT_MASTER:
        from data.instrument_master import master

        return {"rows": master.refresh(force=bool(payload.get("force", False)))}
    if operation is WriteOperation.BACKFILL_SIGNALS:
        from signals.backfill import backfill_symbol_signals

        return backfill_symbol_signals(_required_symbol(payload))
    if operation is WriteOperation.COMPUTE_BIF:
        from bif.engine import BIFEngine

        return BIFEngine().compute_all(_required_symbol(payload), persist=True)
    if operation in {
        WriteOperation.TRAIN_NIFTY_MODELS,
        WriteOperation.TRAIN_STOCK_MODELS,
        WriteOperation.TRAIN_SINGLE_STOCK,
    }:
        raise WriterRequestFailed(
            "model training is not permitted on the DuckDB writer queue"
        )
    if operation is WriteOperation.ENSURE_ML_TRAINING_LOG:
        return _ensure_ml_training_log()
    if operation is WriteOperation.RECORD_ML_TRAINING_LOG:
        return _record_ml_training_log(payload)
    if operation is WriteOperation.RUN_OHLC_FORECASTS:
        return _run_ohlc_forecast(payload)
    if operation is WriteOperation.APPLY_RETENTION:
        from operations.retention import apply_runtime_retention

        return apply_runtime_retention(database)
    if operation is WriteOperation.RECOVER_BACKTESTS:
        from backtest.engine import recover_interrupted_runs

        return {"recovered": recover_interrupted_runs()}
    if operation is WriteOperation.INGEST_OHLCV:
        import main

        rows = payload.get("rows")
        if set(payload) != {"rows"} or not isinstance(rows, list):
            raise ValueError("invalid OHLCV ingestion payload")
        return {"inserted": main._insert_rows(rows)}
    if operation is WriteOperation.DOWNLOAD_HISTORICAL:
        import main

        from_day = _required_date(payload, "from_date")
        to_day = _required_date(payload, "to_date")
        if set(payload) != {"from_date", "to_date"} or from_day > to_day:
            raise ValueError("invalid historical download payload")
        main._run_download(from_day, to_day)
        return main._status_snapshot()
    if operation is WriteOperation.BACKFILL_PRICE_ACTION:
        from signals.backfill import backfill_price_action_signals

        return backfill_price_action_signals()
    if operation is WriteOperation.LOG_PREDICTION:
        from ml.monitor import log_prediction

        expected = {
            "model_id",
            "prediction",
            "probabilities",
            "confidence",
            "features",
            "attribution",
        }
        if (
            set(payload) != expected
            or not isinstance(payload["probabilities"], dict)
            or not isinstance(payload["features"], dict)
            or not isinstance(payload["attribution"], dict)
        ):
            raise ValueError("invalid prediction-log payload")
        return log_prediction(
            str(payload["model_id"]),
            None if payload["prediction"] is None else str(payload["prediction"]),
            payload["probabilities"],
            payload["confidence"],
            payload["features"],
            attribution=payload["attribution"],
        )
    if operation is WriteOperation.PREDICT_AND_LOG:
        from ml.service import predict_all_models

        if set(payload) != {"symbol", "signals"} or not isinstance(payload["signals"], dict):
            raise ValueError("invalid prediction payload")
        return predict_all_models(_required_symbol(payload), payload["signals"], log=True)
    if operation is WriteOperation.RECORD_PREDICTION_OUTCOME:
        from ml.monitor import record_prediction_outcome

        if set(payload) != {"prediction_id", "actual"}:
            raise ValueError("invalid prediction-outcome payload")
        return record_prediction_outcome(str(payload["prediction_id"]), str(payload["actual"]))
    if operation is WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES:
        from ml.monitor import resolve_pending_prediction_outcomes

        if set(payload) - {"model_ids", "limit"}:
            raise ValueError("invalid outcome-resolution payload")
        model_ids = payload.get("model_ids")
        return resolve_pending_prediction_outcomes(
            list(model_ids) if model_ids is not None else None,
            limit=int(payload.get("limit", 5000)),
        )
    if operation is WriteOperation.SAVE_BACKTEST_STARTED:
        from backtest.engine import save_run_started

        if set(payload) != {"job_id", "params"} or not isinstance(payload["params"], dict):
            raise ValueError("invalid backtest-start payload")
        save_run_started(str(payload["job_id"]), payload["params"])
        return {"job_id": str(payload["job_id"]), "status": "queued"}
    if operation is WriteOperation.RUN_BACKTEST_JOB:
        import main

        if set(payload) != {"job_id", "params"} or not isinstance(payload["params"], dict):
            raise ValueError("invalid backtest-job payload")
        main._run_backtest_job(str(payload["job_id"]), payload["params"])
        return main._get_backtest_job(str(payload["job_id"])) or {}
    if operation is WriteOperation.RUN_FULL_BACKFILL:
        import main

        expected = {"scope", "years_equity", "years_fo", "months_intraday"}
        if set(payload) != expected:
            raise ValueError("invalid full-backfill payload")
        main._run_full_backfill(
            str(payload["scope"]),
            int(payload["years_equity"]),
            int(payload["years_fo"]),
            int(payload["months_intraday"]),
        )
        return main._backfill_snapshot()
    if operation is WriteOperation.RUN_PARSER_TESTS:
        import main

        main._run_test_parsers()
        return main._get_parser_report()
    raise ValueError(f"unsupported typed writer operation: {operation.value}")


ML_TRAINING_LOG_STATUSES = frozenset(
    {
        # Traced live from real code, not assumed (2026-09-03):
        # trainer.py's train_all_models sets "READY" on a successful
        # artifact and derives "INSUFFICIENT_DATA"/"FAILED" otherwise
        # (compute/ml/trainer.py:44-60); ml/models/common.py's
        # _unavailable_status is the shared vocabulary a model's own
        # train_model() result can carry through unchanged when
        # trainer.py doesn't overwrite it (INSUFFICIENT_DATA, QUARANTINED,
        # STALE, FAILED, UNTRAINED); READY_ANALYTICAL covers a model whose
        # own train_model() reports readiness without a persisted artifact.
        "READY",
        "READY_ANALYTICAL",
        "INSUFFICIENT_DATA",
        "QUARANTINED",
        "STALE",
        "UNTRAINED",
        "FAILED",
    }
)

_ML_TRAINING_LOG_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_training_log (
    symbol                  TEXT        NOT NULL,
    model_id                TEXT        NOT NULL,
    trained_at              TIMESTAMP   NOT NULL,
    n_rows                  INTEGER,
    n_features              INTEGER,
    train_score             DOUBLE,
    val_score               DOUBLE,
    feature_importance_top5 TEXT,
    bif_rows_used           INTEGER,
    ohlcv_rows_used         INTEGER,
    status                  TEXT        NOT NULL,
    error_msg               TEXT
)
"""


def _ensure_ml_training_log() -> dict[str, Any]:
    connection = writer_connection()
    try:
        connection.execute(_ML_TRAINING_LOG_CREATE_SQL)
    finally:
        connection.close()
    return {"ready": True}


def _record_ml_training_log(payload: dict[str, Any]) -> dict[str, Any]:
    columns = (
        "symbol",
        "model_id",
        "trained_at",
        "n_rows",
        "n_features",
        "train_score",
        "val_score",
        "feature_importance_top5",
        "bif_rows_used",
        "ohlcv_rows_used",
        "status",
        "error_msg",
    )
    if set(payload) != set(columns):
        raise ValueError("invalid ML training-log payload schema")
    symbol = _required_symbol(payload)
    model_id = str(payload["model_id"]).strip()
    status = str(payload["status"]).strip().upper()
    if not model_id or len(model_id) > 100:
        raise ValueError("invalid ML training-log model_id")
    if status not in ML_TRAINING_LOG_STATUSES:
        # Fail closed on an unrecognized status rather than silently
        # dropping/renaming it -- this table exists to be trustworthy
        # observability, and this exact bug (validating a narrower
        # "success"/"failed" pair against the real, wider vocabulary
        # trainer.py/stock_trainer.py/models/common.py actually emit --
        # READY, READY_ANALYTICAL, INSUFFICIENT_DATA, QUARANTINED, STALE,
        # UNTRAINED, FAILED -- verified live 2026-09-03, submit_write with
        # status="READY" raised ValueError) has been silently discarding
        # most non-failure training log rows in production. Extend
        # ML_TRAINING_LOG_STATUSES, with evidence, rather than loosen this.
        raise ValueError(f"invalid ML training-log status: {status!r}")
    connection = writer_connection()
    try:
        connection.execute(_ML_TRAINING_LOG_CREATE_SQL)
        connection.execute(
            """
            INSERT INTO ml_training_log
                (symbol, model_id, trained_at, n_rows, n_features,
                 train_score, val_score, feature_importance_top5,
                 bif_rows_used, ohlcv_rows_used, status, error_msg)
            VALUES (?, ?, CAST(? AS TIMESTAMP), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                symbol,
                model_id,
                payload["trained_at"],
                payload["n_rows"],
                payload["n_features"],
                payload["train_score"],
                payload["val_score"],
                str(payload["feature_importance_top5"]),
                payload["bif_rows_used"],
                payload["ohlcv_rows_used"],
                status,
                payload["error_msg"],
            ],
        )
    finally:
        connection.close()
    return {"symbol": symbol, "model_id": model_id, "status": status}


def _upsert_equity_sectors(database: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Bulk-upsert `fo_universe.sector` (and the required NOT NULL name/lot_size
    columns) for equities the caller has already resolved a real sector for.
    No sector is invented here -- rows with no resolved sector are simply
    omitted by the caller, which is why `insufficient` reflects the honest
    count of equities valuation still cannot classify."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("equity sector payload requires a non-empty rows list")
    required = {"symbol", "name", "lot_size", "sector"}
    parsed: list[tuple[str, str, int, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("invalid equity sector row schema")
        symbol = str(row["symbol"]).strip().upper()
        sector = str(row["sector"]).strip()
        lot_size = int(row["lot_size"])
        if not symbol or not sector or lot_size <= 0:
            raise ValueError(f"invalid equity sector row for {row.get('symbol')!r}")
        parsed.append((symbol, str(row["name"]).strip() or symbol, lot_size, sector))

    connection = writer_connection()
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fo_universe (
                symbol      TEXT        PRIMARY KEY,
                name        TEXT,
                lot_size    INTEGER     NOT NULL,
                sector      TEXT,
                is_active   INTEGER     DEFAULT 1,
                updated_at  TIMESTAMP   NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO fo_universe (symbol, name, lot_size, sector, is_active, updated_at)
            VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (symbol) DO UPDATE SET
                name = excluded.name,
                lot_size = excluded.lot_size,
                sector = excluded.sector,
                is_active = 1,
                updated_at = excluded.updated_at
            """,
            [(symbol, name, lot_size, sector) for symbol, name, lot_size, sector in parsed],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {"upserted": len(parsed)}


from options.iv_contract import OBSERVATION_COLUMNS as _IV_OBSERVATION_COLUMNS


def _validated_iv_daily(connection: Any, symbols: list[str], payload: dict[str, Any]) -> list[dict[str, Any]]:
    from options.iv_contract import daily_atm_rows, timestamp
    import datetime as dt
    columns = {
        row[0]
        for row in connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'iv_history'"
        ).fetchall()
    }
    if not set(_IV_OBSERVATION_COLUMNS).issubset(columns) or not symbols:
        return []  # legacy schema is stored, never promoted to validated history
    as_of = timestamp(payload.get('as_of')) if payload.get('as_of') else dt.datetime.now(dt.timezone.utc)
    if as_of is None:
        raise ValueError('as_of must be timezone-aware')
    normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    placeholders = ", ".join("?" for _ in normalized_symbols)
    rows = connection.execute(
        f"SELECT {', '.join(_IV_OBSERVATION_COLUMNS)} FROM iv_history "
        f"WHERE upper(symbol) IN ({placeholders}) AND date >= CAST(? AS DATE) AND date < CAST(? AS DATE) "
        "ORDER BY date, expiry, strike LIMIT 100000",
        [*normalized_symbols, payload.get('from_date', '1900-01-01'), as_of.date().isoformat()],
    ).fetchall()
    observations = []
    for raw in rows:
        row = dict(zip(_IV_OBSERVATION_COLUMNS, raw))
        for key in ('date', 'expiry', 'risk_free_observation_date'):
            if row[key] is not None:
                row[key] = row[key].isoformat() if hasattr(row[key], 'isoformat') else str(row[key])
        observations.append(row)
    return daily_atm_rows(
        observations,
        as_of=as_of,
        observation_kind=str(payload.get('observation_kind') or 'HISTORICAL_RECONSTRUCTION'),
    )


def _write_iv_rows(connection: Any, rows: Any) -> dict[str, Any]:
    """Called only inside the WriterService-owned transaction.

    Exact replay is a no-op. Conflicting values/kinds never replace history.
    Whole batch validation precedes insertion; outer writer ledger guarantees
    crash recovery and atomicity, including calls from COMPUTE_BIF.
    """
    from options.iv_contract import validate_observation
    if _ACTIVE_WRITER_CONNECTION.get() is None:
        raise RuntimeError('IV persistence requires an active WriterService transaction')
    if not isinstance(rows, list) or not rows or len(rows) > 2000:
        raise ValueError('IV observation batch requires 1..2000 rows')
    unique = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_IV_OBSERVATION_COLUMNS):
            raise ValueError('invalid IV observation row schema')
        validate_observation(row)
        key = str(row['observation_id'])
        if key in unique and unique[key] != row:
            raise ValueError('conflicting duplicate IV key')
        unique[key] = row
    inserted = 0
    for key, row in unique.items():
        existing = connection.execute(
            f"SELECT {', '.join(_IV_OBSERVATION_COLUMNS)} FROM iv_history "
            "WHERE observation_id=?", [key],
        ).fetchone()
        values = tuple(row[name] for name in _IV_OBSERVATION_COLUMNS)
        if existing:
            # Compare via typed SQL casts, ignoring creation time on a replay.
            comparison = [name for name in _IV_OBSERVATION_COLUMNS if name != 'created_at']
            equal = connection.execute(
                'SELECT ' + ' AND '.join(f'{name} IS NOT DISTINCT FROM ?' for name in comparison)
                + ' FROM iv_history WHERE observation_id=?',
                [*[row[name] for name in comparison], key],
            ).fetchone()[0]
            if not equal:
                raise ValueError('IV key already exists with different inputs or provenance')
            continue
        connection.execute(
            f"INSERT INTO iv_history ({', '.join(_IV_OBSERVATION_COLUMNS)}) VALUES ({', '.join('?' for _ in values)})", values,
        )
        inserted += 1
    return {'upserted': inserted}


def _upsert_iv_observations(database: Path, payload: dict[str, Any]) -> dict[str, Any]:
    connection = writer_connection()
    try:
        return _write_iv_rows(connection, payload.get('rows'))
    finally:
        connection.close()


_RATE_CURVE_COLUMNS = (
    "observation_date", "tenor_days", "tenor_label", "rate_pct",
    "rate_decimal", "source", "source_url", "published_at", "retrieved_at",
    "quote_convention", "rate_convention", "conversion_method",
)


def _upsert_risk_free_rate_curve(database: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Bulk-upsert risk_free_rate_history rows fetched from a real external
    provider (see data/risk_free_rate.py). Never invents a rate: the caller
    must already have a real fetched (observation_date, tenor_days, rate)
    triple for every row."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("risk-free rate payload requires a non-empty rows list")
    if len(rows) > 2000:
        raise ValueError("risk-free rate batch exceeds bounded size (max 2000 rows)")
    parsed: list[tuple[Any, ...]] = []
    from data.risk_free_rate import (
        FBIL_QUOTE_CONVENTION,
        FBIL_SOURCE_ID,
        FBIL_TBILL_3M_HISTORY_URL,
        FBIL_TBILL_LATEST_URL,
        OPTION_RATE_CONVENTION,
        RATE_CONVERSION_METHOD,
        TenorPoint,
    )
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_RATE_CURVE_COLUMNS):
            raise ValueError("invalid risk-free rate row schema")
        rate_pct = row.get("rate_pct")
        if not isinstance(rate_pct, (int, float)) or not (0.0 < float(rate_pct) < 50.0):
            raise ValueError(f"implausible risk-free rate rejected: {row!r}")
        tenor_days = row.get("tenor_days")
        if not isinstance(tenor_days, int) or tenor_days <= 0:
            raise ValueError(f"invalid risk-free rate tenor_days: {row!r}")
        if (
            row.get("source") != FBIL_SOURCE_ID
            or row.get("source_url") not in {FBIL_TBILL_LATEST_URL, FBIL_TBILL_3M_HISTORY_URL}
            or row.get("quote_convention") != FBIL_QUOTE_CONVENTION
            or row.get("rate_convention") != OPTION_RATE_CONVENTION
            or row.get("conversion_method") != RATE_CONVERSION_METHOD
        ):
            raise ValueError("unverified risk-free rate convention")
        expected_rate = TenorPoint(
            observation_date=_required_date(row, "observation_date"),
            tenor_days=tenor_days,
            tenor_label=str(row["tenor_label"]),
            rate_pct=float(rate_pct),
            published_at=None,
            source=str(row["source"]),
            source_url=str(row["source_url"]),
        ).rate_decimal
        if not isinstance(row.get("rate_decimal"), (int, float)) or not math.isclose(
            float(row["rate_decimal"]), expected_rate, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("risk-free continuous-rate conversion mismatch")
        retrieved_at = _aware_datetime(row.get("retrieved_at"))
        published_at = _aware_datetime(row.get("published_at")) if row.get("published_at") is not None else None
        if retrieved_at is None:
            raise ValueError("risk-free retrieval timestamp must be timezone-aware")
        if row.get("published_at") is not None and published_at is None:
            raise ValueError("risk-free publication timestamp must be timezone-aware")
        observation_start = dt.datetime.combine(
            _required_date(row, "observation_date"), dt.time.min,
            dt.timezone(dt.timedelta(hours=5, minutes=30)),
        ).astimezone(dt.timezone.utc)
        if retrieved_at < observation_start or (published_at is not None and retrieved_at < published_at):
            raise ValueError("risk-free retrieval precedes its source observation")
        parsed.append(tuple(row[name] for name in _RATE_CURVE_COLUMNS))

    connection = writer_connection()
    try:
        connection.execute("BEGIN TRANSACTION")
        inserted = 0
        for values in parsed:
            existing = connection.execute(
                "SELECT tenor_label, rate_pct, rate_decimal, source, quote_convention, "
                "rate_convention, conversion_method FROM risk_free_rate_history "
                "WHERE observation_date=CAST(? AS DATE) AND tenor_days=?",
                [values[0], values[1]],
            ).fetchone()
            semantic = (values[2], values[3], values[4], values[5], values[9], values[10], values[11])
            if existing is not None:
                if tuple(existing) != semantic:
                    raise ValueError("risk-free observation conflicts with immutable stored rate")
                continue
            connection.execute(
                """
                INSERT INTO risk_free_rate_history
                    (observation_date, tenor_days, tenor_label, rate_pct,
                     rate_decimal, source, source_url, published_at, retrieved_at,
                     quote_convention, rate_convention, conversion_method)
                VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMP), CAST(? AS TIMESTAMP), ?, ?, ?)
                """,
                values,
            )
            inserted += 1
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {"upserted": inserted}


def _upsert_fii_dii(database: Path, payload: dict[str, Any]) -> dict[str, Any]:
    columns = (
        "date",
        "fii_buy",
        "fii_sell",
        "fii_net",
        "dii_buy",
        "dii_sell",
        "dii_net",
        "fii_net_futures",
        "raw_json",
    )
    if set(payload) != set(columns):
        raise ValueError("invalid FII/DII payload schema")
    _required_date(payload, "date")
    connection = writer_connection()
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fii_dii_daily (
                date DATE PRIMARY KEY, fii_buy DOUBLE, fii_sell DOUBLE, fii_net DOUBLE,
                dii_buy DOUBLE, dii_sell DOUBLE, dii_net DOUBLE,
                fii_net_futures DOUBLE, raw_json TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO fii_dii_daily
            (date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net,
             fii_net_futures, raw_json)
            VALUES (CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [payload[name] for name in columns],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {"date": payload["date"], "persisted": True}


def _record_ingestion_manifest(database: Path, payload: dict[str, Any]) -> dict[str, Any]:
    columns = (
        "source_id",
        "source_path",
        "source_sha256",
        "source_identity",
        "expected_scope_json",
        "observed_record_count",
        "ingestion_completed_at",
        "validation_completed_at",
        "completeness_status",
        "downstream_commit_id",
    )
    if set(payload) != set(columns):
        raise ValueError("invalid ingestion manifest payload schema")
    digest = str(payload["source_sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("invalid ingestion manifest SHA-256")
    if str(payload["completeness_status"]) != "COMPLETE":
        raise ValueError("only validated complete ingestion may be manifested")
    connection = writer_connection()
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            INSERT OR REPLACE INTO ingestion_manifests
            (source_id, source_path, source_sha256, source_identity,
             expected_scope_json, observed_record_count, ingestion_completed_at,
             validation_completed_at, completeness_status, downstream_commit_id,
             retention_decision)
            VALUES (?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMP), CAST(? AS TIMESTAMP), ?, ?, 'RETAIN')
            """,
            [payload[name] for name in columns],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {"source_id": payload["source_id"], "retention_decision": "RETAIN"}


def _run_ohlc_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    from ml.feature_engineering import build_context
    from ml.models.m21_ohlc_forecast import OHLCForecaster

    symbol = _required_symbol(payload)
    context = build_context(symbol, {})
    result = OHLCForecaster().predict(
        symbol,
        context.ohlcv,
        context.current_spot,
        log_forecast=True,
        include_accuracy=True,
    )
    return result


def _required_date(payload: dict[str, Any], key: str) -> dt.date:
    value = payload.get(key)
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"invalid {key}") from error


def _aware_datetime(value: Any) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _required_symbol(payload: dict[str, Any]) -> str:
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol or len(symbol) > 64:
        raise ValueError("invalid symbol")
    return symbol


def _utc_text() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
