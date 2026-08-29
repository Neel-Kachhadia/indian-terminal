from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar, Token
import hashlib
import os
import re
from threading import Lock, RLock
from typing import Any, Iterator

import duckdb

from data.paths import DUCKDB_PATH, ensure_dirs
from data.market_data_quality import DECISION_ELIGIBLE_OHLCV_VIEW_SQL

DB_PATH = DUCKDB_PATH
DB_WRITE_LOCK = Lock()
_ACTIVE_WRITER_CONNECTION: ContextVar[duckdb.DuckDBPyConnection | None] = ContextVar(
    "active_duckdb_writer_connection",
    default=None,
)

# A WriterService holds its own DuckDB connection open continuously for its
# entire running lifetime (registered here at the start of its run loop,
# removed when it exits) so that DuckDB's OS-level file exclusivity is held
# without gaps, closing the window where an external, non-cooperating
# `duckdb.connect(path, read_only=False)` could otherwise succeed between
# per-request transactions. In-process observational reads from other threads
# (e.g. FastAPI request handlers) reuse this connection's `.cursor()` instead
# of opening a second, differently-configured connection, which DuckDB
# rejects outright for same-process same-file access. Cross-process readers
# (the standalone collector) must not rely on this registry at all -- they go
# through db.write_service.submit_read() instead, since this registry only
# has meaning within the process that owns the connection.
_WRITER_CONNECTION_REGISTRY_LOCK = Lock()
_WRITER_CONNECTIONS: dict[str, duckdb.DuckDBPyConnection] = {}
# Serializes cursor access to a registered writer connection from threads
# other than the writer's own (which uses the connection directly, never a
# cursor of it -- see ObservationalConnection.execute() below).
_WRITER_CONNECTION_LOCK = RLock()


def register_writer_connection(database: Path, connection: duckdb.DuckDBPyConnection) -> None:
    key = str(Path(database).resolve())
    with _WRITER_CONNECTION_REGISTRY_LOCK:
        _WRITER_CONNECTIONS[key] = connection


def unregister_writer_connection(database: Path) -> None:
    key = str(Path(database).resolve())
    with _WRITER_CONNECTION_REGISTRY_LOCK:
        _WRITER_CONNECTIONS.pop(key, None)


def _lookup_writer_connection(database: Path) -> duckdb.DuckDBPyConnection | None:
    key = str(Path(database).resolve())
    with _WRITER_CONNECTION_REGISTRY_LOCK:
        return _WRITER_CONNECTIONS.get(key)


class ObservationalConnection:
    """Materializing query capability with no retained DuckDB handle."""

    _ALLOWED = re.compile(r"^\s*(?:SELECT|WITH|SHOW|DESCRIBE|EXPLAIN)\b", re.IGNORECASE)
    _MUTATION = re.compile(
        r"\b(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|COPY|ATTACH|DETACH|CALL|EXPORT|IMPORT|INSTALL|LOAD|VACUUM|CHECKPOINT)\b",
        re.IGNORECASE,
    )

    def __init__(self, database: Path) -> None:
        self.__database = Path(database).resolve()
        self.__rows: list[Any] = []
        self.__columns: list[str] = []
        self.__position = 0
        self.__closed = False

    def execute(self, query: str, parameters: Any = None) -> "ObservationalConnection":
        if self.__closed:
            raise RuntimeError("observational DuckDB connection is closed")
        if not isinstance(query, str) or not self._ALLOWED.match(query) or self._MUTATION.search(query):
            raise PermissionError("observational DuckDB connection rejected non-read query")
        active_writer = _ACTIVE_WRITER_CONNECTION.get()
        if active_writer is not None:
            # Same thread, nested inside the writer's own active transaction
            # (e.g. a dispatch handler reading history it, or an earlier step
            # of the same operation, already wrote). Must reuse the connection
            # itself, not a `.cursor()` of it: a cursor gets its own MVCC
            # snapshot and does not see the parent's own uncommitted writes
            # (verified directly -- mid-transaction, a cursor read returned
            # the pre-transaction count while the connection's own .execute()
            # returned the correct in-transaction count). Safe because this
            # call is synchronously nested in the writer's own call stack on
            # its own thread; no concurrent use of this connection object can
            # occur here.
            if parameters is None:
                result = active_writer.execute(query)
            else:
                result = active_writer.execute(query, parameters)
            self.__columns = [str(item[0]) for item in (result.description or [])]
            self.__rows = result.fetchall()
            self.__position = 0
            return self
        writer_connection = _lookup_writer_connection(self.__database)
        if writer_connection is not None:
            # No active writer transaction on this thread, but a WriterService
            # in this same process owns a live connection to this file (e.g.
            # this is a FastAPI request-handler thread reading while the
            # writer thread is idle or mid-transaction elsewhere). Reuse it
            # via a fresh cursor -- correct snapshot isolation, never sees a
            # concurrent in-flight transaction's uncommitted state -- and
            # never a raw independent connect, which DuckDB would refuse
            # outright as a same-process, differently-configured connection.
            with _WRITER_CONNECTION_LOCK:
                cursor = writer_connection.cursor()
                try:
                    if parameters is None:
                        result = cursor.execute(query)
                    else:
                        result = cursor.execute(query, parameters)
                    self.__columns = [str(item[0]) for item in (result.description or [])]
                    self.__rows = result.fetchall()
                    self.__position = 0
                finally:
                    cursor.close()
            return self
        with _database_access_lock(self.__database):
            reader = duckdb.connect(str(self.__database), read_only=True)
            try:
                if parameters is None:
                    result = reader.execute(query)
                else:
                    result = reader.execute(query, parameters)
                self.__columns = [str(item[0]) for item in (result.description or [])]
                self.__rows = result.fetchall()
                self.__position = 0
            finally:
                reader.close()
        return self

    def fetchone(self) -> Any:
        if self.__position >= len(self.__rows):
            return None
        row = self.__rows[self.__position]
        self.__position += 1
        return row

    def fetchall(self) -> list[Any]:
        rows = self.__rows[self.__position :]
        self.__position = len(self.__rows)
        return list(rows)

    def fetchdf(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise TypeError("observational fetchdf does not accept conversion options")
        import pandas as pd

        rows = self.fetchall()
        return pd.DataFrame.from_records(rows, columns=self.__columns)

    def df(self, *args: Any, **kwargs: Any) -> Any:
        return self.fetchdf(*args, **kwargs)

    def close(self) -> None:
        self.__closed = True
        self.__rows = []
        self.__columns = []

    def __enter__(self) -> "ObservationalConnection":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class WriterTransactionConnection:
    """Non-owning facade for one service-owned outer DuckDB transaction."""

    _TRANSACTION_CONTROL = re.compile(
        r"^\s*(?:BEGIN(?:\s+TRANSACTION)?|COMMIT|ROLLBACK)\s*;?\s*$",
        re.IGNORECASE,
    )

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def execute(self, query: str, parameters: Any = None) -> "WriterTransactionConnection":
        if self._TRANSACTION_CONTROL.match(query):
            return self
        if parameters is None:
            self._connection.execute(query)
        else:
            self._connection.execute(query, parameters)
        return self

    def executemany(self, query: str, parameters: Any) -> "WriterTransactionConnection":
        self._connection.executemany(query, parameters)
        return self

    def commit(self) -> "WriterTransactionConnection":
        return self

    def rollback(self) -> "WriterTransactionConnection":
        return self

    def close(self) -> None:
        return None

    def __enter__(self) -> "WriterTransactionConnection":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def connect_observational(database: Path) -> ObservationalConnection:
    """Return a capability that opens only short-lived physical read-only handles."""
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(f"DuckDB database does not exist: {path}")
    return ObservationalConnection(path)


@contextmanager
def _database_access_lock(database: Path, *, timeout_seconds: float = 120.0) -> Iterator[None]:
    """Serialize cross-process DuckDB opens without granting a write capability."""
    resolved = str(Path(database).resolve()).casefold()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        name = "Local\\BharatTerminal-DuckDB-" + hashlib.sha256(
            resolved.encode("utf-8")
        ).hexdigest()
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
        wait_result = kernel32.WaitForSingleObject(handle, max(1, int(timeout_seconds * 1000)))
        if wait_result not in (0x00000000, 0x00000080):
            kernel32.CloseHandle(handle)
            if wait_result == 0x00000102:
                raise TimeoutError("timed out waiting for DuckDB access lock")
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        try:
            yield
        finally:
            kernel32.ReleaseMutex(handle)
            kernel32.CloseHandle(handle)
        return

    lock_path = Path(database).resolve().with_suffix(Path(database).suffix + ".access.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def open_read_connection(database: Path | None = None) -> ObservationalConnection:
    """Return the only connection capability available to normal callers."""
    return connect_observational(Path(database or DB_PATH))


def get_connection(*, read_only: bool = True) -> ObservationalConnection:
    """Deprecated observational-only alias retained for explicit bypass denial tests."""
    del read_only
    return open_read_connection()


def writer_connection() -> WriterTransactionConnection:
    """Return the active service transaction or deny unowned write authority."""
    active = _ACTIVE_WRITER_CONNECTION.get()
    if active is None:
        raise PermissionError("DuckDB write authority belongs to WriterService")
    return WriterTransactionConnection(active)


@contextmanager
def _writer_transaction_scope(
    connection: duckdb.DuckDBPyConnection,
) -> Iterator[WriterTransactionConnection]:
    """Internal service capability. Never exposed to request/collector code."""
    token: Token[duckdb.DuckDBPyConnection | None] = _ACTIVE_WRITER_CONNECTION.set(connection)
    try:
        yield WriterTransactionConnection(connection)
    finally:
        _ACTIVE_WRITER_CONNECTION.reset(token)


def _create_schema_objects_at(path: Path) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol      TEXT    NOT NULL,
                date        DATE    NOT NULL,
                open        DOUBLE  NOT NULL,
                high        DOUBLE  NOT NULL,
                low         DOUBLE  NOT NULL,
                close       DOUBLE  NOT NULL,
                volume      BIGINT  NOT NULL,
                source      TEXT    NOT NULL,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        con.execute(DECISION_ELIGIBLE_OHLCV_VIEW_SQL)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fo_eod (
                symbol      TEXT    NOT NULL,
                expiry      DATE    NOT NULL,
                strike      DOUBLE  NOT NULL,
                option_type TEXT    NOT NULL,
                date        DATE    NOT NULL,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      BIGINT,
                oi          BIGINT,
                oi_change   BIGINT,
                source      TEXT    NOT NULL DEFAULT 'UNKNOWN',
                source_format TEXT  NOT NULL DEFAULT 'unknown',
                price_quality TEXT  NOT NULL DEFAULT 'UNKNOWN',
                PRIMARY KEY (symbol, expiry, strike, option_type, date)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS iv_history (
                symbol      TEXT    NOT NULL,
                expiry      DATE    NOT NULL,
                strike      DOUBLE  NOT NULL,
                option_type TEXT    NOT NULL,
                date        DATE    NOT NULL,
                iv          DOUBLE  NOT NULL,
                PRIMARY KEY (symbol, expiry, strike, option_type, date)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS signals_history (
                date        DATE    NOT NULL,
                signal_name TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                value       DOUBLE,
                PRIMARY KEY (date, signal_name, symbol)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_store (
                date        DATE    NOT NULL,
                symbol      TEXT    NOT NULL,
                features    TEXT    NOT NULL,
                PRIMARY KEY (date, symbol)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS instruments (
                token           TEXT        NOT NULL,
                symbol          TEXT        NOT NULL,
                name            TEXT,
                exchange        TEXT        NOT NULL,
                instrument_type TEXT,
                expiry          DATE,
                strike          DOUBLE,
                option_type     TEXT,
                lot_size        INTEGER,
                tick_size       DOUBLE,
                isin            TEXT,
                updated_at      TIMESTAMP   NOT NULL,
                PRIMARY KEY (token, exchange)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS intraday_5min (
                symbol      TEXT        NOT NULL,
                datetime    TIMESTAMP   NOT NULL,
                open        DOUBLE      NOT NULL,
                high        DOUBLE      NOT NULL,
                low         DOUBLE      NOT NULL,
                close       DOUBLE      NOT NULL,
                volume      BIGINT      NOT NULL,
                vwap        DOUBLE,
                PRIMARY KEY (symbol, datetime)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS gift_nifty_eod (
                date        DATE        PRIMARY KEY,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                premium     DOUBLE,
                source      TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id              TEXT        PRIMARY KEY,
                model_id        TEXT        NOT NULL,
                predicted_at    TIMESTAMP   NOT NULL,
                features_hash   TEXT,
                prediction      TEXT        NOT NULL,
                probabilities   TEXT        NOT NULL,
                confidence      REAL        NOT NULL,
                symbol          TEXT,
                data_as_of      TEXT,
                model_version   TEXT,
                artifact_hash   TEXT,
                feature_schema_version TEXT,
                release_version TEXT,
                validation_status TEXT,
                fallback_status TEXT,
                outcome         TEXT,
                outcome_at      TIMESTAMP,
                correct         INTEGER,
                outcome_error   REAL,
                brier_score     REAL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlc_forecasts (
                id                  BIGINT      PRIMARY KEY,
                symbol              TEXT        NOT NULL,
                forecast_date       DATE        NOT NULL,
                forecast_open       DOUBLE,
                forecast_high       DOUBLE,
                forecast_low        DOUBLE,
                forecast_close      DOUBLE,
                actual_open         DOUBLE,
                actual_high         DOUBLE,
                actual_low          DOUBLE,
                actual_close        DOUBLE,
                close_mae           DOUBLE,
                high_mae            DOUBLE,
                low_mae             DOUBLE,
                expected_range      DOUBLE,
                expected_range_pct  DOUBLE,
                confidence          TEXT,
                key_levels_json     TEXT,
                created_at          TIMESTAMP   NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tft_predictions (
                id              TEXT        PRIMARY KEY,
                symbol          TEXT        NOT NULL,
                predicted_at    TIMESTAMP   NOT NULL,
                return_p10_1d   DOUBLE,
                return_p50_1d   DOUBLE,
                return_p90_1d   DOUBLE,
                return_p50_3d   DOUBLE,
                return_p50_5d   DOUBLE,
                attention_json  TEXT,
                outcome_1d      DOUBLE,
                outcome_3d      DOUBLE,
                outcome_5d      DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prophet_state (
                symbol          TEXT        PRIMARY KEY,
                trained_at      TIMESTAMP   NOT NULL,
                trend_today     DOUBLE,
                residual_today  DOUBLE,
                residual_z      DOUBLE,
                forecast_7d     DOUBLE,
                forecast_30d    DOUBLE,
                regime          TEXT,
                model_pkl_path  TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS alpha_signals (
                signal_name     TEXT        NOT NULL,
                symbol          TEXT        NOT NULL,
                computed_at     TIMESTAMP   NOT NULL,
                ic_30d          DOUBLE,
                ic_90d          DOUBLE,
                ic_stability    DOUBLE,
                ic_decay_3d     DOUBLE,
                ic_decay_5d     DOUBLE,
                pca_factor      TEXT,
                PRIMARY KEY (signal_name, symbol)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS narratives (
                id              BIGINT      PRIMARY KEY,
                generated_at    TIMESTAMP   NOT NULL,
                symbol          TEXT        NOT NULL,
                brief           TEXT        NOT NULL,
                brief_type      TEXT,
                model           TEXT,
                signals_hash    TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_history (
                date            DATE        NOT NULL,
                symbol          TEXT        NOT NULL,
                hmm_state       INTEGER,
                state_label     TEXT,
                state_prob_0    DOUBLE,
                state_prob_1    DOUBLE,
                state_prob_2    DOUBLE,
                state_prob_3    DOUBLE,
                change_point    INTEGER     DEFAULT 0,
                PRIMARY KEY (date, symbol)
            )
            """
        )
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
        con.execute(
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
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS bif_history (
                date            DATE        NOT NULL,
                symbol          TEXT        NOT NULL,
                bms             DOUBLE,
                bms_label       TEXT,
                ngi             DOUBLE,
                ngi_direction   TEXT,
                gravity_center  DOUBLE,
                iad             DOUBLE,
                iad_label       TEXT,
                wyckoff_stage   TEXT,
                vrc_position    DOUBLE,
                vrc_label       TEXT,
                vrp             DOUBLE,
                nfve_fair       DOUBLE,
                nfve_diff       DOUBLE,
                nfve_valuation  TEXT,
                cds             DOUBLE,
                cds_label       TEXT,
                cds_confidence  DOUBLE,
                action          TEXT,
                anomaly_active  INTEGER     DEFAULT 0,
                computed_at     TIMESTAMP   NOT NULL,
                PRIMARY KEY (date, symbol)
            )
            """
        )
        con.execute(
            """
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
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                job_id          TEXT        PRIMARY KEY,
                strategy_name   TEXT        NOT NULL,
                symbol          TEXT        NOT NULL,
                from_date       DATE        NOT NULL,
                to_date         DATE        NOT NULL,
                initial_capital DOUBLE      NOT NULL,
                status          TEXT        NOT NULL,
                progress        DOUBLE      DEFAULT 0,
                stage           TEXT,
                started_at      TIMESTAMP   NOT NULL,
                completed_at    TIMESTAMP,
                error           TEXT,
                result_json     TEXT,
                groq_report     TEXT,
                net_return_pct  DOUBLE,
                sharpe_ratio    DOUBLE,
                max_drawdown    DOUBLE,
                win_rate        DOUBLE,
                total_trades    INTEGER,
                chronological_segment_return DOUBLE,
                validity_status TEXT NOT NULL DEFAULT 'INVALID_LEGACY_PRICE_SEMANTICS',
                price_policy_version TEXT,
                invalid_reason TEXT,
                mc_p50_return   DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_trades (
                job_id          TEXT        NOT NULL,
                trade_id        TEXT        NOT NULL,
                symbol          TEXT        NOT NULL,
                strategy        TEXT        NOT NULL,
                entry_date      DATE        NOT NULL,
                exit_date       DATE,
                legs_json       TEXT        NOT NULL,
                entry_prices    TEXT,
                exit_prices     TEXT,
                gross_pnl       DOUBLE,
                net_pnl         DOUBLE,
                costs           DOUBLE,
                regime_at_entry TEXT,
                bif_cds_entry   DOUBLE,
                iv_at_entry     DOUBLE,
                PRIMARY KEY (job_id, trade_id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_manifests (
                source_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                source_identity TEXT NOT NULL,
                expected_scope_json TEXT NOT NULL,
                observed_record_count BIGINT NOT NULL,
                ingestion_completed_at TIMESTAMP NOT NULL,
                validation_completed_at TIMESTAMP NOT NULL,
                completeness_status TEXT NOT NULL,
                downstream_commit_id TEXT NOT NULL,
                retention_decision TEXT NOT NULL DEFAULT 'RETAIN',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol
            ON ohlcv(symbol)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fo_symbol_expiry
            ON fo_eod(symbol, expiry)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_iv_symbol
            ON iv_history(symbol, expiry)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instr_sym
            ON instruments(symbol, exchange)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instr_name
            ON instruments(name, exchange)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_instr_type
            ON instruments(instrument_type, exchange)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intraday_sym
            ON intraday_5min(symbol, datetime)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fo_sym_date
            ON fo_eod(symbol, date)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fo_expiry
            ON fo_eod(expiry, symbol)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ml_model
            ON ml_predictions(model_id, predicted_at)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlc_forecasts_symbol_date
            ON ohlc_forecasts(symbol, forecast_date)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_narratives_symbol
            ON narratives(symbol, generated_at)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bif_sym_date
            ON bif_history(symbol, date)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backtest_runs_status
            ON backtest_runs(status, started_at)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backtest_trades_job
            ON backtest_trades(job_id)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ingestion_manifest_retention
            ON ingestion_manifests(retention_decision, validation_completed_at)
            """
        )
    finally:
        con.close()


def init_schema() -> None:
    ensure_dirs()
    from db.migrations import migrate_database

    migrate_database(DB_PATH, schema_initializer=_create_schema_objects_at)
