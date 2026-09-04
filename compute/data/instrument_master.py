"""
Angel One instrument master.

Downloaded daily, cached when parquet support is available, and loaded into
DuckDB for token, lot-size, and F&O universe lookups.
"""

from datetime import datetime
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import requests

from data.error_logger import _logger as elog
from data.index_aliases import CANONICAL_INDEX_SYMBOLS, INDEX_FEED_ALIASES
from data.instrument_schema import (
    INSTRUMENT_SCHEMA_VERSION,
    InstrumentSchemaError,
    canonicalize_instrument_frame,
    read_instrument_frame,
)
from data.paths import INSTRUMENTS_CACHE
from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read
from operations.storage import require_free_space

MASTER_URL = (
    "https://margincalculator.angelone.in/"
    "OpenAPI_File/files/OpenAPIScripMaster.json"
)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS instruments (
    schema_version  INTEGER     NOT NULL,
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
);
CREATE TABLE IF NOT EXISTS instrument_master_state (
    singleton       INTEGER     PRIMARY KEY CHECK (singleton = 1),
    cache_sha256    TEXT        NOT NULL,
    schema_version  INTEGER     NOT NULL,
    row_count       BIGINT      NOT NULL,
    updated_at      TIMESTAMP   NOT NULL
);
"""

CREATE_STAGING_TABLE = """
DROP TABLE IF EXISTS instruments_next;
CREATE TABLE instruments_next (
    schema_version  INTEGER     NOT NULL,
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
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_instr_sym
    ON instruments(symbol, exchange);
CREATE INDEX IF NOT EXISTS idx_instr_name
    ON instruments(name, exchange);
CREATE INDEX IF NOT EXISTS idx_instr_type
    ON instruments(instrument_type, exchange);
"""


class InstrumentMaster:
    def resolve_live_feed_universe(self) -> list[dict[str, Any]]:
        """Return canonical NSE cash equities plus supported indices for one feed."""
        result = resolve_read(ReadOperation.RESOLVE_LIVE_FEED_UNIVERSE, {})
        rows = list(result.get("instruments") or [])
        symbols = [str(row.get("symbol") or "") for row in rows]
        tokens = [str(row.get("token") or "") for row in rows]
        if not rows or len(symbols) != len(set(symbols)) or len(tokens) != len(set(tokens)):
            raise InstrumentSchemaError("live feed universe contains missing or duplicate identities")
        if any(not token.isascii() or not token.isdigit() for token in tokens):
            raise InstrumentSchemaError("live feed universe contains malformed token")
        return rows

    def resolve_index_feed(self, symbol: str) -> dict[str, Any]:
        normalized = "".join(symbol.strip().upper().split())
        if normalized not in INDEX_FEED_ALIASES:
            raise InstrumentSchemaError("unsupported index feed symbol")
        aliases = sorted(INDEX_FEED_ALIASES[normalized])
        # Proactively dispatches against this process's own writer connection
        # when it owns one, else routes through the cross-process typed read
        # queue -- never a raw read against a file another process may hold
        # as writer authority. See db.write_service.resolve_read().
        result = resolve_read(ReadOperation.RESOLVE_INDEX_INSTRUMENT, {"symbol": normalized})
        rows = [
            (
                row["schema_version"],
                row["token"],
                row["symbol"],
                row["name"],
                row["exchange"],
                row["instrument_type"],
            )
            for row in result["rows"]
        ]
        tokens = {str(row[1]).strip() for row in rows if str(row[1]).strip()}
        if len(tokens) != 1:
            raise InstrumentSchemaError(f"{normalized} index token is missing or ambiguous")
        row = rows[0]
        token = str(row[1]).strip()
        if not token.isascii() or not token.isdigit():
            raise InstrumentSchemaError(f"{normalized} index token is malformed")
        canonical_candidates = [
            symbol
            for symbol in CANONICAL_INDEX_SYMBOLS
            if symbol in INDEX_FEED_ALIASES.get(normalized, frozenset())
        ]
        if len(canonical_candidates) != 1:
            raise InstrumentSchemaError(f"{normalized} has no unambiguous canonical symbol")
        canonical = canonical_candidates[0]
        return {
            "schema_version": int(row[0]),
            "token": token,
            "symbol": canonical,
            "canonical_symbol": str(row[2]),
            "name": str(row[3] or ""),
            "exchange": str(row[4]),
            "instrument_type": str(row[5]),
        }

    def refresh(self, force: bool = False) -> int:
        """
        Download and load instrument master.

        Skips if parquet cache is less than 24 hours old and force=False.
        Returns number of rows loaded; returns 0 when cache is fresh.
        """
        if not force and INSTRUMENTS_CACHE.exists():
            age = datetime.now().timestamp() - INSTRUMENTS_CACHE.stat().st_mtime
            if age < 86400:
                try:
                    cached = read_instrument_frame(INSTRUMENTS_CACHE)
                    cache_sha256 = self._file_sha256(INSTRUMENTS_CACHE)
                    if self._database_matches_cache(cache_sha256, len(cached)):
                        return 0
                    # A process can die after atomically publishing the cache
                    # but before its enclosing DuckDB transaction commits.
                    # The explicit cache identity makes that crash window
                    # detectable and lets the next run repair the DB from the
                    # already-validated cache instead of incorrectly treating
                    # cache freshness as proof that both stores agree.
                    self._load_to_duckdb(cached, cache_sha256=cache_sha256)
                    return len(cached)
                except (OSError, ValueError):
                    pass

        try:
            response = requests.get(MASTER_URL, timeout=30)
            response.raise_for_status()
            df = pd.DataFrame(response.json())
        except Exception as error:
            elog.log("instrument_master", "download_failed", str(error))
            if not INSTRUMENTS_CACHE.exists():
                raise
            df = read_instrument_frame(INSTRUMENTS_CACHE)

        df = self._normalise(df)
        cache_sha256 = self._write_cache(df)
        self._load_to_duckdb(df, cache_sha256=cache_sha256)
        return len(df)

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        return canonicalize_instrument_frame(df)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _write_cache(self, df: pd.DataFrame) -> str:
        temporary: Path | None = None
        try:
            INSTRUMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            estimated = max(1024 * 1024, int(df.memory_usage(index=True, deep=True).sum()))
            require_free_space(INSTRUMENTS_CACHE, estimated * 2, "instrument master cache")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{INSTRUMENTS_CACHE.name}.",
                suffix=".tmp",
                dir=INSTRUMENTS_CACHE.parent,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            df.to_parquet(temporary, index=False)
            stored = read_instrument_frame(temporary)
            if len(stored) != len(df):
                raise InstrumentSchemaError("instrument cache row count changed during publication")
            cache_sha256 = self._file_sha256(temporary)
            # "rb" (read-only) here reproducibly raises OSError: [Errno 9]
            # Bad file descriptor on Windows -- os.fsync()/FlushFileBuffers
            # needs a handle opened with write access to succeed, even
            # though nothing is actually written through it; this was
            # surfacing on every real installed startup as "Instrument
            # master refresh failed", not a timing-dependent race. "r+b"
            # (the file already exists from df.to_parquet() above, so this
            # never creates or truncates it) is the minimal write-capable
            # mode that fixes it.
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, INSTRUMENTS_CACHE)
            temporary = None
            return cache_sha256
        except Exception as error:
            elog.log("instrument_master", "cache_write_failed", str(error))
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _database_matches_cache(self, cache_sha256: str, row_count: int) -> bool:
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                exists = con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'main' AND table_name = 'instrument_master_state'"
                ).fetchone()
                if not exists or int(exists[0]) != 1:
                    return False
                row = con.execute(
                    "SELECT cache_sha256, schema_version, row_count, "
                    "(SELECT COUNT(*) FROM instruments) "
                    "FROM instrument_master_state WHERE singleton = 1"
                ).fetchone()
                return bool(
                    row
                    and str(row[0]) == cache_sha256
                    and int(row[1]) == INSTRUMENT_SCHEMA_VERSION
                    and int(row[2]) == row_count
                    and int(row[3]) == row_count
                )
            finally:
                con.close()

    def _load_to_duckdb(self, df: pd.DataFrame, *, cache_sha256: str) -> None:
        cols = [
            "schema_version",
            "token",
            "symbol",
            "name",
            "exchange",
            "instrument_type",
            "expiry",
            "strike",
            "option_type",
            "lot_size",
            "tick_size",
            "isin",
            "updated_at",
        ]
        df_db = df[cols]
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                con.execute("BEGIN")
                con.execute(CREATE_TABLE)
                con.register("df_instr", df_db)
                # Instrument master is a complete snapshot. Build an empty
                # constrained replacement first, then swap it atomically.
                # Deleting from the live PK-indexed table and issuing
                # INSERT OR REPLACE for the same keys triggered a native
                # DuckDB fast-fail (0xC0000409) on the production-sized
                # snapshot; staging also avoids DuckDB's same-transaction
                # delete/reinsert uniqueness hazard.
                con.execute(CREATE_STAGING_TABLE)
                con.execute(
                    """
                    INSERT INTO instruments_next
                    (schema_version, token, symbol, name, exchange, instrument_type, expiry,
                     strike, option_type, lot_size, tick_size, isin, updated_at)
                    SELECT schema_version, token, symbol, name, exchange, instrument_type, expiry,
                           strike, option_type, lot_size, tick_size, isin, updated_at
                    FROM df_instr
                    WHERE token <> '' AND exchange <> ''
                    """
                )
                staged = con.execute("SELECT COUNT(*) FROM instruments_next").fetchone()
                if not staged or int(staged[0]) != len(df_db):
                    raise InstrumentSchemaError(
                        "instrument DB row count changed during staging"
                    )
                con.execute("DROP TABLE instruments")
                con.execute("ALTER TABLE instruments_next RENAME TO instruments")
                con.execute(CREATE_INDEXES)
                con.execute(
                    """
                    INSERT OR REPLACE INTO instrument_master_state
                    (singleton, cache_sha256, schema_version, row_count, updated_at)
                    SELECT 1, ?, ?, COUNT(*), CURRENT_TIMESTAMP FROM instruments
                    """,
                    [cache_sha256, INSTRUMENT_SCHEMA_VERSION],
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.close()

    def get_fno_stocks(self) -> list[str]:
        result = resolve_read(ReadOperation.GET_FNO_UNDERLYINGS, {})
        return result["underlyings"]

    def get_token(
        self,
        symbol: str,
        exchange: str = "NSE",
        instrument_type: str = "EQ",
        expiry: Any | None = None,
        strike: float | None = None,
        option_type: str | None = None,
    ) -> str | None:
        normalized = symbol.strip().upper()
        normalized_exchange = exchange.strip().upper()
        normalized_type = instrument_type.strip().upper()

        payload: dict[str, Any] = {
            "symbol": normalized,
            "exchange": normalized_exchange,
            "instrument_type": normalized_type,
        }
        if expiry:
            payload["expiry"] = str(expiry)
        if strike is not None:
            payload["strike"] = strike
        if option_type:
            payload["option_type"] = option_type.strip().upper()
        result = resolve_read(ReadOperation.GET_INSTRUMENT_TOKEN, payload)
        return result["token"]

    def get_lot_size(self, symbol: str, expiry: str | None = None) -> int:
        normalized = symbol.strip().upper()
        payload: dict[str, Any] = {"symbol": normalized}
        if expiry:
            payload["expiry"] = _normalise_expiry(expiry)
        result = resolve_read(ReadOperation.GET_LOT_SIZE, payload)
        lot_size = result.get("lot_size")
        if not isinstance(lot_size, int) or lot_size <= 0:
            suffix = f" for expiry {expiry}" if expiry else ""
            raise InstrumentSchemaError(f"lot size unavailable for {normalized}{suffix}")
        return lot_size


master = InstrumentMaster()


def _normalise_expiry(value: str) -> str:
    text = str(value or "").strip()
    for pattern in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    raise InstrumentSchemaError(f"invalid expiry: {value!r}")
