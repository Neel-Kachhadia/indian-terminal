from pathlib import Path
from threading import Lock

import duckdb

DB_PATH = Path.home() / ".indian-terminal" / "data" / "terminal.db"
DB_WRITE_LOCK = Lock()


def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def init_schema() -> None:
    con = get_connection()
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
    finally:
        con.close()
