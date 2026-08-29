"""SQLite error and progress logger for idempotent data backfills."""

import sqlite3
from datetime import datetime
from typing import Any

from data.paths import ERROR_LOG_PATH
from security.log_scrubber import scrub_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS data_errors (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT     NOT NULL,
    source        TEXT     NOT NULL,
    symbol        TEXT,
    date_str      TEXT,
    error_type    TEXT     NOT NULL,
    error_msg     TEXT,
    fallback_used TEXT,
    resolved      INTEGER  DEFAULT 0
);
CREATE TABLE IF NOT EXISTS backfill_progress (
    stage         TEXT     NOT NULL,
    date_str      TEXT     NOT NULL,
    status        TEXT     NOT NULL,
    error         TEXT,
    updated_at    TEXT     NOT NULL,
    PRIMARY KEY (stage, date_str)
);
"""


class ErrorLogger:
    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(ERROR_LOG_PATH)) as con:
            con.executescript(SCHEMA)

    def _write_connection(self) -> sqlite3.Connection:
        self.initialize()
        return sqlite3.connect(str(ERROR_LOG_PATH))

    def _read_connection(self) -> sqlite3.Connection | None:
        if not ERROR_LOG_PATH.is_file():
            return None
        return sqlite3.connect(f"file:{ERROR_LOG_PATH.as_posix()}?mode=ro", uri=True)

    def log(
        self,
        source: str,
        error_type: str,
        error_msg: Any,
        symbol: str | None = None,
        date_str: str | None = None,
        fallback_used: str | None = None,
    ) -> None:
        with self._write_connection() as con:
            con.execute(
                """
                INSERT INTO data_errors
                (timestamp, source, symbol, date_str,
                 error_type, error_msg, fallback_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    datetime.now().isoformat(),
                    source,
                    symbol,
                    date_str,
                    error_type,
                    scrub_text(str(error_msg))[:1000],
                    fallback_used,
                ],
            )

    def mark(
        self,
        stage: str,
        date_str: str,
        status: str,
        error: Any | None = None,
    ) -> None:
        with self._write_connection() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO backfill_progress
                (stage, date_str, status, error, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    stage,
                    date_str,
                    status,
                    scrub_text(str(error))[:500] if error else None,
                    datetime.now().isoformat(),
                ],
            )

    def is_done(self, stage: str, date_str: str) -> bool:
        connection = self._read_connection()
        if connection is None:
            return False
        with connection as con:
            row = con.execute(
                """
                SELECT 1 FROM backfill_progress
                WHERE stage = ? AND date_str = ?
                  AND status IN ('downloaded', 'loaded', 'skipped')
                """,
                [stage, date_str],
            ).fetchone()
            return row is not None

    def get_failed(self, stage: str) -> list[str]:
        connection = self._read_connection()
        if connection is None:
            return []
        with connection as con:
            rows = con.execute(
                """
                SELECT date_str FROM backfill_progress
                WHERE stage = ? AND status = 'failed'
                ORDER BY date_str
                """,
                [stage],
            ).fetchall()
            return [str(row[0]) for row in rows]

    def get_recent_errors(
        self,
        limit: int = 20,
        unresolved_only: bool = False,
    ) -> list[dict[str, Any]]:
        where = "WHERE resolved = 0" if unresolved_only else ""
        connection = self._read_connection()
        if connection is None:
            return []
        with connection as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                SELECT * FROM data_errors
                {where}
                ORDER BY timestamp DESC LIMIT ?
                """,
                [limit],
            ).fetchall()
            return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        connection = self._read_connection()
        if connection is None:
            return {"total_errors": 0, "unresolved_errors": 0, "progress_by_stage": []}
        with connection as con:
            total = con.execute("SELECT COUNT(*) FROM data_errors").fetchone()[0]
            unresolved = con.execute(
                "SELECT COUNT(*) FROM data_errors WHERE resolved = 0"
            ).fetchone()[0]
            by_stage = con.execute(
                """
                SELECT stage, status, COUNT(*) as count
                FROM backfill_progress
                GROUP BY stage, status
                ORDER BY stage, status
                """
            ).fetchall()
            return {
                "total_errors": total,
                "unresolved_errors": unresolved,
                "progress_by_stage": [
                    {"stage": row[0], "status": row[1], "count": row[2]}
                    for row in by_stage
                ],
            }


_logger = ErrorLogger()
