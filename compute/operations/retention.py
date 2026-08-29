from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

from data.paths import DATA_DIR, EQUITY_BHAV_DIR, FO_BHAV_DIR, QUARANTINE_DIR
from db.duckdb_client import writer_connection
from db.writer_lease import require_writer_lease


RAW_BHAV_RETENTION_DAYS = 90
AUTOMATIC_RAW_INPUT_DELETION_ENABLED = False
QUARANTINE_RETENTION_DAYS = 180
SIGNAL_RETENTION_DAYS = 7 * 366
PREDICTION_RETENTION_DAYS = 2 * 366
TRAINING_RETENTION_DAYS = 2 * 366
FORECAST_RETENTION_DAYS = 2 * 366
BACKTEST_RETENTION_DAYS = 2 * 366
BACKTEST_MINIMUM_RUNS = 500
_DATE_TOKEN = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
RETENTION_STATE_PATH = DATA_DIR / "retention-state.json"


def apply_runtime_retention(
    database: Path,
    *,
    now: dt.datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    instant = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    if not force and _ran_today(instant):
        return {"skipped": True, "reason": "retention already completed today"}
    require_writer_lease(database)
    database_counts = apply_database_retention(database, now=instant)
    raw_counts = prune_loaded_bhavcopy_files(database, now=instant)
    quarantine_count = prune_files_older_than(
        QUARANTINE_DIR,
        instant - dt.timedelta(days=QUARANTINE_RETENTION_DAYS),
    )
    report = {
        "skipped": False,
        "database_rows_removed": database_counts,
        "raw_bhav_files_removed": raw_counts,
        "raw_input_deletion": "DISABLED_PENDING_COMPLETENESS_MANIFEST",
        "quarantine_files_removed": quarantine_count,
    }
    RETENTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RETENTION_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"completed_at": instant.isoformat(), "report": report}, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, RETENTION_STATE_PATH)
    return report


def apply_database_retention(
    database: Path,
    *,
    now: dt.datetime | None = None,
) -> dict[str, int]:
    instant = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    cutoffs = {
        "signals_history": ("date", instant - dt.timedelta(days=SIGNAL_RETENTION_DAYS)),
        "ml_predictions": ("predicted_at", instant - dt.timedelta(days=PREDICTION_RETENTION_DAYS)),
        "ml_training_log": ("trained_at", instant - dt.timedelta(days=TRAINING_RETENTION_DAYS)),
        "ohlc_forecasts": ("forecast_date", instant - dt.timedelta(days=FORECAST_RETENTION_DAYS)),
    }
    counts: dict[str, int] = {}
    connection = writer_connection()
    try:
        tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
        for table, (column, cutoff) in cutoffs.items():
            if table not in tables:
                continue
            cutoff_text = (
                cutoff.date().isoformat()
                if column in {"date", "forecast_date"}
                else cutoff.isoformat()
            )
            cast = "DATE" if column in {"date", "forecast_date"} else "TIMESTAMP"
            counts[table] = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} < CAST(? AS {cast})",
                    [cutoff_text],
                ).fetchone()[0]
            )
            connection.execute(
                f"DELETE FROM {table} WHERE {column} < CAST(? AS {cast})",
                [cutoff_text],
            )

        if {"backtest_runs", "backtest_trades"}.issubset(tables):
            cutoff = (instant - dt.timedelta(days=BACKTEST_RETENTION_DAYS)).isoformat()
            stale_ids = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT job_id FROM backtest_runs
                    WHERE completed_at < CAST(? AS TIMESTAMP)
                      AND status IN ('completed', 'failed')
                      AND job_id NOT IN (
                          SELECT job_id FROM backtest_runs
                          WHERE status IN ('completed', 'failed')
                          ORDER BY completed_at DESC NULLS LAST
                          LIMIT ?
                      )
                    """,
                    [cutoff, BACKTEST_MINIMUM_RUNS],
                ).fetchall()
            ]
            counts["backtest_runs"] = len(stale_ids)
            if stale_ids:
                placeholders = ", ".join("?" for _ in stale_ids)
                connection.execute(
                    f"DELETE FROM backtest_trades WHERE job_id IN ({placeholders})",
                    stale_ids,
                )
                connection.execute(
                    f"DELETE FROM backtest_runs WHERE job_id IN ({placeholders})",
                    stale_ids,
                )
    finally:
        connection.close()
    return counts


def prune_loaded_bhavcopy_files(
    database: Path,
    *,
    now: dt.datetime | None = None,
) -> dict[str, int]:
    del database, now
    counts = {"equity": 0, "fo": 0}
    return counts


def prune_files_older_than(root: Path, cutoff: dt.datetime) -> int:
    if not root.is_dir():
        return 0
    cutoff_utc = cutoff.astimezone(dt.timezone.utc)
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        modified = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        if modified >= cutoff_utc:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _date_from_filename(path: Path) -> dt.date | None:
    match = _DATE_TOKEN.search(path.stem)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _ran_today(instant: dt.datetime) -> bool:
    if not RETENTION_STATE_PATH.is_file():
        return False
    try:
        payload = json.loads(RETENTION_STATE_PATH.read_text(encoding="utf-8"))
        completed = dt.datetime.fromisoformat(str(payload["completed_at"]))
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=dt.timezone.utc)
        return completed.astimezone(dt.timezone.utc).date() == instant.date()
    except (OSError, ValueError, KeyError, TypeError):
        return False
