"""
Status dashboard for the smart collector.

Run:
    python auto_collector/monitor.py

Operator diagnostic. Database status uses the same bounded typed-read queue
as production callers, so running the monitor can never create an independent
DuckDB handle while another process owns the canonical file.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from auto_collector.supervision import status_snapshot
from data.paths import DATA_DIR, DUCKDB_PATH
from db.write_service import ReadOperation, resolve_read

LOG_DB = DATA_DIR / "collector_log.db"
DB_PATH = DUCKDB_PATH
HEARTBEAT_FILE = DATA_DIR / "collector_heartbeat.json"


def show() -> int:
    print()
    print("=" * 72)
    print("BHARATTERMINAL - SMART COLLECTOR STATUS")
    print(f"Report time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)
    show_recent_activity()
    show_database_state()
    show_storage()
    show_process()
    print()
    print(f"Full log: {DATA_DIR / 'collector.log'}")
    print("=" * 72)
    print()
    return 0


def show_recent_activity() -> None:
    print("\nRECENT COLLECTION ACTIVITY (last 7 days)")
    print("-" * 72)
    if not LOG_DB.exists():
        print("  collector_log.db not found yet")
        return
    try:
        with sqlite3.connect(str(LOG_DB)) as con:
            rows = con.execute(
                """
                SELECT task,
                       COUNT(*) AS total,
                       SUM(CASE WHEN status = 'SUCCEEDED' THEN 1 ELSE 0 END) AS ok,
                       SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS fail,
                       MAX(started_at) AS last_run,
                       MAX(CASE WHEN status = 'SUCCEEDED' THEN details END) AS last_details
                FROM collector_runs
                WHERE started_at >= datetime('now', '-7 days')
                GROUP BY task
                ORDER BY last_run DESC
                """
            ).fetchall()
    except sqlite3.Error as error:
        print(f"  log DB error: {error}")
        return

    if not rows:
        print("  no runs in last 7 days")
        return

    fmt = "{:<24} {:>5} {:>4} {:>4}  {:<19}  {}"
    print(fmt.format("TASK", "RUNS", "OK", "FAIL", "LAST RUN", "DETAILS"))
    for task, total, ok, fail, last_run, details in rows:
        print(
            fmt.format(
                task,
                total or 0,
                ok or 0,
                fail or 0,
                str(last_run or "")[:19],
                str(details or "")[:32],
            )
        )


def show_database_state() -> None:
    print("\nDATABASE STATE")
    print("-" * 72)
    try:
        result = resolve_read(ReadOperation.LOAD_DATA_SUMMARY_STATS, {})
    except Exception as error:
        print(f"  DuckDB unavailable: {error}")
        return
    rows = (
        ("ohlcv", result["eq"][0], result["eq"][2]),
        ("fo_eod", result["fo"][0], result["fo"][2]),
        ("signals_history", result["sig"][0], result["sig"][1]),
        ("bif_history", result["bif"][0], result["bif"][1]),
        ("fii_dii_daily", result["fii"][0], result["fii"][1]),
        ("ohlc_forecasts", result["ohlc_fc"][0], result["ohlc_fc"][1]),
    )
    for name, count, latest in rows:
        print(f"  {name:<24} {int(count or 0):>12,} rows  latest: {latest}")


def show_storage() -> None:
    print("\nSTORAGE")
    print("-" * 72)
    try:
        import shutil

        free_gb = shutil.disk_usage(DATA_DIR).free / (1024**3)
        db_gb = DB_PATH.stat().st_size / (1024**3) if DB_PATH.exists() else 0.0
        print(f"  DuckDB:  {db_gb:.2f} GB")
        print(f"  Data volume free: {free_gb:.1f} GB")
    except Exception as error:
        print(f"  storage check failed: {error}")


def show_process() -> None:
    print("\nCOLLECTOR PROCESS")
    print("-" * 72)
    running = collector_process_running()
    if running:
        print("  Collector process appears to be running")
    else:
        print("  Collector may not be running")
        print("  Start: schtasks /Run /TN BharatTerminal_SmartCollector")


def collector_process_running() -> bool:
    return bool(status_snapshot(LOG_DB).get("collector_running"))


if __name__ == "__main__":
    raise SystemExit(show())
