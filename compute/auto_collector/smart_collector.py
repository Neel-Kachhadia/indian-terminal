"""
Standalone smart data collector.

Runs as a background process. Does not import FastAPI, Tauri, or Redis app
state. It only uses local compute modules and the configured application-data
directory.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


COMPUTE_DIR = Path(__file__).resolve().parents[1]
if str(COMPUTE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPUTE_DIR))

from data.paths import DATA_DIR, DUCKDB_PATH, FO_BHAV_DIR, INSTRUMENTS_CACHE, ensure_dirs
from data.exchange_calendar import is_trading_day, now_ist
from db.write_service import (
    ReadOperation,
    WriteOperation,
    WriterService,
    queue_path_for,
    submit_read,
    submit_write,
    writer_service_status,
)
from auto_collector.supervision import CollectorSupervisor, ensure_schema
from operations.logging_config import configure_structured_logging


LOG_DB = DATA_DIR / "collector_log.db"
LOG_FILE = DATA_DIR / "collector.log"
LOCK_FILE = DATA_DIR / "collector.lock"
HEARTBEAT_FILE = DATA_DIR / "collector_heartbeat.json"
MODELS_DIR = DATA_DIR / "ml" / "saved_models"
CHECK_INTERVAL_SECONDS = 30 * 60
MAX_BACK_DAYS = 10
_COLLECTOR_READ_DATASETS: dict[str, tuple[str, str]] = {
    "bif_history": ("bif_history", "date"),
    "fii_dii_daily": ("fii_dii_daily", "date"),
    "fo_eod": ("fo_eod", "date"),
    "ohlcv": ("ohlcv", "date"),
    "signals_history": ("signals_history", "date"),
}

log = logging.getLogger("collector")
_LOCK_HANDLE: Any | None = None
_SUPERVISOR: CollectorSupervisor | None = None
_LOCAL_WRITER: WriterService | None = None


def setup_logging() -> None:
    ensure_dirs()
    configure_structured_logging(LOG_FILE)


def acquire_single_instance() -> bool:
    global _LOCK_HANDLE
    ensure_dirs()
    handle = LOCK_FILE.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return False
    _LOCK_HANDLE = handle
    atexit.register(release_single_instance)
    return True


def release_single_instance() -> None:
    global _LOCK_HANDLE
    if _LOCK_HANDLE is None:
        return
    try:
        import msvcrt

        _LOCK_HANDLE.seek(0)
        msvcrt.locking(_LOCK_HANDLE.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    _LOCK_HANDLE.close()
    _LOCK_HANDLE = None


def write_heartbeat(state: str, pending_count: int | None = None) -> None:
    supervisor = _require_supervisor()
    supervisor.set_activity(state, pending_count)


def ist_now() -> datetime:
    return now_ist()


def ist_hour() -> float:
    now = ist_now()
    return now.hour + now.minute / 60.0


def ist_today() -> date:
    return ist_now().date()


def ensure_log_db() -> None:
    ensure_dirs()
    ensure_schema(LOG_DB)


def task_done(task: str, for_date: date) -> bool:
    try:
        with sqlite3.connect(str(LOG_DB)) as con:
            row = con.execute(
                """
                SELECT 1 FROM collector_runs
                WHERE task = ? AND date = ? AND status = 'SUCCEEDED'
                """,
                [task, for_date.isoformat()],
            ).fetchone()
            return row is not None
    except sqlite3.Error:
        return False


def days_since_last_success(task: str) -> int | None:
    try:
        with sqlite3.connect(str(LOG_DB)) as con:
            row = con.execute(
                "SELECT MAX(date) FROM collector_runs WHERE task = ? AND status = 'SUCCEEDED'",
                [task],
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    last = _coerce_date(row[0])
    return (ist_today() - last).days if last else None


def log_start(task: str, for_date: date) -> None:
    supervisor = _require_supervisor()
    supervisor.queue_job(task, for_date)
    supervisor.claim_job(task, for_date)


def log_success(task: str, for_date: date, details: str, duration_s: float) -> None:
    _require_supervisor().finish_job(task, for_date, True, details, duration_s)


def log_failure(task: str, for_date: date, details: str, duration_s: float) -> None:
    _require_supervisor().finish_job(task, for_date, False, details, duration_s)


def _require_supervisor() -> CollectorSupervisor:
    if _SUPERVISOR is None:
        raise RuntimeError("collector supervisor is not active")
    return _SUPERVISOR


def start_supervision() -> bool:
    global _SUPERVISOR
    ensure_log_db()
    supervisor = CollectorSupervisor(LOG_DB, HEARTBEAT_FILE)
    if not supervisor.start():
        return False
    _SUPERVISOR = supervisor
    return True


def stop_supervision() -> None:
    global _SUPERVISOR
    if _SUPERVISOR is not None:
        _SUPERVISOR.stop()
        _SUPERVISOR = None


def _collector_read_dataset(dataset: str) -> tuple[str, str]:
    try:
        return _COLLECTOR_READ_DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"collector dataset is not allowlisted: {dataset!r}") from exc


def latest_date_in(dataset: str, symbol: str | None = None) -> date | None:
    _collector_read_dataset(dataset)  # local allowlist check; server re-checks independently
    payload: dict[str, Any] = {"dataset": dataset}
    if symbol is not None:
        payload["symbol"] = symbol
    result = submit_read(ReadOperation.LATEST_DATE_IN, payload)
    raw_date = result.get("date")
    if raw_date is None:
        return None
    parsed = _coerce_date(raw_date)
    if parsed is None:
        raise ValueError(f"writer returned an invalid latest date for {dataset}: {raw_date!r}")
    return parsed


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def missing_trading_days(since: date | None, max_back: int = MAX_BACK_DAYS) -> list[date]:
    today = ist_today()
    missing: list[date] = []
    if since is None:
        # Fresh install / never-run dataset: no lower bound to anchor a forward
        # walk from, so seed with the most recent trading days.
        current = today
        checked = 0
        while checked < 30 and len(missing) < max_back:
            if is_trading_day(current):
                missing.append(current)
            current -= timedelta(days=1)
            checked += 1
        return sorted(missing)
    # Walk forward from the oldest gap first. Walking backward from `today` (as
    # this used to) silently and permanently drops any gap older than
    # `max_back` trading days: each cycle only ever discovers the most recent
    # `max_back` missing days, `since` then advances to that batch's newest
    # date, and the older days never re-enter any future scan. A laptop off
    # for more than ~2 calendar weeks (~10 trading days) would never recover
    # the earliest days of that gap. Walking forward from `since` instead
    # means every cycle closes the oldest outstanding gap first, so downtime
    # of any length is fully backfilled over successive cycles.
    current = since + timedelta(days=1)
    checked = 0
    while current <= today and len(missing) < max_back and checked < 3650:
        if is_trading_day(current):
            missing.append(current)
        current += timedelta(days=1)
        checked += 1
    return missing


def instrument_master_age_hours() -> float:
    if not INSTRUMENTS_CACHE.exists():
        return 999.0
    return (datetime.now() - datetime.fromtimestamp(INSTRUMENTS_CACHE.stat().st_mtime)).total_seconds() / 3600.0


def model_age_days() -> int:
    if not MODELS_DIR.exists():
        return 999
    ages = [
        (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        for path in MODELS_DIR.rglob("*.skops")
    ]
    return min(ages) if ages else 999


def row_count_for_date(dataset: str, for_date: date) -> int:
    _collector_read_dataset(dataset)  # local allowlist check; server re-checks independently
    result = submit_read(
        ReadOperation.ROW_COUNT_FOR_DATE,
        {"dataset": dataset, "date": for_date.isoformat()},
    )
    if "count" not in result:
        raise ValueError(f"writer returned no row count for {dataset}")
    count = int(result["count"])
    if count < 0:
        raise ValueError(f"writer returned a negative row count for {dataset}: {count}")
    return count


def ensure_fii_dii_table() -> None:
    status = writer_service_status(queue_path_for(DUCKDB_PATH))
    if not status.get("live"):
        raise RuntimeError("canonical DuckDB writer service unavailable")


def run_instrument_master(_: date) -> str:
    rows = int(
        submit_write(WriteOperation.REFRESH_INSTRUMENT_MASTER, {"force": False})["rows"]
    )
    return f"rows={rows}"


def run_bhavcopy_equity(for_date: date) -> str:
    from data.nse_bhavcopy import NSEBhavcopy

    bhav = NSEBhavcopy()
    success, source = bhav.download_equity_day(for_date)
    if source == "holiday" and for_date == ist_today():
        return "not_available_yet"
    if source == "invalid_response":
        # Distinct from ordinary not-yet-published: the server answered but
        # the payload was garbage (zero-byte/HTML/corrupt archive). Must not
        # carry "not_available_yet" in the details string -- run_logged_task
        # keys its WAIT/FAIL presentation tag off that exact substring, and a
        # persistent parsing/validation break must surface as FAIL, not be
        # silently absorbed into the same quiet tag as a normal holiday wait.
        return f"data_validation_failed source={source}"
    if not success:
        return f"not_available_yet source={source}"
    if source in {"holiday", "pre_2024_skip"}:
        return f"{source}"

    loaded = int(
        submit_write(
            WriteOperation.LOAD_EQUITY_BHAVCOPY,
            {"from_date": for_date.isoformat()},
        )["rows"]
    )
    rows_on_date = row_count_for_date("ohlcv", for_date)
    if rows_on_date == 0 and for_date == ist_today():
        return "not_available_yet rows_on_date=0"
    return f"source={source} loaded={loaded} rows_on_date={rows_on_date}"


def run_bhavcopy_fo(for_date: date) -> str:
    from data.nse_bhavcopy import NSEBhavcopy

    bhav = NSEBhavcopy()
    success, source = bhav.download_fo_day(for_date)
    if source == "holiday" and for_date == ist_today():
        return "not_available_yet"
    if source == "invalid_response":
        return f"data_validation_failed source={source}"
    if not success:
        return f"not_available_yet source={source}"
    if source in {"holiday", "pre_2024_skip"}:
        return f"{source}"

    loaded = int(
        submit_write(
            WriteOperation.LOAD_FO_BHAVCOPY,
            {"from_date": for_date.isoformat()},
        )["rows"]
    )
    rows_on_date = row_count_for_date("fo_eod", for_date)
    if rows_on_date == 0 and for_date == ist_today():
        return "not_available_yet rows_on_date=0"
    return f"source={source} loaded={loaded} rows_on_date={rows_on_date}"


def run_fii_dii(for_date: date) -> str:
    from signals.market_signals import fetch_fii_dii

    payload = fetch_fii_dii()
    if not isinstance(payload, dict) or payload.get("error"):
        return f"not_available_yet error={payload.get('error') if isinstance(payload, dict) else 'bad payload'}"

    payload_date = _parse_market_date(str(payload.get("date") or "")) or for_date
    if payload_date < for_date:
        return f"not_available_yet latest={payload_date.isoformat()}"
    if payload_date > for_date:
        return f"historical_unavailable latest={payload_date.isoformat()}"

    row = {
        "date": payload_date.isoformat(),
        "fii_buy": _num(payload.get("fii_buy_cash")),
        "fii_sell": _num(payload.get("fii_sell_cash")),
        "fii_net": _num(payload.get("fii_net_cash")),
        "dii_buy": _num(payload.get("dii_buy_cash")),
        "dii_sell": _num(payload.get("dii_sell_cash")),
        "dii_net": _num(payload.get("dii_net_cash")),
        "fii_net_futures": _num(payload.get("fii_net_futures")),
        "raw_json": json.dumps(payload, sort_keys=True),
    }
    submit_write(WriteOperation.UPSERT_FII_DII, row)
    return (
        f"date={payload_date.isoformat()} "
        f"fii_net={row['fii_net']:.2f} dii_net={row['dii_net']:.2f}"
    )


def run_signals(_: date) -> str:
    result = submit_write(WriteOperation.BACKFILL_SIGNALS, {"symbol": "NIFTY"})
    return _compact_details(result)


def run_bif(_: date) -> str:
    result = submit_write(WriteOperation.COMPUTE_BIF, {"symbol": "NIFTY"})
    cds = result.get("cds", {}).get("cds")
    if isinstance(cds, (int, float)) and math.isfinite(float(cds)):
        return f"cds={float(cds):.1f}"
    return "cds=unavailable"


def run_ohlc_forecasts(_: date) -> str:
    symbols = forecast_symbols(limit=50)
    completed = 0
    errors = 0
    for symbol in symbols:
        try:
            submit_write(WriteOperation.RUN_OHLC_FORECASTS, {"symbol": symbol})
            completed += 1
        except Exception as error:
            errors += 1
            log.warning("OHLC forecast failed for %s: %s", symbol, error)
    if symbols and completed == 0:
        return f"not_available_yet symbols={completed} errors={errors}"
    return f"symbols={completed} errors={errors}"


def run_weekly_retrain(_: date) -> str:
    report = submit_write(WriteOperation.TRAIN_NIFTY_MODELS, {})
    models = report.get("models", {})
    trained = sum(1 for item in models.values() if isinstance(item, dict) and item.get("is_trained"))
    return f"models={trained} feature_rows={report.get('feature_rows')}"


def run_stock_training(_: date) -> str:
    result = submit_write(WriteOperation.TRAIN_STOCK_MODELS, {"force": False})
    return _compact_details(result)


TaskRunner = Callable[[date], str]
TASK_RUNNERS: dict[str, TaskRunner] = {
    "instrument_master": run_instrument_master,
    "bhavcopy_equity": run_bhavcopy_equity,
    "bhavcopy_fo": run_bhavcopy_fo,
    "fii_dii": run_fii_dii,
    "signals": run_signals,
    "bif": run_bif,
    "ohlc_forecasts": run_ohlc_forecasts,
    "weekly_retrain": run_weekly_retrain,
    "stock_training": run_stock_training,
}


def run_task(task: str, for_date: date) -> tuple[bool, str]:
    runner = TASK_RUNNERS.get(task)
    if runner is None:
        return False, f"unknown task: {task}"
    try:
        details = runner(for_date)
    except Exception as error:
        return False, str(error)
    # "data_validation_failed" (invalid/garbage response content, distinct
    # from an ordinary not-yet-published wait -- see run_bhavcopy_equity/fo)
    # must also count as incomplete here: it deliberately does NOT contain
    # "not_available_yet" so run_logged_task's WAIT/FAIL presentation tag
    # correctly renders it as a loud FAIL, but the task itself still isn't
    # done and must not be marked successful.
    if "not_available_yet" in details or "data_validation_failed" in details:
        return False, details
    return True, details


def get_pending_tasks() -> list[dict[str, Any]]:
    today = ist_today()
    now_hour = ist_hour()
    tasks: list[dict[str, Any]] = []

    if instrument_master_age_hours() >= 24 and not task_done("instrument_master", today):
        tasks.append(
            {
                "task": "instrument_master",
                "date": today,
                "reason": "instrument cache stale",
                "priority": 1,
            }
        )

    if now_hour >= 16.25:
        latest_eq = latest_date_in("ohlcv")
        for missing in missing_trading_days(latest_eq):
            if not task_done("bhavcopy_equity", missing):
                tasks.append(
                    {
                        "task": "bhavcopy_equity",
                        "date": missing,
                        "reason": f"missing equity {missing}",
                        "priority": 2,
                    }
                )

        latest_fo = latest_date_in("fo_eod")
        for missing in missing_trading_days(latest_fo):
            if not task_done("bhavcopy_fo", missing):
                tasks.append(
                    {
                        "task": "bhavcopy_fo",
                        "date": missing,
                        "reason": f"missing F&O {missing}",
                        "priority": 2,
                    }
                )

    if now_hour >= 18.0:
        try:
            ensure_fii_dii_table()
            latest_fii = latest_date_in("fii_dii_daily")
        except Exception as error:
            log.warning("FII/DII pending check skipped: %s", error)
            latest_fii = ist_today()
        for missing in missing_trading_days(latest_fii):
            if not task_done("fii_dii", missing):
                tasks.append(
                    {
                        "task": "fii_dii",
                        "date": missing,
                        "reason": f"missing FII/DII {missing}",
                        "priority": 3,
                    }
                )

    latest_eq = latest_date_in("ohlcv")
    latest_sig = latest_date_in("signals_history", "NIFTY")
    if latest_eq and (latest_sig is None or latest_sig < latest_eq):
        task_date = latest_eq
        if not task_done("signals", task_date):
            tasks.append(
                {
                    "task": "signals",
                    "date": task_date,
                    "reason": f"signals behind ohlcv {latest_eq}",
                    "priority": 4,
                }
            )

    latest_sig = latest_date_in("signals_history", "NIFTY")
    latest_bif = latest_date_in("bif_history", "NIFTY")
    if latest_sig and (latest_bif is None or latest_bif < latest_sig):
        task_date = latest_sig
        if not task_done("bif", task_date):
            tasks.append(
                {
                    "task": "bif",
                    "date": task_date,
                    "reason": f"BIF behind signals {latest_sig}",
                    "priority": 5,
                }
            )

    if now_hour >= 16.0 and not task_done("ohlc_forecasts", today):
        tasks.append(
            {
                "task": "ohlc_forecasts",
                "date": today,
                "reason": "daily OHLC forecasts",
                "priority": 6,
            }
        )

    is_sunday = today.weekday() == 6
    retrain_age_days = days_since_last_success("weekly_retrain")
    # `days_since_last_success` (the collector's own run ledger for this exact
    # task) is used instead of model_age_days() (the newest .skops mtime
    # anywhere under MODELS_DIR): the latter is a global min across every
    # model family, so any unrelated artifact write anywhere in the tree
    # (e.g. stock_training) resets it and masks weekly_retrain's own real
    # staleness -- reproduced live: a Sunday run FAILED, then model_age_days()
    # read back 0 (some other family's artifact was fresh) even though it had
    # genuinely been 10 days since weekly_retrain itself last succeeded, so
    # the catch-up condition never fired and would have silently waited a
    # full week for the next Sunday trigger instead of catching up sooner.
    retrain_catch_up = retrain_age_days is None or retrain_age_days > 7
    if (is_sunday or retrain_catch_up) and not task_done("weekly_retrain", today):
        tasks.append(
            {
                "task": "weekly_retrain",
                "date": today,
                "reason": "Sunday retrain"
                if is_sunday
                else f"weekly_retrain {retrain_age_days if retrain_age_days is not None else 'never'}d stale",
                "priority": 7,
            }
        )

    is_saturday = today.weekday() == 5
    stock_age_days = days_since_last_success("stock_training")
    stock_catch_up = stock_age_days is None or stock_age_days > 7
    if (is_saturday or stock_catch_up) and not task_done("stock_training", today):
        tasks.append(
            {
                "task": "stock_training",
                "date": today,
                "reason": "Saturday stock training"
                if is_saturday
                else f"stock training {stock_age_days if stock_age_days is not None else 'never'}d stale",
                "priority": 8,
            }
        )

    return sorted(tasks, key=lambda item: (item["priority"], item["date"], item["task"]))


def _console(tag: str, category: str = "", detail: str | None = None) -> dict[str, Any]:
    """Build the `extra=` payload consumed by `HumanConsoleFormatter`. Purely
    presentational -- the JSON file formatter ignores `console_meta`
    entirely, so this never changes what's durably logged."""
    meta: dict[str, Any] = {"tag": tag}
    if category:
        meta["category"] = category
    if detail:
        meta["detail"] = detail
    return {"console_meta": meta}


_TASK_CATEGORY = {
    "bhavcopy_equity": "EQUITY",
    "bhavcopy_fo": "F&O",
    "bhavcopy_index": "INDEX",
    "instrument_master": "MASTER",
    "fii_dii": "FII/DII",
    "signals": "SIGNALS",
    "ohlc_forecasts": "FORECAST",
    "weekly_retrain": "TRAIN",
    "stock_training": "TRAIN",
}


def _task_category(task: str) -> str:
    return _TASK_CATEGORY.get(task, task.upper()[:10])


class _GracefulStopRequested(BaseException):
    """Raised from the signal handler so an in-progress `time.sleep()` or
    task is interrupted promptly rather than waiting for the current sleep
    interval or task to finish on its own."""


def _install_graceful_stop_handler() -> None:
    def _handler(signum: int, _frame: object) -> None:
        raise _GracefulStopRequested()

    # CTRL_BREAK_EVENT is what job_supervisor.py (and Task Scheduler's own
    # stop path via that supervisor) delivers for a graceful stop. Without an
    # explicit handler, CPython's default action for SIGBREAK on Windows is
    # immediate termination with no chance to release the writer authority,
    # the collector lease, or the single-instance lock -- unlike SIGINT,
    # SIGBREAK is not auto-translated into a catchable KeyboardInterrupt.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, _handler)


def run_collector_forever() -> int:
    setup_logging()
    if not acquire_single_instance():
        log.info(
            "Another smart_collector.py instance is already running. Exiting.",
            extra=_console("SKIP", "START"),
        )
        return 2
    _install_graceful_stop_handler()
    try:
        if not start_supervision():
            log.error(
                "Another collector owns a live database lease. Exiting.",
                extra=_console("ERROR", "WRITER"),
            )
            return 2
        write_heartbeat("bootstrapping")
        log.info(
            "Collector bootstrap starting: instance=%s",
            _require_supervisor().instance_id,
            extra=_console("START", "START"),
        )
        bootstrap(logging_ready=True)
        write_heartbeat("started")
        _log_startup_banner()

        while True:
            run_pending_cycle()
            write_heartbeat("sleeping")
            minutes = CHECK_INTERVAL_SECONDS // 60
            log.info(
                "Next collection check in %sm", minutes,
                extra=_console("IDLE", "", f"sleeping {minutes}m"),
            )
            time.sleep(CHECK_INTERVAL_SECONDS)
    except _GracefulStopRequested:
        log.info(
            "Graceful stop requested; releasing writer authority and lease.",
            extra=_console("SHUTDOWN", "SHUTDOWN"),
        )
    finally:
        _release_writer_authority()
        stop_supervision()
        release_single_instance()
    return 0


def _log_startup_banner() -> None:
    """Plain, deterministic lines always go to the durable JSON log (and to
    a non-interactive console). An interactive console additionally gets a
    compact boxed header -- presentation only, shown once at startup."""
    log.info("=" * 60)
    log.info("BharatTerminal Smart Collector started")
    log.info("Data dir: %s", DATA_DIR)
    log.info("DuckDB:   %s", DUCKDB_PATH)
    log.info("Log DB:   %s", LOG_DB)
    log.info("=" * 60)

    from operations.logging_config import console_supports_unicode, is_console_interactive

    if not is_console_interactive():
        return
    from operations.console_format import render_banner, short_path

    unicode_ok = console_supports_unicode()
    writer_status = "OWNED" if _LOCAL_WRITER is not None else "REMOTE"
    fields = {
        "Status": "ACTIVE",
        "Data root": short_path(str(DATA_DIR), keep=2, unicode=unicode_ok),
        "Database": DUCKDB_PATH.name,
        "Writer": writer_status,
        "Started": ist_now().strftime("%Y-%m-%d %H:%M:%S IST"),
    }
    for line in render_banner(fields, unicode=unicode_ok).splitlines():
        log.info("%s", line, extra=_console("INFO", ""))


def bootstrap(logging_ready: bool = False) -> None:
    if not logging_ready:
        setup_logging()
    ensure_dirs()
    ensure_log_db()
    _try_become_writer_authority()
    ensure_fii_dii_table()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FO_BHAV_DIR.mkdir(parents=True, exist_ok=True)


def _try_become_writer_authority() -> None:
    """Start a WriterService owned by this collector process when no other
    process already owns one (e.g. the desktop app is closed). Reproduced
    live: without this, every collector write fails with
    "DuckDB writer service unavailable" whenever the desktop backend has
    never been opened, because WriterService was previously only ever
    started from main.py. WriterService.start() itself already refuses
    (raises) if a live owner exists elsewhere via the cross-process writer
    lease, so this is safe to attempt unconditionally: if the desktop backend
    already owns it, this collector process simply falls through to using
    submit_write()/submit_read() against that remote owner, exactly as
    before."""
    global _LOCAL_WRITER
    if _LOCAL_WRITER is not None:
        return
    service = WriterService(DUCKDB_PATH)
    try:
        service.start()
    except Exception:
        return
    _LOCAL_WRITER = service
    log.info(
        "Collector became the DuckDB writer authority (no other owner present)",
        extra=_console("WRITER", "WRITER"),
    )


def _release_writer_authority() -> None:
    global _LOCAL_WRITER
    if _LOCAL_WRITER is None:
        return
    try:
        _LOCAL_WRITER.stop()
    finally:
        _LOCAL_WRITER = None


def run_pending_cycle() -> None:
    try:
        write_heartbeat("checking")
        log.info("Checking for pending tasks...", extra=_console("SCAN", "SCAN"))
        pending = get_pending_tasks()
        write_heartbeat("pending", len(pending))
        if not pending:
            log.info("No tasks pending. All data current.", extra=_console("SCAN", "SCAN"))
            return

        log.info("Found %s pending tasks", len(pending), extra=_console("SCAN", "SCAN"))
        for item in pending:
            log.info(
                "[%s] %s for %s: %s",
                item["priority"],
                item["task"],
                item["date"],
                item["reason"],
            )

        for item in pending:
            write_heartbeat(f"running:{item['task']}", len(pending))
            run_logged_task(str(item["task"]), item["date"])
            time.sleep(3)
    except Exception as error:
        log.exception("Collector loop error: %s", error, extra=_console("ERROR", "SCAN"))


def run_logged_task(task: str, for_date: date) -> tuple[bool, str]:
    category = _task_category(task)
    if task_done(task, for_date):
        # Reproduced live: `--task NAME` (manual/operator re-run, unlike the
        # automatic pending-cycle path which never re-runs an already-done
        # task) called log_start() unconditionally. queue_job() correctly
        # leaves an already-SUCCEEDED ledger row untouched and returns False
        # without transitioning it to QUEUED, but claim_job() unconditionally
        # required QUEUED -> RUNNING regardless, crashing with "collector job
        # transition lost" on any manual re-run of a task already done today.
        log.info(
            "Already succeeded today: %s for %s", task, for_date,
            extra=_console("SKIP", category, "already succeeded today"),
        )
        return True, "already_succeeded_today"
    log.info("Running: %s for %s", task, for_date, extra=_console("RUN", category, f"{for_date}"))
    log_start(task, for_date)
    started = time.time()
    try:
        success, details = run_task(task, for_date)
    except Exception as error:
        success = False
        details = f"{type(error).__name__}: {error}"
        log.exception("Task crashed: %s for %s", task, for_date, extra=_console("ERROR", category))
    duration = time.time() - started
    if success:
        log_success(task, for_date, details, duration)
        log.info(
            "Done: %s for %s: %s (%.1fs)", task, for_date, details, duration,
            extra=_console("OK", category, f"{duration:.1f}s"),
        )
    else:
        log_failure(task, for_date, details, duration)
        # "not_available_yet" is this codebase's own established marker for
        # an expected, intentionally-deferred NSE/task state (holiday, not
        # yet published, etc.) -- reused here for presentation only, exactly
        # the same signal `run_task`/tests already treat as non-fatal, so a
        # normal deferred condition is never rendered as a red ERROR.
        tag = "WAIT" if "not_available_yet" in details else "FAIL"
        log.warning("Failed: %s for %s: %s", task, for_date, details, extra=_console(tag, category))
    return success, details


def run_once() -> int:
    setup_logging()
    if not acquire_single_instance():
        print("Collector already running")
        return 2
    try:
        if not start_supervision():
            print("Collector database lease already owned")
            return 2
        bootstrap(logging_ready=True)
        pending = get_pending_tasks()
        print(f"Pending tasks: {len(pending)}")
        for item in pending:
            print(f"  {item['task']} for {item['date']}: {item['reason']}")
        for item in pending:
            success, details = run_logged_task(str(item["task"]), item["date"])
            prefix = "OK " if success else "ERR"
            print(f"{prefix} {item['task']}: {details}")
        return 0
    finally:
        _release_writer_authority()
        stop_supervision()
        release_single_instance()


def run_named_task(task: str) -> int:
    setup_logging()
    if not acquire_single_instance():
        print("Collector already running")
        return 2
    try:
        if not start_supervision():
            print("Collector database lease already owned")
            return 2
        bootstrap(logging_ready=True)
        success, details = run_logged_task(task, ist_today())
        print(("Success: " if success else "Failed: ") + details)
        return 0 if success else 1
    finally:
        _release_writer_authority()
        stop_supervision()
        release_single_instance()


def show_status(limit: int = 30) -> int:
    ensure_dirs()
    ensure_log_db()
    with sqlite3.connect(str(LOG_DB)) as con:
        rows = con.execute(
            """
            SELECT task, date, status, details, started_at, duration_s
            FROM collector_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

    print(f"\n{'TASK':<24} {'DATE':<12} {'STATUS':<9} {'DETAILS':<44} {'STARTED':<19} DUR")
    print("-" * 120)
    for task, day, status, details, started_at, duration_s in rows:
        duration = f"{duration_s:.0f}s" if duration_s else ""
        print(
            f"{task:<24} {day:<12} {status:<9} "
            f"{str(details or '')[:44]:<44} {str(started_at or '')[:19]:<19} {duration}"
        )
    print()
    return 0


def forecast_symbols(limit: int = 50) -> list[str]:
    symbols: list[str] = ["NIFTY", "BANKNIFTY"]
    if MODELS_DIR.exists():
        symbols.extend(path.name for path in MODELS_DIR.iterdir() if path.is_dir())

    try:
        result = submit_read(ReadOperation.FORECAST_SYMBOLS, {})
        symbols.extend(str(item) for item in result.get("symbols", []) if item)
    except Exception:
        pass

    output: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = str(symbol).strip().upper()
        if not clean or clean in seen:
            continue
        output.append(clean)
        seen.add(clean)
        if len(output) >= limit:
            break
    return output


def load_ohlcv_full_frame(symbol: str) -> pd.DataFrame:
    candidates = _symbol_candidates(symbol)
    result = submit_read(ReadOperation.LOAD_OHLCV_FULL_FRAME, {"symbols": candidates})
    frame = pd.DataFrame.from_records(result.get("rows", []), columns=result.get("columns", []))
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _symbol_candidates(symbol: str) -> list[str]:
    normalized = symbol.strip().upper()
    if normalized in {"NIFTY", "NIFTY_50", "NIFTY 50"}:
        return ["NIFTY 50", "NIFTY"]
    if normalized in {"BANKNIFTY", "NIFTY_BANK", "NIFTY BANK"}:
        return ["NIFTY BANK", "BANKNIFTY"]
    return [normalized]


def _parse_market_date(value: str) -> date | None:
    clean = value.strip()
    if not clean:
        return None
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(clean, fmt).date()
        except ValueError:
            continue
    return None


def _num(value: Any) -> float:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _compact_details(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    parts: list[str] = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)[:1000] if parts else json.dumps(payload, sort_keys=True)[:1000]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BharatTerminal Smart Collector")
    parser.add_argument("--once", action="store_true", help="Run pending tasks once and exit")
    parser.add_argument("--status", action="store_true", help="Show recent collector runs")
    parser.add_argument("--task", type=str, help="Run one task for today's IST date")
    # argv=None (the sys.argv[1:] default) is only correct for a standalone
    # `python smart_collector.py` invocation. The frozen product's
    # "collector-run" child (see auto_collector.job_supervisor._spawn_collector
    # and main.py's `__main__` dispatch) is `[installed_exe, "collector-run",
    # ...]`, so main.py passes sys.argv[2:] explicitly here -- reading the
    # process's real sys.argv would otherwise leave the literal token
    # "collector-run" as an unrecognized positional argument and argparse
    # would exit 2 before the collector ever ran.
    args = parser.parse_args(argv)

    if args.status:
        return show_status()
    if args.once:
        return run_once()
    if args.task:
        return run_named_task(args.task)

    return run_collector_forever()


if __name__ == "__main__":
    raise SystemExit(main())
