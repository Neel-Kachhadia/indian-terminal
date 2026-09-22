from __future__ import annotations

import datetime as dt
import json
import os
import socket
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Protocol


LEASE_SECONDS = 90

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _own_process_start_time() -> int | None:
    """Windows process creation time as a single integer, paired with a PID
    to prove *which* process holds a lease, not just which PID (PIDs get
    reused). Mirrors `db.write_service._own_process_start_time` -- duplicated
    rather than imported so this lightweight supervision module keeps its
    existing no-DuckDB-dependency property."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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
    """True if the exact (pid, creation-time) process is alive, False if
    provably dead (or the PID was reused by an unrelated process), None if
    liveness cannot be established (caller falls back to lease-TTL policy).
    Mirrors `db.write_service._owner_liveness`."""
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
HEARTBEAT_SECONDS = 20
HISTORY_RETENTION_DAYS = 180
INSTANCE_RUNNING = "RUNNING"
INSTANCE_STOPPED = "STOPPED"
INSTANCE_ABANDONED = "ABANDONED"
JOB_STATES = frozenset({"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "ABANDONED"})
LEGAL_JOB_TRANSITIONS = {
    "QUEUED": frozenset({"RUNNING", "CANCELLED"}),
    "RUNNING": frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "ABANDONED"}),
    "FAILED": frozenset({"QUEUED", "CANCELLED"}),
    "ABANDONED": frozenset({"QUEUED", "CANCELLED"}),
    "CANCELLED": frozenset({"QUEUED"}),
    "SUCCEEDED": frozenset(),
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Clock(Protocol):
    def now(self) -> dt.datetime: ...


class RealClock:
    def now(self) -> dt.datetime:
        return utc_now()


def iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_schema(log_db: Path, *, now: dt.datetime | None = None) -> None:
    log_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(log_db)) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS collector_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                date TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT DEFAULT '',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                duration_s REAL,
                instance_id TEXT,
                lease_expires_at TEXT,
                attempt INTEGER NOT NULL DEFAULT 0,
                UNIQUE(task, date) ON CONFLICT REPLACE
            )
            """
        )
        columns = {row[1] for row in con.execute("PRAGMA table_info(collector_runs)")}
        for name, definition in (
            ("instance_id", "TEXT"),
            ("lease_expires_at", "TEXT"),
            ("attempt", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                con.execute(f"ALTER TABLE collector_runs ADD COLUMN {name} {definition}")
        con.execute(
            """
            UPDATE collector_runs SET status = CASE lower(status)
                WHEN 'running' THEN 'RUNNING'
                WHEN 'success' THEN 'SUCCEEDED'
                WHEN 'failed' THEN 'FAILED'
                WHEN 'cancelled' THEN 'CANCELLED'
                WHEN 'abandoned' THEN 'ABANDONED'
                WHEN 'queued' THEN 'QUEUED'
                ELSE status END
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS collector_instances (
                instance_id TEXT PRIMARY KEY,
                pid INTEGER NOT NULL,
                hostname TEXT NOT NULL,
                started_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL,
                state TEXT NOT NULL,
                activity TEXT NOT NULL,
                pending_count INTEGER,
                stopped_at TEXT
            )
            """
        )
        instance_columns = {row[1] for row in con.execute("PRAGMA table_info(collector_instances)")}
        if "process_start_time" not in instance_columns:
            con.execute("ALTER TABLE collector_instances ADD COLUMN process_start_time INTEGER")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_collector_runs_started ON collector_runs(started_at)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_collector_instances_lease "
            "ON collector_instances(state, lease_expires_at)"
        )
        cutoff = iso_utc((now or utc_now()) - dt.timedelta(days=HISTORY_RETENTION_DAYS))
        con.execute(
            "DELETE FROM collector_runs WHERE finished_at < ? AND status IN ('SUCCEEDED', 'FAILED', 'CANCELLED', 'ABANDONED')",
            [cutoff],
        )
        con.execute(
            "DELETE FROM collector_instances WHERE stopped_at < ? AND state != 'RUNNING'",
            [cutoff],
        )


def assert_job_transition(current: str, target: str) -> None:
    if current not in JOB_STATES or target not in JOB_STATES:
        raise ValueError(f"unknown collector job state: {current} -> {target}")
    if target not in LEGAL_JOB_TRANSITIONS[current]:
        raise ValueError(f"invalid collector job transition: {current} -> {target}")


class CollectorSupervisor:
    def __init__(
        self,
        log_db: Path,
        heartbeat_file: Path,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.log_db = Path(log_db)
        self.heartbeat_file = Path(heartbeat_file)
        self.instance_id = str(uuid.uuid4())
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.clock = clock or RealClock()
        self._activity = "BOOTSTRAPPING"
        self._pending_count: int | None = None
        self._activity_lock = threading.Lock()
        self._heartbeat_write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self, now: dt.datetime | None = None, *, start_thread: bool = True) -> bool:
        current = now or self.clock.now()
        ensure_schema(self.log_db, now=current)
        now_text = iso_utc(current)
        lease_text = iso_utc(current + dt.timedelta(seconds=LEASE_SECONDS))
        own_start_time = _own_process_start_time()
        with sqlite3.connect(str(self.log_db), isolation_level=None) as con:
            con.execute("BEGIN IMMEDIATE")
            self._reconcile_locked(con, current)
            owner = con.execute(
                """
                SELECT instance_id, pid, process_start_time FROM collector_instances
                WHERE state = 'RUNNING' AND lease_expires_at > ?
                ORDER BY heartbeat_at DESC LIMIT 1
                """,
                [now_text],
            ).fetchone()
            if owner is not None and owner[0] != self.instance_id:
                # A forced-kill of the previous owner leaves its lease
                # technically unexpired for up to LEASE_SECONDS -- reproduced
                # live: a real Task Scheduler restart immediately after the
                # supervisor's Job Object killed a prior owner still hit this
                # branch and refused for the remainder of the TTL, even
                # though the owner process was provably gone. Prove the
                # owner is actually dead (same pattern already proven for
                # the DuckDB writer lease in write_service.py) before
                # falling back to the conservative TTL wait.
                if _owner_liveness(owner[1], owner[2]) is False:
                    con.execute(
                        "UPDATE collector_instances SET state = 'ABANDONED', stopped_at = ? "
                        "WHERE instance_id = ? AND state = 'RUNNING'",
                        [now_text, owner[0]],
                    )
                else:
                    con.rollback()
                    return False
            con.execute(
                """
                INSERT INTO collector_instances
                (instance_id, pid, hostname, started_at, heartbeat_at,
                 lease_expires_at, state, activity, pending_count, process_start_time)
                VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', ?, ?, ?)
                """,
                [
                    self.instance_id,
                    self.pid,
                    self.hostname,
                    now_text,
                    now_text,
                    lease_text,
                    self._activity,
                    self._pending_count,
                    own_start_time,
                ],
            )
            con.commit()
        self._started = True
        self._write_heartbeat(current, lease_text)
        if start_thread:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="collector-lease-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return True

    def set_activity(self, activity: str, pending_count: int | None = None) -> None:
        with self._activity_lock:
            self._activity = str(activity).upper()
            self._pending_count = pending_count
        if self._started:
            self.renew()

    def renew(self, now: dt.datetime | None = None) -> None:
        if not self._started:
            raise RuntimeError("collector supervisor not started")
        current = now or self.clock.now()
        now_text = iso_utc(current)
        lease_text = iso_utc(current + dt.timedelta(seconds=LEASE_SECONDS))
        with self._activity_lock:
            activity = self._activity
            pending_count = self._pending_count
        with sqlite3.connect(str(self.log_db)) as con:
            updated = con.execute(
                """
                UPDATE collector_instances
                SET heartbeat_at = ?, lease_expires_at = ?, activity = ?, pending_count = ?
                WHERE instance_id = ? AND state = 'RUNNING'
                """,
                [now_text, lease_text, activity, pending_count, self.instance_id],
            ).rowcount
            if updated != 1:
                raise RuntimeError("collector lease ownership lost")
            con.execute(
                """
                UPDATE collector_runs SET lease_expires_at = ?
                WHERE instance_id = ? AND status = 'RUNNING'
                """,
                [lease_text, self.instance_id],
            )
        self._write_heartbeat(current, lease_text)

    def stop(self, now: dt.datetime | None = None) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=HEARTBEAT_SECONDS + 2)
        if not self._started:
            return
        current = now or self.clock.now()
        with sqlite3.connect(str(self.log_db)) as con:
            con.execute(
                """
                UPDATE collector_instances
                SET state = 'STOPPED', stopped_at = ?, lease_expires_at = ?
                WHERE instance_id = ? AND state = 'RUNNING'
                """,
                [iso_utc(current), iso_utc(current), self.instance_id],
            )
            con.execute(
                """
                UPDATE collector_runs
                SET status = 'CANCELLED', finished_at = ?,
                    details = CASE WHEN details = '' THEN 'collector stopped' ELSE details END
                WHERE instance_id = ? AND status = 'RUNNING'
                """,
                [iso_utc(current), self.instance_id],
            )
        self._write_heartbeat(current, iso_utc(current), state=INSTANCE_STOPPED)
        self._started = False

    def queue_job(self, task: str, for_date: dt.date) -> bool:
        now_text = iso_utc(self.clock.now())
        day = for_date.isoformat()
        with sqlite3.connect(str(self.log_db), isolation_level=None) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM collector_runs WHERE task = ? AND date = ?",
                [task, day],
            ).fetchone()
            if row is None:
                con.execute(
                    """
                    INSERT INTO collector_runs
                    (task, date, status, details, started_at, attempt)
                    VALUES (?, ?, 'QUEUED', '', ?, 0)
                    """,
                    [task, day, now_text],
                )
                con.commit()
                return True
            current = str(row[0]).upper()
            if current == "SUCCEEDED" or current in {"QUEUED", "RUNNING"}:
                con.commit()
                return current == "QUEUED"
            assert_job_transition(current, "QUEUED")
            con.execute(
                """
                UPDATE collector_runs
                SET status = 'QUEUED', details = '', finished_at = NULL,
                    instance_id = NULL, lease_expires_at = NULL
                WHERE task = ? AND date = ? AND status = ?
                """,
                [task, day, current],
            )
            con.commit()
            return True

    def claim_job(self, task: str, for_date: dt.date) -> None:
        current = self.clock.now()
        lease = current + dt.timedelta(seconds=LEASE_SECONDS)
        self._transition_job(
            task,
            for_date,
            "QUEUED",
            "RUNNING",
            details="",
            started_at=iso_utc(current),
            instance_id=self.instance_id,
            lease_expires_at=iso_utc(lease),
            increment_attempt=True,
        )

    def finish_job(
        self,
        task: str,
        for_date: dt.date,
        success: bool,
        details: str,
        duration_s: float,
    ) -> None:
        target = "SUCCEEDED" if success else "FAILED"
        self._transition_job(
            task,
            for_date,
            "RUNNING",
            target,
            details=details[:1000],
            finished_at=iso_utc(self.clock.now()),
            duration_s=duration_s,
            lease_expires_at=None,
            owner_required=True,
        )

    def reconcile(self, now: dt.datetime | None = None) -> tuple[int, int]:
        current = now or self.clock.now()
        with sqlite3.connect(str(self.log_db), isolation_level=None) as con:
            con.execute("BEGIN IMMEDIATE")
            result = self._reconcile_locked(con, current)
            con.commit()
            return result

    def _reconcile_locked(
        self,
        con: sqlite3.Connection,
        current: dt.datetime,
    ) -> tuple[int, int]:
        now_text = iso_utc(current)
        instances = con.execute(
            """
            UPDATE collector_instances SET state = 'ABANDONED', stopped_at = ?
            WHERE state = 'RUNNING' AND lease_expires_at <= ?
            """,
            [now_text, now_text],
        ).rowcount
        jobs = con.execute(
            """
            UPDATE collector_runs
            SET status = 'ABANDONED', finished_at = ?,
                details = CASE WHEN details = '' THEN 'owner lease expired' ELSE details END
            WHERE status = 'RUNNING'
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
            """,
            [now_text, now_text],
        ).rowcount
        return instances, jobs

    def _transition_job(
        self,
        task: str,
        for_date: dt.date,
        expected: str,
        target: str,
        *,
        owner_required: bool = False,
        increment_attempt: bool = False,
        **updates: Any,
    ) -> None:
        assert_job_transition(expected, target)
        assignments = ["status = ?"]
        values: list[Any] = [target]
        for column in (
            "details",
            "started_at",
            "finished_at",
            "duration_s",
            "instance_id",
            "lease_expires_at",
        ):
            if column in updates:
                assignments.append(f"{column} = ?")
                values.append(updates[column])
        if increment_attempt:
            assignments.append("attempt = attempt + 1")
        values.extend([task, for_date.isoformat(), expected])
        owner_clause = " AND instance_id = ?" if owner_required else ""
        if owner_required:
            values.append(self.instance_id)
        with sqlite3.connect(str(self.log_db)) as con:
            changed = con.execute(
                f"UPDATE collector_runs SET {', '.join(assignments)} "
                f"WHERE task = ? AND date = ? AND status = ?{owner_clause}",
                values,
            ).rowcount
        if changed != 1:
            raise RuntimeError(
                f"collector job transition lost: {task}/{for_date} {expected} -> {target}"
            )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(HEARTBEAT_SECONDS):
            try:
                self.renew()
            except Exception:
                self._stop_event.set()
                return

    def _write_heartbeat(
        self,
        current: dt.datetime,
        lease_text: str,
        *,
        state: str = INSTANCE_RUNNING,
    ) -> None:
        with self._activity_lock:
            payload = {
                "schema_version": 1,
                "instance_id": self.instance_id,
                "pid": self.pid,
                "hostname": self.hostname,
                "state": state,
                "activity": self._activity,
                "pending_count": self._pending_count,
                "heartbeat_at": iso_utc(current),
                "lease_expires_at": lease_text,
            }
        with self._heartbeat_write_lock:
            self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.heartbeat_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.heartbeat_file)


def status_snapshot(
    log_db: Path,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    if not log_db.exists():
        return {
            "collector_running": False,
            "instance": None,
            "error": "collector supervision DB not found",
        }
    now_text = iso_utc(now or utc_now())
    try:
        with sqlite3.connect(f"file:{log_db.as_posix()}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            instance = con.execute(
                """
                SELECT instance_id, pid, hostname, started_at, heartbeat_at,
                       lease_expires_at, state, activity, pending_count
                FROM collector_instances
                ORDER BY heartbeat_at DESC LIMIT 1
                """
            ).fetchone()
            abandoned = con.execute(
                "SELECT COUNT(*) FROM collector_runs WHERE status = 'ABANDONED'"
            ).fetchone()[0]
    except sqlite3.Error as error:
        return {
            "collector_running": False,
            "instance": None,
            "error": f"collector supervision unavailable: {error}",
        }
    payload = dict(instance) if instance is not None else None
    running = bool(
        payload
        and payload["state"] == INSTANCE_RUNNING
        and payload["lease_expires_at"] > now_text
    )
    return {
        "collector_running": running,
        "instance": payload,
        "abandoned_jobs": int(abandoned),
    }


# --- Watchdog: recovers an unexpectedly-dead collector on Windows -----------
#
# Windows Task Scheduler's own RestartOnFailure cannot be relied on for this.
# Reproduced live against the real installed task: after a forced kill of the
# supervisor process, `schtasks /query ... /v` reported
# `Last Result: 267014` == 0x41306 == SCHED_S_TASK_TERMINATED ("the last run
# of the task was terminated by the user") -- a real, documented Task
# Scheduler status class, distinct from a genuine application failure exit
# code. RestartOnFailure only fires on the latter; Windows itself does not
# consider an externally-terminated run eligible for it, so the task sat
# Idle indefinitely with no restart. This is Windows' own semantics, not a
# code defect in job_supervisor.py's already-correct in-process restart loop
# (which DOES correctly recover a `collector-run` CHILD dying while the
# supervisor stays alive -- see run() above; that path was never broken).
#
# The gap is specifically: supervisor process itself dies (crash, or
# something kills its whole job-object tree, or the outer PyInstaller
# bootloader Task Scheduler actually tracks dies). Recovering THAT requires a
# mechanism outside the dead process. This watchdog is a second, independent,
# time-triggered Task Scheduler task: periodically check whether the
# collector is actually alive via the same `status_snapshot()` used
# everywhere else in this module (single source of truth, no new liveness
# signal invented), and if not, ask Task Scheduler to (re)start the real
# collector task via `schtasks /run` -- reusing Task Scheduler's own
# MultipleInstancesPolicy=IgnoreNew on that task as a second, independent
# guard against ever creating two logical collectors, on top of this
# process's own DB-backed check.
#
# Restart-rate bounding must be PERSISTENT (a file, not in-memory), because
# each watchdog invocation is a fresh short-lived process -- an in-memory
# counter like job_supervisor.py's own `attempts` list would silently reset
# every single watchdog run and could never actually bound anything.

WATCHDOG_MAX_RESTARTS = 5
WATCHDOG_RESTART_WINDOW_SECONDS = 60 * 60


def load_watchdog_attempts(state_file: Path, *, now: float) -> list[float]:
    """Read persisted restart-attempt timestamps (epoch seconds), pruned to
    the trailing WATCHDOG_RESTART_WINDOW_SECONDS. Missing/corrupt state reads
    as "no prior attempts" -- never as a reason to skip a needed restart."""
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        attempts = [float(value) for value in raw.get("attempts", [])]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []
    return [value for value in attempts if now - value < WATCHDOG_RESTART_WINDOW_SECONDS]


def save_watchdog_attempts(state_file: Path, attempts: list[float]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
    os.replace(temporary, state_file)


def watchdog_decision(
    *,
    collector_running: bool,
    attempts: list[float],
    max_restarts: int = WATCHDOG_MAX_RESTARTS,
    writer_responsive: bool | None = None,
) -> tuple[bool, str]:
    """Pure policy decision, independent of I/O so it's exhaustively
    unit-testable. Returns (should_restart, reason).

    `writer_responsive` closes the blind spot proven live in production: the
    collector-lease heartbeat (`collector_running`, above) and the DuckDB
    WriterService's own `writer_runtime.state`/heartbeat (see
    `db.write_service.writer_service_status()`) are two independent signals
    from two independent threads inside the SAME collector-run process. One
    thread dying does not kill the other, so a live collector lease alone
    does not prove the writer is alive. Pass `writer_responsive=None` (the
    default) when the caller has not checked writer health at all -- this
    preserves the pre-existing collector-only policy exactly, byte for byte.
    Pass an explicit bool once checked: only `False` (genuinely proven
    unresponsive/FAILED/stale-heartbeat) escalates a live collector lease
    into a restart decision; a sleeping-but-healthy collector (writer
    heartbeats continuously regardless of collection activity) always reads
    `True` and must never be mistaken for a dead writer.
    """
    writer_dead_under_live_collector = collector_running and writer_responsive is False
    if collector_running and not writer_dead_under_live_collector:
        return False, "collector is already running"
    if len(attempts) >= max_restarts:
        return False, (
            f"collector has been restarted {len(attempts)} times within "
            f"{WATCHDOG_RESTART_WINDOW_SECONDS}s; refusing to restart again "
            "(failing closed rather than restart-looping a persistently "
            "broken collector)"
        )
    if writer_dead_under_live_collector:
        return True, (
            "collector lease is live but its DuckDB WriterService is "
            "unresponsive (FAILED state or stale heartbeat); treating the "
            "worker as unhealthy and restart budget remains"
        )
    return True, "collector is not running and restart budget remains"
