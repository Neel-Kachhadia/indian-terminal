"""Windows Job Object process-tree owner for the scheduled collector.

Task Scheduler's `BharatTerminal_SmartCollector` action must invoke this
module (not `uv.exe` / `smart_collector.py` directly). It creates a Windows
Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, assigns itself to it,
then spawns the real collector as a child: in the installed product this is
a `collector-run` invocation of the SAME frozen executable this supervisor
is itself running as (see `_collector_command()`), never `uv.exe` / a repo
source path -- the `uv run ... smart_collector.py` form only applies when a
developer runs this module unfrozen, straight out of the repo. Because a
child process is automatically added to its parent's job
(no `CREATE_BREAKAWAY_FROM_JOB` is used here), the collector and anything it
spawns (training workers, subprocess calls) are members of the same job.

The OS -- not a heartbeat, not a PID file, not `taskkill /T` -- guarantees the
invariant this fixes: whenever every handle to the job object closes, which
happens automatically when this supervisor process exits for *any* reason
(clean exit, crash, or an external `taskkill /F` on THIS process), Windows
kills every remaining process still in the job. Reuses the same
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` mechanism already proven in
`src-tauri/src/main.rs` for the Tauri -> Python compute child relationship;
this is that same pattern applied to the Task Scheduler -> ... -> Python
collector relationship.

That parenthetical "on THIS process" is load-bearing. Reproduced live against
a real frozen build: PyInstaller's Windows onefile bootloader is itself a
TWO-PROCESS chain -- the exe Task Scheduler launches and tracks as the task
root extracts to a temp dir, then spawns a SECOND copy of itself as a child
to actually run the bundled Python (and waits, relaying stdout/stderr/exit
code). This module's own `create_kill_on_close_job()` call only ever runs in
that second, inner process, so the Job Object it creates is owned by the
inner process, not the one Task Scheduler (or a bare `taskkill /F <root
pid>`, no `/T`) actually terminates. Killing only the outer bootloader PID
left the inner supervisor and its collector-run child running and orphaned
in a live reproduction (confirmed writing to the real production DuckDB).
`_watch_parent_liveness()` closes that gap: it polls this process's own
immediate parent (the outer bootloader under a frozen onefile build; Task
Scheduler itself when run unfrozen/onedir, where no such split exists) and
self-terminates via the same `os.kill(os.getpid(), SIGTERM)` graceful path
already used for the CTRL_LOGOFF/CTRL_SHUTDOWN handler below the instant that
parent is gone -- which then lets THIS process's own job-object membership
do the rest.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

COMPUTE_DIR = Path(__file__).resolve().parents[1]

# Name of the real collector task the watchdog asks Task Scheduler to
# (re)start -- must match auto_collector.setup_autostart.TASK_NAME exactly.
# Not imported directly to keep this module importable standalone in tests
# without pulling in the XML-generation module's own dependencies.
COLLECTOR_TASK_NAME = "BharatTerminal_SmartCollector"

# Same resolved-absolute-path pattern as setup_autostart.SCHTASKS (duplicated
# rather than imported, for the same standalone-importability reason above):
# a bare "schtasks" argv[0] resolves via PATH search order at spawn time,
# which is unnecessary risk for an unattended production binary when the
# real system32 location is always known ahead of time.
SCHTASKS = (Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve() / "System32" / "schtasks.exe").resolve()

MAX_RESTARTS = 5
RESTART_WINDOW_SECONDS = 60 * 60
GRACEFUL_STOP_TIMEOUT_SECONDS = 30
# Exit codes the child uses to signal "not a crash, don't restart":
# 2 == smart_collector.py's own single-instance/lease-owned-elsewhere exit.
NO_RESTART_EXIT_CODES = frozenset({2})

log = logging.getLogger("job_supervisor")

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
CTRL_BREAK_EVENT = 1
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000
STILL_ACTIVE = 259

PARENT_WATCHDOG_INTERVAL_SECONDS = 2.0


def _configure_lifecycle_logging(component: str) -> Path:
    """Route headless supervisor/watchdog lifecycle events to durable JSONL.

    Task Scheduler launches the frozen windowed executable with no usable
    stderr, so a stream-only handler silently discarded every restart and
    recovery event.  Keep the two processes in separate rotating files to
    avoid cross-process rotation races while sharing the same structured
    logging directory and secret-scrubbing policy as the backend.
    """
    from data.paths import DATA_DIR
    from operations.logging_config import configure_structured_logging

    path = DATA_DIR / "logs" / f"{component}.jsonl"
    configure_structured_logging(path, redirect_prints=False)
    return path


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _new_kill_on_close_job() -> int:
    """Create a bare Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` set.
    Does NOT assign any process to it -- callers decide membership. Split out
    from `create_kill_on_close_job()` so tests can assign arbitrary *other*
    processes to a job without also pulling the test process itself in (see
    `create_kill_on_close_job()`, which self-assigns and would otherwise take
    the caller down together with the job on a simulated owner death)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        raise OSError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")
    return job


def create_kill_on_close_job() -> int:
    """Create a Job Object that kills every member process when the last
    handle to it closes, assign the CALLING process to it, and return the
    raw handle (int). Only the supervisor should call this: it makes the
    calling process itself a job member, so if the job's last handle later
    closes (including via this same process being force-killed), the OS
    kills every other process still in the job too."""
    job = _new_kill_on_close_job()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    # GetCurrentProcess() returns a pseudo-handle (-1 as HANDLE); assigning
    # this supervisor process itself means the job's "kill on close" limit
    # applies to this process's own children too, once it exits.
    current_process = ctypes.c_void_p(-1)
    ok = kernel32.AssignProcessToJobObject(job, current_process)
    if not ok:
        raise OSError(f"AssignProcessToJobObject (self) failed: {ctypes.get_last_error()}")
    return job


def _process_alive(pid: int) -> bool:
    """True if `pid` still exists and has not exited. Used only to poll our
    own immediate parent -- see `_watch_parent_liveness()` -- never the job's
    member processes, which the OS job-object mechanism already owns."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle(handle)


def _watch_parent_liveness(
    parent_pid: int,
    on_parent_dead,
    *,
    interval_seconds: float = PARENT_WATCHDOG_INTERVAL_SECONDS,
    alive_check=_process_alive,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Poll `parent_pid` on a background thread; call `on_parent_dead()`
    once, the moment it's no longer alive, then stop. See the module
    docstring for why this exists: PyInstaller's Windows onefile bootloader
    is a two-process chain, so the process Task Scheduler actually manages
    (and can force-kill) is this supervisor's PARENT, not itself -- the OS
    job-object "kill on close" limit only protects against THIS process
    dying, never against an ancestor dying out from under it while this
    process is still alive. `stop_event` lets callers (real shutdown, tests)
    end the thread without waiting for a dead parent."""
    stop_event = stop_event or threading.Event()
    fired = threading.Event()

    def _loop() -> None:
        while not stop_event.wait(interval_seconds):
            if not alive_check(parent_pid):
                if not fired.is_set():
                    fired.set()
                    on_parent_dead()
                return

    thread = threading.Thread(target=_loop, name="parent-liveness-watchdog", daemon=True)
    thread.start()
    return thread


def _resolve_uv_exe() -> str:
    home = Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "uv.exe"
    if home.exists():
        return str(home)
    found = shutil.which("uv.exe") or shutil.which("uv")
    if not found:
        raise RuntimeError("canonical uv runtime launcher not found")
    return found


def _collector_command() -> tuple[list[str], str]:
    """Command + cwd for the collector child.

    When this supervisor is itself the frozen installed executable
    (`sys.frozen`, set by the PyInstaller bootloader -- see
    `auto_collector.setup_autostart.runtime_layout`), the collector child
    must be a "collector-run" invocation of that SAME executable, never
    `uv run .../smart_collector.py`: the installed product ships no repo
    checkout, no `compute/.venv`, and no `uv` runtime for a production
    child to depend on. Only the dev (non-frozen) path -- a developer running
    `python job_supervisor.py` directly out of the repo -- uses `uv`.
    """
    if getattr(sys, "frozen", False):
        executable = str(Path(sys.executable).resolve())
        return [executable, "collector-run"], str(Path(executable).parent)
    uv_exe = _resolve_uv_exe()
    return (
        [
            uv_exe,
            "run",
            "--project",
            str(COMPUTE_DIR),
            "--frozen",
            "python",
            str(COMPUTE_DIR / "auto_collector" / "smart_collector.py"),
        ],
        str(COMPUTE_DIR),
    )


def _spawn_collector() -> subprocess.Popen:
    command, cwd = _collector_command()
    return subprocess.Popen(
        command,
        cwd=cwd,
        creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
    )


class _StopRequested(Exception):
    pass


# Kept alive for the process lifetime: ctypes does not hold a reference to a
# callback registered with a WinAPI function, so a locally-scoped
# WINFUNCTYPE instance would be garbage-collected while Windows still holds
# a raw pointer to it -- the next console-control event would then call
# freed memory.
_console_ctrl_handler_ref = None

# CTRL_LOGOFF_EVENT (5) and CTRL_SHUTDOWN_EVENT (6): unlike CTRL_C_EVENT and
# CTRL_BREAK_EVENT, Python's `signal` module has no mapping for these at
# all -- they are only observable via the raw Win32 console-control-handler
# API. Reproduced live: the real installed collector task terminated with
# Task Scheduler recording exit code 0xC000013A (STATUS_CONTROL_C_EXIT) with
# zero final log line -- not the graceful-shutdown message this module's own
# SIGBREAK/SIGINT/SIGTERM handler would have logged, meaning the OS's
# *default* console-control handler ran and called ExitProcess() before any
# Python-level signal callback (even for the types Python *can* see) got a
# chance to execute. The exact external trigger could not be pinned down
# (Task Scheduler's Operational event log was disabled with no historical
# record, and Windows does not audit GenerateConsoleCtrlEvent callers) but a
# LogonType=InteractiveToken task shares the interactive session's console
# state, and a CTRL_LOGOFF/CTRL_SHUTDOWN-class event does not require a real
# logoff/shutdown to be delivered to console-attached processes. Registering
# a native handler is the only way to get any chance at a graceful flush for
# these two event types; Windows still force-kills after its own ~5s budget
# regardless of what a handler does, so this narrows the window rather than
# eliminating it (a residual risk this can't remove).
_CTRL_LOGOFF_EVENT = 5
_CTRL_SHUTDOWN_EVENT = 6


def handle_console_ctrl_event(ctrl_type: int, notify=None) -> int:
    """Pure logic, split out from the raw ctypes callback for testability
    (the callback itself can't be invoked directly -- it's only reachable
    via a real Win32 console-control event). Returns 0 (not "handled") so
    the default handler still runs after us either way."""
    notify = notify or (lambda: os.kill(os.getpid(), signal.SIGTERM))
    if ctrl_type in (_CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT):
        log.warning("console control event %s received; attempting graceful stop", ctrl_type)
        try:
            notify()
        except Exception:
            pass
    return 0


def _install_console_ctrl_handler() -> None:
    global _console_ctrl_handler_ref
    if os.name != "nt":
        return
    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32)
    _console_ctrl_handler_ref = HANDLER_ROUTINE(handle_console_ctrl_event)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetConsoleCtrlHandler.argtypes = [HANDLER_ROUTINE, ctypes.c_int]
    kernel32.SetConsoleCtrlHandler.restype = ctypes.c_int
    if not kernel32.SetConsoleCtrlHandler(_console_ctrl_handler_ref, 1):
        log.warning("SetConsoleCtrlHandler registration failed: %s", ctypes.get_last_error())


def _install_graceful_stop_handler() -> None:
    def _handler(signum: int, _frame: object) -> None:
        raise _StopRequested()

    # CTRL_BREAK_EVENT is what Task Scheduler / `schtasks /End` deliver to a
    # console process group on Windows; SIGTERM has no native Windows
    # delivery path but is kept for parity when invoked from POSIX-like
    # supervisors in tests.
    signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    _install_console_ctrl_handler()


def _graceful_stop_child(child: subprocess.Popen) -> None:
    try:
        child.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        child.wait(timeout=GRACEFUL_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def run() -> int:
    _configure_lifecycle_logging("collector-supervisor")
    log.info("collector supervisor starting")
    try:
        create_kill_on_close_job()
    except OSError:
        log.exception("Job Object setup failed; refusing to spawn an unowned collector process tree")
        return 1

    _install_graceful_stop_handler()

    # See the module docstring: under a frozen onefile build, the process
    # Task Scheduler tracks and can force-kill is this supervisor's PARENT
    # (the PyInstaller bootloader), not this process itself, so the job
    # object above does not by itself protect against that ancestor dying.
    watchdog_stop = threading.Event()
    if os.name == "nt":
        parent_pid = os.getppid()
        if not parent_pid:
            log.error("parent pid unavailable; refusing to spawn without ancestor ownership")
            return 1
        _watch_parent_liveness(
            parent_pid,
            lambda: os.kill(os.getpid(), signal.SIGTERM),
            stop_event=watchdog_stop,
        )

    try:
        attempts: list[float] = []
        while True:
            child = _spawn_collector()
            log.info("collector child started pid=%s", child.pid)
            try:
                exit_code = child.wait()
            except _StopRequested:
                log.info("graceful stop requested; forwarding to collector child")
                _graceful_stop_child(child)
                return 0
            log.info("collector child exited code=%s", exit_code)

            if exit_code in NO_RESTART_EXIT_CODES:
                return exit_code

            now = time.monotonic()
            attempts = [t for t in attempts if now - t < RESTART_WINDOW_SECONDS]
            attempts.append(now)
            if len(attempts) > MAX_RESTARTS:
                log.error(
                    "collector child crashed %s times within %ss; not restarting again",
                    len(attempts),
                    RESTART_WINDOW_SECONDS,
                )
                return exit_code or 1
            log.warning("restarting collector child (attempt %s/%s)", len(attempts), MAX_RESTARTS)
    finally:
        watchdog_stop.set()


def run_watchdog_check() -> int:
    """Entry point for the `collector-watchdog` Task Scheduler task (see
    module docstring's "Watchdog" section in supervision.py for why this
    exists -- RestartOnFailure cannot recover the top-level supervisor
    process dying on Windows). Short-lived: check, maybe ask Task Scheduler
    to (re)start the real collector task, then exit. Safe to run
    concurrently with itself only via its OWN task's
    MultipleInstancesPolicy=IgnoreNew (set by setup_autostart.py); safe to
    run while a real collector is alive because it only ever *asks* Task
    Scheduler to start the main task, which has its own IgnoreNew guard, and
    it does not even ask unless status_snapshot() says the collector is down.
    """
    from auto_collector import supervision
    from data.paths import DATA_DIR

    _configure_lifecycle_logging("collector-watchdog")
    log.info("collector watchdog starting")
    log_db = DATA_DIR / "collector_log.db"
    state_file = DATA_DIR / "collector_watchdog_state.json"

    snapshot = supervision.status_snapshot(log_db)
    now = time.time()
    attempts = supervision.load_watchdog_attempts(state_file, now=now)
    should_restart, reason = supervision.watchdog_decision(
        collector_running=bool(snapshot.get("collector_running")),
        attempts=attempts,
    )
    log.info("watchdog check: collector_running=%s decision=%s (%s)", snapshot.get("collector_running"), should_restart, reason)
    if not should_restart:
        return 0

    attempts.append(now)
    supervision.save_watchdog_attempts(state_file, attempts)
    try:
        result = subprocess.run(
            [str(SCHTASKS), "/run", "/tn", COLLECTOR_TASK_NAME],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log.error("watchdog: failed to invoke schtasks /run: %s", error)
        return 1
    if result.returncode != 0:
        log.error(
            "watchdog: schtasks /run exited %s: %s", result.returncode, result.stderr.strip()
        )
        return result.returncode
    log.warning("watchdog: collector was down, requested restart via Task Scheduler")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "watchdog":
        raise SystemExit(run_watchdog_check())
    raise SystemExit(run())
