"""Regression coverage for the Windows Job Object process-tree owner.

These tests only ever spawn short-lived disposable `python -c` processes of
their own -- they never touch the real registered
`BharatTerminal_SmartCollector` scheduled task or the live collector.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object test")

from auto_collector.job_supervisor import (  # noqa: E402
    _install_console_ctrl_handler,
    _new_kill_on_close_job,
    _process_alive,
    _terminate_stale_writer_owner,
    _watch_parent_liveness,
    handle_console_ctrl_event,
)
from auto_collector import job_supervisor  # noqa: E402


def _pid_alive(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    STILL_ACTIVE = 259
    exit_code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
    kernel32.CloseHandle(handle)
    return exit_code.value == STILL_ACTIVE


def test_job_object_kills_child_when_owner_handle_closes() -> None:
    """This IS the fix for the reproduced orphan-python.exe finding: a
    process assigned to a JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE job dies the
    instant the job's last handle closes, which happens automatically when
    the owning process exits for any reason -- including a forceful,
    non-cooperating termination."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int

    job = _new_kill_on_close_job()

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
    )
    try:
        ok = kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(int(child._handle)))
        assert ok, f"AssignProcessToJobObject failed: {ctypes.get_last_error()}"
        assert _pid_alive(child.pid)

        # Simulate the owner process disappearing: close our only handle to
        # the job. In production this happens implicitly when the owning
        # process itself is force-killed; here we close the handle directly
        # to isolate the OS behavior from process-lifetime plumbing.
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle(ctypes.c_void_p(job))

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _pid_alive(child.pid):
            time.sleep(0.1)
        assert not _pid_alive(child.pid), "child survived job-object handle close"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_child_processes_auto_inherit_owner_job_without_breakaway() -> None:
    """A process assigned to a job propagates membership to its own children
    by default (no CREATE_BREAKAWAY_FROM_JOB used anywhere in
    job_supervisor.py), which is what lets one job own the whole
    supervisor -> uv.exe -> python.exe collector tree."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    kernel32.IsProcessInJob.restype = ctypes.c_int

    job = _new_kill_on_close_job()
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print(c.pid); sys.stdout.flush(); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ok = kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(int(parent._handle)))
        assert ok
        grandchild_pid_line = parent.stdout.readline().strip()
        assert grandchild_pid_line, "grandchild never reported its pid"
        grandchild_pid = int(grandchild_pid_line)
        assert _pid_alive(grandchild_pid)

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, grandchild_pid)
        assert handle
        is_in_job = ctypes.c_int(0)
        kernel32.IsProcessInJob(handle, ctypes.c_void_p(job), ctypes.byref(is_in_job))
        kernel32.CloseHandle(handle)
        assert is_in_job.value == 1, "grandchild did not auto-inherit job membership"

        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle(ctypes.c_void_p(job))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _pid_alive(grandchild_pid):
            time.sleep(0.1)
        assert not _pid_alive(grandchild_pid), "grandchild survived job-object handle close"
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def test_console_ctrl_handler_registers_with_real_winapi() -> None:
    """Reproduces a real incident: the installed collector task terminated
    with Task Scheduler recording exit code 0xC000013A
    (STATUS_CONTROL_C_EXIT) and zero final log line -- not the graceful
    message this module's SIGBREAK/SIGINT/SIGTERM handler would log,
    meaning the OS's default console-control handler ran and exited the
    process before any Python signal callback got scheduled. CTRL_LOGOFF
    and CTRL_SHUTDOWN have no Python `signal` mapping at all; a real
    Win32 SetConsoleCtrlHandler registration is the only way to get any
    chance at a graceful flush for those two event types."""
    _install_console_ctrl_handler()  # must not raise; real WinAPI call


def test_console_ctrl_handler_forwards_only_logoff_and_shutdown() -> None:
    calls: list[int] = []
    for ctrl_type in range(0, 7):
        handle_console_ctrl_event(ctrl_type, notify=lambda: calls.append(ctrl_type))
    assert calls == [5, 6]  # CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT only


def test_console_ctrl_handler_swallows_notify_exception() -> None:
    def _boom() -> None:
        raise RuntimeError("should not propagate to the WinAPI callback")

    result = handle_console_ctrl_event(5, notify=_boom)
    assert result == 0


def test_process_alive_reflects_real_process_lifetime() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _process_alive(child.pid)
    finally:
        child.kill()
        child.wait(timeout=5)
    assert not _process_alive(child.pid)


def test_parent_liveness_watchdog_fires_once_when_watched_process_dies() -> None:
    """Reproduces the real gap this closes: PyInstaller's Windows onefile
    bootloader is a two-process chain, so the process Task Scheduler tracks
    and can force-kill (this supervisor's PARENT under a frozen build) is not
    the same process the Job Object above protects (THIS process). Live
    reproduction against a real frozen build: `taskkill /F` on only the
    outer bootloader PID left the inner job-owning supervisor and its
    collector-run child running, still writing to the production DuckDB.
    This test proves the watchdog detects that "owning ancestor is gone"
    condition and fires exactly once, without needing a real PyInstaller
    build -- any disposable process stands in for "the parent"."""
    fake_parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    fired: list[int] = []
    stop_event = threading.Event()
    try:
        thread = _watch_parent_liveness(
            fake_parent.pid,
            lambda: fired.append(1),
            interval_seconds=0.05,
            stop_event=stop_event,
        )
        time.sleep(0.2)
        assert not fired, "must not fire while the watched process is alive"

        fake_parent.kill()
        fake_parent.wait(timeout=5)

        thread.join(timeout=5)
        assert not thread.is_alive(), "watchdog thread must exit after firing"
        assert fired == [1], "must fire exactly once"
    finally:
        stop_event.set()
        if fake_parent.poll() is None:
            fake_parent.kill()
            fake_parent.wait(timeout=5)


def test_parent_liveness_watchdog_stop_event_prevents_late_fire() -> None:
    """A graceful shutdown that sets `stop_event` before the watched process
    actually dies must not later call `on_parent_dead` -- that would race a
    clean exit with a spurious extra SIGTERM-equivalent self-signal."""
    fake_parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    fired: list[int] = []
    stop_event = threading.Event()
    try:
        thread = _watch_parent_liveness(
            fake_parent.pid,
            lambda: fired.append(1),
            interval_seconds=0.05,
            stop_event=stop_event,
        )
        stop_event.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        fake_parent.kill()
        fake_parent.wait(timeout=5)
        time.sleep(0.2)
        assert fired == []
    finally:
        if fake_parent.poll() is None:
            fake_parent.kill()
            fake_parent.wait(timeout=5)


def test_supervisor_refuses_to_spawn_when_job_object_ownership_fails(monkeypatch) -> None:
    spawned: list[bool] = []

    def fail_job_setup() -> int:
        raise OSError("job unavailable")

    monkeypatch.setattr(job_supervisor, "create_kill_on_close_job", fail_job_setup)
    monkeypatch.setattr(job_supervisor, "_spawn_collector", lambda: spawned.append(True))
    monkeypatch.setattr(job_supervisor, "_configure_lifecycle_logging", lambda _component: None)

    assert job_supervisor.run() == 1
    assert spawned == []


# --- Watchdog writer-blind-spot recovery: terminate the exact recorded owner
# PID (never a swept target) before the collector task can be re-run, so a
# fresh collector-run claims one clean owner (see run_watchdog_check). -------


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_terminate_stale_writer_owner_succeeds(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return _FakeCompletedProcess(0, stdout="SUCCESS: terminated")

    monkeypatch.setattr(job_supervisor.subprocess, "run", fake_run)

    terminated, detail = _terminate_stale_writer_owner(30060)

    assert terminated is True
    assert "terminated" in detail
    [argv] = calls
    assert str(job_supervisor.TASKKILL) in argv
    assert "/PID" in argv and "30060" in argv
    assert "/F" in argv


def test_terminate_stale_writer_owner_treats_already_gone_pid_as_success(monkeypatch) -> None:
    """taskkill exit 128 ('PID not found') means there is already no owner
    left to conflict with a restart -- a successful outcome for this call's
    purpose, not a failure to report/retry."""
    monkeypatch.setattr(
        job_supervisor.subprocess, "run",
        lambda argv, **_kwargs: _FakeCompletedProcess(128, stderr="not found"),
    )
    terminated, _detail = _terminate_stale_writer_owner(999999)
    assert terminated is True


def test_terminate_stale_writer_owner_reports_genuine_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        job_supervisor.subprocess, "run",
        lambda argv, **_kwargs: _FakeCompletedProcess(5, stderr="ERROR: Access is denied."),
    )
    terminated, detail = _terminate_stale_writer_owner(30060)
    assert terminated is False
    assert "denied" in detail.lower()
