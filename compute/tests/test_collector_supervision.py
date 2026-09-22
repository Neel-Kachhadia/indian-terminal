from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from auto_collector.supervision import (
    LEASE_SECONDS,
    WATCHDOG_RESTART_WINDOW_SECONDS,
    CollectorSupervisor,
    assert_job_transition,
    load_watchdog_attempts,
    save_watchdog_attempts,
    status_snapshot,
    watchdog_decision,
)


class ManualClock:
    def __init__(self, current: dt.datetime) -> None:
        self.current = current

    def now(self) -> dt.datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += dt.timedelta(seconds=seconds)


def test_legal_and_invalid_job_transitions() -> None:
    assert_job_transition("QUEUED", "RUNNING")
    assert_job_transition("RUNNING", "SUCCEEDED")
    with pytest.raises(ValueError, match="invalid"):
        assert_job_transition("SUCCEEDED", "RUNNING")
    with pytest.raises(ValueError, match="unknown"):
        assert_job_transition("MISSING", "RUNNING")


def test_dead_owner_lease_reclaimed_immediately_not_after_full_ttl(tmp_path) -> None:
    """Reproduces a real deployed defect found via a live forced-parent-death
    test against the actual installed Task Scheduler task: the Job Object
    fix correctly killed the whole process tree instantly, but the very next
    real `schtasks /Run` (well within LEASE_SECONDS=90 of the kill) was
    refused with "Another collector owns a live database lease. Exiting." --
    the previous owner was provably dead (OpenProcess on its PID fails
    outright) yet `start()` only checked lease_expires_at > now, so recovery
    was bounded by the full lease TTL instead of being immediate, unlike the
    sibling DuckDB writer-authority lease in write_service.py which already
    had this dead-owner fast-path."""
    import subprocess
    import sys

    db = tmp_path / "collector.db"
    heartbeat = tmp_path / "heartbeat.json"

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = dead.pid
    assert dead.wait(timeout=10) == 0

    now = dt.datetime(2026, 8, 26, 12, 0, 0, tzinfo=dt.timezone.utc)
    from auto_collector.supervision import ensure_schema

    ensure_schema(db, now=now)
    with sqlite3.connect(str(db)) as con:
        con.execute(
            """
            INSERT INTO collector_instances
            (instance_id, pid, hostname, started_at, heartbeat_at,
             lease_expires_at, state, activity, pending_count, process_start_time)
            VALUES ('dead-instance', ?, 'host', ?, ?, ?, 'RUNNING', 'SLEEPING', NULL, 1)
            """,
            [
                dead_pid,
                now.isoformat(),
                now.isoformat(),
                (now + dt.timedelta(seconds=LEASE_SECONDS - 5)).isoformat(),
            ],
        )

    new_owner = CollectorSupervisor(db, heartbeat)
    # Still well within the dead owner's lease TTL -- must reclaim immediately.
    assert new_owner.start(now + dt.timedelta(seconds=1), start_thread=False)


def test_lease_ownership_and_health(tmp_path) -> None:
    db = tmp_path / "collector.db"
    heartbeat = tmp_path / "heartbeat.json"
    first = CollectorSupervisor(db, heartbeat)
    second = CollectorSupervisor(db, heartbeat)
    now = dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)
    assert first.start(now, start_thread=False)
    assert not second.start(now, start_thread=False)
    snapshot = status_snapshot(db, now)
    assert snapshot["collector_running"]
    assert snapshot["instance"]["instance_id"] == first.instance_id
    first.stop(now + dt.timedelta(seconds=1))
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["state"] == "STOPPED"
    assert payload["lease_expires_at"] == payload["heartbeat_at"]


def test_heartbeat_file_alone_is_never_proof_of_life(tmp_path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text('{"state":"RUNNING","pid":123}', encoding="utf-8")
    snapshot = status_snapshot(tmp_path / "missing.db")
    assert snapshot["collector_running"] is False


def test_startup_reconciles_abandoned_instance_and_job(tmp_path) -> None:
    db = tmp_path / "collector.db"
    heartbeat = tmp_path / "heartbeat.json"
    day = dt.date(2026, 8, 22)
    started = dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)
    clock = ManualClock(started)
    first = CollectorSupervisor(db, heartbeat, clock=clock)
    assert first.start(started, start_thread=False)
    assert first.queue_job("equity", day)
    first.claim_job("equity", day)

    restart = started + dt.timedelta(seconds=LEASE_SECONDS + 1)
    clock.current = restart
    second = CollectorSupervisor(db, heartbeat, clock=clock)
    assert second.start(restart, start_thread=False)
    with sqlite3.connect(str(db)) as con:
        state = con.execute(
            "SELECT status FROM collector_runs WHERE task = 'equity'"
        ).fetchone()[0]
        owner_state = con.execute(
            "SELECT state FROM collector_instances WHERE instance_id = ?",
            [first.instance_id],
        ).fetchone()[0]
    assert state == "ABANDONED"
    assert owner_state == "ABANDONED"
    second.stop(restart)


def test_one_clock_controls_claim_heartbeat_expiry_and_restart(tmp_path) -> None:
    started = dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)
    clock = ManualClock(started)
    db = tmp_path / "collector.db"
    day = started.date()
    first = CollectorSupervisor(db, tmp_path / "heartbeat.json", clock=clock)
    assert first.start(start_thread=False)
    assert first.queue_job("equity", day)
    first.claim_job("equity", day)

    clock.advance(LEASE_SECONDS - 1)
    assert first.reconcile() == (0, 0)
    with sqlite3.connect(str(db)) as con:
        assert con.execute("SELECT status FROM collector_runs").fetchone()[0] == "RUNNING"

    first.renew()
    clock.advance(LEASE_SECONDS - 1)
    assert first.reconcile() == (0, 0)

    clock.advance(2)
    assert first.reconcile() == (1, 1)
    restart = CollectorSupervisor(db, tmp_path / "restart.json", clock=clock)
    assert restart.start(start_thread=False)
    with sqlite3.connect(str(db)) as con:
        assert con.execute("SELECT status FROM collector_runs").fetchone()[0] == "ABANDONED"
    restart.stop()


def test_durable_job_retry_and_completion(tmp_path) -> None:
    db = tmp_path / "collector.db"
    supervisor = CollectorSupervisor(db, tmp_path / "heartbeat.json")
    assert supervisor.start(start_thread=False)
    day = dt.date(2026, 8, 23)
    assert supervisor.queue_job("fo", day)
    supervisor.claim_job("fo", day)
    supervisor.finish_job("fo", day, False, "network", 1.2)
    assert supervisor.queue_job("fo", day)
    supervisor.claim_job("fo", day)
    supervisor.finish_job("fo", day, True, "loaded", 2.5)
    with sqlite3.connect(str(db)) as con:
        row = con.execute(
            "SELECT status, attempt, details FROM collector_runs WHERE task = 'fo'"
        ).fetchone()
    assert row == ("SUCCEEDED", 2, "loaded")
    supervisor.stop()


# --- Watchdog policy (RestartOnFailure cannot recover a killed supervisor on
# Windows -- reproduced live via Last Result == 0x41306 == SCHED_S_TASK_
# TERMINATED, a status class RestartOnFailure does not act on) ------------


def test_watchdog_does_not_restart_a_healthy_collector() -> None:
    should_restart, reason = watchdog_decision(collector_running=True, attempts=[])
    assert should_restart is False
    assert "already running" in reason


def test_watchdog_restarts_a_dead_collector_with_budget_remaining() -> None:
    should_restart, reason = watchdog_decision(collector_running=False, attempts=[])
    assert should_restart is True
    assert "budget remains" in reason


def test_watchdog_fails_closed_after_exhausting_restart_budget() -> None:
    should_restart, reason = watchdog_decision(
        collector_running=False, attempts=[1.0, 2.0, 3.0, 4.0, 5.0], max_restarts=5
    )
    assert should_restart is False
    assert "refusing to restart" in reason


def test_watchdog_restarts_again_once_old_attempts_age_out_of_the_window(tmp_path) -> None:
    """Discriminating case: a persistently-failing collector must eventually
    be retried again once enough time has passed -- fail-closed is bounded,
    not permanent, matching job_supervisor.py's own MAX_RESTARTS/
    RESTART_WINDOW_SECONDS design for the child-process restart loop."""
    state_file = tmp_path / "watchdog_state.json"
    now = 10_000.0
    old_attempts = [now - WATCHDOG_RESTART_WINDOW_SECONDS - 1] * 5
    save_watchdog_attempts(state_file, old_attempts)

    loaded = load_watchdog_attempts(state_file, now=now)
    assert loaded == []  # all pruned, outside the window

    should_restart, _ = watchdog_decision(collector_running=False, attempts=loaded)
    assert should_restart is True


def test_watchdog_attempts_state_is_corruption_safe(tmp_path) -> None:
    state_file = tmp_path / "watchdog_state.json"
    state_file.write_text("not valid json {{{", encoding="utf-8")
    assert load_watchdog_attempts(state_file, now=1000.0) == []


def test_watchdog_attempts_round_trip(tmp_path) -> None:
    state_file = tmp_path / "watchdog_state.json"
    save_watchdog_attempts(state_file, [1.0, 2.0, 3.0])
    assert load_watchdog_attempts(state_file, now=3.0 + 10) == [1.0, 2.0, 3.0]


# --- Watchdog writer-blind-spot: collector lease and WriterService heartbeat
# are two independent threads inside the same collector-run process. A live
# collector lease alone must never be read as proof the writer is alive. ----


def test_watchdog_untouched_when_writer_health_not_checked() -> None:
    """Byte-for-byte back-compat: omitting writer_responsive (None) must
    behave exactly like the pre-existing collector-only policy."""
    should_restart, reason = watchdog_decision(collector_running=True, attempts=[])
    assert should_restart is False
    assert "already running" in reason


def test_watchdog_does_not_restart_a_healthy_writer() -> None:
    """A sleeping-but-healthy collector (writer heartbeats continuously
    regardless of collection activity) must never be mistaken for a dead
    writer."""
    should_restart, reason = watchdog_decision(
        collector_running=True, attempts=[], writer_responsive=True,
    )
    assert should_restart is False
    assert "already running" in reason


def test_watchdog_restarts_when_writer_failed_under_a_live_collector_lease() -> None:
    """The exact production incident: collector lease fresh, but
    writer_runtime.state == FAILED (or its heartbeat is stale) in the same
    process. Must escalate to a restart decision even though the collector
    lease alone says 'running'."""
    should_restart, reason = watchdog_decision(
        collector_running=True, attempts=[], writer_responsive=False,
    )
    assert should_restart is True
    assert "WriterService" in reason or "writer" in reason.lower()


def test_watchdog_writer_failure_respects_bounded_restart_budget() -> None:
    should_restart, reason = watchdog_decision(
        collector_running=True,
        attempts=[1.0, 2.0, 3.0, 4.0, 5.0],
        max_restarts=5,
        writer_responsive=False,
    )
    assert should_restart is False
    assert "refusing to restart" in reason


def test_watchdog_writer_failure_restart_window_exhaustion_then_recovery(tmp_path) -> None:
    """Same bounded-restart mechanism as the collector-down path: exhausted
    budget fails closed, then recovers once old attempts age out -- a single
    shared restart accounting for both failure classes, not a second,
    parallel restart budget."""
    state_file = tmp_path / "watchdog_state.json"
    now = 10_000.0
    save_watchdog_attempts(state_file, [now - 1, now - 2, now - 3, now - 4, now - 5])
    exhausted = load_watchdog_attempts(state_file, now=now)
    should_restart, _reason = watchdog_decision(
        collector_running=True, attempts=exhausted, writer_responsive=False,
    )
    assert should_restart is False

    old_attempts = [now - WATCHDOG_RESTART_WINDOW_SECONDS - 1] * 5
    save_watchdog_attempts(state_file, old_attempts)
    recovered = load_watchdog_attempts(state_file, now=now)
    assert recovered == []
    should_restart, _reason = watchdog_decision(
        collector_running=True, attempts=recovered, writer_responsive=False,
    )
    assert should_restart is True
