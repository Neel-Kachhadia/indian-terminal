from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

import auto_collector.setup_autostart as setup_autostart
import auto_collector.smart_collector as smart_collector
from auto_collector.setup_autostart import (
    SUPERVISOR,
    TASK_NAME,
    TASK_NS,
    WATCHDOG_TASK_NAME,
    _access_denied,
    runtime_layout,
    setup,
    task_xml,
    validate_task_xml,
    watchdog_runtime,
    watchdog_task_xml,
)
from auto_collector.smart_collector import (
    MAX_BACK_DAYS,
    _collector_read_dataset,
    missing_trading_days,
    run_logged_task,
    run_ohlc_forecasts,
    run_task,
)


def test_task_xml_has_validated_paths_and_restart_policy() -> None:
    xml = task_xml(r"HOST\user")
    assert f'xmlns="{TASK_NS}"' in xml
    assert str(SUPERVISOR) in xml
    assert validate_task_xml(xml) == []


def test_watchdog_task_name_is_distinct_from_primary_task() -> None:
    assert WATCHDOG_TASK_NAME != TASK_NAME
    assert WATCHDOG_TASK_NAME.startswith(TASK_NAME)


def test_watchdog_runtime_targets_the_watchdog_argv_not_the_supervisor() -> None:
    installed = runtime_layout(frozen=True, executable=SUPERVISOR.parent / "app.exe")
    watchdog = watchdog_runtime(installed)
    assert watchdog.command == installed.command
    assert watchdog.arguments == "collector-watchdog"
    assert watchdog.arguments != installed.arguments


def test_watchdog_task_xml_uses_a_time_trigger_and_ignores_new_overlap() -> None:
    installed = runtime_layout(frozen=True, executable=SUPERVISOR.parent / "app.exe")
    xml = watchdog_task_xml(r"HOST\user", installed)
    assert f'xmlns="{TASK_NS}"' in xml
    assert "<TimeTrigger>" in xml
    assert "<LogonTrigger>" not in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "collector-watchdog" in xml
    # Bounded execution so a hung watchdog run can't block every subsequent
    # tick forever under IgnoreNew.
    assert "<ExecutionTimeLimit>PT2M</ExecutionTimeLimit>" in xml
    # Reproduces a real deployed defect: a TimeTrigger without a
    # StartBoundary is invalid per the Task Scheduler XML schema --
    # registration failed closed on a real install with "the task XML is
    # missing a required element or attribute... StartBoundary", silently
    # leaving the whole watchdog recovery mechanism unregistered/inert.
    assert "<StartBoundary>" in xml


def test_watchdog_task_xml_registers_on_real_windows_task_scheduler() -> None:
    """Reproduces a real deployed defect (see test above) with the strongest
    available proof: actually asks the real OS Task Scheduler to accept this
    exact XML via `schtasks /create /xml`, using a disposable task name so it
    never collides with or disturbs the real installed
    BharatTerminal_SmartCollector_Watchdog task. A syntactically-plausible
    XML string is not sufficient evidence -- only the real Windows schema
    validator is authoritative for "is this a legal task definition"."""
    import subprocess
    import sys
    import tempfile
    import uuid
    from pathlib import Path as _Path

    if sys.platform != "win32":
        import pytest as _pytest

        _pytest.skip("Task Scheduler is Windows-only")

    from auto_collector.setup_autostart import _current_user

    installed = runtime_layout(frozen=True, executable=SUPERVISOR.parent / "app.exe")
    # Real resolvable account, not the "HOST\user" placeholder other tests in
    # this file use for pure string assertions -- schtasks /create with an
    # InteractiveToken LogonType needs a genuine account to resolve, and this
    # test's whole point is to isolate whether the StartBoundary fix itself
    # is what the real Task Scheduler validator accepts.
    xml = watchdog_task_xml(_current_user(), installed)
    disposable_name = f"BharatTerminal_TestWatchdogXml_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as temp_dir:
        xml_path = _Path(temp_dir) / "task.xml"
        xml_path.write_text(xml, encoding="utf-16")
        try:
            result = subprocess.run(
                ["schtasks", "/create", "/tn", disposable_name, "/xml", str(xml_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"real Task Scheduler rejected the generated XML: "
                f"{result.stdout!r} {result.stderr!r}"
            )
        finally:
            subprocess.run(
                ["schtasks", "/delete", "/tn", disposable_name, "/f"],
                capture_output=True,
                text=True,
                timeout=30,
            )


def test_task_xml_arguments_are_not_double_quoted() -> None:
    r"""Reproduces a real deployed defect: the Action is a direct .venv
    python.exe invocation with no cmd.exe layer (unlike the old
    launcher.bat-via-cmd.exe Action, which legitimately needed
    `""{launcher}""` double-quoting so cmd.exe's own `/c` argument parsing
    left one literal quoted path). Carrying that same double-quote template
    over to a direct-exe Action produces `<Arguments>""D:\...py""</Arguments>`,
    which Windows' CreateProcess argv parser does not resolve to a plain
    path -- the installed task's every real run failed with
    Last Result 2 (ERROR_FILE_NOT_FOUND) until this was caught and fixed."""
    xml = task_xml(r"HOST\user")
    assert '""' not in xml
    assert f'<Arguments>"{SUPERVISOR}"</Arguments>' in xml


def test_task_xml_validation_rejects_wrong_working_directory() -> None:
    xml = task_xml(r"HOST\user").replace(
        "<WorkingDirectory>",
        "<WorkingDirectory>C:\\wrong<!--",
        1,
    )
    errors = validate_task_xml(xml)
    assert errors


@pytest.mark.parametrize(
    "candidate",
    ["ohlcv; DROP TABLE ohlcv", "provider_table", "signals-history", "OHLCV"],
)
def test_collector_dynamic_dataset_identifiers_are_closed(candidate: str) -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        _collector_read_dataset(candidate)


def test_stock_training_catches_up_when_saturday_is_missed(monkeypatch, tmp_path) -> None:
    """Reproduces a live gap: weekly_retrain (Sunday) had a staleness-based
    catch-up fallback (age_days > 7) but stock_training (Saturday-only) had
    none -- if the machine was off every Saturday for a stretch, stock model
    training would never run again, silently, forever."""
    log_db = tmp_path / "collector_log.db"
    monkeypatch.setattr(smart_collector, "LOG_DB", log_db)
    smart_collector.ensure_log_db()

    monkeypatch.setattr(smart_collector, "ist_today", lambda: date(2026, 8, 25))  # Tuesday

    # never run: must catch up
    assert smart_collector.days_since_last_success("stock_training") is None

    with sqlite3.connect(str(log_db)) as con:
        con.execute(
            "INSERT INTO collector_runs (task, date, status, started_at, attempt) "
            "VALUES ('stock_training', '2026-08-09', 'SUCCEEDED', '2026-08-09T10:00:00+00:00', 1)"
        )
    # last success 16 days ago (missed Saturdays since): must catch up
    assert smart_collector.days_since_last_success("stock_training") == 16

    with sqlite3.connect(str(log_db)) as con:
        con.execute("UPDATE collector_runs SET date = ? WHERE task = 'stock_training'", ["2026-08-22"])
    # last success 3 days ago: no catch-up needed yet
    assert smart_collector.days_since_last_success("stock_training") == 3


def test_weekly_retrain_catch_up_uses_own_ledger_not_global_model_mtime(monkeypatch, tmp_path) -> None:
    """Reproduces a live defect: the Sunday catch-up fallback used
    model_age_days() -- the newest .skops mtime across every model family
    under MODELS_DIR, not just weekly_retrain's own NIFTY models. A real
    weekly_retrain run FAILED (missing skops dependency); ten days later
    model_age_days() still read back 0 because an unrelated family's artifact
    had been touched more recently, so the "catch up when stale" condition
    never fired and would have silently waited a full week for the next
    Sunday. days_since_last_success() -- this task's own run ledger -- is not
    fooled by unrelated artifact writes elsewhere in the tree."""
    log_db = tmp_path / "collector_log.db"
    monkeypatch.setattr(smart_collector, "LOG_DB", log_db)
    smart_collector.ensure_log_db()
    monkeypatch.setattr(smart_collector, "ist_today", lambda: date(2026, 8, 26))  # Wednesday

    assert smart_collector.days_since_last_success("weekly_retrain") is None

    with sqlite3.connect(str(log_db)) as con:
        con.execute(
            "INSERT INTO collector_runs (task, date, status, started_at, attempt) "
            "VALUES ('weekly_retrain', '2026-08-16', 'SUCCEEDED', '2026-08-16T12:09:15+00:00', 1)"
        )
        con.execute(
            "INSERT INTO collector_runs (task, date, status, started_at, attempt) "
            "VALUES ('weekly_retrain', '2026-08-23', 'FAILED', '2026-08-23T06:34:15+00:00', 1)"
        )
    # last success was 10 days ago even though the most recent attempt failed
    # only 3 days ago -- staleness must be measured from the last SUCCESS.
    assert smart_collector.days_since_last_success("weekly_retrain") == 10


def test_ohlc_forecasts_total_failure_is_not_reported_as_success(monkeypatch) -> None:
    """Reproduces a live defect: every symbol failing (e.g. the writer-process
    same-connection-config clash previously in db/duckdb_client.py) used to
    return a details string with no 'not_available_yet' marker, so run_task()
    logged the day as SUCCEEDED with zero rows produced."""
    monkeypatch.setattr(smart_collector, "forecast_symbols", lambda limit=50: ["NIFTY", "BANKNIFTY"])

    def _always_fails(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Connection Error: Can't open a connection to same database file")

    monkeypatch.setattr(smart_collector, "submit_write", _always_fails)

    details = run_ohlc_forecasts(date(2026, 8, 25))
    assert "not_available_yet" in details

    success, _ = run_task("ohlc_forecasts", date(2026, 8, 25))
    assert success is False


def _weekday_is_trading_day(day: date) -> bool:
    return day.weekday() < 5


def test_missing_trading_days_recovers_gap_older_than_max_back_across_cycles(monkeypatch) -> None:
    """Reproduces a live defect: missing_trading_days() used to walk backward
    from "today", capped at max_back. For downtime longer than max_back
    trading days, that only ever discovered the most recent max_back missing
    days; since then advances to that batch's newest date, so the oldest days
    in the gap were never found by any future cycle -- a laptop off for more
    than ~2 weeks would permanently lose the earliest days of the gap."""
    monkeypatch.setattr(smart_collector, "is_trading_day", _weekday_is_trading_day)
    today = date(2026, 8, 25)  # Tuesday
    monkeypatch.setattr(smart_collector, "ist_today", lambda: today)

    since = today - timedelta(days=25)  # ~18 trading days ago
    first_cycle = missing_trading_days(since, max_back=MAX_BACK_DAYS)
    assert len(first_cycle) == MAX_BACK_DAYS
    # The oldest outstanding day must be discovered first (ascending from since),
    # not skipped in favor of the most recent max_back days.
    assert first_cycle == sorted(first_cycle)
    assert first_cycle[0] < today - timedelta(days=MAX_BACK_DAYS)

    # Simulate the collector completing this batch, then running a second
    # cycle: every trading day between the original `since` and `today` must
    # eventually be covered with no gap and no duplicate.
    second_cycle = missing_trading_days(first_cycle[-1], max_back=MAX_BACK_DAYS)
    all_days = first_cycle + second_cycle
    assert len(all_days) == len(set(all_days))
    trading_days_in_range = [
        since + timedelta(days=offset)
        for offset in range(1, (today - since).days + 1)
        if _weekday_is_trading_day(since + timedelta(days=offset))
    ]
    assert sorted(all_days) == trading_days_in_range


def test_missing_trading_days_weekend_gap_flags_only_monday(monkeypatch) -> None:
    monkeypatch.setattr(smart_collector, "is_trading_day", _weekday_is_trading_day)
    friday = date(2026, 8, 21)
    monday = date(2026, 8, 24)
    monkeypatch.setattr(smart_collector, "ist_today", lambda: monday)
    assert missing_trading_days(friday, max_back=MAX_BACK_DAYS) == [monday]


def test_missing_trading_days_fresh_install_seeds_recent_days_backward(monkeypatch) -> None:
    monkeypatch.setattr(smart_collector, "is_trading_day", _weekday_is_trading_day)
    today = date(2026, 8, 25)
    monkeypatch.setattr(smart_collector, "ist_today", lambda: today)
    result = missing_trading_days(None, max_back=5)
    assert len(result) == 5
    assert result == sorted(result)
    assert result[-1] == today


def test_collector_read_failures_never_become_missing_dates_or_zero_rows(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("writer unavailable")

    monkeypatch.setattr(smart_collector, "submit_read", unavailable)

    with pytest.raises(RuntimeError, match="writer unavailable"):
        smart_collector.latest_date_in("ohlcv")
    with pytest.raises(RuntimeError, match="writer unavailable"):
        smart_collector.row_count_for_date("ohlcv", date(2026, 8, 25))


def test_collector_rejects_malformed_read_responses(monkeypatch) -> None:
    monkeypatch.setattr(
        smart_collector,
        "submit_read",
        lambda *_args, **_kwargs: {"date": "not-a-date"},
    )
    with pytest.raises(ValueError, match="invalid latest date"):
        smart_collector.latest_date_in("ohlcv")

    monkeypatch.setattr(
        smart_collector,
        "submit_read",
        lambda *_args, **_kwargs: {},
    )
    with pytest.raises(ValueError, match="no row count"):
        smart_collector.row_count_for_date("ohlcv", date(2026, 8, 25))


def test_signal_pending_check_uses_its_decision_eligible_source_domain(monkeypatch) -> None:
    """Raw OHLCV can advance before its rows become decision eligible. The
    signal scheduler must compare signals with the exact view used by the
    backfill engine, otherwise it records a zero-row false success and masks
    the still-ineligible date for the rest of the day."""
    today = date(2026, 8, 31)
    monkeypatch.setattr(smart_collector, "ist_today", lambda: today)
    monkeypatch.setattr(smart_collector, "ist_hour", lambda: 12.0)
    monkeypatch.setattr(smart_collector, "instrument_master_age_hours", lambda: 0.0)
    monkeypatch.setattr(smart_collector, "task_done", lambda *_args: False)
    monkeypatch.setattr(smart_collector, "days_since_last_success", lambda *_args: 0)

    def latest(dataset: str, symbol: str | None = None):
        values = {
            ("ohlcv", None): today,
            ("signals_source_ohlcv", "NIFTY 50"): date(2026, 8, 26),
            ("signals_history", "NIFTY"): date(2026, 8, 26),
            ("bif_history", "NIFTY"): date(2026, 8, 26),
        }
        return values[(dataset, symbol)]

    monkeypatch.setattr(smart_collector, "latest_date_in", latest)

    pending = smart_collector.get_pending_tasks()

    assert all(item["task"] != "signals" for item in pending)
    assert _collector_read_dataset("signals_source_ohlcv") == (
        "decision_eligible_ohlcv",
        "date",
    )


def test_signal_pending_check_uses_the_real_decision_eligible_symbol_spelling(monkeypatch) -> None:
    """decision_eligible_ohlcv (data/market_data_quality.py's view SQL) selects
    `symbol` straight through from raw ohlcv with no alias canonicalization, so
    index rows sit under the literal source spelling "NIFTY 50" -- never the
    bare "NIFTY" signals_history/bif_history key their own rows under.
    Querying this dataset with "NIFTY" always returns None, which makes
    `if latest_eq and (...)` structurally unreachable no matter how fresh the
    real decision-eligible index data is. Reproduced live 2026-09-02: with
    genuinely fresh index bhav data, signals_history stayed frozen 11 days
    behind because this exact check was silently never true."""
    today = date(2026, 9, 2)
    monkeypatch.setattr(smart_collector, "ist_today", lambda: today)
    monkeypatch.setattr(smart_collector, "ist_hour", lambda: 12.0)
    monkeypatch.setattr(smart_collector, "instrument_master_age_hours", lambda: 0.0)
    monkeypatch.setattr(smart_collector, "task_done", lambda *_args: False)
    monkeypatch.setattr(smart_collector, "days_since_last_success", lambda *_args: 0)

    def latest(dataset: str, symbol: str | None = None):
        values = {
            ("ohlcv", None): today,
            ("signals_source_ohlcv", "NIFTY 50"): today,
            ("signals_history", "NIFTY"): date(2026, 8, 22),
            ("bif_history", "NIFTY"): date(2026, 8, 22),
        }
        return values[(dataset, symbol)]

    monkeypatch.setattr(smart_collector, "latest_date_in", latest)

    pending = smart_collector.get_pending_tasks()

    signal_tasks = [item for item in pending if item["task"] == "signals"]
    assert signal_tasks, "fresh decision-eligible NIFTY 50 data must schedule a signals catch-up task"
    assert signal_tasks[0]["date"] == today


def test_manual_rerun_of_already_succeeded_task_does_not_crash(monkeypatch, tmp_path) -> None:
    """Reproduces a real live crash: `--task NAME` (a manual/operator re-run,
    unlike the automatic pending-cycle path which never re-invokes an
    already-done task) called log_start() unconditionally. queue_job()
    correctly leaves an already-SUCCEEDED ledger row alone and returns False
    instead of transitioning it to QUEUED, but the old run_logged_task()
    ignored that and called claim_job() unconditionally, which requires a
    QUEUED row and crashed with `RuntimeError: collector job transition
    lost: stub_task/... QUEUED -> RUNNING` on any manual re-run of a task
    already completed today."""
    log_db = tmp_path / "collector_log.db"
    heartbeat = tmp_path / "heartbeat.json"
    monkeypatch.setattr(smart_collector, "LOG_DB", log_db)
    monkeypatch.setattr(smart_collector, "HEARTBEAT_FILE", heartbeat)
    monkeypatch.setattr(smart_collector, "run_task", lambda task, for_date: (True, "stub ok"))

    assert smart_collector.start_supervision()
    try:
        today = smart_collector.ist_today()
        first = run_logged_task("stub_task", today)
        assert first == (True, "stub ok")

        # The manual re-run path: must report the existing SUCCEEDED state,
        # not crash.
        second = run_logged_task("stub_task", today)
        assert second == (True, "already_succeeded_today")
    finally:
        smart_collector.stop_supervision()


def test_ohlc_forecasts_partial_success_still_reports_success(monkeypatch) -> None:
    monkeypatch.setattr(smart_collector, "forecast_symbols", lambda limit=50: ["NIFTY", "BANKNIFTY"])

    calls = {"n": 0}

    def _fail_first(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {}

    monkeypatch.setattr(smart_collector, "submit_write", _fail_first)

    details = run_ohlc_forecasts(date(2026, 8, 25))
    assert "not_available_yet" not in details

    success, _ = run_task("ohlc_forecasts", date(2026, 8, 25))
    assert success is True


def test_access_denied_matches_only_that_error_class() -> None:
    assert _access_denied("ERROR: ERROR: Access is denied.")
    assert _access_denied("access IS DENIED")
    assert not _access_denied("ERROR: task not found")
    assert not _access_denied("")


def test_setup_retries_via_elevation_when_schtasks_reports_access_denied(monkeypatch) -> None:
    """Reproduces a real live defect: Windows blocks LogonTrigger task
    registration (both XML import and the plain `/SC ONLOGON` form -- both
    reproduced directly against schtasks.exe) from an unelevated process
    even when the calling account is itself an administrator, because UAC
    hands out a filtered token by default. A TimeTrigger-only task with the
    identical Principal registers fine unelevated, isolating the cause to
    LogonTrigger specifically, not the account or the XML content. setup()
    must retry once via a UAC-elevated schtasks child rather than failing
    outright."""
    import subprocess as subprocess_module

    monkeypatch.setattr(setup_autostart, "validate_environment", lambda runtime=None: [])
    monkeypatch.setattr(setup_autostart, "validate_installed", lambda: 0)
    monkeypatch.setattr(setup_autostart, "_current_user", lambda: r"HOST\user")
    # Not under test here (see test_autostart_data_root_consistency.py); must
    # be isolated like every other real side-effecting call setup() makes,
    # or it hits the real registry (see conftest.py's winreg guard).
    monkeypatch.setattr(setup_autostart, "_sync_app_data_override", lambda: None)

    calls: list[str] = []

    def fake_run(args, **_kwargs):
        calls.append("unelevated")
        return subprocess_module.CompletedProcess(args, 1, "", "ERROR: Access is denied.")

    monkeypatch.setattr(setup_autostart.subprocess, "run", fake_run)

    def fake_elevated(_xml_path, _task_name=setup_autostart.TASK_NAME):
        calls.append("elevated")
        return True, ""

    monkeypatch.setattr(setup_autostart, "_create_task_elevated", fake_elevated)

    # setup() now registers TWO independent tasks (the primary collector
    # task and the watchdog task that recovers it -- see
    # WATCHDOG_TASK_NAME), each hitting this same unelevated-then-elevated
    # retry path once.
    assert setup() == 0
    assert calls == ["unelevated", "elevated", "unelevated", "elevated"]


def test_setup_reports_elevation_decline_without_crashing(monkeypatch) -> None:
    import subprocess as subprocess_module

    monkeypatch.setattr(setup_autostart, "validate_environment", lambda runtime=None: [])
    monkeypatch.setattr(setup_autostart, "_current_user", lambda: r"HOST\user")
    monkeypatch.setattr(setup_autostart, "_sync_app_data_override", lambda: None)
    monkeypatch.setattr(
        setup_autostart.subprocess,
        "run",
        lambda args, **_kwargs: subprocess_module.CompletedProcess(
            args, 1, "", "ERROR: Access is denied."
        ),
    )
    monkeypatch.setattr(
        setup_autostart,
        "_create_task_elevated",
        lambda _xml_path, _task_name=setup_autostart.TASK_NAME: (
            False,
            "user declined the elevation consent prompt",
        ),
    )

    assert setup() == 1


def test_setup_does_not_attempt_elevation_for_unrelated_failures(monkeypatch) -> None:
    import subprocess as subprocess_module

    monkeypatch.setattr(setup_autostart, "validate_environment", lambda runtime=None: [])
    monkeypatch.setattr(setup_autostart, "_current_user", lambda: r"HOST\user")
    monkeypatch.setattr(setup_autostart, "_sync_app_data_override", lambda: None)
    monkeypatch.setattr(
        setup_autostart.subprocess,
        "run",
        lambda args, **_kwargs: subprocess_module.CompletedProcess(
            args, 1, "", "ERROR: The system cannot find the file specified."
        ),
    )

    def _unexpected_elevation(_xml_path, _task_name=setup_autostart.TASK_NAME):
        raise AssertionError("must not attempt elevation for a non-access-denied failure")

    monkeypatch.setattr(setup_autostart, "_create_task_elevated", _unexpected_elevation)

    assert setup() == 1
