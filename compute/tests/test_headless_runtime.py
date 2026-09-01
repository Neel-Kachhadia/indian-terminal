from __future__ import annotations

import subprocess as subprocess_module

import pytest

from auto_collector import job_supervisor, setup_autostart


def test_frozen_build_is_windowed_not_console() -> None:
    """Reproduced live: the installed collector/supervisor/watchdog ran as a
    console-subsystem executable, and Task Scheduler's InteractiveToken logon
    (required for Credential Manager/DPAPI access) does not suppress that
    process's window the way it does for a service-account logon -- a
    console kept popping up while BharatTerminal was installed/running."""
    import build_compute

    source = build_compute.Path(build_compute.__file__).read_text(encoding="utf-8")
    assert '"--windowed"' in source
    assert source.index('"--windowed"') < source.index('"--name"')


def test_spawn_collector_uses_create_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        captured["command"] = command
        return object()

    monkeypatch.setattr(job_supervisor, "_collector_command", lambda: (["exe.exe", "collector-run"], "."))
    monkeypatch.setattr(job_supervisor.subprocess, "Popen", fake_popen)

    job_supervisor._spawn_collector()

    assert captured["creationflags"] & job_supervisor.CREATE_NO_WINDOW
    assert captured["creationflags"] & job_supervisor.CREATE_NEW_PROCESS_GROUP


def test_watchdog_schtasks_run_uses_create_no_window(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from auto_collector import supervision

    captured: dict = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return subprocess_module.CompletedProcess(args, 0, "SUCCESS", "")

    monkeypatch.setattr(job_supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(
        supervision, "status_snapshot", lambda log_db: {"collector_running": False}
    )
    monkeypatch.setattr(supervision, "load_watchdog_attempts", lambda state_file, now=None: [])
    monkeypatch.setattr(supervision, "save_watchdog_attempts", lambda state_file, attempts: None)
    monkeypatch.setattr(supervision, "watchdog_decision", lambda **_kwargs: (True, "collector not running"))

    import data.paths as data_paths

    monkeypatch.setattr(data_paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(job_supervisor, "_configure_lifecycle_logging", lambda _component: None)

    job_supervisor.run_watchdog_check()

    assert captured["creationflags"] & job_supervisor.CREATE_NO_WINDOW


@pytest.mark.parametrize(
    ("factory", "expected_argv_fragment"),
    [
        (lambda: setup_autostart._current_user(), str(setup_autostart.WHOAMI)),
        (lambda: setup_autostart.query_xml(), "/Query"),
        (lambda: setup_autostart._register_task("Fake_Task", "<Task/>"), "/Create"),
        (lambda: setup_autostart.remove(), "/Delete"),
    ],
)
def test_setup_autostart_subprocess_calls_use_create_no_window(
    monkeypatch: pytest.MonkeyPatch, factory, expected_argv_fragment
) -> None:
    """Every setup_autostart call that shells out to a real Windows console
    utility (whoami.exe, schtasks.exe) must suppress its window -- these run
    routinely (every setup()/validate()/remove() call, which itself runs on
    every collector-task CLI invocation from the Tauri GUI), not just once."""
    calls: list[dict] = []

    def fake_run(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return subprocess_module.CompletedProcess(args, 0, "some-user\n", "")

    monkeypatch.setattr(setup_autostart.subprocess, "run", fake_run)
    factory()

    assert calls, "expected at least one subprocess.run call"
    matching = [call for call in calls if expected_argv_fragment in " ".join(call["args"])]
    assert matching, f"no captured call contained {expected_argv_fragment!r}: {calls}"
    for call in matching:
        assert call.get("creationflags") == subprocess_module.CREATE_NO_WINDOW


def test_safe_stream_or_devnull_tolerates_missing_stdio() -> None:
    """A `--windowed` frozen process launched with no stdio redirection (the
    real production Task Scheduler path) has `sys.stdout`/`sys.stderr` set to
    `None` by the interpreter. Every caller that used to assume a real stream
    (`.isatty()`, `.encoding`, `logging.StreamHandler(stream)`,
    `logging.basicConfig(stream=...)`) must keep working against the
    substitute instead of crashing the background collector."""
    from operations.logging_config import safe_stream_or_devnull

    real = object()
    assert safe_stream_or_devnull(real) is real

    substitute = safe_stream_or_devnull(None)
    assert substitute.isatty() is False
    assert substitute.encoding
    substitute.write("ignored")
    substitute.flush()


def test_job_supervisor_lifecycle_logging_is_durable_and_structured(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Headless Task Scheduler processes have no durable stderr.  Lifecycle
    and recovery events must use the same scrubbed rotating JSONL facility
    as the backend, under the production data log directory."""
    import data.paths as data_paths
    import operations.logging_config as logging_config

    captured: dict = {}
    monkeypatch.setattr(data_paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        logging_config,
        "configure_structured_logging",
        lambda path, **kwargs: captured.update(path=path, kwargs=kwargs),
    )

    path = job_supervisor._configure_lifecycle_logging("collector-watchdog")

    assert path == tmp_path / "logs" / "collector-watchdog.jsonl"
    assert captured == {"path": path, "kwargs": {"redirect_prints": False}}
    source = job_supervisor.Path(job_supervisor.__file__).read_text(encoding="utf-8")
    assert "logging.basicConfig(" not in source
    assert '_configure_lifecycle_logging("collector-supervisor")' in source
    assert '_configure_lifecycle_logging("collector-watchdog")' in source
