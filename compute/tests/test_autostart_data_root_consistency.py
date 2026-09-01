from __future__ import annotations

import sys
import types

import pytest

from auto_collector import setup_autostart


class _FakeWinreg:
    HKEY_CURRENT_USER = object()
    KEY_SET_VALUE = object()
    REG_EXPAND_SZ = object()

    def __init__(self) -> None:
        self.set_calls: list[tuple[str, object, str]] = []

    def CreateKeyEx(self, hive, subkey, reserved, access):
        assert hive is self.HKEY_CURRENT_USER
        assert subkey == "Environment"
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def SetValueEx(self, key, name, reserved, value_type, value):
        assert value_type is self.REG_EXPAND_SZ
        self.set_calls.append((name, key, value))


@pytest.fixture
def fake_winreg(monkeypatch: pytest.MonkeyPatch) -> _FakeWinreg:
    fake = _FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", fake)
    return fake


def test_no_override_set_never_touches_the_registry(
    monkeypatch: pytest.MonkeyPatch, fake_winreg: _FakeWinreg
) -> None:
    monkeypatch.delenv("BHARATTERMINAL_APP_DATA", raising=False)
    setup_autostart._sync_app_data_override()
    assert fake_winreg.set_calls == []


def test_override_present_is_persisted_for_task_scheduler_to_inherit(
    monkeypatch: pytest.MonkeyPatch, fake_winreg: _FakeWinreg
) -> None:
    monkeypatch.setenv("BHARATTERMINAL_APP_DATA", r"D:\BharatTerminal")
    setup_autostart._sync_app_data_override()
    assert fake_winreg.set_calls == [("BHARATTERMINAL_APP_DATA", fake_winreg, r"D:\BharatTerminal")]


def test_real_winreg_is_never_reachable_from_a_test_process() -> None:
    """conftest.py must have replaced sys.modules['winreg'] with a guard
    before this module (or setup_autostart) ever gets a chance to import the
    real one. Reproduced live: a pre-existing test called setup() with
    subprocess mocked but _sync_app_data_override() unmocked, and the real
    winreg call persisted this test session's disposable temp data root into
    the real machine's HKCU\\Environment, corrupting the production
    collector's data-root resolution on its next Task Scheduler launch."""
    import winreg as real_or_guarded_winreg

    with pytest.raises(Exception):
        real_or_guarded_winreg.SetValueEx(None, "x", 0, None, "y")


def test_validate_task_xml_fails_closed_on_a_disabled_task() -> None:
    """Reproduced live: both production tasks were left Enabled=false (from
    an earlier controlled test drill) and validate_installed() reported
    "Task valid" anyway, because validate_task_xml() never inspected the
    Enabled element at all -- the collector could sit fully disabled
    indefinitely with the acceptance tooling reporting all-clear."""
    from auto_collector.setup_autostart import runtime_layout, task_xml, validate_task_xml

    runtime = runtime_layout(
        frozen=True, executable=setup_autostart.Path(r"C:\installed\bharatterminal-compute.exe")
    )
    generated = task_xml(r"HOST\user", runtime)
    settings_start = generated.index("<Settings>")
    xml_text = (
        generated[:settings_start]
        + generated[settings_start:].replace("<Enabled>true</Enabled>", "<Enabled>false</Enabled>", 1)
    )
    errors = validate_task_xml(xml_text, runtime)
    assert any("Enabled" in error for error in errors)


def test_validate_task_xml_accepts_a_genuinely_enabled_task_missing_the_element() -> None:
    """Reproduced live: `schtasks /Query /XML` never round-trips <Enabled>
    for an enabled task -- Windows only emits the element at all once a task
    has been explicitly disabled. A freshly re-registered, genuinely-enabled
    task (Get-ScheduledTask -> State "Ready") therefore has no <Enabled>
    element in its live-queried XML at all, and the naive "must equal the
    literal string true" check added for the sibling test above flagged
    every successful setup() as a failure."""
    from auto_collector.setup_autostart import runtime_layout, task_xml, validate_task_xml

    runtime = runtime_layout(
        frozen=True, executable=setup_autostart.Path(r"C:\installed\bharatterminal-compute.exe")
    )
    generated = task_xml(r"HOST\user", runtime)
    settings_start = generated.index("<Settings>")
    settings_end = generated.index("</Settings>")
    assert "<Enabled>true</Enabled>" in generated[settings_start:settings_end]
    xml_text = (
        generated[:settings_start]
        + generated[settings_start:settings_end].replace("<Enabled>true</Enabled>\n", "", 1)
        + generated[settings_end:]
    )
    assert "<Enabled>true</Enabled>" not in xml_text[settings_start:]
    errors = validate_task_xml(xml_text, runtime)
    assert not any("Enabled" in error for error in errors)
