"""Idempotent Windows Task Scheduler installer for the durable collector."""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path

from defusedxml import ElementTree as ET


TASK_NAME = "BharatTerminal_SmartCollector"
# Second, independent task: recovers the collector when the top-level
# supervisor process itself dies. See auto_collector.supervision's
# "Watchdog" section for why RestartOnFailure on TASK_NAME alone cannot do
# this (reproduced live: Windows classifies a killed run as
# SCHED_S_TASK_TERMINATED, which RestartOnFailure does not act on). A
# TimeTrigger-only task (no LogonTrigger) registers without needing UAC
# elevation -- see _create_task_elevated()'s docstring, which already
# documented this exact asymmetry from the primary task's own install.
WATCHDOG_TASK_NAME = "BharatTerminal_SmartCollector_Watchdog"
WATCHDOG_INTERVAL_MINUTES = 5
LAUNCHER = Path(__file__).with_name("launcher.bat").resolve()
COMPUTE_DIR = LAUNCHER.parent.parent.resolve()
# The Task Scheduler Action invokes this interpreter directly (no cmd.exe,
# no uv.exe ancestor layer) so the process Task Scheduler tracks as the task
# root IS the Job Object owner (job_supervisor.py). job_supervisor.py uses
# only the standard library, so it needs no `uv run` resolution step itself;
# it invokes uv internally for the real collector child. A cmd.exe/uv.exe
# ancestor above the job owner would not be covered by
# JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (that limit only protects the job
# owner's own descendants, never its ancestors), which is exactly how the
# previously reproduced orphan-python.exe-after-force-kill failure occurred.
SUPERVISOR = Path(__file__).with_name("job_supervisor.py").resolve()
VENV_PYTHON = (COMPUTE_DIR / ".venv" / "Scripts" / "python.exe").resolve()
UV_EXE = shutil.which("uv")
SYSTEM_ROOT = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
WHOAMI = (SYSTEM_ROOT / "System32" / "whoami.exe").resolve()
SCHTASKS = (SYSTEM_ROOT / "System32" / "schtasks.exe").resolve()
TASK_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"


@dataclass(frozen=True)
class TaskRuntime:
    command: Path
    arguments: str
    working_directory: Path
    installed: bool


def runtime_layout(
    *,
    frozen: bool | None = None,
    executable: Path | None = None,
) -> TaskRuntime:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if is_frozen:
        command = (executable or Path(sys.executable)).resolve()
        return TaskRuntime(
            command=command,
            arguments="collector-supervisor",
            working_directory=command.parent,
            installed=True,
        )
    return TaskRuntime(
        command=VENV_PYTHON,
        arguments=f'"{SUPERVISOR}"',
        working_directory=COMPUTE_DIR,
        installed=False,
    )


def _current_user() -> str:
    result = subprocess.run(
        [str(WHOAMI)], capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("unable to determine current Windows user")
    return result.stdout.strip()


def validate_environment(runtime: TaskRuntime | None = None) -> list[str]:
    runtime = runtime or runtime_layout()
    errors: list[str] = []
    if os.name != "nt":
        errors.append("Windows Task Scheduler is only supported on Windows")
    if not runtime.command.is_file():
        errors.append(f"collector runtime missing: {runtime.command}")
    if not runtime.working_directory.is_dir():
        errors.append(f"working directory missing: {runtime.working_directory}")
    if runtime.installed:
        serialized = f"{runtime.command} {runtime.arguments} {runtime.working_directory}".casefold()
        for forbidden in (".venv", "uv.exe", "job_supervisor.py", "\\hermes\\", "\\.codex\\"):
            if forbidden in serialized:
                errors.append(f"installed task contains forbidden development path class: {forbidden}")
    else:
        if not SUPERVISOR.is_file():
            errors.append(f"job supervisor missing: {SUPERVISOR}")
        if not VENV_PYTHON.is_file():
            errors.append(f"canonical project venv python missing: {VENV_PYTHON}")
        if not UV_EXE:
            errors.append("canonical uv runtime launcher missing from PATH")
    if not WHOAMI.is_file():
        errors.append(f"whoami missing: {WHOAMI}")
    if not SCHTASKS.is_file():
        errors.append(f"Task Scheduler CLI missing: {SCHTASKS}")
    if not os.environ.get("APPDATA"):
        errors.append("APPDATA is unavailable")
    return errors


def task_xml(user_id: str, runtime: TaskRuntime | None = None) -> str:
    runtime = runtime or runtime_layout()
    # quote=False: these are XML element TEXT content, not attribute values --
    # only &, <, > need escaping there. html.escape()'s default quote=True
    # also rewrites literal `"` to `&quot;`, which XML parsers decode back to
    # `"` correctly (so validate_task_xml(), which reads parsed node.text,
    # never saw a problem) but produces a raw <Arguments> string that no
    # longer contains the literal quoted path Task Scheduler's Action expects
    # to see documented/matched verbatim.
    command = escape(str(runtime.command), quote=False)
    arguments = escape(runtime.arguments, quote=False)
    working_directory = escape(str(runtime.working_directory), quote=False)
    user = escape(user_id, quote=False)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="{TASK_NS}">
  <RegistrationInfo>
    <Description>BharatTerminal durable data collector</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <Delay>PT1M</Delay>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{working_directory}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def watchdog_runtime(runtime: TaskRuntime | None = None) -> TaskRuntime:
    """Same command/working-directory as the primary collector task, just
    invoked with the `collector-watchdog` argv instead of
    `collector-supervisor` -- see main.py's argv dispatch."""
    base = runtime or runtime_layout()
    return TaskRuntime(
        command=base.command,
        arguments="collector-watchdog" if base.installed else f'"{SUPERVISOR}" watchdog',
        working_directory=base.working_directory,
        installed=base.installed,
    )


def watchdog_task_xml(user_id: str, runtime: TaskRuntime | None = None) -> str:
    runtime = watchdog_runtime(runtime)
    command = escape(str(runtime.command), quote=False)
    arguments = escape(runtime.arguments, quote=False)
    working_directory = escape(str(runtime.working_directory), quote=False)
    user = escape(user_id, quote=False)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="{TASK_NS}">
  <RegistrationInfo>
    <Description>BharatTerminal collector watchdog -- recovers the collector if its supervisor process dies unexpectedly (Task Scheduler's own RestartOnFailure cannot do this, see supervision.py)</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Enabled>true</Enabled>
      <!-- Required by the Task Scheduler XML schema for any TimeTrigger.
           Registration fails closed without it ("task XML is missing a
           required element or attribute... StartBoundary"), reproduced
           live on a real install. A fixed anchor safely in the past is the
           standard pattern for an "every N minutes, forever" schedule: the
           Repetition/Interval below is what actually governs firing, this
           boundary only has to be a valid, already past starting point for
           it to originate from. -->
      <StartBoundary>2024-01-01T00:00:00</StartBoundary>
      <Repetition>
        <Interval>PT{WATCHDOG_INTERVAL_MINUTES}M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT2M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{working_directory}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def validate_task_xml(xml_text: str, runtime: TaskRuntime | None = None) -> list[str]:
    runtime = runtime or runtime_layout()
    errors: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        return [f"task XML invalid: {error}"]
    namespace = {"t": TASK_NS}

    def value(path: str) -> str:
        node = root.find(path, namespace)
        return (node.text or "").strip() if node is not None else ""

    expected = {
        ".//t:Exec/t:Command": str(runtime.command),
        ".//t:Exec/t:WorkingDirectory": str(runtime.working_directory),
        ".//t:Settings/t:MultipleInstancesPolicy": "IgnoreNew",
        ".//t:Settings/t:RestartOnFailure/t:Interval": "PT1M",
        ".//t:Settings/t:RestartOnFailure/t:Count": "3",
        ".//t:Settings/t:ExecutionTimeLimit": "PT0S",
    }
    for path, expected_value in expected.items():
        actual = value(path)
        if actual.casefold() != expected_value.casefold():
            errors.append(f"task field mismatch {path}: {actual!r}")
    # `schtasks /Query /XML` never round-trips `<Enabled>` for an enabled
    # task -- Windows only emits the element at all when a task has been
    # explicitly disabled (`<Enabled>false</Enabled>`). Reproduced live:
    # a freshly re-registered, genuinely-enabled task (confirmed via
    # Get-ScheduledTask -> State "Ready") still reported this field as
    # mismatched on every successful setup(), because absent was compared
    # against the literal string "true" instead of being treated as the
    # enabled default. Only an explicit non-true value is a real defect.
    enabled = value(".//t:Settings/t:Enabled")
    if enabled and enabled.casefold() != "true":
        errors.append(f"task field mismatch .//t:Settings/t:Enabled: {enabled!r}")
    arguments = value(".//t:Exec/t:Arguments")
    if arguments.casefold() != runtime.arguments.casefold():
        errors.append("task arguments do not match the expected collector supervisor entrypoint")
    if runtime.installed:
        serialized = f"{value('.//t:Exec/t:Command')} {arguments} {value('.//t:Exec/t:WorkingDirectory')}".casefold()
        if any(token in serialized for token in (".venv", "uv.exe", "job_supervisor.py")):
            errors.append("installed task references a development runtime")
    return errors


def query_xml(task_name: str = TASK_NAME) -> tuple[str | None, str | None]:
    result = subprocess.run(
        [str(SCHTASKS), "/Query", "/TN", task_name, "/XML"],
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        return None, (result.stderr.strip() or result.stdout.strip() or "task not found")
    return result.stdout, None


def validate_installed() -> int:
    runtime = runtime_layout()
    errors = validate_environment(runtime)
    xml_text, query_error = query_xml()
    if query_error:
        errors.append(query_error)
    elif xml_text:
        errors.extend(validate_task_xml(xml_text, runtime))

    watchdog = watchdog_runtime(runtime)
    watchdog_xml, watchdog_query_error = query_xml(WATCHDOG_TASK_NAME)
    if watchdog_query_error:
        errors.append(f"watchdog task: {watchdog_query_error}")
    elif watchdog_xml:
        # Reuse the same field-shape checks by validating against the
        # watchdog's own expected command/arguments/working-directory.
        watchdog_errors = validate_task_xml(watchdog_xml, watchdog)
        # The primary task's RestartOnFailure/ExecutionTimeLimit checks in
        # validate_task_xml() don't apply to the watchdog task's simpler
        # settings -- filter those two expected mismatches out, they're
        # by design (see watchdog_task_xml()), not a real defect.
        errors.extend(
            error
            for error in watchdog_errors
            if "RestartOnFailure" not in error and "ExecutionTimeLimit" not in error
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Task valid: {TASK_NAME}")
    print(f"Runtime: {runtime.command}")
    print(f"Working directory: {runtime.working_directory}")
    print("Restart: 3 attempts, 1 minute interval")
    print(f"Watchdog valid: {WATCHDOG_TASK_NAME} (every {WATCHDOG_INTERVAL_MINUTES}m)")
    return 0


def _access_denied(message: str) -> bool:
    return "access is denied" in message.lower()


def _create_task_elevated(xml_path: Path, task_name: str = TASK_NAME) -> tuple[bool, str]:
    """Retry task registration via a UAC-elevated `schtasks.exe` child.

    Windows blocks a LogonTrigger task registration (XML import AND the
    plain `/SC ONLOGON` form -- both reproduced) from an unelevated process
    even when the calling user account is itself an administrator, because
    UAC hands out a filtered, non-elevated token by default. This is a real
    OS-level boundary this account hit live, not a bug in the XML: a
    TimeTrigger-only task with the exact same Principal/UserId registers
    fine unelevated. `ShellExecuteExW` with `lpVerb="runas"` is the standard
    Windows pattern for a narrowly-scoped one-time elevated operation (it
    elevates only this `schtasks.exe` child, never this process or any
    other part of the product) -- it raises a real UAC consent prompt only a
    human can approve, so this can only ever succeed with the signed-in
    user's explicit one-time consent, never silently.
    """
    if os.name != "nt":
        return False, "elevation only supported on Windows"
    import ctypes
    from ctypes import wintypes

    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    SW_HIDE = 0

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", ctypes.c_ulong),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hKeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIconOrMonitor", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    parameters = f'/Create /TN "{task_name}" /XML "{xml_path}" /F'
    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = str(SCHTASKS)
    info.lpParameters = parameters
    info.nShow = SW_HIDE
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        error = ctypes.get_last_error()
        # ERROR_CANCELLED (1223): the user declined the UAC consent prompt.
        if error == 1223:
            return False, "user declined the elevation consent prompt"
        return False, f"elevated launch failed: WinError {error}"
    WAIT_TIMEOUT_MS = 30_000
    WAIT_OBJECT_0 = 0
    wait_result = ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, WAIT_TIMEOUT_MS)
    if wait_result != WAIT_OBJECT_0:
        ctypes.windll.kernel32.CloseHandle(info.hProcess)
        return False, "elevated schtasks did not complete in time"
    exit_code = wintypes.DWORD()
    ctypes.windll.kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code))
    ctypes.windll.kernel32.CloseHandle(info.hProcess)
    if exit_code.value != 0:
        return False, f"elevated schtasks exited with code {exit_code.value}"
    return True, ""


def _register_task(task_name: str, xml_text: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-16",
        suffix=".xml",
        delete=False,
    ) as temporary:
        temporary.write(xml_text)
        xml_path = Path(temporary.name)
    try:
        result = subprocess.run(
            [str(SCHTASKS), "/Create", "/TN", task_name, "/XML", str(xml_path), "/F"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0 and _access_denied(result.stderr or result.stdout or ""):
            elevated_ok, elevated_error = _create_task_elevated(xml_path, task_name)
            if not elevated_ok:
                return False, (
                    f"{result.stderr.strip() or result.stdout.strip()}; "
                    f"elevated retry failed: {elevated_error}"
                )
            return True, ""
    finally:
        xml_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    return True, ""


def _sync_app_data_override() -> None:
    """Persist BHARATTERMINAL_APP_DATA to the user's registry environment
    block so Task Scheduler-launched runs resolve the same data root as
    this (interactive) process.

    Task Scheduler never inherits a variable set with the shell builtin
    `set` -- that is local to the setting process and whatever it directly
    launches as a child. It DOES build each task action's environment
    fresh from the user's persisted `HKCU\\Environment` block at spawn
    time, the same registry key `setx` writes to. Without this, a machine
    where an operator has set BHARATTERMINAL_APP_DATA in their interactive
    session (to point the collector at a non-default data root) ends up
    with two silently diverging data directories: the interactive/dev
    launch path honors the override, while every Task Scheduler-driven run
    -- the actual unattended production path -- falls back to
    data.paths._base_dir()'s default (%APPDATA%\\BharatTerminal) instead,
    reproduced live: the real production DuckDB went untouched for hours
    while a healthy, correctly-recovering collector kept writing to the
    wrong directory with no error, no crash, and no log difference to
    distinguish it from success.

    A machine with no override set is untouched: normal installs keep
    using the documented %APPDATA%\\BharatTerminal default everywhere.
    """
    override = os.environ.get("BHARATTERMINAL_APP_DATA")
    if not override:
        return
    import ctypes
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "BHARATTERMINAL_APP_DATA", 0, winreg.REG_EXPAND_SZ, override)
    # Broadcast so already-running processes that read the environment at
    # startup (e.g. Explorer, for anything launched from the Start Menu)
    # see the change too. Task Scheduler itself re-reads the registry
    # directly on every launch and does not need this broadcast, but a
    # human-launched process later in the same session might.
    HWND_BROADCAST = 0xFFFF
    WM_SETTINGCHANGE = 0x001A
    SMTO_ABORTIFHUNG = 0x0002
    ctypes.windll.user32.SendMessageTimeoutW(
        HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", SMTO_ABORTIFHUNG, 5000, None
    )


def setup() -> int:
    runtime = runtime_layout()
    errors = validate_environment(runtime)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    _sync_app_data_override()

    ok, error = _register_task(TASK_NAME, task_xml(_current_user(), runtime))
    if not ok:
        print(f"ERROR: {error}")
        return 1

    watchdog_ok, watchdog_error = _register_task(
        WATCHDOG_TASK_NAME, watchdog_task_xml(_current_user(), runtime)
    )
    if not watchdog_ok:
        print(f"ERROR: watchdog task registration failed: {watchdog_error}")
        return 1

    if validate_installed() != 0:
        return 1
    print(f"Task installed idempotently: {TASK_NAME}")
    print(f"Start: schtasks /Run /TN {TASK_NAME}")
    return 0


def remove() -> int:
    overall = 0
    for task_name in (TASK_NAME, WATCHDOG_TASK_NAME):
        result = subprocess.run(
            [str(SCHTASKS), "/Delete", "/TN", task_name, "/F"],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            print(f"Removed: {task_name}")
        else:
            print(result.stderr.strip() or result.stdout.strip() or f"{task_name} not found")
            overall = result.returncode or 1
    return overall


def cli(arguments: list[str] | None = None) -> int:
    selected = list(arguments or [])
    if selected in (["remove"], ["--remove"]):
        return remove()
    if selected in (["validate"], ["--validate"]):
        return validate_installed()
    if selected not in ([], ["install"], ["setup"]):
        print("Usage: collector-task [install|validate|remove]")
        return 2
    return setup()


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
