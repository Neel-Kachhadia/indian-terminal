from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Callable, Iterable

# Loaded before pytest collects test modules. Establish process-wide isolation
# before importing DuckDB or any application module; a fixture is too late for
# modules that bind persistent paths during collection.
_ORIGINAL_APPDATA = os.environ.get("APPDATA")
_ORIGINAL_LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
_TEST_ROOT_OWNER = tempfile.TemporaryDirectory(prefix="bharatterminal-tests-")
TEST_APP_DATA_ROOT = Path(_TEST_ROOT_OWNER.name).resolve(strict=True)
TEST_DATA_ROOT = TEST_APP_DATA_ROOT / "data"
TEST_DB_PATH = TEST_DATA_ROOT / "terminal.db"
TEST_ROAMING_ROOT = TEST_APP_DATA_ROOT / "roaming"
TEST_LOCAL_ROOT = TEST_APP_DATA_ROOT / "local"
TEST_REDIS_CREDENTIAL = secrets.token_urlsafe(32)
TEST_REDIS_URL = f"redis://:{TEST_REDIS_CREDENTIAL}@127.0.0.1:1/15"
os.environ["BHARATTERMINAL_APP_DATA"] = str(TEST_APP_DATA_ROOT)
os.environ["BHARATTERMINAL_DATA_ROOT"] = str(TEST_DATA_ROOT)
os.environ["BHARATTERMINAL_DB_PATH"] = str(TEST_DB_PATH)
os.environ["APPDATA"] = str(TEST_ROAMING_ROOT)
os.environ["LOCALAPPDATA"] = str(TEST_LOCAL_ROOT)
os.environ["BHARATTERMINAL_ALLOW_UNTRUSTED_DEV"] = "1"
os.environ["BHARATTERMINAL_TEST_HARNESS_ID"] = secrets.token_hex(16)
os.environ["BHARATTERMINAL_TEST_REDIS_URL"] = TEST_REDIS_URL
os.environ["BHARATTERMINAL_REDIS_URL"] = TEST_REDIS_URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _canonical_path(path: os.PathLike[str] | str) -> str:
    return os.path.normcase(os.path.abspath(os.path.realpath(os.path.expandvars(os.fspath(path)))))


def _production_database_paths() -> tuple[Path, ...]:
    candidates = {
        _canonical_path(WORKSPACE_ROOT / "data" / "terminal.db"),
        _canonical_path(PROJECT_ROOT / "data" / "terminal.db"),
    }
    for appdata in (_ORIGINAL_APPDATA, _ORIGINAL_LOCALAPPDATA):
        if appdata:
            candidates.add(
                _canonical_path(Path(appdata) / "BharatTerminal" / "data" / "terminal.db")
            )
    return tuple(Path(candidate) for candidate in sorted(candidates))


PRODUCTION_DATABASE_PATHS = _production_database_paths()


def pytest_configure(config) -> None:
    # A unique, session-owned base avoids inheriting stale Windows ACLs from
    # pytest's shared per-user temp directory.
    config.option.basetemp = str(TEST_APP_DATA_ROOT / "pytest-tmp")


class ProductionDatabaseAccessError(RuntimeError):
    """Raised before a test process can open a protected production database."""


def paths_refer_to_same_file(left: os.PathLike[str] | str, right: os.PathLike[str] | str) -> bool:
    if _canonical_path(left) == _canonical_path(right):
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError, ValueError):
        return False


def assert_test_database_path(
    database: os.PathLike[str] | str,
    *,
    protected_paths: Iterable[os.PathLike[str] | str] = PRODUCTION_DATABASE_PATHS,
) -> None:
    raw = os.fspath(database)
    if raw == ":memory:":
        return
    if any(paths_refer_to_same_file(raw, protected) for protected in protected_paths):
        raise ProductionDatabaseAccessError(
            f"test process refused protected production database path: {_canonical_path(raw)}"
        )


def make_guarded_duckdb_connect(
    connect: Callable[..., object],
    *,
    protected_paths: Iterable[os.PathLike[str] | str] = PRODUCTION_DATABASE_PATHS,
) -> Callable[..., object]:
    protected = tuple(protected_paths)

    def guarded_connect(database: os.PathLike[str] | str = ":memory:", *args, **kwargs):
        assert_test_database_path(database, protected_paths=protected)
        return connect(os.fspath(database), *args, **kwargs)

    return guarded_connect


import duckdb  # noqa: E402 -- test root must exist before this import

_REAL_DUCKDB_CONNECT = duckdb.connect
duckdb.connect = make_guarded_duckdb_connect(_REAL_DUCKDB_CONNECT)  # type: ignore[assignment]


class ProductionRegistryAccessError(RuntimeError):
    """Raised before a test process can write to the real Windows registry.

    Reproduced live: BHARATTERMINAL_APP_DATA is set process-wide above (line
    22) so every test resolves data paths under the disposable
    TEST_APP_DATA_ROOT instead of production. That is correct for path
    resolution, but auto_collector.setup_autostart._sync_app_data_override()
    treats *any* set BHARATTERMINAL_APP_DATA as an operator's real intent and
    persists it to HKCU\\Environment so Task Scheduler inherits it -- a test
    that exercises setup() without mocking that one call silently overwrote
    the real production registry value with this session's disposable temp
    path, corrupting the real collector's data-root resolution on next
    launch. sys.modules is pre-seeded with this guard so `import winreg`
    inside that function hits it by default; a test that legitimately wants
    to exercise the real persistence logic must explicitly install its own
    fake via monkeypatch.setitem(sys.modules, "winreg", fake) (see
    test_autostart_data_root_consistency.py), which shadows this guard only
    for that test and is reverted automatically at teardown.
    """


import sys as _sys  # noqa: E402
import winreg as _real_winreg  # noqa: E402

# Functions that can mutate the real registry. Read-only calls (OpenKey,
# QueryValueEx, ConnectRegistry, EnumKey, CloseKey, ...) pass through
# untouched -- third-party libraries legitimately read the registry during
# collection (e.g. dateutil.tz.win reads HKEY_LOCAL_MACHINE timezone data via
# winreg.ConnectRegistry when pandas is imported), and blocking those breaks
# unrelated tests for no safety benefit.
_MUTATING_WINREG_CALLS = frozenset(
    {
        "CreateKey",
        "CreateKeyEx",
        "SetValue",
        "SetValueEx",
        "SetValueExW",
        "DeleteKey",
        "DeleteKeyEx",
        "DeleteValue",
    }
)


class _GuardedWinreg:
    """Proxies the real winreg module, refusing only registry writes.

    Reproduced live: BHARATTERMINAL_APP_DATA is set process-wide above (line
    22) so every test resolves data paths under the disposable
    TEST_APP_DATA_ROOT instead of production. That is correct for path
    resolution, but auto_collector.setup_autostart._sync_app_data_override()
    treats *any* set BHARATTERMINAL_APP_DATA as an operator's real intent and
    persists it to HKCU\\Environment so Task Scheduler inherits it -- a test
    that exercises setup() without mocking that one call silently overwrote
    the real production registry value with this session's disposable temp
    path, corrupting the real collector's data-root resolution on next
    launch. A test that legitimately wants to exercise the real persistence
    logic must explicitly install its own fake via
    monkeypatch.setitem(sys.modules, "winreg", fake) (see
    test_autostart_data_root_consistency.py), which shadows this guard only
    for that test and is reverted automatically at teardown.
    """

    def __getattr__(self, name: str):
        if name in _MUTATING_WINREG_CALLS:
            def _refuse(*_args, **_kwargs):
                raise ProductionRegistryAccessError(
                    f"test process attempted real winreg.{name}(...) -- mock "
                    "winreg explicitly if this test intends to exercise "
                    "registry-writing logic"
                )

            return _refuse
        return getattr(_real_winreg, name)


# Unconditional, not setdefault: some stdlib/third-party module (e.g.
# mimetypes, dateutil.tz.win) may have already cached the real winreg in
# sys.modules by this point, and the whole point of this guard is that no
# code path -- first-party or incidental -- writes to the real registry
# during a test run.
_sys.modules["winreg"] = _GuardedWinreg()

import pytest  # noqa: E402

from db.writer_lease import release_writer_lease  # noqa: E402


@pytest.fixture
def isolated_app_data(tmp_path: Path) -> Path:
    root = (tmp_path / "app-data").resolve()
    assert_test_database_path(root / "data" / "terminal.db")
    root.mkdir()
    return root


def pytest_sessionstart(session: pytest.Session) -> None:
    configured = Path(os.environ["BHARATTERMINAL_APP_DATA"])
    if not paths_refer_to_same_file(configured, TEST_APP_DATA_ROOT):
        raise pytest.UsageError("test bootstrap data root changed before session start")
    assert_test_database_path(configured / "data" / "terminal.db")
    expected = {
        "BHARATTERMINAL_DATA_ROOT": str(TEST_DATA_ROOT),
        "BHARATTERMINAL_DB_PATH": str(TEST_DB_PATH),
        "BHARATTERMINAL_REDIS_URL": TEST_REDIS_URL,
        "APPDATA": str(TEST_ROAMING_ROOT),
        "LOCALAPPDATA": str(TEST_LOCAL_ROOT),
    }
    changed = [name for name, value in expected.items() if os.environ.get(name) != value]
    if changed:
        raise pytest.UsageError(f"test bootstrap isolation changed: {', '.join(changed)}")


@pytest.fixture(autouse=True)
def _release_database_writer_after_test():
    yield
    release_writer_lease()
