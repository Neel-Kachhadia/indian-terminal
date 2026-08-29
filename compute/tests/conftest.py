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
