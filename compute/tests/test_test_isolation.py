from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from conftest import (
    PRODUCTION_DATABASE_PATHS,
    TEST_APP_DATA_ROOT,
    TEST_DATA_ROOT,
    TEST_DB_PATH,
    TEST_LOCAL_ROOT,
    TEST_REDIS_URL,
    TEST_ROAMING_ROOT,
    ProductionDatabaseAccessError,
    make_guarded_duckdb_connect,
    paths_refer_to_same_file,
)
from data.paths import DUCKDB_PATH


@pytest.mark.contract
def test_application_database_is_bound_to_session_isolation_before_collection() -> None:
    assert paths_refer_to_same_file(DUCKDB_PATH, TEST_DB_PATH)
    assert not any(paths_refer_to_same_file(DUCKDB_PATH, path) for path in PRODUCTION_DATABASE_PATHS)


@pytest.mark.contract
def test_all_safety_critical_environment_authorities_are_forced() -> None:
    import os

    assert os.environ["BHARATTERMINAL_DATA_ROOT"] == str(TEST_DATA_ROOT)
    assert os.environ["BHARATTERMINAL_DB_PATH"] == str(TEST_DB_PATH)
    assert os.environ["BHARATTERMINAL_REDIS_URL"] == TEST_REDIS_URL
    assert os.environ["APPDATA"] == str(TEST_ROAMING_ROOT)
    assert os.environ["LOCALAPPDATA"] == str(TEST_LOCAL_ROOT)


@pytest.mark.contract
def test_protected_database_is_rejected_before_open_and_rows_remain_unchanged(tmp_path: Path) -> None:
    # Disposable stand-in proves pre-open rejection without touching a live database.
    protected = tmp_path / "production-stand-in.db"
    connection = duckdb.connect(protected)
    try:
        connection.execute("CREATE TABLE sentinel (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO sentinel VALUES (1), (2)")
        before = connection.execute("SELECT COUNT(*) FROM sentinel").fetchone()[0]
    finally:
        connection.close()

    guarded_connect = make_guarded_duckdb_connect(duckdb.connect, protected_paths=(protected,))
    with pytest.raises(ProductionDatabaseAccessError, match="refused protected production"):
        guarded_connect(protected)

    connection = duckdb.connect(protected, read_only=True)
    try:
        after = connection.execute("SELECT COUNT(*) FROM sentinel").fetchone()[0]
    finally:
        connection.close()
    assert after == before


@pytest.mark.contract
def test_existing_alias_of_protected_database_is_rejected(tmp_path: Path) -> None:
    protected = tmp_path / "protected.db"
    connection = duckdb.connect(protected)
    connection.close()
    alias = tmp_path / "alias.db"
    try:
        alias.symlink_to(protected)
    except OSError:
        try:
            alias.hardlink_to(protected)
        except OSError:
            pytest.skip("test host does not permit filesystem alias creation")

    guarded_connect = make_guarded_duckdb_connect(duckdb.connect, protected_paths=(protected,))
    with pytest.raises(ProductionDatabaseAccessError):
        guarded_connect(alias)
