from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

# Any production module opening the canonical live DuckDB file directly
# (`duckdb.connect(`) can collide with a different process holding it as
# writer authority -- DuckDB has no cross-process concurrent-reader mode.
# Normal callers must go through db.duckdb_client.open_read_connection() /
# db.write_service.resolve_read() instead, which proactively decide
# local-vs-remote before ever opening a second raw connection. This is a
# tiny, explicit allowlist of the modules that ARE the database authority
# layer itself -- not a directory-wide exemption.
ALLOWED_DIRECT_CONNECT_FILES = frozenset(
    {
        "db/write_service.py",
        "db/duckdb_client.py",
        "db/migrations.py",
    }
)

_EXCLUDED_DIR_PARTS = frozenset(
    {
        ".venv",
        ".venv-canonical",
        ".venv-clean",
        ".venv-release",
        ".uv-cache",
        "node_modules",
        "tests",
        ".test-tmp",
        "__pycache__",
    }
)

_CONNECT_PATTERN = re.compile(r"duckdb\.connect\s*\(")


def _production_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        relative_parts = path.relative_to(root).parts
        if any(
            part in _EXCLUDED_DIR_PARTS
            or part.startswith((".venv", ".uv-cache", ".test-tmp"))
            for part in relative_parts
        ):
            continue
        files.append(path)
    return files


@pytest.mark.contract
def test_no_production_module_opens_duckdb_directly_outside_allowlist() -> None:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in _production_python_files(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        if not _CONNECT_PATTERN.search(source):
            continue
        if relative in ALLOWED_DIRECT_CONNECT_FILES:
            continue
        violations.append(relative)
    assert violations == [], (
        "production module(s) open DuckDB directly outside the tiny "
        "authority-layer allowlist -- route through "
        "db.duckdb_client.open_read_connection() or "
        "db.write_service.resolve_read()/submit_write() instead: "
        f"{violations}"
    )


@pytest.mark.contract
def test_allowlist_entries_still_exist_and_still_need_the_exemption() -> None:
    """Keeps the allowlist honest: fails if an entry no longer exists (typo,
    rename) or no longer actually calls duckdb.connect( (the exemption is
    stale and should be removed)."""
    root = Path(__file__).resolve().parents[1]
    for relative in ALLOWED_DIRECT_CONNECT_FILES:
        path = root / relative
        assert path.is_file(), f"allowlisted file no longer exists: {relative}"
        source = path.read_text(encoding="utf-8")
        assert _CONNECT_PATTERN.search(source), (
            f"allowlisted file no longer calls duckdb.connect( -- remove "
            f"the now-stale exemption: {relative}"
        )


def test_operator_monitor_uses_typed_read_authority(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from auto_collector import monitor

    captured: dict[str, Any] = {}

    def _summary(operation: Any, payload: dict[str, Any]) -> dict[str, list[Any]]:
        captured.update(operation=operation, payload=payload)
        return {
            "eq": [10, "2026-08-01", "2026-08-28", 1],
            "fo": [20, "2026-08-01", "2026-08-28", 1],
            "intra": [0, None, None],
            "sig": [30, "2026-08-28"],
            "bif": [1, "2026-08-28"],
            "fii": [2, "2026-08-28"],
            "ohlc_fc": [3, "2026-08-29"],
            "instr": [100, 2],
            "gift": [0, None, None],
        }

    monkeypatch.setattr(monitor, "resolve_read", _summary)
    monitor.show_database_state()

    assert captured == {
        "operation": monitor.ReadOperation.LOAD_DATA_SUMMARY_STATS,
        "payload": {},
    }
    assert "DuckDB unavailable" not in capsys.readouterr().out
