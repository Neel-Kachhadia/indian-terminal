from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from db.writer_lease import release_writer_lease, require_writer_lease


def test_repeated_writer_lease_metadata_remains_bounded(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    lease_path = database.with_suffix(database.suffix + ".writer.lock")

    for _ in range(20):
        require_writer_lease(database)
        release_writer_lease()

    assert lease_path.stat().st_size == 32
    assert lease_path.read_text(encoding="ascii").strip() == str(os.getpid())


@pytest.mark.recovery
def test_concurrent_process_cannot_acquire_database_writer_lease(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    require_writer_lease(database)
    compute_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(compute_root)
    script = """
import sys
from pathlib import Path
from db.writer_lease import WriterLeaseError, require_writer_lease
try:
    require_writer_lease(Path(sys.argv[1]))
except WriterLeaseError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    try:
        child = subprocess.run(
            [sys.executable, "-c", script, str(database)],
            cwd=compute_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        release_writer_lease()

    assert child.returncode == 0, child.stderr
