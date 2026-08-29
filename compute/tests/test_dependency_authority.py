from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_uv_lock_generated_dependency_exports_are_byte_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/dependency_exports.py"],
        cwd=root,
        check=True,
    )
