from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "# GENERATED FILE - DO NOT EDIT\n"
    "# source: pyproject.toml + uv.lock\n"
    "# generation command: python scripts/dependency_exports.py --write\n"
)
EXPORTS = {
    "requirements.txt": ("--no-dev", "--no-hashes"),
    "requirements-dev.txt": ("--all-groups", "--no-hashes"),
    "requirements.lock": ("--no-dev",),
}


def rendered_exports() -> dict[str, bytes]:
    rendered: dict[str, bytes] = {}
    for filename, selection in EXPORTS.items():
        command = [
            "uv",
            "export",
            "--frozen",
            "--no-emit-project",
            "--format",
            "requirements.txt",
            "--no-header",
            *selection,
        ]
        environment = os.environ.copy()
        environment.setdefault(
            "UV_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "bharatterminal-uv-export-cache"),
        )
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            env=environment,
        )
        body = result.stdout.replace(b"\r\n", b"\n")
        rendered[filename] = HEADER.encode("utf-8") + body
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = rendered_exports()
    stale: list[str] = []
    for filename, content in expected.items():
        path = ROOT / filename
        if args.write:
            path.write_bytes(content)
        elif not path.is_file() or path.read_bytes().replace(b"\r\n", b"\n") != content:
            stale.append(filename)
    if stale:
        raise SystemExit("dependency exports stale: " + ", ".join(sorted(stale)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
