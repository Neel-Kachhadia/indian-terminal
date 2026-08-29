from __future__ import annotations

from pathlib import Path


def safe_path(base_dir: Path, user_input: str) -> Path:
    base = base_dir.resolve()
    resolved = (base / user_input).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("Path traversal attempt blocked")
    return resolved
