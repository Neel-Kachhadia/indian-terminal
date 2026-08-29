from __future__ import annotations

import shutil
from pathlib import Path


DEFAULT_HEADROOM_BYTES = 512 * 1024 * 1024


class LowDiskSpaceError(RuntimeError):
    pass


def free_bytes(path: Path) -> int:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return int(shutil.disk_usage(candidate).free)


def require_free_space(
    path: Path,
    estimated_write_bytes: int,
    operation: str,
    *,
    headroom_bytes: int = DEFAULT_HEADROOM_BYTES,
) -> int:
    estimated = max(0, int(estimated_write_bytes))
    headroom = max(0, int(headroom_bytes))
    available = free_bytes(path)
    required = estimated + headroom
    if available < required:
        raise LowDiskSpaceError(
            f"insufficient disk for {operation}: need {required} bytes, have {available} bytes"
        )
    return available
