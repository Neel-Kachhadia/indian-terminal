from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "release_version",
    "frontend_version",
    "rust_version",
    "python_api_version",
    "database_schema_version",
    "bootstrap_data_version",
    "bootstrap_data_sha256",
    "model_manifest_version",
    "model_manifest_sha256",
    "integrity_manifest_version",
    "build_status",
}


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("BHARATTERMINAL_RELEASE_IDENTITY")
    if override:
        candidates.append(Path(override))
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "release-identity.json")
    candidates.append(Path(__file__).resolve().parents[1] / "release" / "release-identity.json")
    return candidates


@lru_cache(maxsize=1)
def load_release_identity() -> dict[str, Any]:
    for path in _candidate_paths():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(REQUIRED_FIELDS.difference(payload))
        if missing:
            raise RuntimeError(f"release identity missing fields: {', '.join(missing)}")
        return payload
    raise RuntimeError("release identity not found")
