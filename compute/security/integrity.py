from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


TRUSTED_KEY_ID = "e205031af11b9da1"
TRUSTED_PUBLIC_KEY_HEX = "1aa6ed25f36c421b0d308c3996f192818826a42145ffe633a7de41085eec5878"


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def integrity_check(*, expected_phase: str = "FINAL") -> dict[str, Any]:
    """`expected_phase` defaults to "FINAL" -- the real installed product
    NEVER passes anything else; `main.py`'s normal `integrity_check()` call
    site is unchanged. The only other caller is `main.py`'s
    "release-openapi-probe" entrypoint, reachable ONLY via that literal argv
    token (never how Task Scheduler or the Tauri parent launch this sidecar
    -- see `python_binary_path()`/`launch_python_subprocess` in
    src-tauri/src/main.rs, which spawn it with no arguments at all in a
    release build). That narrower "PROBE" phase exists solely so
    `release_gate.py`'s build-time OpenAPI smoke test can launch the frozen
    sidecar standalone and get a real (not forged, not bypassed) signature
    check against the subset of release artifacts that legitimately exist
    before `cargo tauri build` has produced the Tauri executable -- see
    `verify_release_assets()` for exactly what PROBE still enforces and what
    it structurally cannot (the Tauri executable and frontend bundle don't
    exist yet at that point in the pipeline)."""
    manifest_path = _manifest_path()
    if manifest_path is None:
        return _degraded("trusted release manifest missing")
    signature_path = manifest_path.with_suffix(".sig")
    root = _protected_root()
    if root is None:
        return _degraded("protected runtime root unavailable")
    executable = os.environ.get("BHARATTERMINAL_TAURI_EXECUTABLE")
    if not executable and expected_phase == "FINAL":
        return _degraded("Tauri executable path unavailable")
    return verify_release_assets(
        manifest_path,
        signature_path,
        root,
        tauri_executable_path=Path(executable) if executable else None,
        expected_phase=expected_phase,
    )


def verify_release_assets(
    manifest_path: Path,
    signature_path: Path,
    protected_root: Path,
    *,
    public_key_hex: str = TRUSTED_PUBLIC_KEY_HEX,
    expected_key_id: str = TRUSTED_KEY_ID,
    tauri_executable_path: Path | None = None,
    expected_phase: str = "FINAL",
) -> dict[str, Any]:
    if expected_phase not in ("FINAL", "PROBE"):
        expected_phase = "FINAL"
    if not manifest_path.is_file():
        return _degraded("trusted release manifest missing")
    if not signature_path.is_file():
        return _degraded("release manifest signature missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _degraded(f"release integrity metadata invalid: {error}")
    if manifest.get("status") != "COHERENT":
        return _degraded("release manifest is not coherent")
    if manifest.get("phase") != expected_phase:
        return _degraded(f"release manifest is not {expected_phase.lower()}")
    if manifest.get("signature_required") is not True:
        return _degraded("release manifest does not require a signature")
    if manifest.get("signing_key_id") != expected_key_id:
        return _degraded("release manifest key is not trusted")

    payload = canonical_manifest_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    reasons: list[str] = []
    if signature.get("status") != "SIGNED":
        reasons.append("release manifest is not signed")
    if signature.get("algorithm") != "Ed25519":
        reasons.append("unsupported release signature algorithm")
    if signature.get("key_id") != expected_key_id:
        reasons.append("release signature key is not trusted")
    if signature.get("manifest_sha256") != digest:
        reasons.append("release manifest digest mismatch")
    if not reasons:
        try:
            raw_signature = base64.b64decode(signature["signature_base64"], validate=True)
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
                raw_signature,
                payload,
            )
        except (InvalidSignature, ValueError, TypeError, KeyError) as error:
            reasons.append(f"release manifest signature invalid: {type(error).__name__}")
    if reasons:
        return _degraded(*reasons)

    if expected_phase == "FINAL":
        executable_record = manifest.get("tauri_executable")
        if not isinstance(executable_record, dict):
            return _degraded("Tauri executable integrity record missing")
        if tauri_executable_path is not None:
            try:
                executable = tauri_executable_path.resolve(strict=True)
                if executable_record.get("name") != executable.name:
                    return _degraded("Tauri executable identity mismatch")
                if executable.stat().st_size != int(executable_record.get("size", -1)):
                    return _degraded("Tauri executable size mismatch")
                if _sha256_tauri_executable(executable) != executable_record.get("sha256"):
                    return _degraded("Tauri executable hash mismatch")
            except (OSError, TypeError, ValueError):
                return _degraded("Tauri executable unavailable")

    root = protected_root.resolve()
    entries = manifest.get("protected_assets")
    protected_roots = manifest.get("protected_roots")
    if not isinstance(entries, list) or not isinstance(protected_roots, list):
        return _degraded("protected asset contract missing")
    expected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return _degraded("protected asset record invalid")
        relative = _safe_relative(entry.get("runtime_path"))
        if relative is None or relative in expected:
            return _degraded("protected asset path invalid or duplicated")
        expected[relative] = entry

    scope_reasons = _protected_scope_reasons(
        set(expected), set(protected_roots), expected_phase=expected_phase
    )
    if scope_reasons:
        return _degraded(*scope_reasons)

    actual: set[str] = set()
    for relative in expected:
        path = root / Path(relative)
        if path.is_file():
            actual.add(relative)
    for raw_root in protected_roots:
        relative_root = _safe_relative(raw_root)
        if relative_root is None:
            return _degraded("protected root path invalid")
        directory = root / Path(relative_root)
        if directory.is_dir():
            actual.update(
                path.relative_to(root).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            )

    missing = sorted(set(expected).difference(actual))
    additions = sorted(actual.difference(expected))
    modifications: list[str] = []
    for relative in sorted(set(expected).intersection(actual)):
        path = root / Path(relative)
        record = expected[relative]
        if path.stat().st_size != int(record.get("size", -1)):
            modifications.append(relative)
            continue
        if _sha256(path) != record.get("sha256"):
            modifications.append(relative)

    if missing or additions or modifications:
        return {
            "status": "DEGRADED",
            "trusted": False,
            "key_id": expected_key_id,
            "checked": sorted(actual),
            "missing": missing,
            "additions": additions,
            "modifications": modifications,
            "reasons": ["protected runtime asset set or content mismatch"],
        }
    return {
        "status": "TRUSTED",
        "trusted": True,
        "key_id": expected_key_id,
        "checked": sorted(actual),
        "missing": [],
        "additions": [],
        "modifications": [],
        "reasons": [],
    }


def _manifest_path() -> Path | None:
    override = os.environ.get("BHARATTERMINAL_RESOURCE_MANIFEST")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    if getattr(sys, "frozen", False):
        resource_root = os.environ.get("BHARATTERMINAL_RESOURCE_ROOT")
        if not resource_root:
            return None
        # Fixed, single canonical location -- NOT an unconstrained rglob.
        # tauri.conf.json declares this manifest as a resource at the
        # literal path "../release/resource-manifest.json"; Tauri's own
        # `_up_` indirection (see src-tauri/src/main.rs's
        # resolve_resource_root, which BHARATTERMINAL_RESOURCE_ROOT is
        # already derived from) means it always lands at exactly
        # "<resource_root>/release/resource-manifest.json" in both a real
        # install and this build's own release-openapi-probe self-test.
        # A recursive rglob() instead required *exactly one* match anywhere
        # under resource_root, which is fragile against cargo/Tauri's own
        # build-staging output -- e.g. `cargo tauri build` leaves stale
        # `target/{debug,release}/_up_/release/resource-manifest.json`
        # copies sitting near a source tree used as a probe resource root,
        # any one of which makes the count != 1 and fails the whole
        # integrity check closed even though the real manifest is right
        # there at the canonical path.
        candidate = Path(resource_root) / "release" / "resource-manifest.json"
        return candidate if candidate.is_file() else None
    candidate = Path(__file__).resolve().parents[2] / "release" / "resource-manifest.json"
    return candidate if candidate.is_file() else None


def _protected_root() -> Path | None:
    override = os.environ.get("BHARATTERMINAL_PROTECTED_ASSET_ROOT")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        value = os.environ.get("BHARATTERMINAL_RESOURCE_ROOT")
        return Path(value) if value else None
    return Path(__file__).resolve().parents[2]


def _safe_relative(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# `cargo tauri bundle` embeds a placeholder `__TAURI_BUNDLE_TYPE_VAR_UNK` at
# link time, then patches those trailing 3 bytes in place to record which
# bundle format actually produced the Tauri executable ("NSS" for NSIS).
# Whichever build step last touched the exe determines which variant is on
# disk when `scripts/release_gate.py` hashes it for the manifest, so this
# same normalization has to run here too before comparing against that
# manifest hash at runtime -- must stay byte-for-byte identical to
# `_normalize_tauri_bundle_type_marker` in scripts/release_gate.py and
# `normalize_tauri_bundle_type_marker` in src-tauri/src/release_integrity.rs.
# Only these two are legitimate Tauri-owned values for this product (NSIS
# only); any other 3-byte value at this position is left untouched so it
# still changes the digest, including a value an attacker deliberately set
# to something outside this set to try to slip through as "just another
# bundle-type variant".
_TAURI_BUNDLE_TYPE_MARKER_PREFIX = b"__TAURI_BUNDLE_TYPE_VAR_"
_TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX = b"UNK"
_TAURI_BUNDLE_TYPE_KNOWN_SUFFIXES = (b"UNK", b"NSS")


def _sha256_tauri_executable(path: Path) -> str:
    data = bytearray(path.read_bytes())
    marker_length = len(_TAURI_BUNDLE_TYPE_MARKER_PREFIX) + len(
        _TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX
    )
    start = 0
    while True:
        index = data.find(_TAURI_BUNDLE_TYPE_MARKER_PREFIX, start)
        if index == -1:
            break
        suffix_start = index + len(_TAURI_BUNDLE_TYPE_MARKER_PREFIX)
        suffix_end = suffix_start + len(_TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX)
        if bytes(data[suffix_start:suffix_end]) in _TAURI_BUNDLE_TYPE_KNOWN_SUFFIXES:
            data[suffix_start:suffix_end] = _TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX
        start = index + marker_length
    return hashlib.sha256(bytes(data)).hexdigest()


def _protected_scope_reasons(
    paths: set[str], roots: set[str], *, expected_phase: str = "FINAL"
) -> list[str]:
    all_paths = paths | {f"{root.rstrip('/')}/" for root in roots if isinstance(root, str)}
    requirements = {
        "Python sidecar executable": lambda value: value.startswith("compute/dist/")
        and value.lower().endswith(".exe"),
        "model assets": lambda value: value.startswith("compute/models/"),
        "database bootstrap": lambda value: value.startswith("bundle-data/"),
        "release identity/configuration": lambda value: value.startswith("release/"),
    }
    # The frontend bundle ("dist/") does not exist yet at the point in the
    # release pipeline where a PROBE-phase manifest is verified (`pnpm build`
    # runs AFTER the openapi smoke test that PROBE exists for) -- FINAL is
    # the only phase that ever ships to a real install, and keeps this
    # requirement exactly as before.
    if expected_phase == "FINAL":
        requirements["frontend bundle"] = lambda value: value.startswith("dist/")
    return [
        f"protected scope omits {name}"
        for name, predicate in requirements.items()
        if not any(predicate(value) for value in all_paths)
    ]


def _degraded(*reasons: str) -> dict[str, Any]:
    return {
        "status": "DEGRADED",
        "trusted": False,
        "key_id": TRUSTED_KEY_ID,
        "checked": [],
        "missing": [],
        "additions": [],
        "modifications": [],
        "reasons": list(reasons),
    }
