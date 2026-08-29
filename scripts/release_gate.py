from __future__ import annotations

import argparse
import base64
from collections import deque
import gzip
import hashlib
import json
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
SPEC_PATH = RELEASE_DIR / "release-spec.json"
IDENTITY_PATH = RELEASE_DIR / "release-identity.json"
REQUIRED_API_PATH = RELEASE_DIR / "required-api.json"
PACKAGED_OPENAPI_PATH = RELEASE_DIR / "packaged-openapi.json"
RESOURCE_MANIFEST_PATH = RELEASE_DIR / "resource-manifest.json"
RESOURCE_SIGNATURE_PATH = RELEASE_DIR / "resource-manifest.sig"
TAURI_EXECUTABLE_PATH = ROOT / "src-tauri" / "target" / "release" / "indian-terminal.exe"
INTEGRITY_PUBLIC_KEY_PATH = ROOT / "contracts" / "trusted-release-key.json"
SIGNING_KEY_SERVICE = "BharatTerminal.ReleaseSigning"
SIGNING_KEY_ACCOUNT = "ed25519-private-seed-v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_signing_key() -> Ed25519PrivateKey:
    encoded = os.environ.get("BHARATTERMINAL_RELEASE_SIGNING_KEY", "").strip()
    if not encoded:
        import keyring

        encoded = keyring.get_password(SIGNING_KEY_SERVICE, SIGNING_KEY_ACCOUNT) or ""
    try:
        seed = bytes.fromhex(encoded)
    except ValueError as error:
        raise RuntimeError("release signing key is not valid hexadecimal") from error
    if len(seed) != 32:
        raise RuntimeError(
            "release signing key unavailable; provision a 32-byte Ed25519 seed in Windows Credential Manager"
        )
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    trusted = read_json(INTEGRITY_PUBLIC_KEY_PATH)
    actual_public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    if actual_public != trusted.get("public_key_hex"):
        raise RuntimeError("release signing key does not match the trusted public key")
    return private_key


def sign_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    private_key = load_signing_key()
    trusted = read_json(INTEGRITY_PUBLIC_KEY_PATH)
    payload = canonical_json_bytes(manifest)
    signature = {
        "status": "SIGNED",
        "algorithm": "Ed25519",
        "key_id": trusted["key_id"],
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "signature_base64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    write_json(RESOURCE_SIGNATURE_PATH, signature)
    return signature


def verify_manifest_signature(manifest: dict[str, Any]) -> None:
    trusted = read_json(INTEGRITY_PUBLIC_KEY_PATH)
    signature = read_json(RESOURCE_SIGNATURE_PATH)
    payload = canonical_json_bytes(manifest)
    if signature.get("status") != "SIGNED" or signature.get("algorithm") != "Ed25519":
        raise RuntimeError("release manifest signature is missing or unsupported")
    if signature.get("key_id") != trusted.get("key_id"):
        raise RuntimeError("release manifest signing key is not trusted")
    if signature.get("manifest_sha256") != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("release manifest signed digest mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted["public_key_hex"])).verify(
            base64.b64decode(signature["signature_base64"], validate=True),
            payload,
        )
    except (InvalidSignature, ValueError, TypeError, KeyError) as error:
        raise RuntimeError("release manifest signature verification failed") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# `cargo tauri bundle` embeds a placeholder `__TAURI_BUNDLE_TYPE_VAR_UNK` at
# link time, then patches those trailing 3 bytes in place to identify the
# bundle format actually produced ("NSS" for NSIS here -- confirmed by
# diffing a freshly-linked exe against the same exe after `cargo tauri
# bundle --bundles nsis`, and matches that command's own "Patching ... with
# bundle type information: nsis" log line). Whether TAURI_EXECUTABLE_PATH on
# disk is pre- or post-patch at the moment this script hashes it depends on
# exactly which build step ran last, which is not something this pipeline
# can pin down reliably across `cargo tauri build --no-bundle` followed by
# `cargo tauri bundle` (a relink of the same object files reintroduces the
# unpatched placeholder). Canonicalizing this ONE known, Tauri-owned,
# fixed-length, fixed-offset-pattern marker before hashing -- the same
# principle Authenticode uses to exclude its own signature field from its
# own file hash -- makes the hash stable regardless of patch timing while
# still catching any OTHER byte change, including this exact field being set
# to some third, unrecognized value.
_TAURI_BUNDLE_TYPE_MARKER_PREFIX = b"__TAURI_BUNDLE_TYPE_VAR_"
_TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX = b"UNK"
# Only these two are legitimate Tauri-owned values for THIS product (we only
# ever ship NSIS): "UNK" is the pre-patch linker placeholder, "NSS" is what
# `cargo tauri bundle --bundles nsis` patches it to. Any OTHER 3-byte value
# at this position -- including a suffix deliberately set to something
# outside this set specifically to slip through as "just another bundle-type
# variant" -- must NOT be normalized away, or this stops being a real
# integrity check. Must stay byte-for-byte identical to the matching
# KNOWN_SUFFIXES allowlist in src-tauri/src/release_integrity.rs.
_TAURI_BUNDLE_TYPE_KNOWN_SUFFIXES = (b"UNK", b"NSS")


def _normalize_tauri_bundle_type_marker(data: bytes) -> bytes:
    marker_length = len(_TAURI_BUNDLE_TYPE_MARKER_PREFIX) + len(
        _TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX
    )
    normalized = bytearray(data)
    start = 0
    while True:
        index = normalized.find(_TAURI_BUNDLE_TYPE_MARKER_PREFIX, start)
        if index == -1:
            break
        suffix_start = index + len(_TAURI_BUNDLE_TYPE_MARKER_PREFIX)
        suffix_end = suffix_start + len(_TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX)
        if bytes(normalized[suffix_start:suffix_end]) in _TAURI_BUNDLE_TYPE_KNOWN_SUFFIXES:
            normalized[suffix_start:suffix_end] = _TAURI_BUNDLE_TYPE_CANONICAL_SUFFIX
        start = index + marker_length
    return bytes(normalized)


def sha256_tauri_executable(path: Path) -> str:
    return hashlib.sha256(_normalize_tauri_bundle_type_marker(path.read_bytes())).hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def source_files() -> list[Path]:
    excluded = {
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        ".venv-clean",
        ".venv-release",
        "__pycache__",
        "build",
        "dist",
        "logs",
        "node_modules",
        "saved_models",
        "target",
    }

    files: list[Path] = []
    for base in (
        ROOT / "compute",
        ROOT / "contracts",
        ROOT / "public",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "src-tauri" / "src",
    ):
        for current, directories, names in os.walk(base, topdown=True):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in excluded
                and not directory.startswith((".", "_"))
            )
            current_path = Path(current)
            files.extend(current_path / name for name in sorted(names))
    files.extend(
        ROOT / relative
        for relative in (
            "package.json",
            "pnpm-lock.yaml",
            ".gitignore",
            "index.html",
            "tailwind.config.ts",
            "tsconfig.json",
            "vite.config.ts",
            "src-tauri/Cargo.toml",
            "src-tauri/Cargo.lock",
            "src-tauri/tauri.conf.json",
            "src-tauri/build.rs",
            "src-tauri/installer-hooks.nsh",
            "build_release.ps1",
            "release/release-spec.json",
            "release/required-api.json",
        )
    )
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def cargo_version() -> str:
    cargo = tomllib.loads((ROOT / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8"))
    return str(cargo["package"]["version"])


def validate_versions(spec: dict[str, Any]) -> tuple[str, str, str]:
    frontend = str(read_json(ROOT / "package.json")["version"])
    rust = cargo_version()
    tauri = str(read_json(ROOT / "src-tauri" / "tauri.conf.json")["version"])
    release = str(spec["release_version"])
    mismatches = {
        "frontend": frontend,
        "rust": rust,
        "tauri": tauri,
    }
    bad = {name: version for name, version in mismatches.items() if version != release}
    if bad:
        raise RuntimeError(f"release version skew: expected {release}, got {bad}")
    return frontend, rust, str(spec["python_api_version"])


def bootstrap_path() -> Path:
    gzip_path = ROOT / "bundle-data" / "terminal.db.gz"
    if gzip_path.is_file():
        return gzip_path
    raise RuntimeError("bootstrap database archive missing")


def validate_bootstrap(path: Path) -> None:
    if path.stat().st_size < 1024:
        raise RuntimeError(f"bootstrap database archive is implausibly small: {path}")
    with gzip.open(path, "rb") as stream:
        if not stream.read(16):
            raise RuntimeError("bootstrap database archive is empty")


def generate_identity() -> dict[str, Any]:
    spec = read_json(SPEC_PATH)
    frontend, rust, python_api = validate_versions(spec)
    bootstrap = bootstrap_path()
    validate_bootstrap(bootstrap)
    model_root = ROOT / "compute" / "models"
    identity = {
        "release_version": str(spec["release_version"]),
        "frontend_version": frontend,
        "rust_version": rust,
        "python_api_version": python_api,
        "database_schema_version": int(spec["database_schema_version"]),
        "bootstrap_data_version": str(spec["bootstrap_data_version"]),
        "bootstrap_data_sha256": sha256_file(bootstrap),
        "model_manifest_version": str(spec["model_manifest_version"]),
        "model_manifest_sha256": sha256_tree(model_root),
        "integrity_manifest_version": str(spec["integrity_manifest_version"]),
        "source_sha256": source_fingerprint(),
        "build_status": "COHERENT",
    }
    write_json(IDENTITY_PATH, identity)
    return identity


def validate_openapi(schema: dict[str, Any], required: dict[str, Any] | None = None) -> None:
    required = required or read_json(REQUIRED_API_PATH)
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("OpenAPI document has no paths object")
    failures: list[str] = []
    for item in required.get("operations", []):
        path = str(item["path"])
        method = str(item["method"]).lower()
        operation = paths.get(path, {}).get(method)
        label = f"{method.upper()} {path}"
        if not isinstance(operation, dict):
            failures.append(f"missing {label}")
            continue
        responses = operation.get("responses", {})
        if "200" not in responses and "201" not in responses:
            failures.append(f"{label} has no success response")
        if item.get("request_body") and "requestBody" not in operation:
            failures.append(f"{label} has no request body schema")
    if failures:
        raise RuntimeError("OpenAPI compatibility failed:\n" + "\n".join(failures))


def validate_source_openapi() -> dict[str, Any]:
    compute = ROOT / "compute"
    sys.path.insert(0, str(compute))
    previous = os.environ.get("BHARATTERMINAL_APP_DATA")
    with tempfile.TemporaryDirectory(prefix="bharatterminal-source-api-") as temp_dir:
        os.environ["BHARATTERMINAL_APP_DATA"] = temp_dir
        try:
            from main import app  # noqa: PLC0415

            schema = app.openapi()
            validate_openapi(schema)
            return schema
        finally:
            if previous is None:
                os.environ.pop("BHARATTERMINAL_APP_DATA", None)
            else:
                os.environ["BHARATTERMINAL_APP_DATA"] = previous


def _drain(stream: Any, lines: queue.Queue[str]) -> None:
    for line in iter(stream.readline, ""):
        lines.put(line.rstrip("\r\n"))


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
        taskkill = system_root / "System32" / "taskkill.exe"
        if not taskkill.is_file():
            raise RuntimeError(f"trusted taskkill binary missing: {taskkill}")
        subprocess.run(
            [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    elif process.poll() is None:
        process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def obtain_packaged_openapi(sidecar: Path, timeout_seconds: int = 180) -> dict[str, Any]:
    if not sidecar.is_file():
        raise RuntimeError(f"sidecar missing: {sidecar}")
    port = 18765
    with tempfile.TemporaryDirectory(prefix="bharatterminal-release-gate-") as temp_dir:
        env = os.environ.copy()
        env["BHARATTERMINAL_APP_DATA"] = str(Path(temp_dir) / "app-data")
        env["BHARATTERMINAL_PORT"] = str(port)
        # Points the sidecar's real integrity_check() at this repo's own
        # release/ dir (a PROBE-phase manifest must already be there --
        # `generate-probe-manifest` runs immediately before this step) so it
        # is genuinely, cryptographically self-verified rather than launched
        # with checks bypassed. See security.integrity.integrity_check.
        env["BHARATTERMINAL_RESOURCE_ROOT"] = str(ROOT)
        token = secrets.token_hex(32)
        env["BHARATTERMINAL_PROCESS_TOKEN"] = token
        process = subprocess.Popen(
            [str(sidecar), "release-openapi-probe"],
            cwd=temp_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines: queue.Queue[str] = queue.Queue()
        assert process.stdout is not None and process.stderr is not None
        drain_threads = [
            threading.Thread(target=_drain, args=(process.stdout, lines), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, lines), daemon=True),
        ]
        for drain_thread in drain_threads:
            drain_thread.start()
        deadline = time.monotonic() + timeout_seconds
        recent_output: deque[str] = deque(maxlen=50)
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    while True:
                        try:
                            recent_output.append(lines.get_nowait())
                        except queue.Empty:
                            break
                    details = "\n".join(recent_output)
                    raise RuntimeError(
                        f"packaged sidecar exited early with code {process.returncode}"
                        + (f"\n{details}" if details else "")
                    )
                try:
                    line = lines.get(timeout=0.25)
                except queue.Empty:
                    line = ""
                if line:
                    recent_output.append(line)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/openapi.json",
                    headers={"X-Process-Token": token},
                )
                try:
                    with urllib.request.urlopen(request, timeout=2) as response:
                        schema = json.load(response)
                    validate_openapi(schema)
                    write_json(PACKAGED_OPENAPI_PATH, schema)
                    return schema
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    continue
            raise RuntimeError("packaged sidecar OpenAPI timed out")
        finally:
            _terminate_process_tree(process)
            process.stdout.close()
            process.stderr.close()
            for drain_thread in drain_threads:
                drain_thread.join(timeout=5)


def artifact_record(path: Path, version: str, metadata: dict[str, Any]) -> dict[str, Any]:
    digest = sha256_tauri_executable(path) if path == TAURI_EXECUTABLE_PATH else sha256_file(path)
    return {
        "artifact": path.relative_to(ROOT).as_posix(),
        "version": version,
        "size": path.stat().st_size,
        "sha256": digest,
        "compatibility": metadata,
    }


# `compute/dist/` on disk in the SOURCE tree holds two copies of the
# compute sidecar (see compute/build_compute.py): the plain-named
# `bharatterminal-compute.exe`, which tauri.conf.json's `resources` array
# bundles under the resource root at "compute/dist/bharatterminal-compute.exe"
# (verifiable at runtime), and a target-triple-suffixed copy that exists
# ONLY to satisfy Tauri's `externalBin` sidecar-bundling convention at BUILD
# time -- Tauri installs THAT one to the app's top-level install directory
# (see src-tauri/src/main.rs's `release_sidecar_candidates()`, which finds
# it there, not under the resource root), never under
# "compute/dist/<name>" in the installed resource tree. Walking
# `compute/dist/` blindly for protected assets picked up both, and the
# runtime integrity check then looked for the triple-suffixed one under the
# resource root, where it was never actually placed -- reproduced live: a
# real installed build panicked with "protected asset missing:
# compute/dist/bharatterminal-compute-x86_64-pc-windows-msvc.exe" on first
# launch.
#
# That exclusion is only correct for FINAL: FINAL's protected_root is the
# real bundled resource directory, which genuinely never contains the
# triple-suffixed copy. PROBE and PREBUNDLE instead verify against the raw
# SOURCE tree (release_gate.py invoked unfrozen, `_protected_root()`'s
# non-frozen branch returns the repo root) -- there, `compute/dist/` is
# `compute/build_compute.py`'s own build output directory and genuinely
# holds BOTH copies side by side, since Tauri hasn't run its bundler/
# externalBin staging step yet. Applying the same exclusion unconditionally
# at those phases produces the mirror-image bug: the file is present on
# disk under a scanned protected_root but missing from `protected_assets`,
# so `verify_release_assets()` flags it as an unexpected "addition" and
# fails release-openapi-probe on every real build -- reproduced live via
# `build_release.ps1`'s packaged-openapi step.
_EXCLUDED_PROTECTED_ASSET_NAMES_BY_PHASE = {
    "FINAL": frozenset({"bharatterminal-compute-x86_64-pc-windows-msvc.exe"}),
    "PREBUNDLE": frozenset(),
    "PROBE": frozenset(),
}


def protected_asset_paths(*, phase: str = "FINAL") -> list[Path]:
    excluded = _EXCLUDED_PROTECTED_ASSET_NAMES_BY_PHASE.get(phase, frozenset())
    paths = [bootstrap_path(), IDENTITY_PATH]
    if phase != "PROBE":
        # packaged-openapi.json is produced BY the same openapi smoke test
        # that verifies a PROBE-phase manifest, so it cannot be one of its
        # own protected inputs yet.
        paths.append(PACKAGED_OPENAPI_PATH)
    protected_roots = [
        ROOT / "bundle-data",
        ROOT / "compute" / "models",
        ROOT / "compute" / "dist",
    ]
    if phase != "PROBE":
        # `dist/` (the pnpm frontend bundle) does not exist yet at the point
        # in the pipeline where PROBE is verified -- pnpm build runs AFTER
        # the openapi smoke test.
        protected_roots.append(ROOT / "dist")
    for protected_root in protected_roots:
        if protected_root.is_dir():
            paths.extend(
                path
                for path in protected_root.rglob("*")
                if path.is_file() and path.name not in excluded
            )
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def protected_asset_record(path: Path) -> dict[str, Any]:
    return {
        "runtime_path": path.relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def generate_resource_manifest(*, phase: str = "FINAL") -> dict[str, Any]:
    if phase not in {"PROBE", "PREBUNDLE", "FINAL"}:
        raise ValueError("unsupported release manifest phase")
    identity = read_json(IDENTITY_PATH)
    if identity.get("build_status") != "COHERENT":
        raise RuntimeError("release identity is not coherent")
    current_source = source_fingerprint()
    if identity.get("source_sha256") != current_source:
        raise RuntimeError("release identity source hash is stale; regenerate release identity")
    if phase != "PROBE":
        # PROBE exists precisely so the openapi smoke test that PRODUCES
        # packaged-openapi.json can run against a trusted sidecar -- it
        # can't also require that file as one of its own inputs.
        validate_openapi(read_json(PACKAGED_OPENAPI_PATH))
    release_version = str(identity["release_version"])
    paths = [
        ROOT / "compute" / "dist" / "bharatterminal-compute.exe",
        ROOT / "compute" / "dist" / "bharatterminal-compute-x86_64-pc-windows-msvc.exe",
        bootstrap_path(),
        IDENTITY_PATH,
        ROOT / "compute" / "models" / "manifest.json",
    ]
    if phase != "PROBE":
        paths.append(PACKAGED_OPENAPI_PATH)
        # `dist/` (pnpm frontend bundle) doesn't exist yet at PROBE time --
        # pnpm build runs AFTER the openapi smoke test PROBE unblocks.
        paths.extend(sorted(path for path in (ROOT / "dist").rglob("*") if path.is_file()))
    if phase == "FINAL":
        paths.append(TAURI_EXECUTABLE_PATH)
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"required release artifact missing: {path}")
    protected_paths = protected_asset_paths(phase=phase)
    for path in protected_paths:
        if not path.is_file():
            raise RuntimeError(f"protected release asset missing: {path}")
    trusted_key = read_json(INTEGRITY_PUBLIC_KEY_PATH)
    metadata = {
        "python_api_version": identity["python_api_version"],
        "database_schema_version": identity["database_schema_version"],
        "required_api_contract_version": read_json(REQUIRED_API_PATH)["contract_version"],
    }
    protected_roots = ["bundle-data", "compute/models", "compute/dist"]
    if phase != "PROBE":
        protected_roots.append("dist")
    manifest = {
        "release_version": release_version,
        "status": "COHERENT",
        "phase": phase,
        "signature_required": True,
        "signing_key_id": trusted_key["key_id"],
        "identity_sha256": sha256_file(IDENTITY_PATH),
        "source_sha256": current_source,
        "artifacts": [artifact_record(path, release_version, metadata) for path in paths],
        "protected_roots": protected_roots,
        "protected_assets": [protected_asset_record(path) for path in protected_paths],
    }
    if phase == "FINAL":
        manifest["tauri_executable"] = {
            "name": TAURI_EXECUTABLE_PATH.name,
            "size": TAURI_EXECUTABLE_PATH.stat().st_size,
            "sha256": sha256_tauri_executable(TAURI_EXECUTABLE_PATH),
        }
    write_json(RESOURCE_MANIFEST_PATH, manifest)
    sign_manifest(manifest)
    return manifest


def validate_manifest() -> None:
    manifest = read_json(RESOURCE_MANIFEST_PATH)
    identity = read_json(IDENTITY_PATH)
    if manifest.get("status") != "COHERENT":
        raise RuntimeError("resource manifest is not coherent")
    if manifest.get("phase") != "FINAL":
        raise RuntimeError("resource manifest is not final")
    if manifest.get("identity_sha256") != sha256_file(IDENTITY_PATH):
        raise RuntimeError("release identity hash mismatch")
    if manifest.get("source_sha256") != source_fingerprint():
        raise RuntimeError("release source fingerprint mismatch; rebuild all release artifacts")
    if identity.get("source_sha256") != manifest.get("source_sha256"):
        raise RuntimeError("release identity and resource manifest source hashes differ")
    verify_manifest_signature(manifest)
    executable = manifest.get("tauri_executable")
    if not isinstance(executable, dict):
        raise RuntimeError("Tauri executable integrity record missing")
    if executable.get("name") != TAURI_EXECUTABLE_PATH.name:
        raise RuntimeError("Tauri executable identity mismatch")
    if not TAURI_EXECUTABLE_PATH.is_file():
        raise RuntimeError("Tauri executable missing")
    if executable.get("size") != TAURI_EXECUTABLE_PATH.stat().st_size:
        raise RuntimeError("Tauri executable size mismatch")
    if executable.get("sha256") != sha256_tauri_executable(TAURI_EXECUTABLE_PATH):
        raise RuntimeError("Tauri executable hash mismatch")
    for artifact in manifest.get("artifacts", []):
        path = ROOT / str(artifact["artifact"])
        if not path.is_file():
            raise RuntimeError(f"manifest artifact missing: {path}")
        if path.stat().st_size != int(artifact["size"]):
            raise RuntimeError(f"manifest size mismatch: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {path}")
    expected = {
        str(record["runtime_path"]): record for record in manifest.get("protected_assets", [])
    }
    actual: set[str] = set()
    for relative in expected:
        path = ROOT / relative
        if path.is_file():
            actual.add(relative)
    for relative_root in manifest.get("protected_roots", []):
        directory = ROOT / str(relative_root)
        if directory.is_dir():
            actual.update(
                path.relative_to(ROOT).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
                and path.name not in _EXCLUDED_PROTECTED_ASSET_NAMES_BY_PHASE["FINAL"]
            )
    if set(expected) != actual:
        missing = sorted(set(expected).difference(actual))
        additions = sorted(actual.difference(expected))
        raise RuntimeError(f"protected path set mismatch: missing={missing}, additions={additions}")
    for relative, record in expected.items():
        path = ROOT / relative
        if path.stat().st_size != int(record["size"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"protected asset mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "generate-identity",
            "source-openapi",
            "generate-probe-manifest",
            "packaged-openapi",
            "generate-prebundle-manifest",
            "generate-manifest",
            "validate",
        ),
    )
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args()
    if args.action == "generate-identity":
        generate_identity()
    elif args.action == "source-openapi":
        validate_source_openapi()
    elif args.action == "generate-probe-manifest":
        generate_resource_manifest(phase="PROBE")
    elif args.action == "packaged-openapi":
        sidecar = args.sidecar or ROOT / "compute" / "dist" / "bharatterminal-compute.exe"
        obtain_packaged_openapi(sidecar.resolve())
    elif args.action == "generate-prebundle-manifest":
        generate_resource_manifest(phase="PREBUNDLE")
    elif args.action == "generate-manifest":
        generate_resource_manifest()
    else:
        validate_versions(read_json(SPEC_PATH))
        validate_manifest()
        validate_openapi(read_json(PACKAGED_OPENAPI_PATH))
    print(f"release-gate: {args.action} PASS")


if __name__ == "__main__":
    main()
