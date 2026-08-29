from __future__ import annotations

import base64
import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from security.integrity import canonical_manifest_bytes, verify_release_assets


def _write_signed_contract(root: Path) -> tuple[Path, Path, str, str]:
    protected = root / "models"
    protected.mkdir(parents=True)
    model = protected / "model.bin"
    model.write_bytes(b"trusted-model")
    identity = root / "release-identity.json"
    identity.write_text("identity", encoding="utf-8")
    sidecar = root / "compute" / "dist" / "compute.exe"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"sidecar")
    frontend = root / "dist" / "assets" / "app.js"
    frontend.parent.mkdir(parents=True)
    frontend.write_bytes(b"frontend")
    bootstrap = root / "bundle-data" / "terminal.db.gz"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_bytes(b"bootstrap")
    compute_model = root / "compute" / "models" / "model.bin"
    compute_model.parent.mkdir(parents=True)
    compute_model.write_bytes(model.read_bytes())
    release_identity = root / "release" / "release-identity.json"
    release_identity.parent.mkdir(parents=True)
    release_identity.write_bytes(identity.read_bytes())
    tauri_executable = root / "indian-terminal.exe"
    tauri_executable.write_bytes(b"tauri-executable")
    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    key_id = hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()[:16]
    manifest = {
        "status": "COHERENT",
        "phase": "FINAL",
        "signature_required": True,
        "signing_key_id": key_id,
        "tauri_executable": {
            "name": tauri_executable.name,
            "size": tauri_executable.stat().st_size,
            "sha256": hashlib.sha256(tauri_executable.read_bytes()).hexdigest(),
        },
        "protected_roots": ["models", "compute/models", "compute/dist", "dist", "bundle-data", "release"],
        "protected_assets": [
            {
                "runtime_path": "models/model.bin",
                "size": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            },
            {
                "runtime_path": "release-identity.json",
                "size": identity.stat().st_size,
                "sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
            },
            *[
                {
                    "runtime_path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in (sidecar, frontend, bootstrap)
            ],
            {
                "runtime_path": "compute/models/model.bin",
                "size": compute_model.stat().st_size,
                "sha256": hashlib.sha256(compute_model.read_bytes()).hexdigest(),
            },
            {
                "runtime_path": "release/release-identity.json",
                "size": release_identity.stat().st_size,
                "sha256": hashlib.sha256(release_identity.read_bytes()).hexdigest(),
            },
        ],
    }
    payload = canonical_manifest_bytes(manifest)
    manifest_path = root / "resource-manifest.json"
    signature_path = root / "resource-manifest.sig"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    signature_path.write_text(
        json.dumps(
            {
                "status": "SIGNED",
                "algorithm": "Ed25519",
                "key_id": key_id,
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "signature_base64": base64.b64encode(private.sign(payload)).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, signature_path, public_hex, key_id


def _verify(root: Path) -> dict:
    manifest, signature, public, key_id = _write_signed_contract(root)
    return verify_release_assets(
        manifest,
        signature,
        root,
        public_key_hex=public,
        expected_key_id=key_id,
    )


def test_signed_exact_path_set_is_trusted(tmp_path: Path) -> None:
    assert _verify(tmp_path)["status"] == "TRUSTED"


def test_modification_is_detected(tmp_path: Path) -> None:
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    (tmp_path / "models" / "model.bin").write_bytes(b"modified")
    result = verify_release_assets(
        manifest, signature, tmp_path, public_key_hex=public, expected_key_id=key_id
    )
    assert result["modifications"] == ["models/model.bin"]


@pytest.mark.parametrize(
    "relative",
    [
        "compute/dist/compute.exe",
        "dist/assets/app.js",
        "compute/models/model.bin",
    ],
)
def test_financially_material_asset_modification_is_detected(
    tmp_path: Path, relative: str
) -> None:
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    (tmp_path / relative).write_bytes(b"tampered")
    result = verify_release_assets(
        manifest, signature, tmp_path, public_key_hex=public, expected_key_id=key_id
    )
    assert relative in result["modifications"]


def test_tauri_executable_modification_is_detected(tmp_path: Path) -> None:
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    executable = tmp_path / "indian-terminal.exe"
    executable.write_bytes(b"tampered")
    result = verify_release_assets(
        manifest,
        signature,
        tmp_path,
        public_key_hex=public,
        expected_key_id=key_id,
        tauri_executable_path=executable,
    )
    assert result["status"] == "DEGRADED"
    assert result["reasons"] == ["Tauri executable size mismatch"]


def test_tauri_executable_bundle_type_patch_does_not_trip_the_hash_check(tmp_path: Path) -> None:
    """Reproduces a real deployed defect: `cargo tauri bundle` patches
    indian-terminal.exe in place after release_gate.py may have already
    hashed it for the manifest (a `__TAURI_BUNDLE_TYPE_VAR_UNK` link-time
    placeholder becomes `..._NSS`). This module's own `_sha256_tauri_executable`
    has to normalize that one known marker the same way release_gate.py and
    release_integrity.rs do, or a real installed app's Python sidecar
    rejects its own unmodified executable -- reproduced live: the sidecar
    crash-looped with "protected release integrity verification failed"
    on every real launch until this normalization was added here too."""
    root = tmp_path
    unpatched = b"prefix __TAURI_BUNDLE_TYPE_VAR_UNK suffix"
    patched = b"prefix __TAURI_BUNDLE_TYPE_VAR_NSS suffix"
    executable = root / "indian-terminal.exe"
    executable.write_bytes(patched)  # what's actually on disk at runtime

    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    key_id = hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()[:16]
    manifest = {
        "status": "COHERENT",
        "phase": "FINAL",
        "signature_required": True,
        "signing_key_id": key_id,
        "tauri_executable": {
            "name": executable.name,
            "size": len(patched),
            # what release_gate.py recorded, hashed pre-patch:
            "sha256": hashlib.sha256(unpatched).hexdigest(),
        },
        "protected_roots": [],
        "protected_assets": [],
    }
    payload = canonical_manifest_bytes(manifest)
    manifest_path = root / "resource-manifest.json"
    signature_path = root / "resource-manifest.sig"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    signature_path.write_text(
        json.dumps(
            {
                "status": "SIGNED",
                "algorithm": "Ed25519",
                "key_id": key_id,
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "signature_base64": base64.b64encode(private.sign(payload)).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    result = verify_release_assets(
        manifest_path,
        signature_path,
        root,
        public_key_hex=public_hex,
        expected_key_id=key_id,
        tauri_executable_path=executable,
    )
    # Empty protected_roots/protected_assets means this minimal fixture still
    # fails the (unrelated) protected-scope requirements -- what matters here
    # is that the bundle-type patch alone never produces a hash mismatch.
    assert "Tauri executable hash mismatch" not in result["reasons"]
    assert "Tauri executable size mismatch" not in result["reasons"]
    assert "Tauri executable identity mismatch" not in result["reasons"]


def test_deletion_is_detected(tmp_path: Path) -> None:
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    (tmp_path / "models" / "model.bin").unlink()
    result = verify_release_assets(
        manifest, signature, tmp_path, public_key_hex=public, expected_key_id=key_id
    )
    assert result["missing"] == ["models/model.bin"]


def test_addition_is_detected(tmp_path: Path) -> None:
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    (tmp_path / "models" / "injected.bin").write_bytes(b"injected")
    result = verify_release_assets(
        manifest, signature, tmp_path, public_key_hex=public, expected_key_id=key_id
    )
    assert result["additions"] == ["models/injected.bin"]


def test_extra_protected_executable_is_detected(tmp_path: Path) -> None:
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    (tmp_path / "compute" / "dist" / "injected.exe").write_bytes(b"injected")
    result = verify_release_assets(
        manifest, signature, tmp_path, public_key_hex=public, expected_key_id=key_id
    )
    assert "compute/dist/injected.exe" in result["additions"]


def test_missing_or_tampered_manifest_never_self_blesses(tmp_path: Path) -> None:
    missing = verify_release_assets(tmp_path / "missing", tmp_path / "missing.sig", tmp_path)
    assert missing["status"] == "DEGRADED"
    manifest, signature, public, key_id = _write_signed_contract(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["protected_assets"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    tampered = verify_release_assets(
        manifest, signature, tmp_path, public_key_hex=public, expected_key_id=key_id
    )
    assert tampered["status"] == "DEGRADED"
    assert any("digest" in reason for reason in tampered["reasons"])


def test_attacker_manifest_and_public_key_pair_cannot_replace_pinned_root(tmp_path: Path) -> None:
    manifest, signature, attacker_public, attacker_key_id = _write_signed_contract(tmp_path)
    (tmp_path / "integrity-public-key.json").write_text(
        json.dumps({"key_id": attacker_key_id, "public_key_hex": attacker_public}),
        encoding="utf-8",
    )
    result = verify_release_assets(manifest, signature, tmp_path)
    assert result["status"] == "DEGRADED"
    assert result["trusted"] is False
    assert any("not trusted" in reason for reason in result["reasons"])


def test_frozen_manifest_discovery_uses_fixed_path_not_ambiguous_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen-exe manifest lookup must resolve the single canonical
    "<resource_root>/release/resource-manifest.json" path directly. It must
    NOT do a resource_root-wide recursive search expecting exactly one
    match: `cargo tauri build`'s own `_up_` resource-staging convention (see
    src-tauri/src/main.rs's resolve_resource_root) is known to leave extra,
    unrelated resource-manifest.json copies under target/{debug,release}/
    near a real resource root -- a search-based lookup finds >1 match there
    and fails the whole integrity check closed even though the real,
    canonical manifest is sitting exactly where it should be.
    """
    import sys

    import security.integrity as integrity_module

    resource_root = tmp_path / "resource_root"
    real_release_dir = resource_root / "release"
    real_release_dir.mkdir(parents=True)
    real_manifest = real_release_dir / "resource-manifest.json"
    real_manifest.write_text("{}", encoding="utf-8")

    # A same-named file elsewhere under the resource root -- exactly the
    # shape of cargo/Tauri's own incidental build-staging duplicates. A
    # search expecting uniqueness breaks here; a fixed-path lookup does not.
    decoy_dir = resource_root / "some" / "other" / "staging" / "release"
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "resource-manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("BHARATTERMINAL_RESOURCE_ROOT", str(resource_root))
    monkeypatch.delenv("BHARATTERMINAL_RESOURCE_MANIFEST", raising=False)

    found = integrity_module._manifest_path()
    assert found == real_manifest


def test_rust_build_gate_uses_compiled_trust_source_not_release_key_file() -> None:
    root = Path(__file__).resolve().parents[2]
    build = (root / "src-tauri" / "build.rs").read_text(encoding="utf-8")
    assert "trusted_release_keys::public_key_hex" in build
    assert 'fs::read(root.join("release/integrity-public-key.json"))' not in build
    package = (root / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    assert "integrity-public-key.json" not in package
    release_gate = (root / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    assert '"contracts" / "trusted-release-key.json"' in release_gate


def test_production_public_key_matches_release_metadata() -> None:
    key_path = Path(__file__).resolve().parents[2] / "contracts" / "trusted-release-key.json"
    metadata = json.loads(key_path.read_text(encoding="utf-8"))
    from security.integrity import TRUSTED_KEY_ID, TRUSTED_PUBLIC_KEY_HEX

    assert metadata["key_id"] == TRUSTED_KEY_ID
    assert metadata["public_key_hex"] == TRUSTED_PUBLIC_KEY_HEX


def test_failed_integrity_gate_precedes_all_startup_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    events: list[str] = []
    monkeypatch.delenv("BHARATTERMINAL_ALLOW_UNTRUSTED_DEV", raising=False)
    monkeypatch.setattr(
        main,
        "integrity_check",
        lambda **_: events.append("integrity") or {"trusted": False, "reasons": ["tampered"]},
    )
    monkeypatch.setattr(
        main,
        "configure_structured_logging",
        lambda *_: events.append("logging"),
    )

    async def start() -> None:
        async with main.lifespan(main.app):
            raise AssertionError("untrusted startup entered application lifespan")

    with pytest.raises(RuntimeError, match="integrity"):
        asyncio.run(start())
    assert events == ["integrity"]
