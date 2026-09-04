from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from release_gate import (  # noqa: E402
    _normalize_tauri_bundle_type_marker,
    canonicalize_source_bytes,
    read_json,
    sha256_source_file,
    sha256_tauri_executable,
    source_fingerprint,
    source_files,
    validate_openapi,
    validate_versions,
)


@pytest.mark.release
def test_current_source_satisfies_desktop_api_contract(isolated_app_data: Path) -> None:
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    validate_openapi(module.app.openapi())


@pytest.mark.release
def test_openapi_gate_rejects_missing_required_operation() -> None:
    required = {"operations": [{"method": "GET", "path": "/required"}]}
    with pytest.raises(RuntimeError, match="missing GET /required"):
        validate_openapi({"paths": {}}, required)


@pytest.mark.release
def test_release_component_versions_are_bound() -> None:
    spec = read_json(ROOT / "release" / "release-spec.json")
    frontend, rust, python_api = validate_versions(spec)
    assert frontend == rust == python_api == spec["release_version"]


@pytest.mark.release
def test_release_spec_schema_version_matches_enforced_migration_target() -> None:
    """Reproduces a real found drift: release-spec.json declared
    database_schema_version=4 while db/migrations.py::TARGET_SCHEMA_VERSION
    (the value actually enforced against every real database, proven live
    at schema 5 with all 5 migrations applied) had moved on to 5 with no
    test catching the two falling out of sync. release_gate.py reads
    database_schema_version directly from this JSON file with no
    cross-check against the real migration target, so a stale value here
    would silently bake a wrong schema version into the signed release
    identity."""
    from db.migrations import TARGET_SCHEMA_VERSION

    spec = read_json(ROOT / "release" / "release-spec.json")
    assert spec["database_schema_version"] == TARGET_SCHEMA_VERSION


@pytest.mark.release
def test_tauri_declares_all_coherence_resources() -> None:
    config = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    resources = set(config["bundle"]["resources"])
    assert {
        "../release/release-identity.json",
        "../release/resource-manifest.json",
        "../release/packaged-openapi.json",
    }.issubset(resources)


@pytest.mark.release
def test_nsis_installer_purges_stale_resources_before_reinstall() -> None:
    """Reproduces a real deployed defect: NSIS additively copies files into
    $INSTDIR on install/upgrade -- it never removes a PREVIOUS install's
    files that the new build no longer produces under the same name. Vite
    emits content-hashed chunk filenames (dist/assets/index-<hash>.js) that
    change on every source edit, so reinstalling over an existing install
    left the old build's now-orphaned chunks sitting next to the new ones
    under $INSTDIR\\_up_\\dist\\assets. The signed manifest's protected_assets
    set reflects only ONE build's file list, so it can never match an
    install directory holding the union of two builds' outputs -- the app's
    own FINAL-phase integrity check correctly refused to start, panicking
    with 'extra protected asset: dist/assets/index-<stale-hash>.js' on
    every real reinstall. Fix: an NSIS_HOOK_PREINSTALL hook purges the
    entire `_up_` resource tree (build-derived only, never user data --
    that lives under %APPDATA%\\BharatTerminal\\data\\, outside $INSTDIR)
    before Tauri's own file-copy macros run."""
    config = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    hook_relative = config["bundle"]["windows"]["nsis"]["installerHooks"]
    hook_path = ROOT / "src-tauri" / hook_relative
    assert hook_path.is_file(), f"declared installerHooks file missing: {hook_path}"
    hook_source = hook_path.read_text(encoding="utf-8")
    assert "NSIS_HOOK_PREINSTALL" in hook_source
    assert "_up_" in hook_source


@pytest.mark.release
def test_nsis_installer_stops_collector_tasks_before_reinstall() -> None:
    """Reproduces a real deployed defect: bharatterminal-compute.exe at
    $INSTDIR root (the Tauri externalBin sidecar copy) is the same
    executable the Task Scheduler-launched collector
    (BharatTerminal_SmartCollector) and its watchdog
    (BharatTerminal_SmartCollector_Watchdog) run from directly, independent
    of whether the Tauri host app is open. Reproduced live: with the
    scheduled collector still running during a real reinstall, Windows could
    not replace the locked exe -- the installer's copy step silently no-op'd
    on that one file (exit code 0, no error) and the collector kept running
    the OLD sidecar binary after an "upgrade" every other artifact and the
    signed manifest correctly reflected as new. Fix: end both tasks in the
    NSIS_HOOK_PREINSTALL hook before the copy step runs; the watchdog's
    already-proven auto-recovery picks the new binary back up on its own."""
    config = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    hook_relative = config["bundle"]["windows"]["nsis"]["installerHooks"]
    hook_path = ROOT / "src-tauri" / hook_relative
    hook_source = hook_path.read_text(encoding="utf-8")
    assert "BharatTerminal_SmartCollector" in hook_source
    assert "BharatTerminal_SmartCollector_Watchdog" in hook_source
    assert "schtasks" in hook_source.lower()
    watchdog_end_index = hook_source.index('/end /tn "BharatTerminal_SmartCollector_Watchdog"')
    collector_end_index = hook_source.index('/end /tn "BharatTerminal_SmartCollector"')
    purge_index = hook_source.index('RMDir /r "$INSTDIR\\_up_"')
    assert watchdog_end_index < collector_end_index, (
        "the watchdog must be ended BEFORE the main collector task -- reproduced live: "
        "ending them in the other order left a race where the watchdog's own periodic "
        "check can see the main task as unexpectedly stopped and revive it (with the "
        "still-old exe) before the copy step replaces the file"
    )
    assert collector_end_index < purge_index, "tasks must be ended before the file copy/purge step runs"
    assert "RMDir" in hook_source


@pytest.mark.release
def test_nsis_installer_disables_collector_tasks_for_full_install_duration() -> None:
    """Reproduces a real deployed defect that survived the /end-ordering fix
    above: `_up_`'s resource copy (in particular the terminal.db.gz seed
    blob, hundreds of MB) takes minutes -- long enough for the watchdog's own
    normal 5-minute schedule to elapse and refire on its own, re-launching
    and re-locking $INSTDIR\\bharatterminal-compute.exe between the hook's
    one-time /end calls and the externalBin `File` instruction that runs
    near the end of Section Install. Reproduced live across three
    consecutive real installs: every other artifact updated correctly while
    this exe's mtime never advanced past the very first install of the
    session. A one-time /end only stops an already-running instance; it does
    nothing to stop a LATER trigger from firing again before the section
    finishes. Fix: /disable both tasks in NSIS_HOOK_PREINSTALL (closing the
    window for the section's entire duration, not just at t=0), and
    /enable them again in NSIS_HOOK_POSTINSTALL once the new binary is
    safely in place -- leaving them disabled after a successful install
    would silently stop all future data collection with no visible error."""
    config = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    hook_relative = config["bundle"]["windows"]["nsis"]["installerHooks"]
    hook_path = ROOT / "src-tauri" / hook_relative
    hook_source = hook_path.read_text(encoding="utf-8")
    assert "NSIS_HOOK_POSTINSTALL" in hook_source
    preinstall_start = hook_source.index("!macro NSIS_HOOK_PREINSTALL")
    postinstall_start = hook_source.index("!macro NSIS_HOOK_POSTINSTALL")
    preinstall_body = hook_source[preinstall_start:postinstall_start]
    postinstall_body = hook_source[postinstall_start:]
    disable_watchdog_index = preinstall_body.index(
        '/change /tn "BharatTerminal_SmartCollector_Watchdog" /disable'
    )
    disable_collector_index = preinstall_body.index('/change /tn "BharatTerminal_SmartCollector" /disable')
    end_watchdog_index = preinstall_body.index('/end /tn "BharatTerminal_SmartCollector_Watchdog"')
    assert disable_watchdog_index < end_watchdog_index, (
        "both tasks must be disabled BEFORE the /end calls -- otherwise a trigger landing "
        "between /end and /disable could still relaunch the collector"
    )
    assert disable_collector_index < end_watchdog_index
    assert '/change /tn "BharatTerminal_SmartCollector" /enable' in postinstall_body
    assert '/change /tn "BharatTerminal_SmartCollector_Watchdog" /enable' in postinstall_body


@pytest.mark.release
def test_protected_assets_exclude_externalbin_only_sidecar_copy_at_final_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces two real, mirror-image deployed defects around
    `compute/dist/`'s two sidecar copies (compute/build_compute.py writes
    both -- the plain-named one and a target-triple-suffixed copy that
    exists solely for Tauri's `externalBin` convention):

    1. FINAL's protected_root is the real bundled resource directory,
       which tauri.conf.json's `resources` array populates with only the
       plain-named copy -- the triple-suffixed one installs to the app's
       top-level directory instead (Tauri's externalBin convention), never
       under the resource root. Including it in FINAL's protected assets
       made the runtime integrity check look for it where it was never
       placed: a real installed build panicked with "protected asset
       missing: compute/dist/bharatterminal-compute-x86_64-pc-windows-msvc.exe"
       on first launch.

    2. PROBE and PREBUNDLE instead verify against the raw SOURCE tree,
       where `compute/dist/` is build_compute.py's own output directory and
       genuinely holds BOTH copies side by side (Tauri hasn't bundled/
       staged externalBin yet). Excluding the triple-suffixed copy there
       too flips the failure mode: the file is present on disk under a
       scanned protected_root but absent from protected_assets, so the
       runtime check flags it as an unexpected "addition" and fails
       release-openapi-probe on every real build.

    So the exclusion must be phase-specific: present (and verified) at
    PROBE/PREBUNDLE, where it genuinely exists in the scanned root; absent
    only at FINAL, where it genuinely doesn't.

    Sandboxed against a synthetic tree, not the real repo: build_release.ps1
    runs pytest BEFORE compute/build_compute.py, so compute/dist/ never
    genuinely holds either sidecar copy at the moment this test executes on
    a truly pristine checkout -- a version of this test that scanned the
    real ROOT only ever passed because a stale compute/dist/ from an
    earlier manual build happened to be lying around, silently depending on
    an artifact this exact pipeline invocation had not produced yet.
    """
    import release_gate

    monkeypatch.setattr(release_gate, "ROOT", tmp_path)
    monkeypatch.setattr(release_gate, "IDENTITY_PATH", tmp_path / "release" / "release-identity.json")
    monkeypatch.setattr(
        release_gate, "PACKAGED_OPENAPI_PATH", tmp_path / "release" / "packaged-openapi.json"
    )
    (tmp_path / "bundle-data").mkdir(parents=True)
    (tmp_path / "bundle-data" / "terminal.db.gz").write_bytes(b"synthetic")
    (tmp_path / "compute" / "models").mkdir(parents=True)
    (tmp_path / "compute" / "dist").mkdir(parents=True)
    msvc_name = "bharatterminal-compute-x86_64-pc-windows-msvc.exe"
    (tmp_path / "compute" / "dist" / msvc_name).write_bytes(b"synthetic")
    (tmp_path / "compute" / "dist" / "bharatterminal-compute.exe").write_bytes(b"synthetic")
    (tmp_path / "dist").mkdir(parents=True)
    (tmp_path / "dist" / "index.html").write_bytes(b"synthetic")

    from release_gate import protected_asset_paths

    for phase in ("PROBE", "PREBUNDLE"):
        names = {path.name for path in protected_asset_paths(phase=phase)}
        assert msvc_name in names
        assert "bharatterminal-compute.exe" in names
    names = {path.name for path in protected_asset_paths(phase="FINAL")}
    assert msvc_name not in names
    assert "bharatterminal-compute.exe" in names


def test_tauri_bundle_type_marker_normalizes_regardless_of_patch_state(tmp_path: Path) -> None:
    """Reproduces a real deployed defect: `cargo tauri bundle` patches
    indian-terminal.exe in place (a `__TAURI_BUNDLE_TYPE_VAR_UNK` link-time
    placeholder becomes `..._NSS`), so whether release_gate.py hashes the
    exe before or after that patch depends on fragile build-step timing.
    On a real NSIS install this surfaced as the installed app's own
    integrity self-check panicking with "Tauri executable modified" on
    first launch -- the manifest recorded whichever variant happened to be
    on disk at generate-manifest time, not the one actually shipped."""
    unpatched = b"prefix __TAURI_BUNDLE_TYPE_VAR_UNK suffix"
    patched = b"prefix __TAURI_BUNDLE_TYPE_VAR_NSS suffix"
    assert _normalize_tauri_bundle_type_marker(unpatched) == _normalize_tauri_bundle_type_marker(
        patched
    )
    assert b"__TAURI_BUNDLE_TYPE_VAR_UNK" in _normalize_tauri_bundle_type_marker(patched)

    # Two files differing ONLY in that one known marker must hash identically...
    unpatched_file = tmp_path / "unpatched.exe"
    patched_file = tmp_path / "patched.exe"
    unpatched_file.write_bytes(unpatched)
    patched_file.write_bytes(patched)
    assert sha256_tauri_executable(unpatched_file) == sha256_tauri_executable(patched_file)

    # ...but any OTHER byte change, including this exact field holding some
    # third, unrecognized value, must still change the digest.
    tampered_file = tmp_path / "tampered.exe"
    tampered_file.write_bytes(b"prefix __TAURI_BUNDLE_TYPE_VAR_XXX suffix")
    assert sha256_tauri_executable(tampered_file) != sha256_tauri_executable(unpatched_file)

    other_byte_changed = tmp_path / "other-change.exe"
    other_byte_changed.write_bytes(unpatched.replace(b"prefix", b"PREFIX"))
    assert sha256_tauri_executable(other_byte_changed) != sha256_tauri_executable(unpatched_file)


def test_final_signed_manifest_covers_tauri_executable_before_bundle() -> None:
    gate = (ROOT / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    rust_integrity = (ROOT / "src-tauri" / "src" / "release_integrity.rs").read_text(
        encoding="utf-8"
    )
    build = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert 'manifest["tauri_executable"]' in gate
    assert 'manifest["phase"] != "FINAL"' in rust_integrity
    assert "verify_tauri_executable(&manifest)?" in rust_integrity
    assert build.index("tauri build --no-bundle") < build.index(
        "release_gate.py generate-manifest"
    )
    assert build.index("release_gate.py generate-manifest") < build.index(
        "tauri bundle --bundles nsis"
    )


def test_release_identity_generated_before_anything_imports_main() -> None:
    """Reproduced live: main.py binds RELEASE_IDENTITY at module import time
    (main.py: `RELEASE_IDENTITY = load_release_identity()`), which raises
    RuntimeError("release identity not found") when no
    release/release-identity.json exists yet -- true on a genuine clean
    checkout. build_release.ps1 used to run `pytest` (which imports main.py
    via test_integrity.py etc.) and `source-openapi` (which imports main.py
    to introspect the FastAPI app) BEFORE `generate-identity`, so a pristine
    checkout's release pipeline crashed on its very first step."""
    build = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    generate_identity_pos = build.index("release_gate.py generate-identity")
    assert build.index("-m pytest") > generate_identity_pos
    assert build.index("release_gate.py source-openapi") > generate_identity_pos
    # generate-identity itself hashes bundle-data/terminal.db.gz, which only
    # Compress-TerminalDb produces -- must run strictly before it. rindex,
    # not index: the first occurrence of this token is the `function
    # Compress-TerminalDb {` definition itself, which always sits near the
    # top of the file regardless of call order -- only the call site (the
    # last occurrence) matters here.
    assert build.rindex("Compress-TerminalDb") < generate_identity_pos


def test_release_checkpoints_duckdb_wal_before_compressing_snapshot() -> None:
    """A raw terminal.db copy omits committed rows still present only in its
    WAL. The canonical build must checkpoint and reject a non-empty WAL before
    it compresses the bootstrap database into the installer."""
    build = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    checkpoint_pos = build.rindex("Checkpoint-TerminalDb -Python")
    compress_pos = build.rindex("Compress-TerminalDb")
    assert checkpoint_pos < compress_pos
    assert 'connection.execute("CHECKPOINT")' in build
    assert "WAL remained non-empty after CHECKPOINT" in build


@pytest.mark.release
def test_release_source_fingerprint_excludes_volatile_build_state() -> None:
    volatile_parts = {
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        ".venv-clean",
        ".venv-release",
        "__pycache__",
        "build",
        "catboost_info",
        "dist",
        "node_modules",
        "target",
    }
    for path in source_files():
        relative_parts = path.relative_to(ROOT).parts
        assert not volatile_parts.intersection(relative_parts)
        assert not any(
            part.startswith((".test-tmp", ".venv", ".uv-cache"))
            for part in relative_parts
        )


@pytest.mark.release
def test_rust_source_fingerprint_excludes_any_dot_or_underscore_directory() -> None:
    """`src-tauri/build.rs` reimplements source_files()'s exclusion rules in
    Rust to independently re-verify, at compile time, that source hasn't
    drifted since release_gate.py computed and signed source_sha256 into the
    PREBUNDLE manifest. That only works if both implementations walk the
    EXACT same file set. Python excludes any directory whose name starts
    with "." or "_" as a blanket rule (this function's own `os.walk` filter,
    above); a prior version of build.rs only excluded three specific
    dot-prefixes (".test-tmp", ".venv", ".uv-cache"), so any other
    dot/underscore-prefixed directory under a scanned base -- e.g. a real
    leftover scratch directory like `compute/.route-audit` -- silently
    walked into on the Rust side but skipped on the Python side, making the
    two fingerprints permanently, deterministically unequal regardless of
    whether the source ever actually changed. Reproduced live: `cargo tauri
    build`'s PREBUNDLE coherence check failed on every single build with
    "release source changed after artifacts were built" even with zero real
    drift, purely because of this cross-implementation exclusion mismatch."""
    build_rs = (ROOT / "src-tauri" / "build.rs").read_text(encoding="utf-8")
    assert "name.starts_with('.')" in build_rs
    assert "name.starts_with('_')" in build_rs
    # The set of explicitly-named exclusions must also stay in sync between
    # the two implementations (this function's own hardcoded set, above).
    excluded_from_source_files = {
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        ".venv-clean",
        ".venv-release",
        "__pycache__",
        "build",
        "catboost_info",
        "dist",
        "logs",
        "node_modules",
        "saved_models",
        "target",
    }
    for name in excluded_from_source_files:
        assert f'"{name}"' in build_rs, f"build.rs source_fingerprint() is missing exclusion: {name}"


@pytest.mark.release
def test_release_source_fingerprint_excludes_arbitrary_dot_or_underscore_scratch_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discriminating case for the same defect as the test above, but
    exercised directly against Python's own source_files() with a real
    on-disk scratch directory shaped exactly like the one that triggered
    it (`compute/.route-audit`) -- proves the Python side already behaves
    correctly (it always did; the bug was Rust's narrower mirror of it)."""
    import release_gate

    fake_root = tmp_path / "repo"
    scratch_dir = fake_root / "compute" / ".route-audit" / "data"
    scratch_dir.mkdir(parents=True)
    (scratch_dir / "leftover.json").write_text("{}", encoding="utf-8")
    real_file = fake_root / "compute" / "main.py"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.write_text("# real source\n", encoding="utf-8")
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
        "build_release.ps1",
        "release/release-spec.json",
        "release/required-api.json",
    ):
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

    monkeypatch.setattr(release_gate, "ROOT", fake_root)
    files = set(release_gate.source_files())
    assert real_file in files
    assert not any(".route-audit" in path.parts for path in files)


@pytest.mark.release
def test_release_source_fingerprint_ignores_catboost_training_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces a live defect: CatBoost's default train_dir writes
    compute/catboost_info/ into the process cwd as a side effect of any real
    training run, including test_determinism_source_freeze.py's in-suite
    CatBoost training. That directory does not start with "." or "_", so
    (unlike .route-audit above) it was NOT covered by source_files()'s
    blanket dot/underscore-prefix skip -- it needed its own explicit
    exclusion, exactly like build/dist/logs/__pycache__ below it.

    Reproduced live: `build_release.ps1` -> generate-identity (runs BEFORE
    pytest, catboost_info/ does not exist yet) -> pytest, in which
    test_determinism_source_freeze.py trains a real CatBoost model partway
    through the suite, leaving compute/catboost_info/ behind -> by the time
    test_release_identity_binds_canonical_source_fingerprint ran later in
    the SAME pytest invocation, source_fingerprint() picked up the new
    directory and produced a different hash than the one already baked into
    release-identity.json -- a real, deterministic release-pipeline defect,
    not test flakiness or drift."""
    import release_gate

    fake_root = tmp_path / "repo"
    catboost_dir = fake_root / "compute" / "catboost_info" / "learn"
    catboost_dir.mkdir(parents=True)
    (catboost_dir / "events.out.tfevents").write_bytes(b"\x00")
    (fake_root / "compute" / "catboost_info" / "learn_error.tsv").write_text("", encoding="utf-8")
    real_file = fake_root / "compute" / "main.py"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.write_text("# real source\n", encoding="utf-8")
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
    ):
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

    monkeypatch.setattr(release_gate, "ROOT", fake_root)
    before = set(release_gate.source_files())
    assert real_file in before
    assert not any("catboost_info" in path.parts for path in before)

    # The determinism property the original bug violated: adding the
    # catboost_info/ directory after computing a first fingerprint must not
    # change source_fingerprint()'s result.
    fingerprint_before = release_gate.source_fingerprint()
    (catboost_dir / "another_run_output.tsv").write_text("late", encoding="utf-8")
    fingerprint_after = release_gate.source_fingerprint()
    assert fingerprint_before == fingerprint_after


@pytest.mark.release
def test_release_source_fingerprint_includes_frontend_public_assets() -> None:
    files = set(source_files())
    assert ROOT / "public" / "plotly.min.js" in files
    assert ROOT / "public" / "pdf.min.mjs" in files
    assert ROOT / "public" / "pdf.worker.min.mjs" in files


@pytest.mark.release
def test_release_identity_binds_canonical_source_fingerprint() -> None:
    from release_gate import IDENTITY_PATH, source_fingerprint

    identity = read_json(IDENTITY_PATH)
    assert identity["source_sha256"] == source_fingerprint()


@pytest.mark.release
def test_sidecar_build_bundles_shared_exchange_contract() -> None:
    build_compute = importlib.import_module("build_compute")
    source = inspect.getsource(build_compute.main)
    assert build_compute.CONTRACTS == ROOT / "contracts"
    assert 'f"{CONTRACTS};contracts"' in source


@pytest.mark.release
def test_sidecar_build_bundles_every_dynamically_imported_skops_submodule() -> None:
    """Reproduces a real deployed defect: skops/io/_persist.py loads its
    backend submodules via `importlib.import_module()` with a
    string-constructed module name (`modules = ["._general", "._numpy",
    "._scipy", "._sklearn", "._quantile_forest"]`), a pattern PyInstaller's
    static import analysis cannot trace -- listing only the parent "skops"
    package as a hidden import bundled skops.io itself but silently
    dropped every one of these. Reproduced live: a real installed build
    logged "bif: ML provider failed: No module named 'skops.io._sklearn'"
    on every startup, meaning the BIF ML-backed provider silently never
    worked in the shipped product even though it worked in dev mode."""
    import build_compute
    import skops.io._persist as skops_persist

    dynamically_imported = {
        f"skops.io{name}" for name in skops_persist.modules
    }
    missing = dynamically_imported.difference(build_compute.HIDDEN_IMPORTS)
    assert not missing, f"build_compute.py HIDDEN_IMPORTS is missing: {sorted(missing)}"


@pytest.mark.release
def test_sidecar_build_copies_pandas_ta_openbb_package_metadata() -> None:
    """Reproduces a real deployed defect: pandas_ta/maps.py reads its own
    package metadata at import time via
    importlib.metadata.distribution("pandas-ta-openbb"). PyInstaller's
    onefile bundling only bundles importable code, not .dist-info metadata
    directories, so every import of pandas_ta raised
    importlib.metadata.PackageNotFoundError in the frozen build. Reproduced
    live: a real installed release returned "500 Internal Server Error" on
    every /indicators call (EMA/RSI/MACD/BB), for every symbol,
    unconditionally -- the entire technical-indicators chart never worked
    in the shipped product even though it worked in dev mode."""
    build_compute = importlib.import_module("build_compute")
    source = inspect.getsource(build_compute.main)
    assert '"--copy-metadata", "pandas-ta-openbb"' in source


# --- release source fingerprint: checkout-EOL independence -----------------
#
# Reproduces a real forensic finding: core.autocrlf=true on Windows left six
# already-tracked compute/ files (duckdb_client.py, write_service.py, and
# four compute/tests/*.py files) holding CRLF bytes on disk even though
# .gitattributes declares `* text=auto eol=lf` and the committed git blobs
# are LF-normalized -- adding the attribute does not retroactively rewrite
# bytes already sitting in a working tree. source_fingerprint() used to hash
# those raw worktree bytes directly, so the signed release source fingerprint
# diverged from a clean `git archive HEAD` hash of the exact same committed
# content. The fix canonicalizes CRLF -> LF at hash time for every file in
# the fingerprint's scope, so the same committed content produces the same
# fingerprint regardless of which EOL bytes a given checkout carries.

_REQUIRED_EXPLICIT_SOURCE_FILES = (
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


def _write_minimal_fake_repo(root: Path, *, compute_main_content: bytes) -> None:
    for relative in _REQUIRED_EXPLICIT_SOURCE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    compute_main = root / "compute" / "main.py"
    compute_main.parent.mkdir(parents=True, exist_ok=True)
    compute_main.write_bytes(compute_main_content)


@pytest.mark.release
def test_canonicalize_source_bytes_normalizes_crlf_only() -> None:
    # CRLF collapses to LF ...
    assert canonicalize_source_bytes(b"line one\r\nline two\r\n") == b"line one\nline two\n"
    # ... a bare LF is left untouched ...
    assert canonicalize_source_bytes(b"line one\nline two\n") == b"line one\nline two\n"
    # ... and a lone CR (not followed by LF) is NOT stripped -- this is a
    # narrow CRLF->LF transform, not a blanket "delete all \r" pass.
    assert canonicalize_source_bytes(b"a\rb") == b"a\rb"


@pytest.mark.release
def test_source_file_hash_identical_for_crlf_and_lf_checkout(tmp_path: Path) -> None:
    lf_file = tmp_path / "lf.py"
    crlf_file = tmp_path / "crlf.py"
    content = b"def handler():\n    return 1\n\nclass Foo:\n    pass\n"
    lf_file.write_bytes(content)
    crlf_file.write_bytes(content.replace(b"\n", b"\r\n"))
    assert sha256_source_file(lf_file) == sha256_source_file(crlf_file)


@pytest.mark.release
def test_source_fingerprint_stable_across_eol_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import release_gate

    content = b"import os\n\n\ndef run():\n    return os.getcwd()\n"

    lf_root = tmp_path / "lf-checkout"
    crlf_root = tmp_path / "crlf-checkout"
    _write_minimal_fake_repo(lf_root, compute_main_content=content)
    _write_minimal_fake_repo(crlf_root, compute_main_content=content.replace(b"\n", b"\r\n"))

    monkeypatch.setattr(release_gate, "ROOT", lf_root)
    lf_fingerprint = release_gate.source_fingerprint()
    monkeypatch.setattr(release_gate, "ROOT", crlf_root)
    crlf_fingerprint = release_gate.source_fingerprint()

    assert lf_fingerprint == crlf_fingerprint


@pytest.mark.release
def test_source_fingerprint_reacts_to_real_content_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import release_gate

    root = tmp_path / "repo"
    _write_minimal_fake_repo(root, compute_main_content=b"def run():\n    return 1\n")
    monkeypatch.setattr(release_gate, "ROOT", root)
    baseline = release_gate.source_fingerprint()

    (root / "compute" / "main.py").write_bytes(b"def run():\n    return 2\n")
    changed = release_gate.source_fingerprint()

    assert baseline != changed


@pytest.mark.release
def test_source_fingerprint_changes_when_scoped_file_is_added_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import release_gate

    root = tmp_path / "repo"
    _write_minimal_fake_repo(root, compute_main_content=b"def run():\n    return 1\n")
    monkeypatch.setattr(release_gate, "ROOT", root)
    baseline = release_gate.source_fingerprint()

    (root / "compute" / "extra_module.py").write_bytes(b"VALUE = 1\n")
    with_extra_file = release_gate.source_fingerprint()
    assert with_extra_file != baseline

    (root / "compute" / "extra_module.py").unlink()
    back_to_baseline = release_gate.source_fingerprint()
    assert back_to_baseline == baseline


@pytest.mark.release
def test_source_fingerprint_is_deterministic_across_repeated_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import release_gate

    root = tmp_path / "repo"
    _write_minimal_fake_repo(root, compute_main_content=b"def run():\n    return 1\n")
    (root / "compute" / "aaa.py").write_bytes(b"A = 1\n")
    (root / "compute" / "zzz.py").write_bytes(b"Z = 1\n")
    monkeypatch.setattr(release_gate, "ROOT", root)

    fingerprints = {release_gate.source_fingerprint() for _ in range(5)}
    assert len(fingerprints) == 1


@pytest.mark.release
def test_source_fingerprint_fails_closed_when_required_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import release_gate

    root = tmp_path / "repo"
    _write_minimal_fake_repo(root, compute_main_content=b"def run():\n    return 1\n")
    (root / "package.json").unlink()
    monkeypatch.setattr(release_gate, "ROOT", root)

    with pytest.raises(FileNotFoundError):
        release_gate.source_fingerprint()


@pytest.mark.release
def test_untracked_root_scratch_files_are_outside_source_fingerprint_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local acceptance-evidence scratch files (.playwright-cli/, output/,
    f0_snapshot.json) live at repo root, outside every base directory
    source_files() walks and outside its explicit file list, so they must
    never affect the signed release source fingerprint."""
    import release_gate

    root = tmp_path / "repo"
    _write_minimal_fake_repo(root, compute_main_content=b"def run():\n    return 1\n")
    monkeypatch.setattr(release_gate, "ROOT", root)
    baseline = release_gate.source_fingerprint()

    (root / "f0_snapshot.json").write_bytes(b'{"scratch": true}')
    (root / "output").mkdir()
    (root / "output" / "evidence.json").write_bytes(b"{}")
    (root / ".playwright-cli").mkdir()
    (root / ".playwright-cli" / "trace.zip").write_bytes(b"binary")

    assert release_gate.source_fingerprint() == baseline


@pytest.mark.release
def test_six_previously_observed_crlf_files_hash_stable_regardless_of_checkout_eol() -> None:
    """Directly verifies the six files the forensic pass observed holding
    CRLF worktree bytes (compute/db/duckdb_client.py,
    compute/db/write_service.py, compute/tests/test_instrument_contract.py,
    compute/tests/test_signals_backfill.py, compute/tests/test_spot_provenance.py,
    compute/tests/test_write_service.py): whatever EOL bytes these files
    physically carry on THIS checkout, canonicalizing before hashing must
    make their fingerprint contribution identical to a synthetic pure-LF
    copy of the same content -- no source rewrite required to satisfy this."""
    observed_files = (
        ROOT / "compute" / "db" / "duckdb_client.py",
        ROOT / "compute" / "db" / "write_service.py",
        ROOT / "compute" / "tests" / "test_instrument_contract.py",
        ROOT / "compute" / "tests" / "test_signals_backfill.py",
        ROOT / "compute" / "tests" / "test_spot_provenance.py",
        ROOT / "compute" / "tests" / "test_write_service.py",
    )
    for path in observed_files:
        assert path.is_file(), f"expected fingerprint-scope file missing: {path}"
        raw = path.read_bytes()
        synthetic_lf = raw.replace(b"\r\n", b"\n")
        assert canonicalize_source_bytes(raw) == canonicalize_source_bytes(synthetic_lf)
        assert hashlib.sha256(canonicalize_source_bytes(raw)).hexdigest() == hashlib.sha256(
            synthetic_lf
        ).hexdigest()


@pytest.mark.release
def test_release_identity_fingerprint_matches_git_archive_of_head() -> None:
    """The ultimate reproduction of the original defect: a clean `git archive
    HEAD` materialization (pure LF, no local checkout artifacts) must hash to
    the exact same canonical source fingerprint as this working checkout,
    proving the fingerprint no longer depends on incidental Windows CRLF
    checkout bytes. Skips gracefully if git/archive tooling is unavailable in
    the sandbox running this test."""
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        pytest.skip("not running inside a git checkout")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "head.tar"
        result = subprocess.run(
            ["git", "archive", "-o", str(archive_path), "HEAD"],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            pytest.skip(f"git archive unavailable: {result.stderr.decode(errors='replace')}")

        extract_root = tmp_path / "extracted"
        extract_root.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_root, filter="data")  # trusted local git archive of our own HEAD

        import release_gate

        original_root = release_gate.ROOT
        try:
            release_gate.ROOT = extract_root
            archive_fingerprint = release_gate.source_fingerprint()
        finally:
            release_gate.ROOT = original_root

        assert archive_fingerprint == source_fingerprint()


@pytest.mark.release
def test_rust_build_script_canonicalizes_source_bytes_before_hashing() -> None:
    """Cross-language contract check, matching this file's existing pattern
    of re-verifying build.rs's independent reimplementation of
    source_fingerprint() stays in lockstep with release_gate.py's: both must
    canonicalize CRLF -> LF before hashing each source file, using the exact
    same transform, or the two independently-computed fingerprints (Python
    signs it, Rust re-verifies it at PREBUNDLE compile time) would
    permanently disagree on any checkout carrying CRLF bytes."""
    build_rs = (ROOT / "src-tauri" / "build.rs").read_text(encoding="utf-8")
    assert "fn canonicalize_source_bytes(" in build_rs
    canonicalize_pos = build_rs.index("fn canonicalize_source_bytes(")
    fingerprint_pos = build_rs.index("fn source_fingerprint(")
    assert canonicalize_pos < fingerprint_pos, (
        "canonicalize_source_bytes must be defined before source_fingerprint uses it"
    )
    assert "canonicalize_source_bytes(&raw)" in build_rs
    digest_call = build_rs.index("digest.update(Sha256::digest(canonicalize_source_bytes(&raw)))")
    assert digest_call > fingerprint_pos
