from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.utils.hooks import collect_data_files

from operations.storage import require_free_space

ROOT = Path(__file__).resolve().parent
RELEASE_IDENTITY = ROOT.parent / "release" / "release-identity.json"
CONTRACTS = ROOT.parent / "contracts"
MAIN = ROOT / "main.py"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = BUILD / "bharatterminal-compute.spec"

# Hidden imports collect importable Python modules only. XGBoost's Windows
# wheel loads this native runtime with ctypes from xgboost/lib at import time,
# so PyInstaller cannot discover it by following Python imports. Keep the
# package list deliberately narrow: every other compiled production package
# currently has a working upstream PyInstaller hook and is verified by the
# frozen-runtime release gate.
NATIVE_BINARY_PACKAGES = ("xgboost",)
EXPECTED_FROZEN_NATIVE_BINARIES = frozenset({"xgboost/lib/xgboost.dll"})
EXPECTED_FROZEN_PACKAGE_DATA = frozenset({"xgboost/VERSION"})

HIDDEN_IMPORTS = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "duckdb",
    "redis",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "skops",
    # skops/io/_persist.py imports these via importlib.import_module() with
    # string-constructed module names -- PyInstaller's static import
    # analysis cannot trace that, so only the parent "skops" package got
    # bundled and every one of these silently failed to load at runtime.
    # Reproduced live: a real installed build logged "bif: ML provider
    # failed: No module named 'skops.io._sklearn'" on every startup.
    "skops.io._audit",
    "skops.io._general",
    "skops.io._numpy",
    "skops.io._protocol",
    "skops.io._quantile_forest",
    "skops.io._scipy",
    "skops.io._sklearn",
    "skops.io._trusted_types",
    "skops.io._utils",
    "skops.io.old._general_v0",
    "skops.io.old._numpy_v0",
    "skops.io.old._numpy_v1",
    "xgboost",
    "lightgbm",
    "catboost",
    "py_vollib",
    "pandas_ta",
    "keyring",
    "keyring.backends.Windows",
    "cryptography",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "SmartApi",
    "pyotp",
    "groq",
]


def main() -> None:
    require_free_space(DIST, 4 * 1024 * 1024 * 1024, "compute sidecar build", headroom_bytes=1024 * 1024 * 1024)
    if not RELEASE_IDENTITY.is_file():
        raise RuntimeError("release identity missing; run release gate generate-identity first")
    identity = __import__("json").loads(RELEASE_IDENTITY.read_text(encoding="utf-8"))
    if identity.get("build_status") != "COHERENT":
        raise RuntimeError("release identity is not coherent")
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--onefile",
        # Windows GUI subsystem, not console. Task Scheduler runs our tasks
        # under an InteractiveToken logon (required for the collector to
        # reach the user's own Credential Manager/DPAPI store), and Windows
        # does not suppress a console-subsystem process's window for that
        # logon type the way it does for a service account -- reproduced
        # live as a console window repeatedly appearing while the installed
        # collector/watchdog ran. A GUI-subsystem executable never gets an
        # OS-allocated console under any launch path, while explicitly
        # redirected stdio (Tauri's Stdio::piped(), release_gate.py's
        # subprocess capture) keeps working unchanged -- Windows only
        # allocates a console when no stdio handle was already provided.
        "--windowed",
        "--name",
        "bharatterminal-compute",
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD),
        "--specpath",
        str(BUILD),
    ]
    for module in HIDDEN_IMPORTS:
        command.extend(["--hidden-import", module])

    # pandas_ta/maps.py reads its own package metadata at import time via
    # importlib.metadata.distribution("pandas-ta-openbb") to detect which
    # fork is installed. PyInstaller's onefile bundling does not include
    # .dist-info metadata directories by default (only importable code), so
    # this raised importlib.metadata.PackageNotFoundError on every /indicators
    # call in the frozen build -- reproduced live: a real installed release
    # returned "500 Internal Server Error" for every chart's technical
    # indicators (EMA/RSI/MACD/BB), for every symbol, unconditionally.
    command.extend(["--copy-metadata", "pandas-ta-openbb"])
    for package in NATIVE_BINARY_PACKAGES:
        command.extend(["--collect-binaries", package])
    for source, destination in required_xgboost_package_data():
        command.extend(["--add-data", f"{source};{destination}"])

    models_dir = ROOT / "models"
    if models_dir.exists():
        command.extend(["--add-data", f"{models_dir};models"])
    command.extend(["--add-data", f"{CONTRACTS};contracts"])
    command.extend(["--add-data", f"{RELEASE_IDENTITY};."])

    command.append(str(MAIN))
    subprocess.run(command, cwd=ROOT, check=True)
    exe = DIST / "bharatterminal-compute.exe"
    verify_frozen_native_payload(exe)
    tauri_sidecar = DIST / "bharatterminal-compute-x86_64-pc-windows-msvc.exe"
    if exe.exists():
        shutil.copy2(exe, tauri_sidecar)
    print(f"Compute sidecar at: {exe}")
    print(f"Tauri sidecar at: {tauri_sidecar}")


def frozen_archive_entries(executable: Path) -> set[str]:
    """Return normalized onefile archive paths without extracting payloads."""
    if not executable.is_file():
        raise RuntimeError(f"frozen sidecar missing: {executable}")
    return {name.replace("\\", "/") for name in CArchiveReader(str(executable)).toc}


def validate_native_payload_entries(entries: set[str]) -> None:
    missing = sorted(EXPECTED_FROZEN_NATIVE_BINARIES.difference(entries))
    if missing:
        raise RuntimeError(f"frozen native payload missing: {', '.join(missing)}")
    missing_data = sorted(EXPECTED_FROZEN_PACKAGE_DATA.difference(entries))
    if missing_data:
        raise RuntimeError(f"frozen package data missing: {', '.join(missing_data)}")


def required_xgboost_package_data() -> list[tuple[str, str]]:
    files = collect_data_files("xgboost", includes=["VERSION"])
    normalized = {(Path(source).name, destination.replace("\\", "/")) for source, destination in files}
    if normalized != {("VERSION", "xgboost")}:
        raise RuntimeError(f"unexpected XGBoost package data layout: {sorted(normalized)}")
    return files


def verify_frozen_native_payload(executable: Path) -> None:
    validate_native_payload_entries(frozen_archive_entries(executable))


if __name__ == "__main__":
    main()
