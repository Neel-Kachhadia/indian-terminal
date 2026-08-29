from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

from operations.storage import require_free_space

ROOT = Path(__file__).resolve().parent
RELEASE_IDENTITY = ROOT.parent / "release" / "release-identity.json"
CONTRACTS = ROOT.parent / "contracts"
MAIN = ROOT / "main.py"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / "bharatterminal-compute.spec"

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
        "--name",
        "bharatterminal-compute",
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD),
        "--specpath",
        str(ROOT),
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

    models_dir = ROOT / "models"
    if models_dir.exists():
        command.extend(["--add-data", f"{models_dir};models"])
    command.extend(["--add-data", f"{CONTRACTS};contracts"])
    command.extend(["--add-data", f"{RELEASE_IDENTITY};."])

    command.append(str(MAIN))
    subprocess.run(command, cwd=ROOT, check=True)
    exe = DIST / "bharatterminal-compute.exe"
    tauri_sidecar = DIST / "bharatterminal-compute-x86_64-pc-windows-msvc.exe"
    if exe.exists():
        shutil.copy2(exe, tauri_sidecar)
    if SPEC.exists():
        SPEC.unlink()
    print(f"Compute sidecar at: {exe}")
    print(f"Tauri sidecar at: {tauri_sidecar}")


if __name__ == "__main__":
    main()
