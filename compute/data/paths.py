from __future__ import annotations

import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    override = os.environ.get("BHARATTERMINAL_APP_DATA")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        appdata = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        return (appdata / "BharatTerminal").resolve()
    return Path(__file__).resolve().parents[3]


BASE_DIR = _base_dir()

DATA_DIR = Path(os.environ.get("BHARATTERMINAL_DATA_ROOT", BASE_DIR / "data")).expanduser().resolve()
DUCKDB_PATH = Path(
    os.environ.get("BHARATTERMINAL_DB_PATH", DATA_DIR / "terminal.db")
).expanduser().resolve()
SQLITE_PATH = DATA_DIR / "config.db"
ERROR_LOG_PATH = DATA_DIR / "data_errors.db"
MODELS_DIR = BASE_DIR / "models"
BHAVCOPY_DIR = DATA_DIR / "bhavcopy"
EQUITY_BHAV_DIR = BHAVCOPY_DIR / "equity"
FO_BHAV_DIR = BHAVCOPY_DIR / "fo"
INTRADAY_DIR = DATA_DIR / "intraday"
INSTRUMENTS_CACHE = DATA_DIR / "instruments_master.parquet"
QUARANTINE_DIR = DATA_DIR / "quarantine"


def ensure_dirs() -> None:
    """Create all runtime data directories."""
    for directory in [
        DATA_DIR,
        MODELS_DIR,
        BHAVCOPY_DIR,
        EQUITY_BHAV_DIR,
        FO_BHAV_DIR,
        INTRADAY_DIR,
        QUARANTINE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
