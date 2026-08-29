"""
NSE Bhavcopy downloader and parser.

Fallback chain:
1. jugaad-data
2. NSE archive direct URLs

Each download/load is checkpointed in data_errors.db. Unknown formats are
quarantined and logged so backfills continue.
"""

import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from backtest.price_quality import quality_for_ingest
from data.exchange_calendar import is_trading_day_candidate
from data.error_logger import _logger as elog
from data.paths import DUCKDB_PATH, EQUITY_BHAV_DIR, FO_BHAV_DIR, QUARANTINE_DIR, ensure_dirs
from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read
from operations.storage import require_free_space

# Bumping this gives a rerun a clean checkpoint namespace without deleting any
# prior backfill_progress rows -- legacy rows simply live under a different,
# unversioned stage key and can never satisfy is_done() for the current
# pipeline. See error_logger.backfill_progress / _load_stage below.
PIPELINE_VERSION = "p1_provenance"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com",
}

FORMAT_META_KEYS = {"version"}
FO_INSTRUMENT_TYPE_ALIASES = {
    "IDF": "FUTIDX",
    "STF": "FUTSTK",
    "IDO": "OPTIDX",
    "STO": "OPTSTK",
}
FO_INSTRUMENT_TYPES = {
    "FUTIDX",
    "FUTSTK",
    "OPTIDX",
    "OPTSTK",
    *FO_INSTRUMENT_TYPE_ALIASES.keys(),
}
FO_FUTURE_INSTRUMENT_TYPES = {"FUTIDX", "FUTSTK", "IDF", "STF"}
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_CSV_BYTES = 512 * 1024 * 1024
EQUITY_INSTRUMENT_TYPES = {"EQ", "STK"}
INDEX_FO_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
NSE_CONTENT_URL_CUTOFF = date(2024, 1, 1)

EQUITY_FORMATS = [
    {
        "version": "v3_unified",
        "series": ["SctySrs"],
        "instrument": ["FinInstrmTp"],
        "isin": ["ISIN"],
        "symbol": ["TckrSymb"],
        "date": ["TradDt"],
        "open": ["OpnPric"],
        "high": ["HghPric"],
        "low": ["LwPric"],
        "close": ["ClsPric"],
        "volume": ["TtlTradgVol"],
    },
    {
        "version": "v2_post2023",
        "symbol": ["TckrSymb"],
        "date": ["TradDt"],
        "open": ["OpnPric"],
        "high": ["HghPric"],
        "low": ["LwPric"],
        "close": ["ClsePric"],
        "volume": ["TtlTradgVol"],
    },
    {
        "version": "v1_pre2023",
        "symbol": ["SYMBOL"],
        "series": ["SERIES"],
        "date": ["TIMESTAMP", "DATE1"],
        "open": ["OPEN", "OPEN_PRICE"],
        "high": ["HIGH", "HIGH_PRICE"],
        "low": ["LOW", "LOW_PRICE"],
        "close": ["CLOSE", "CLOSE_PRICE"],
        "volume": ["TOTTRDQTY", "TTL_TRD_QNTY"],
    },
]

FO_FORMATS = [
    {
        "version": "v3_unified",
        "instrument": ["FinInstrmTp"],
        "symbol": ["TckrSymb"],
        "expiry": ["XpryDt"],
        "strike": ["StrkPric"],
        "option_type": ["OptnTp"],
        "open": ["OpnPric"],
        "high": ["HghPric"],
        "low": ["LwPric"],
        "close": ["ClsPric"],
        "volume": ["TtlTradgVol"],
        "oi": ["OpnIntrst"],
        "oi_change": ["ChngInOpnIntrst"],
        "date": ["TradDt"],
    },
    {
        "version": "v2_post2023",
        "instrument": ["FinInstrmTp"],
        "symbol": ["TckrSymb"],
        "expiry": ["XpryDt"],
        "strike": ["StrkPric"],
        "option_type": ["OptnTp"],
        "open": ["OpnPric"],
        "high": ["HghPric"],
        "low": ["LwPric"],
        "close": ["ClsePric"],
        "volume": ["TtlTradgVol"],
        "oi": ["OpnIntrst"],
        "oi_change": ["ChngInOpnIntrst"],
        "date": ["TradDt"],
    },
    {
        "version": "v1_pre2023",
        "instrument": ["INSTRUMENT"],
        "symbol": ["SYMBOL"],
        "expiry": ["EXPIRY_DT"],
        "strike": ["STRIKE_PR"],
        "option_type": ["OPTION_TYP"],
        "open": ["OPEN"],
        "high": ["HIGH"],
        "low": ["LOW"],
        "close": ["CLOSE"],
        "volume": ["CONTRACTS"],
        "oi": ["OPEN_INT"],
        "oi_change": ["CHG_IN_OI"],
        "date": ["TIMESTAMP"],
    },
]


class RawSourceValidationError(ValueError):
    """Raised when downloaded content must never be persisted as a raw source file."""


_HTML_SIGNATURES = (b"<!doctype html", b"<html", b"<head", b"<body")


def _validate_csv_payload(content: bytes, *, context: str) -> None:
    """Gate every write to EQUITY_BHAV_DIR/FO_BHAV_DIR.

    The historical archive contains 16 corrupt files -- 15 are saved HTML
    error/challenge pages, 1 is a 0-byte file -- because nothing validated
    downloaded bytes before writing them to disk as ".csv". This is the single
    choke point all raw-file writes now pass through so that can never repeat.
    """
    if len(content) == 0:
        raise RawSourceValidationError(f"{context}: empty payload, refusing to persist")
    head = content[:512].lstrip().lower()
    if any(head.startswith(sig) for sig in _HTML_SIGNATURES) or b"<html" in head[:100]:
        raise RawSourceValidationError(f"{context}: HTML payload (error/challenge page), refusing to persist")
    first_line = content.split(b"\n", 1)[0]
    if b"," not in first_line:
        raise RawSourceValidationError(f"{context}: no delimiter in first line, does not look like CSV")


_LEGACY_STEM_DATE = re.compile(r"^[a-zA-Z]*(\d{1,2})([a-zA-Z]{3})(\d{4})[a-zA-Z]*$")
_LEGACY_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_stem_date(stem: str) -> date | None:
    """Resolve a raw bhavcopy filename stem to its business date. Handles both
    the current `YYYYMMDD...` convention and the legacy `cmDDMonYYYYbhav` /
    `foDDMonYYYYbhav` convention used by the older locally-archived files
    (over 1,300 of them in the equity archive alone). Previously only the
    digit-prefixed format was recognized, so every legacy file fell through
    as an unparseable date on every single load call regardless of
    `from_date` -- re-hashed and re-processed on every run instead of being
    correctly skipped once already loaded -- and any file that reached the
    manifest step was written under a shared literal "UNKNOWN" business-date
    key instead of its own real date, so unrelated legacy files collided on
    the same primary key (reproduced live: a same-key INSERT OR REPLACE
    retry after an earlier failure hit a DuckDB-internal fatal index error,
    "Failed to delete all rows from index", permanently invalidating the
    connection it ran on)."""
    if len(stem) >= 8 and stem[:8].isdigit():
        try:
            return date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
        except ValueError:
            return None
    match = _LEGACY_STEM_DATE.match(stem)
    if not match:
        return None
    day_text, month_text, year_text = match.groups()
    month = _LEGACY_MONTH_ABBR.get(month_text.lower())
    if month is None:
        return None
    try:
        return date(int(year_text), month, int(day_text))
    except ValueError:
        return None


class NSEBhavcopy:
    def __init__(self) -> None:
        ensure_dirs()
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)

    def download_equity_day(self, day: date) -> tuple[bool, str]:
        stage = "equity_download"
        date_str = day.strftime("%Y%m%d")
        filepath = EQUITY_BHAV_DIR / f"{date_str}.csv"

        if day < NSE_CONTENT_URL_CUTOFF:
            elog.mark(
                stage,
                date_str,
                "skipped",
                error="pre_2024_not_available_on_this_url",
            )
            return True, "pre_2024_skip"

        if filepath.exists() and filepath.stat().st_size > 5000:
            elog.mark(stage, date_str, "downloaded")
            return True, "cached"
        if filepath.exists() and elog.is_done(stage, date_str):
            return True, "cached"

        try:
            from jugaad_data.nse import bhavcopy_save

            require_free_space(filepath, 64 * 1024 * 1024, "equity bhavcopy download")
            bhavcopy_save(day, str(EQUITY_BHAV_DIR))
            saved = self._find_jugaad_file(EQUITY_BHAV_DIR, day)
            if saved and os.path.getsize(saved) > 5000:
                self._move_to_standard_name(saved, filepath)
                elog.mark(stage, date_str, "downloaded")
                return True, "jugaad"
        except Exception as error:
            elog.log(
                "nse_bhavcopy_equity",
                "jugaad_failed",
                str(error),
                date_str=date_str,
                fallback_used="nse_direct",
            )

        try:
            url = (
                "https://nsearchives.nseindia.com/content/cm/"
                f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv.zip"
            )
            response = self._nse_get(url)
            if response.status_code == 200:
                self._extract_first_csv(response.content, filepath)
                elog.mark(stage, date_str, "downloaded")
                return True, "nse_direct"
            if response.status_code == 404:
                elog.mark(stage, date_str, "skipped", error="holiday or non-trading day")
                return True, "holiday"
            elog.log(
                "nse_bhavcopy_equity",
                "direct_status",
                f"HTTP {response.status_code}",
                date_str=date_str,
            )
        except (RawSourceValidationError, zipfile.BadZipFile) as error:
            # Distinct from a plain network/status failure: the server answered
            # 200 but the bytes are not real bhavcopy data (zero-byte, an HTML
            # error/challenge page, or a corrupt/truncated archive). This must
            # never be folded into the generic "unavailable" bucket below --
            # that bucket is presented to the operator as a quiet WAIT (see
            # smart_collector.run_logged_task), which is correct for "NSE
            # hasn't published yet" but would silently hide a persistent
            # parsing/validation break behind the same quiet tag forever.
            elog.log("nse_bhavcopy_equity", "direct_invalid_content", str(error), date_str=date_str)
            elog.mark(stage, date_str, "failed", error=f"invalid_response: {error}")
            return False, "invalid_response"
        except Exception as error:
            elog.log("nse_bhavcopy_equity", "direct_failed", str(error), date_str=date_str)

        elog.mark(stage, date_str, "failed", error="all sources exhausted")
        return False, "unavailable"

    def download_fo_day(self, day: date) -> tuple[bool, str]:
        date_str = day.strftime("%Y%m%d")
        stage = "fo_download"
        filepath = FO_BHAV_DIR / f"{date_str}.csv"

        if day < NSE_CONTENT_URL_CUTOFF:
            elog.mark(
                stage,
                date_str,
                "skipped",
                error="pre_2024_not_available_on_this_url",
            )
            return True, "pre_2024_skip"

        if filepath.exists() and filepath.stat().st_size > 10000:
            elog.mark(stage, date_str, "downloaded")
            return True, "cached"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nseindia.com",
            "Accept": "application/zip, application/octet-stream, */*",
        }
        url = (
            "https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
        )

        invalid_content_detected = False
        for attempt in range(3):
            try:
                print(f"fo_download: downloading {date_str} (attempt {attempt + 1})")
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 404:
                    print(f"fo_download: {date_str} not found (holiday?)")
                    elog.mark(stage, date_str, "skipped", error="404_holiday")
                    return True, "holiday"

                if response.status_code != 200:
                    print(f"fo_download: {date_str} HTTP {response.status_code}")
                    continue

                if len(response.content) == 0 or not response.content.strip():
                    # Zero-byte or whitespace-only body on a 200: this is not a
                    # "try again later" condition, it's a malformed response --
                    # classify distinctly (see equity path above for why).
                    print(f"fo_download: {date_str} empty/whitespace-only response body")
                    elog.log(
                        "nse_bhavcopy_fo",
                        "empty_response",
                        f"{len(response.content)} bytes",
                        date_str=date_str,
                    )
                    invalid_content_detected = True
                    break

                if len(response.content) < 1000:
                    print(
                        f"fo_download: {date_str} response too small "
                        f"({len(response.content)} bytes)"
                    )
                    continue
                if len(response.content) > MAX_ARCHIVE_BYTES:
                    raise RuntimeError("F&O bhavcopy archive exceeds the download bound")

                print(f"fo_download: extracting {date_str} ({len(response.content)} bytes)")
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = archive.namelist()
                    csv_name = next(
                        (name for name in names if name.lower().endswith(".csv")),
                        names[0] if names else None,
                    )
                    if not csv_name:
                        print(f"fo_download: no CSV in ZIP for {date_str}")
                        continue

                    info = archive.getinfo(csv_name)
                    if info.file_size > MAX_EXTRACTED_CSV_BYTES:
                        raise RuntimeError("F&O bhavcopy CSV exceeds the extraction bound")
                    require_free_space(filepath, info.file_size * 2, "F&O bhavcopy extraction")
                    csv_bytes = archive.read(csv_name)
                    _validate_csv_payload(csv_bytes, context=filepath.name)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(csv_bytes)
                    print(
                        f"fo_download: saved {date_str} -> {filepath} "
                        f"({len(csv_bytes)} bytes)"
                    )

                if filepath.exists() and filepath.stat().st_size > 10000:
                    elog.mark(stage, date_str, "downloaded")
                    return True, "nse_direct"

                print(f"fo_download: save failed for {date_str}")
            except (zipfile.BadZipFile, RawSourceValidationError) as error:
                print(f"fo_download: invalid content for {date_str}: {error}")
                elog.log("nse_bhavcopy_fo", "bad_zip_or_invalid_content", str(error), date_str=date_str)
                invalid_content_detected = True
                break
            except Exception as error:
                print(f"fo_download: error {date_str} attempt {attempt + 1}: {error}")
                time.sleep(2)

        if invalid_content_detected:
            elog.mark(stage, date_str, "failed", error="invalid_response_content")
            return False, "invalid_response"

        elog.mark(stage, date_str, "failed", error="all_attempts_exhausted")
        elog.log(
            "nse_bhavcopy_fo",
            "download_failed",
            "all attempts failed",
            date_str=date_str,
        )
        return False, "unavailable"

    def download_equity_range(
        self,
        from_date: date,
        to_date: date,
        on_progress=None,
    ) -> dict[str, int]:
        return self._download_range("equity", from_date, to_date, on_progress)

    def download_fo_range(
        self,
        from_date: date,
        to_date: date,
        on_progress=None,
    ) -> dict[str, int]:
        return self._download_range("fo", from_date, to_date, on_progress)

    def load_equity_to_duckdb(
        self,
        symbols: list[str] | str = "ALL",
        from_date: date | None = None,
    ) -> int:
        files = sorted(glob.glob(str(EQUITY_BHAV_DIR / "*.csv")))
        total_inserted = 0
        stage = self._load_stage("equity_load", symbols)
        symbol_filter = self._symbol_filter(symbols)
        fo_symbol_filter = self._fo_symbol_filter(symbols)

        for filepath in files:
            date_str = Path(filepath).stem
            if from_date and self._file_before(filepath, from_date):
                continue

            file_sha256 = self._file_sha256(filepath)
            progress_key = self._progress_key(date_str, file_sha256)
            if elog.is_done(stage, progress_key):
                continue

            try:
                df_raw = pd.read_csv(filepath, low_memory=False)
                rows_parsed = int(len(df_raw))
                fmt = self._resolve_format(df_raw, EQUITY_FORMATS)
                if fmt is None:
                    self._quarantine(filepath, list(df_raw.columns))
                    elog.mark(stage, progress_key, "failed", error="unknown format")
                    self._record_manifest(
                        filepath=filepath,
                        file_sha256=file_sha256,
                        source_kind="NSE_EQUITY_BHAVCOPY",
                        business_date=self._business_date_from_stem(date_str),
                        parser_schema_version="unknown",
                        rows_parsed=rows_parsed,
                        rows_accepted=0,
                        rows_rejected=rows_parsed,
                        rejection_reasons=["unknown_format"],
                    )
                    continue

                normalised = self._normalise_equity_frame(df_raw, fmt)
                business_date = self._business_date_from_frame(normalised)
                frame = normalised.dropna(subset=["symbol", "date", "close"])
                if symbol_filter is not None:
                    frame = frame[frame["symbol"].isin(symbol_filter)]

                if not frame.empty:
                    inserted = self._insert_ohlcv(frame)
                    total_inserted += inserted

                if fmt.get("version") == "v3_unified":
                    # FO_FORMATS[0] is the raw column-candidate spec (list-valued);
                    # _normalise_fo_frame needs a *resolved* mapping (single column
                    # name per field), the same shape self._resolve_format already
                    # produces for the equity `fmt` above -- resolve it against this
                    # same unified file before use.
                    fo_fmt = self._resolve_format(df_raw, [FO_FORMATS[0]])
                    if fo_fmt is not None:
                        fo_frame = self._normalise_fo_frame(
                            df_raw,
                            fo_fmt,
                            require_fo_types=True,
                        )
                        fo_frame = fo_frame.dropna(
                            subset=["symbol", "expiry", "date", "option_type"]
                        )
                        if fo_symbol_filter is not None:
                            fo_frame = fo_frame[fo_frame["symbol"].isin(fo_symbol_filter)]
                        if not fo_frame.empty:
                            self._insert_fo_eod(fo_frame)

                elog.mark(stage, progress_key, "loaded" if not frame.empty else "skipped")
                rows_accepted = int(len(frame))
                self._record_manifest(
                    filepath=filepath,
                    file_sha256=file_sha256,
                    source_kind="NSE_EQUITY_BHAVCOPY",
                    business_date=business_date,
                    parser_schema_version=str(fmt.get("version") or "unknown"),
                    rows_parsed=rows_parsed,
                    rows_accepted=rows_accepted,
                    rows_rejected=max(0, int(len(normalised)) - rows_accepted),
                    rejection_reasons=[]
                    if rows_accepted == int(len(normalised))
                    else ["symbol_filter_or_missing_fields"],
                )
            except Exception as error:
                elog.log(
                    "nse_bhavcopy_equity",
                    "load_failed",
                    str(error),
                    date_str=date_str,
                )
                elog.mark(stage, progress_key, "failed", error=str(error))
                self._record_manifest(
                    filepath=filepath,
                    file_sha256=file_sha256,
                    source_kind="NSE_EQUITY_BHAVCOPY",
                    business_date=self._business_date_from_stem(date_str),
                    parser_schema_version="unknown",
                    rows_parsed=0,
                    rows_accepted=0,
                    rows_rejected=0,
                    rejection_reasons=[str(error)[:200]],
                )

        return total_inserted

    def load_fo_to_duckdb(
        self,
        symbols: list[str] | str = "ALL",
        from_date: date | None = None,
    ) -> int:
        total_inserted = 0
        stage = self._load_stage("fo_load", symbols)
        symbol_filter = self._fo_symbol_filter(symbols)

        for filepath in sorted(glob.glob(str(FO_BHAV_DIR / "*.csv"))):
            date_str = Path(filepath).stem
            if from_date and self._file_before(filepath, from_date):
                continue

            file_sha256 = self._file_sha256(filepath)
            progress_key = self._progress_key(date_str, file_sha256)
            if elog.is_done(stage, progress_key):
                continue

            try:
                df_raw = pd.read_csv(filepath, low_memory=False)
                rows_parsed = int(len(df_raw))
                fmt = self._resolve_format(df_raw, FO_FORMATS)
                if fmt is None:
                    self._quarantine(filepath, list(df_raw.columns))
                    elog.mark(stage, progress_key, "failed", error="unknown format")
                    self._record_manifest(
                        filepath=filepath,
                        file_sha256=file_sha256,
                        source_kind="NSE_FO_BHAVCOPY",
                        business_date=self._business_date_from_stem(date_str),
                        parser_schema_version="unknown",
                        rows_parsed=rows_parsed,
                        rows_accepted=0,
                        rows_rejected=rows_parsed,
                        rejection_reasons=["unknown_format"],
                    )
                    continue

                normalised = self._normalise_fo_frame(
                    df_raw,
                    fmt,
                    require_fo_types=fmt.get("version") == "v3_unified",
                )
                business_date = self._business_date_from_frame(normalised)
                frame = normalised.dropna(subset=["symbol", "expiry", "date", "option_type"])
                if symbol_filter is not None:
                    frame = frame[frame["symbol"].isin(symbol_filter)]

                if frame.empty:
                    elog.mark(stage, progress_key, "skipped")
                    self._record_manifest(
                        filepath=filepath,
                        file_sha256=file_sha256,
                        source_kind="NSE_FO_BHAVCOPY",
                        business_date=business_date,
                        parser_schema_version=str(fmt.get("version") or "unknown"),
                        rows_parsed=rows_parsed,
                        rows_accepted=0,
                        rows_rejected=int(len(normalised)),
                        rejection_reasons=["symbol_filter_or_missing_fields"],
                    )
                    continue

                inserted = self._insert_fo_eod(frame)
                total_inserted += inserted
                elog.mark(stage, progress_key, "loaded")
                rows_accepted = int(len(frame))
                self._record_manifest(
                    filepath=filepath,
                    file_sha256=file_sha256,
                    source_kind="NSE_FO_BHAVCOPY",
                    business_date=business_date,
                    parser_schema_version=str(fmt.get("version") or "unknown"),
                    rows_parsed=rows_parsed,
                    rows_accepted=rows_accepted,
                    rows_rejected=max(0, int(len(normalised)) - rows_accepted),
                    rejection_reasons=[]
                    if rows_accepted == int(len(normalised))
                    else ["symbol_filter_or_missing_fields"],
                )
            except Exception as error:
                elog.log("nse_bhavcopy_fo", "load_failed", str(error), date_str=date_str)
                elog.mark(stage, progress_key, "failed", error=str(error))
                self._record_manifest(
                    filepath=filepath,
                    file_sha256=file_sha256,
                    source_kind="NSE_FO_BHAVCOPY",
                    business_date=self._business_date_from_stem(date_str),
                    parser_schema_version="unknown",
                    rows_parsed=0,
                    rows_accepted=0,
                    rows_rejected=0,
                    rejection_reasons=[str(error)[:200]],
                )

        return total_inserted

    def _normalise_equity_frame(self, df_raw: pd.DataFrame, fmt: dict[str, str]) -> pd.DataFrame:
        source = "NSE_EQUITY_BHAVCOPY"
        frame = pd.DataFrame(
            {
                "symbol": self._text_series(df_raw[fmt["symbol"]]).str.upper(),
                "date": pd.to_datetime(df_raw[fmt["date"]], errors="coerce").dt.date,
                "open": pd.to_numeric(df_raw[fmt["open"]], errors="coerce"),
                "high": pd.to_numeric(df_raw[fmt["high"]], errors="coerce"),
                "low": pd.to_numeric(df_raw[fmt["low"]], errors="coerce"),
                "close": pd.to_numeric(df_raw[fmt["close"]], errors="coerce"),
                "volume": pd.to_numeric(df_raw[fmt["volume"]], errors="coerce"),
                "source": source,
            }
        )
        mask = pd.Series(True, index=df_raw.index)
        if fmt.get("series"):
            series = self._text_series(df_raw[fmt["series"]]).str.upper()
            mask &= series.eq("EQ")
        if fmt.get("instrument"):
            instrument = self._text_series(df_raw[fmt["instrument"]]).str.upper()
            mask &= instrument.isin(EQUITY_INSTRUMENT_TYPES)
        return frame[mask].copy()

    def _normalise_fo_frame(
        self,
        df_raw: pd.DataFrame,
        fmt: dict[str, str],
        require_fo_types: bool,
    ) -> pd.DataFrame:
        instrument = self._text_series(df_raw[fmt["instrument"]]).str.upper()
        source = df_raw
        if require_fo_types:
            source = df_raw[instrument.isin(FO_INSTRUMENT_TYPES)]
            instrument = instrument.loc[source.index]

        option_type = self._text_series(source[fmt["option_type"]]).str.upper()
        option_type = option_type.mask(option_type.isin(["", "XX", "NAN"]), "FUT")
        option_type = option_type.mask(instrument.isin(FO_FUTURE_INSTRUMENT_TYPES), "FUT")

        frame = pd.DataFrame(
            {
                "symbol": self._text_series(source[fmt["symbol"]]).str.upper(),
                "expiry": pd.to_datetime(source[fmt["expiry"]], errors="coerce").dt.date,
                "strike": pd.to_numeric(source[fmt["strike"]], errors="coerce").fillna(0),
                "option_type": option_type,
                "date": pd.to_datetime(source[fmt["date"]], errors="coerce").dt.date,
                "open": pd.to_numeric(source[fmt["open"]], errors="coerce"),
                "high": pd.to_numeric(source[fmt["high"]], errors="coerce"),
                "low": pd.to_numeric(source[fmt["low"]], errors="coerce"),
                "close": pd.to_numeric(source[fmt["close"]], errors="coerce"),
                "volume": pd.to_numeric(source[fmt["volume"]], errors="coerce"),
                "oi": pd.to_numeric(source[fmt["oi"]], errors="coerce"),
                "oi_change": pd.to_numeric(source[fmt["oi_change"]], errors="coerce"),
                "source": "NSE_FO_BHAVCOPY",
                "source_format": str(fmt.get("version") or "unknown"),
            }
        )
        frame["price_quality"] = frame.apply(
            lambda row: quality_for_ingest(row.to_dict()),
            axis=1,
        )
        return frame

    def validate_vs_angel(
        self,
        symbol: str,
        angel_df: pd.DataFrame,
        n_days: int = 30,
    ) -> dict[str, Any]:
        result = resolve_read(
            ReadOperation.LOAD_RAW_OHLCV_FOR_VALIDATION,
            {"symbol": symbol.upper(), "n_days": n_days},
        )
        bhav_rows = result["rows"]

        if not bhav_rows:
            return {
                "matches": 0,
                "mismatches": 0,
                "max_diff_pct": None,
                "recommendation": "no_bhavcopy_data",
            }
        if angel_df.empty:
            return {
                "matches": 0,
                "mismatches": 0,
                "max_diff_pct": None,
                "recommendation": "no_angel_data",
            }

        bhav_df = pd.DataFrame(bhav_rows, columns=["date", "close"])
        bhav_df["date"] = pd.to_datetime(bhav_df["date"]).dt.normalize()
        bhav_df = bhav_df.set_index("date")

        angel_close = angel_df["Close"] if "Close" in angel_df.columns else angel_df["close"]
        angel_close = angel_close.copy()
        angel_close.index = pd.to_datetime(angel_close.index).normalize()

        common = bhav_df.index.intersection(angel_close.index)
        if common.empty:
            return {
                "matches": 0,
                "mismatches": 0,
                "max_diff_pct": None,
                "recommendation": "no_common_dates",
            }

        diff = (
            (bhav_df.loc[common, "close"] - angel_close.loc[common]).abs()
            / bhav_df.loc[common, "close"]
            * 100
        )
        mismatches = int((diff > 0.5).sum())
        return {
            "matches": int(len(common) - mismatches),
            "mismatches": mismatches,
            "max_diff_pct": round(float(diff.max()), 4),
            "recommendation": (
                "both_agree"
                if mismatches == 0
                else "use_bhavcopy"
                if mismatches < 3
                else "investigate"
            ),
        }

    def inspect_equity_files(
        self,
        symbol: str,
        from_date: date | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rows": 0,
            "formats": [],
            "parse_errors": [],
        }
        symbol_filter = symbol.strip().upper()
        for filepath in sorted(glob.glob(str(EQUITY_BHAV_DIR / "*.csv"))):
            if from_date and self._file_before(filepath, from_date):
                continue
            try:
                df_raw = pd.read_csv(filepath, low_memory=False)
                resolved = self._resolve_format_with_name(df_raw, EQUITY_FORMATS)
                if resolved is None:
                    result["parse_errors"].append(f"{Path(filepath).name}: unknown format")
                    continue
                fmt, format_name = resolved
                result["formats"].append(format_name)
                frame = self._normalise_equity_frame(df_raw, fmt)
                result["rows"] += int(frame["symbol"].eq(symbol_filter).sum())
            except Exception as error:
                result["parse_errors"].append(f"{Path(filepath).name}: {error}")

        result["formats"] = sorted(set(result["formats"]))
        return result

    def inspect_fo_files(
        self,
        symbol: str,
        from_date: date | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rows": 0,
            "formats": [],
            "instrument_types": [],
            "parse_errors": [],
        }
        symbol_filter = symbol.strip().upper()
        for filepath, require_fo_types in self._fo_source_files(from_date):
            if from_date and self._file_before(filepath, from_date):
                continue
            try:
                df_raw = pd.read_csv(filepath, low_memory=False)
                resolved = self._resolve_format_with_name(df_raw, FO_FORMATS)
                if resolved is None:
                    if require_fo_types:
                        continue
                    result["parse_errors"].append(f"{Path(filepath).name}: unknown format")
                    continue
                fmt, format_name = resolved
                if require_fo_types and fmt.get("version") != "v3_unified":
                    continue
                result["formats"].append(format_name)
                symbols = self._text_series(df_raw[fmt["symbol"]]).str.upper()
                instrument = self._text_series(df_raw[fmt["instrument"]]).str.upper()
                if require_fo_types:
                    fo_mask = instrument.isin(FO_INSTRUMENT_TYPES)
                    symbols = symbols[fo_mask]
                    instrument = instrument[fo_mask]
                matches = symbols.eq(symbol_filter)
                result["rows"] += int(matches.sum())
                instruments = (
                    self._canonical_instrument_types(instrument.loc[matches])
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )
                result["instrument_types"].extend(str(item) for item in instruments)
            except Exception as error:
                result["parse_errors"].append(f"{Path(filepath).name}: {error}")

        result["formats"] = sorted(set(result["formats"]))
        result["instrument_types"] = sorted(set(result["instrument_types"]))
        return result

    def parser_coverage(self, from_date: date | None = None) -> dict[str, Any]:
        coverage: dict[str, set[str]] = {
            "equity_formats_tested": set(),
            "fo_formats_tested": set(),
            "instrument_types_tested": set(),
        }

        for filepath in sorted(glob.glob(str(EQUITY_BHAV_DIR / "*.csv"))):
            if from_date and self._file_before(filepath, from_date):
                continue
            try:
                df_raw = pd.read_csv(filepath, nrows=5, low_memory=False)
                resolved = self._resolve_format_with_name(df_raw, EQUITY_FORMATS)
                if resolved:
                    coverage["equity_formats_tested"].add(resolved[1])
            except Exception as error:
                elog.log("nse_bhavcopy_equity", "coverage_failed", str(error), date_str=Path(filepath).stem)

        for filepath in sorted(glob.glob(str(FO_BHAV_DIR / "*.csv"))):
            if from_date and self._file_before(filepath, from_date):
                continue
            try:
                df_raw = pd.read_csv(filepath, low_memory=False)
                resolved = self._resolve_format_with_name(df_raw, FO_FORMATS)
                if not resolved:
                    continue
                fmt, format_name = resolved
                coverage["fo_formats_tested"].add(format_name)
                instruments = (
                    self._canonical_instrument_types(df_raw[fmt["instrument"]])
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )
                coverage["instrument_types_tested"].update(str(item) for item in instruments)
            except Exception as error:
                elog.log("nse_bhavcopy_fo", "coverage_failed", str(error), date_str=Path(filepath).stem)

        for filepath in sorted(glob.glob(str(EQUITY_BHAV_DIR / "*.csv"))):
            if from_date and self._file_before(filepath, from_date):
                continue
            try:
                df_raw = pd.read_csv(filepath, low_memory=False)
                resolved = self._resolve_format_with_name(df_raw, FO_FORMATS)
                if not resolved:
                    continue
                fmt, format_name = resolved
                if fmt.get("version") != "v3_unified":
                    continue
                instruments = self._text_series(df_raw[fmt["instrument"]]).str.upper()
                fo_instruments = instruments[instruments.isin(FO_INSTRUMENT_TYPES)]
                if not fo_instruments.empty:
                    coverage["fo_formats_tested"].add(format_name)
                    coverage["instrument_types_tested"].update(
                        str(item)
                        for item in self._canonical_instrument_types(fo_instruments)
                        .dropna()
                        .unique()
                        .tolist()
                    )
            except Exception as error:
                elog.log("nse_bhavcopy_fo", "unified_coverage_failed", str(error), date_str=Path(filepath).stem)

        return {
            "equity_formats_tested": sorted(coverage["equity_formats_tested"]),
            "fo_formats_tested": sorted(coverage["fo_formats_tested"]),
            "instrument_types_tested": sorted(coverage["instrument_types_tested"]),
            "notes": "pre_2023 format only visible in older archived files",
        }

    def _download_range(
        self,
        kind: str,
        from_date: date,
        to_date: date,
        on_progress,
    ) -> dict[str, int]:
        result = {"downloaded": 0, "skipped": 0, "failed": 0, "holiday": 0}
        current = from_date
        consecutive_failures = 0

        while current <= to_date:
            if not is_trading_day_candidate(current):
                current += timedelta(days=1)
                continue

            if kind == "equity":
                success, source = self.download_equity_day(current)
            else:
                success, source = self.download_fo_day(current)

            if source == "holiday":
                result["holiday"] += 1
            elif source in {"cached", "pre_2024_skip"}:
                result["skipped"] += 1
            elif success:
                result["downloaded"] += 1
                consecutive_failures = 0
            else:
                result["failed"] += 1
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print("bhavcopy: 5 consecutive failures, pausing 5 minutes")
                    time.sleep(300)
                    consecutive_failures = 0

            if on_progress:
                on_progress(current, dict(result))

            current += timedelta(days=1)
            time.sleep(0.5)

        return result

    def _insert_ohlcv(self, df: pd.DataFrame) -> int:
        count = int(len(df))
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                con.execute("BEGIN")
                con.register("df_eq", df)
                con.execute(
                    """
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, date, open, high, low, close, volume, source)
                    SELECT symbol, date, open, high, low, close,
                           CAST(COALESCE(volume, 0) AS BIGINT), source
                    FROM df_eq
                    WHERE close > 0
                    """
                )
                con.execute("COMMIT")
                return count
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.close()

    def _insert_fo_eod(self, df: pd.DataFrame) -> int:
        count = int(len(df))
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                con.execute("BEGIN")
                con.register("df_fo", df)
                con.execute(
                    """
                    INSERT OR REPLACE INTO fo_eod
                    (symbol, expiry, strike, option_type, date,
                     open, high, low, close, volume, oi, oi_change,
                     source, source_format, price_quality)
                    SELECT symbol, expiry, COALESCE(strike, 0), option_type, date,
                           open, high, low, close,
                           CAST(COALESCE(volume, 0) AS BIGINT),
                           CAST(COALESCE(oi, 0) AS BIGINT),
                           CAST(COALESCE(oi_change, 0) AS BIGINT),
                           source, source_format, price_quality
                    FROM df_fo
                    WHERE close > 0 AND expiry IS NOT NULL
                    """
                )
                con.execute("COMMIT")
                return count
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.close()

    def _nse_get(self, url: str) -> requests.Response:
        try:
            self.session.get("https://www.nseindia.com", timeout=8)
            time.sleep(0.25)
        except Exception:
            pass
        return self.session.get(url, timeout=30)

    def _extract_first_csv(self, content: bytes, filepath: Path) -> None:
        if len(content) > MAX_ARCHIVE_BYTES:
            raise RuntimeError("equity bhavcopy archive exceeds the download bound")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            info = archive.getinfo(csv_name)
            if info.file_size > MAX_EXTRACTED_CSV_BYTES:
                raise RuntimeError("equity bhavcopy CSV exceeds the extraction bound")
            require_free_space(filepath, info.file_size * 2, "equity bhavcopy extraction")
            with archive.open(csv_name) as csv_file:
                extracted = csv_file.read()
            _validate_csv_payload(extracted, context=filepath.name)
            filepath.write_bytes(extracted)

    def _resolve_format(
        self,
        df: pd.DataFrame,
        formats: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        resolved = self._resolve_format_with_name(df, formats)
        return resolved[0] if resolved else None

    def _resolve_format_with_name(
        self,
        df: pd.DataFrame,
        formats: list[dict[str, Any]],
    ) -> tuple[dict[str, str], str] | None:
        columns = {str(column).strip(): column for column in df.columns}
        for index, fmt in enumerate(formats):
            resolved: dict[str, str] = {}
            ok = True
            for target, candidates in fmt.items():
                if target in FORMAT_META_KEYS:
                    continue
                match = next((columns[candidate] for candidate in candidates if candidate in columns), None)
                if match is None:
                    if target == "series":
                        continue
                    ok = False
                    break
                resolved[target] = match
            if ok:
                version = str(fmt.get("version", "post_2023" if index == 0 else "pre_2023"))
                resolved["version"] = version
                return resolved, version
        return None

    def _quarantine(self, filepath: str, columns: list[Any]) -> None:
        source_path = Path(filepath)
        dest = QUARANTINE_DIR / f"{source_path.parent.name}_{source_path.name}"
        shutil.copy(filepath, dest)
        elog.log(
            "nse_bhavcopy",
            "unknown_format",
            f"columns: {columns}",
            date_str=source_path.name,
        )
        print(f"bhavcopy: unknown format quarantined: {filepath}")

    def _find_jugaad_file(self, directory: Path, day: date) -> str | None:
        patterns = [
            f"*{day.strftime('%d%m%Y')}*",
            f"*{day.strftime('%Y%m%d')}*",
            f"*{day.strftime('%d-%m-%Y')}*",
        ]
        return self._first_match(directory, patterns)

    def _find_jugaad_fo_file(self, directory: Path, day: date) -> str | None:
        patterns = [
            f"fo{day.strftime('%d%b%Y').upper()}*",
            f"*{day.strftime('%Y%m%d')}*fo*",
            f"*fo*{day.strftime('%d%m%Y')}*",
        ]
        return self._first_match(directory, patterns)

    def _first_match(self, directory: Path, patterns: list[str]) -> str | None:
        for pattern in patterns:
            matches = glob.glob(str(directory / pattern))
            if matches:
                return matches[0]
        return None

    def _move_to_standard_name(self, source: str, filepath: Path) -> None:
        _validate_csv_payload(Path(source).read_bytes(), context=Path(source).name)
        if Path(source).resolve() == filepath.resolve():
            return
        os.replace(source, filepath)

    def _fo_source_files(self, from_date: date | None = None) -> list[tuple[str, bool]]:
        files: list[tuple[str, bool]] = []
        for filepath in sorted(glob.glob(str(EQUITY_BHAV_DIR / "*.csv"))):
            if not from_date or not self._file_before(filepath, from_date):
                files.append((filepath, True))
        for filepath in sorted(glob.glob(str(FO_BHAV_DIR / "*.csv"))):
            if not from_date or not self._file_before(filepath, from_date):
                files.append((filepath, False))
        return files

    def _file_before(self, filepath: str, from_date: date) -> bool:
        parsed = _parse_stem_date(Path(filepath).stem)
        return parsed < from_date if parsed else False

    def _symbol_filter(self, symbols: list[str] | str) -> set[str] | None:
        if symbols == "ALL":
            return None
        if isinstance(symbols, str):
            return {symbols.strip().upper()} if symbols.strip() else set()
        return {symbol.strip().upper() for symbol in symbols if symbol.strip()}

    def _fo_symbol_filter(self, symbols: list[str] | str) -> set[str] | None:
        symbol_filter = self._symbol_filter(symbols)
        if symbol_filter is not None:
            symbol_filter.update(INDEX_FO_SYMBOLS)
        return symbol_filter

    def _load_stage(self, prefix: str, symbols: list[str] | str) -> str:
        if symbols == "ALL":
            return f"{prefix}_{PIPELINE_VERSION}_all"
        key = ",".join(sorted(symbol.strip().upper() for symbol in symbols if symbol.strip()))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{PIPELINE_VERSION}_{digest}"

    def _progress_key(self, date_str: str, file_sha256: str) -> str:
        # Folds the file's content hash into the checkpoint key so a changed
        # file (same business date, different bytes) gets a fresh checkpoint
        # instead of being silently skipped forever.
        return f"{date_str}:{file_sha256[:12]}"

    def _business_date_from_stem(self, date_str: str) -> str | None:
        parsed = _parse_stem_date(date_str)
        return parsed.isoformat() if parsed else None

    def _business_date_from_frame(self, frame: pd.DataFrame) -> str | None:
        if frame.empty or "date" not in frame.columns:
            return None
        values = frame["date"].dropna()
        if values.empty:
            return None
        try:
            return str(values.mode(dropna=True).iloc[0])
        except Exception:
            return str(values.iloc[0])

    def _file_sha256(self, filepath: str) -> str:
        digest = hashlib.sha256()
        with open(filepath, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_manifest(
        self,
        *,
        filepath: str,
        file_sha256: str,
        source_kind: str,
        business_date: str | None,
        parser_schema_version: str,
        rows_parsed: int,
        rows_accepted: int,
        rows_rejected: int,
        rejection_reasons: list[str],
    ) -> None:
        source_id = f"{source_kind}:{business_date or 'UNKNOWN'}:{file_sha256[:16]}"
        payload = {
            "source_id": source_id,
            "source_path": str(Path(filepath).resolve()),
            "source_sha256": file_sha256,
            "source_identity": source_kind,
            "expected_scope_json": json.dumps(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "source_kind": source_kind,
                    "business_date": business_date,
                    "parser_schema_version": parser_schema_version,
                    "rows_parsed": rows_parsed,
                    "rows_accepted": rows_accepted,
                    "rows_rejected": rows_rejected,
                    "rejection_reasons": rejection_reasons,
                },
                separators=(",", ":"),
            ),
            "observed_record_count": rows_accepted,
            "ingestion_completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "validation_completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completeness_status": "COMPLETE",
            "downstream_commit_id": source_id,
        }
        try:
            # This loader itself only ever runs inside an active WriterService
            # dispatch (writer_connection() raises PermissionError otherwise --
            # see db/duckdb_client.py), so the manifest row is written through
            # the same active transaction directly. Routing it back through
            # submit_write() would queue a request behind the very dispatch
            # call that's blocking on this line -- a guaranteed self-deadlock.
            from db.write_service import _record_ingestion_manifest

            _record_ingestion_manifest(DUCKDB_PATH, payload)
        except Exception as error:
            # Manifest recording is an audit trail on top of the eligibility
            # contract, not the source of truth -- never fail ingestion over it.
            elog.log(
                "nse_bhavcopy_manifest",
                "manifest_write_failed",
                str(error),
                date_str=business_date,
            )

    def _text_series(self, series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip()

    def _canonical_instrument_types(self, series: pd.Series) -> pd.Series:
        return self._text_series(series).str.upper().replace(FO_INSTRUMENT_TYPE_ALIASES)
