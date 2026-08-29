from __future__ import annotations

import datetime as _dt
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Callable

import duckdb
import pytest

import data.error_logger as error_logger_module
import data.nse_bhavcopy as nse_bhavcopy_module
from data.error_logger import ErrorLogger
from data.nse_bhavcopy import _parse_stem_date
from db.duckdb_client import _create_schema_objects_at, _writer_transaction_scope, open_read_connection
from db.migrations import migrate_database
from db.write_service import WriteOperation, _dispatch_operation
from db.writer_lease import release_writer_lease, require_writer_lease


EQUITY_V3_HEADER = "SctySrs,FinInstrmTp,ISIN,TckrSymb,TradDt,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol"
FO_V3_HEADER = (
    "FinInstrmTp,TckrSymb,XpryDt,StrkPric,OptnTp,TradDt,"
    "OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,OpnIntrst,ChngInOpnIntrst"
)
EQUITY_PRE2023_HEADER = "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY"


def equity_v3_csv(symbol: str, trade_date: str, close: float = 105.0) -> str:
    return (
        f"{EQUITY_V3_HEADER}\n"
        f"EQ,STK,INE000A00000,{symbol},{trade_date},100,110,90,{close},1000\n"
    )


def equity_pre2023_csv(symbol: str, trade_date: str, close: float = 63.55) -> str:
    return (
        f"{EQUITY_PRE2023_HEADER}\n"
        f"{symbol},EQ,{trade_date},60.75,64.10,59.85,{close},125996\n"
    )


def fo_v3_csv(symbol: str, trade_date: str, expiry: str) -> str:
    return (
        f"{FO_V3_HEADER}\n"
        f"FUTSTK,{symbol},{expiry},0,,{trade_date},100,110,90,105,500,1000,10\n"
    )


class ReingestEnv:
    """Wires nse_bhavcopy/error_logger at their real module paths onto a tmp
    disposable DB + tmp raw-file dirs, using the exact same dispatch path
    (WriteOperation -> _dispatch_operation under a writer transaction scope)
    the production WriterService uses -- no parallel test-only code path."""

    def __init__(self, db_path: Path, equity_dir: Path, fo_dir: Path, quarantine_dir: Path) -> None:
        self.db_path = db_path
        self.equity_dir = equity_dir
        self.fo_dir = fo_dir
        self.quarantine_dir = quarantine_dir

    def dispatch(self, operation: WriteOperation, payload: dict[str, Any]) -> Any:
        connection = duckdb.connect(str(self.db_path))
        try:
            with _writer_transaction_scope(connection):
                return _dispatch_operation(operation, payload, self.db_path)
        finally:
            connection.close()

    def load_equity(self) -> dict[str, Any]:
        return self.dispatch(WriteOperation.LOAD_EQUITY_BHAVCOPY, {"from_date": "2000-01-01"})

    def load_fo(self) -> dict[str, Any]:
        return self.dispatch(WriteOperation.LOAD_FO_BHAVCOPY, {"from_date": "2000-01-01"})

    def read(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        con = open_read_connection(self.db_path)
        try:
            return con.execute(sql, params or []).fetchall()
        finally:
            con.close()

    def seed_legacy_progress(self, stage: str, date_str: str, status: str = "loaded") -> None:
        ErrorLogger().initialize()
        with sqlite3.connect(str(error_logger_module.ERROR_LOG_PATH)) as con:
            con.execute(
                "INSERT OR REPLACE INTO backfill_progress (stage, date_str, status, error, updated_at) "
                "VALUES (?, ?, ?, NULL, '2020-01-01T00:00:00')",
                [stage, date_str, status],
            )


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReingestEnv:
    equity_dir = tmp_path / "bhavcopy" / "equity"
    fo_dir = tmp_path / "bhavcopy" / "fo"
    quarantine_dir = tmp_path / "quarantine"
    equity_dir.mkdir(parents=True)
    fo_dir.mkdir(parents=True)
    quarantine_dir.mkdir(parents=True)

    monkeypatch.setattr(nse_bhavcopy_module, "EQUITY_BHAV_DIR", equity_dir)
    monkeypatch.setattr(nse_bhavcopy_module, "FO_BHAV_DIR", fo_dir)
    monkeypatch.setattr(nse_bhavcopy_module, "QUARANTINE_DIR", quarantine_dir)
    monkeypatch.setattr(error_logger_module, "ERROR_LOG_PATH", tmp_path / "data_errors.db")

    db_path = tmp_path / "terminal.db"
    require_writer_lease(db_path)
    monkeypatch.setattr(nse_bhavcopy_module, "DUCKDB_PATH", db_path)
    migrate_database(db_path, schema_initializer=_create_schema_objects_at)
    try:
        yield ReingestEnv(db_path, equity_dir, fo_dir, quarantine_dir)
    finally:
        release_writer_lease()


def test_legacy_unversioned_progress_does_not_block_current_pipeline(env: ReingestEnv) -> None:
    (env.equity_dir / "20240102.csv").write_text(equity_v3_csv("TESTSYM", "2024-01-02"))
    # Simulate a pre-fix run's checkpoint: same date, unversioned stage name.
    env.seed_legacy_progress("equity_load_all", "20240102", status="loaded")

    result = env.load_equity()

    assert result["rows"] == 1
    rows = env.read("SELECT upper(source) FROM ohlcv WHERE symbol = 'TESTSYM'")
    assert rows == [("NSE_EQUITY_BHAVCOPY",)]


def test_identical_file_idempotently_skipped_on_rerun(env: ReingestEnv) -> None:
    (env.equity_dir / "20240102.csv").write_text(equity_v3_csv("TESTSYM", "2024-01-02"))

    first = env.load_equity()
    second = env.load_equity()

    assert first["rows"] == 1
    assert second["rows"] == 0
    assert env.read("SELECT count(*) FROM ohlcv")[0][0] == 1
    assert env.read("SELECT count(*) FROM ingestion_manifests")[0][0] == 1


def test_changed_file_hash_is_reprocessed_not_silently_skipped(env: ReingestEnv) -> None:
    path = env.equity_dir / "20240102.csv"
    path.write_text(equity_v3_csv("TESTSYM", "2024-01-02", close=105.0))
    env.load_equity()

    path.write_text(equity_v3_csv("TESTSYM", "2024-01-02", close=999.0))
    result = env.load_equity()

    assert result["rows"] == 1
    rows = env.read("SELECT close FROM ohlcv WHERE symbol = 'TESTSYM'")
    assert rows == [(999.0,)]
    assert env.read("SELECT count(*) FROM ingestion_manifests")[0][0] == 2


def test_reprocessing_same_file_produces_no_duplicate_rows(env: ReingestEnv) -> None:
    (env.equity_dir / "20240102.csv").write_text(equity_v3_csv("TESTSYM", "2024-01-02"))
    (env.fo_dir / "20240102.csv").write_text(fo_v3_csv("TESTSYM", "2024-01-02", "2024-01-25"))

    env.load_equity()
    env.load_fo()
    env.load_equity()
    env.load_fo()

    assert env.read(
        "SELECT count(*) FROM ohlcv WHERE symbol = 'TESTSYM' AND date = DATE '2024-01-02'"
    )[0][0] == 1
    assert env.read("SELECT count(*) FROM ingestion_manifests")[0][0] == 2


def test_source_provenance_is_canonical_never_copied_from_existing_db_value(
    env: ReingestEnv,
) -> None:
    connection = duckdb.connect(str(env.db_path))
    try:
        with _writer_transaction_scope(connection) as writer:
            writer.execute(
                "INSERT INTO ohlcv VALUES ('TESTSYM', DATE '2024-01-02', 1, 1, 1, 1, 1, 'bhavcopy')"
            )
    finally:
        connection.close()

    (env.equity_dir / "20240102.csv").write_text(equity_v3_csv("TESTSYM", "2024-01-02"))
    env.load_equity()

    rows = env.read("SELECT upper(source) FROM ohlcv WHERE symbol = 'TESTSYM'")
    assert rows == [("NSE_EQUITY_BHAVCOPY",)]


def test_fo_rows_never_land_in_ohlcv_or_become_decision_eligible(env: ReingestEnv) -> None:
    (env.fo_dir / "20240102.csv").write_text(fo_v3_csv("TESTSYM", "2024-01-02", "2024-01-25"))

    env.load_fo()

    assert env.read("SELECT count(*) FROM ohlcv WHERE symbol = 'TESTSYM'")[0][0] == 0
    assert (
        env.read("SELECT count(*) FROM decision_eligible_ohlcv WHERE symbol = 'TESTSYM'")[0][0]
        == 0
    )
    fo_rows = env.read("SELECT upper(source) FROM fo_eod WHERE symbol = 'TESTSYM'")
    assert fo_rows == [("NSE_FO_BHAVCOPY",)]


def test_unknown_format_file_is_quarantined_and_stays_non_eligible(env: ReingestEnv) -> None:
    bad_path = env.equity_dir / "20240103.csv"
    bad_path.write_text("GARBAGE,COLUMNS\nx,y\n")

    env.load_equity()

    assert (env.quarantine_dir / "equity_20240103.csv").exists()
    assert env.read("SELECT count(*) FROM ohlcv")[0][0] == 0
    manifest = env.read(
        "SELECT observed_record_count, completeness_status, expected_scope_json "
        "FROM ingestion_manifests"
    )
    assert len(manifest) == 1
    assert manifest[0][0] == 0
    assert manifest[0][1] == "COMPLETE"
    assert '"rows_rejected":1' in manifest[0][2]
    # original file is copied into quarantine, never deleted or moved
    assert bad_path.exists()


def test_csv_parse_crash_still_gets_a_manifest_row(env: ReingestEnv) -> None:
    # A file that fails inside pd.read_csv itself (inconsistent column counts,
    # not just "unrecognised columns") must still produce an audit trail row --
    # not silently vanish from ingestion_manifests.
    (env.equity_dir / "20240103.csv").write_text("A,B,C\n1,2\n3,4,5,6\n")

    env.load_equity()

    manifest = env.read(
        "SELECT observed_record_count, completeness_status FROM ingestion_manifests"
    )
    assert len(manifest) == 1
    assert manifest[0] == (0, "COMPLETE")
    assert env.read("SELECT count(*) FROM ohlcv")[0][0] == 0


def test_verified_equity_bhavcopy_becomes_decision_eligible(env: ReingestEnv) -> None:
    (env.equity_dir / "20240102.csv").write_text(equity_v3_csv("TESTSYM", "2024-01-02"))
    (env.equity_dir / "cm21Jun2021bhav.csv").write_text(
        equity_pre2023_csv("LEGACYSYM", "21-Jun-2021")
    )

    env.load_equity()

    rows = env.read(
        "SELECT symbol, quality, decision_eligible FROM decision_eligible_ohlcv ORDER BY symbol"
    )
    assert rows == [
        ("LEGACYSYM", "OBSERVED_VERIFIED", True),
        ("TESTSYM", "OBSERVED_VERIFIED", True),
    ]


def test_full_loader_rerun_is_idempotent(env: ReingestEnv) -> None:
    (env.equity_dir / "20240102.csv").write_text(equity_v3_csv("TESTSYM", "2024-01-02"))
    (env.fo_dir / "20240102.csv").write_text(fo_v3_csv("TESTSYM", "2024-01-02", "2024-01-25"))

    env.load_equity()
    env.load_fo()
    before_ohlcv = env.read("SELECT count(*) FROM ohlcv")[0][0]
    before_fo = env.read("SELECT count(*) FROM fo_eod")[0][0]
    before_manifests = env.read("SELECT count(*) FROM ingestion_manifests")[0][0]

    eq_rerun = env.load_equity()
    fo_rerun = env.load_fo()

    assert eq_rerun["rows"] == 0
    assert fo_rerun["rows"] == 0
    assert env.read("SELECT count(*) FROM ohlcv")[0][0] == before_ohlcv
    assert env.read("SELECT count(*) FROM fo_eod")[0][0] == before_fo
    assert env.read("SELECT count(*) FROM ingestion_manifests")[0][0] == before_manifests


def test_raw_source_file_is_never_deleted_or_modified(env: ReingestEnv) -> None:
    good_path = env.equity_dir / "20240102.csv"
    good_path.write_text(equity_v3_csv("TESTSYM", "2024-01-02"))
    bad_path = env.equity_dir / "20240103.csv"
    bad_path.write_text("GARBAGE,COLUMNS\nx,y\n")

    good_before = good_path.read_bytes()
    good_mtime_before = good_path.stat().st_mtime_ns
    bad_before = bad_path.read_bytes()
    bad_mtime_before = bad_path.stat().st_mtime_ns

    env.load_equity()

    assert good_path.exists() and good_path.read_bytes() == good_before
    assert good_path.stat().st_mtime_ns == good_mtime_before
    assert bad_path.exists() and bad_path.read_bytes() == bad_before
    assert bad_path.stat().st_mtime_ns == bad_mtime_before


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("20260825", _dt.date(2026, 8, 25)),
        ("cm31Mar2025bhav", _dt.date(2025, 3, 31)),
        ("cm08Aug2022bhav", _dt.date(2022, 8, 8)),
        ("cm15Nov2024bhav", _dt.date(2024, 11, 15)),
        ("fo01Jan2024bhav", _dt.date(2024, 1, 1)),
        ("not_a_date_at_all", None),
        ("BhavCopy_NSE_CM_0_0_0_20260825_F_0000", None),
    ],
)
def test_parse_stem_date_resolves_legacy_and_current_filenames(
    stem: str, expected: _dt.date | None
) -> None:
    """Reproduces a live defect: legacy locally-archived files named
    cmDDMonYYYYbhav.csv (1,343 of them in the current equity archive) never
    matched the digit-prefixed YYYYMMDD pattern this used to require, so
    every one of them was (a) never date-filterable -- rescanned and
    re-hashed on every single load call regardless of from_date -- and (b)
    written to ingestion_manifests, if it got that far, under a shared
    literal "UNKNOWN" business-date key that collided with every other
    unparseable file. A same-key retry after a prior failed write hit a real
    DuckDB-internal fatal index error live during this pass."""
    assert _parse_stem_date(stem) == expected
