from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from data.market_data_quality import DECISION_ELIGIBLE_OHLCV_VIEW_SQL
from db import duckdb_client
from db.write_service import WriteOperation, WriterService, queue_path_for, submit_write
from signals.backfill import MINIMUM_LOOKBACK_ROWS, _write_signal_rows


def _seed_nifty_ohlcv(database: Path, rows: int) -> None:
    import duckdb

    con = duckdb.connect(str(database))
    try:
        con.execute(
            "CREATE TABLE ohlcv(symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
            "low DOUBLE, close DOUBLE, volume BIGINT, source VARCHAR)"
        )
        con.execute(DECISION_ELIGIBLE_OHLCV_VIEW_SQL)
        con.execute(
            """
            CREATE TABLE signals_history (
                date        DATE    NOT NULL,
                signal_name TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                value       DOUBLE,
                PRIMARY KEY (date, signal_name, symbol)
            )
            """
        )
        start = dt.date(2020, 1, 1)
        values = []
        price = 100.0
        for i in range(rows):
            day = start + dt.timedelta(days=i)
            price += 0.1
            values.append(
                (
                    "NIFTY 50",
                    day.isoformat(),
                    price,
                    price + 1,
                    price - 1,
                    price + 0.5,
                    1000,
                    "NSE_INDEX_BHAVCOPY",
                )
            )
        con.executemany("INSERT INTO ohlcv VALUES (?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?)", values)
    finally:
        con.close()


@pytest.mark.integration
def test_signal_backfill_first_run_writes_then_second_run_is_idempotent(
    monkeypatch, tmp_path: Path
) -> None:
    database = tmp_path / "terminal.db"
    queue = queue_path_for(database)
    monkeypatch.setattr(duckdb_client, "DB_PATH", database)
    _seed_nifty_ohlcv(database, MINIMUM_LOOKBACK_ROWS + 50)

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        first = submit_write(WriteOperation.BACKFILL_SIGNALS, {"symbol": "NIFTY"}, database=database, queue_path=queue)
        assert first["rows_written"] > 0
        assert first["dates_written"] == 51  # 250 seeded rows - 200 lookback + 1

        second = submit_write(WriteOperation.BACKFILL_SIGNALS, {"symbol": "NIFTY"}, database=database, queue_path=queue)
        assert second["rows_written"] == 0
        assert second["dates_written"] == 0
        assert second["dates_skipped_existing"] == 51
    finally:
        service.stop()

    import duckdb

    con = duckdb.connect(str(database))
    try:
        total = con.execute("SELECT count(*) FROM signals_history").fetchone()[0]
        distinct = con.execute(
            "SELECT count(*) FROM (SELECT DISTINCT date, signal_name, symbol FROM signals_history)"
        ).fetchone()[0]
        assert total == distinct, "duplicate business keys after idempotent rerun"
    finally:
        con.close()


@pytest.mark.integration
def test_signal_backfill_never_overwrites_an_existing_row(monkeypatch, tmp_path: Path) -> None:
    """Conflict behavior must be preserved by the set-based rewrite: an
    existing (date, signal_name, symbol) row is left untouched, never
    replaced, exactly like the old ON CONFLICT DO NOTHING semantics."""
    database = tmp_path / "terminal.db"
    queue = queue_path_for(database)
    monkeypatch.setattr(duckdb_client, "DB_PATH", database)
    _seed_nifty_ohlcv(database, MINIMUM_LOOKBACK_ROWS + 10)

    service = WriterService(database, queue_path=queue)
    service.start()
    submit_write(WriteOperation.BACKFILL_SIGNALS, {"symbol": "NIFTY"}, database=database, queue_path=queue)
    service.stop()

    import duckdb

    raw = duckdb.connect(str(database))
    try:
        existing = raw.execute(
            "SELECT date, signal_name, symbol FROM signals_history LIMIT 1"
        ).fetchone()
        raw.execute(
            "UPDATE signals_history SET value = -999999 "
            "WHERE date = ? AND signal_name = ? AND symbol = ?",
            list(existing),
        )
    finally:
        raw.close()

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        submit_write(WriteOperation.BACKFILL_SIGNALS, {"symbol": "NIFTY"}, database=database, queue_path=queue)
    finally:
        service.stop()

    raw = duckdb.connect(str(database))
    try:
        value = raw.execute(
            "SELECT value FROM signals_history WHERE date = ? AND signal_name = ? AND symbol = ?",
            list(existing),
        ).fetchone()[0]
    finally:
        raw.close()
    assert value == -999999, "existing row was overwritten by the anti-join insert"


def test_write_signal_rows_scales_against_a_large_pre_existing_table() -> None:
    """Permanent scalability regression: proves the set-based/bulk
    architecture, not a runtime microbenchmark against production-scale
    (3.2M-row) data -- that lives in the disposable one-off benchmark used to
    diagnose and fix the original defect, not in CI. Here a moderately large
    pre-existing table (200k rows, disjoint business keys) plus a realistic
    candidate batch (2,000 dates x 15 signals = 30k rows) must insert well
    under the ceiling a row-by-row ON CONFLICT DO NOTHING re-check against a
    large target would blow through (reproduced separately: even 10k
    candidate rows against a 3.2M-row target exceeded a 90s ceiling). The
    ceiling here is generous (10s vs. an observed ~0.2-1s) specifically to
    avoid a flaky microbenchmark while still catching a regression back to
    per-row conflict checking.
    """
    import time

    import duckdb

    from db import duckdb_client as _duckdb_client

    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE signals_history (
            date DATE NOT NULL, signal_name TEXT NOT NULL, symbol TEXT NOT NULL,
            value DOUBLE, PRIMARY KEY (date, signal_name, symbol)
        )
        """
    )
    # 200,000 disjoint pre-existing rows (different symbol universe than the
    # NIFTY candidate batch below).
    con.execute(
        """
        INSERT INTO signals_history
        SELECT DATE '2000-01-01' + CAST(i % 2000 AS INTEGER),
               'sig_' || CAST((i // 2000) % 20 AS VARCHAR),
               'STOCK_' || CAST(i // (2000 * 20) AS VARCHAR),
               random()
        FROM range(200000) t(i)
        """
    )

    base_date = dt.date(2015, 1, 1)
    rows = [
        ((base_date + dt.timedelta(days=d)).isoformat(), f"price_action.sig_{s}", "NIFTY", float(d + s))
        for d in range(2000)
        for s in range(15)
    ]

    token = _duckdb_client._ACTIVE_WRITER_CONNECTION.set(con)
    try:
        start = time.perf_counter()
        written = _write_signal_rows(rows)
        elapsed = time.perf_counter() - start
    finally:
        _duckdb_client._ACTIVE_WRITER_CONNECTION.reset(token)

    assert written == len(rows)
    assert elapsed < 10.0, f"set-based insert took {elapsed:.2f}s -- possible regression to row-by-row conflict checking"

    total = con.execute("SELECT count(*) FROM signals_history").fetchone()[0]
    distinct = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT date, signal_name, symbol FROM signals_history)"
    ).fetchone()[0]
    assert total == distinct


def test_write_signal_rows_deduplicates_within_a_single_batch() -> None:
    """The set-based INSERT has no per-row ON CONFLICT to fall back on, so a
    batch containing a duplicate business key (defensive case; the real
    caller already de-duplicates by construction) must not raise a PRIMARY
    KEY violation -- it must resolve to one row."""
    import duckdb

    from db import duckdb_client

    tmp_db = duckdb.connect(":memory:")
    tmp_db.execute(
        """
        CREATE TABLE signals_history (
            date DATE NOT NULL, signal_name TEXT NOT NULL, symbol TEXT NOT NULL,
            value DOUBLE, PRIMARY KEY (date, signal_name, symbol)
        )
        """
    )
    token = duckdb_client._ACTIVE_WRITER_CONNECTION.set(tmp_db)
    try:
        rows = [
            ("2024-01-01", "sig", "NIFTY", 1.0),
            ("2024-01-01", "sig", "NIFTY", 2.0),  # duplicate key, later value wins
        ]
        written = _write_signal_rows(rows)
        assert written == 1
        result = tmp_db.execute("SELECT value FROM signals_history").fetchall()
        assert result == [(2.0,)]
    finally:
        duckdb_client._ACTIVE_WRITER_CONNECTION.reset(token)


@pytest.mark.integration
def test_banknifty_signal_backfill_resolves_through_the_canonical_alias_table(
    monkeypatch, tmp_path: Path
) -> None:
    """Reproduces a live defect: signals.backfill._load_symbol_ohlcv
    hand-duplicated its own partial index candidate table -- only
    NIFTY <-> "NIFTY 50", never BANKNIFTY <-> "NIFTY BANK" -- the same bug
    class as ml/stock_trainer.py's _load_ohlcv. Verified actual OHLCV for the
    index is stored under "NIFTY BANK", so per-symbol signal backfill (called
    from ml.stock_trainer.StockModelTrainer.train_symbol for every symbol,
    including BANKNIFTY) silently saw 0 OHLCV rows and wrote nothing."""
    import duckdb

    database = tmp_path / "terminal.db"
    queue = queue_path_for(database)
    monkeypatch.setattr(duckdb_client, "DB_PATH", database)

    con = duckdb.connect(str(database))
    try:
        con.execute(
            "CREATE TABLE ohlcv(symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
            "low DOUBLE, close DOUBLE, volume BIGINT, source VARCHAR)"
        )
        con.execute(DECISION_ELIGIBLE_OHLCV_VIEW_SQL)
        con.execute(
            """
            CREATE TABLE signals_history (
                date        DATE    NOT NULL,
                signal_name TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                value       DOUBLE,
                PRIMARY KEY (date, signal_name, symbol)
            )
            """
        )
        start = dt.date(2020, 1, 1)
        values = []
        price = 100.0
        for i in range(MINIMUM_LOOKBACK_ROWS + 50):
            day = start + dt.timedelta(days=i)
            price += 0.1
            values.append(
                ("NIFTY BANK", day.isoformat(), price, price + 1, price - 1, price + 0.5, 1000, "NSE_INDEX_BHAVCOPY")
            )
        con.executemany("INSERT INTO ohlcv VALUES (?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?)", values)
    finally:
        con.close()

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        result = submit_write(
            WriteOperation.BACKFILL_SIGNALS, {"symbol": "BANKNIFTY"}, database=database, queue_path=queue
        )
        assert result["rows_written"] > 0, "BANKNIFTY signal backfill must resolve via the 'NIFTY BANK' verified rows"
        assert result["source_symbol"] == "NIFTY BANK"
    finally:
        service.stop()
