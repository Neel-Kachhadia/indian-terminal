"""Source-freeze closure pass -- Phase 2 OOS failure matrix (case_id-tagged).

Isolated app-data / temp DuckDB only. Never touches production DuckDB.
Cases not covered here are covered by pre-existing tests (cited in
oos_failure_matrix.csv) or are NOT_REACHABLE_BY_CONTRACT (also cited there).
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from db import duckdb_client
from db.duckdb_client import _writer_transaction_scope
from db.write_service import WriteOperation, WriterService, submit_write
from db.writer_lease import release_writer_lease, require_writer_lease


def _attribution(symbol: str = "TESTSTK", data_as_of: str = "2026-01-05", **overrides) -> dict:
    base = {
        "requested_symbol": symbol,
        "trained_symbol": symbol,
        "data_as_of": data_as_of,
        "model_version": "v1",
        "artifact_hash": "a" * 64,
        "feature_schema_version": "features-v1:test",
        "release_version": "0.1.0",
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
    }
    base.update(overrides)
    return base


@pytest.mark.contract
def test_case16_m01_vol_regime_resolves_prematurely_before_5day_maturity(
    isolated_app_data: Path,
) -> None:
    """CASE 16 (resolver before target maturity) for m01_vol_regime.

    monitor._RESOLVABLE_TARGETS declares m01_vol_regime's horizon as 5
    trading days. vol_regime_target's realised-vol window is
    `future.std(axis=1, ddof=0)` over 5 forward-shifted return columns.
    pandas .std(skipna=True) computes over however many of those 5 columns
    are non-NaN -- it does NOT wait for all 5. With only 1 future row
    available (T+1), the row is [return_T1, NaN, NaN, NaN, NaN]; std of a
    single value with ddof=0 is 0.0, not NaN -- so realised=0.0 and the
    label resolves to "LOW" one trading day after the prediction, four
    trading days before the documented 5-day maturity.

    This FAILS the OOS contract: resolve_pending_prediction_outcomes must
    not resolve a row before its target has matured.
    """
    from db.write_service import WriteOperation as WO

    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"

    # 6 trading days: prediction anchored at index 4 (2026-01-12), with only
    # ONE future row (index 5, 2026-01-13) actually present -- 4 short of
    # the 5-day maturity horizon.
    dates = [d.date().isoformat() for d in pd.bdate_range("2026-01-05", periods=6)]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 101.0]

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                for day, close in zip(dates, closes):
                    connection.execute(
                        "INSERT INTO ohlcv VALUES ('TESTSTK', CAST(? AS DATE), ?, ?, ?, ?, 1000, 'NSE_EQUITY_BHAVCOPY')",
                        [day, close, close + 1, close - 1, close],
                    )

        anchor_day = dates[4]  # only dates[5] exists after it: 1 of 5 needed
        prediction_id = submit_write(
            WO.LOG_PREDICTION,
            {
                "model_id": "m01_vol_regime",
                "prediction": "MEDIUM",
                "probabilities": {"LOW": 0.3, "MEDIUM": 0.5, "HIGH": 0.2},
                "confidence": 0.5,
                "features": {"spot": 1.0},
                "attribution": _attribution(data_as_of=anchor_day),
            },
            database=database,
            queue_path=queue,
        )

        summary = submit_write(
            WO.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m01_vol_regime"]},
            database=database,
            queue_path=queue,
        )

        with duckdb.connect(str(database)) as connection:
            row = connection.execute(
                "SELECT outcome FROM ml_predictions WHERE id = ?", [prediction_id]
            ).fetchone()

        # EXPECTED (contract): outcome IS NULL, not_yet_resolvable == 1
        # ACTUAL (defect): outcome gets set to "LOW" one day after prediction
        if row[0] is not None:
            pytest.fail(
                f"CASE16-FAIL: m01_vol_regime resolved with outcome={row[0]!r} "
                f"after only 1/5 required future trading days (premature "
                f"resolution using a partial-window std, see "
                f"compute/ml/feature_engineering.py vol_regime_target). "
                f"summary={summary}"
            )
    finally:
        service.stop()


@pytest.mark.contract
def test_case17_m01_vol_regime_resolves_correctly_at_full_5day_maturity(
    isolated_app_data: Path,
) -> None:
    """CASE 17 (resolver exactly at maturity) -- control case: with the full
    5 future trading days present, resolution must succeed with a label
    computed over the full window (contrast with case 16)."""
    from db.write_service import WriteOperation as WO

    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"

    dates = [d.date().isoformat() for d in pd.bdate_range("2026-01-05", periods=6)]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 130.0]  # big final move -> HIGH vol

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                for day, close in zip(dates, closes):
                    connection.execute(
                        "INSERT INTO ohlcv VALUES ('TESTSTK', CAST(? AS DATE), ?, ?, ?, ?, 1000, 'NSE_EQUITY_BHAVCOPY')",
                        [day, close, close + 1, close - 1, close],
                    )

        anchor_day = dates[0]  # 5 future rows exist: dates[1..5]
        prediction_id = submit_write(
            WO.LOG_PREDICTION,
            {
                "model_id": "m01_vol_regime",
                "prediction": "HIGH",
                "probabilities": {"LOW": 0.1, "MEDIUM": 0.2, "HIGH": 0.7},
                "confidence": 0.7,
                "features": {"spot": 1.0},
                "attribution": _attribution(data_as_of=anchor_day),
            },
            database=database,
            queue_path=queue,
        )
        submit_write(
            WO.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m01_vol_regime"]},
            database=database,
            queue_path=queue,
        )
        with duckdb.connect(str(database)) as connection:
            row = connection.execute(
                "SELECT outcome FROM ml_predictions WHERE id = ?", [prediction_id]
            ).fetchone()
        assert row[0] == "HIGH"
    finally:
        service.stop()


@pytest.mark.contract
def test_case6_7_8_non_finite_and_out_of_range_probability_rejected(
    isolated_app_data: Path,
) -> None:
    """CASE 6/7/8: non-finite / <0 / >1 confidence must raise, not log."""
    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    require_writer_lease(database)
    try:
        from ml.monitor import log_prediction

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                for bad_confidence in (float("nan"), float("inf"), -0.1, 1.1):
                    with pytest.raises(ValueError):
                        log_prediction(
                            "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.4}, bad_confidence,
                            {"f1": 1.0}, attribution=_attribution(),
                        )
        with duckdb.connect(str(database), read_only=True) as connection:
            count = connection.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
        assert count == 0, "no row may be logged for any non-finite/out-of-range confidence"
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_case9_empty_probabilities_rejected(isolated_app_data: Path) -> None:
    """CASE 9: empty probability output must raise, not log with {}."""
    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    require_writer_lease(database)
    try:
        from ml.monitor import log_prediction

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                with pytest.raises(ValueError):
                    log_prediction(
                        "m02_direction", "UP", {}, 0.6,
                        {"f1": 1.0}, attribution=_attribution(),
                    )
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_case25_already_resolved_row_is_not_touched_again(isolated_app_data: Path) -> None:
    """CASE 25: re-running the resolver on an already-resolved row must be a
    no-op (covered end-to-end via idempotency, but assert immutable columns
    explicitly stay byte-identical across the second run)."""
    from db.write_service import WriteOperation as WO

    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"

    dates = [d.date().isoformat() for d in pd.bdate_range("2026-01-05", periods=3)]
    closes = [100.0, 101.0, 100.0]

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                for day, close in zip(dates, closes):
                    connection.execute(
                        "INSERT INTO ohlcv VALUES ('TESTSTK', CAST(? AS DATE), ?, ?, ?, ?, 1000, 'NSE_EQUITY_BHAVCOPY')",
                        [day, close, close + 1, close - 1, close],
                    )
        prediction_id = submit_write(
            WO.LOG_PREDICTION,
            {
                "model_id": "m02_direction",
                "prediction": "UP",
                "probabilities": {"UP": 0.6, "DOWN": 0.4},
                "confidence": 0.6,
                "features": {"spot": 1.0},
                "attribution": _attribution(data_as_of=dates[0]),
            },
            database=database,
            queue_path=queue,
        )
        submit_write(WO.RESOLVE_PENDING_PREDICTION_OUTCOMES, {"model_ids": ["m02_direction"]},
                      database=database, queue_path=queue)
        with duckdb.connect(str(database)) as connection:
            first = connection.execute(
                "SELECT prediction, outcome, outcome_at, correct, data_as_of FROM ml_predictions WHERE id=?",
                [prediction_id],
            ).fetchone()
        second_summary = submit_write(WO.RESOLVE_PENDING_PREDICTION_OUTCOMES, {"model_ids": ["m02_direction"]},
                                       database=database, queue_path=queue)
        with duckdb.connect(str(database)) as connection:
            second = connection.execute(
                "SELECT prediction, outcome, outcome_at, correct, data_as_of FROM ml_predictions WHERE id=?",
                [prediction_id],
            ).fetchone()
        assert first == second, "immutable columns + outcome must not change on re-resolution"
        assert second_summary["resolved"] == 0
        assert second_summary["already_resolved_skipped"] == 0  # row no longer a candidate at all
    finally:
        service.stop()
