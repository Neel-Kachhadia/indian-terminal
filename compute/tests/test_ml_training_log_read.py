from __future__ import annotations

import duckdb
import pytest

from db.write_service import ReadOperation, _dispatch_read


def _seed(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        "CREATE TABLE ml_training_log ("
        "symbol TEXT NOT NULL, model_id TEXT NOT NULL, trained_at TIMESTAMP NOT NULL, "
        "n_rows INTEGER, n_features INTEGER, train_score DOUBLE, val_score DOUBLE, "
        "feature_importance_top5 TEXT, bif_rows_used INTEGER, ohlcv_rows_used INTEGER, "
        "status TEXT NOT NULL, error_msg TEXT)"
    )
    connection.execute(
        """
        INSERT INTO ml_training_log VALUES
            ('RELIANCE', 'm02_direction', TIMESTAMP '2026-08-01 00:00:00', 500, 30, 0.6, 0.52, '{}', 100, 500, 'FAILED', 'val_score below floor'),
            ('RELIANCE', 'm02_direction', TIMESTAMP '2026-08-08 00:00:00', 520, 30, 0.61, 0.55, '{}', 100, 520, 'FAILED', 'val_score below floor'),
            ('TCS',      'm01_vol_regime', TIMESTAMP '2026-08-07 00:00:00', 400, 25, 0.7, 0.68, '{}', 90, 400, 'READY', NULL)
        """
    )


def test_returns_bounded_typed_rows_ordered_newest_first() -> None:
    connection = duckdb.connect(":memory:")
    try:
        _seed(connection)
        result = _dispatch_read(ReadOperation.LOAD_ML_TRAINING_LOG.value, {}, connection)
        assert result["columns"][:3] == ["symbol", "model_id", "trained_at"]
        assert len(result["rows"]) == 3
        assert result["rows"][0][0] == "RELIANCE"
        assert result["rows"][0][2].startswith("2026-08-08")
    finally:
        connection.close()


def test_filters_by_model_id_symbol_and_status() -> None:
    connection = duckdb.connect(":memory:")
    try:
        _seed(connection)
        by_model = _dispatch_read(
            ReadOperation.LOAD_ML_TRAINING_LOG.value, {"model_id": "m01_vol_regime"}, connection
        )
        assert len(by_model["rows"]) == 1
        assert by_model["rows"][0][1] == "m01_vol_regime"

        by_symbol = _dispatch_read(
            ReadOperation.LOAD_ML_TRAINING_LOG.value, {"symbol": "reliance"}, connection
        )
        assert len(by_symbol["rows"]) == 2

        by_status = _dispatch_read(
            ReadOperation.LOAD_ML_TRAINING_LOG.value, {"status": "FAILED"}, connection
        )
        assert len(by_status["rows"]) == 2
        assert all(row[7] == "FAILED" for row in by_status["rows"])

        by_status_lowercase_input = _dispatch_read(
            ReadOperation.LOAD_ML_TRAINING_LOG.value, {"status": "failed"}, connection
        )
        assert len(by_status_lowercase_input["rows"]) == 2, "status filter must not be case-sensitive to caller casing"
    finally:
        connection.close()


def test_limit_is_bounded_not_arbitrary_sql() -> None:
    connection = duckdb.connect(":memory:")
    try:
        _seed(connection)
        with pytest.raises(ValueError):
            _dispatch_read(ReadOperation.LOAD_ML_TRAINING_LOG.value, {"limit": 0}, connection)
        with pytest.raises(ValueError):
            _dispatch_read(ReadOperation.LOAD_ML_TRAINING_LOG.value, {"limit": 5001}, connection)
        capped = _dispatch_read(ReadOperation.LOAD_ML_TRAINING_LOG.value, {"limit": 2}, connection)
        assert len(capped["rows"]) == 2


    finally:
        connection.close()


def test_empty_table_returns_empty_rows_not_an_error() -> None:
    connection = duckdb.connect(":memory:")
    try:
        result = _dispatch_read(ReadOperation.LOAD_ML_TRAINING_LOG.value, {}, connection)
        assert result["rows"] == []
    finally:
        connection.close()
