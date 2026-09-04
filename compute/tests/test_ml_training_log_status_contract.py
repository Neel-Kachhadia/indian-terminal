from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db.write_service import ML_TRAINING_LOG_STATUSES, WriteOperation, WriterService, submit_write


def _payload(**overrides) -> dict:
    base = {
        "symbol": "NIFTY",
        "model_id": "m01_vol_regime",
        "trained_at": "2026-08-30T15:00:00+00:00",
        "n_rows": 250,
        "n_features": 12,
        "train_score": 0.75,
        "val_score": 0.7,
        "feature_importance_top5": "{}",
        "bif_rows_used": 0,
        "ohlcv_rows_used": 250,
        "status": "READY",
        "error_msg": None,
    }
    base.update(overrides)
    return base


def _service(tmp_path: Path):
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    service = WriterService(database, queue_path=queue)
    service.start()
    return service, database, queue


@pytest.mark.parametrize("status", sorted(ML_TRAINING_LOG_STATUSES))
def test_every_real_training_status_is_accepted(tmp_path: Path, status: str) -> None:
    """The exact vocabulary traced live from trainer.py/stock_trainer.py/
    ml.models.common._unavailable_status (2026-09-03) -- not an assumed list."""
    service, database, queue = _service(tmp_path)
    try:
        result = submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG,
            _payload(status=status),
            database=database,
            queue_path=queue,
            timeout=2,
        )
        assert result["status"] == status
    finally:
        service.stop()


@pytest.mark.parametrize(
    "bad_status",
    ["success", "unknown", "READY_STATE", "", "TRAINED", "REJECTED"],
)
def test_illegitimate_status_fails_closed(tmp_path: Path, bad_status: str) -> None:
    """Unknown states must fail closed, not be silently coerced or dropped --
    this is exactly the fix for the discovered defect (old code accepted
    only "success"/"failed", neither of which real code ever emits)."""
    service, database, queue = _service(tmp_path)
    try:
        with pytest.raises(Exception):
            submit_write(
                WriteOperation.RECORD_ML_TRAINING_LOG,
                _payload(status=bad_status),
                database=database,
                queue_path=queue,
                timeout=2,
            )
    finally:
        service.stop()


def test_whitespace_padded_real_status_is_normalized_not_rejected(tmp_path: Path) -> None:
    service, database, queue = _service(tmp_path)
    try:
        result = submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG,
            _payload(status="READY "),
            database=database,
            queue_path=queue,
            timeout=2,
        )
        assert result["status"] == "READY"
    finally:
        service.stop()


def test_mixed_case_of_a_real_status_is_normalized_not_rejected(tmp_path: Path) -> None:
    service, database, queue = _service(tmp_path)
    try:
        result = submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG,
            _payload(status="ready"),
            database=database,
            queue_path=queue,
            timeout=2,
        )
        assert result["status"] == "READY"
    finally:
        service.stop()


def test_none_status_is_rejected(tmp_path: Path) -> None:
    service, database, queue = _service(tmp_path)
    try:
        with pytest.raises(Exception):
            submit_write(
                WriteOperation.RECORD_ML_TRAINING_LOG,
                _payload(status=None),
                database=database,
                queue_path=queue,
                timeout=2,
            )
    finally:
        service.stop()


def test_duplicate_write_creates_two_distinct_log_rows(tmp_path: Path) -> None:
    """Unlike ml_predictions (idempotent on a natural key) and ohlc_forecasts
    (idempotent on symbol+date), ml_training_log is an append-only audit
    trail of every attempt -- two real training runs for the same
    model/symbol/day are two real events and must both be visible."""
    service, database, queue = _service(tmp_path)
    try:
        submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG, _payload(), database=database, queue_path=queue, timeout=2
        )
        submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG, _payload(), database=database, queue_path=queue, timeout=2
        )
    finally:
        service.stop()
    connection = duckdb.connect(database, read_only=True)
    try:
        count = connection.execute("SELECT COUNT(*) FROM ml_training_log").fetchone()[0]
        assert count == 2
    finally:
        connection.close()


def test_writer_restart_does_not_lose_or_duplicate_prior_rows(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG, _payload(), database=database, queue_path=queue, timeout=2
        )
    finally:
        service.stop()

    service2 = WriterService(database, queue_path=queue)
    service2.start()
    try:
        submit_write(
            WriteOperation.RECORD_ML_TRAINING_LOG,
            _payload(model_id="m02_direction"),
            database=database,
            queue_path=queue,
            timeout=2,
        )
    finally:
        service2.stop()

    connection = duckdb.connect(database, read_only=True)
    try:
        rows = connection.execute("SELECT model_id FROM ml_training_log ORDER BY model_id").fetchall()
        assert rows == [("m01_vol_regime",), ("m02_direction",)]
    finally:
        connection.close()
