from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db import duckdb_client
from db.duckdb_client import _writer_transaction_scope
from db.writer_lease import release_writer_lease, require_writer_lease


def _attribution(symbol: str = "RELIANCE", data_as_of: str = "2026-09-03", **overrides) -> dict:
    base = {
        "requested_symbol": symbol,
        "trained_symbol": symbol,
        "data_as_of": data_as_of,
        "model_version": "v1",
        "artifact_hash": "abc123",
        "feature_schema_version": "features-v1:deadbeef",
        "release_version": "0.1.0",
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
    }
    base.update(overrides)
    return base


@pytest.mark.contract
def test_relogging_same_model_symbol_data_as_of_never_creates_a_duplicate_row(
    isolated_app_data: Path,
) -> None:
    """Same bug class already found and fixed once in ohlc_forecasts
    (see test_ohlc_forecast_logging_integrity.py): log_prediction previously
    did a bare INSERT with a random uuid id and no natural-key dedup check.
    A scheduled snapshot task re-run for a (model_id, symbol, data_as_of)
    already logged would silently add another full duplicate observation."""
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.monitor import log_prediction

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                id_1 = log_prediction(
                    "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.3, "FLAT": 0.1}, 0.6,
                    {"f1": 1.0}, attribution=_attribution(),
                )
        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                # Same natural key, different prediction payload -- must
                # still be treated as the same observation, not a new one.
                id_2 = log_prediction(
                    "m02_direction", "DOWN", {"UP": 0.2, "DOWN": 0.7, "FLAT": 0.1}, 0.7,
                    {"f1": 2.0}, attribution=_attribution(),
                )
        assert id_1 == id_2

        with duckdb.connect(str(database), read_only=True) as connection:
            rows = connection.execute(
                "SELECT COUNT(*), MAX(prediction), MAX(confidence) FROM ml_predictions "
                "WHERE model_id = 'm02_direction' AND symbol = 'RELIANCE' AND data_as_of = '2026-09-03'"
            ).fetchone()
        assert rows[0] == 1, "a second log for an already-logged (model, symbol, data_as_of) must not duplicate"
        # The FIRST logged observation's payload is retained -- outcome
        # resolution must never be second-guessed by a later re-run.
        assert rows[1] == "UP"
        assert rows[2] == pytest.approx(0.6)
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_different_symbols_and_dates_are_logged_independently(isolated_app_data: Path) -> None:
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.monitor import log_prediction

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                log_prediction(
                    "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.4}, 0.6,
                    {"f1": 1.0}, attribution=_attribution(symbol="RELIANCE"),
                )
                log_prediction(
                    "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.4}, 0.6,
                    {"f1": 1.0}, attribution=_attribution(symbol="TCS"),
                )
                log_prediction(
                    "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.4}, 0.6,
                    {"f1": 1.0}, attribution=_attribution(symbol="RELIANCE", data_as_of="2026-09-04"),
                )
        with duckdb.connect(str(database), read_only=True) as connection:
            count = connection.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
        assert count == 3
    finally:
        release_writer_lease()


@pytest.mark.contract
def test_missing_natural_key_attribution_refuses_rather_than_logs_unattributed(
    isolated_app_data: Path,
) -> None:
    database = isolated_app_data / "data" / "terminal.db"
    duckdb_client.DB_PATH = database
    duckdb_client.init_schema()
    require_writer_lease(database)
    try:
        from ml.monitor import log_prediction

        with duckdb.connect(str(database)) as connection:
            with _writer_transaction_scope(connection):
                with pytest.raises(ValueError):
                    log_prediction(
                        "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.4}, 0.6,
                        {"f1": 1.0}, attribution=_attribution(symbol=""),
                    )
                with pytest.raises(ValueError):
                    log_prediction(
                        "m02_direction", "UP", {"UP": 0.6, "DOWN": 0.4}, 0.6,
                        {"f1": 1.0}, attribution=_attribution(data_as_of=""),
                    )
        with duckdb.connect(str(database), read_only=True) as connection:
            count = connection.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
        assert count == 0
    finally:
        release_writer_lease()
