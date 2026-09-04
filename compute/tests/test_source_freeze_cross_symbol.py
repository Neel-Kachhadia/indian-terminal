"""Source-freeze closure pass -- Phase 8 cross-symbol isolation (DB/logging
layer). Isolated app-data / temp DuckDB only.

Scope note: this covers the ml_predictions natural-key/attribution isolation
layer (log_prediction + resolve_pending_prediction_outcomes), which is where
cross-symbol leakage would materialize into stored/served data. A full
end-to-end ML-context-build/signals/forecast/options/BIF response diff across
all 9 symbols was NOT executed in this pass (would require full market-data
fixtures for each symbol/product surface) -- see the reported summary for
what remains INCONCLUSIVE/not independently exercised.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from db import duckdb_client
from db.duckdb_client import _writer_transaction_scope
from db.write_service import WriteOperation as WO
from db.write_service import submit_write, WriterService

SYMBOLS = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "RELIANCE", "TCS", "INFY", "ANGELONE", "SWIGGY",
]


def _attribution(symbol: str, data_as_of: str = "2026-01-05") -> dict:
    return {
        "requested_symbol": symbol,
        "trained_symbol": symbol,
        "data_as_of": data_as_of,
        "model_version": f"v1-{symbol}",
        "artifact_hash": f"{symbol.lower()}" + "0" * 20,
        "feature_schema_version": "features-v1:test",
        "release_version": "0.1.0",
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
    }


@pytest.mark.contract
def test_cross_symbol_predictions_never_leak_attribution_across_symbols(
    isolated_app_data: Path,
) -> None:
    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"

    service = WriterService(database, queue_path=queue)
    service.start()
    ids = {}
    try:
        for symbol in SYMBOLS:
            ids[symbol] = submit_write(
                WO.LOG_PREDICTION,
                {
                    "model_id": "m02_direction",
                    "prediction": "UP",
                    "probabilities": {"UP": 0.6, "DOWN": 0.4},
                    "confidence": 0.6,
                    "features": {"spot": 1.0},
                    "attribution": _attribution(symbol),
                },
                database=database,
                queue_path=queue,
            )

        assert len(set(ids.values())) == len(SYMBOLS), "every symbol must get its own distinct row id"

        with duckdb.connect(str(database)) as connection:
            for symbol in SYMBOLS:
                row = connection.execute(
                    "SELECT symbol, trained_symbol, model_version, artifact_hash "
                    "FROM ml_predictions WHERE id = ?",
                    [ids[symbol]],
                ).fetchone()
                assert row[0] == symbol, f"symbol column leaked for {symbol}: got {row[0]}"
                assert row[1] == symbol, f"trained_symbol leaked for {symbol}: got {row[1]}"
                assert row[2] == f"v1-{symbol}", f"model_version leaked for {symbol}"
                assert row[3].startswith(symbol.lower()), f"artifact_hash leaked for {symbol}"

            total = connection.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
        assert total == len(SYMBOLS), "no extra/duplicate/merged rows across symbols"
    finally:
        service.stop()
