from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pytest
import duckdb

from db import duckdb_client
from operations import health as health_module
from operations.health import health_snapshot
from operations.logging_config import JsonFormatter, MAX_LOG_BYTES, LOG_BACKUPS
from security.log_scrubber import SecretScrubber


class FakeRedis:
    def __init__(self, *, available: bool = True, values: dict[str, str] | None = None) -> None:
        self.available = available
        self.values = values or {}

    def ping(self) -> bool:
        if not self.available:
            raise ConnectionError("offline")
        return True

    def get(self, key: str):
        if not self.available:
            raise ConnectionError("offline")
        return self.values.get(key)

    def ttl(self, key: str) -> int:
        return 30 if key in self.values else -2


def _database(root: Path) -> Path:
    path = root / "data" / "terminal.db"
    duckdb_client.DB_PATH = path
    duckdb_client.init_schema()
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            "INSERT INTO ohlcv VALUES ('NIFTY', '2026-08-21', 1, 2, 1, 2, 100, 'TEST')"
        )
        connection.execute(
            """
            INSERT INTO fo_eod
            (symbol, expiry, strike, option_type, date, open, high, low, close,
             volume, oi, oi_change, source, source_format, price_quality)
            VALUES ('NIFTY', '2026-08-25', 25000, 'CE', '2026-08-21',
                    1, 2, 1, 2, 100, 100, 1, 'TEST', 'fixture', 'EXECUTABLE')
            """
        )
        connection.execute(
            "INSERT INTO iv_history VALUES ('NIFTY', '2026-08-25', 25000, 'CE', '2026-08-21', 12)"
        )
        connection.execute(
            "INSERT INTO signals_history VALUES ('2026-08-21', 'test', 'NIFTY', 1)"
        )
    return path


@pytest.mark.contract
def test_health_separates_liveness_readiness_and_degradation(
    isolated_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(isolated_app_data)
    monkeypatch.setattr(
        health_module,
        "registry_health",
        lambda: {
            "state": "EMPTY",
            "validated_models": 0,
            "corrupt_models": 0,
            "latest_training_timestamp": None,
            "reason": "model registry not created",
        },
    )
    now = dt.datetime(2026, 8, 23, 6, 0, tzinfo=dt.timezone.utc)
    snapshot = health_snapshot(
        database_path=database,
        collector_db=isolated_app_data / "data" / "collector_log.db",
        redis_client=FakeRedis(),
        feed_status=lambda: {"state": "DEGRADED", "connected": False},
        release={"release_version": "1.0.0", "build_status": "COHERENT"},
        integrity={"status": "TRUSTED", "trusted": True},
        external_success_times={},
        now=now,
    )
    assert snapshot["liveness"]["status"] == "ALIVE"
    assert snapshot["readiness"] == {
        "status": "READY",
        "ready": True,
        "blocking_components": [],
    }
    assert snapshot["status"] == "DEGRADED"
    assert snapshot["components"]["angel_stream"]["state"] == "SUSPENDED"
    assert snapshot["components"]["data_freshness"]["state"] == "HEALTHY"
    assert "collector" in snapshot["degraded_state"]["components"]


@pytest.mark.recovery
def test_redis_failure_blocks_readiness_without_false_green(
    isolated_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(isolated_app_data)
    monkeypatch.setattr(health_module, "registry_health", lambda: {"state": "HEALTHY"})
    snapshot = health_snapshot(
        database_path=database,
        collector_db=isolated_app_data / "missing.db",
        redis_client=FakeRedis(available=False),
        feed_status=lambda: {"state": "LIVE", "connected": True, "symbols_active": 1},
        release={"release_version": "1.0.0", "build_status": "COHERENT"},
        integrity={"status": "TRUSTED", "trusted": True},
        external_success_times={},
        now=dt.datetime(2026, 8, 21, 4, 30, tzinfo=dt.timezone.utc),
    )
    assert snapshot["liveness"]["status"] == "ALIVE"
    assert snapshot["readiness"]["ready"] is False
    assert "redis" in snapshot["readiness"]["blocking_components"]
    assert snapshot["status"] == "NOT_READY"


def test_redis_ping_cache_does_not_leak_stale_result_across_client_identity(
    isolated_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dict keyed by id(client) (the pre-fix implementation) can hand a
    brand-new client's health check a stale "healthy" result left behind by
    a different, already garbage-collected client that happened to reuse the
    same id() -- CPython is free to recycle a freed object's id immediately.
    The WeakKeyDictionary fix must never do this: its entry for a client is
    tied to that exact object and is evicted the instant the object is
    collected, so a colliding id can never inherit a stale cache entry.
    """
    import gc

    database = _database(isolated_app_data)
    monkeypatch.setattr(health_module, "registry_health", lambda: {"state": "HEALTHY"})

    def snapshot_for(client: FakeRedis) -> dict:
        return health_snapshot(
            database_path=database,
            collector_db=isolated_app_data / "missing.db",
            redis_client=client,
            feed_status=lambda: {"state": "LIVE", "connected": True, "symbols_active": 1},
            release={"release_version": "1.0.0", "build_status": "COHERENT"},
            integrity={"status": "TRUSTED", "trusted": True},
            external_success_times={},
            now=dt.datetime(2026, 8, 21, 4, 30, tzinfo=dt.timezone.utc),
        )

    healthy_client = FakeRedis(available=True)
    snapshot_for(healthy_client)  # populates the cache with ok=True for this object
    assert len(health_module._redis_ping_cache) == 1

    del healthy_client
    gc.collect()
    # Mechanism-level proof: the WeakKeyDictionary must have dropped the
    # entry the instant its key was collected -- a plain dict keyed by
    # id(client) would still hold a "healthy" entry under `healthy_id` here.
    assert len(health_module._redis_ping_cache) == 0

    unavailable_client = FakeRedis(available=False)
    snapshot = snapshot_for(unavailable_client)
    assert snapshot["readiness"]["ready"] is False
    assert "redis" in snapshot["readiness"]["blocking_components"]
    assert snapshot["components"]["redis"]["state"] == "UNAVAILABLE"


@pytest.mark.recovery
def test_database_unavailable_blocks_readiness_without_creating_it(
    isolated_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = isolated_app_data / "data" / "missing.db"
    monkeypatch.setattr(health_module, "registry_health", lambda: {"state": "HEALTHY"})
    snapshot = health_snapshot(
        database_path=missing,
        collector_db=isolated_app_data / "missing-collector.db",
        redis_client=FakeRedis(),
        feed_status=lambda: {"state": "DEGRADED", "connected": False},
        release={"release_version": "1.0.0", "build_status": "COHERENT"},
        integrity={"status": "TRUSTED", "trusted": True},
        external_success_times={},
        now=dt.datetime(2026, 8, 21, 4, 30, tzinfo=dt.timezone.utc),
    )
    assert snapshot["components"]["duckdb_writer"]["state"] == "UNAVAILABLE"
    assert snapshot["readiness"]["ready"] is False
    assert "duckdb_writer" in snapshot["readiness"]["blocking_components"]
    assert not missing.exists()


@pytest.mark.contract
def test_structured_log_is_timestamped_bounded_and_scrubbed() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="api_key=super-secret-value",
        args=(),
        exc_info=None,
    )
    assert SecretScrubber().filter(record) is True
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "ERROR"
    assert payload["logger"] == "test"
    assert payload["timestamp"].endswith("+00:00")
    assert "super-secret-value" not in payload["event"]
    assert MAX_LOG_BYTES == 5 * 1024 * 1024
    assert LOG_BACKUPS == 5
