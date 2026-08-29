from __future__ import annotations

import asyncio
import json
import time

import pytest

from live.multi_feed import (
    FEED_CONNECTION_OWNER,
    FEED_WRITER_OWNER,
    FeedPhase,
    _backoff_delay,
    get_status,
    run,
    valid_transition,
)


def test_feed_state_rejects_false_live_transition() -> None:
    assert not valid_transition(FeedPhase.STARTING_BACKEND, FeedPhase.LIVE)


def test_feed_state_accepts_reconnect_cycle() -> None:
    phases = [
        FeedPhase.STARTING_BACKEND,
        FeedPhase.WAITING_FOR_HEALTH,
        FeedPhase.AUTHENTICATED,
        FeedPhase.CONNECTING_FEED,
        FeedPhase.SUBSCRIBING,
        FeedPhase.WAITING_FOR_DATA,
        FeedPhase.LIVE,
        FeedPhase.RECONNECTING,
        FeedPhase.WAITING_FOR_HEALTH,
    ]

    assert all(valid_transition(current, next_phase) for current, next_phase in zip(phases, phases[1:]))


def test_backoff_is_bounded_and_deterministic_without_jitter() -> None:
    assert _backoff_delay(1, 0.0) == 1.0
    assert _backoff_delay(120, 0.0) == 30.0


class FakeRedis:
    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload

    def get(self, key: str) -> str | None:
        assert key == "health:angel_stream"
        return json.dumps(self.payload) if self.payload is not None else None


def test_python_is_consumer_only_and_reports_rust_owner() -> None:
    status = get_status(
        FakeRedis(
            {
                "state": "LIVE",
                "source": FEED_WRITER_OWNER,
                "last_success_ms": int(time.time() * 1000),
            }
        )
    )
    assert status["connected"] is True
    assert status["connection_owner"] == "PYTHON_SECURE_FEED_BRIDGE"
    assert status["writer_owner"] == "RUST_TAURI_INGESTER"


def test_standalone_feed_connection_path_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="backend lifecycle"):
        asyncio.run(run())
