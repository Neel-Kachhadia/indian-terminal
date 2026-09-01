from __future__ import annotations

import asyncio
import time

import pytest

from db.write_service import WriteOperation


def test_stalled_writer_does_not_block_past_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduced live: the FastAPI lifespan previously called submit_write()
    (a synchronous polling loop with its own internal 120s timeout) directly
    inside the async lifespan coroutine, once with no exception handling at
    all. A stalled or unreachable writer authority (the collector, in
    remote-client topology) therefore either blocked the whole process for up
    to two minutes, or crashed startup outright with an uncaught exception --
    in both cases keeping 127.0.0.1:8765 from ever becoming ready.
    main._startup_submit_write() must return control within its `budget`
    regardless of how long the underlying writer call actually takes, and the
    event loop must remain free to run other coroutines while it waits."""
    import main

    def stalled_submit_write(operation, payload, **_kwargs):
        time.sleep(5)
        return {"recovered": 0}

    monkeypatch.setattr(main, "submit_write", stalled_submit_write)

    async def scenario() -> tuple[float, int]:
        heartbeat_ticks: list[float] = []
        stop = asyncio.Event()

        async def heartbeat() -> None:
            while not stop.is_set():
                heartbeat_ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        heartbeat_task = asyncio.create_task(heartbeat())
        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await main._startup_submit_write(WriteOperation.RECOVER_BACKTESTS, {}, budget=0.2)
        elapsed = time.monotonic() - started
        stop.set()
        await heartbeat_task
        return elapsed, len(heartbeat_ticks)

    elapsed, tick_count = asyncio.run(scenario())

    assert elapsed < 1.0, f"startup call blocked past its budget: {elapsed}s"
    # The event loop must have kept ticking the heartbeat coroutine while the
    # stalled writer call was in flight -- a genuinely blocking call would
    # have starved it for the whole 0.2s+ wait.
    assert tick_count >= 3, f"event loop appears to have been blocked: {tick_count} ticks"


def test_healthy_writer_returns_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def fast_submit_write(operation, payload, **_kwargs):
        return {"recovered": 2}

    monkeypatch.setattr(main, "submit_write", fast_submit_write)

    result = asyncio.run(main._startup_submit_write(WriteOperation.RECOVER_BACKTESTS, {}, budget=5.0))

    assert result == {"recovered": 2}


def test_writer_exception_propagates_for_caller_to_degrade(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def failing_submit_write(operation, payload, **_kwargs):
        raise RuntimeError("writer request failed")

    monkeypatch.setattr(main, "submit_write", failing_submit_write)

    with pytest.raises(RuntimeError, match="writer request failed"):
        asyncio.run(main._startup_submit_write(WriteOperation.RECOVER_BACKTESTS, {}, budget=5.0))


def test_remote_client_startup_enqueues_no_database_maintenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collector-first topology: an additional API process is a read client,
    so merely starting it must not enqueue recovery, retention, or instrument
    refresh writes against the production database."""
    import main

    calls: list[object] = []

    async def unexpected_submit(*_args, **_kwargs):
        calls.append("write")
        raise AssertionError("remote client must not enqueue startup writes")

    monkeypatch.setattr(main, "_startup_submit_write", unexpected_submit)
    monkeypatch.setattr(
        main,
        "_refresh_instrument_master_safely",
        lambda: calls.append("instrument_refresh"),
    )

    ran = asyncio.run(main._run_writer_owner_startup_maintenance(owns_writer=False))
    refresh_started = main._start_instrument_refresh_if_owner(owns_writer=False)

    assert ran is False
    assert refresh_started is False
    assert calls == []


def test_local_writer_startup_runs_expected_database_maintenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import main

    calls: list[object] = []

    async def record_submit(operation, payload):
        calls.append(operation)
        return {"recovered": 0} if operation is WriteOperation.RECOVER_BACKTESTS else {}

    class InlineThread:
        def __init__(self, *, target, daemon):
            assert daemon is True
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(main, "_startup_submit_write", record_submit)
    monkeypatch.setattr(main, "Thread", InlineThread)
    monkeypatch.setattr(
        main,
        "_refresh_instrument_master_safely",
        lambda: calls.append("instrument_refresh"),
    )

    ran = asyncio.run(main._run_writer_owner_startup_maintenance(owns_writer=True))
    refresh_started = main._start_instrument_refresh_if_owner(owns_writer=True)

    assert ran is True
    assert refresh_started is True
    assert calls == [
        WriteOperation.RECOVER_BACKTESTS,
        WriteOperation.APPLY_RETENTION,
        "instrument_refresh",
    ]
