from __future__ import annotations

import datetime as dt
import json
import time
import weakref
from pathlib import Path
from typing import Any, Callable

from auto_collector.supervision import status_snapshot
from data.exchange_calendar import exchange_calendar
from db.duckdb_client import connect_observational
from db.writer_lease import writer_lease_status
from ml.registry import registry_health
from security.log_scrubber import scrub_text


PROCESS_STARTED_MONOTONIC = time.monotonic()
HEALTHY_STATES = {"HEALTHY", "SUSPENDED"}


def health_snapshot(
    *,
    database_path: Path,
    collector_db: Path,
    redis_client: Any,
    feed_status: Callable[[], dict[str, Any]],
    release: dict[str, Any],
    integrity: dict[str, Any],
    external_success_times: dict[str, float],
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    instant = now or dt.datetime.now(dt.timezone.utc)
    checked_at = instant.isoformat()
    calendar = exchange_calendar()
    session = calendar.session_state(instant)
    components: dict[str, dict[str, Any]] = {
        "backend": {
            "state": "HEALTHY",
            "critical": True,
            "uptime_seconds": round(time.monotonic() - PROCESS_STARTED_MONOTONIC, 3),
            "checked_at": checked_at,
        }
    }
    redis_state = _redis_health(redis_client, session, checked_at)
    components["redis"] = redis_state
    components["angel_stream"] = _feed_health(
        redis_client, redis_state, feed_status, session, checked_at
    )
    components["nse_chain_poller"] = _poller_health(redis_client, redis_state, session, checked_at)
    components["collector"] = _collector_health(collector_db, checked_at)
    database, freshness = _database_health(database_path, instant, checked_at)
    components["duckdb_writer"] = database
    components["external_providers"] = _external_health(
        external_success_times,
        checked_at,
        now_epoch=instant.timestamp(),
    )
    model_registry = registry_health()
    components["model_registry"] = {
        **model_registry,
        "critical": False,
        "checked_at": checked_at,
    }
    release_healthy = bool(
        release.get("build_status") == "COHERENT" and integrity.get("trusted") is True
    )
    components["release_identity"] = {
        "state": "HEALTHY" if release_healthy else "DEGRADED",
        "critical": True,
        "release_version": release.get("release_version"),
        "build_status": release.get("build_status"),
        "integrity_status": integrity.get("status"),
        "trusted": bool(integrity.get("trusted")),
        "reason": None if release_healthy else "release identity or protected assets are not trusted",
        "checked_at": checked_at,
    }
    components["data_freshness"] = {
        "state": _aggregate_freshness(freshness),
        "critical": False,
        "datasets": freshness,
        "checked_at": checked_at,
    }

    critical_failures = [
        name
        for name, state in components.items()
        if state.get("critical") and state.get("state") not in HEALTHY_STATES
    ]
    degraded = [name for name, state in components.items() if state.get("state") not in HEALTHY_STATES]
    ready = not critical_failures
    return {
        "status": "READY" if ready and not degraded else "DEGRADED" if ready else "NOT_READY",
        "liveness": {"status": "ALIVE", "uptime_seconds": components["backend"]["uptime_seconds"]},
        "readiness": {
            "status": "READY" if ready else "NOT_READY",
            "ready": ready,
            "blocking_components": critical_failures,
        },
        "degraded_state": {
            "degraded": bool(degraded),
            "components": degraded,
        },
        "market_session": session,
        "timestamp": checked_at,
        "release": release,
        "components": components,
    }


# A monitoring probe polling /health during a Redis outage must not pay the
# full socket_connect_timeout/socket_timeout (2s each in redis_config.py) on
# every single request -- that turns one dependency outage into a multi-second
# latency spike on every health check. Cache the raw ping outcome briefly so
# repeated probes during an outage (or while healthy) reuse the last real
# attempt instead of re-blocking; checked_at/market_session are always
# re-stamped so the response still reflects the current request.
_REDIS_PING_CACHE_TTL_SECONDS = 3.0
# Keyed by the client object itself via a WeakKeyDictionary, NOT id(client):
# a plain dict keyed by id() holds no reference to the client, so once a
# short-lived client (a fresh FakeRedis per test, or any client that falls
# out of scope) is garbage-collected, CPython is free to reuse that exact id
# for a brand-new, unrelated object within the TTL window -- causing the new
# client to silently inherit a stale cached ping outcome from a client it
# never was. A WeakKeyDictionary hashes/compares by real object identity and
# auto-evicts its entry the moment the client is collected, so a reused id
# can never collide with a live cache entry.
_redis_ping_cache: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()


def _redis_health(client: Any, session: str, checked_at: str) -> dict[str, Any]:
    now = time.monotonic()
    cache = _redis_ping_cache.get(client)
    if cache is not None and (now - cache["checked_monotonic"]) < _REDIS_PING_CACHE_TTL_SECONDS:
        ok = cache["ok"]
        latency_ms = cache["latency_ms"]
        reason = cache["reason"]
    else:
        started = time.perf_counter()
        try:
            result = client.ping()
            if result is not True:
                raise RuntimeError("PING did not return success")
            ok, reason = True, None
        except Exception as error:
            ok, reason = False, f"redis probe failed: {type(error).__name__}"
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        try:
            _redis_ping_cache[client] = {
                "checked_monotonic": now,
                "ok": ok,
                "reason": reason,
                "latency_ms": latency_ms,
            }
        except TypeError:
            # client isn't weakly-referenceable (e.g. a bare object() in a
            # test double) -- skip caching rather than fail the health check.
            pass

    if not ok:
        return {
            "state": "UNAVAILABLE",
            "critical": True,
            "latency_ms": latency_ms,
            "reason": reason,
            "recovery": "retrying on next health probe",
            "checked_at": checked_at,
        }
    return {
        "state": "HEALTHY",
        "critical": True,
        "latency_ms": latency_ms,
        "market_session": session,
        "checked_at": checked_at,
    }


def _feed_health(
    client: Any,
    redis_state: dict[str, Any],
    feed_status: Callable[[], dict[str, Any]],
    session: str,
    checked_at: str,
) -> dict[str, Any]:
    # `feed_status` (live.multi_feed.get_status) opens its OWN fresh Redis
    # client and pays a full socket_connect_timeout when called with no
    # client argument -- a cost `_redis_health`'s cache above cannot see or
    # amortize, since it is a different client object entirely. When we
    # already know redis is unavailable this call can only ever resolve to
    # the same "missing feed health" degraded outcome it produces internally
    # on a RedisError, so skip the live probe and report that outcome
    # directly instead of paying the connection timeout again.
    if redis_state["state"] != "HEALTHY":
        raw: dict[str, Any] = {"state": "FAILED", "connected": False, "last_error": "redis unavailable"}
    else:
        try:
            raw = feed_status()
        except Exception as error:
            raw = {"state": "FAILED", "connected": False, "last_error": type(error).__name__}
    last_tick_age = int(raw.get("last_tick_ms_ago", -1) or -1)
    connected = bool(raw.get("connected"))
    active = int(raw.get("symbols_active", 0) or 0)
    rust = _runtime_health_key(client, redis_state, "health:angel_stream")
    python_healthy = connected and active > 0 and 0 <= last_tick_age <= 60_000
    rust_healthy = rust.get("state") == "HEALTHY"
    if session != "OPEN":
        state = "SUSPENDED"
        reason = "market session closed"
    elif rust_healthy or python_healthy:
        state = "HEALTHY"
        reason = None
    else:
        state = "DEGRADED"
        reason = "feed is not live with fresh active symbols"
    return {
        "state": state,
        "critical": False,
        "producers": {
            "rust_desktop_ingester": rust,
            "python_multi_feed": {
                "state": "HEALTHY" if python_healthy else "DEGRADED",
                "phase": raw.get("state", "UNKNOWN"),
                "connected": connected,
                "symbols_subscribed": int(raw.get("symbols_subscribed", 0) or 0),
                "symbols_active": active,
                "last_tick_ms_ago": last_tick_age,
                "consecutive_failures": int(raw.get("consecutive_failures", 0) or 0),
                "last_error": scrub_text(str(raw.get("last_error"))) if raw.get("last_error") else None,
            },
        },
        "reason": reason,
        "recovery": "automatic reconnect with bounded backoff" if state == "DEGRADED" else None,
        "checked_at": checked_at,
    }


def _poller_health(
    client: Any,
    redis_state: dict[str, Any],
    session: str,
    checked_at: str,
) -> dict[str, Any]:
    if session != "OPEN":
        return {
            "state": "SUSPENDED",
            "critical": False,
            "reason": "market session closed",
            "checked_at": checked_at,
        }
    if redis_state["state"] != "HEALTHY":
        return {
            "state": "UNKNOWN",
            "critical": True,
            "reason": "redis unavailable; poller output cannot be observed",
            "checked_at": checked_at,
        }
    runtime = _runtime_health_key(client, redis_state, "health:nse_chain_poller")
    healthy = runtime.get("state") == "HEALTHY"
    return {
        "state": "HEALTHY" if healthy else "STALE",
        "critical": False,
        "runtime": runtime,
        "reason": None if healthy else "no live NSE chain output with a positive TTL",
        "recovery": "poller retries session and provider failures" if not healthy else None,
        "checked_at": checked_at,
    }


def _runtime_health_key(
    client: Any, redis_state: dict[str, Any], key: str
) -> dict[str, Any]:
    if redis_state.get("state") != "HEALTHY":
        return {"state": "UNKNOWN", "ttl_seconds": None, "reason": "redis unavailable"}
    try:
        raw = client.get(key)
        ttl = int(client.ttl(key)) if raw is not None else -2
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="strict")
        payload = json.loads(str(raw)) if raw is not None else {}
    except Exception as error:
        return {
            "state": "UNAVAILABLE",
            "ttl_seconds": None,
            "reason": f"runtime health read failed: {type(error).__name__}",
        }
    return {
        "state": "HEALTHY" if raw is not None and ttl > 0 else "STALE",
        "ttl_seconds": ttl,
        "source": payload.get("source"),
        "last_success_ms": payload.get("last_success_ms"),
    }


def _collector_health(path: Path, checked_at: str) -> dict[str, Any]:
    raw = status_snapshot(path)
    instance = raw.get("instance") or {}
    running = bool(raw.get("collector_running"))
    state = "HEALTHY" if running else "NOT_RUNNING"
    return {
        "state": state,
        "critical": False,
        "running": running,
        "activity": instance.get("activity", "MISSING"),
        "heartbeat_at": instance.get("heartbeat_at"),
        "lease_expires_at": instance.get("lease_expires_at"),
        "abandoned_jobs": int(raw.get("abandoned_jobs", 0) or 0),
        "reason": scrub_text(str(raw.get("error"))) if raw.get("error") else None,
        "recovery": "Windows Task Scheduler restarts failed process" if not running else None,
        "checked_at": checked_at,
    }


def _database_health(
    path: Path, instant: dt.datetime, checked_at: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    freshness: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        with connect_observational(path) as connection:
            connection.execute("SELECT 1").fetchone()
            schema_row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
            for table in ("ohlcv", "fo_eod", "iv_history", "signals_history"):
                row = connection.execute(f"SELECT MAX(date) FROM {table}").fetchone()
                freshness[table] = _freshness_record(row[0], instant)
    except Exception as error:
        return (
            {
                "state": "UNAVAILABLE",
                "critical": True,
                "path": str(path),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "reason": f"database read probe failed: {type(error).__name__}",
                "checked_at": checked_at,
            },
            {"state": "UNKNOWN", "reason": "database unavailable"},
        )
    lease = writer_lease_status()
    return (
        {
            "state": "HEALTHY",
            "critical": True,
            "path": str(path),
            "schema_version": int(schema_row[0] or 0),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "writer_lease": lease,
            "checked_at": checked_at,
        },
        freshness,
    )


def _freshness_record(value: Any, instant: dt.datetime) -> dict[str, Any]:
    if value is None:
        return {"state": "EMPTY", "latest_date": None, "expected_date": _expected_eod(instant).isoformat()}
    latest = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value)[:10])
    expected = _expected_eod(instant)
    return {
        "state": "HEALTHY" if latest >= expected else "STALE",
        "latest_date": latest.isoformat(),
        "expected_date": expected.isoformat(),
        "lag_calendar_days": max(0, (expected - latest).days),
    }


def _expected_eod(instant: dt.datetime) -> dt.date:
    calendar = exchange_calendar()
    local = calendar.as_ist(instant)
    day = local.date()
    if calendar.is_trading_day(day) and local.time().replace(tzinfo=None) >= dt.time(18, 0):
        return day
    return calendar.previous_trading_day(day - dt.timedelta(days=1))


def _external_health(
    success_times: dict[str, float], checked_at: str, *, now_epoch: float
) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for name, ttl in (("cross_asset", 120), ("sentiment", 240)):
        timestamp = float(success_times.get(name, 0.0) or 0.0)
        age = max(0.0, now_epoch - timestamp) if timestamp else None
        providers[name] = {
            "state": "HEALTHY" if age is not None and age <= ttl else "UNKNOWN" if age is None else "STALE",
            "last_success_epoch": timestamp or None,
            "age_seconds": round(age, 3) if age is not None else None,
            "freshness_budget_seconds": ttl,
        }
    state = "HEALTHY" if all(item["state"] == "HEALTHY" for item in providers.values()) else "DEGRADED"
    return {
        "state": state,
        "critical": False,
        "providers": providers,
        "checked_at": checked_at,
    }


def _aggregate_freshness(freshness: dict[str, Any]) -> str:
    if not freshness:
        return "UNKNOWN"
    states = {item.get("state") for item in freshness.values() if isinstance(item, dict)}
    if states == {"HEALTHY"}:
        return "HEALTHY"
    if "STALE" in states:
        return "STALE"
    return "DEGRADED"
