from __future__ import annotations

import json
import logging
import random
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import redis

from data.index_aliases import CANONICAL_INDEX_SYMBOLS, INDEX_CANONICAL_NAME
from data.instrument_schema import read_instrument_frame
from data.paths import INSTRUMENTS_CACHE


FEED_CONNECTION_OWNER = "PYTHON_SECURE_FEED_BRIDGE"
FEED_WRITER_OWNER = "RUST_TAURI_INGESTER"
INSTRUMENTS_PATH = INSTRUMENTS_CACHE
MAX_BACKOFF_SECONDS = 30
HEALTH_MAX_AGE_MS = 120_000

# Derived from the canonical alias registry (compute/data/index_aliases.py)
# so this file's own index-name knowledge never drifts from the DB-backed
# resolver's (data.instrument_master.resolve_index_feed / db.write_service's
# RESOLVE_INDEX_INSTRUMENT) -- this used to be a third hand-maintained copy
# of the same alias strings, which is exactly how MIDCPNIFTY ended up never
# tracked here at all even after being added to the canonical registry.
# "NIFTY MIDCAP 100" and "INDIA VIX" are outside the canonical 4-index
# registry (not part of resolve_index_feed's supported set) but are real,
# separately-useful live-feed targets, so they stay listed explicitly here.
INDEX_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (
        canonical_symbol,
        tuple(sorted({canonical_symbol, INDEX_CANONICAL_NAME[canonical_symbol]})),
    )
    for canonical_symbol in CANONICAL_INDEX_SYMBOLS
) + (
    ("NIFTY MIDCAP 100", ("NIFTY MIDCAP 100",)),
    ("INDIA VIX", ("INDIA VIX",)),
)


class FeedPhase(StrEnum):
    STARTING_BACKEND = "STARTING_BACKEND"
    WAITING_FOR_HEALTH = "WAITING_FOR_HEALTH"
    AUTHENTICATED = "AUTHENTICATED"
    CONNECTING_FEED = "CONNECTING_FEED"
    SUBSCRIBING = "SUBSCRIBING"
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"
    SHUTTING_DOWN = "SHUTTING_DOWN"


_logger = logging.getLogger("live_feed_consumer")


async def run() -> None:
    raise RuntimeError(
        "standalone feed loop disabled; secure bridge is owned by the local backend lifecycle"
    )


def get_status(redis_client: Any | None = None) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    if redis_client is None:
        from security.redis_config import authenticated_redis_client

        client = authenticated_redis_client(decode_responses=True)
    else:
        client = redis_client
    try:
        raw = client.get("health:angel_stream")
        payload = json.loads(raw) if raw else None
    except (redis.RedisError, TypeError, ValueError):
        payload = None
    if not isinstance(payload, dict):
        return _status(FeedPhase.DEGRADED, now_ms, None, "missing feed health", None)
    source = str(payload.get("source") or "")
    last_success_ms = _int_or_zero(payload.get("last_success_ms"))
    if source != FEED_WRITER_OWNER:
        return _status(FeedPhase.DEGRADED, now_ms, last_success_ms, "untrusted feed writer", None)
    if last_success_ms <= 0 or now_ms - last_success_ms > HEALTH_MAX_AGE_MS:
        return _status(FeedPhase.DEGRADED, now_ms, last_success_ms, "stale feed health", payload)
    if payload.get("state") != FeedPhase.LIVE.value:
        return _status(FeedPhase.DEGRADED, now_ms, last_success_ms, "feed not live", payload)
    return _status(FeedPhase.LIVE, now_ms, last_success_ms, None, payload)


def _status(
    phase: FeedPhase,
    now_ms: int,
    last_success_ms: int | None,
    error: str | None,
    coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    source = coverage if isinstance(coverage, dict) else {}
    result = {
        "schema_version": 1,
        "state": phase.value,
        "connected": phase is FeedPhase.LIVE,
        "connection_owner": FEED_CONNECTION_OWNER,
        "writer_owner": FEED_WRITER_OWNER,
        "last_tick_ms_ago": (
            max(0, now_ms - last_success_ms) if last_success_ms and last_success_ms > 0 else -1
        ),
        "last_error": error,
        "expected": _int_or_zero(source.get("expected")),
        "symbols_subscribed": _int_or_zero(source.get("subscribed")),
        "symbols_active": _int_or_zero(source.get("receiving")),
        "fresh": _int_or_zero(source.get("fresh")),
        "stale": _int_or_zero(source.get("stale")),
        "unavailable": _int_or_zero(source.get("unavailable")),
        "unknown_tokens": _int_or_zero(source.get("unknown_tokens")),
        "parser_errors": _int_or_zero(source.get("parser_errors")),
        "duplicate_subscriptions": _int_or_zero(source.get("duplicate_subscriptions")),
        "reconnect_count": _int_or_zero(source.get("reconnect_count")),
        "packet_count": _int_or_zero(source.get("packet_count")),
        "quote_updates_per_second": float(source.get("quote_updates_per_second") or 0.0),
        "provider_token_limit": _int_or_zero(source.get("provider_token_limit")),
        "freshness_threshold_ms": _int_or_zero(source.get("freshness_threshold_ms")),
        "coverage_stabilization_ms": source.get("coverage_stabilization_ms"),
        "last_packet_time_ms": source.get("last_packet_time_ms"),
    }
    for name in (
        "expected_symbols",
        "subscribed_symbols",
        "fresh_symbols",
        "stale_symbols",
        "unavailable_symbols",
    ):
        values = source.get(name)
        result[name] = [str(value) for value in values] if isinstance(values, list) else []
    return result


def build_token_list(path: Path = INSTRUMENTS_PATH) -> tuple[list[str], dict[str, str]]:
    frame = read_instrument_frame(path)
    nfo = frame[
        (frame["exchange"] == "NFO")
        & (frame["instrument_type"].isin(["FUTSTK", "OPTSTK"]))
    ]
    underlyings = sorted(
        {
            str(value).strip().upper()
            for value in nfo["name"].dropna().tolist()
            if str(value).strip()
        }
    )
    nse = frame[frame["exchange"] == "NSE"].copy()
    nse["_symbol_norm"] = nse["symbol"].fillna("").astype(str).str.strip().str.upper()
    nse["_name_norm"] = nse["name"].fillna("").astype(str).str.strip().str.upper()
    token_to_symbol: dict[str, str] = {}
    for underlying in underlyings:
        if "NSETEST" in underlying:
            continue
        row = _find_spot_row(nse, underlying)
        if row is None:
            _logger.warning("no NSE spot token for %s", underlying)
            continue
        token = _clean_token(row.get("token"))
        if token:
            token_to_symbol.setdefault(token, underlying)
    _add_index_tokens(nse, token_to_symbol)
    return list(token_to_symbol), token_to_symbol


def _find_spot_row(nse: pd.DataFrame, underlying: str) -> pd.Series | None:
    for column, value in (
        ("_symbol_norm", underlying),
        ("_symbol_norm", f"{underlying}-EQ"),
        ("_name_norm", underlying),
    ):
        matches = nse[nse[column] == value]
        if not matches.empty:
            return matches.iloc[0]
    return None


def _add_index_tokens(nse: pd.DataFrame, token_to_symbol: dict[str, str]) -> None:
    index_rows = nse[nse["instrument_type"] == "AMXIDX"]
    for redis_symbol, aliases in INDEX_TARGETS:
        token = _find_index_token(index_rows, aliases)
        if token:
            token_to_symbol[token] = redis_symbol


def _find_index_token(index_rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized_aliases = {_match_key(alias) for alias in aliases}
    for _, row in index_rows.iterrows():
        if _match_key(row.get("symbol")) in normalized_aliases or _match_key(
            row.get("name")
        ) in normalized_aliases:
            return _clean_token(row.get("token"))
    return None


def _clean_token(value: Any) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") else text


def _match_key(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def valid_transition(current: FeedPhase, next_phase: FeedPhase) -> bool:
    allowed = {
        FeedPhase.STARTING_BACKEND: {
            FeedPhase.WAITING_FOR_HEALTH,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.WAITING_FOR_HEALTH: {
            FeedPhase.AUTHENTICATED,
            FeedPhase.DEGRADED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.AUTHENTICATED: {
            FeedPhase.CONNECTING_FEED,
            FeedPhase.DEGRADED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.CONNECTING_FEED: {
            FeedPhase.SUBSCRIBING,
            FeedPhase.RECONNECTING,
            FeedPhase.DEGRADED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.SUBSCRIBING: {
            FeedPhase.WAITING_FOR_DATA,
            FeedPhase.RECONNECTING,
            FeedPhase.DEGRADED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.WAITING_FOR_DATA: {
            FeedPhase.LIVE,
            FeedPhase.RECONNECTING,
            FeedPhase.DEGRADED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.LIVE: {
            FeedPhase.RECONNECTING,
            FeedPhase.DEGRADED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.DEGRADED: {
            FeedPhase.WAITING_FOR_HEALTH,
            FeedPhase.RECONNECTING,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.RECONNECTING: {
            FeedPhase.WAITING_FOR_HEALTH,
            FeedPhase.CONNECTING_FEED,
            FeedPhase.FAILED,
            FeedPhase.SHUTTING_DOWN,
        },
        FeedPhase.FAILED: {FeedPhase.WAITING_FOR_HEALTH, FeedPhase.SHUTTING_DOWN},
        FeedPhase.SHUTTING_DOWN: set(),
    }
    return next_phase in allowed[current]


def _backoff_delay(base_seconds: int, jitter_fraction: float | None = None) -> float:
    base = float(max(1, min(base_seconds, MAX_BACKOFF_SECONDS)))
    jitter = (
        random.SystemRandom().uniform(0.0, 0.25)
        if jitter_fraction is None
        else jitter_fraction
    )
    return min(base * (1.0 + max(0.0, min(jitter, 0.25))), MAX_BACKOFF_SECONDS)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
