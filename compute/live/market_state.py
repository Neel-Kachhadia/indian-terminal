from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Mapping


QUOTE_SCHEMA_VERSION = 1
STREAM_FRESH_MS = 30_000


class QuoteSource(StrEnum):
    ANGEL_STREAM = "ANGEL_STREAM"
    NSE_CHAIN = "NSE_CHAIN"
    CACHE = "CACHE"
    HISTORICAL = "HISTORICAL"
    DERIVED = "DERIVED"


class QuoteFreshness(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class QuoteQuality(StrEnum):
    EXECUTABLE_STREAM = "EXECUTABLE_STREAM"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


class MarketDataState(StrEnum):
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"
    STARTING = "STARTING"


@dataclass(frozen=True)
class QuoteEnvelope:
    schema_version: int
    symbol: str
    token: str
    ltp: float
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None
    oi: int | None
    bid: float | None
    ask: float | None
    exchange: str
    source_type: QuoteSource
    observed_at_ms: int
    received_at_ms: int
    sequence: int | None
    field_completeness: tuple[str, ...]
    freshness: QuoteFreshness
    quality: QuoteQuality
    fallback: bool
    fallback_reason: str | None
    market_state: MarketDataState
    requested_symbol: str | None = None
    canonical_symbol: str | None = None
    age_ms: int = 0
    connection_state: MarketDataState = MarketDataState.DISCONNECTED
    decision_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["field_completeness"] = list(self.field_completeness)
        return payload


def parse_quote_envelope(data: Mapping[str, Any]) -> QuoteEnvelope:
    required = {
        "schema_version", "symbol", "token", "ltp", "open", "high", "low", "close",
        "volume", "oi", "bid", "ask", "exchange", "source_type", "observed_at_ms",
        "received_at_ms", "sequence", "field_completeness", "freshness", "quality",
        "fallback", "fallback_reason", "market_state",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"quote contract missing fields: {', '.join(missing)}")
    if int(data["schema_version"]) != QUOTE_SCHEMA_VERSION:
        raise ValueError(f"unsupported quote schema_version: {data['schema_version']!r}")

    symbol = str(data["symbol"]).strip().upper()
    token = str(data["token"]).strip()
    exchange = str(data["exchange"]).strip().upper()
    if not symbol or not token or not exchange:
        raise ValueError("quote identity fields must be non-empty")

    ltp = float(data["ltp"])
    if not math.isfinite(ltp) or ltp <= 0:
        raise ValueError("quote ltp must be finite and positive")
    optional_prices = {
        name: _optional_finite_float(data[name])
        for name in ("open", "high", "low", "close", "bid", "ask")
    }
    optional_counts = {
        name: _optional_nonnegative_int(data[name])
        for name in ("volume", "oi")
    }

    observed_at_ms = int(data["observed_at_ms"])
    received_at_ms = int(data["received_at_ms"])
    if observed_at_ms <= 0 or received_at_ms <= 0:
        raise ValueError("quote timestamps must be positive epoch milliseconds")

    completeness = tuple(str(value) for value in data["field_completeness"])
    if "ltp" not in completeness:
        raise ValueError("quote field_completeness must include ltp")
    optional_values = optional_prices | optional_counts
    inconsistent = sorted(
        name for name in completeness if name in optional_values and optional_values[name] is None
    )
    if inconsistent:
        raise ValueError(
            "quote field_completeness lists absent fields: " + ", ".join(inconsistent)
        )

    sequence_value = data["sequence"]
    sequence = None if sequence_value is None else int(sequence_value)
    fallback = data["fallback"]
    if not isinstance(fallback, bool):
        raise ValueError("quote fallback must be boolean")
    fallback_reason = data["fallback_reason"]
    if fallback and not str(fallback_reason or "").strip():
        raise ValueError("fallback quote must carry fallback_reason")

    return QuoteEnvelope(
        schema_version=QUOTE_SCHEMA_VERSION,
        symbol=symbol,
        token=token,
        ltp=ltp,
        open=optional_prices["open"],
        high=optional_prices["high"],
        low=optional_prices["low"],
        close=optional_prices["close"],
        volume=optional_counts["volume"],
        oi=optional_counts["oi"],
        bid=optional_prices["bid"],
        ask=optional_prices["ask"],
        exchange=exchange,
        source_type=QuoteSource(str(data["source_type"])),
        observed_at_ms=observed_at_ms,
        received_at_ms=received_at_ms,
        sequence=sequence,
        field_completeness=completeness,
        freshness=QuoteFreshness(str(data["freshness"])),
        quality=QuoteQuality(str(data["quality"])),
        fallback=fallback,
        fallback_reason=None if fallback_reason is None else str(fallback_reason),
        market_state=MarketDataState(str(data["market_state"])),
        requested_symbol=str(data.get("requested_symbol") or symbol).strip().upper(),
        canonical_symbol=str(data.get("canonical_symbol") or symbol).strip().upper(),
        age_ms=max(0, int(data.get("age_ms") or 0)),
        connection_state=MarketDataState(
            str(data.get("connection_state") or data["market_state"])
        ),
        decision_eligible=bool(data.get("decision_eligible", False)),
    )


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("optional quote prices must be finite when present")
    return number


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError("optional quote counts must be non-negative when present")
    return number


def evaluate_market_state(
    quote: QuoteEnvelope,
    *,
    now_ms: int | None = None,
    expected_feed_live: bool,
    expected_feed_starting: bool = False,
) -> QuoteEnvelope:
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    age_ms = max(0, current_ms - quote.observed_at_ms)
    freshness = QuoteFreshness.FRESH if age_ms <= STREAM_FRESH_MS else QuoteFreshness.STALE
    if freshness is QuoteFreshness.STALE:
        state = MarketDataState.STALE
    elif expected_feed_starting:
        state = MarketDataState.STARTING
    elif (
        quote.fallback
        or quote.source_type is not QuoteSource.ANGEL_STREAM
        or quote.quality is not QuoteQuality.EXECUTABLE_STREAM
    ):
        state = MarketDataState.DEGRADED
    elif expected_feed_live:
        state = MarketDataState.LIVE
    else:
        state = MarketDataState.DISCONNECTED
    decision_eligible = bool(
        state is MarketDataState.LIVE
        and freshness is QuoteFreshness.FRESH
        and quote.source_type is QuoteSource.ANGEL_STREAM
        and quote.quality is QuoteQuality.EXECUTABLE_STREAM
        and not quote.fallback
    )
    return replace(
        quote,
        freshness=freshness,
        market_state=state,
        age_ms=age_ms,
        connection_state=state,
        decision_eligible=decision_eligible,
    )


def should_accept_update(incoming: QuoteEnvelope, existing: QuoteEnvelope) -> bool:
    if incoming.sequence is not None and existing.sequence is not None and incoming.sequence != existing.sequence:
        return incoming.sequence > existing.sequence
    if incoming.observed_at_ms != existing.observed_at_ms:
        return incoming.observed_at_ms > existing.observed_at_ms
    return incoming.received_at_ms > existing.received_at_ms
