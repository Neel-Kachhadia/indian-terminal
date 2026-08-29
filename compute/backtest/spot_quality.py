from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from data.exchange_calendar import CalendarCoverageError, exchange_calendar
from data.market_data_quality import INDEX_SYMBOLS, VERIFIED_EQUITY_SOURCES, VERIFIED_INDEX_SOURCES

INVALID_DERIVED_SPOT_SOURCE = "FO_EOD_FUTIDX_DERIVED"
VERIFIED_INDEX_SPOT_SOURCES = VERIFIED_INDEX_SOURCES
VERIFIED_EQUITY_SPOT_SOURCES = VERIFIED_EQUITY_SOURCES
VERIFIED_OHLCV_SOURCES = VERIFIED_INDEX_SPOT_SOURCES | VERIFIED_EQUITY_SPOT_SOURCES


class SpotQuality(StrEnum):
    VERIFIED = "VERIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_SOURCE = "INVALID_SOURCE"
    INVALID_VALUE = "INVALID_VALUE"
    FUTURE_OBSERVATION = "FUTURE_OBSERVATION"
    STALE = "STALE"


class SpotFallbackPolicy(StrEnum):
    NO_FALLBACK = "NO_FALLBACK"
    PREVIOUS_SESSION = "PREVIOUS_SESSION"


@dataclass(frozen=True)
class SpotObservation:
    price: float | None
    requested_as_of: dt.date
    actual_observed_at: dt.datetime | None
    source: str
    quality: SpotQuality
    reason: str
    staleness_sessions: int | None
    maximum_staleness_sessions: int
    fallback_used: bool
    fallback_policy: SpotFallbackPolicy
    instrument: str = ""

    @property
    def valid(self) -> bool:
        return self.quality is SpotQuality.VERIFIED

    @property
    def value(self) -> float | None:
        return self.price

    @property
    def decision_eligible(self) -> bool:
        return self.valid

    @property
    def freshness(self) -> str:
        if self.quality is SpotQuality.STALE:
            return "STALE"
        if not self.valid:
            return "UNAVAILABLE"
        return "POLICY_ALLOWED_FALLBACK" if self.fallback_used else "FRESH"

    @property
    def observed_on(self) -> dt.date | None:
        return self.actual_observed_at.date() if self.actual_observed_at else None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "requested_as_of": self.requested_as_of.isoformat(),
            "actual_observed_at": (
                self.actual_observed_at.isoformat() if self.actual_observed_at else None
            ),
            "source": self.source,
            "quality": self.quality.value,
            "reason": self.reason,
            "staleness_sessions": self.staleness_sessions,
            "maximum_staleness_sessions": self.maximum_staleness_sessions,
            "fallback_used": self.fallback_used,
            "fallback_policy": self.fallback_policy.value,
            "freshness": self.freshness,
            "decision_eligible": self.decision_eligible,
        }


def verified_spot_sources(symbol: str) -> frozenset[str]:
    normalized = str(symbol or "").strip().upper()
    if normalized in INDEX_SYMBOLS:
        return VERIFIED_INDEX_SPOT_SOURCES
    return VERIFIED_EQUITY_SPOT_SOURCES


def classify_spot_row(
    row: Mapping[str, Any] | None,
    *,
    symbol: str,
    requested_as_of: dt.date,
    fallback_policy: SpotFallbackPolicy = SpotFallbackPolicy.NO_FALLBACK,
    maximum_staleness_sessions: int = 0,
) -> SpotObservation:
    policy = SpotFallbackPolicy(fallback_policy)
    normalized_symbol = str(symbol or "").strip().upper()
    max_staleness = max(int(maximum_staleness_sessions), 0)
    if row is None:
        return SpotObservation(
            None,
            requested_as_of,
            None,
            "NONE",
            SpotQuality.UNAVAILABLE,
            "missing_verified_spot_observation",
            None,
            max_staleness,
            False,
            policy,
            normalized_symbol,
        )

    source = str(row.get("source") or "UNKNOWN").strip().upper()
    observed_on = _date_or_none(row.get("observed_at") or row.get("date"))
    actual_observed_at = (
        dt.datetime.combine(observed_on, dt.time.min, dt.timezone.utc) if observed_on else None
    )
    price = _finite_or_none(row.get("close"))
    if source not in verified_spot_sources(symbol):
        return SpotObservation(
            price,
            requested_as_of,
            actual_observed_at,
            source,
            SpotQuality.INVALID_SOURCE,
            "unverified_spot_source",
            None,
            max_staleness,
            False,
            policy,
            normalized_symbol,
        )
    if observed_on is None or price is None or price <= 0:
        return SpotObservation(
            price,
            requested_as_of,
            actual_observed_at,
            source,
            SpotQuality.INVALID_VALUE,
            "invalid_spot_value_or_timestamp",
            None,
            max_staleness,
            False,
            policy,
            normalized_symbol,
        )
    if observed_on > requested_as_of:
        return SpotObservation(
            price,
            requested_as_of,
            actual_observed_at,
            source,
            SpotQuality.FUTURE_OBSERVATION,
            "future_spot_observation",
            None,
            max_staleness,
            False,
            policy,
            normalized_symbol,
        )

    try:
        staleness_sessions = _session_distance(observed_on, requested_as_of)
    except CalendarCoverageError:
        return SpotObservation(
            price,
            requested_as_of,
            actual_observed_at,
            source,
            SpotQuality.STALE,
            "spot_session_staleness_outside_calendar_authority",
            None,
            max_staleness,
            observed_on != requested_as_of,
            policy,
            normalized_symbol,
        )
    fallback_used = observed_on != requested_as_of
    if fallback_used and (
        policy is SpotFallbackPolicy.NO_FALLBACK or staleness_sessions > max_staleness
    ):
        return SpotObservation(
            price,
            requested_as_of,
            actual_observed_at,
            source,
            SpotQuality.STALE,
            "spot_observation_outside_staleness_policy",
            staleness_sessions,
            max_staleness,
            True,
            policy,
            normalized_symbol,
        )
    return SpotObservation(
        price,
        requested_as_of,
        actual_observed_at,
        source,
        SpotQuality.VERIFIED,
        "verified_source_and_freshness",
        staleness_sessions,
        max_staleness,
        fallback_used,
        policy,
        normalized_symbol,
    )


def classify_live_spot(
    quote: Mapping[str, Any] | None,
    *,
    symbol: str,
    requested_at: dt.datetime,
) -> SpotObservation:
    """Convert a fully validated live quote envelope into canonical spot provenance."""
    if requested_at.tzinfo is None or requested_at.utcoffset() is None:
        raise ValueError("requested_at must be timezone-aware")
    normalized = str(symbol or "").strip().upper()
    requested_date = exchange_calendar().as_ist(requested_at).date()
    if not isinstance(quote, Mapping):
        return _live_spot_failure(
            normalized,
            requested_date,
            None,
            "NONE",
            SpotQuality.UNAVAILABLE,
            "missing_verified_live_spot",
        )

    source = str(quote.get("source_type") or "UNKNOWN").strip().upper()
    observed_ms = _integer_or_none(quote.get("observed_at_ms"))
    observed_at = (
        dt.datetime.fromtimestamp(observed_ms / 1000, tz=dt.timezone.utc)
        if observed_ms is not None and observed_ms > 0
        else None
    )
    price = _finite_or_none(quote.get("ltp"))
    quote_symbol = str(quote.get("symbol") or "").strip().upper()
    if quote_symbol != normalized:
        return _live_spot_failure(
            normalized, requested_date, observed_at, source, SpotQuality.INVALID_SOURCE,
            "live_spot_instrument_mismatch", price,
        )
    if price is None or price <= 0 or observed_at is None:
        return _live_spot_failure(
            normalized, requested_date, observed_at, source, SpotQuality.INVALID_VALUE,
            "invalid_live_spot_value_or_timestamp", price,
        )
    if observed_at > requested_at:
        return _live_spot_failure(
            normalized, requested_date, observed_at, source, SpotQuality.FUTURE_OBSERVATION,
            "future_live_spot_observation", price,
        )
    eligible = (
        source == "ANGEL_STREAM"
        and str(quote.get("quality") or "") == "EXECUTABLE_STREAM"
        and str(quote.get("freshness") or "") == "FRESH"
        and str(quote.get("market_state") or "") == "LIVE"
        and quote.get("fallback") is False
    )
    if not eligible:
        return _live_spot_failure(
            normalized, requested_date, observed_at, source, SpotQuality.INVALID_SOURCE,
            "live_spot_provenance_not_decision_eligible", price,
        )
    return SpotObservation(
        price,
        requested_date,
        observed_at,
        source,
        SpotQuality.VERIFIED,
        "verified_live_spot",
        0,
        0,
        False,
        SpotFallbackPolicy.NO_FALLBACK,
        normalized,
    )


def unavailable_spot(symbol: str, requested_at: dt.datetime, reason: str) -> SpotObservation:
    if requested_at.tzinfo is None or requested_at.utcoffset() is None:
        raise ValueError("requested_at must be timezone-aware")
    normalized = str(symbol or "").strip().upper()
    return _live_spot_failure(
        normalized,
        exchange_calendar().as_ist(requested_at).date(),
        None,
        "NONE",
        SpotQuality.UNAVAILABLE,
        reason,
    )


def _live_spot_failure(
    instrument: str,
    requested_as_of: dt.date,
    observed_at: dt.datetime | None,
    source: str,
    quality: SpotQuality,
    reason: str,
    price: float | None = None,
) -> SpotObservation:
    return SpotObservation(
        price,
        requested_as_of,
        observed_at,
        source,
        quality,
        reason,
        None,
        0,
        False,
        SpotFallbackPolicy.NO_FALLBACK,
        instrument,
    )


def _session_distance(observed_on: dt.date, requested_as_of: dt.date) -> int:
    calendar = exchange_calendar()
    calendar.ensure_covered(observed_on)
    calendar.ensure_covered(requested_as_of)
    sessions = 0
    candidate = observed_on + dt.timedelta(days=1)
    while candidate <= requested_as_of:
        if calendar.is_trading_day(candidate):
            sessions += 1
        candidate += dt.timedelta(days=1)
    return sessions


def _date_or_none(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
