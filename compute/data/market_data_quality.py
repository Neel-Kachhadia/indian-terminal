from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class MarketDataQuality(StrEnum):
    OBSERVED_VERIFIED = "OBSERVED_VERIFIED"
    DERIVED_VERIFIED = "DERIVED_VERIFIED"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class DerivationType(StrEnum):
    NONE = "NONE"
    SINGLE_INSTRUMENT = "SINGLE_INSTRUMENT"
    CROSS_EXPIRY_FUTURES = "CROSS_EXPIRY_FUTURES"
    LEGACY_UNKNOWN = "LEGACY_UNKNOWN"


class DecisionContext(StrEnum):
    HISTORICAL_RESEARCH = "HISTORICAL_RESEARCH"
    BACKTEST_SIGNAL = "BACKTEST_SIGNAL"
    BACKTEST_EXECUTION = "BACKTEST_EXECUTION"
    CURRENT_DECISION = "CURRENT_DECISION"
    REFERENCE_DISPLAY = "REFERENCE_DISPLAY"


VERIFIED_INDEX_SOURCES = frozenset({"NSE_INDEX_BHAVCOPY"})
VERIFIED_EQUITY_SOURCES = frozenset({"NSE_EQUITY_BHAVCOPY"})
INVALID_SPOT_SOURCES = frozenset({"UNKNOWN", "FO_EOD_FUTIDX_DERIVED", "LEGACY_UNVERIFIED"})
INDEX_SYMBOLS = frozenset(
    {"NIFTY", "NIFTY 50", "BANKNIFTY", "NIFTY BANK", "BANK NIFTY", "FINNIFTY", "MIDCPNIFTY"}
)

# NSE's own ind_close_all_*.csv "Index Name" column uses full display names that
# don't match the short trading-symbol aliases above (e.g. "Nifty Financial
# Services" for the FINNIFTY futures underlying, "Nifty Midcap Select" for
# MIDCPNIFTY). This maps the real NSE display name (uppercased) to the
# canonical trading symbol so index source rows get labeled correctly instead
# of silently dropped for not matching INDEX_SYMBOLS directly.
NSE_INDEX_DISPLAY_NAME_TO_CANONICAL_SYMBOL = {
    "NIFTY FINANCIAL SERVICES": "FINNIFTY",
    "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
    # NSE rebranded its "CNX"-prefixed index family to "Nifty" branding.
    # Verified via real downloaded ind_close_all files, not name similarity:
    # 2015-11-06 (Fri, last CNX day) "CNX Nifty" close=7954.3 / "CNX Bank"
    # close=17086.5 -> 2015-11-09 (Mon, first Nifty day) "Nifty 50" open=7788.25
    # / "Nifty Bank" open=16667.9 -- ordinary ~2% weekend gap, same price
    # series, not a discontinuity. Same underlying index, renamed in place.
    "CNX NIFTY": "NIFTY 50",
    "S&P CNX NIFTY": "NIFTY 50",
    "CNX BANK": "NIFTY BANK",
}


def canonical_index_symbol(raw_index_name: str) -> str:
    """Resolve a raw NSE index-file 'Index Name' value to its canonical trading symbol."""
    upper = str(raw_index_name or "").strip().upper()
    return NSE_INDEX_DISPLAY_NAME_TO_CANONICAL_SYMBOL.get(upper, upper)

DECISION_ELIGIBLE_OHLCV_VIEW_SQL = """
CREATE OR REPLACE VIEW decision_eligible_ohlcv AS
SELECT
    symbol,
    date,
    CAST(date AS TIMESTAMP) AS observed_at,
    symbol AS instrument,
    open,
    high,
    low,
    close,
    volume,
    upper(source) AS source,
    'OBSERVED_VERIFIED'::TEXT AS quality,
    'NONE'::TEXT AS derivation_type,
    TRUE AS decision_eligible
FROM ohlcv
WHERE date <= CURRENT_DATE
  AND open > 0 AND high > 0 AND low > 0 AND close > 0
  AND high >= greatest(open, low, close)
  AND low <= least(open, high, close)
  AND volume >= 0
  AND (
    (
        upper(symbol) IN (
            'NIFTY', 'NIFTY 50', 'BANKNIFTY', 'NIFTY BANK',
            'BANK NIFTY', 'FINNIFTY', 'MIDCPNIFTY'
        )
        AND upper(source) = 'NSE_INDEX_BHAVCOPY'
    )
    OR (
        upper(symbol) NOT IN (
            'NIFTY', 'NIFTY 50', 'BANKNIFTY', 'NIFTY BANK',
            'BANK NIFTY', 'FINNIFTY', 'MIDCPNIFTY'
        )
        AND upper(source) = 'NSE_EQUITY_BHAVCOPY'
    )
  )
"""


@dataclass(frozen=True)
class MarketBar:
    instrument: str
    observed_at: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    quality: MarketDataQuality
    derivation_type: DerivationType
    decision_eligible: bool


@dataclass(frozen=True)
class MarketDataEligibility:
    context: DecisionContext
    requested_as_of: dt.datetime
    expected_observation_date: dt.date | None
    actual_observed_at: dt.datetime | None
    source: str
    quality: MarketDataQuality
    semantic_quality_eligible: bool
    temporal_eligible: bool
    decision_eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.value,
            "requested_as_of": self.requested_as_of.isoformat(),
            "expected_observation_date": (
                self.expected_observation_date.isoformat()
                if self.expected_observation_date is not None
                else None
            ),
            "actual_observed_at": (
                self.actual_observed_at.isoformat() if self.actual_observed_at is not None else None
            ),
            "source": self.source,
            "quality": self.quality.value,
            "semantic_quality_eligible": self.semantic_quality_eligible,
            "temporal_eligible": self.temporal_eligible,
            "decision_eligible": self.decision_eligible,
            "reasons": list(self.reasons),
        }


def expected_sources(instrument: str) -> frozenset[str]:
    normalized = str(instrument or "").strip().upper()
    return VERIFIED_INDEX_SOURCES if normalized in INDEX_SYMBOLS else VERIFIED_EQUITY_SOURCES


def classify_market_bar(
    row: Mapping[str, Any],
    *,
    requested_as_of: dt.date | None = None,
) -> MarketBar:
    instrument = str(row.get("instrument") or row.get("symbol") or "").strip().upper()
    source = str(row.get("source") or "UNKNOWN").strip().upper()
    derivation = _derivation(row.get("derivation_type"), source)
    observed_at = _observed_at(row.get("observed_at") or row.get("observed_date") or row.get("date"))
    values = [_finite(row.get(name)) for name in ("open", "high", "low", "close")]
    volume = _integer(row.get("volume"))
    structurally_valid = (
        observed_at is not None
        and all(value is not None and value > 0 for value in values)
        and values[1] >= max(values[0], values[2], values[3])
        and values[2] <= min(values[0], values[1], values[3])
        and volume is not None
        and volume >= 0
    )
    future = bool(requested_as_of and observed_at and observed_at.date() > requested_as_of)
    eligible = (
        structurally_valid
        and not future
        and source in expected_sources(instrument)
        and derivation is DerivationType.NONE
    )
    if eligible:
        quality = MarketDataQuality.OBSERVED_VERIFIED
    elif source in INVALID_SPOT_SOURCES or derivation in {
        DerivationType.CROSS_EXPIRY_FUTURES,
        DerivationType.LEGACY_UNKNOWN,
    }:
        quality = MarketDataQuality.UNKNOWN
    else:
        quality = MarketDataQuality.INVALID
    return MarketBar(
        instrument=instrument,
        observed_at=observed_at or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        open=float(values[0] or 0.0),
        high=float(values[1] or 0.0),
        low=float(values[2] or 0.0),
        close=float(values[3] or 0.0),
        volume=int(volume or 0),
        source=source,
        quality=quality,
        derivation_type=derivation,
        decision_eligible=eligible,
    )


def evaluate_market_bar(
    row: Mapping[str, Any] | None,
    *,
    context: DecisionContext,
    requested_as_of: dt.datetime,
) -> MarketDataEligibility:
    """Apply semantic quality and context-specific temporal eligibility centrally."""
    if requested_as_of.tzinfo is None or requested_as_of.utcoffset() is None:
        raise ValueError("requested_as_of must be timezone-aware")
    decision_context = DecisionContext(context)
    if row is None:
        return MarketDataEligibility(
            decision_context,
            requested_as_of,
            _expected_observation_date(decision_context, requested_as_of),
            None,
            "NONE",
            MarketDataQuality.INVALID,
            False,
            False,
            False,
            ("market observation missing",),
        )

    requested_date = _requested_market_date(decision_context, requested_as_of)
    bar = classify_market_bar(row, requested_as_of=requested_date)
    expected = _expected_observation_date(decision_context, requested_as_of)
    reasons: list[str] = []
    semantic = bool(bar.decision_eligible and bar.quality is MarketDataQuality.OBSERVED_VERIFIED)
    if not semantic:
        reasons.append("market observation semantic quality invalid")

    observed_on = bar.observed_at.date() if bar.observed_at != dt.datetime.min.replace(
        tzinfo=dt.timezone.utc
    ) else None
    temporal = False
    if observed_on is None:
        reasons.append("market observation timestamp invalid")
    elif decision_context is DecisionContext.HISTORICAL_RESEARCH:
        temporal = observed_on <= requested_date
    elif decision_context is DecisionContext.REFERENCE_DISPLAY:
        temporal = observed_on <= requested_date
    else:
        temporal = expected is not None and observed_on == expected
    if not temporal:
        reasons.append("market observation outside context freshness requirement")

    eligible = semantic and temporal and decision_context is not DecisionContext.REFERENCE_DISPLAY
    return MarketDataEligibility(
        decision_context,
        requested_as_of,
        expected,
        bar.observed_at if observed_on is not None else None,
        bar.source,
        bar.quality,
        semantic,
        temporal,
        eligible,
        tuple(sorted(set(reasons))),
    )


def _expected_observation_date(
    context: DecisionContext,
    requested_as_of: dt.datetime,
) -> dt.date | None:
    if context in {DecisionContext.HISTORICAL_RESEARCH, DecisionContext.REFERENCE_DISPLAY}:
        return None
    if context in {DecisionContext.BACKTEST_SIGNAL, DecisionContext.BACKTEST_EXECUTION}:
        return requested_as_of.date()

    from data.exchange_calendar import CalendarCoverageError, SessionType, exchange_calendar

    calendar = exchange_calendar()
    local = calendar.as_ist(requested_as_of)
    try:
        session = calendar.session(local.date())
        if (
            session.session_type is not SessionType.HOLIDAY
            and session.close_time is not None
            and local.time().replace(tzinfo=None) >= session.close_time
        ):
            return local.date()
        return calendar.previous_trading_day(local.date() - dt.timedelta(days=1))
    except CalendarCoverageError:
        return None


def _requested_market_date(
    context: DecisionContext,
    requested_as_of: dt.datetime,
) -> dt.date:
    if context is DecisionContext.CURRENT_DECISION:
        from data.exchange_calendar import exchange_calendar

        return exchange_calendar().as_ist(requested_as_of).date()
    return requested_as_of.date()


def _derivation(value: Any, source: str) -> DerivationType:
    text = str(value or "").strip().upper()
    if source == "FO_EOD_FUTIDX_DERIVED" or text == DerivationType.CROSS_EXPIRY_FUTURES:
        return DerivationType.CROSS_EXPIRY_FUTURES
    if not text and source in VERIFIED_INDEX_SOURCES | VERIFIED_EQUITY_SOURCES:
        return DerivationType.NONE
    try:
        return DerivationType(text or DerivationType.LEGACY_UNKNOWN)
    except ValueError:
        return DerivationType.LEGACY_UNKNOWN


def _observed_at(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min, dt.timezone.utc)
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
