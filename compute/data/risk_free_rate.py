"""Real INR risk-free rate provider for option valuation.

Replaces the previous hardcoded `RISK_FREE_RATE = 0.065` constant. Source is
the FBIL (Financial Benchmarks India Pvt Ltd) published T-Bill par-yield
curve -- FBIL is the RBI/FIMMDA-recognised administrator of India's official
money-market benchmarks (see https://www.fbil.org.in), which makes this an
authoritative source, not an arbitrary finance website.

Two public, unauthenticated JSON endpoints were verified live on 2026-09-04:

  GET https://www.fbil.org.in/wasdm/tbill/fetch?authenticated=false
      -> full 14-tenor curve (7 Days .. 12 Months), but only for the most
         recent ~2 published business days. Re-running this fetch on an
         ongoing basis is what accumulates full-curve history going forward.

  GET https://www.fbil.org.in/wasdm/tbill/fetchcommongraph/12M
      -> single fixed tenor ("3 Months") daily series, up to ~241 trading
         days (~12 months) of real lookback. Verified: {period} only takes
         {1M, 3M, 6M, 12M} and always returns the 3-month tenor; it is a
         lookback-window selector, not a tenor selector -- confirmed by
         requesting all four periods and observing an identical
         productName="T-Bill Rate - 3 Months" in every response.

No endpoint offering an explicit historical date-range query, multi-tenor
deep history, or CSV/bulk export was found reachable without an
authenticated FBIL account. This is a real, disclosed coverage boundary,
not a code defect: dates/tenors outside what these two endpoints return are
RATE_UNAVAILABLE, not estimated.

RBI DBIE (data.rbi.org.in/DBIE) was also investigated as a candidate primary
source (91/182/364-day T-Bill auction yields, genuinely free and historically
deep) but its actual report delivery goes through an SAP BusinessObjects
OpenDocument viewer requiring a live authenticated session/token per report
open -- reproduced live: clicking through to a report opened a token-bearing
`BOE/OpenDocument/opendoc/openDocument.jsp` URL, not a stable API or file.
Not usable as a scriptable production data source without embedding a
persistent browser-automation dependency in the analytics backend, which
was judged disproportionate to a rate-lookup call. Kept as a documented,
unimplemented secondary candidate, not silently dropped.
"""

from __future__ import annotations

import datetime as dt
import re
import math
from dataclasses import dataclass
from typing import Any

import requests


FBIL_TBILL_LATEST_URL = "https://www.fbil.org.in/wasdm/tbill/fetch?authenticated=false"
FBIL_TBILL_3M_HISTORY_URL = "https://www.fbil.org.in/wasdm/tbill/fetchcommongraph/12M"
FBIL_SOURCE_ID = "FBIL_TBILL_CURVE"
FBIL_QUOTE_CONVENTION = "SIMPLE_ACT_365_YIELD"
OPTION_RATE_CONVENTION = "CONTINUOUS_ACT_365"
RATE_CONVERSION_METHOD = "LN_1_PLUS_SIMPLE_TIMES_T_OVER_T"

# Deterministic, bounded nearest-tenor matching policy (spec: "deterministic,
# bounded, tested, no extrapolation without explicit policy"). A requested
# maturity is matched to the nearest tenor actually observed on that date
# only if the gap is <= this many days; otherwise the rate is
# RATE_UNAVAILABLE for that observation rather than silently stretched.
# Chosen because adjacent published tenors are never more than 30 days apart
# (7, 14, 30, 60, 90, 120, ... 360), so this always resolves within one
# adjacent-tenor step on a full-curve date, while still refusing to apply
# the single 3-month history point to a materially different maturity.
MAX_TENOR_GAP_DAYS = 30
MAX_RATE_AGE_DAYS = 4

_TENOR_LABEL_RE = re.compile(r"^\s*(\d+)\s*(day|month)s?\s*$", re.IGNORECASE)


class RateProvenanceError(ValueError):
    """Raised when a rate record cannot be trusted / parsed safely."""


@dataclass(frozen=True)
class TenorPoint:
    observation_date: dt.date
    tenor_days: int
    tenor_label: str
    rate_pct: float
    published_at: dt.datetime | None
    source: str
    source_url: str
    retrieved_at: dt.datetime | None = None
    quote_convention: str = FBIL_QUOTE_CONVENTION

    @property
    def rate_decimal(self) -> float:
        """Continuously compounded zero rate preserving published tenor DF.

        RBI money-market T-Bill yield uses ACT/365 simple yield.  For tenor T,
        P/F = 1/(1+y*T); Black/Black-Scholes needs exp(-r*T).  Therefore
        r=log(1+y*T)/T.  Raw `rate_pct` remains unchanged for provenance.
        """
        tenor_years = self.tenor_days / 365.0
        simple = self.rate_pct / 100.0
        return math.log1p(simple * tenor_years) / tenor_years


@dataclass(frozen=True)
class RateSelection:
    observation_date: dt.date
    requested_days: int
    selected_tenor_days: int
    selected_tenor_label: str
    rate_decimal: float
    rate_pct: float
    gap_days: int
    method: str
    source: str
    source_url: str
    published_at: dt.datetime | None = None
    retrieved_at: dt.datetime | None = None
    quote_convention: str = FBIL_QUOTE_CONVENTION
    rate_convention: str = OPTION_RATE_CONVENTION
    conversion_method: str = RATE_CONVERSION_METHOD


def parse_tenor_label(label: str) -> int:
    """'7 Days' -> 7, '3 Months' -> 90 (1 month treated as 30 calendar days
    for tenor-bucket matching purposes only -- a documented matching
    convention, not a claim of calendar precision)."""
    match = _TENOR_LABEL_RE.match(str(label or ""))
    if not match:
        raise RateProvenanceError(f"unrecognised FBIL tenor label: {label!r}")
    count = int(match.group(1))
    unit = match.group(2).lower()
    if count <= 0 or (unit == "month" and count > 12):
        raise RateProvenanceError("unsupported tenor")
    return count if unit == "day" else 365 if count == 12 else count * 30


def _parse_fbil_datetime(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))
        ).astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _parse_fbil_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        pass
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def fetch_fbil_tbill_latest(
    *,
    session: requests.Session | None = None,
    timeout: float = 15.0,
) -> list[TenorPoint]:
    """Full multi-tenor curve for the most recent ~2 published business
    days. Raises RateProvenanceError on any malformed/unexpected response
    rather than returning a partial or guessed result."""
    client = session or requests
    retrieved_at = dt.datetime.now(dt.timezone.utc)
    response = client.get(FBIL_TBILL_LATEST_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RateProvenanceError("FBIL T-Bill latest-curve response was empty or malformed")
    points: list[TenorPoint] = []
    for row in payload:
        observation_date = _parse_fbil_date(row.get("processRunDate"))
        rate_pct = row.get("rate")
        tenor_label = row.get("tenorName")
        if observation_date is None or rate_pct is None or tenor_label is None:
            raise RateProvenanceError(f"malformed FBIL T-Bill row: {row!r}")
        rate_value = float(rate_pct)
        if not (0.0 < rate_value < 50.0):
            raise RateProvenanceError(f"implausible FBIL T-Bill rate rejected: {row!r}")
        points.append(
            TenorPoint(
                observation_date=observation_date,
                tenor_days=parse_tenor_label(tenor_label),
                tenor_label=str(tenor_label).strip(),
                rate_pct=rate_value,
                published_at=_parse_fbil_datetime(row.get("displayTime")),
                source=FBIL_SOURCE_ID,
                source_url=FBIL_TBILL_LATEST_URL,
                retrieved_at=retrieved_at,
            )
        )
    return points


def fetch_fbil_tbill_3m_history(
    *,
    session: requests.Session | None = None,
    timeout: float = 15.0,
) -> list[TenorPoint]:
    """Single fixed 3-month-tenor daily series, ~12 months of real lookback."""
    client = session or requests
    retrieved_at = dt.datetime.now(dt.timezone.utc)
    response = client.get(FBIL_TBILL_3M_HISTORY_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    trend_series = (payload or {}).get("benchMarkTrendData") or []
    if not trend_series:
        raise RateProvenanceError("FBIL T-Bill 3M-history response was empty or malformed")
    product_name = str(trend_series[0].get("productName") or "")
    if "3 Months" not in product_name:
        raise RateProvenanceError(
            f"FBIL 3M-history endpoint returned unexpected tenor product: {product_name!r}"
        )
    trend = trend_series[0].get("benchMarkTrend") or []
    points: list[TenorPoint] = []
    for row in trend:
        observation_date = _parse_fbil_date(row.get("publishDate"))
        rate_pct = row.get("rate")
        if observation_date is None or rate_pct is None:
            raise RateProvenanceError(f"malformed FBIL T-Bill 3M-history row: {row!r}")
        rate_value = float(rate_pct)
        if not (0.0 < rate_value < 50.0):
            raise RateProvenanceError(f"implausible FBIL T-Bill 3M rate rejected: {row!r}")
        points.append(
            TenorPoint(
                observation_date=observation_date,
                tenor_days=90,
                tenor_label="3 Months",
                rate_pct=rate_value,
                published_at=None,
                source=FBIL_SOURCE_ID,
                source_url=FBIL_TBILL_3M_HISTORY_URL,
                retrieved_at=retrieved_at,
            )
        )
    return points


def select_rate_for_tenor(
    curve_for_date: list[TenorPoint],
    requested_days: int,
    *,
    max_gap_days: int = MAX_TENOR_GAP_DAYS,
) -> RateSelection | None:
    """Deterministic nearest-neighbor tenor match within one observation
    date's curve. Returns None (RATE_UNAVAILABLE) rather than extrapolating
    when nothing is close enough, or when the curve for that date is empty."""
    if not curve_for_date or requested_days <= 0 or max_gap_days < 0:
        return None
    if len({point.observation_date for point in curve_for_date}) != 1:
        return None
    if any(
        point.source != FBIL_SOURCE_ID
        or point.source_url not in {FBIL_TBILL_LATEST_URL, FBIL_TBILL_3M_HISTORY_URL}
        or point.retrieved_at is None
        or point.quote_convention != FBIL_QUOTE_CONVENTION
        or point.tenor_days <= 0 or not math.isfinite(point.rate_pct)
        or not 0 < point.rate_pct < 50
        or point.retrieved_at < dt.datetime.combine(
            point.observation_date,
            dt.time.min,
            dt.timezone(dt.timedelta(hours=5, minutes=30)),
        ).astimezone(dt.timezone.utc)
        or (
            point.published_at is not None
            and point.retrieved_at < point.published_at
        )
        for point in curve_for_date
    ):
        return None
    if len({point.tenor_days for point in curve_for_date}) != len(curve_for_date):
        return None
    best = min(curve_for_date, key=lambda point: (abs(point.tenor_days - requested_days), point.tenor_days))
    gap = abs(best.tenor_days - requested_days)
    if gap > max_gap_days:
        return None
    return RateSelection(
        observation_date=best.observation_date,
        requested_days=requested_days,
        selected_tenor_days=best.tenor_days,
        selected_tenor_label=best.tenor_label,
        rate_decimal=best.rate_decimal,
        rate_pct=best.rate_pct,
        gap_days=gap,
        method="NEAREST_NEIGHBOR",
        source=best.source,
        source_url=best.source_url,
        published_at=best.published_at,
        retrieved_at=best.retrieved_at,
        quote_convention=best.quote_convention,
    )


def time_to_expiry_years_to_days(time_to_expiry_years: float) -> int:
    if not math.isfinite(time_to_expiry_years) or time_to_expiry_years <= 0:
        return 0
    return max(1, round(time_to_expiry_years * 365.0))


# Raw FBIL yields remain preserved as published. `TenorPoint.rate_decimal`
# converts RBI's documented simple ACT/365 money-market yield to an equivalent
# continuously compounded zero rate at the published tenor. No cache crosses
# an as-of boundary; failures remain explicit None.
def latest_available_rate(
    time_to_expiry_years: float,
    *,
    as_of: dt.date | dt.datetime | None = None,
    lookback_days: int = MAX_RATE_AGE_DAYS,
) -> RateSelection | None:
    from db.write_service import ReadOperation, resolve_read
    from options.iv_contract import IST, timestamp
    if time_to_expiry_years_to_days(time_to_expiry_years) <= 0 or not 0 <= lookback_days <= MAX_RATE_AGE_DAYS:
        return None
    now = dt.datetime.now(dt.timezone.utc)
    if isinstance(as_of, dt.datetime):
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            return None
        cutoff = as_of.astimezone(dt.timezone.utc)
        reference_date = as_of.astimezone(IST).date()
    elif isinstance(as_of, dt.date):
        reference_date = as_of
        # Date-only cutoff means start-of-day; never grant same-day hindsight.
        cutoff = dt.datetime.combine(reference_date, dt.time.min, IST).astimezone(dt.timezone.utc)
    else:
        cutoff = now
        reference_date = now.astimezone(IST).date()
    try:
        result = resolve_read(ReadOperation.LOAD_RISK_FREE_RATE_CURVE, {
            "from_date": (reference_date - dt.timedelta(days=lookback_days)).isoformat(),
            "to_date": reference_date.isoformat(),
        })
        points = []
        for row in result.get('rows', []):
            published = timestamp(row[7]) if len(row) > 7 else None
            retrieved = timestamp(row[8]) if len(row) > 8 else None
            quote_convention = str(row[9]) if len(row) > 9 else ""
            rate_convention = str(row[10]) if len(row) > 10 else ""
            if quote_convention != FBIL_QUOTE_CONVENTION or rate_convention != OPTION_RATE_CONVENTION:
                continue
            # Same-day curve cannot be used before its actual publication.
            row_date = dt.date.fromisoformat(str(row[0]))
            if row_date == reference_date and (published is None or published > cutoff):
                continue
            if row_date > reference_date or retrieved is None or retrieved > cutoff:
                continue
            # Never trust the read layer's date-range filter alone: verify
            # staleness locally so a caching/query bug cannot silently admit
            # a rate older than the caller's own lookback_days bound.
            if (reference_date - row_date).days > lookback_days:
                continue
            points.append(TenorPoint(
                observation_date=row_date, tenor_days=int(row[1]),
                tenor_label=str(row[2]), rate_pct=float(row[3]), published_at=published,
                source=str(row[5]), source_url=str(row[6]), retrieved_at=retrieved,
                quote_convention=quote_convention,
            ))
        if not points:
            return None
        latest = max(point.observation_date for point in points)
        selection = select_rate_for_tenor([p for p in points if p.observation_date == latest],
            time_to_expiry_years_to_days(time_to_expiry_years))
        return selection
    except (ValueError, TypeError, KeyError, RuntimeError):
        return None


def refresh_risk_free_rate_curve(*, include_deep_history: bool = True) -> dict[str, Any]:
    """Fetch real curve data from FBIL and persist it via the canonical
    WriterService write path. Intended to be called periodically (e.g. once
    per day, similar to the existing instrument-master refresh) by whichever
    process owns write authority; safe to call repeatedly (idempotent
    upsert on (observation_date, tenor_days))."""
    from db.write_service import WriteOperation, submit_write

    points: list[TenorPoint] = list(fetch_fbil_tbill_latest())
    if include_deep_history:
        points.extend(fetch_fbil_tbill_3m_history())

    unique = {}
    for point in points:
        unique.setdefault((point.observation_date, point.tenor_days), point)
    rows = [
        {
            "observation_date": point.observation_date.isoformat(),
            "tenor_days": point.tenor_days,
            "tenor_label": point.tenor_label,
            "rate_pct": point.rate_pct,
            "rate_decimal": point.rate_decimal,
            "source": point.source,
            "source_url": point.source_url,
            "published_at": point.published_at.isoformat() if point.published_at else None,
            "retrieved_at": point.retrieved_at.isoformat() if point.retrieved_at else dt.datetime.now(dt.timezone.utc).isoformat(),
            "quote_convention": point.quote_convention,
            "rate_convention": OPTION_RATE_CONVENTION,
            "conversion_method": RATE_CONVERSION_METHOD,
        }
        for point in unique.values()
    ]
    upserted = 0
    for start in range(0, len(rows), 500):
        batch = rows[start : start + 500]
        result = submit_write(WriteOperation.UPSERT_RISK_FREE_RATE_CURVE, {"rows": batch})
        upserted += int(result["upserted"])
    return {"fetched": len(points), "upserted": upserted}
