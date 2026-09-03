from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from functools import lru_cache
from enum import StrEnum
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


def _calendar_path() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "contracts" / "exchange-calendar.json"
    return Path(__file__).resolve().parents[2] / "contracts" / "exchange-calendar.json"


CALENDAR_PATH = _calendar_path()


class CalendarCoverageError(ValueError):
    """Raised when exchange behavior is requested outside verified metadata."""


class SessionType(StrEnum):
    NORMAL = "NORMAL"
    HOLIDAY = "HOLIDAY"
    SPECIAL = "SPECIAL"
    MUHURAT = "MUHURAT"
    HALF_SESSION = "HALF_SESSION"


@dataclass(frozen=True)
class SessionMetadata:
    date: dt.date
    session_type: SessionType
    open_time: dt.time | None
    close_time: dt.time | None
    special_session: bool
    timing_status: str
    source_url: str | None = None


@dataclass(frozen=True)
class ExchangeCalendar:
    metadata_version: str
    timezone: ZoneInfo
    open_time: dt.time
    close_time: dt.time
    weekly_expiry_weekday: int
    coverage_start: dt.date
    coverage_end: dt.date
    holidays: frozenset[dt.date]
    special_sessions: dict[dt.date, SessionMetadata]
    sources: tuple[dict[str, str], ...]
    fixtures: tuple[dict[str, str], ...]

    def ensure_covered(self, day: dt.date) -> None:
        if not self.coverage_start <= day <= self.coverage_end:
            raise CalendarCoverageError(
                f"NSE calendar metadata does not cover {day.isoformat()} "
                f"({self.coverage_start}..{self.coverage_end})"
            )

    def as_ist(self, instant: dt.datetime | None = None) -> dt.datetime:
        value = instant or dt.datetime.now(dt.timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("exchange timestamps must be timezone-aware")
        return value.astimezone(self.timezone)

    def is_trading_day(self, day: dt.date) -> bool:
        self.ensure_covered(day)
        if day in self.special_sessions:
            return True
        return day.weekday() < 5 and day not in self.holidays

    def session(self, day: dt.date) -> SessionMetadata:
        self.ensure_covered(day)
        special = self.special_sessions.get(day)
        if special is not None:
            return special
        if day.weekday() >= 5 or day in self.holidays:
            return SessionMetadata(
                day,
                SessionType.HOLIDAY,
                None,
                None,
                False,
                "VERIFIED",
            )
        return SessionMetadata(
            day,
            SessionType.NORMAL,
            self.open_time,
            self.close_time,
            False,
            "VERIFIED",
        )

    def session_state(self, instant: dt.datetime | None = None) -> str:
        local = self.as_ist(instant)
        session = self.session(local.date())
        if session.session_type is SessionType.HOLIDAY:
            return "CLOSED"
        if session.open_time is None or session.close_time is None:
            return "SCHEDULE_PENDING"
        if local.time().replace(tzinfo=None) < session.open_time:
            return "PRE_OPEN"
        if local.time().replace(tzinfo=None) < session.close_time:
            return "OPEN"
        return "CLOSED"

    def next_session(self, day: dt.date) -> SessionMetadata:
        candidate = day + dt.timedelta(days=1)
        while True:
            session = self.session(candidate)
            if session.session_type is not SessionType.HOLIDAY:
                return session
            candidate += dt.timedelta(days=1)

    def session_snapshot(self, instant: dt.datetime | None = None) -> dict[str, object]:
        local = self.as_ist(instant)
        current = self.session(local.date())
        next_session = self.next_session(local.date())

        def boundary(day: dt.date, value: dt.time | None) -> str | None:
            return dt.datetime.combine(day, value, self.timezone).isoformat() if value else None

        return {
            "calendar_version": self.metadata_version,
            "exchange": "NSE",
            "session_date": current.date.isoformat(),
            "is_trading_day": current.session_type is not SessionType.HOLIDAY,
            "session_type": current.session_type.value,
            "market_open": boundary(current.date, current.open_time),
            "market_close": boundary(current.date, current.close_time),
            "special_session": current.special_session,
            "timing_status": current.timing_status,
            "current_state": self.session_state(local),
            "next_session": {
                "session_date": next_session.date.isoformat(),
                "session_type": next_session.session_type.value,
                "market_open": boundary(next_session.date, next_session.open_time),
                "market_close": boundary(next_session.date, next_session.close_time),
                "special_session": next_session.special_session,
                "timing_status": next_session.timing_status,
            },
        }

    def next_trading_open(self, instant: dt.datetime | None = None) -> dt.datetime:
        local = self.as_ist(instant)
        candidate = local.date()
        session = self.session(candidate)
        if (
            session.session_type is not SessionType.HOLIDAY
            and (session.open_time is None or session.close_time is None)
        ):
            raise CalendarCoverageError(
                f"NSE session timing pending for {candidate.isoformat()}"
            )
        if (
            session.session_type is not SessionType.HOLIDAY
            and session.open_time is not None
            and local.time().replace(tzinfo=None) < session.open_time
        ):
            return dt.datetime.combine(candidate, session.open_time, self.timezone)
        candidate += dt.timedelta(days=1)
        session = self.session(candidate)
        while session.session_type is SessionType.HOLIDAY:
            candidate += dt.timedelta(days=1)
            session = self.session(candidate)
        if session.open_time is None:
            raise CalendarCoverageError(f"NSE session time pending for {candidate.isoformat()}")
        return dt.datetime.combine(candidate, session.open_time, self.timezone)

    def next_session_change(self, instant: dt.datetime | None = None) -> dt.datetime:
        local = self.as_ist(instant)
        current = self.session(local.date())
        if (
            current.session_type is not SessionType.HOLIDAY
            and (current.open_time is None or current.close_time is None)
        ):
            raise CalendarCoverageError(
                f"NSE session timing pending for {local.date().isoformat()}"
            )
        state = self.session_state(local)
        if state == "PRE_OPEN":
            if current.open_time is None:
                raise CalendarCoverageError(
                    f"NSE session open time pending for {local.date().isoformat()}"
                )
            return dt.datetime.combine(local.date(), current.open_time, self.timezone)
        if state == "OPEN":
            if current.close_time is None:
                raise CalendarCoverageError(
                    f"NSE session close time pending for {local.date().isoformat()}"
                )
            return dt.datetime.combine(local.date(), current.close_time, self.timezone)
        return self.next_trading_open(local)

    def previous_trading_day(self, day: dt.date) -> dt.date:
        candidate = day
        while True:
            self.ensure_covered(candidate)
            if self.is_trading_day(candidate):
                return candidate
            candidate -= dt.timedelta(days=1)

    def adjusted_expiry(self, nominal: dt.date) -> dt.date:
        return self.previous_trading_day(nominal)

    def next_weekly_expiries(
        self,
        count: int = 4,
        instant: dt.datetime | None = None,
    ) -> list[dt.date]:
        if count < 1 or count > 52:
            raise ValueError("expiry count must be between 1 and 52")
        local = self.as_ist(instant)
        today = local.date()
        self.ensure_covered(today)
        days_until = (self.weekly_expiry_weekday - today.weekday()) % 7
        nominal = today + dt.timedelta(days=days_until)
        results: list[dt.date] = []
        while len(results) < count:
            expiry = self.adjusted_expiry(nominal)
            expiry_session = self.session(expiry)
            if expiry_session.close_time is None:
                raise CalendarCoverageError(
                    f"NSE expiry cutoff pending for {expiry.isoformat()}"
                )
            before_cutoff = local.time().replace(tzinfo=None) < expiry_session.close_time
            if expiry > today or (expiry == today and before_cutoff):
                if not results or results[-1] != expiry:
                    results.append(expiry)
            nominal += dt.timedelta(days=7)
            if len(results) < count:
                self.ensure_covered(nominal)
        return results

    def monthly_expiry(self, year: int, month: int) -> dt.date:
        if month == 12:
            next_month = dt.date(year + 1, 1, 1)
        else:
            next_month = dt.date(year, month + 1, 1)
        candidate = next_month - dt.timedelta(days=1)
        while candidate.weekday() != self.weekly_expiry_weekday:
            candidate -= dt.timedelta(days=1)
        return self.adjusted_expiry(candidate)

    def next_monthly_expiries(
        self,
        count: int = 4,
        instant: dt.datetime | None = None,
    ) -> list[dt.date]:
        if count < 1 or count > 52:
            raise ValueError("expiry count must be between 1 and 52")
        local = self.as_ist(instant)
        today = local.date()
        self.ensure_covered(today)
        results: list[dt.date] = []
        year, month = today.year, today.month
        while len(results) < count:
            expiry = self.monthly_expiry(year, month)
            expiry_session = self.session(expiry)
            if expiry_session.close_time is None:
                raise CalendarCoverageError(
                    f"NSE expiry cutoff pending for {expiry.isoformat()}"
                )
            before_cutoff = local.time().replace(tzinfo=None) < expiry_session.close_time
            if expiry > today or (expiry == today and before_cutoff):
                if not results or results[-1] != expiry:
                    results.append(expiry)
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1
            if len(results) < count:
                self.ensure_covered(dt.date(year, month, 1))
        return results


def _parse_time(value: str) -> dt.time:
    return dt.datetime.strptime(value, "%H:%M").time()


@lru_cache(maxsize=1)
def exchange_calendar() -> ExchangeCalendar:
    raw = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 2 or raw.get("exchange") != "NSE":
        raise ValueError("unsupported exchange calendar contract")
    coverage = raw["coverage"]
    session = raw["normal_session"]
    special_sessions = {
        day: SessionMetadata(
            day,
            SessionType(record["session_type"]),
            _parse_time(record["open"]) if record.get("open") else None,
            _parse_time(record["close"]) if record.get("close") else None,
            bool(record.get("special_session")),
            str(record.get("timing_status") or "UNKNOWN"),
            str(record["source_url"]) if record.get("source_url") else None,
        )
        for record in raw.get("special_sessions", [])
        for day in [dt.date.fromisoformat(record["date"])]
    }
    return ExchangeCalendar(
        metadata_version=str(raw["metadata_version"]),
        timezone=ZoneInfo(raw["timezone"]),
        open_time=_parse_time(session["open"]),
        close_time=_parse_time(session["close"]),
        weekly_expiry_weekday=int(raw["weekly_expiry_weekday"]),
        coverage_start=dt.date.fromisoformat(coverage["start"]),
        coverage_end=dt.date.fromisoformat(coverage["end"]),
        holidays=frozenset(dt.date.fromisoformat(value) for value in raw["holidays"]),
        special_sessions=special_sessions,
        sources=tuple(raw.get("sources", [])),
        fixtures=tuple(raw.get("contract_fixtures", [])),
    )


def now_ist(instant: dt.datetime | None = None) -> dt.datetime:
    return exchange_calendar().as_ist(instant)


def today_ist(instant: dt.datetime | None = None) -> dt.date:
    return now_ist(instant).date()


def is_trading_day(day: dt.date) -> bool:
    return exchange_calendar().is_trading_day(day)


def is_trading_day_candidate(day: dt.date) -> bool:
    """Exact within verified coverage; weekday candidate outside it for archival fetches."""
    calendar = exchange_calendar()
    if calendar.coverage_start <= day <= calendar.coverage_end:
        return calendar.is_trading_day(day)
    return day.weekday() < 5


def next_trading_day(day: dt.date) -> dt.date:
    calendar = exchange_calendar()
    candidate = day + dt.timedelta(days=1)
    while not calendar.is_trading_day(candidate):
        candidate += dt.timedelta(days=1)
    return candidate


def next_weekly_expiries(
    count: int = 4,
    instant: dt.datetime | None = None,
) -> list[dt.date]:
    return exchange_calendar().next_weekly_expiries(count, instant)


def next_expiry_labels(
    count: int = 4,
    instant: dt.datetime | None = None,
) -> list[str]:
    return [value.strftime("%d-%b-%Y") for value in next_weekly_expiries(count, instant)]


# NSE permits exactly one true-weekly index-options contract per exchange (SEBI's
# "rationalisation of expiry days" framework); NIFTY holds that slot on NSE.
# BANKNIFTY/FINNIFTY/MIDCPNIFTY and all single-stock F&O are monthly-only, so
# resolving their nearest expiry via the weekly cadence looks up a contract date
# that was never listed (see get_lot_size/get_chain exact-expiry lookups).
_WEEKLY_EXPIRY_SYMBOLS = frozenset({"NIFTY"})


def is_weekly_expiry_symbol(symbol: str) -> bool:
    return symbol.strip().upper() in _WEEKLY_EXPIRY_SYMBOLS


def next_expiries_for_symbol(
    symbol: str,
    count: int = 4,
    instant: dt.datetime | None = None,
) -> list[dt.date]:
    calendar = exchange_calendar()
    if is_weekly_expiry_symbol(symbol):
        return calendar.next_weekly_expiries(count, instant)
    return calendar.next_monthly_expiries(count, instant)


def next_expiry_labels_for_symbol(
    symbol: str,
    count: int = 4,
    instant: dt.datetime | None = None,
) -> list[str]:
    return [
        value.strftime("%d-%b-%Y")
        for value in next_expiries_for_symbol(symbol, count, instant)
    ]
