from __future__ import annotations

import datetime as dt
from dataclasses import replace
import importlib
from pathlib import Path
import sys

import pytest

from data.exchange_calendar import (
    CalendarCoverageError,
    SessionMetadata,
    SessionType,
    exchange_calendar,
    next_expiries_for_symbol,
    next_expiry_labels_for_symbol,
)


def test_shared_contract_fixtures() -> None:
    calendar = exchange_calendar()
    for fixture in calendar.fixtures:
        instant = dt.datetime.fromisoformat(fixture["at_utc"].replace("Z", "+00:00"))
        assert calendar.session_state(instant) == fixture["session"], fixture["name"]
        expiry = calendar.next_weekly_expiries(1, instant)[0]
        assert expiry.strftime("%d-%b-%Y") == fixture["next_weekly_expiry"], fixture["name"]


def test_holiday_adjusts_expiry_to_previous_trading_day() -> None:
    calendar = exchange_calendar()
    assert calendar.adjusted_expiry(dt.date(2026, 3, 31)) == dt.date(2026, 3, 30)


def test_monthly_expiry_handles_month_and_year_end() -> None:
    calendar = exchange_calendar()
    assert calendar.monthly_expiry(2026, 3) == dt.date(2026, 3, 30)
    assert calendar.monthly_expiry(2026, 12) == dt.date(2026, 12, 29)


def test_non_nifty_symbols_resolve_monthly_expiry_not_weekly() -> None:
    """NSE permits one true-weekly index contract per exchange; NIFTY holds that
    slot. BANKNIFTY/FINNIFTY/MIDCPNIFTY and single-stock F&O are monthly-only, so
    resolving their nearest expiry off the weekly cadence looks up a contract date
    that was never listed (regression for lot-size/chain/signal 503s on non-NIFTY
    symbols while NIFTY resolved fine)."""
    instant = dt.datetime(2026, 9, 3, 6, 0, tzinfo=dt.timezone.utc)
    nifty_expiry = next_expiries_for_symbol("NIFTY", 1, instant)[0]
    assert nifty_expiry == dt.date(2026, 9, 8)

    for symbol in ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "RELIANCE", "SWIGGY"):
        expiry = next_expiries_for_symbol(symbol, 1, instant)[0]
        assert expiry == dt.date(2026, 9, 29), symbol
        assert expiry != nifty_expiry, symbol


def test_next_expiry_labels_for_symbol_formats_monthly_cadence() -> None:
    instant = dt.datetime(2026, 9, 3, 6, 0, tzinfo=dt.timezone.utc)
    assert next_expiry_labels_for_symbol("BANKNIFTY", 1, instant) == ["29-Sep-2026"]
    assert next_expiry_labels_for_symbol("NIFTY", 1, instant) == ["08-Sep-2026"]


def test_monthly_only_symbol_rolls_to_next_month_after_expiry_cutoff() -> None:
    instant = dt.datetime(2026, 9, 29, 12, 0, tzinfo=dt.timezone.utc)
    expiries = next_expiries_for_symbol("BANKNIFTY", 2, instant)
    assert expiries[0] == dt.date(2026, 10, 27)
    # 24-Nov-2026 is an NSE holiday; expiry rolls back to the previous trading day.
    assert expiries[1] == dt.date(2026, 11, 23)


def test_timezone_awareness_is_mandatory() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        exchange_calendar().as_ist(dt.datetime(2026, 8, 18, 10, 0))


def test_metadata_fails_closed_outside_verified_year() -> None:
    with pytest.raises(CalendarCoverageError, match="does not cover"):
        exchange_calendar().is_trading_day(dt.date(2027, 1, 1))


def test_leap_year_date_math_remains_gregorian() -> None:
    leap_day = dt.date(2028, 2, 29)
    assert leap_day.weekday() == 1


def test_muhurat_session_is_metadata_driven_not_weekend_closed() -> None:
    calendar = exchange_calendar()
    day = dt.date(2026, 11, 8)
    session = calendar.session(day)
    assert calendar.is_trading_day(day) is True
    assert session.session_type.value == "MUHURAT"
    assert session.special_session is True
    assert session.open_time is None and session.close_time is None
    instant = dt.datetime(2026, 11, 8, 6, 0, tzinfo=dt.timezone.utc)
    snapshot = calendar.session_snapshot(instant)
    assert snapshot["current_state"] == "SCHEDULE_PENDING"
    assert snapshot["calendar_version"] == "NSE_FO_2026.20260823"
    with pytest.raises(CalendarCoverageError, match="timing pending"):
        calendar.next_session_change(instant)
    with pytest.raises(CalendarCoverageError, match="timing pending"):
        calendar.next_trading_open(instant)


def test_special_session_times_drive_transitions_and_expiry_cutoff() -> None:
    base = exchange_calendar()
    day = dt.date(2026, 8, 18)
    special = SessionMetadata(
        day,
        SessionType.SPECIAL,
        dt.time(18, 0),
        dt.time(19, 0),
        True,
        "VERIFIED",
    )
    calendar = replace(base, special_sessions={**base.special_sessions, day: special})
    before = dt.datetime(2026, 8, 18, 12, 0, tzinfo=dt.timezone.utc)
    during = dt.datetime(2026, 8, 18, 12, 45, tzinfo=dt.timezone.utc)
    after = dt.datetime(2026, 8, 18, 13, 30, tzinfo=dt.timezone.utc)

    assert calendar.session_state(before) == "PRE_OPEN"
    assert calendar.next_session_change(before).time() == dt.time(18, 0)
    assert calendar.session_state(during) == "OPEN"
    assert calendar.next_session_change(during).time() == dt.time(19, 0)
    assert calendar.session_state(after) == "CLOSED"
    assert calendar.next_weekly_expiries(1, during) == [day]
    assert calendar.next_weekly_expiries(1, after) == [dt.date(2026, 8, 25)]


def test_frontend_contains_no_exchange_session_inference() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "src/components/layout/StockUniverse.tsx",
        "src/components/layout/NiftyDetail.tsx",
        "src/components/Screener/index.tsx",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "isMarketOpenNow" not in source
        assert "formatToParts" not in source
        assert "useMarketSession" in source
    app_source = (root / "src/App.tsx").read_text(encoding="utf-8")
    for forbidden in ("getDay(", "T15:30:00", "T09:15:00"):
        assert forbidden not in app_source


def test_frozen_calendar_path_uses_explicit_bundle_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("data.exchange_calendar")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert module._calendar_path() == tmp_path / "contracts" / "exchange-calendar.json"
