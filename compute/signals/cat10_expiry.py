from __future__ import annotations

import datetime as _dt
from typing import Any

from data.exchange_calendar import now_ist

from .common import dte, last_tuesday_of_month, next_budget_day, next_tuesday, results_season_week
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "expiry"
RBI_DATES_2026 = [
    _dt.date(2026, 2, 6),
    _dt.date(2026, 4, 3),
    _dt.date(2026, 6, 5),
    _dt.date(2026, 8, 6),
    _dt.date(2026, 10, 1),
    _dt.date(2026, 12, 4),
]

SIGNAL_NAMES = [
    "dte_weekly",
    "dte_monthly",
    "expiry_week_flag",
    "expiry_day_flag",
    "theta_acceleration_zone",
    "monthly_vs_weekly_flag",
    "rbi_policy_dte",
    "budget_dte",
    "results_season_flag",
    "results_season_week",
    "rollover_pct_near_month",
    "rollover_cost_annualised",
    "rollover_direction",
    "near_month_oi_change",
    "next_month_oi_change",
    "expiry_day_avg_return",
    "expiry_week_avg_return",
    "pre_expiry_bias",
    "circuit_breaker_distance_10pct",
    "circuit_breaker_distance_15pct",
    "dte_bucket",
    "theta_pressure_score",
    "calendar_event_risk_score",
    "expiry_pin_window_flag",
    "weekly_monthly_overlap_risk",
]

TYPE_OVERRIDES = {
    "expiry_week_flag": SignalType.BINARY,
    "expiry_day_flag": SignalType.BINARY,
    "theta_acceleration_zone": SignalType.BINARY,
    "monthly_vs_weekly_flag": SignalType.BINARY,
    "results_season_flag": SignalType.BINARY,
    "dte_bucket": SignalType.CATEGORICAL,
    "expiry_pin_window_flag": SignalType.BINARY,
    "weekly_monthly_overlap_risk": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    "rollover_pct_near_month": DataSource.FO_EOD,
    "rollover_cost_annualised": DataSource.FO_EOD,
    "rollover_direction": DataSource.FO_EOD,
    "near_month_oi_change": DataSource.FO_EOD,
    "next_month_oi_change": DataSource.FO_EOD,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.MACRO,
    type_overrides=TYPE_OVERRIDES,
    source_overrides=SOURCE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values = {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx))


def _compute_values(ctx: SignalContext) -> dict[str, Any]:
    today = now_ist().date()
    weekly_dte = dte(ctx.expiry_str, today)
    monthly_expiry = last_tuesday_of_month(today)
    if monthly_expiry < today:
        next_month = (today.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
        monthly_expiry = last_tuesday_of_month(next_month)
    monthly_dte = (monthly_expiry - today).days
    rbi = _next_rbi(today)
    result_week = results_season_week(today)
    spot = ctx.spot
    values = {
        "dte_weekly": weekly_dte,
        "dte_monthly": monthly_dte,
        "expiry_week_flag": int((weekly_dte or 99) <= 6),
        "expiry_day_flag": int(weekly_dte == 0),
        "theta_acceleration_zone": int((weekly_dte or 99) < 3),
        "monthly_vs_weekly_flag": int(monthly_dte == weekly_dte),
        "rbi_policy_dte": (rbi - today).days if rbi is not None else None,
        "budget_dte": (next_budget_day(today) - today).days,
        "results_season_flag": int(result_week is not None),
        "results_season_week": result_week,
        "expiry_day_avg_return": None,
        "expiry_week_avg_return": None,
        "pre_expiry_bias": None,
        "circuit_breaker_distance_10pct": spot * 0.10 if spot > 0 else None,
        "circuit_breaker_distance_15pct": spot * 0.15 if spot > 0 else None,
        "dte_bucket": _bucket(weekly_dte),
        "theta_pressure_score": _theta_score(weekly_dte),
        "calendar_event_risk_score": _calendar_risk((rbi - today).days if rbi is not None else None, result_week),
        "expiry_pin_window_flag": int((weekly_dte or 99) <= 2),
        "weekly_monthly_overlap_risk": int(monthly_dte == weekly_dte and (weekly_dte or 99) <= 6),
    }
    return values


def _next_rbi(today: _dt.date) -> _dt.date | None:
    for day in RBI_DATES_2026:
        if day >= today:
            return day
    return None


def _bucket(dte_value: int | None) -> str | None:
    if dte_value is None:
        return None
    if dte_value == 0:
        return "expiry_day"
    if dte_value < 3:
        return "theta_acceleration"
    if dte_value < 7:
        return "expiry_week"
    return "normal"


def _theta_score(dte_value: int | None) -> float | None:
    if dte_value is None:
        return None
    return max(0.0, min(100.0, 100.0 - dte_value * 15.0))


def _calendar_risk(rbi_dte: int | None, results_week: int | None) -> float:
    score = 0.0
    if rbi_dte is not None and rbi_dte <= 7:
        score += 50.0
    if results_week is not None:
        score += 25.0
    return min(100.0, score)
