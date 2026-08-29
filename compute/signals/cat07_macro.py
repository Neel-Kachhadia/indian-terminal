from __future__ import annotations

import datetime as _dt
from typing import Any

from db.write_service import ReadOperation, resolve_read
from data.exchange_calendar import now_ist

from .common import next_budget_day, next_tuesday, results_season_week
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "macro"
REPO_RATE = 6.25
REVERSE_REPO_RATE = 6.00
RBI_POLICY_DATES_2026 = [
    _dt.date(2026, 2, 6),
    _dt.date(2026, 4, 3),
    _dt.date(2026, 6, 5),
    _dt.date(2026, 8, 6),
    _dt.date(2026, 10, 1),
    _dt.date(2026, 12, 4),
]

SIGNAL_NAMES = [
    "rbi_days_to_next_policy",
    "budget_days_to_next",
    "results_season_active",
    "nifty_expiry_week",
    "fno_ban_list_count",
    "cpi_latest",
    "cpi_vs_rbi_target",
    "wpi_latest",
    "iip_latest",
    "gdp_growth_latest",
    "pmi_manufacturing",
    "pmi_services",
    "gst_collection_crores",
    "trade_deficit_usd_bn",
    "current_account_deficit_pct_gdp",
    "forex_reserves_usd_bn",
    "bank_credit_growth_pct",
    "mclr_1y",
    "repo_rate",
    "reverse_repo_rate",
    "inflation_vs_repo",
    "monsoon_progress_pct",
    "real_policy_rate",
    "policy_restrictiveness_score",
    "macro_risk_score_0_100",
]

TYPE_OVERRIDES = {
    "results_season_active": SignalType.BINARY,
    "nifty_expiry_week": SignalType.BINARY,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.MACRO,
    type_overrides=TYPE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values = _compute_values()
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values = {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx))


def _compute_values() -> dict[str, Any]:
    today = now_ist().date()
    macro = _macro_table_values()
    cpi = macro.get("cpi_latest")
    values = {
        "rbi_days_to_next_policy": _days_to_next_policy(today),
        "budget_days_to_next": (next_budget_day(today) - today).days,
        "results_season_active": int(results_season_week(today) is not None),
        "nifty_expiry_week": int((next_tuesday(today) - today).days <= 6),
        "repo_rate": REPO_RATE,
        "reverse_repo_rate": REVERSE_REPO_RATE,
        "cpi_latest": cpi,
        "cpi_vs_rbi_target": cpi - 4.0 if cpi is not None else None,
        "inflation_vs_repo": cpi - REPO_RATE if cpi is not None else None,
        "real_policy_rate": REPO_RATE - cpi if cpi is not None else None,
        "policy_restrictiveness_score": max(0.0, min(100.0, 50.0 + (REPO_RATE - 6.0) * 20.0)),
    }
    for key in (
        "wpi_latest",
        "iip_latest",
        "gdp_growth_latest",
        "pmi_manufacturing",
        "pmi_services",
        "gst_collection_crores",
        "trade_deficit_usd_bn",
        "current_account_deficit_pct_gdp",
        "forex_reserves_usd_bn",
        "bank_credit_growth_pct",
        "mclr_1y",
        "monsoon_progress_pct",
        "fno_ban_list_count",
    ):
        values[key] = macro.get(key)
    values["macro_risk_score_0_100"] = _macro_risk(values)
    return values


def _days_to_next_policy(today: _dt.date) -> int | None:
    for day in RBI_POLICY_DATES_2026:
        if day >= today:
            return (day - today).days
    return None


def _macro_table_values() -> dict[str, float | None]:
    try:
        result = resolve_read(ReadOperation.LOAD_MACRO_TABLE_LATEST_VALUES, {})
        rows = result["rows"]
    except Exception as error:
        print(f"signals: macro table query failed: {error}")
        return {}
    return {str(metric): _float(value) for metric, value in rows}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _macro_risk(values: dict[str, Any]) -> float | None:
    pieces = [values.get("policy_restrictiveness_score")]
    cpi_gap = values.get("cpi_vs_rbi_target")
    if cpi_gap is not None:
        pieces.append(max(0.0, min(100.0, 50.0 + float(cpi_gap) * 10.0)))
    clean = [float(piece) for piece in pieces if piece is not None]
    return sum(clean) / len(clean) if clean else None
