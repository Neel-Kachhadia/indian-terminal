from __future__ import annotations

import datetime as dt

import numpy as np

from backtest.metrics import _period_returns, _trade_stats


def test_profit_factor_oracle_uses_explicit_gross_and_net_pnl() -> None:
    trades = [
        {"gross_pnl": 120.0, "net_pnl": 100.0, "exit_date": "2026-01-02"},
        {"gross_pnl": -60.0, "net_pnl": -80.0, "exit_date": "2026-01-03"},
        {"gross_pnl": 30.0, "net_pnl": 10.0, "exit_date": "2026-01-04"},
    ]
    metrics = _trade_stats(trades)
    assert metrics["gross_profit_factor"] == 2.5
    assert metrics["net_profit_factor"] == 1.375
    assert metrics["profit_factor"] == metrics["net_profit_factor"]
    assert metrics["net_expectancy"] == 10.0
    assert metrics["net_hit_rate"] == round(2 / 3, 4)


def test_weekly_periods_are_unique_with_multiple_weeks_in_month() -> None:
    rows = _period_returns(
        ["2026-01-02", "2026-01-09", "2026-01-16"],
        np.array([100.0, 101.0, 102.0]),
        "W",
    )
    periods = [row["period"] for row in rows]
    assert len(periods) == len(set(periods)) == 3
    assert all(row["period_type"] == "iso_week" for row in rows)


def test_weekly_period_identity_handles_year_boundary_and_partial_week() -> None:
    rows = _period_returns(
        ["2025-12-31", "2026-01-02", "2026-01-05"],
        np.array([100.0, 101.0, 103.0]),
        "W",
    )
    assert rows[0]["period"] == "2026-W01"
    assert rows[0]["start_date"] == dt.date(2025, 12, 31).isoformat()
    assert rows[0]["end_date"] == dt.date(2026, 1, 2).isoformat()
    assert rows[1]["period"] == "2026-W02"
    assert rows[1]["start_date"] == rows[1]["end_date"] == "2026-01-05"
