from __future__ import annotations

from typing import Any


STRESS_WINDOWS = [
    ("Election 2024", "2024-06-03", "2024-06-07"),
    ("Budget 2024", "2024-07-22", "2024-07-26"),
    ("Lok Sabha 2024 Result Day", "2024-06-04", "2024-06-04"),
    ("June 2026 Expiry Vol", "2026-06-01", "2026-06-19"),
]


def stress_test(daily_records: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    equity_by_date = {
        str(row.get("date")): float(row.get("portfolio_value") or 0.0)
        for row in daily_records
    }
    scenarios = []
    for name, start, end in STRESS_WINDOWS:
        points = [
            (date, value)
            for date, value in equity_by_date.items()
            if start <= date <= end and value > 0
        ]
        if len(points) < 2:
            scenarios.append(
                {
                    "scenario": name,
                    "start": start,
                    "end": end,
                    "return_pct": None,
                    "max_drawdown_pct": None,
                    "trades": _trade_count(trades, start, end),
                    "status": "NO_OVERLAP",
                }
            )
            continue
        values = [value for _, value in points]
        high = values[0]
        max_dd = 0.0
        for value in values:
            high = max(high, value)
            if high > 0:
                max_dd = min(max_dd, value / high - 1.0)
        scenarios.append(
            {
                "scenario": name,
                "start": points[0][0],
                "end": points[-1][0],
                "return_pct": round((values[-1] / values[0] - 1.0) * 100.0, 3) if values[0] else 0.0,
                "max_drawdown_pct": round(max_dd * 100.0, 3),
                "trades": _trade_count(trades, start, end),
                "status": "OK",
            }
        )
    worst = min(
        [row for row in scenarios if row["return_pct"] is not None],
        key=lambda row: float(row["return_pct"]),
        default=None,
    )
    return {
        "scenarios": scenarios,
        "worst_scenario": worst,
        "verdict": "STRESS_ACCEPTABLE" if not worst or float(worst["return_pct"]) > -10 else "STRESS_FRAGILE",
    }


def _trade_count(trades: list[dict[str, Any]], start: str, end: str) -> int:
    return sum(1 for trade in trades if start <= str(trade.get("entry_date") or "") <= end)
