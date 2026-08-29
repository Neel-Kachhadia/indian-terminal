from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_signals(daily_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in daily_records:
        signals = record.get("signals")
        if not isinstance(signals, dict):
            signals = {}
        row = {"date": record.get("date"), "portfolio_value": record.get("portfolio_value")}
        for key, value in signals.items():
            number = _number(value)
            if number is not None:
                row[key] = number
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty or "portfolio_value" not in frame:
        return {"signals_tested": 0, "top_ic": [], "mean_abs_ic": 0.0}

    frame["forward_return"] = frame["portfolio_value"].pct_change().shift(-1)
    rows_out = []
    for column in frame.columns:
        if column in {"date", "portfolio_value", "forward_return"}:
            continue
        pair = frame[[column, "forward_return"]].dropna()
        if len(pair) < 20 or pair[column].nunique() < 3:
            continue
        ic = pair[column].corr(pair["forward_return"], method="spearman")
        if ic != ic:
            continue
        rows_out.append(
            {
                "signal": column,
                "ic": round(float(ic), 4),
                "abs_ic": round(abs(float(ic)), 4),
                "observations": int(len(pair)),
                "direction": "positive" if ic >= 0 else "negative",
            }
        )
    rows_out.sort(key=lambda row: row["abs_ic"], reverse=True)
    mean_abs = float(np.mean([row["abs_ic"] for row in rows_out])) if rows_out else 0.0
    return {
        "signals_tested": len(rows_out),
        "top_ic": rows_out[:12],
        "mean_abs_ic": round(mean_abs, 4),
        "verdict": "USEFUL" if mean_abs >= 0.03 else "WEAK_OR_INSUFFICIENT",
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
