from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import _sharpe


def chronological_segment_summary(
    daily_records: list[dict[str, Any]],
    segments: int = 4,
) -> dict[str, Any]:
    """Describe fixed-strategy chronology; this is not parameter/model walk-forward."""
    equity = np.array(
        [float(row.get("portfolio_value") or 0.0) for row in daily_records],
        dtype=np.float64,
    )
    dates = [str(row.get("date")) for row in daily_records]
    assumptions = {
        "parameters_refit": False,
        "model_retrained": False,
        "untouched_future_evaluation": False,
        "interpretation": "fixed-strategy chronological segment comparison only",
    }
    if equity.size < 20:
        return {
            "segments": [],
            "historical_avg_return": 0.0,
            "subsequent_avg_return": 0.0,
            "historical_sharpe": 0.0,
            "subsequent_sharpe": 0.0,
            "difference_pct": 0.0,
            "verdict": "INSUFFICIENT_DATA",
            "assumptions": assumptions,
        }

    segment_count = max(2, min(int(segments), 8))
    segment_size = max(equity.size // segment_count, 1)
    rows: list[dict[str, Any]] = []
    historical_returns: list[float] = []
    subsequent_returns: list[float] = []
    historical_sharpes: list[float] = []
    subsequent_sharpes: list[float] = []

    for index in range(1, segment_count):
        history_end = index * segment_size
        subsequent_start = history_end
        subsequent_end = (index + 1) * segment_size if index < segment_count - 1 else equity.size
        if subsequent_end - subsequent_start < 2 or history_end < 2:
            continue
        history_equity = equity[:history_end]
        subsequent_equity = equity[subsequent_start:subsequent_end]
        history_return = _segment_return(history_equity)
        subsequent_return = _segment_return(subsequent_equity)
        history_sharpe = _sharpe(_returns(history_equity))
        subsequent_sharpe = _sharpe(_returns(subsequent_equity))
        historical_returns.append(history_return)
        subsequent_returns.append(subsequent_return)
        historical_sharpes.append(history_sharpe)
        subsequent_sharpes.append(subsequent_sharpe)
        rows.append(
            {
                "segment": index,
                "history_start": dates[0],
                "history_end": dates[history_end - 1],
                "subsequent_start": dates[subsequent_start],
                "subsequent_end": dates[subsequent_end - 1],
                "historical_return_pct": round(history_return, 3),
                "subsequent_return_pct": round(subsequent_return, 3),
                "historical_sharpe": round(history_sharpe, 3),
                "subsequent_sharpe": round(subsequent_sharpe, 3),
            }
        )

    historical_avg = float(np.mean(historical_returns)) if historical_returns else 0.0
    subsequent_avg = float(np.mean(subsequent_returns)) if subsequent_returns else 0.0
    historical_sharpe_avg = float(np.mean(historical_sharpes)) if historical_sharpes else 0.0
    subsequent_sharpe_avg = float(np.mean(subsequent_sharpes)) if subsequent_sharpes else 0.0
    difference = subsequent_sharpe_avg - historical_sharpe_avg
    return {
        "segments": rows,
        "historical_avg_return": round(historical_avg, 3),
        "subsequent_avg_return": round(subsequent_avg, 3),
        "historical_sharpe": round(historical_sharpe_avg, 3),
        "subsequent_sharpe": round(subsequent_sharpe_avg, 3),
        "difference_pct": round(difference, 3),
        "verdict": "DESCRIPTIVE_ONLY",
        "assumptions": assumptions,
    }


def _segment_return(equity: np.ndarray) -> float:
    if equity.size < 2 or equity[0] == 0:
        return 0.0
    return float((equity[-1] / equity[0] - 1.0) * 100.0)


def _returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([], dtype=np.float64)
    previous = equity[:-1]
    current = equity[1:]
    return np.divide(
        current - previous,
        previous,
        out=np.zeros_like(current),
        where=previous != 0,
    )
