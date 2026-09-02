from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from ml.feature_engineering import MLContext

from .apex_common import close_series, return_series, train_unavailable, untrained_prediction


MODEL_ID = "m12_prophet"
MINIMUM_ROWS = 90

EVENTS = [
    ("Union Budget window", (1, 20)),
    ("Budget season", (2, 1)),
    ("RBI MPC window", (4, 8)),
    ("Monsoon onset", (6, 1)),
    ("Quarterly results season", (7, 15)),
    ("NIFTY rebalance window", (9, 30)),
    ("Year-end positioning", (12, 20)),
]


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    return train_unavailable(
        MODEL_ID,
        "prophet training not run; Holt-Winters-style fallback active",
        context,
        minimum_rows=MINIMUM_ROWS,
    )


def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    close = close_series(context)
    if len(close) < MINIMUM_ROWS:
        return untrained_prediction(
            MODEL_ID,
            f"insufficient daily rows: {len(close)}/{MINIMUM_ROWS}",
            context,
            minimum_rows=MINIMUM_ROWS,
            extra={
                "trend_direction": None,
                "trend_strength": None,
                "residual_today": None,
                "residual_z_score": None,
                "regime": None,
                "forecast_7d": None,
                "forecast_30d": None,
                "next_event_days": None,
                "next_event_name": None,
            },
        )

    window = close.tail(90)
    x = np.arange(len(window), dtype=float)
    y = np.log(window.to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = np.exp(intercept + slope * x)
    residual = float(window.iloc[-1] - fitted[-1])
    residual_series = pd.Series(window.to_numpy(dtype=float) - fitted)
    residual_std = float(residual_series.std(ddof=0))
    residual_z = residual / residual_std if residual_std > 1e-12 else 0.0
    daily_trend_pct = (np.exp(slope) - 1.0) * 100.0
    trend_strength = min(1.0, abs(daily_trend_pct) / 0.35)
    trend_direction = "UP" if daily_trend_pct > 0.04 else "DOWN" if daily_trend_pct < -0.04 else "FLAT"
    regime = _regime(trend_direction, residual_z, return_series(context).tail(20).std(ddof=0))
    today = dt.date.today()
    event_name, event_days = _next_event(today)
    forecast_7d = float(window.iloc[-1] * np.exp(slope * 7))
    forecast_30d = float(window.iloc[-1] * np.exp(slope * 30))
    weekly_effect = _weekly_effect(context)
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "READY_ANALYTICAL",
        "reason_code": None,
        "model_used": "holt_winters_fallback",
        "prediction": regime,
        "trend_direction": trend_direction,
        "trend_strength": round(float(trend_strength), 6),
        "weekly_effect_today": (
            round(float(weekly_effect), 6) if weekly_effect is not None else None
        ),
        "yearly_effect_today": None,
        "residual_today": round(float(residual), 6),
        "residual_z_score": round(float(residual_z), 6),
        "regime": regime,
        "forecast_7d": round(forecast_7d, 6),
        "forecast_30d": round(forecast_30d, 6),
        "next_event_days": int(event_days),
        "next_event_name": event_name,
        "confidence": None,
        "confidence_formula": None,
        "reason": "Prophet optional; deterministic trend/seasonality fallback used",
    }


def _regime(trend: str, residual_z: float, vol: float) -> str:
    if abs(residual_z) >= 1.5:
        return "OVERBOUGHT" if residual_z > 0 else "MEAN_REVERSION_SETUP"
    if vol >= 1.2:
        return "VOLATILE_TREND"
    return {"UP": "TREND_UP", "DOWN": "TREND_DOWN"}.get(trend, "RANGE")


def _weekly_effect(context: MLContext) -> float | None:
    returns = return_series(context)
    if returns.empty:
        return None
    today_dow = dt.date.today().weekday()
    by_day = returns.groupby(pd.Series(returns.index).map(lambda item: item.weekday()).to_numpy()).mean()
    try:
        return float(by_day.loc[today_dow])
    except Exception:
        return None


def _next_event(today: dt.date) -> tuple[str, int]:
    candidates: list[tuple[str, dt.date]] = []
    for year in (today.year, today.year + 1):
        for name, (month, day) in EVENTS:
            candidates.append((name, dt.date(year, month, day)))
    future = [(name, day) for name, day in candidates if day >= today]
    name, day = min(future, key=lambda item: item[1])
    return name, (day - today).days
