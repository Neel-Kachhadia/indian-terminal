from __future__ import annotations

from typing import Any

import numpy as np

from ml.feature_engineering import MLContext, current_value


MODEL_ID = "m07_iv_forecast"
INPUT_DATASET = "iv_history"
REQUIRED_INPUTS = ("iv_history.atm_iv",)
MINIMUM_ROWS = 60


def train_model(context: MLContext, force: bool = False) -> dict[str, Any]:
    del force
    rows = len(context.iv_history)
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": rows >= MINIMUM_ROWS,
        "status": "READY_ANALYTICAL" if rows >= MINIMUM_ROWS else "INSUFFICIENT_DATA",
        "reason_code": None if rows >= MINIMUM_ROWS else "INSUFFICIENT_IV_HISTORY",
        "train_rows": rows,
        "model_used": "holt_winters" if rows >= MINIMUM_ROWS else None,
    }

def predict(context: MLContext, confidence_multiplier: float = 1.0) -> dict[str, Any]:
    del confidence_multiplier
    if len(context.iv_history) < MINIMUM_ROWS:
        current_iv = current_value(context, ["atm_iv", "options.atm_iv"])
        return {
            "model_id": MODEL_ID,
            "is_trained": False,
            "prediction": None,
            "iv_forecast_1d": None,
            "iv_forecast_3d": None,
            "iv_forecast_5d": None,
            "confidence_interval_1d": None,
            "model_used": None,
            "mape_30d": None,
            "available_rows": len(context.iv_history),
            "minimum_rows": MINIMUM_ROWS,
            "reason": "iv_history needs 60 rows for sequence forecast",
            "status": "INSUFFICIENT_DATA",
            "reason_code": "INSUFFICIENT_IV_HISTORY",
            "is_valid": False,
            "current_atm_iv": current_iv,
        }
    try:
        return _holt_winters(context)
    except Exception as error:
        return {
            "model_id": MODEL_ID,
            "is_trained": False,
            "prediction": None,
            "iv_forecast_1d": None,
            "iv_forecast_3d": None,
            "iv_forecast_5d": None,
            "confidence_interval_1d": None,
            "model_used": None,
            "mape_30d": None,
            "available_rows": len(context.iv_history),
            "minimum_rows": MINIMUM_ROWS,
            "reason": str(error),
            "status": "FAILED",
            "reason_code": "SEQUENCE_FORECAST_FAILED",
            "is_valid": False,
        }


def _holt_winters(context: MLContext) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = context.iv_history.sort_values("date")["atm_iv"].astype(float)
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=5,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(5)
    residual = np.asarray(fit.resid, dtype=float)
    sigma = float(np.nanstd(residual[-30:])) if residual.size else 0.0
    one_day = float(forecast.iloc[0])
    lower = max(0.0, one_day - 1.2816 * sigma)
    upper = one_day + 1.2816 * sigma
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    actual = np.asarray(series, dtype=float)
    if len(actual) >= 30 and len(fitted) == len(actual):
        denom = np.where(np.abs(actual[-30:]) < 1e-9, np.nan, np.abs(actual[-30:]))
        mape = float(np.nanmean(np.abs((actual[-30:] - fitted[-30:]) / denom)) * 100.0)
    else:
        mape = None
    return {
        "model_id": MODEL_ID,
        "is_trained": False,
        "is_valid": True,
        "status": "READY_ANALYTICAL",
        "reason_code": None,
        "prediction": round(float(forecast.iloc[2]), 6),
        "iv_forecast_1d": round(one_day, 6),
        "iv_forecast_3d": round(float(forecast.iloc[2]), 6),
        "iv_forecast_5d": round(float(forecast.iloc[4]), 6),
        "confidence_interval_1d": [round(lower, 6), round(upper, 6)],
        "model_used": "holt_winters",
        "mape_30d": round(mape, 6) if mape is not None else None,
        "confidence": None,
        "confidence_formula": None,
    }
