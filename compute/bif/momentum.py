from __future__ import annotations

import datetime as _dt
import math
from typing import Any

from data.exchange_calendar import today_ist


RETURN_WEIGHTS = {
    "1d": 0.05,
    "3d": 0.15,
    "5d": 0.25,
    "10d": 0.30,
    "20d": 0.25,
}


def compute_bms(
    ohlcv_rows: list[dict[str, Any]],
    option_signals: dict[str, Any],
    market_signals: dict[str, Any],
    expiry: str,
    ml_prediction: dict[str, Any],
    flow_history: dict[str, list[float]],
    signal_values: dict[str, float],
) -> dict[str, Any]:
    """
    Bharat Momentum Score.

    Formula follows BHARAT_INTELLIGENCE_FRAMEWORK.md:
    BMS = (PM * 0.30 + OFM * 0.40 + IFM * 0.30) * EA * RM.
    """
    closes = [_finite(row.get("close")) for row in ohlcv_rows]
    closes = [value for value in closes if value is not None and value > 0]
    component_returns = _component_returns(closes)
    price_ready = all(component_returns.get(key) is not None for key in RETURN_WEIGHTS)
    weighted_return = (
        sum(
            float(component_returns[key]) / 100.0 * weight
            for key, weight in RETURN_WEIGHTS.items()
        )
        if price_ready
        else None
    )
    price_momentum = (
        math.tanh(weighted_return * 10.0) * 100.0
        if weighted_return is not None
        else None
    )

    pcr = _first_number(option_signals, signal_values, "pcr", "options.pcr_total")
    pcr_trend = _first_number(option_signals, signal_values, "pcr_trend", "options.pcr_trend_5")
    pcr_change = (pcr - pcr_trend) if pcr is not None and pcr_trend is not None else None
    gex = _first_number(option_signals, signal_values, "gex", "options.gex")
    gex_direction = (
        1.0 if gex is not None and gex > 0
        else -1.0 if gex is not None and gex < 0
        else 0.0 if gex == 0
        else None
    )
    iv_rank = _first_number(option_signals, signal_values, "iv_rank", "options.iv_rank")
    iv_inv = ((100.0 - iv_rank) / 50.0 - 1.0) if iv_rank is not None else None
    skew_direction = _first_number(
        option_signals,
        signal_values,
        "iv_skew_direction",
        "options.iv_skew_direction",
    )
    skew_direction = (
        1.0 if skew_direction is not None and skew_direction > 0
        else -1.0 if skew_direction is not None and skew_direction < 0
        else 0.0 if skew_direction == 0
        else None
    )
    option_parts = {
        "pcr_change": (math.tanh(pcr_change * 5.0) * 100.0 if pcr_change is not None else None),
        "gex_direction": (gex_direction * 100.0 if gex_direction is not None else None),
        "iv_rank_inverse": (iv_inv * 100.0 if iv_inv is not None else None),
        "skew_direction": (skew_direction * 100.0 if skew_direction is not None else None),
    }
    option_weights = {
        "pcr_change": 0.30,
        "gex_direction": 0.25,
        "iv_rank_inverse": 0.25,
        "skew_direction": 0.20,
    }
    available_option_parts = {
        key: value for key, value in option_parts.items() if value is not None
    }
    option_weight = sum(option_weights[key] for key in available_option_parts)
    options_flow = (
        sum(
            float(value) * option_weights[key]
            for key, value in available_option_parts.items()
        ) / option_weight
        if option_weight >= 0.50
        else None
    )

    fii_net = _first_available(
        _flow_sum(flow_history.get("fii_net_cash")),
        _latest_flow(market_signals, "fii_net_cash"),
    )
    dii_net = _first_available(
        _flow_sum(flow_history.get("dii_net_cash")),
        _latest_flow(market_signals, "dii_net_cash"),
    )
    fii_norm = math.tanh(fii_net / 10_000.0) * 100.0 if fii_net is not None else None
    dii_norm = math.tanh(dii_net / 10_000.0) * 100.0 if dii_net is not None else None
    institutional_parts = {
        "fii": (fii_norm, 0.60),
        "dii": (dii_norm, 0.40),
    }
    institutional_weight = sum(
        weight for value, weight in institutional_parts.values() if value is not None
    )
    institutional = (
        sum(
            float(value) * weight
            for value, weight in institutional_parts.values()
            if value is not None
        ) / institutional_weight
        if institutional_weight > 0
        else None
    )

    expiry_adjustment = _expiry_adjustment(expiry)
    regime_multiplier = _regime_multiplier(ml_prediction)
    missing_inputs = [
        name
        for name, value in {
            "price_momentum_20d": price_momentum,
            "options_flow": options_flow,
            "institutional_flow": institutional,
            "valid_expiry": expiry_adjustment,
            "validated_vol_regime": regime_multiplier,
        }.items()
        if value is None
    ]
    is_valid = not missing_inputs
    bms = (
        _clamp(
            (
                float(price_momentum) * 0.30
                + float(options_flow) * 0.40
                + float(institutional) * 0.30
            ) * float(expiry_adjustment) * float(regime_multiplier),
            -100.0,
            100.0,
        )
        if is_valid
        else None
    )

    return {
        "bms": round(bms, 3) if bms is not None else None,
        "bms_label": _label(bms) if bms is not None else None,
        "price_momentum": (
            round(_clamp(price_momentum, -100.0, 100.0), 3)
            if price_momentum is not None else None
        ),
        "options_flow_momentum": (
            round(_clamp(options_flow, -100.0, 100.0), 3)
            if options_flow is not None else None
        ),
        "institutional_momentum": (
            round(_clamp(institutional, -100.0, 100.0), 3)
            if institutional is not None else None
        ),
        "expiry_adjustment": expiry_adjustment,
        "regime_multiplier": regime_multiplier,
        "component_returns": component_returns,
        "option_component_coverage": round(option_weight, 4),
        "option_components": option_parts,
        "missing_inputs": missing_inputs,
        "reason_code": None if is_valid else "MISSING_REQUIRED_INPUTS",
        "is_valid": is_valid,
    }


def _component_returns(closes: list[float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if not closes:
        return {key: None for key in RETURN_WEIGHTS}
    last_close = closes[-1]
    for key in RETURN_WEIGHTS:
        days = int(key[:-1])
        if len(closes) <= days or closes[-days - 1] <= 0:
            result[key] = None
            continue
        result[key] = round((last_close / closes[-days - 1] - 1.0) * 100.0, 4)
    return result


def _expiry_adjustment(expiry: str) -> float | None:
    dte = _days_to_expiry(expiry)
    if dte is None:
        return None
    if dte > 5:
        return 1.0
    if dte >= 3:
        return 0.7
    return 0.4


def _days_to_expiry(expiry: str) -> int | None:
    parsed = _parse_expiry(expiry)
    if parsed is None:
        return None
    return max((parsed - today_ist()).days, 0)


def _parse_expiry(expiry: str) -> _dt.date | None:
    clean = str(expiry or "").strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(clean, fmt).date()
        except ValueError:
            pass
    return None


def _regime_multiplier(ml_prediction: dict[str, Any]) -> float | None:
    regime = None
    components = ml_prediction.get("component_outputs")
    if isinstance(components, dict):
        vol = components.get("m01_vol_regime")
        if isinstance(vol, dict) and vol.get("is_trained") is True:
            regime = vol.get("regime") or vol.get("prediction")
    if regime is None:
        regime = ml_prediction.get("vol_regime") or ml_prediction.get("regime")

    text = str(regime or "").upper()
    if "LOW" in text:
        return 1.2
    if "HIGH" in text:
        return 0.7
    if "MEDIUM" in text:
        return 1.0
    return None


def _latest_flow(market_signals: dict[str, Any], key: str) -> float | None:
    fii_dii = market_signals.get("fii_dii")
    if not isinstance(fii_dii, dict):
        return None
    return _finite(fii_dii.get(key))


def _flow_sum(values: list[float] | None) -> float | None:
    if not values:
        return None
    clean = [number for value in values[-5:] if (number := _finite(value)) is not None]
    return float(sum(clean)) if clean else None


def _first_available(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _first_number(
    option_signals: dict[str, Any],
    signal_values: dict[str, float],
    *keys: str,
) -> float | None:
    for key in keys:
        value = _finite(option_signals.get(key))
        if value is not None:
            return value
        value = _finite(signal_values.get(key))
        if value is not None:
            return value
    return None


def _label(value: float) -> str:
    if value > 60:
        return "STRONG BULL"
    if value > 30:
        return "BULL"
    if value > 10:
        return "MILD BULL"
    if value > -10:
        return "NEUTRAL"
    if value > -30:
        return "MILD BEAR"
    if value > -60:
        return "BEAR"
    return "STRONG BEAR"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
