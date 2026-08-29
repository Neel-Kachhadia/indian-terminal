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
    weighted_return = sum(
        (component_returns.get(key) or 0.0) / 100.0 * weight
        for key, weight in RETURN_WEIGHTS.items()
    )
    price_momentum = math.tanh(weighted_return * 10.0) * 100.0

    pcr = _first_number(option_signals, signal_values, "pcr", "options.pcr_total")
    pcr_trend = _first_number(option_signals, signal_values, "pcr_trend", "options.pcr_trend_5")
    pcr_change = (pcr - pcr_trend) if pcr is not None and pcr_trend is not None else 0.0
    gex = _first_number(option_signals, signal_values, "gex", "options.gex")
    gex_direction = 1.0 if (gex or 0.0) > 0 else -1.0 if (gex or 0.0) < 0 else 0.0
    iv_rank = _first_number(option_signals, signal_values, "iv_rank", "options.iv_rank")
    iv_inv = ((100.0 - iv_rank) / 50.0 - 1.0) if iv_rank is not None else 0.0
    skew_direction = _first_number(
        option_signals,
        signal_values,
        "iv_skew_direction",
        "options.iv_skew_direction",
    ) or 0.0
    skew_direction = 1.0 if skew_direction > 0 else -1.0 if skew_direction < 0 else 0.0
    options_flow = (
        math.tanh(pcr_change * 5.0) * 30.0
        + gex_direction * 25.0
        + iv_inv * 25.0
        + skew_direction * 20.0
    )

    fii_net = _flow_sum(flow_history.get("fii_net_cash")) or _latest_flow(market_signals, "fii_net_cash")
    dii_net = _flow_sum(flow_history.get("dii_net_cash")) or _latest_flow(market_signals, "dii_net_cash")
    fii_norm = math.tanh(fii_net / 10_000.0) * 100.0
    dii_norm = math.tanh(dii_net / 10_000.0) * 100.0
    institutional = fii_norm * 0.60 + dii_norm * 0.40

    expiry_adjustment = _expiry_adjustment(expiry)
    regime_multiplier = _regime_multiplier(ml_prediction)
    raw_bms = (
        price_momentum * 0.30
        + options_flow * 0.40
        + institutional * 0.30
    ) * expiry_adjustment * regime_multiplier
    bms = _clamp(raw_bms, -100.0, 100.0)

    return {
        "bms": round(bms, 3),
        "bms_label": _label(bms),
        "price_momentum": round(_clamp(price_momentum, -100.0, 100.0), 3),
        "options_flow_momentum": round(_clamp(options_flow, -100.0, 100.0), 3),
        "institutional_momentum": round(_clamp(institutional, -100.0, 100.0), 3),
        "expiry_adjustment": expiry_adjustment,
        "regime_multiplier": regime_multiplier,
        "component_returns": component_returns,
        "is_valid": len(closes) >= 21,
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


def _expiry_adjustment(expiry: str) -> float:
    dte = _days_to_expiry(expiry)
    if dte is None:
        return 1.0
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


def _regime_multiplier(ml_prediction: dict[str, Any]) -> float:
    regime = None
    components = ml_prediction.get("component_outputs")
    if isinstance(components, dict):
        vol = components.get("m01_vol_regime")
        if isinstance(vol, dict):
            regime = vol.get("regime") or vol.get("prediction")
    if regime is None:
        regime = ml_prediction.get("vol_regime") or ml_prediction.get("regime")

    text = str(regime or "").upper()
    if "LOW" in text:
        return 1.2
    if "HIGH" in text:
        return 0.7
    return 1.0


def _latest_flow(market_signals: dict[str, Any], key: str) -> float:
    fii_dii = market_signals.get("fii_dii")
    if not isinstance(fii_dii, dict):
        return 0.0
    return _finite(fii_dii.get(key)) or 0.0


def _flow_sum(values: list[float] | None) -> float:
    if not values:
        return 0.0
    return float(sum(values[-5:]))


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
