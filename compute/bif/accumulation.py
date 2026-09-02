from __future__ import annotations

import math
from typing import Any


def compute_iad(
    ohlcv_rows: list[dict[str, Any]],
    chain_legs: list[dict[str, Any]],
    flow_history: dict[str, list[float]],
    market_signals: dict[str, Any],
) -> dict[str, Any]:
    """
    Institutional Accumulation Detector.

    Uses price-volume behavior, gradual FII flow, and option hedge build-up.
    """
    closes = [_finite(row.get("close")) for row in ohlcv_rows]
    volumes = [_finite(row.get("volume")) for row in ohlcv_rows]
    close_values = [value for value in closes if value is not None and value > 0]
    volume_values = [value for value in volumes if value is not None and value > 0]

    pv_score = _price_volume_score(close_values, volume_values)
    fii_score = _fii_flow_score(flow_history.get("fii_net_cash"), market_signals)
    options_score = _options_hedge_score(chain_legs, close_values)
    delivery_score = None

    weights = {
        "price_volume": 0.30,
        "fii_flow": 0.35,
        "options_hedge": 0.25,
        "delivery": 0.10,
    }
    components = {
        "price_volume": pv_score,
        "fii_flow": fii_score,
        "options_hedge": options_score,
        "delivery": delivery_score,
    }
    available = {key: value for key, value in components.items() if value is not None}
    available_weight = sum(weights[key] for key in available)
    coverage = available_weight / sum(weights.values())
    required_available = pv_score is not None and (
        fii_score is not None or options_score is not None
    )
    is_valid = required_available and coverage >= 0.55
    iad_score = (
        _clamp(
            sum(float(value) * weights[key] for key, value in available.items())
            / available_weight,
            -100.0,
            100.0,
        )
        if is_valid and available_weight > 0
        else None
    )
    price_20d = _return_pct(close_values, 20)

    return {
        "iad_score": round(iad_score, 3) if iad_score is not None else None,
        "iad_label": _label(iad_score) if iad_score is not None else None,
        "wyckoff_stage": (
            _wyckoff_stage(iad_score, price_20d) if iad_score is not None else None
        ),
        "price_volume_score": round(pv_score, 3) if pv_score is not None else None,
        "fii_flow_score": round(fii_score, 3) if fii_score is not None else None,
        "options_hedge_score": round(options_score, 3) if options_score is not None else None,
        "delivery_score": delivery_score,
        "component_coverage": round(coverage, 4),
        "available_components": sorted(available),
        "missing_components": sorted(set(components) - set(available)),
        "weight_normalization": "renormalized_available_components",
        "reason_code": None if is_valid else "INSUFFICIENT_COMPONENT_COVERAGE",
        "is_valid": is_valid,
    }


def _price_volume_score(closes: list[float], volumes: list[float]) -> float | None:
    if len(closes) < 12 or len(volumes) < 12:
        return None

    scores: list[float] = []
    start = max(1, len(closes) - 10)
    for index in range(start, len(closes)):
        previous_close = closes[index - 1]
        if previous_close <= 0:
            continue
        price_change_pct = (closes[index] / previous_close - 1.0) * 100.0
        avg_start = max(0, index - 20)
        volume_window = volumes[avg_start:index]
        avg_volume = sum(volume_window) / len(volume_window) if volume_window else volumes[index]
        if avg_volume <= 0:
            continue
        volume_ratio = volumes[index] / avg_volume
        daily_score = 0.0
        if price_change_pct <= 0.2 and volume_ratio > 1.0:
            daily_score += volume_ratio * 20.0
        if price_change_pct >= -0.2 and volume_ratio > 1.0:
            daily_score -= volume_ratio * 20.0
        scores.append(_clamp(daily_score, -100.0, 100.0))

    if not scores:
        return None
    return _clamp(sum(scores) / len(scores), -100.0, 100.0)


def _fii_flow_score(
    values: list[float] | None,
    market_signals: dict[str, Any],
) -> float | None:
    history = []
    for value in values or []:
        number = _finite(value)
        if number is not None:
            history.append(number)
    if len(history) >= 3:
        recent = history[-10:]
        today = recent[-1]
        avg = sum(recent) / len(recent)
        if abs(avg) < 1e-9:
            consistency = 0.3
            direction = 1.0 if today > 0 else -1.0 if today < 0 else 0.0
        else:
            consistency = 1.0 - abs(today - avg) / abs(avg)
            consistency = _clamp(consistency, 0.0, 1.0)
            direction = 1.0 if avg > 0 else -1.0
        return _clamp(direction * consistency * 100.0, -100.0, 100.0)

    fii_dii = market_signals.get("fii_dii")
    latest = None
    if isinstance(fii_dii, dict):
        latest = _finite(fii_dii.get("fii_net_cash"))
    if latest is None:
        return None
    if latest == 0:
        return 0.0
    return 50.0 if latest > 0 else -50.0


def _options_hedge_score(
    chain_legs: list[dict[str, Any]],
    closes: list[float],
) -> float | None:
    if not chain_legs:
        return None

    put_oi_change = 0.0
    put_legs = 0
    for leg in chain_legs:
        option_type = str(leg.get("option_type") or "").strip().upper()
        if option_type not in {"PE", "PUT", "P"}:
            continue
        put_legs += 1
        change = _finite(leg.get("oi_change"))
        if change is None:
            return None
        put_oi_change += change

    if put_legs == 0:
        return None

    price_change = _return_pct(closes, 1)
    stable_price = price_change is not None and abs(price_change) <= 0.5
    if put_oi_change > 0 and stable_price:
        return 50.0
    if put_oi_change < 0:
        return -30.0
    return 0.0


def _return_pct(closes: list[float], days: int) -> float | None:
    if len(closes) <= days or closes[-days - 1] <= 0:
        return None
    return (closes[-1] / closes[-days - 1] - 1.0) * 100.0


def _wyckoff_stage(iad_score: float, price_20d: float | None) -> str:
    price_20d = price_20d if price_20d is not None else 0.0
    if iad_score > 40 and price_20d < 2:
        return "ACCUMULATION"
    if iad_score > 20 and price_20d > 2:
        return "MARKUP"
    if iad_score < -40 and price_20d > -2:
        return "DISTRIBUTION"
    if iad_score < -20 and price_20d < -2:
        return "MARKDOWN"
    return "TRANSITION"


def _label(value: float) -> str:
    if value > 60:
        return "STRONG ACCUMULATION"
    if value > 30:
        return "ACCUMULATION"
    if value > 10:
        return "MILD ACCUMULATION"
    if value > -10:
        return "NEUTRAL"
    if value > -30:
        return "MILD DISTRIBUTION"
    if value > -60:
        return "DISTRIBUTION"
    return "STRONG DISTRIBUTION"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
