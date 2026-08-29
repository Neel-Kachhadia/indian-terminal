from __future__ import annotations

from typing import Any

from .common import safe_div
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "cross_asset"

SIGNAL_NAMES = [
    "usdinr_rate",
    "usdinr_change_pct_today",
    "usdinr_direction",
    "usdinr_vs_20d_avg_pct",
    "usdinr_regime",
    "brent_price",
    "brent_change_pct_today",
    "brent_regime_impact",
    "gold_price",
    "gold_change_pct_today",
    "gold_safe_haven_signal",
    "dow_futures_change_pct",
    "sp500_futures_change_pct",
    "nasdaq_futures_change_pct",
    "us_market_bias",
    "us_10y_yield",
    "us_10y_change_bps",
    "us_10y_regime",
    "us_2y_yield",
    "yield_curve_slope",
    "nikkei_change_pct",
    "hangseng_change_pct",
    "asia_market_bias",
    "premarket_bias_score",
    "premarket_bias_label",
    "gift_nifty_premium",
    "nifty_sp500_correlation_30d",
    "nifty_usdinr_correlation_30d",
    "nifty_crude_correlation_30d",
    "global_risk_score_0_100",
    "currency_pressure_score",
    "commodity_inflation_score",
    "rates_pressure_score",
    "risk_on_off_signal",
    "external_gap_risk_score",
]

TYPE_OVERRIDES = {
    "usdinr_direction": SignalType.DIRECTIONAL,
    "usdinr_regime": SignalType.CATEGORICAL,
    "brent_regime_impact": SignalType.CATEGORICAL,
    "gold_safe_haven_signal": SignalType.DIRECTIONAL,
    "us_market_bias": SignalType.CATEGORICAL,
    "us_10y_regime": SignalType.CATEGORICAL,
    "asia_market_bias": SignalType.CATEGORICAL,
    "premarket_bias_label": SignalType.CATEGORICAL,
    "risk_on_off_signal": SignalType.DIRECTIONAL,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.CROSS_ASSET,
    type_overrides=TYPE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values = {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx))


def _compute_values(ctx: SignalContext) -> dict[str, Any]:
    payload = ctx.cross_asset_snapshot or {}
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else payload
    bias = payload.get("bias") if isinstance(payload.get("bias"), dict) else {}
    if not snapshot:
        return {}

    usdinr = _asset(snapshot, "usdinr")
    brent = _asset(snapshot, "brent")
    gold = _asset(snapshot, "gold")
    dow = _asset(snapshot, "dow_futures")
    spx = _asset(snapshot, "sp500_futures")
    nasdaq = _asset(snapshot, "nasdaq_futures")
    ten_y = _asset(snapshot, "us_10y_yield")
    two_y = _asset(snapshot, "us_2y_yield")
    nikkei = _asset(snapshot, "nikkei")
    hangseng = _asset(snapshot, "hangseng")
    gift = _asset(snapshot, "gift_nifty")

    us_score = _mean([
        _number(dow.get("change_pct")),
        _number(spx.get("change_pct")),
        _number(nasdaq.get("change_pct")),
    ])
    asia_score = _mean([
        _number(nikkei.get("change_pct")),
        _number(hangseng.get("change_pct")),
    ])

    values = {
        "usdinr_rate": _number(usdinr.get("price")),
        "usdinr_change_pct_today": _number(usdinr.get("change_pct")),
        "usdinr_direction": _direction(usdinr.get("direction"), _number(usdinr.get("change_pct"))),
        "usdinr_regime": _usdinr_regime(_number(usdinr.get("price"))),
        "brent_price": _number(brent.get("price")),
        "brent_change_pct_today": _number(brent.get("change_pct")),
        "brent_regime_impact": _brent_impact(_number(brent.get("change_pct"))),
        "gold_price": _number(gold.get("price")),
        "gold_change_pct_today": _number(gold.get("change_pct")),
        "gold_safe_haven_signal": _safe_haven(_number(gold.get("change_pct")), ctx.spot),
        "dow_futures_change_pct": _number(dow.get("change_pct")),
        "sp500_futures_change_pct": _number(spx.get("change_pct")),
        "nasdaq_futures_change_pct": _number(nasdaq.get("change_pct")),
        "us_market_bias": _bias_label(us_score),
        "us_10y_yield": _number(ten_y.get("price")),
        "us_10y_change_bps": _number(ten_y.get("change")),
        "us_10y_regime": _yield_regime(_number(ten_y.get("price"))),
        "us_2y_yield": _number(two_y.get("price")),
        "yield_curve_slope": _diff(_number(ten_y.get("price")), _number(two_y.get("price"))),
        "nikkei_change_pct": _number(nikkei.get("change_pct")),
        "hangseng_change_pct": _number(hangseng.get("change_pct")),
        "asia_market_bias": _bias_label(asia_score),
        "premarket_bias_score": _number(bias.get("score")),
        "premarket_bias_label": bias.get("bias"),
        "gift_nifty_premium": _number(gift.get("premium")),
        "global_risk_score_0_100": _global_risk_score(us_score, asia_score, _number(usdinr.get("change_pct")), _number(brent.get("change_pct"))),
        "currency_pressure_score": _pressure(_number(usdinr.get("change_pct")), scale=1.0),
        "commodity_inflation_score": _pressure(_number(brent.get("change_pct")), scale=2.0),
        "rates_pressure_score": _pressure(_number(ten_y.get("change")), scale=10.0),
        "risk_on_off_signal": _risk_signal(us_score, asia_score, _number(gold.get("change_pct"))),
        "external_gap_risk_score": _external_gap_risk(us_score, asia_score, _number(gift.get("premium"))),
    }
    values["usdinr_vs_20d_avg_pct"] = None
    values["nifty_sp500_correlation_30d"] = None
    values["nifty_usdinr_correlation_30d"] = None
    values["nifty_crude_correlation_30d"] = None
    return values


def _asset(snapshot: dict[str, Any], key: str) -> dict[str, Any]:
    value = snapshot.get(key)
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _direction(label: Any, change_pct: float | None) -> int | None:
    text = str(label or "").lower()
    if text == "up":
        return 1
    if text == "down":
        return -1
    if change_pct is None:
        return None
    if change_pct > 0.05:
        return 1
    if change_pct < -0.05:
        return -1
    return 0


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _usdinr_regime(rate: float | None) -> str | None:
    if rate is None:
        return None
    if rate < 83:
        return "strong"
    if rate > 85:
        return "weak"
    return "normal"


def _brent_impact(change_pct: float | None) -> str | None:
    if change_pct is None:
        return None
    if change_pct > 2:
        return "inflationary"
    if change_pct < -2:
        return "deflationary"
    return "neutral"


def _safe_haven(gold_change: float | None, spot: float) -> int | None:
    if gold_change is None:
        return None
    return 1 if gold_change > 0 and spot > 0 else 0


def _bias_label(score: float | None) -> str | None:
    if score is None:
        return None
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _yield_regime(yield_value: float | None) -> str | None:
    if yield_value is None:
        return None
    if yield_value < 3.5:
        return "low"
    if yield_value > 4.5:
        return "high"
    return "normal"


def _pressure(value: float | None, *, scale: float) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, 50.0 + value / scale * 50.0))


def _global_risk_score(us_score: float | None, asia_score: float | None, usdinr_change: float | None, brent_change: float | None) -> float | None:
    pieces = []
    if us_score is not None:
        pieces.append(50.0 - us_score * 25.0)
    if asia_score is not None:
        pieces.append(50.0 - asia_score * 25.0)
    if usdinr_change is not None:
        pieces.append(50.0 + usdinr_change * 20.0)
    if brent_change is not None:
        pieces.append(50.0 + brent_change * 10.0)
    if not pieces:
        return None
    return max(0.0, min(100.0, sum(pieces) / len(pieces)))


def _risk_signal(us_score: float | None, asia_score: float | None, gold_change: float | None) -> int | None:
    score = _mean([us_score, asia_score])
    if score is None:
        return None
    if score > 0.2 and gold_change is not None and gold_change <= 0:
        return 1
    if score < -0.2 or (gold_change is not None and gold_change > 0.5):
        return -1
    return 0


def _external_gap_risk(us_score: float | None, asia_score: float | None, gift_premium: float | None) -> float | None:
    pieces = []
    if us_score is not None:
        pieces.append(abs(us_score) * 25.0)
    if asia_score is not None:
        pieces.append(abs(asia_score) * 25.0)
    if gift_premium is not None:
        pieces.append(abs(gift_premium) / 100.0 * 50.0)
    return min(100.0, sum(pieces)) if pieces else None
