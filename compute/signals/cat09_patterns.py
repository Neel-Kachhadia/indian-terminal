from __future__ import annotations

from typing import Any

from .common import (
    bollinger,
    current_price,
    last,
    numeric_series,
    option_chain_summary,
    safe_div,
)
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "patterns"

SIGNAL_NAMES = [
    "cdl_doji",
    "cdl_hammer",
    "cdl_hanging_man",
    "cdl_engulfing_bullish",
    "cdl_engulfing_bearish",
    "cdl_morning_star",
    "cdl_evening_star",
    "cdl_shooting_star",
    "cdl_inverted_hammer",
    "cdl_three_white_soldiers",
    "cdl_three_black_crows",
    "cdl_harami_bullish",
    "cdl_harami_bearish",
    "cdl_piercing_line",
    "cdl_dark_cloud_cover",
    "pattern_52w_high_breakout",
    "pattern_52w_low_breakdown",
    "pattern_20d_high_breakout",
    "pattern_20d_low_breakdown",
    "pattern_range_breakout",
    "pattern_bb_squeeze_breakout",
    "nearest_support",
    "nearest_resistance",
    "support_distance_pct",
    "resistance_distance_pct",
    "range_position",
    "gap_up_today",
    "gap_down_today",
    "gap_filled",
    "inside_day",
    "outside_day",
    "narrow_range_7",
    "wide_range_7",
    "close_above_prev_high",
    "close_below_prev_low",
    "bullish_reversal_cluster",
    "bearish_reversal_cluster",
    "support_resistance_balance",
    "pattern_score_0_100",
    "breakout_confirmation_score",
]

TYPE_OVERRIDES = {
    name: SignalType.DIRECTIONAL
    for name in SIGNAL_NAMES
    if name.startswith("cdl_")
}
TYPE_OVERRIDES.update({
    "pattern_52w_high_breakout": SignalType.BINARY,
    "pattern_52w_low_breakdown": SignalType.BINARY,
    "pattern_20d_high_breakout": SignalType.BINARY,
    "pattern_20d_low_breakdown": SignalType.BINARY,
    "pattern_range_breakout": SignalType.BINARY,
    "pattern_bb_squeeze_breakout": SignalType.BINARY,
    "gap_up_today": SignalType.BINARY,
    "gap_down_today": SignalType.BINARY,
    "gap_filled": SignalType.BINARY,
    "inside_day": SignalType.BINARY,
    "outside_day": SignalType.BINARY,
    "narrow_range_7": SignalType.BINARY,
    "wide_range_7": SignalType.BINARY,
    "close_above_prev_high": SignalType.BINARY,
    "close_below_prev_low": SignalType.BINARY,
})

SOURCE_OVERRIDES = {
    "nearest_support": DataSource.OPTIONS_CHAIN,
    "nearest_resistance": DataSource.OPTIONS_CHAIN,
    "support_distance_pct": DataSource.OPTIONS_CHAIN,
    "resistance_distance_pct": DataSource.OPTIONS_CHAIN,
    "range_position": DataSource.OPTIONS_CHAIN,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OHLCV_DAILY,
    type_overrides=TYPE_OVERRIDES,
    source_overrides=SOURCE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values = {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx))


def _compute_values(ctx: SignalContext) -> dict[str, Any]:
    df = ctx.ohlcv_df
    values: dict[str, Any] = {}
    if df is not None and not getattr(df, "empty", True):
        open_ = numeric_series(df, "open")
        high = numeric_series(df, "high")
        low = numeric_series(df, "low")
        close = numeric_series(df, "close")
        price = current_price(df, ctx.spot)
        body = (close - open_).abs()
        candle_range = (high - low).replace(0, None)
        upper_shadow = high - open_.where(close >= open_, close)
        lower_shadow = open_.where(close >= open_, close) - low
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        rolling_range = (high - low).rolling(7, min_periods=4).mean()
        bb_upper, bb_mid, bb_lower = bollinger(close, 20, 2.0)
        bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, None) * 100.0

        values.update({
            "cdl_doji": int((last(body) or 0.0) < 0.1 * ((last(candle_range) or 0.0))),
            "cdl_hammer": _hammer(last(body), last(upper_shadow), last(lower_shadow), bullish=True),
            "cdl_hanging_man": _hammer(last(body), last(upper_shadow), last(lower_shadow), bullish=False),
            "cdl_engulfing_bullish": _engulfing(last(open_), last(close), last(prev_open), last(prev_close), bullish=True),
            "cdl_engulfing_bearish": _engulfing(last(open_), last(close), last(prev_open), last(prev_close), bullish=False),
            "cdl_morning_star": _morning_evening(open_, close, bullish=True),
            "cdl_evening_star": _morning_evening(open_, close, bullish=False),
            "cdl_shooting_star": _shooting_inverted(last(body), last(upper_shadow), last(lower_shadow), shooting=True),
            "cdl_inverted_hammer": _shooting_inverted(last(body), last(upper_shadow), last(lower_shadow), shooting=False),
            "cdl_three_white_soldiers": _three_candles(open_, close, bullish=True),
            "cdl_three_black_crows": _three_candles(open_, close, bullish=False),
            "cdl_harami_bullish": _harami(open_, close, bullish=True),
            "cdl_harami_bearish": _harami(open_, close, bullish=False),
            "cdl_piercing_line": _piercing(open_, close, prev_open, prev_close),
            "cdl_dark_cloud_cover": _dark_cloud(open_, close, prev_open, prev_close),
            "pattern_52w_high_breakout": int(price > (high.rolling(252, min_periods=20).max().shift(1).iloc[-1] if len(high) > 20 else float("inf"))),
            "pattern_52w_low_breakdown": int(price < (low.rolling(252, min_periods=20).min().shift(1).iloc[-1] if len(low) > 20 else -float("inf"))),
            "pattern_20d_high_breakout": int(price > (high.rolling(20, min_periods=10).max().shift(1).iloc[-1] if len(high) > 20 else float("inf"))),
            "pattern_20d_low_breakdown": int(price < (low.rolling(20, min_periods=10).min().shift(1).iloc[-1] if len(low) > 20 else -float("inf"))),
            "pattern_range_breakout": _range_breakout(price, high, low),
            "pattern_bb_squeeze_breakout": _bb_breakout(price, bb_upper, bb_lower, bb_width),
            "gap_up_today": int((last(open_) or 0.0) > (last(prev_close) or float("inf")) * 1.003),
            "gap_down_today": int((last(open_) or 0.0) < (last(prev_close) or 0.0) * 0.997),
            "gap_filled": _gap_filled(last(open_), last(high), last(low), last(prev_close)),
            "inside_day": int((last(high) or 0.0) < (last(prev_high) or -float("inf")) and (last(low) or 0.0) > (last(prev_low) or float("inf"))),
            "outside_day": int((last(high) or 0.0) > (last(prev_high) or float("inf")) and (last(low) or 0.0) < (last(prev_low) or -float("inf"))),
            "narrow_range_7": int((last(high - low) or float("inf")) < (last(rolling_range) or 0.0) * 0.6),
            "wide_range_7": int((last(high - low) or 0.0) > (last(rolling_range) or float("inf")) * 1.6),
            "close_above_prev_high": int((last(close) or 0.0) > (last(prev_high) or float("inf"))),
            "close_below_prev_low": int((last(close) or 0.0) < (last(prev_low) or -float("inf"))),
        })

    summary = option_chain_summary(ctx.chain_legs, ctx.spot)
    support = max([strike for strike in summary["put_oi"] if strike <= ctx.spot], default=None) if ctx.spot > 0 else None
    resistance = min([strike for strike in summary["call_oi"] if strike >= ctx.spot], default=None) if ctx.spot > 0 else None
    values.update({
        "nearest_support": support,
        "nearest_resistance": resistance,
        "support_distance_pct": _pct(ctx.spot - support, ctx.spot) if support is not None else None,
        "resistance_distance_pct": _pct(resistance - ctx.spot, ctx.spot) if resistance is not None else None,
        "range_position": safe_div(ctx.spot - support, resistance - support) if support is not None and resistance is not None else None,
    })
    bullish = sum(1 for name in values if name.startswith("cdl_") and values.get(name) == 1)
    bearish = sum(1 for name in values if name.startswith("cdl_") and values.get(name) == -1)
    values["bullish_reversal_cluster"] = int(bullish >= 2)
    values["bearish_reversal_cluster"] = int(bearish >= 2)
    values["support_resistance_balance"] = safe_div(values.get("support_distance_pct"), values.get("resistance_distance_pct"))
    values["pattern_score_0_100"] = max(0.0, min(100.0, 50.0 + (bullish - bearish) * 10.0))
    values["breakout_confirmation_score"] = _breakout_score(values)
    return values


def _hammer(body: float | None, upper: float | None, lower: float | None, *, bullish: bool) -> int:
    if body is None or upper is None or lower is None:
        return 0
    active = lower > 2.0 * max(body, 1e-9) and upper < body
    return 1 if active and bullish else (-1 if active and not bullish else 0)


def _shooting_inverted(body: float | None, upper: float | None, lower: float | None, *, shooting: bool) -> int:
    if body is None or upper is None or lower is None:
        return 0
    active = upper > 2.0 * max(body, 1e-9) and lower < body
    return -1 if active and shooting else (1 if active and not shooting else 0)


def _engulfing(open_now: float | None, close_now: float | None, open_prev: float | None, close_prev: float | None, *, bullish: bool) -> int:
    if None in (open_now, close_now, open_prev, close_prev):
        return 0
    if bullish and close_prev < open_prev and close_now > open_now and close_now > open_prev and open_now < close_prev:
        return 1
    if not bullish and close_prev > open_prev and close_now < open_now and close_now < open_prev and open_now > close_prev:
        return -1
    return 0


def _morning_evening(open_: Any, close: Any, *, bullish: bool) -> int:
    if len(close.dropna()) < 3:
        return 0
    c1, c2, c3 = close.iloc[-3], close.iloc[-2], close.iloc[-1]
    o1, o2, o3 = open_.iloc[-3], open_.iloc[-2], open_.iloc[-1]
    if bullish and c1 < o1 and abs(c2 - o2) < abs(c1 - o1) * 0.5 and c3 > o3 and c3 > (o1 + c1) / 2:
        return 1
    if not bullish and c1 > o1 and abs(c2 - o2) < abs(c1 - o1) * 0.5 and c3 < o3 and c3 < (o1 + c1) / 2:
        return -1
    return 0


def _three_candles(open_: Any, close: Any, *, bullish: bool) -> int:
    if len(close.dropna()) < 3:
        return 0
    candles = [(open_.iloc[-idx], close.iloc[-idx]) for idx in (3, 2, 1)]
    if bullish and all(c > o for o, c in candles):
        return 1
    if not bullish and all(c < o for o, c in candles):
        return -1
    return 0


def _harami(open_: Any, close: Any, *, bullish: bool) -> int:
    if len(close.dropna()) < 2:
        return 0
    prev_body = sorted([open_.iloc[-2], close.iloc[-2]])
    curr_body = sorted([open_.iloc[-1], close.iloc[-1]])
    inside = curr_body[0] >= prev_body[0] and curr_body[1] <= prev_body[1]
    if bullish and inside and close.iloc[-1] > open_.iloc[-1]:
        return 1
    if not bullish and inside and close.iloc[-1] < open_.iloc[-1]:
        return -1
    return 0


def _piercing(open_: Any, close: Any, prev_open: Any, prev_close: Any) -> int:
    if len(close.dropna()) < 2:
        return 0
    midpoint = (prev_open.iloc[-1] + prev_close.iloc[-1]) / 2.0
    return int(prev_close.iloc[-1] < prev_open.iloc[-1] and open_.iloc[-1] < prev_close.iloc[-1] and close.iloc[-1] > midpoint)


def _dark_cloud(open_: Any, close: Any, prev_open: Any, prev_close: Any) -> int:
    if len(close.dropna()) < 2:
        return 0
    midpoint = (prev_open.iloc[-1] + prev_close.iloc[-1]) / 2.0
    return -1 if prev_close.iloc[-1] > prev_open.iloc[-1] and open_.iloc[-1] > prev_close.iloc[-1] and close.iloc[-1] < midpoint else 0


def _range_breakout(price: float, high: Any, low: Any) -> int:
    upper = high.rolling(20, min_periods=10).max().shift(1).iloc[-1] if len(high) > 20 else None
    lower = low.rolling(20, min_periods=10).min().shift(1).iloc[-1] if len(low) > 20 else None
    return int((upper is not None and price > upper) or (lower is not None and price < lower))


def _bb_breakout(price: float, upper: Any, lower: Any, width: Any) -> int:
    if len(width.dropna()) < 20:
        return 0
    squeeze = width.iloc[-2] < width.dropna().iloc[-252:].quantile(0.2)
    return int(squeeze and (price > upper.iloc[-1] or price < lower.iloc[-1]))


def _gap_filled(open_: float | None, high: float | None, low: float | None, prev_close: float | None) -> int | None:
    if None in (open_, high, low, prev_close):
        return None
    if open_ > prev_close:
        return int(low <= prev_close)
    if open_ < prev_close:
        return int(high >= prev_close)
    return 0


def _pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or abs(base) < 1e-12:
        return None
    return value / base * 100.0


def _breakout_score(values: dict[str, Any]) -> float:
    score = 50.0
    score += 15.0 if values.get("pattern_20d_high_breakout") else 0.0
    score -= 15.0 if values.get("pattern_20d_low_breakdown") else 0.0
    score += 10.0 if values.get("close_above_prev_high") else 0.0
    score -= 10.0 if values.get("close_below_prev_low") else 0.0
    return max(0.0, min(100.0, score))
