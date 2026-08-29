from __future__ import annotations

import math
from typing import Any

import numpy as np

from .common import (
    atr,
    bollinger,
    cci,
    cross_signal,
    current_price,
    days_since_cross,
    ema,
    historical_vol,
    last,
    linear_slope_r2,
    macd,
    nth_from_end,
    numeric_series,
    percentile,
    roc,
    rsi,
    safe_div,
    sma,
    stochastic_k,
    trix,
    true_range,
    williams_r,
    z_score,
)
from .framework import (
    DataSource,
    SignalRegime,
    SignalType,
    build_signals,
    source_availability,
    specs_from_names,
    SignalContext,
)


CATEGORY = "price_action"

SIGNAL_NAMES = [
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_200",
    "price_vs_ema9_pct",
    "price_vs_ema21_pct",
    "price_vs_ema50_pct",
    "price_vs_ema200_pct",
    "ema_alignment_score",
    "ema9_21_cross",
    "ema21_50_cross",
    "ema50_200_cross",
    "golden_cross_days",
    "adx_14",
    "adx_di_plus",
    "adx_di_minus",
    "adx_di_spread",
    "supertrend_7_3",
    "supertrend_distance_pct",
    "linear_reg_slope_20",
    "linear_reg_r2_20",
    "price_above_all_emas",
    "rsi_14",
    "rsi_7",
    "rsi_21",
    "rsi_divergence_10d",
    "rsi_regime",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "macd_histogram_expanding",
    "macd_zero_cross_days",
    "stoch_k_14",
    "stoch_d_14",
    "stoch_signal",
    "williams_r_14",
    "roc_5",
    "roc_10",
    "roc_20",
    "cci_20",
    "trix_14",
    "kst_signal",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width_pct",
    "bb_position",
    "bb_squeeze",
    "bb_expansion",
    "atr_14",
    "atr_pct",
    "atr_z_score",
    "keltner_upper",
    "keltner_lower",
    "keltner_breakout",
    "hv_10",
    "hv_20",
    "hv_30",
    "hv_ratio_short_long",
    "parkinson_vol_10",
    "garman_klass_vol_10",
    "price_gap_today",
    "high_52w_distance",
    "low_52w_distance",
    "return_1d_pct",
    "return_3d_pct",
    "return_5d_pct",
    "return_20d_pct",
    "ema9_slope_5d",
    "ema21_slope_5d",
    "ema50_slope_10d",
    "price_channel_20d_position",
    "donchian_20d_upper",
    "donchian_20d_lower",
    "donchian_55d_upper",
    "donchian_55d_lower",
    "daily_range_pct",
    "close_location_value",
    "true_range_pct",
    "gap_fade_signal",
    "momentum_composite_0_100",
    "trend_composite_0_100",
]

TYPE_OVERRIDES = {
    "ema9_21_cross": SignalType.DIRECTIONAL,
    "ema21_50_cross": SignalType.DIRECTIONAL,
    "ema50_200_cross": SignalType.DIRECTIONAL,
    "supertrend_7_3": SignalType.DIRECTIONAL,
    "price_above_all_emas": SignalType.BINARY,
    "rsi_divergence_10d": SignalType.DIRECTIONAL,
    "rsi_regime": SignalType.CATEGORICAL,
    "macd_histogram_expanding": SignalType.BINARY,
    "stoch_signal": SignalType.DIRECTIONAL,
    "kst_signal": SignalType.DIRECTIONAL,
    "bb_squeeze": SignalType.BINARY,
    "bb_expansion": SignalType.BINARY,
    "keltner_breakout": SignalType.DIRECTIONAL,
    "gap_fade_signal": SignalType.DIRECTIONAL,
}

REGIME_OVERRIDES = {
    "adx_14": SignalRegime.TRENDING,
    "adx_di_spread": SignalRegime.TRENDING,
    "supertrend_7_3": SignalRegime.TRENDING,
    "linear_reg_slope_20": SignalRegime.TRENDING,
    "linear_reg_r2_20": SignalRegime.TRENDING,
    "bb_squeeze": SignalRegime.LOW_VOL,
    "bb_expansion": SignalRegime.HIGH_VOL,
    "atr_z_score": SignalRegime.HIGH_VOL,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OHLCV_DAILY,
    type_overrides=TYPE_OVERRIDES,
    regime_overrides=REGIME_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values, histories = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values, histories = {}, {}
    return build_signals(
        CATEGORY,
        SPECS,
        values,
        source_availability(ctx),
        histories=histories,
    )


def _compute_values(ctx: SignalContext) -> tuple[dict[str, Any], dict[str, Any]]:
    df = ctx.ohlcv_df
    if df is None or getattr(df, "empty", True):
        return {}, {}

    close = numeric_series(df, "close")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    open_ = numeric_series(df, "open")
    price = current_price(df, ctx.spot)

    ema_9 = ema(close, 9)
    ema_21 = ema(close, 21)
    ema_50 = ema(close, 50)
    ema_200 = ema(close, 200)
    rsi_14 = rsi(close, 14)
    rsi_7 = rsi(close, 7)
    rsi_21 = rsi(close, 21)
    macd_line, macd_sig, macd_hist = macd(close)
    stoch_k = stochastic_k(df, 14)
    stoch_d = stoch_k.rolling(3, min_periods=2).mean()
    bb_upper, bb_middle, bb_lower = bollinger(close, 20, 2.0)
    atr_14 = atr(df, 14)
    tr = true_range(df)
    hv_10 = historical_vol(close, 10)
    hv_20 = historical_vol(close, 20)
    hv_30 = historical_vol(close, 30)

    latest_high = last(high)
    latest_low = last(low)
    latest_open = last(open_)
    prev_close = nth_from_end(close, 1)
    latest_close = last(close, price) or price
    price_for_distance = price or latest_close

    adx, plus_di, minus_di = _adx(df, 14)
    supertrend_line, supertrend_dir = _supertrend(df, 7, 3.0)
    slope_20, r2_20 = linear_slope_r2(close, 20)
    width = safe_div(last(bb_upper) - last(bb_lower), last(bb_middle))
    width_pct = width * 100.0 if width is not None else None
    bb_position = safe_div(price_for_distance - last(bb_lower), last(bb_upper) - last(bb_lower))

    keltner_middle = ema(close, 20)
    kel_upper = keltner_middle + 2.0 * atr_14
    kel_lower = keltner_middle - 2.0 * atr_14
    high_52w = high.rolling(252, min_periods=20).max()
    low_52w = low.rolling(252, min_periods=20).min()
    donchian_20_upper = high.rolling(20, min_periods=10).max()
    donchian_20_lower = low.rolling(20, min_periods=10).min()
    donchian_55_upper = high.rolling(55, min_periods=20).max()
    donchian_55_lower = low.rolling(55, min_periods=20).min()
    cci_20 = cci(df, 20)
    trix_14 = trix(close, 14)
    kst = _kst(close)

    values: dict[str, Any] = {
        "ema_9": last(ema_9),
        "ema_21": last(ema_21),
        "ema_50": last(ema_50),
        "ema_200": last(ema_200),
        "price_vs_ema9_pct": _pct_distance(price_for_distance, last(ema_9)),
        "price_vs_ema21_pct": _pct_distance(price_for_distance, last(ema_21)),
        "price_vs_ema50_pct": _pct_distance(price_for_distance, last(ema_50)),
        "price_vs_ema200_pct": _pct_distance(price_for_distance, last(ema_200)),
        "ema_alignment_score": sum(
            1
            for value in (last(ema_9), last(ema_21), last(ema_50), last(ema_200))
            if value is not None and price_for_distance > value
        ),
        "ema9_21_cross": cross_signal(ema_9, ema_21),
        "ema21_50_cross": cross_signal(ema_21, ema_50),
        "ema50_200_cross": cross_signal(ema_50, ema_200),
        "golden_cross_days": days_since_cross(ema_50, ema_200),
        "adx_14": last(adx),
        "adx_di_plus": last(plus_di),
        "adx_di_minus": last(minus_di),
        "adx_di_spread": _diff(last(plus_di), last(minus_di)),
        "supertrend_7_3": supertrend_dir,
        "supertrend_distance_pct": _pct_distance(price_for_distance, last(supertrend_line)),
        "linear_reg_slope_20": slope_20,
        "linear_reg_r2_20": r2_20,
        "price_above_all_emas": int(
            all(value is not None and price_for_distance > value for value in (last(ema_9), last(ema_21), last(ema_50), last(ema_200)))
        ),
        "rsi_14": last(rsi_14),
        "rsi_7": last(rsi_7),
        "rsi_21": last(rsi_21),
        "rsi_divergence_10d": _rsi_divergence(close, rsi_14),
        "rsi_regime": _rsi_regime(last(rsi_14)),
        "macd_line": last(macd_line),
        "macd_signal": last(macd_sig),
        "macd_histogram": last(macd_hist),
        "macd_histogram_expanding": _hist_expanding(macd_hist),
        "macd_zero_cross_days": _zero_cross_days(macd_line),
        "stoch_k_14": last(stoch_k),
        "stoch_d_14": last(stoch_d),
        "stoch_signal": _stoch_signal(stoch_k, stoch_d),
        "williams_r_14": last(williams_r(df, 14)),
        "roc_5": last(roc(close, 5)),
        "roc_10": last(roc(close, 10)),
        "roc_20": last(roc(close, 20)),
        "cci_20": last(cci_20),
        "trix_14": last(trix_14),
        "kst_signal": _kst_signal(kst),
        "bb_upper": last(bb_upper),
        "bb_middle": last(bb_middle),
        "bb_lower": last(bb_lower),
        "bb_width_pct": width_pct,
        "bb_position": bb_position,
        "bb_squeeze": int((width_pct or math.inf) < (_percentile_series(_bb_width_pct(bb_upper, bb_middle, bb_lower), 20) or -math.inf)),
        "bb_expansion": int((width_pct or -math.inf) > (_percentile_series(_bb_width_pct(bb_upper, bb_middle, bb_lower), 80) or math.inf)),
        "atr_14": last(atr_14),
        "atr_pct": _pct_of(last(atr_14), price_for_distance),
        "atr_z_score": z_score(last(atr_14), atr_14.dropna().iloc[-30:].tolist()),
        "keltner_upper": last(kel_upper),
        "keltner_lower": last(kel_lower),
        "keltner_breakout": _keltner_breakout(price_for_distance, last(kel_upper), last(kel_lower)),
        "hv_10": last(hv_10),
        "hv_20": last(hv_20),
        "hv_30": last(hv_30),
        "hv_ratio_short_long": safe_div(last(hv_10), last(hv_30)),
        "parkinson_vol_10": last(_parkinson_vol(high, low, 10)),
        "garman_klass_vol_10": last(_garman_klass_vol(open_, high, low, close, 10)),
        "price_gap_today": _gap_pct(latest_open, prev_close),
        "high_52w_distance": _pct_distance(price_for_distance, last(high_52w)),
        "low_52w_distance": _pct_distance(price_for_distance, last(low_52w)),
        "return_1d_pct": last(roc(close, 1)),
        "return_3d_pct": last(roc(close, 3)),
        "return_5d_pct": last(roc(close, 5)),
        "return_20d_pct": last(roc(close, 20)),
        "ema9_slope_5d": _slope_pct(ema_9, 5),
        "ema21_slope_5d": _slope_pct(ema_21, 5),
        "ema50_slope_10d": _slope_pct(ema_50, 10),
        "price_channel_20d_position": safe_div(price_for_distance - last(donchian_20_lower), last(donchian_20_upper) - last(donchian_20_lower)),
        "donchian_20d_upper": last(donchian_20_upper),
        "donchian_20d_lower": last(donchian_20_lower),
        "donchian_55d_upper": last(donchian_55_upper),
        "donchian_55d_lower": last(donchian_55_lower),
        "daily_range_pct": _pct_of(_diff(latest_high, latest_low), price_for_distance),
        "close_location_value": safe_div(latest_close - latest_low, latest_high - latest_low),
        "true_range_pct": _pct_of(last(tr), price_for_distance),
        "gap_fade_signal": _gap_fade(latest_open, latest_close, prev_close),
    }
    values["momentum_composite_0_100"] = _momentum_score(values)
    values["trend_composite_0_100"] = _trend_score(values)

    histories = {
        "rsi_14": rsi_14.tolist(),
        "atr_14": atr_14.tolist(),
        "bb_width_pct": _bb_width_pct(bb_upper, bb_middle, bb_lower).tolist(),
        "hv_20": hv_20.tolist(),
        "return_1d_pct": roc(close, 1).tolist(),
        "price_vs_ema200_pct": ((close - ema_200) / ema_200.replace(0, np.nan) * 100.0).tolist(),
    }
    return values, histories


def _pct_distance(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or abs(base) < 1e-12:
        return None
    return (value - base) / base * 100.0


def _pct_of(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or abs(base) < 1e-12:
        return None
    return value / base * 100.0


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _adx(df: Any, period: int):
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_series = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * np.asarray(plus_dm) / atr_series.replace(0, np.nan)
    minus_di = 100.0 * np.asarray(minus_dm) / atr_series.replace(0, np.nan)
    plus_di = plus_di.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_di = minus_di.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def _supertrend(df: Any, period: int, multiplier: float):
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    close = numeric_series(df, "close")
    atr_series = atr(df, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr_series
    lower = hl2 - multiplier * atr_series
    line = lower.copy()
    direction = 1
    for index in range(1, len(close)):
        if close.iloc[index] > upper.iloc[index - 1]:
            direction = 1
        elif close.iloc[index] < lower.iloc[index - 1]:
            direction = -1
        line.iloc[index] = lower.iloc[index] if direction == 1 else upper.iloc[index]
    return line, direction


def _rsi_regime(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 30:
        return "oversold"
    if value > 70:
        return "overbought"
    return "neutral"


def _rsi_divergence(close: Any, rsi_series: Any) -> int:
    if len(close.dropna()) < 12 or len(rsi_series.dropna()) < 12:
        return 0
    price_now = close.iloc[-1]
    price_prev = close.iloc[-10]
    rsi_now = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-10]
    if price_now < price_prev and rsi_now > rsi_prev:
        return 1
    if price_now > price_prev and rsi_now < rsi_prev:
        return -1
    return 0


def _hist_expanding(hist: Any) -> int:
    current = last(hist)
    previous = nth_from_end(hist, 1)
    if current is None or previous is None:
        return 0
    return int(abs(current) > abs(previous))


def _zero_cross_days(series: Any) -> int | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    sign = 1 if clean.iloc[-1] >= 0 else -1
    for offset in range(1, len(clean)):
        current = clean.iloc[-offset]
        previous = clean.iloc[-offset - 1]
        if previous <= 0 < current or previous >= 0 > current:
            return offset if sign > 0 else -offset
    return len(clean) if sign > 0 else -len(clean)


def _stoch_signal(k: Any, d: Any) -> int:
    if len(k.dropna()) < 2 or len(d.dropna()) < 2:
        return 0
    crossed_up = k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1]
    crossed_down = k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] < d.iloc[-1]
    if crossed_up and k.iloc[-1] < 30:
        return 1
    if crossed_down and k.iloc[-1] > 70:
        return -1
    return 0


def _kst(close: Any):
    r1 = roc(close, 10).rolling(10, min_periods=5).mean()
    r2 = roc(close, 15).rolling(10, min_periods=5).mean()
    r3 = roc(close, 20).rolling(10, min_periods=5).mean()
    r4 = roc(close, 30).rolling(15, min_periods=5).mean()
    return r1 + 2 * r2 + 3 * r3 + 4 * r4


def _kst_signal(kst: Any) -> int:
    signal = kst.rolling(9, min_periods=3).mean()
    return cross_signal(kst, signal)


def _bb_width_pct(upper: Any, middle: Any, lower: Any):
    return (upper - lower) / middle.replace(0, np.nan) * 100.0


def _percentile_series(series: Any, pct: float) -> float | None:
    clean = series.dropna().iloc[-252:]
    if len(clean) < 20:
        return None
    return float(np.percentile(clean, pct))


def _keltner_breakout(price: float | None, upper: float | None, lower: float | None) -> int:
    if price is None or upper is None or lower is None:
        return 0
    if price > upper:
        return 1
    if price < lower:
        return -1
    return 0


def _parkinson_vol(high: Any, low: Any, period: int):
    estimator = (np.log(high / low.replace(0, np.nan)) ** 2) / (4.0 * math.log(2.0))
    return np.sqrt(estimator.rolling(period, min_periods=max(5, period // 2)).mean() * 252.0) * 100.0


def _garman_klass_vol(open_: Any, high: Any, low: Any, close: Any, period: int):
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    estimator = 0.5 * log_hl ** 2 - (2.0 * math.log(2.0) - 1.0) * log_co ** 2
    return np.sqrt(estimator.clip(lower=0).rolling(period, min_periods=max(5, period // 2)).mean() * 252.0) * 100.0


def _gap_pct(open_value: float | None, prev_close: float | None) -> float | None:
    return _pct_distance(open_value, prev_close)


def _slope_pct(series: Any, period: int) -> float | None:
    current = last(series)
    previous = nth_from_end(series, period)
    return _pct_distance(current, previous)


def _gap_fade(open_value: float | None, close_value: float | None, prev_close: float | None) -> int:
    gap = _gap_pct(open_value, prev_close)
    if gap is None or close_value is None or prev_close is None:
        return 0
    if gap > 0.3 and close_value < open_value:
        return -1
    if gap < -0.3 and close_value > open_value:
        return 1
    return 0


def _momentum_score(values: dict[str, Any]) -> float | None:
    pieces = []
    rsi_value = values.get("rsi_14")
    if rsi_value is not None:
        pieces.append(max(0.0, min(100.0, float(rsi_value))))
    for name in ("roc_5", "roc_10", "roc_20"):
        value = values.get(name)
        if value is not None:
            pieces.append(max(0.0, min(100.0, 50.0 + float(value) * 5.0)))
    hist = values.get("macd_histogram")
    if hist is not None:
        pieces.append(60.0 if hist > 0 else 40.0)
    return sum(pieces) / len(pieces) if pieces else None


def _trend_score(values: dict[str, Any]) -> float | None:
    pieces = []
    ema_alignment = values.get("ema_alignment_score")
    if ema_alignment is not None:
        pieces.append(float(ema_alignment) / 4.0 * 100.0)
    price_above_all_emas = values.get("price_above_all_emas")
    if price_above_all_emas is not None:
        pieces.append(100.0 if price_above_all_emas else 0.0)
    adx_value = values.get("adx_14")
    adx_di_spread = values.get("adx_di_spread")
    if adx_value is not None and adx_di_spread is not None:
        direction = 1.0 if adx_di_spread >= 0 else 0.0
        pieces.append(min(100.0, float(adx_value) * 4.0) * direction)
    return sum(pieces) / len(pieces) if pieces else None
