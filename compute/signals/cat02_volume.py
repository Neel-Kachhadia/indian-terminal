from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    current_price,
    last,
    linear_slope_r2,
    nth_from_end,
    numeric_series,
    safe_div,
    sma,
    z_score,
)
from .framework import (
    DataSource,
    SignalRegime,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "volume"

SIGNAL_NAMES = [
    "volume_today",
    "volume_avg_5d",
    "volume_avg_20d",
    "volume_avg_60d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_z_score_252d",
    "volume_spike_2x",
    "volume_spike_3x",
    "volume_price_confirmation",
    "up_volume_ratio",
    "vpt",
    "obv",
    "obv_5d_slope",
    "obv_divergence",
    "ad_line",
    "cmf_20",
    "mfi_14",
    "ease_of_movement_14",
    "force_index_13",
    "delivery_pct",
    "block_deal_today",
    "bulk_deal_today",
    "unusual_volume_z_score",
    "nvi_trend",
    "pvi_trend",
    "volume_ema_20",
    "volume_ema_50",
    "volume_trend_20d",
    "price_volume_corr_20d",
    "dollar_volume_today",
    "dollar_volume_z_score",
    "range_volume_efficiency",
    "accumulation_days_20d",
    "distribution_days_20d",
    "volume_dry_up_signal",
    "climax_volume_signal",
    "relative_volume_percentile_52w",
    "turnover_breakout_20d",
    "participation_composite_0_100",
]

TYPE_OVERRIDES = {
    "volume_spike_2x": SignalType.BINARY,
    "volume_spike_3x": SignalType.BINARY,
    "volume_price_confirmation": SignalType.DIRECTIONAL,
    "obv_divergence": SignalType.DIRECTIONAL,
    "delivery_pct": SignalType.RATIO,
    "block_deal_today": SignalType.BINARY,
    "bulk_deal_today": SignalType.BINARY,
    "nvi_trend": SignalType.DIRECTIONAL,
    "pvi_trend": SignalType.DIRECTIONAL,
    "volume_dry_up_signal": SignalType.BINARY,
    "climax_volume_signal": SignalType.BINARY,
    "turnover_breakout_20d": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    "delivery_pct": DataSource.FO_EOD,
    "block_deal_today": DataSource.FO_EOD,
    "bulk_deal_today": DataSource.FO_EOD,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OHLCV_DAILY,
    type_overrides=TYPE_OVERRIDES,
    source_overrides=SOURCE_OVERRIDES,
    regime_overrides={
        "volume_spike_2x": SignalRegime.HIGH_VOL,
        "volume_spike_3x": SignalRegime.HIGH_VOL,
        "climax_volume_signal": SignalRegime.HIGH_VOL,
        "volume_dry_up_signal": SignalRegime.LOW_VOL,
    },
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
    volume = numeric_series(df, "volume")
    price = current_price(df, ctx.spot)
    volume_today = last(volume)
    avg_5 = last(sma(volume, 5))
    avg_20 = last(sma(volume, 20))
    avg_60 = last(sma(volume, 60))
    ret = close.pct_change()
    up_day = close > close.shift(1)
    down_day = close < close.shift(1)

    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    vpt = (volume * ret.fillna(0.0)).cumsum()
    money_flow_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_vol = money_flow_mult.fillna(0.0) * volume
    ad_line = money_flow_vol.cumsum()
    cmf = money_flow_vol.rolling(20, min_periods=10).sum() / volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    mfi = _mfi(high, low, close, volume, 14)
    eom = ((high + low) / 2.0).diff() * (high - low) / volume.replace(0, np.nan)
    force = close.diff() * volume
    nvi = _volume_index(close, volume, negative=True)
    pvi = _volume_index(close, volume, negative=False)
    dollar_volume = close * volume

    obv_slope, _ = linear_slope_r2(obv, 5)
    values = {
        "volume_today": volume_today,
        "volume_avg_5d": avg_5,
        "volume_avg_20d": avg_20,
        "volume_avg_60d": avg_60,
        "volume_ratio_5d": safe_div(volume_today, avg_5),
        "volume_ratio_20d": safe_div(volume_today, avg_20),
        "volume_z_score_252d": z_score(volume_today, volume.dropna().iloc[-252:].tolist()),
        "volume_spike_2x": int((volume_today or 0.0) > 2.0 * (avg_20 or float("inf"))),
        "volume_spike_3x": int((volume_today or 0.0) > 3.0 * (avg_20 or float("inf"))),
        "volume_price_confirmation": _price_volume_confirmation(close, volume),
        "up_volume_ratio": safe_div(volume.where(up_day).tail(5).sum(), volume.tail(5).sum()),
        "vpt": last(vpt),
        "obv": last(obv),
        "obv_5d_slope": obv_slope,
        "obv_divergence": _divergence(close, obv),
        "ad_line": last(ad_line),
        "cmf_20": last(cmf),
        "mfi_14": last(mfi),
        "ease_of_movement_14": last(eom.rolling(14, min_periods=7).mean()),
        "force_index_13": last(force.ewm(span=13, adjust=False, min_periods=7).mean()),
        "unusual_volume_z_score": z_score(volume_today, volume.dropna().iloc[-252:].tolist()),
        "nvi_trend": _index_trend(nvi),
        "pvi_trend": _index_trend(pvi),
        "volume_ema_20": last(volume.ewm(span=20, adjust=False, min_periods=10).mean()),
        "volume_ema_50": last(volume.ewm(span=50, adjust=False, min_periods=20).mean()),
        "volume_trend_20d": _pct(last(sma(volume, 5)), nth_from_end(sma(volume, 20), 20)),
        "price_volume_corr_20d": last(close.pct_change().rolling(20, min_periods=10).corr(volume.pct_change())),
        "dollar_volume_today": price * volume_today if volume_today is not None else None,
        "dollar_volume_z_score": z_score(last(dollar_volume), dollar_volume.dropna().iloc[-252:].tolist()),
        "range_volume_efficiency": safe_div(abs((last(close) or 0.0) - (nth_from_end(close, 1) or 0.0)), (volume_today or 0.0)),
        "accumulation_days_20d": int(((up_day) & (volume > volume.rolling(20, min_periods=5).mean())).tail(20).sum()),
        "distribution_days_20d": int(((down_day) & (volume > volume.rolling(20, min_periods=5).mean())).tail(20).sum()),
        "volume_dry_up_signal": int((volume_today or float("inf")) < 0.5 * (avg_20 or 0.0)),
        "climax_volume_signal": int((volume_today or 0.0) > 2.5 * (avg_20 or float("inf"))),
        "relative_volume_percentile_52w": _percentile(volume_today, volume.dropna().iloc[-252:].tolist()),
        "turnover_breakout_20d": int((last(dollar_volume) or 0.0) > (dollar_volume.rolling(20, min_periods=10).max().shift(1).iloc[-1] if len(dollar_volume) > 20 else float("inf"))),
    }
    values["participation_composite_0_100"] = _participation_score(values)

    histories = {
        "volume_today": volume.tolist(),
        "volume_ratio_20d": (volume / sma(volume, 20).replace(0, np.nan)).tolist(),
        "dollar_volume_today": dollar_volume.tolist(),
        "mfi_14": mfi.tolist(),
        "cmf_20": cmf.tolist(),
    }
    return values, histories


def _mfi(high: Any, low: Any, close: Any, volume: Any, period: int):
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(period, min_periods=period).sum().abs()
    ratio = positive / negative.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def _price_volume_confirmation(close: Any, volume: Any) -> int:
    if len(close.dropna()) < 2 or len(volume.dropna()) < 2:
        return 0
    price_up = close.iloc[-1] > close.iloc[-2]
    volume_up = volume.iloc[-1] > volume.iloc[-2]
    if price_up and volume_up:
        return 1
    if price_up and not volume_up:
        return -1
    if not price_up and volume_up:
        return -1
    return 0


def _divergence(price: Any, indicator: Any) -> int:
    if len(price.dropna()) < 10 or len(indicator.dropna()) < 10:
        return 0
    if price.iloc[-1] < price.iloc[-10] and indicator.iloc[-1] > indicator.iloc[-10]:
        return 1
    if price.iloc[-1] > price.iloc[-10] and indicator.iloc[-1] < indicator.iloc[-10]:
        return -1
    return 0


def _volume_index(close: Any, volume: Any, *, negative: bool):
    output = [1000.0]
    for index in range(1, len(close)):
        condition = volume.iloc[index] < volume.iloc[index - 1]
        if not negative:
            condition = volume.iloc[index] > volume.iloc[index - 1]
        if condition and close.iloc[index - 1] != 0:
            output.append(output[-1] * (1 + (close.iloc[index] - close.iloc[index - 1]) / close.iloc[index - 1]))
        else:
            output.append(output[-1])
    return close.copy().__class__(output, index=close.index)


def _index_trend(series: Any) -> int:
    current = last(series)
    mean_20 = last(sma(series, 20))
    if current is None or mean_20 is None:
        return 0
    if current > mean_20:
        return 1
    if current < mean_20:
        return -1
    return 0


def _pct(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or abs(previous) < 1e-12:
        return None
    return (current - previous) / previous * 100.0


def _percentile(value: float | None, history: list[float]) -> float | None:
    if value is None or not history:
        return None
    return sum(1 for item in history if item <= value) / len(history) * 100.0


def _participation_score(values: dict[str, Any]) -> float | None:
    pieces = []
    ratio = values.get("volume_ratio_20d")
    if ratio is not None:
        pieces.append(min(100.0, max(0.0, float(ratio) * 50.0)))
    mfi_value = values.get("mfi_14")
    if mfi_value is not None:
        pieces.append(min(100.0, max(0.0, float(mfi_value))))
    cmf_value = values.get("cmf_20")
    if cmf_value is not None:
        pieces.append(min(100.0, max(0.0, 50.0 + float(cmf_value) * 100.0)))
    return sum(pieces) / len(pieces) if pieces else None
