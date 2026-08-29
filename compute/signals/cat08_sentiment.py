from __future__ import annotations

from collections import deque
from typing import Any

from .common import current_price, ema, last, numeric_series, option_chain_summary, safe_div
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "sentiment"
_SCORE_HISTORY: deque[float] = deque(maxlen=20)

SIGNAL_NAMES = [
    "sentiment_score_4h",
    "sentiment_label",
    "sentiment_article_count_4h",
    "sentiment_bullish_count",
    "sentiment_bearish_count",
    "sentiment_neutral_count",
    "sentiment_high_impact_count",
    "sentiment_momentum",
    "sentiment_equity_score",
    "sentiment_rates_score",
    "sentiment_fx_score",
    "sentiment_macro_score",
    "nse_announcement_count_today",
    "nse_high_impact_count_today",
    "nse_announcement_sentiment",
    "et_sentiment_score",
    "cnbc_sentiment_score",
    "gtrends_nifty_india",
    "gtrends_stock_market_india",
    "gtrends_sip_india",
    "twitter_nifty_mention_volume",
    "twitter_nifty_sentiment",
    "reddit_india_invest_sentiment",
    "fear_greed_composite",
    "vix_fear_component",
    "pcr_fear_component",
    "iv_rank_fear_component",
    "momentum_fear_component",
    "breadth_fear_component",
    "headline_velocity_4h",
    "high_impact_news_ratio",
    "negative_news_pressure",
    "positive_news_pressure",
    "announcement_intensity_score",
    "sentiment_risk_off_signal",
    "sentiment_complacency_signal",
    "media_dispersion_score",
    "news_flow_quality_score",
    "sentiment_shock_score",
    "news_staleness_warning",
]

TYPE_OVERRIDES = {
    "sentiment_label": SignalType.CATEGORICAL,
    "sentiment_momentum": SignalType.DIRECTIONAL,
    "nse_announcement_sentiment": SignalType.DIRECTIONAL,
    "sentiment_risk_off_signal": SignalType.BINARY,
    "sentiment_complacency_signal": SignalType.BINARY,
    "news_staleness_warning": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    "fear_greed_composite": DataSource.SENTIMENT,
    "vix_fear_component": DataSource.FII_DII,
    "pcr_fear_component": DataSource.OPTIONS_CHAIN,
    "iv_rank_fear_component": DataSource.OPTIONS_CHAIN,
    "momentum_fear_component": DataSource.OHLCV_DAILY,
    "breadth_fear_component": DataSource.FO_EOD,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.SENTIMENT,
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
    data = ctx.sentiment_data or {}
    score = data.get("score") if isinstance(data.get("score"), dict) else {}
    headlines = data.get("headlines") if isinstance(data.get("headlines"), list) else []
    announcements = data.get("announcements") if isinstance(data.get("announcements"), list) else []
    sentiment_score = _number(score.get("score"))
    if sentiment_score is not None:
        _SCORE_HISTORY.append(sentiment_score)

    breakdown = score.get("breakdown") if isinstance(score.get("breakdown"), dict) else {}
    source_scores = _source_scores(headlines)
    high_impact = _number(score.get("high_impact_count")) or 0.0
    article_count = _number(score.get("article_count")) or 0.0
    bullish = _number(score.get("bullish_count")) or 0.0
    bearish = _number(score.get("bearish_count")) or 0.0
    neutral = _number(score.get("neutral_count")) or 0.0
    vix_component = _vix_component(ctx.india_vix)
    pcr_component, iv_component = _options_fear_components(ctx)
    momentum_component = _momentum_component(ctx)
    fear_greed = _mean([100.0 - vix_component if vix_component is not None else None, pcr_component, iv_component, momentum_component])

    values = {
        "sentiment_score_4h": sentiment_score,
        "sentiment_label": score.get("label"),
        "sentiment_article_count_4h": article_count,
        "sentiment_bullish_count": bullish,
        "sentiment_bearish_count": bearish,
        "sentiment_neutral_count": neutral,
        "sentiment_high_impact_count": high_impact,
        "sentiment_momentum": _momentum(),
        "sentiment_equity_score": _number(breakdown.get("equity")),
        "sentiment_rates_score": _number(breakdown.get("rates")),
        "sentiment_fx_score": _number(breakdown.get("currency")),
        "sentiment_macro_score": _number(breakdown.get("macro")),
        "nse_announcement_count_today": len(announcements),
        "nse_high_impact_count_today": sum(1 for item in announcements if str(item.get("impact") or "").lower() == "high"),
        "nse_announcement_sentiment": 0,
        "et_sentiment_score": source_scores.get("economic times"),
        "cnbc_sentiment_score": source_scores.get("cnbc"),
        "fear_greed_composite": fear_greed,
        "vix_fear_component": vix_component,
        "pcr_fear_component": pcr_component,
        "iv_rank_fear_component": iv_component,
        "momentum_fear_component": momentum_component,
        "headline_velocity_4h": article_count / 4.0,
        "high_impact_news_ratio": safe_div(high_impact, article_count),
        "negative_news_pressure": safe_div(bearish, article_count),
        "positive_news_pressure": safe_div(bullish, article_count),
        "announcement_intensity_score": min(100.0, len(announcements) * 10.0),
        "sentiment_risk_off_signal": int(
            (sentiment_score is not None and sentiment_score < 40.0) or high_impact >= 2
        ),
        "sentiment_complacency_signal": int(
            sentiment_score is not None and sentiment_score > 65.0 and high_impact == 0
        ),
        "media_dispersion_score": _dispersion([bullish, bearish, neutral]),
        "news_flow_quality_score": _quality(article_count, high_impact),
        "sentiment_shock_score": (
            None if sentiment_score is None else abs(sentiment_score - 50.0) + high_impact * 5.0
        ),
        "news_staleness_warning": int(article_count == 0),
    }
    return values


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _momentum() -> int | None:
    if len(_SCORE_HISTORY) < 2:
        return None
    if _SCORE_HISTORY[-1] > _SCORE_HISTORY[0] + 2:
        return 1
    if _SCORE_HISTORY[-1] < _SCORE_HISTORY[0] - 2:
        return -1
    return 0


def _source_scores(headlines: list[dict[str, Any]]) -> dict[str, float]:
    by_source: dict[str, list[float]] = {}
    for item in headlines:
        source = str(item.get("source") or "").lower()
        if not source:
            continue
        sentiment = str(item.get("sentiment") or "").lower()
        score = 50.0
        if sentiment == "bullish":
            score = 65.0
        elif sentiment == "bearish":
            score = 35.0
        by_source.setdefault(source, []).append(score)
    return {key: sum(values) / len(values) for key, values in by_source.items() if values}


def _vix_component(vix: float | None) -> float | None:
    if vix is None:
        return None
    return max(0.0, min(100.0, (float(vix) - 10.0) / 25.0 * 100.0))


def _options_fear_components(ctx: SignalContext) -> tuple[float | None, float | None]:
    summary = option_chain_summary(ctx.chain_legs, ctx.spot)
    pcr = summary.get("pcr")
    pcr_component = None
    if pcr is not None:
        pcr_component = max(0.0, min(100.0, 50.0 + (float(pcr) - 1.0) * 50.0))
    atm_call = summary.get("atm_call") or {}
    atm_iv = atm_call.get("iv")
    iv_component = max(0.0, min(100.0, float(atm_iv) * 300.0)) if atm_iv is not None else None
    return pcr_component, iv_component


def _momentum_component(ctx: SignalContext) -> float | None:
    df = ctx.ohlcv_df
    if df is None or getattr(df, "empty", True):
        return None
    close = numeric_series(df, "close")
    ema_200 = ema(close, 200)
    price = current_price(df, ctx.spot)
    base = last(ema_200)
    if base is None or base == 0:
        return None
    return max(0.0, min(100.0, 50.0 + (price - base) / base * 500.0))


def _dispersion(values: list[float]) -> float | None:
    total = sum(values)
    if total <= 0:
        return None
    shares = [value / total for value in values]
    return (1.0 - max(shares)) * 100.0


def _quality(article_count: float, high_impact: float) -> float:
    return min(100.0, article_count * 3.0 + high_impact * 10.0)
