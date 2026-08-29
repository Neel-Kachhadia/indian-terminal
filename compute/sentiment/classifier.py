from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any


BULLISH_KEYWORDS = [
    "rally", "surge", "jump", "soar", "climb", "gain", "rise", "up",
    "bull", "bullish", "breakout", "record high", "52-week high",
    "all-time high", "outperform", "upgrade", "buy", "accumulate",
    "GDP growth", "strong results", "beat estimates", "profit", "revenue up",
    "FII buying", "inflow", "rate cut", "RBI accommodative", "dovish",
    "inflation falls", "CPI down", "IIP up", "PMI expansion",
    "FII net buyer", "upper circuit", "SIP inflow", "GST collection up",
    "monsoon normal", "capex", "infrastructure spending",
]

BEARISH_KEYWORDS = [
    "fall", "drop", "crash", "plunge", "slide", "decline", "down",
    "bear", "bearish", "breakdown", "52-week low", "sell-off",
    "underperform", "downgrade", "sell", "caution", "warning",
    "miss estimates", "loss", "revenue down", "FII selling", "outflow",
    "rate hike", "RBI hawkish", "inflation rises", "CPI up", "IIP down",
    "PMI contraction", "deficit", "fiscal stress",
    "FII net seller", "lower circuit", "redemption", "GST shortfall",
    "monsoon deficit", "rupee weakens", "current account deficit",
    "SEBI action", "fraud", "scam", "penalty",
]

HIGH_IMPACT_SOURCES = ["nse_announcement", "rbi_press", "sebi_press"]

HIGH_IMPACT_WORDS = (
    "board meeting",
    "results",
    "dividend",
    "bonus",
    "split",
    "merger",
    "acquisition",
    "buyback",
    "delisting",
    "sebi order",
    "rbi policy",
    "budget",
    "fraud",
    "scam",
)

ASSET_CLASS_KEYWORDS = {
    "currency": (
        "rupee",
        "forex",
        "usd/inr",
        "usd-inr",
        "dollar",
        "currency",
    ),
    "rates": (
        "rbi",
        "repo",
        "rate",
        "bond",
        "yield",
        "g-sec",
        "monetary policy",
    ),
    "macro": (
        "gdp",
        "inflation",
        "iip",
        "pmi",
        "trade deficit",
        "cpi",
        "fiscal",
        "current account",
    ),
    "equity": (
        "stock",
        "share",
        "nifty",
        "sensex",
        "equity",
        "market",
        "ipo",
        "reliance",
        "hdfc",
        "icici",
        "infosys",
        "tcs",
        "bank nifty",
    ),
}


def classify_headline(
    title: str,
    summary: str = "",
    source: str = "",
) -> dict:
    """
    Classify a single Indian financial headline with keyword rules.
    """
    text = f"{title} {summary}".casefold()
    title_text = title.casefold()
    bullish_matches = _matched_keywords(text, BULLISH_KEYWORDS)
    bearish_matches = _matched_keywords(text, BEARISH_KEYWORDS)

    bullish_count = len(bullish_matches)
    bearish_count = len(bearish_matches)
    denominator = bullish_count + bearish_count + 1
    score = (bullish_count - bearish_count) / denominator

    if score > 0.1:
        sentiment = "bullish"
    elif score < -0.1:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    confidence = max(0.0, min(1.0, abs(score)))
    high_impact = (
        source in HIGH_IMPACT_SOURCES
        or any(_keyword_found(title_text, word) for word in HIGH_IMPACT_WORDS)
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "score": score,
        "matched_keywords": bullish_matches + bearish_matches,
        "impact": "high" if high_impact else "medium" if confidence > 0.3 else "low",
        "asset_class": _detect_asset_class(text),
    }


def compute_sentiment_score(
    articles: list[dict],
    window_hours: int = 4,
) -> dict:
    """
    Compute rolling 0-100 sentiment from classified articles.
    """
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=window_hours)
    counts = {
        "bullish": 0,
        "bearish": 0,
        "neutral": 0,
        "high_impact": 0,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    breakdown_sum = {key: 0.0 for key in ("equity", "rates", "currency", "macro")}
    breakdown_weight = {key: 0.0 for key in breakdown_sum}

    for article in articles:
        published_at = _article_datetime(article, now)
        if published_at < cutoff:
            continue

        classification = _classification_for_article(article)
        sentiment = classification["sentiment"]
        impact = str(classification.get("impact") or "low")
        asset_class = str(classification.get("asset_class") or "unknown")
        score = _finite_float(classification.get("score"))
        weight = _impact_weight(impact)

        if (now - published_at).total_seconds() < 3600:
            weight *= 1.5

        counts[sentiment] += 1
        if impact == "high":
            counts["high_impact"] += 1

        weighted_sum += score * weight
        total_weight += weight

        if asset_class in breakdown_sum:
            breakdown_sum[asset_class] += score * weight
            breakdown_weight[asset_class] += weight

    score_0_100: float | None = None
    if total_weight > 0:
        score_0_100 = max(0.0, min(100.0, 50.0 + (weighted_sum / total_weight) * 50.0))

    return {
        "score": score_0_100,
        "label": _score_label(score_0_100) if score_0_100 is not None else None,
        "article_count": counts["bullish"] + counts["bearish"] + counts["neutral"],
        "bullish_count": counts["bullish"],
        "bearish_count": counts["bearish"],
        "neutral_count": counts["neutral"],
        "high_impact_count": counts["high_impact"],
        "breakdown": {
            key: _breakdown_score(breakdown_sum[key], breakdown_weight[key])
            for key in breakdown_sum
        },
    }


def _classification_for_article(article: dict[str, Any]) -> dict[str, Any]:
    if all(key in article for key in ("sentiment", "confidence", "score")):
        return article
    return classify_headline(
        str(article.get("title") or ""),
        str(article.get("summary") or ""),
        str(article.get("source") or ""),
    )


def _matched_keywords(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if _keyword_found(text, keyword)]


def _keyword_found(text: str, keyword: str) -> bool:
    return re.search(_keyword_pattern(keyword.casefold()), text) is not None


def _keyword_pattern(keyword: str) -> str:
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")
    if " " in keyword or re.search(r"[^a-z0-9]", keyword):
        return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    if keyword.endswith("y") and len(keyword) > 2:
        stem = re.escape(keyword[:-1])
        return rf"\b{stem}(?:y|ies|ied|ying)\b"
    return rf"\b{escaped}(?:s|es|ed|ing)?\b"


def _detect_asset_class(text: str) -> str:
    for asset_class in ("currency", "rates", "macro", "equity"):
        if any(_keyword_found(text, keyword) for keyword in ASSET_CLASS_KEYWORDS[asset_class]):
            return asset_class
    return "unknown"


def _impact_weight(impact: str) -> float:
    if impact == "high":
        return 3.0
    if impact == "medium":
        return 1.5
    return 1.0


def _article_datetime(article: dict[str, Any], now: dt.datetime) -> dt.datetime:
    value = str(article.get("published_at") or "").strip()
    if value:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except ValueError:
            pass

    age_minutes = _finite_float(article.get("age_minutes"))
    return now - dt.timedelta(minutes=max(0.0, age_minutes))


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _breakdown_score(weighted_sum: float, total_weight: float) -> float | None:
    if total_weight <= 0:
        return None
    return max(0.0, min(100.0, 50.0 + (weighted_sum / total_weight) * 50.0))


def _score_label(score: float) -> str:
    if score >= 70:
        return "Strongly Bullish"
    if score > 55:
        return "Bullish"
    if score <= 30:
        return "Strongly Bearish"
    if score < 45:
        return "Bearish"
    return "Neutral"
