"""Regression: unavailable sentiment must not surface as a fabricated 50.0.

Two independent sites previously coalesced "no classified articles in the
scoring window" into a fake neutral 50.0 score / "Neutral" label:

  - sentiment.classifier.compute_sentiment_score / _breakdown_score, when
    total_weight is 0 (no articles, or none within the window/known asset
    classes).
  - main._empty_sentiment_payload, used whenever the cache has never been
    populated (fresh startup) or a refresh fails with nothing cached yet.

Both are real financial-semantics defects: the fabricated 50.0 flowed
straight into compute/signals/cat08_sentiment.py's sentiment_score feature
(and thence into ML/strategy inputs) and into the /sentiment/current API
consumed by the frontend SentimentPanel, in both cases indistinguishable
from a genuinely observed neutral reading.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentiment.classifier import compute_sentiment_score


def test_compute_sentiment_score_returns_none_when_no_articles() -> None:
    result = compute_sentiment_score([])
    assert result["score"] is None
    assert result["label"] is None
    assert result["article_count"] == 0


def test_compute_sentiment_score_breakdown_is_none_for_asset_classes_with_no_articles() -> None:
    articles = [
        {
            "title": "RBI holds repo rate steady",
            "summary": "",
            "source": "et_markets",
            "sentiment": "bullish",
            "confidence": 0.8,
            "score": 0.6,
            "asset_class": "rates",
        }
    ]
    result = compute_sentiment_score(articles)
    assert result["score"] is not None
    assert result["breakdown"]["rates"] is not None
    # equity/currency/macro had zero weighted articles -> must stay None,
    # not fabricate 50.0.
    assert result["breakdown"]["equity"] is None
    assert result["breakdown"]["currency"] is None
    assert result["breakdown"]["macro"] is None


def test_compute_sentiment_score_real_signal_still_computed_when_articles_present() -> None:
    articles = [
        {
            "title": "Nifty rallies on strong earnings",
            "summary": "",
            "source": "et_markets",
            "sentiment": "bullish",
            "confidence": 0.9,
            "score": 0.8,
            "asset_class": "equity",
        }
    ]
    result = compute_sentiment_score(articles)
    assert result["score"] is not None
    assert result["score"] > 50.0
    assert result["label"] is not None


def test_empty_sentiment_payload_score_is_none_not_fabricated_neutral() -> None:
    import main as main_module

    payload = main_module._empty_sentiment_payload()
    assert payload["score"] is None


def test_sentiment_score_value_helper_is_none_safe() -> None:
    import main as main_module

    assert main_module._sentiment_score_value(None) is None
    assert main_module._sentiment_score_value({}) is None
    assert main_module._sentiment_score_value({"score": None}) is None
    assert main_module._sentiment_score_value({"score": {"score": None}}) is None
    assert main_module._sentiment_score_value({"score": {"score": 61.5}}) == 61.5
