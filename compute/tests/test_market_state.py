from __future__ import annotations

import pytest

from live.market_state import (
    MarketDataState,
    QuoteEnvelope,
    QuoteFreshness,
    QuoteQuality,
    QuoteSource,
    evaluate_market_state,
    parse_quote_envelope,
    should_accept_update,
)


def quote(**overrides: object) -> QuoteEnvelope:
    values: dict[str, object] = {
        "schema_version": 1,
        "symbol": "NIFTY",
        "token": "99926000",
        "ltp": 25_000.0,
        "open": 24_900.0,
        "high": 25_050.0,
        "low": 24_850.0,
        "close": 24_950.0,
        "volume": 10,
        "oi": 0,
        "bid": 0.0,
        "ask": 0.0,
        "exchange": "NSE",
        "source_type": QuoteSource.ANGEL_STREAM,
        "observed_at_ms": 10_000,
        "received_at_ms": 10_001,
        "sequence": 5,
        "field_completeness": ("ltp", "open", "high", "low", "close", "volume"),
        "freshness": QuoteFreshness.FRESH,
        "quality": QuoteQuality.EXECUTABLE_STREAM,
        "fallback": False,
        "fallback_reason": None,
        "market_state": MarketDataState.STARTING,
    }
    values.update(overrides)
    return QuoteEnvelope(**values)  # type: ignore[arg-type]


def test_stale_stream_cannot_be_live() -> None:
    result = evaluate_market_state(quote(), now_ms=20_000, expected_feed_live=True)
    assert result.market_state is MarketDataState.STALE
    assert result.freshness is QuoteFreshness.STALE


def test_fallback_is_visibly_degraded() -> None:
    result = evaluate_market_state(
        quote(source_type=QuoteSource.NSE_CHAIN, quality=QuoteQuality.REFERENCE_ONLY, fallback=True, fallback_reason="fallback"),
        now_ms=10_100,
        expected_feed_live=True,
    )
    assert result.market_state is MarketDataState.DEGRADED


def test_out_of_order_sequence_is_rejected() -> None:
    assert not should_accept_update(quote(sequence=4, received_at_ms=10_002), quote(sequence=5))
    assert should_accept_update(quote(sequence=6), quote(sequence=5))


def test_legacy_or_malformed_payload_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        parse_quote_envelope({"symbol": "NIFTY", "ltp": 25_000})
    with pytest.raises(ValueError, match="fallback_reason"):
        parse_quote_envelope(quote(fallback=True).to_dict())


def test_reference_fallback_accepts_explicitly_absent_optional_fields() -> None:
    payload = quote(
        token="NSE_CHAIN:NIFTY",
        open=None,
        high=None,
        low=None,
        close=None,
        volume=None,
        oi=None,
        bid=None,
        ask=None,
        source_type=QuoteSource.NSE_CHAIN,
        quality=QuoteQuality.REFERENCE_ONLY,
        fallback=True,
        fallback_reason="Angel stream unavailable; NSE chain underlying reference",
        field_completeness=("ltp",),
        sequence=None,
        market_state=MarketDataState.DEGRADED,
    ).to_dict()

    parsed = parse_quote_envelope(payload)

    assert parsed.ltp == 25_000.0
    assert parsed.open is None
    assert parsed.volume is None


def test_field_completeness_cannot_claim_an_absent_value() -> None:
    payload = quote(open=None).to_dict()
    with pytest.raises(ValueError, match="lists absent fields: open"):
        parse_quote_envelope(payload)
