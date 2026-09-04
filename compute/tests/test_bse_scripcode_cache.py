"""Regression test for the BSE scrip-code cache-poisoning bug: a cache key
of `bse_scripcode:{symbol}` (no ISIN) meant a scrip code resolved for one
ISIN could be handed back for a *different* ISIN on a later call for the
same symbol -- silently bypassing the identity check `_resolve_bse_scripcode`
exists to enforce. Fixed by keying the cache on (symbol, ISIN)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import main


class _FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, payload, ttl_seconds: int) -> None:
        self.store[key] = payload


class _FakeBseResponse:
    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.text = text


@pytest.fixture()
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> _FakeCache:
    cache = _FakeCache()
    monkeypatch.setattr(main, "_equity_cache_get", cache.get)
    monkeypatch.setattr(main, "_equity_cache_set", cache.set)
    return cache


def _peer_search_response(symbol: str, isin: str, scripcode: str) -> _FakeBseResponse:
    return _FakeBseResponse(
        f"<li><a>SOME COMPANY LTD<br /><span><strong>{symbol}</strong>"
        f"&nbsp;&nbsp;&nbsp;{isin}&nbsp;&nbsp;&nbsp;{scripcode}</span></a></li>"
    )


def test_cache_is_keyed_by_symbol_and_isin_not_symbol_alone(
    fake_cache: _FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_get(url: str, **kwargs):
        calls.append(url)
        # BSE's own real, unambiguous answer for this symbol never changes.
        return _peer_search_response("EXAMPLE", "INE000A00001", "999999")

    monkeypatch.setattr("data.bse_session.bse", SimpleNamespace(get=fake_get))

    # A(symbol, isin1) -- isin1 matches BSE's real answer -- resolves and caches.
    result1 = main._resolve_bse_scripcode("EXAMPLE", "INE000A00001")
    assert result1 == "999999"
    assert len(calls) == 1

    # A(symbol, isin2) -- a DIFFERENT (e.g. stale-NSE-reported) ISIN for the
    # same symbol -- must never return isin1's cached scrip code. It must
    # re-query BSE, find the same real answer, and reject it as a mismatch
    # against isin2, exactly as a fresh (uncached) call would.
    result2 = main._resolve_bse_scripcode("EXAMPLE", "INE999Z99999")
    assert result2 is None
    assert len(calls) == 2  # re-queried BSE rather than trusting isin1's cache entry

    # A(symbol, isin1) again -- must still return the correct original result.
    result3 = main._resolve_bse_scripcode("EXAMPLE", "INE000A00001")
    assert result3 == "999999"
    # Served from the (symbol, isin1)-keyed cache entry -- no third network call.
    assert len(calls) == 2


def test_cache_key_includes_isin_literally(fake_cache: _FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "data.bse_session.bse",
        SimpleNamespace(get=lambda url, **kwargs: _peer_search_response("EXAMPLE", "INE000A00001", "999999")),
    )
    main._resolve_bse_scripcode("EXAMPLE", "INE000A00001")
    assert any("INE000A00001" in key for key in fake_cache.store)
    assert "bse_scripcode:EXAMPLE" not in fake_cache.store
