from __future__ import annotations

import pytest

from backtest.spot_quality import VERIFIED_INDEX_SPOT_SOURCES, verified_spot_sources
from data.market_data_quality import INDEX_SYMBOLS


@pytest.mark.parametrize("symbol", sorted(INDEX_SYMBOLS))
def test_every_canonical_index_symbol_resolves_to_index_sources(symbol: str) -> None:
    # verified_spot_sources() used to hardcode its own partial copy of
    # INDEX_SYMBOLS (missing FINNIFTY/MIDCPNIFTY), silently routing those
    # symbols to equity-only sources and starving ml/feature_engineering's
    # _load_ohlcv of data that decision_eligible_ohlcv already marks eligible.
    assert verified_spot_sources(symbol) == VERIFIED_INDEX_SPOT_SOURCES


def test_equity_symbol_still_resolves_to_equity_sources() -> None:
    from backtest.spot_quality import VERIFIED_EQUITY_SPOT_SOURCES

    assert verified_spot_sources("RELIANCE") == VERIFIED_EQUITY_SPOT_SOURCES
