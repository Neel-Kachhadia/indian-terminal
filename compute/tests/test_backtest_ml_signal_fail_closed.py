from __future__ import annotations

import datetime as dt

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.position import Portfolio
from backtest.strategies.base import _condition_ok


class _NoSignalGuard:
    """Simulates a date where the ML model produced no output for this symbol."""

    def get_ohlcv(self, symbol, as_of_date, lookback=260):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    def get_chain_snapshot(self, symbol, as_of_date, expiry):
        return []

    def get_bif_value(self, column, symbol, as_of_date):
        return None

    def get_bif_label(self, column, symbol, as_of_date):
        return None

    def get_signal(self, signal_name, symbol, as_of_date):
        return None


class _StubSpotObservation:
    valid = False
    price = None
    source = "unavailable"
    quality = type("Q", (), {"value": "unavailable"})()
    observed_on = None

    def as_provenance(self):
        return {}


class _StubPricer:
    def get_spot(self, symbol, trade_date):
        return None

    def get_spot_observation(self, symbol, as_of_date):
        return _StubSpotObservation()

    def find_nearest_expiry(self, symbol, as_of_date, min_dte):
        return None

    def get_lot_size(self, symbol):
        return 75


def _build_signals():
    from backtest.strategies.base import BaseStrategy

    strategy = BaseStrategy("test_strategy", {"name": "test", "symbol": "NIFTY"})
    day = dt.date(2026, 1, 5)
    return strategy.generate_signals(day, day, _NoSignalGuard(), _StubPricer())


def test_missing_ml_signal_stays_unavailable_not_fabricated_neutral():
    signals = _build_signals()
    assert signals["ml.model_agreement"] is None
    assert signals["ml.composite_probabilities.UP"] is None


def test_condition_gate_fails_closed_regardless_of_threshold_direction_on_missing_signal():
    signals = _build_signals()
    above_half = {"signal": "ml.composite_probabilities.UP", "operator": ">", "value": 0.55}
    below_half = {"signal": "ml.composite_probabilities.UP", "operator": "<", "value": 0.45}
    assert _condition_ok(above_half, signals) is False
    assert _condition_ok(below_half, signals) is False


def test_position_sizer_does_not_fabricate_kelly_edge_when_ml_probability_unavailable():
    engine = BacktestEngine(
        strategy="test_strategy",
        from_date="2026-01-01",
        to_date="2026-01-31",
        position_sizer="quarter_kelly",
        symbol="NIFTY",
    )
    signals = _build_signals()
    signals["spot"] = 24000.0
    portfolio = Portfolio(500000.0)

    lots_no_signal = engine._compute_position_size(signals, dt.date(2026, 1, 5), portfolio, _StubPricer())

    signals_with_fabricated_edge = dict(signals)
    signals_with_fabricated_edge["ml.composite_probabilities.UP"] = 0.55
    lots_with_old_fabricated_default = engine._compute_position_size(
        signals_with_fabricated_edge, dt.date(2026, 1, 5), portfolio, _StubPricer()
    )

    assert lots_no_signal <= lots_with_old_fabricated_default
