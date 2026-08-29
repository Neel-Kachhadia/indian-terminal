from __future__ import annotations

import datetime as dt

import pytest

from backtest.price_quality import HistoricalPriceObservation, HistoricalPriceQuality
from backtest.strategies.base import BaseStrategy
from backtest.engine import BacktestEngine, BacktestValidity, expected_execution_session
from backtest import engine as engine_module
from data.exchange_calendar import CalendarCoverageError


class RecordingPricer:
    def __init__(self) -> None:
        self.strike_requests: list[dt.date] = []
        self.price_requests: list[tuple[dt.date, str]] = []

    def get_lot_size(self, symbol: str) -> int:
        return 75

    def get_atm_strike(
        self,
        symbol: str,
        spot: float,
        expiry: dt.date,
        trade_date: dt.date,
    ) -> float:
        self.strike_requests.append(trade_date)
        return 24_000.0

    def get_option_observation(
        self,
        symbol: str,
        expiry: dt.date,
        strike: float,
        option_type: str,
        trade_date: dt.date,
        price_type: str,
    ) -> HistoricalPriceObservation:
        self.price_requests.append((trade_date, price_type))
        return HistoricalPriceObservation(
            100.0,
            HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION,
            "NSE_FO_BHAVCOPY",
            trade_date,
            "verified_ohlc_volume_source",
        )


def test_contract_selection_uses_prior_close_and_fill_uses_next_open() -> None:
    strategy = BaseStrategy(
        "timing",
        {
            "name": "timing",
            "legs": [{"type": "CE", "action": "BUY", "strike": "ATM"}],
        },
    )
    pricer = RecordingPricer()
    signal_day = dt.date(2026, 8, 20)
    execution_day = dt.date(2026, 8, 21)
    legs = strategy.build_legs(
        {
            "signal_date": signal_day.isoformat(),
            "spot": 24_010.0,
            "expiry": "2026-08-27",
        },
        execution_day,
        1,
        pricer,
        object(),
    )
    assert len(legs) == 1
    assert pricer.strike_requests == [signal_day]
    assert pricer.price_requests == [(execution_day, "open")]
    assert legs[0].entry_execution_point == "NEXT_SESSION_OPEN"


def test_execution_session_is_calendar_adjacent_across_weekend_holiday_and_year_end() -> None:
    assert expected_execution_session(dt.date(2026, 8, 21)) == dt.date(2026, 8, 24)
    assert expected_execution_session(dt.date(2026, 1, 14)) == dt.date(2026, 1, 16)
    assert expected_execution_session(dt.date(2026, 11, 6)) == dt.date(2026, 11, 8)
    assert expected_execution_session(dt.date(2026, 11, 30)) == dt.date(2026, 12, 1)
    with pytest.raises(CalendarCoverageError):
        expected_execution_session(dt.date(2026, 12, 31))


def test_actual_engine_rejects_t_plus_two_row_when_expected_session_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_day = dt.date(2026, 8, 20)
    t_plus_two = dt.date(2026, 8, 24)
    calls: list[str] = []

    class Connection:
        def close(self) -> None:
            return None

    class Pricer:
        price_observation_events: list[object] = []
        spot_observation_events: list[object] = []

        def __init__(self, _connection: object) -> None:
            pass

        def get_trading_days(self, *_args: object) -> list[dt.date]:
            return [signal_day, t_plus_two]

    class Strategy:
        name = "timing-trap"

        def generate_signals(self, *_args: object) -> dict[str, object]:
            calls.append("generated-on-t-plus-two")
            return {}

    class Portfolio:
        def __init__(self, _capital: float) -> None:
            self.open_positions: list[object] = []
            self.closed_trades: list[dict[str, object]] = []
            self.realized_pnl = 0.0

        def total_value(self, *_args: object) -> float:
            return 500_000.0

        def unrealized_pnl(self, *_args: object) -> float:
            return 0.0

        def close_all(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(engine_module, "writer_connection", lambda: Connection())
    monkeypatch.setattr(engine_module, "ensure_backtest_schema", lambda _connection: None)
    monkeypatch.setattr(engine_module, "HistoricalOptionsPricer", Pricer)
    monkeypatch.setattr(engine_module, "get_strategy", lambda *_args: Strategy())
    monkeypatch.setattr(engine_module, "Portfolio", Portfolio)
    monkeypatch.setattr(engine_module, "_benchmark_records", lambda *_args: [])
    monkeypatch.setattr(
        engine_module,
        "compute_metrics",
        lambda **_kwargs: {"total_trades": 0, "strategy_name": "timing-trap"},
    )
    monkeypatch.setattr(engine_module, "regime_attribution", lambda *_args: {})
    monkeypatch.setattr(engine_module, "validate_signals", lambda *_args: {})
    monkeypatch.setattr(engine_module, "stress_test", lambda *_args: {})
    monkeypatch.setattr(
        engine_module.BacktestReporter,
        "generate_full_report",
        lambda *_args: {"text": "", "model": "none", "live": False},
    )

    result = BacktestEngine(
        "timing-trap",
        signal_day,
        t_plus_two,
        chronological_segments=False,
        monte_carlo=False,
    ).run()

    assert calls == []
    assert result["validity_status"] == BacktestValidity.INVALID_TIMING.value
    assert result["daily_records"][-1]["signals"]["execution_skipped"] == "MISSING_EXECUTION_SESSION"
