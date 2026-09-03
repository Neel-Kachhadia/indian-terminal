from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import main
from intelligence.options_flow import compute_options_flow
from risk.greeks_dashboard import compute_greeks_dashboard
from risk.stress_test import compute_stress_tests
from risk.var_engine import compute_var
from signals.aggregator import compute_all_signals


def test_apex_inputs_preserves_unavailable_live_spot_as_none(monkeypatch) -> None:
    observation = SimpleNamespace(
        price=24_000.0,
        decision_eligible=False,
        as_provenance=lambda: {"decision_eligible": False, "reason": "stale"},
    )
    eligibility = SimpleNamespace(decision_eligible=False, to_dict=lambda: {"decision_eligible": False})
    monkeypatch.setattr(main, "get_next_expiries", lambda _count, _symbol=None: ["2026-09-01"])
    monkeypatch.setattr(main, "get_chain_from_redis", lambda *_args: [])
    monkeypatch.setattr(main, "_read_chain_legs_from_redis", lambda *_args: [])
    monkeypatch.setattr(main, "_read_spot_observation_from_redis", lambda _symbol: observation)
    monkeypatch.setattr(main.instrument_master, "get_lot_size", lambda *_args: 65)
    monkeypatch.setattr(main, "_load_ohlcv_frame", lambda _symbol: pd.DataFrame())
    monkeypatch.setattr(main, "_current_market_data_eligibility", lambda _frame: eligibility)
    monkeypatch.setattr(main, "get_signals", lambda _symbol: {})
    monkeypatch.setattr(main, "get_all_signals", lambda _symbol: {})

    result = main._apex_inputs("NIFTY")

    assert result["spot"] is None
    assert result["signals"]["spot"] is None
    assert result["spot_provenance"]["decision_eligible"] is False


def test_current_risk_and_options_metrics_fail_closed_without_verified_spot() -> None:
    frame = pd.DataFrame({"close": range(100, 140)})
    legs = [{"strike": 100.0, "option_type": "CE", "oi": 1_000, "iv": 0.2}]

    outputs = (
        compute_options_flow("NIFTY", legs, None),
        compute_greeks_dashboard("NIFTY", legs, None),
        compute_var("NIFTY", frame, spot=None, lot_size=65),
        compute_stress_tests("NIFTY", frame, spot=None, lot_size=65),
    )

    for output in outputs:
        assert output["error"] == "verified live spot unavailable"
        assert "spot" not in output or output["spot"] is None


def test_signal_catalogue_does_not_compute_with_zero_spot_sentinel() -> None:
    output = compute_all_signals(
        "NIFTY",
        pd.DataFrame({"close": [24_000.0, 24_100.0]}),
        [],
        None,
        "2026-09-01",
        {},
        {},
        {},
        None,
        lot_size=65,
    )

    assert output["spot"] is None
    assert output["error"] == "verified live spot unavailable"
    assert output["categories"] == {}
    assert output["duckdb_rows_written"] == 0


def test_var_and_stress_use_explicit_verified_spot() -> None:
    frame = pd.DataFrame({"close": range(100, 140)})

    var = compute_var("NIFTY", frame, spot=24_000.0, lot_size=65)
    stress = compute_stress_tests("NIFTY", frame, spot=24_000.0, lot_size=65)

    assert var["notional_per_lot"] == 1_560_000.0
    assert stress["historical_scenarios"][0]["estimated_spot"] == 14_880.0
