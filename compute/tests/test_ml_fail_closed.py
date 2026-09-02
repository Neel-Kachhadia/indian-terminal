import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml import service as ml_service
from ml.service import _apex_action
from ml.models import m10_ensemble
from bif.composite import compute_cds
from bif.engine import BIFEngine
from bif import engine as bif_engine_module
from backtest.spot_quality import classify_live_spot
from data.market_data_quality import (
    DecisionContext,
    MarketDataEligibility,
    MarketDataQuality,
)


def _validated_model_output(now: dt.datetime) -> dict[str, object]:
    def output(model_id: str) -> dict[str, object]:
        return {
            "model_id": model_id,
            "is_trained": True,
            "calibrated": True,
            "model_attribution": {
                "artifact_hash": f"sha256:{model_id}",
                "model_version": "1",
                "training_data_horizon": "2025-01-01..2026-08-20",
                "feature_schema_version": "schema-1",
                "artifact_verified": True,
                "registry_status": "VALIDATED",
                "feature_schema_compatible": True,
                "validation_status": "VALIDATED",
            },
        }

    components = {
        model_id: output(model_id)
        for model_id in ("m02_direction", "m17_conformal", "m18_bandit", "m19_dqn", "m20_bnn")
    }
    ensemble = output("m10_ensemble")
    ensemble.update(
        {
            "composite_probabilities": {"UP": 0.999, "DOWN": 0.001},
            "composite_direction": "UP",
            "composite_confidence": 0.999,
            "component_outputs": components,
            "decision_input_evidence": {
                "market_data_quality": "OBSERVED_VERIFIED",
                "market_data_freshness": "CURRENT_SESSION_ELIGIBLE",
                "market_data_provenance": "NSE_INDEX_BHAVCOPY+ANGEL_STREAM",
                "prediction_as_of": (now - dt.timedelta(minutes=1)).isoformat(),
                "requested_decision_at": now.isoformat(),
                "fallback_state": "NONE",
                "input_feature_schema": "schema-1",
            },
        }
    )
    return ensemble


def _market_eligibility(
    now: dt.datetime,
    *,
    valid: bool,
    missing_quality: bool = False,
) -> MarketDataEligibility:
    return MarketDataEligibility(
        DecisionContext.CURRENT_DECISION,
        now,
        now.date(),
        now - dt.timedelta(hours=1),
        "NSE_INDEX_BHAVCOPY" if valid else "NONE" if missing_quality else "UNKNOWN",
        (
            MarketDataQuality.OBSERVED_VERIFIED
            if valid
            else MarketDataQuality.INVALID
            if missing_quality
            else MarketDataQuality.UNKNOWN
        ),
        valid,
        valid,
        valid,
        () if valid else ("market observation semantic quality invalid",),
    )


def test_unvalidated_components_never_emit_actionable_language() -> None:
    result = _apex_action(
        {"composite_direction": "UP", "composite_confidence": 0.99, "uncertainty_level": "LOW", "spot": 24000},
        {"recommended_strategy": "LONG_CALL"},
        {"recommended_size": 1.0},
        {"interval_90": [-1.0, 2.0]},
        {},
        eligibility={"eligible": False, "state": "UNVALIDATED", "reasons": ["artifact missing"]},
    )
    assert result["trade"] is False
    assert result["action"] == "UNVALIDATED"
    assert "LONG" not in result["action"] and "SHORT" not in result["action"]


@pytest.mark.parametrize(
    ("market_valid", "current_spot", "ohlcv_empty"),
    [
        (False, 0.0, True),
        (True, 0.0, False),
        (True, 24_000.0, True),
    ],
)
def test_embedded_ohlc_forecast_fails_closed_without_eligible_spot_and_bars(
    monkeypatch,
    market_valid: bool,
    current_spot: float,
    ohlcv_empty: bool,
) -> None:
    """Regression: /ml/predict once embedded a 0.00-price OHLC forecast
    while the dedicated forecast endpoint correctly reported unavailable.
    """
    now = dt.datetime.now(dt.timezone.utc)
    context = SimpleNamespace(
        market_data_eligibility=_market_eligibility(now, valid=market_valid),
        current_spot=current_spot,
        ohlcv=SimpleNamespace(empty=ohlcv_empty),
    )

    class UnexpectedForecaster:
        def __init__(self) -> None:
            raise AssertionError("forecaster must not run with ineligible decision inputs")

    monkeypatch.setattr(ml_service, "OHLCForecaster", UnexpectedForecaster)

    assert ml_service._ohlc_forecast(context) is None


def test_actual_bif_cds_path_blocks_extreme_untrained_model_output() -> None:
    strong_bull = {"bms": 100.0}
    strong_ngi = {"ngi": 100.0}
    strong_iad = {"iad_score": 100.0}
    strong_vrc = {"clock_label": "COMPRESSION"}
    strong_nfve = {"valuation": "UNDERVALUED"}
    unvalidated_ml = {
        "composite_probabilities": {"UP": 0.9999, "DOWN": 0.0001},
        "composite_direction": "UP",
        "composite_confidence": 0.9999,
        "is_trained": False,
        "model_attribution": {"validation_status": "UNVALIDATED"},
    }

    result = compute_cds(
        strong_bull,
        strong_ngi,
        strong_iad,
        strong_vrc,
        strong_nfve,
        unvalidated_ml,
    )

    assert result["recommendation_state"] == "UNVALIDATED"
    assert result["action"] == "UNVALIDATED"
    assert result["is_valid"] is False
    assert "LONG" not in result["action"] and "SHORT" not in result["action"]


def test_analytical_ensemble_cannot_emit_action_contract(monkeypatch) -> None:
    monkeypatch.setattr(m10_ensemble, "_direction_vectors", lambda *_: ([], []))
    monkeypatch.setattr(m10_ensemble, "_accuracy_tracking", lambda: {})
    context = object()

    result = m10_ensemble.predict(context, {})

    assert "action_recommendation" not in result
    assert "trade" not in result
    assert result["is_trained"] is False


def test_bif_service_boundary_preserves_unvalidated_non_actionable_state(monkeypatch) -> None:
    rows = [
        {
            "symbol": "NIFTY",
            "date": f"2026-07-{day:02d}",
            "open": 24_000 + day,
            "high": 24_020 + day,
            "low": 23_980 + day,
            "close": 24_010 + day,
            "volume": 1_000_000 + day,
        }
        for day in range(1, 29)
    ]
    monkeypatch.setattr(bif_engine_module, "_load_ohlcv_rows", lambda _symbol: rows)
    monkeypatch.setattr(bif_engine_module, "_load_iv_history", lambda _symbol: [])
    monkeypatch.setattr(bif_engine_module, "_load_signal_history", lambda _symbol: {})
    unvalidated_ml = {
        "composite_probabilities": {"UP": 0.9999, "DOWN": 0.0001},
        "composite_direction": "UP",
        "composite_confidence": 0.9999,
        "is_trained": False,
        "model_attribution": {"validation_status": "UNVALIDATED"},
    }
    service = BIFEngine(
        providers={
            "signals": lambda _symbol: {},
            "chain": lambda _symbol: [],
            "spot": lambda _symbol: 24_000,
            "market": lambda: {},
            "cross_asset": lambda: {},
            "ml": lambda *_args: unvalidated_ml,
        }
    )
    monkeypatch.setattr(service, "get_history", lambda *_args, **_kwargs: [])

    result = service.get_cds_only("NIFTY")

    assert result["recommendation_state"] == "UNVALIDATED"
    assert result["action"] == "UNVALIDATED"
    assert result["is_valid"] is False


@pytest.mark.parametrize("missing_quality", [False, True])
def test_actual_bif_business_path_blocks_valid_models_with_invalid_or_missing_market_quality(
    monkeypatch,
    missing_quality: bool,
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    model = _validated_model_output(now)
    spot = classify_live_spot(
        {
            "symbol": "NIFTY",
            "ltp": 24_000,
            "observed_at_ms": int((now - dt.timedelta(seconds=1)).timestamp() * 1000),
            "source_type": "ANGEL_STREAM",
            "quality": "EXECUTABLE_STREAM",
            "freshness": "FRESH",
            "market_state": "LIVE",
            "fallback": False,
        },
        symbol="NIFTY",
        requested_at=now,
    )
    rows = [
        {
            "symbol": "NIFTY",
            "date": now.date().isoformat(),
            "open": 24_000,
            "high": 24_100,
            "low": 23_900,
            "close": 24_050,
            "volume": 1_000_000,
        }
    ]
    monkeypatch.setattr(
        bif_engine_module,
        "_load_current_ohlcv_rows",
        lambda _symbol, requested_at: (
            rows,
            _market_eligibility(
                requested_at,
                valid=False,
                missing_quality=missing_quality,
            ),
        ),
    )
    monkeypatch.setattr(bif_engine_module, "_load_iv_history", lambda _symbol: [])
    monkeypatch.setattr(bif_engine_module, "_load_signal_history", lambda _symbol: {})
    service = BIFEngine(
        providers={
            "signals": lambda _symbol: {},
            "chain": lambda _symbol: [],
            "spot": lambda _symbol: spot,
            "market": lambda: {},
            "cross_asset": lambda: {},
            "ml": lambda *_args: model,
        }
    )
    monkeypatch.setattr(service, "get_history", lambda *_args, **_kwargs: [])

    result = service.get_cds_only("NIFTY")

    assert result["recommendation_state"] == "UNVALIDATED"
    assert result["action"] == "UNVALIDATED"
    assert result["data_quality_state"] == "INVALID"
    assert result["is_valid"] is False


def test_actual_bif_rejects_naked_float_and_does_not_rescue_from_option_strike(
    monkeypatch,
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    rows = [
        {
            "symbol": "NIFTY",
            "date": now.date().isoformat(),
            "open": 24_000,
            "high": 24_100,
            "low": 23_900,
            "close": 24_050,
            "volume": 1_000_000,
        }
    ]
    monkeypatch.setattr(
        bif_engine_module,
        "_load_current_ohlcv_rows",
        lambda _symbol, requested_at: (rows, _market_eligibility(requested_at, valid=True)),
    )
    monkeypatch.setattr(bif_engine_module, "_load_iv_history", lambda _symbol: [])
    monkeypatch.setattr(bif_engine_module, "_load_signal_history", lambda _symbol: {})
    service = BIFEngine(
        providers={
            "signals": lambda _symbol: {},
            "chain": lambda _symbol: {
                "expiry": "2026-08-27",
                "legs": [{"strike": 24_100, "option_type": "CE", "ltp": 100}],
            },
            "spot": lambda _symbol: 24_000.0,
            "market": lambda: {},
            "cross_asset": lambda: {},
            "ml": lambda *_args: _validated_model_output(now),
        }
    )
    monkeypatch.setattr(service, "get_history", lambda *_args, **_kwargs: [])

    full = service.compute_all("NIFTY")

    assert full["spot"] is None
    assert full["spot_provenance"]["decision_eligible"] is False
    assert full["spot_provenance"]["reason"] == "spot_provider_did_not_return_provenance"
    assert full["cds"]["recommendation_state"] == "UNVALIDATED"
    assert full["cds"]["is_valid"] is False


def test_bif_frontend_never_reconstructs_action_from_numeric_score() -> None:
    component_root = Path(__file__).resolve().parents[2] / "src" / "components"
    bif = (component_root / "BIFPanel" / "index.tsx").read_text(encoding="utf-8")
    assert 'recommendation_state === "ACTIONABLE"' in bif
    assert "cds > 80" not in bif
    assert "cds < 20" not in bif
    for relative in ("MLPanel/index.tsx", "DecisionIntelligencePanel/index.tsx"):
        source = (component_root / relative).read_text(encoding="utf-8")
        assert "actionEligible" in source
        assert '"NON-ACTIONABLE"' in source
