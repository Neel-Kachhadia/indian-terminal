from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bif.accumulation import compute_iad
from bif.composite import compute_cds
from bif.engine import _current_iv_observations, _validated_chain
from bif.fair_value import compute_nfve
from bif.gravity import compute_ngi
from bif.momentum import compute_bms
from data.exchange_calendar import today_ist
from data.index_aliases import index_symbol_candidates
from ml.feature_engineering import MLContext
from ml.models import m04_iv_reversion, m05_anomaly, m06_regime_return, m10_ensemble
from ml.monitor import log_prediction


def _context(frame: pd.DataFrame, *, iv_rows: int = 0) -> MLContext:
    return MLContext(
        symbol="NIFTY",
        signal_payload={},
        current_features={},
        feature_frame=frame,
        ohlcv=pd.DataFrame(),
        iv_history=pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=iv_rows),
                "atm_iv": np.linspace(0.15, 0.20, iv_rows),
            }
        ),
        bif_history=pd.DataFrame(),
        fo_eod_rows=0,
        current_spot=None,
        spot_provenance={},
        market_data_eligibility=None,
        decision_requested_at=dt.datetime.now(dt.timezone.utc),
    )


def _eligible_direction(probabilities: dict[str, float]) -> dict[str, object]:
    return {
        "is_trained": True,
        "calibrated": True,
        "feature_schema_compatible": True,
        "prediction": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
        "model_attribution": {
            "artifact_verified": True,
            "registry_status": "VALIDATED",
        },
    }


def _accuracy_stub(*_args, **_kwargs) -> dict[str, object]:
    return {"accuracy": None, "sample_count": 0}


def test_canonical_index_alias_includes_persisted_nifty_name() -> None:
    assert index_symbol_candidates("NIFTY") == ["NIFTY 50", "NIFTY"]
    assert index_symbol_candidates("NIFTY50") == ["NIFTY 50", "NIFTY", "NIFTY50"]
    assert index_symbol_candidates("BANKNIFTY") == ["NIFTY BANK", "BANKNIFTY"]


def test_anomaly_preprocessor_excludes_current_row_and_all_null_features() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=31),
            "feature_b": np.arange(31, dtype=float),
            "feature_a": np.arange(31, dtype=float) * 2,
            "all_null": [np.nan] * 31,
        }
    )
    frame.loc[30, "feature_b"] = 10_000.0
    prepared = m05_anomaly._prepare_matrix(_context(frame))

    assert prepared["matrix"].shape == (30, 2)
    assert prepared["columns"] == ["feature_b", "feature_a"]
    assert prepared["excluded_features"] == ["all_null"]
    assert prepared["matrix"][-1, 0] == 29.0
    assert prepared["current"][0] == 10_000.0


def test_anomaly_insufficient_history_is_unavailable_not_normal() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=30),
            "feature": np.arange(30, dtype=float),
        }
    )
    output = m05_anomaly.predict(_context(frame))
    assert output["status"] == "INSUFFICIENT_DATA"
    assert output["prediction"] is None
    assert output["is_anomalous"] is None
    assert output["anomaly_score"] is None
    assert output["confidence_multiplier"] is None


@pytest.mark.parametrize(
    ("left", "right", "expected_direction", "expected_agreement"),
    [
        (
            {"UP": 0.8, "FLAT": 0.1, "DOWN": 0.1},
            {"UP": 0.7, "FLAT": 0.2, "DOWN": 0.1},
            "UP",
            1.0,
        ),
        (
            {"UP": 0.8, "FLAT": 0.1, "DOWN": 0.1},
            {"UP": 0.1, "FLAT": 0.1, "DOWN": 0.8},
            "UP",
            0.5,
        ),
    ],
)
def test_ensemble_agreement_is_weighted_constituent_vote_share(
    monkeypatch,
    left: dict[str, float],
    right: dict[str, float],
    expected_direction: str,
    expected_agreement: float,
) -> None:
    monkeypatch.setattr(m10_ensemble, "accuracy_for_model", _accuracy_stub)
    output = m10_ensemble.predict(
        object(),
        {
            "m02_direction": _eligible_direction(left),
            "m06_regime_return": _eligible_direction(right),
            "m05_anomaly": {"is_valid": True, "is_anomalous": False, "confidence_multiplier": 1.0},
        },
    )
    assert output["composite_direction"] == expected_direction
    assert output["model_agreement"] == expected_agreement
    assert output["eligible_component_count"] == 2
    assert output["weight_sum"] == 2.0


def test_unavailable_ensemble_model_has_no_vote_or_denominator_weight(monkeypatch) -> None:
    monkeypatch.setattr(m10_ensemble, "accuracy_for_model", _accuracy_stub)
    components = {
        "m02_direction": _eligible_direction({"UP": 0.8, "FLAT": 0.1, "DOWN": 0.1}),
        "m06_regime_return": {
            "is_trained": False,
            "probabilities": {"UP": 0.0, "FLAT": 1.0, "DOWN": 0.0},
            "reason_code": "ARTIFACT_NOT_REGISTERED",
        },
    }
    output = m10_ensemble.predict(object(), components)
    assert output["eligible_component_count"] == 1
    assert output["weight_sum"] == 1.0
    assert output["evidence_coverage"] == 0.5
    assert output["composite_direction"] == "UP"
    assert output["composite_confidence"] == pytest.approx(0.4)
    assert output["excluded_components"] == [
        {"model_id": "m06_regime_return", "reason": "ARTIFACT_NOT_REGISTERED"}
    ]


def test_majority_unavailable_ensemble_is_explicitly_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(m10_ensemble, "accuracy_for_model", _accuracy_stub)
    output = m10_ensemble.predict(object(), {})
    assert output["status"] == "UNAVAILABLE"
    assert output["composite_direction"] is None
    assert output["composite_probabilities"] == {}
    assert output["composite_confidence"] is None
    assert output["model_agreement"] is None
    assert output["weight_sum"] == 0.0


def test_valid_anomaly_gate_suppresses_confidence_but_unavailable_gate_does_not(
    monkeypatch,
) -> None:
    monkeypatch.setattr(m10_ensemble, "accuracy_for_model", _accuracy_stub)
    direction = _eligible_direction({"UP": 0.8, "FLAT": 0.1, "DOWN": 0.1})
    base = {"m02_direction": direction, "m06_regime_return": direction}
    gated = m10_ensemble.predict(
        object(),
        {**base, "m05_anomaly": {"is_valid": True, "is_anomalous": True, "confidence_multiplier": 0.2}},
    )
    unavailable = m10_ensemble.predict(
        object(),
        {**base, "m05_anomaly": {"is_valid": False, "confidence_multiplier": None}},
    )
    assert gated["composite_confidence"] == pytest.approx(0.16)
    assert unavailable["composite_confidence"] == pytest.approx(0.8)


def test_model_inference_never_trains_missing_m04_or_m06_artifact(monkeypatch) -> None:
    frame = pd.DataFrame({"date": [dt.date(2026, 1, 1)], "feature": [1.0]})
    context = _context(frame, iv_rows=2)
    calls: list[bool] = []

    def fake_train(*_args, **kwargs):
        calls.append(kwargs["train_if_missing"])
        return None, "model not trained for symbol", 1

    monkeypatch.setattr(m04_iv_reversion, "train_classifier_artifact", fake_train)
    monkeypatch.setattr(m06_regime_return, "train_classifier_artifact", fake_train)
    monkeypatch.setattr(m04_iv_reversion, "accuracy_for_model", _accuracy_stub)
    monkeypatch.setattr(m06_regime_return, "accuracy_for_model", _accuracy_stub)
    assert m04_iv_reversion.predict(context)["prediction"] is None
    assert m06_regime_return.predict(context)["prediction"] is None
    assert calls == [False, False]


def test_ngi_missing_inputs_stay_missing_and_real_zero_is_preserved() -> None:
    expiry = (today_ist() + dt.timedelta(days=4)).isoformat()
    missing = compute_ngi([], 0.0, expiry)
    assert missing["ngi"] is None
    assert missing["ngi_direction"] is None
    assert missing["pin_probability"] is None
    assert missing["is_valid"] is False

    balanced = compute_ngi(
        [
            {"strike": 110, "oi": 1000, "option_type": "CE"},
            {"strike": 90, "oi": 1000, "option_type": "PE"},
        ],
        100.0,
        expiry,
    )
    assert balanced["is_valid"] is True
    assert balanced["ngi"] == 0.0
    assert balanced["ngi_direction"] == "NEUTRAL"


def test_bms_and_iad_do_not_treat_missing_options_as_zero() -> None:
    rows = [
        {"close": 100.0 + index, "volume": 1_000_000 + index * 1000}
        for index in range(30)
    ]
    expiry = (today_ist() + dt.timedelta(days=4)).isoformat()
    bms = compute_bms(
        rows,
        {},
        {},
        expiry,
        {"component_outputs": {"m01_vol_regime": {"is_trained": True, "prediction": "MEDIUM"}}},
        {"fii_net_cash": [100.0], "dii_net_cash": [50.0]},
        {},
    )
    assert bms["options_flow_momentum"] is None
    assert bms["bms"] is None
    assert bms["is_valid"] is False

    iad = compute_iad(
        rows,
        [],
        {"fii_net_cash": [100.0, 110.0, 120.0]},
        {},
    )
    assert iad["options_hedge_score"] is None
    assert iad["iad_score"] is not None
    assert iad["is_valid"] is True
    assert iad["component_coverage"] == pytest.approx(0.65)


def test_nfve_cannot_classify_without_measured_financial_inputs() -> None:
    output = compute_nfve("NIFTY", 24_000.0, [], {}, {}, {}, {})
    assert output["is_valid"] is False
    assert output["fair_value"] is None
    assert output["valuation"] is None
    assert "index_eps" in output["missing_inputs"]


def test_cds_coverage_caps_confidence_and_missing_components_stay_none() -> None:
    output = compute_cds(
        {"bms": 0.0, "is_valid": True},
        {"ngi": 0.0, "is_valid": True},
        {"iad_score": None, "is_valid": False, "reason_code": "MISSING"},
        {"is_valid": False, "reason_code": "NO_CURRENT_IV"},
        {"is_valid": False, "reason_code": "MISSING_MEASURED_FINANCIAL_INPUTS"},
        {},
    )
    assert output["cds"] == 50.0
    assert output["component_scores"]["iad"] is None
    assert output["component_weight_coverage"] == pytest.approx(0.4)
    assert output["cds_confidence"] == pytest.approx(0.4)


def test_chain_requires_fresh_provenance_and_future_iv_rows_are_real_only() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    legs = [
        {"strike": 100.0, "option_type": "CE", "iv": 20.0, "oi": 100},
        {"strike": 100.0, "option_type": "PE", "iv": 22.0, "oi": 100},
    ]
    invalid_legs, _, invalid_provenance = _validated_chain(
        legs,
        "2026-09-24",
        {},
        now,
    )
    assert invalid_legs == []
    assert invalid_provenance["decision_eligible"] is False

    validated_legs, expiry, provenance = _validated_chain(
        legs,
        "2026-09-24",
        {
            "source": "NSE_GET_SYMBOL_DERIVATIVES_DATA",
            "observed_at": (now - dt.timedelta(seconds=1)).isoformat(),
            "freshness": "FRESH",
            "decision_eligible": True,
        },
        now,
    )
    observations = _current_iv_observations(
        "NIFTY",
        validated_legs,
        100.0,
        expiry,
        "2026-09-02",
        provenance,
        True,
    )
    assert [row["iv"] for row in observations] == [0.2, 0.22]
    assert all(row["quality"] == "CURRENT_DECISION_ELIGIBLE" for row in observations)


def test_unavailable_prediction_is_rejected_before_oos_log_write() -> None:
    with pytest.raises(ValueError, match="unavailable prediction"):
        log_prediction(
            "m02_direction",
            None,
            {},
            None,
            {},
            attribution={},
        )
