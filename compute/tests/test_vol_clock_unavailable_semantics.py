"""Regression coverage for the vol_clock/composite null-semantics fix.

Prior behavior (the bug): when IV data was unavailable or history too sparse
to compute a percentile, `bif/vol_clock.py::compute_vrc` fabricated
`clock_label: "COMPRESSION"` and `iv_percentile: 50.0`. Because "COMPRESSION"
is *also* a genuine volatility-regime label, `bif/composite.py::compute_cds`
could not distinguish "we have no IV data" from "the market is genuinely in
compression", and injected `VRC_SCORE["COMPRESSION"] = 70.0` into the
composite decision score as if it were a real observation.

The fix: unavailable/insufficient-history states return an explicit
`is_valid: False` result with reason-code labels ("NO_DATA",
"INSUFFICIENT_HISTORY") that are not keys in VRC_SCORE, and
`compute_cds` excludes the vrc component from the weighted composite
(renormalizing remaining weights) whenever `vrc.get("is_valid")` is falsy,
instead of defaulting to a fabricated score.
"""

import pytest

from bif.composite import VRC_SCORE, compute_cds
from bif.vol_clock import compute_vrc


def test_compute_vrc_returns_explicit_unavailable_when_no_iv_data() -> None:
    result = compute_vrc(
        iv_history=[],
        option_signals={},
        ohlcv_rows=[],
        signal_values={},
    )
    assert result["is_valid"] is False
    assert result["clock_label"] == "NO_DATA"
    assert result["iv_percentile"] is None
    assert result["clock_position"] is None
    assert result["clock_label"] not in VRC_SCORE


def test_compute_vrc_returns_explicit_unavailable_when_history_too_sparse() -> None:
    result = compute_vrc(
        iv_history=[],
        option_signals={"atm_iv": 0.20},
        ohlcv_rows=[],
        signal_values={},
    )
    assert result["is_valid"] is False
    assert result["clock_label"] == "INSUFFICIENT_HISTORY"
    assert result["iv_percentile"] is None
    assert result["clock_label"] not in VRC_SCORE


def test_compute_vrc_resolves_real_regime_with_sufficient_history() -> None:
    iv_history = [{"atm_iv": 0.15 + 0.01 * (i % 5)} for i in range(40)]
    result = compute_vrc(
        iv_history=iv_history,
        option_signals={"atm_iv": 0.19},
        ohlcv_rows=[],
        signal_values={},
    )
    assert result["is_valid"] is True
    assert result["clock_label"] in VRC_SCORE
    assert isinstance(result["iv_percentile"], float)
    assert 0.0 <= result["iv_percentile"] <= 100.0


def _flat_components(vrc: dict) -> dict:
    return dict(
        bms={"bms": 0.0},
        ngi={"ngi": 0.0},
        iad={"iad_score": 0.0},
        vrc=vrc,
        nfve={"valuation": "FAIRLY_VALUED"},
        ml_prediction={},
    )


def test_compute_cds_excludes_unavailable_vrc_instead_of_fabricating_a_score() -> None:
    """All other components normalize to exactly 50.0 by construction, so any
    deviation from cds == 50.0 must come from the vrc component. An
    unavailable vrc read (no `is_valid: True`) must contribute nothing --
    not the pre-fix behavior of injecting VRC_SCORE["COMPRESSION"] = 70.0.
    """
    unavailable_vrc = {"clock_label": "COMPRESSION"}  # no is_valid -> unavailable
    result = compute_cds(**_flat_components(unavailable_vrc))
    assert result["component_scores"]["vrc"] is None
    assert result["cds"] == pytest.approx(50.0, abs=1e-6)


def test_compute_cds_uses_vrc_only_when_explicitly_valid() -> None:
    valid_vrc = {"clock_label": "COMPRESSION", "is_valid": True}
    result = compute_cds(**_flat_components(valid_vrc))
    expected_cds = 50.0 * 0.85 + VRC_SCORE["COMPRESSION"] * 0.15
    assert result["component_scores"]["vrc"] == pytest.approx(70.0)
    assert result["cds"] == pytest.approx(expected_cds, abs=1e-6)
    assert result["cds"] != pytest.approx(50.0, abs=1e-6)


def test_compute_cds_reason_code_labels_never_collide_with_a_real_regime() -> None:
    reason_code_vrc = {"clock_label": "NO_DATA", "is_valid": False}
    result = compute_cds(**_flat_components(reason_code_vrc))
    assert result["component_scores"]["vrc"] is None
    assert result["cds"] == pytest.approx(50.0, abs=1e-6)
