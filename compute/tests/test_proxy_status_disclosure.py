from __future__ import annotations

import pytest

from ml.service import _normalise_runtime_status


@pytest.mark.contract
def test_own_symbol_trained_output_stays_ready() -> None:
    output = {
        "is_trained": True,
        "prediction": "UP",
        "model_attribution": {"fallback_status": "PRIMARY", "trained_symbol": "RELIANCE"},
    }
    _normalise_runtime_status(output)
    assert output["status"] == "READY"


@pytest.mark.contract
def test_proxy_served_output_is_never_indistinguishable_from_ready() -> None:
    """Reproduces a live defect: _normalise_runtime_status set status="READY"
    purely from is_trained+prediction, never checking model_attribution --
    so a component served off PROXY_SYMBOL's (NIFTY's) artifact for a symbol
    with no artifact of its own (reproduced live: LTM, 116 own rows) reported
    the exact same top-level "READY" a genuinely own-symbol-trained
    component reports. requested_symbol != trained_symbol must be obvious
    from the top-level status a naive consumer/GUI reads, not only from the
    nested model_attribution dict."""
    output = {
        "is_trained": True,
        "prediction": "FLAT",
        "model_attribution": {
            "fallback_status": "PROXY",
            "trained_symbol": "NIFTY",
            "requested_symbol": "LTM",
            "feature_schema_compatible": False,
        },
    }
    _normalise_runtime_status(output)
    assert output["status"] == "READY_PROXY"
    assert output["status"] != "READY"
    assert "NIFTY" in str(output["reason_code"])


@pytest.mark.contract
def test_explicit_status_is_never_overridden() -> None:
    output = {"status": "FAILED", "is_trained": True, "prediction": "UP"}
    _normalise_runtime_status(output)
    assert output["status"] == "FAILED"
