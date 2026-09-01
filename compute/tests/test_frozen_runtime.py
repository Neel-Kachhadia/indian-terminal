from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import build_compute
from ml import frozen_runtime
from ml.models import m01_vol_regime, m02_direction
from ml.models import common as model_common


def test_xgboost_native_payload_is_explicitly_collected_and_verified() -> None:
    source = build_compute.Path(build_compute.__file__).read_text(encoding="utf-8")
    assert build_compute.NATIVE_BINARY_PACKAGES == ("xgboost",)
    assert build_compute.EXPECTED_FROZEN_NATIVE_BINARIES == {
        "xgboost/lib/xgboost.dll"
    }
    assert build_compute.EXPECTED_FROZEN_PACKAGE_DATA == {"xgboost/VERSION"}
    assert '["--collect-binaries", package]' in source
    assert '["--add-data", f"{source};{destination}"]' in source
    assert source.index("subprocess.run(command") < source.index("verify_frozen_native_payload(exe)")


def test_native_payload_contract_rejects_disposable_package_without_xgboost_dll() -> None:
    with pytest.raises(RuntimeError, match=r"xgboost/lib/xgboost\.dll"):
        build_compute.validate_native_payload_entries(
            {
                "lightgbm/bin/lib_lightgbm.dll",
                "catboost/_catboost.pyd",
                "xgboost/VERSION",
            }
        )


def test_native_payload_contract_accepts_expected_xgboost_destination() -> None:
    build_compute.validate_native_payload_entries(
        {"xgboost/lib/xgboost.dll", "xgboost/VERSION"}
    )


def test_native_payload_contract_rejects_missing_xgboost_version_data() -> None:
    with pytest.raises(RuntimeError, match=r"xgboost/VERSION"):
        build_compute.validate_native_payload_entries({"xgboost/lib/xgboost.dll"})


def test_canonical_frozen_sweep_covers_all_21_model_modules() -> None:
    from ml.models.m21_ohlc_forecast import MODEL_ID as OHLC_MODEL_ID
    from ml.trainer import MODEL_MODULES

    expected = tuple(model_id for model_id, _ in MODEL_MODULES) + (OHLC_MODEL_ID,)
    assert len(expected) == 21
    assert frozen_runtime.canonical_model_ids() == expected


def test_model_sweep_reports_absent_artifacts_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frozen_runtime, "latest_validated_record", lambda *_args: None)
    result = frozen_runtime.model_runtime_sweep("NIFTY")
    assert result["canonical_model_count"] == 21
    assert result["loaded"] == 0
    assert result["honestly_unavailable"] == 21
    assert result["failed"] == 0
    assert all(row["load_result"] == "HONESTLY_UNAVAILABLE" for row in result["models"])


def test_runtime_dependency_failure_classification_is_not_honest_unavailability() -> None:
    error = RuntimeError("Cannot find XGBoost Library\n- C:/temp/_MEI/xgboost/lib/xgboost.dll")
    assert frozen_runtime._classify_load_failure(error) == "FAILED_RUNTIME_DEPENDENCY"


def test_native_load_error_console_is_concise_deduplicated_and_durable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model_common._ARTIFACT_LOAD_FAILURES.clear()
    record = {"artifact_path": "NIFTY/m01_vol_regime_probe.skops"}
    error = RuntimeError(
        "Cannot find XGBoost Library in the candidate path. List of candidates:\n"
        "- C:/temp/_MEI/xgboost/lib/xgboost.dll\n"
        "- C:/temp/_MEI/lib/xgboost.dll"
    )
    with caplog.at_level(logging.ERROR, logger="ml.runtime"):
        try:
            raise error
        except RuntimeError as caught:
            model_common._report_artifact_load_failure("m01_vol_regime", record, caught)
        try:
            raise error
        except RuntimeError as caught:
            model_common._report_artifact_load_failure("m01_vol_regime", record, caught)

    records = [record for record in caplog.records if record.name == "ml.runtime"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None
    message = records[0].getMessage()
    assert "model=m01_vol_regime" in message
    assert "artifact=m01_vol_regime_probe.skops" in message
    assert "state=FAIL_CLOSED" in message
    assert "\n" not in message


def test_missing_xgboost_artifact_never_fabricates_regime_or_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(available_rows=500)
    monkeypatch.setattr(
        m01_vol_regime,
        "train_classifier_artifact",
        lambda *_args, **_kwargs: (None, "runtime dependency unavailable", 500),
    )
    monkeypatch.setattr(m01_vol_regime, "accuracy_for_model", lambda *_args: {"accuracy": None})
    result = m01_vol_regime.predict(context)  # type: ignore[arg-type]
    assert result["is_trained"] is False
    assert result["regime"] is None
    assert result["probabilities"] == {}
    assert result["confidence"] is None


def test_missing_lightgbm_artifact_never_fabricates_direction_or_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(available_rows=500)
    monkeypatch.setattr(
        m02_direction,
        "_train",
        lambda *_args, **_kwargs: (None, "runtime dependency unavailable", 500),
    )
    monkeypatch.setattr(m02_direction, "accuracy_for_model", lambda *_args: {"accuracy": None})
    result = m02_direction.predict(context)  # type: ignore[arg-type]
    assert result["is_trained"] is False
    assert result["direction"] is None
    assert result["probabilities"] == {}
    assert result["confidence"] is None


def test_release_pipeline_runs_frozen_gate_before_openapi_probe() -> None:
    build = (build_compute.ROOT.parent / "build_release.ps1").read_text(encoding="utf-8")
    assert build.index("release_gate.py generate-probe-manifest") < build.index(
        "release_gate.py frozen-runtime"
    )
    assert build.index("release_gate.py frozen-runtime") < build.index(
        "release_gate.py packaged-openapi"
    )
