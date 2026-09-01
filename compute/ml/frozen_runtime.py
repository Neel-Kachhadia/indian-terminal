from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from ml.models.common import SAVED_MODELS_DIR, load_verified_artifact
from ml.registry import latest_validated_record, read_verified_artifact_bytes, validated_records


CANONICAL_MODEL_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("m01_vol_regime", "XGBoost classifier", "xgboost/lib/xgboost.dll"),
    ("m02_direction", "LightGBM classifier", "lightgbm/bin/lib_lightgbm.dll"),
    ("m03_strategy", "strategy selector", None),
    ("m04_iv_reversion", "IV mean-reversion classifier", None),
    ("m05_anomaly", "anomaly detector", None),
    ("m06_regime_return", "CatBoost classifier", "catboost/_catboost.pyd"),
    ("m07_iv_forecast", "IV forecast", None),
    ("m08_gp_regime", "Gaussian-process regime model", None),
    ("m09_rl_sizer", "reinforcement-learning sizer", None),
    ("m10_ensemble", "validated ensemble", None),
    ("m11_tft", "temporal fusion model", None),
    ("m12_prophet", "trend forecast", None),
    ("m13_nbeats", "N-BEATS forecast", None),
    ("m14_wavelet", "wavelet model", None),
    ("m15_kalman", "Kalman model", None),
    ("m16_survival", "survival model", None),
    ("m17_conformal", "conformal wrapper", None),
    ("m18_bandit", "contextual bandit", None),
    ("m19_dqn", "DQN policy", None),
    ("m20_bnn", "Bayesian neural model", None),
    ("m21_ohlc", "OHLC weighted forecast", None),
)

FROZEN_PROBE_MODEL_IDS = ("m01_vol_regime", "m02_direction", "m06_regime_return")


def canonical_model_ids() -> tuple[str, ...]:
    return tuple(model_id for model_id, _, _ in CANONICAL_MODEL_SPECS)


def model_runtime_sweep(
    symbol: str | None = None,
    *,
    model_ids: Iterable[str] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    selected = set(model_ids) if model_ids is not None else None
    requested = symbol.strip().upper() if symbol else None
    checked_at = now or dt.datetime.now(dt.timezone.utc)
    rows: list[dict[str, Any]] = []
    for model_id, framework, native_dependency in CANONICAL_MODEL_SPECS:
        if selected is not None and model_id not in selected:
            continue
        row: dict[str, Any] = {
            "model": model_id,
            "symbol": requested,
            "artifact": None,
            "framework": framework,
            "model_version": None,
            "load_result": "HONESTLY_UNAVAILABLE",
            "native_dependency": native_dependency,
            "artifact_hash_check": "NOT_APPLICABLE",
            "schema_compatibility": "NOT_APPLICABLE",
            "freshness_state": "UNKNOWN",
            "actionability_state": "NOT_EVALUATED_RUNTIME_ONLY",
            "failure_reason": "no validated SKOPS artifact registered",
        }
        record = _latest_record(model_id, requested)
        if record is None:
            rows.append(row)
            continue

        row.update(
            {
                "artifact": str(record["artifact_path"]),
                "symbol": str(record["symbol"]),
                "model_version": str(record["model_version"]),
                "artifact_hash_check": "FAIL",
                "schema_compatibility": "UNKNOWN",
                "failure_reason": None,
            }
        )
        verified = read_verified_artifact_bytes(
            record,
            SAVED_MODELS_DIR,
            maximum_bytes=512 * 1024 * 1024,
        )
        if verified is None:
            row["load_result"] = "FAILED_INTEGRITY"
            row["failure_reason"] = "artifact hash, size, path, or file-type verification failed"
            rows.append(row)
            continue
        row["artifact_hash_check"] = "PASS"
        try:
            _, artifact = load_verified_artifact(record, SAVED_MODELS_DIR)
        except Exception as error:
            row["load_result"] = _classify_load_failure(error)
            row["failure_reason"] = _concise_error(error)
            rows.append(row)
            continue

        schema_ok = artifact.get("feature_schema_version") == record.get("feature_schema_version")
        row["schema_compatibility"] = "PASS" if schema_ok else "FAIL"
        row["freshness_state"] = _freshness_state(str(record["training_timestamp"]), checked_at)
        if not schema_ok:
            row["load_result"] = "FAILED_SCHEMA"
            row["failure_reason"] = "artifact feature schema does not match registry"
        else:
            row["load_result"] = "PASS"
        rows.append(row)

    return {
        "symbol": requested or "CANONICAL_LATEST",
        "canonical_model_count": len(rows),
        "loaded": sum(row["load_result"] == "PASS" for row in rows),
        "honestly_unavailable": sum(row["load_result"] == "HONESTLY_UNAVAILABLE" for row in rows),
        "failed": sum(
            row["load_result"] not in {"PASS", "HONESTLY_UNAVAILABLE"} for row in rows
        ),
        "failed_runtime_dependency": sum(
            row["load_result"] == "FAILED_RUNTIME_DEPENDENCY" for row in rows
        ),
        "failed_deserialization": sum(
            row["load_result"] == "FAILED_DESERIALIZATION" for row in rows
        ),
        "models": rows,
    }


def frozen_probe() -> dict[str, Any]:
    if not getattr(sys, "frozen", False):
        raise RuntimeError("frozen-runtime probe requires a frozen executable")
    extraction_root = Path(str(getattr(sys, "_MEIPASS", ""))).resolve()
    if not extraction_root.is_dir():
        raise RuntimeError("PyInstaller extraction root unavailable")

    import catboost
    import catboost._catboost as catboost_native
    import lightgbm
    import lightgbm.basic as lightgbm_basic
    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn
    import skops
    import xgboost
    import xgboost.libpath as xgboost_libpath

    native_paths = {
        "xgboost": Path(xgboost_libpath.find_lib_path()[0]).resolve(),
        "lightgbm": Path(str(lightgbm_basic._LIB._name)).resolve(),
        "catboost": Path(str(catboost_native.__file__)).resolve(),
    }
    package_modules = {
        "numpy": numpy,
        "pandas": pandas,
        "scipy": scipy,
        "scikit-learn": sklearn,
        "skops": skops,
        "pyarrow": pyarrow,
        "lightgbm": lightgbm,
        "catboost": catboost,
        "xgboost": xgboost,
    }
    for name, module in package_modules.items():
        module_path = Path(str(module.__file__)).resolve()
        if not module_path.is_relative_to(extraction_root):
            raise RuntimeError(f"{name} resolved outside frozen payload")
    outside = {
        name: str(path)
        for name, path in native_paths.items()
        if not path.is_file() or not path.is_relative_to(extraction_root)
    }
    if outside:
        raise RuntimeError(f"native runtime resolved outside frozen payload: {sorted(outside)}")
    dev_paths = [
        value
        for value in sys.path
        if ".venv" in value.lower() or "indian-terminal" in value.lower()
    ]
    if dev_paths:
        raise RuntimeError("frozen sys.path contains development paths")

    sweep = model_runtime_sweep("FROZEN_PROBE", model_ids=FROZEN_PROBE_MODEL_IDS)
    if sweep["loaded"] != len(FROZEN_PROBE_MODEL_IDS) or sweep["failed"]:
        raise RuntimeError(f"frozen model artifact load failed: {sweep}")
    return {
        "status": "PASS",
        "frozen": True,
        "xgboost_version": str(xgboost.__version__),
        "package_versions": {
            name: str(getattr(module, "__version__", "UNKNOWN"))
            for name, module in package_modules.items()
        },
        "native_payload_local": True,
        "dev_path_dependency": False,
        "global_python_dependency": False,
        "models": sweep,
    }


def print_sweep(symbol: str | None = None) -> int:
    result = model_runtime_sweep(symbol)
    print("FROZEN_MODEL_SWEEP=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 1 if result["failed"] else 0


def print_probe() -> int:
    result = frozen_probe()
    print("FROZEN_RUNTIME_PROBE=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _freshness_state(timestamp: str, now: dt.datetime) -> str:
    try:
        trained_at = dt.datetime.fromisoformat(timestamp)
    except ValueError:
        return "INVALID"
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=dt.timezone.utc)
    age = now.astimezone(dt.timezone.utc) - trained_at.astimezone(dt.timezone.utc)
    return "FRESH" if 0 <= age.total_seconds() <= 24 * 3600 else "STALE"


def _latest_record(model_id: str, symbol: str | None) -> dict[str, Any] | None:
    if symbol is not None:
        return latest_validated_record(model_id, symbol)
    records = validated_records([model_id])
    return records[0] if records else None


def _classify_load_failure(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    markers = (
        "cannot find xgboost library",
        "xgboost.dll",
        "lib_lightgbm",
        "_catboost",
        "dll load failed",
        "no module named",
        "packagenotfounderror",
    )
    if isinstance(error, (ImportError, OSError)) or any(marker in text for marker in markers):
        return "FAILED_RUNTIME_DEPENDENCY"
    return "FAILED_DESERIALIZATION"


def _concise_error(error: Exception) -> str:
    text = str(error).strip()
    return next((line.strip() for line in text.splitlines() if line.strip()), type(error).__name__)[:240]
