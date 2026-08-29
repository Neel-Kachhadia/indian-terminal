from __future__ import annotations

import datetime as dt
from typing import Any

from ml.feature_engineering import build_context
from ml.monitor import accuracy_for_model, accuracy_report
from ml.models import (
    m01_vol_regime,
    m02_direction,
    m03_strategy_selector,
    m04_iv_reversion,
    m05_anomaly,
    m06_regime_return,
    m07_iv_forecast,
    m08_gp_regime,
    m09_rl_sizer,
    m10_ensemble,
    m11_tft,
    m12_prophet,
    m13_nbeats,
    m14_wavelet,
    m15_kalman,
    m16_survival,
    m17_conformal,
    m18_bandit,
    m19_dqn,
    m20_bnn,
)
from ml.models.common import SAVED_MODELS_DIR, model_status
from ml.training_log import log_training_result


MODEL_MODULES = [
    ("m01_vol_regime", m01_vol_regime),
    ("m02_direction", m02_direction),
    ("m03_strategy", m03_strategy_selector),
    ("m04_iv_reversion", m04_iv_reversion),
    ("m05_anomaly", m05_anomaly),
    ("m06_regime_return", m06_regime_return),
    ("m07_iv_forecast", m07_iv_forecast),
    ("m08_gp_regime", m08_gp_regime),
    ("m09_rl_sizer", m09_rl_sizer),
    ("m10_ensemble", m10_ensemble),
    ("m11_tft", m11_tft),
    ("m12_prophet", m12_prophet),
    ("m13_nbeats", m13_nbeats),
    ("m14_wavelet", m14_wavelet),
    ("m15_kalman", m15_kalman),
    ("m16_survival", m16_survival),
    ("m17_conformal", m17_conformal),
    ("m18_bandit", m18_bandit),
    ("m19_dqn", m19_dqn),
    ("m20_bnn", m20_bnn),
]


def train_all_models(force: bool = False, symbol: str = "NIFTY") -> dict[str, Any]:
    context = build_context(symbol)
    report: dict[str, Any] = {
        "symbol": context.symbol,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "force": force,
        "feature_rows": context.available_rows,
        "models": {},
    }
    for model_id, module in MODEL_MODULES:
        try:
            result = module.train_model(context, force=force)
        except Exception as error:
            result = {
                "model_id": model_id,
                "is_trained": False,
                "reason": str(error),
            }
        report["models"][model_id] = result
        log_training_result(context.symbol, model_id, result, context=context)
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return report


def load_all_models() -> dict[str, Any]:
    return {model_id: model_status(model_id, "NIFTY") for model_id, _ in MODEL_MODULES}


def evaluate_model(model_id: str, n_days: int = 30) -> dict[str, Any]:
    return accuracy_for_model(model_id, n_days)


def status_report() -> dict[str, Any]:
    return {
        "saved_models_dir": str(SAVED_MODELS_DIR),
        "models": load_all_models(),
        "accuracy": accuracy_report(30),
    }
