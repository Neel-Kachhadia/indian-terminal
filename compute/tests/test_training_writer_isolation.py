from __future__ import annotations

from datetime import date

import main
from auto_collector import smart_collector
from ml import stock_trainer, trainer, training_log


def _forbid_writer_training(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("model training must not enter DuckDB writer queue")


def test_collector_model_training_runs_outside_duckdb_writer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smart_collector, "submit_write", _forbid_writer_training)
    monkeypatch.setattr(
        trainer,
        "train_all_models",
        lambda **_kwargs: {"feature_rows": 250, "models": {"m01": {"artifact_ready": True}}},
    )
    monkeypatch.setattr(
        stock_trainer.StockModelTrainer,
        "train_all_fno_stocks",
        lambda self, **_kwargs: {"total": 214, "trained": 3},
    )

    weekly = smart_collector.run_weekly_retrain(date(2026, 8, 30))
    stocks = smart_collector.run_stock_training(date(2026, 8, 30))

    assert "artifacts_ready=1" in weekly
    assert "feature_rows=250" in weekly
    assert "trained=3" in stocks


def test_api_background_model_training_runs_outside_duckdb_writer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main, "submit_write", _forbid_writer_training)
    monkeypatch.setattr(
        trainer,
        "train_all_models",
        lambda **_kwargs: {"models": {}, "feature_rows": 250},
    )
    monkeypatch.setattr(
        stock_trainer.StockModelTrainer,
        "train_all_fno_stocks",
        lambda self, **_kwargs: {"total": 214, "trained": 0},
    )

    main._run_ml_training(force=False)
    main._run_stock_training(force=False)

    assert main._ml_training_state["last_error"] is None
    assert main._ml_training_state["last_report"]["feature_rows"] == 250
    assert main._stock_training_state["stage"] == "complete"
    assert main._stock_training_state["last_report"]["total"] == 214


def test_api_single_stock_training_runs_outside_duckdb_writer(monkeypatch) -> None:
    monkeypatch.setattr(main, "submit_write", _forbid_writer_training)
    monkeypatch.setattr(
        stock_trainer.StockModelTrainer,
        "train_symbol",
        lambda self, symbol, **_kwargs: {
            "symbol": symbol,
            "models_trained": ["m01_vol_regime"],
            "error": None,
        },
    )

    main._run_single_stock_training("RELIANCE", force=False)

    assert main._stock_training_state["stage"] == "complete"
    assert main._stock_training_state["last_report"]["symbol"] == "RELIANCE"


def test_training_audit_uses_short_typed_writer_request(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        training_log,
        "submit_write",
        lambda operation, payload: calls.append((operation, payload)),
    )

    training_log.log_training_result(
        "NIFTY",
        "m01_vol_regime",
        {"is_trained": True, "train_rows": 250, "val_score": 0.7},
    )

    assert calls[0][0] is training_log.WriteOperation.RECORD_ML_TRAINING_LOG
    assert calls[0][1]["symbol"] == "NIFTY"
    assert calls[0][1]["status"] == "READY"
