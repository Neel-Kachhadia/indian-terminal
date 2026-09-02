from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataclasses

from db import duckdb_client
from data.market_data_quality import (
    DecisionContext,
    MarketDataEligibility,
    MarketDataQuality,
)
from ml import registry
from ml.feature_engineering import MLContext
from ml.models import common


def test_artifact_path_change_cannot_change_verified_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    artifact = root / "model.skops"
    trusted = b"trusted-artifact-bytes"
    changed = b"changed-artifact-bytes"
    assert len(trusted) == len(changed)
    artifact.write_bytes(trusted)
    import hashlib

    record = {
        "artifact_path": "model.skops",
        "artifact_size": len(trusted),
        "artifact_hash": hashlib.sha256(trusted).hexdigest(),
    }
    original_read = registry.os.read
    swapped = False

    def swap_after_open_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, count)
        if data and not swapped:
            swapped = True
            artifact.write_bytes(changed)
        return data

    monkeypatch.setattr(registry.os, "read", swap_after_open_read)
    verified = registry.read_verified_artifact_bytes(record, root, maximum_bytes=1024)
    assert swapped is True
    assert artifact.read_bytes() == changed
    assert verified is not None and verified[1] == trusted


def _context(rows: int = 220) -> MLContext:
    dates = pd.bdate_range("2026-01-02", periods=rows)
    feature = np.arange(rows, dtype=float)
    feature[-50:] = 1_000_000.0
    frame = pd.DataFrame({"date": dates.date, "feature": feature})
    ohlcv = pd.DataFrame(
        {
            "date": dates.date,
            "open": 100.0 + np.arange(rows),
            "high": 102.0 + np.arange(rows),
            "low": 99.0 + np.arange(rows),
            "close": 101.0 + np.arange(rows),
            "volume": 1_000,
        }
    )
    return MLContext(
        symbol="NIFTY",
        signal_payload={},
        current_features={},
        feature_frame=frame,
        ohlcv=ohlcv,
        iv_history=pd.DataFrame(),
        bif_history=pd.DataFrame(),
        fo_eod_rows=rows,
        current_spot=float(ohlcv["close"].iloc[-1]),
        spot_provenance={
            "source": "ANGEL_STREAM",
            "decision_eligible": True,
            "fallback_used": False,
        },
        market_data_eligibility=MarketDataEligibility(
            DecisionContext.CURRENT_DECISION,
            dt.datetime(2026, 8, 21, 11, tzinfo=dt.timezone.utc),
            dt.date(2026, 8, 21),
            dt.datetime(2026, 8, 21, tzinfo=dt.timezone.utc),
            "NSE_INDEX_BHAVCOPY",
            MarketDataQuality.OBSERVED_VERIFIED,
            True,
            True,
            True,
            (),
        ),
        decision_requested_at=dt.datetime(2026, 8, 21, 11, tzinfo=dt.timezone.utc),
    )


@pytest.fixture
def isolated_registry(isolated_app_data: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    saved = isolated_app_data / "data" / "ml" / "saved_models"
    monkeypatch.setattr(registry, "REGISTRY_PATH", isolated_app_data / "data" / "ml" / "model_registry.db")
    monkeypatch.setattr(common, "SAVED_MODELS_DIR", saved)
    return saved


@pytest.mark.contract
def test_temporal_split_is_purged_and_non_overlapping() -> None:
    dates = pd.Series(pd.bdate_range("2026-01-01", periods=200))
    split = common.temporal_train_validation_test_split(dates, gap=5)
    train, validation, test = split["slices"]
    assert train.stop + 5 == validation.start
    assert validation.stop + 5 == test.start
    assert split["train_end"] < split["validation_start"]
    assert split["validation_end"] < split["test_start"]


@pytest.mark.contract
def test_classifier_registry_hash_and_train_only_preprocessing(isolated_registry: Path) -> None:
    from sklearn.linear_model import LogisticRegression

    context = _context()

    def target_builder(_: MLContext) -> pd.Series:
        # Threshold on the raw feature column itself (linearly learnable),
        # not an index-parity pattern -- this test exercises registry/hash/
        # preprocessing plumbing, not model skill (see test_classification_
        # skill_floor_* in this file for the skill-floor behavior itself).
        # A parity-style target that a linear model cannot learn would now
        # correctly be rejected by the chance-baseline skill floor.
        values = np.where(context.feature_frame["feature"].to_numpy() > 100.0, "UP", "DOWN")
        return pd.Series(values, index=pd.to_datetime(context.feature_frame["date"]))

    artifact, reason, _ = common.train_classifier_artifact(
        context,
        model_id="test_direction",
        classes=["DOWN", "UP"],
        target_builder=target_builder,
        estimator_factory=lambda: LogisticRegression(max_iter=500),
        minimum_rows=100,
        n_splits=3,
        gap=1,
        calibration_method="sigmoid",
        force=True,
        allow_proxy=False,
    )
    assert reason is None
    assert artifact is not None
    assert artifact["validation_status"] == "VALIDATED"
    assert artifact["feature_medians"]["feature"] < 1_000_000.0
    assert artifact["train_end"] < artifact["validation_start"]
    assert artifact["validation_end"] < artifact["test_start"]
    loaded = common.load_latest_artifact("test_direction", "NIFTY", allow_proxy=False)
    assert loaded is not None
    assert loaded["artifact_hash"] == artifact["artifact_hash"]

    path = common.direct_artifact_path("test_direction", "NIFTY")
    assert path is not None and path.is_relative_to(isolated_registry)
    path.write_bytes(path.read_bytes() + b"tamper")
    assert common.load_latest_artifact("test_direction", "NIFTY", allow_proxy=False) is None


def test_classification_skill_floor_scales_with_validation_sample_size() -> None:
    # Chance baseline for 2 classes is 0.5; the floor sits above it by a
    # margin that shrinks as validation_n grows (tighter statistical
    # confidence with more samples), and never drops below chance itself.
    floor_small_n = common._classification_skill_floor(2, 20)
    floor_large_n = common._classification_skill_floor(2, 2000)
    assert 0.5 < floor_large_n < floor_small_n < 1.0
    # A 3-class problem's floor sits above its own 1/3 chance baseline.
    floor_three_class = common._classification_skill_floor(3, 100)
    assert 1.0 / 3.0 < floor_three_class < 0.5


@pytest.mark.contract
def test_sub_chance_classifier_is_rejected_not_validated(isolated_registry: Path) -> None:
    """A model that cannot beat random guessing must never reach
    registry_status == VALIDATED -- recommendation_eligibility() and
    ml/service.py's confidence gate both treat VALIDATED as a proxy for
    genuine skill, not merely "the pipeline ran".
    """
    from sklearn.linear_model import LogisticRegression

    context = _context()

    def unlearnable_target_builder(_: MLContext) -> pd.Series:
        # Alternates every row regardless of the (monotonic) feature value --
        # not linearly separable, so a single-feature LogisticRegression can
        # do no better than chance on any contiguous temporal slice.
        values = np.where(np.arange(len(context.feature_frame)) % 2 == 0, "UP", "DOWN")
        return pd.Series(values, index=pd.to_datetime(context.feature_frame["date"]))

    artifact, reason, _ = common.train_classifier_artifact(
        context,
        model_id="test_no_skill",
        classes=["DOWN", "UP"],
        target_builder=unlearnable_target_builder,
        estimator_factory=lambda: LogisticRegression(max_iter=500),
        minimum_rows=100,
        n_splits=3,
        gap=1,
        calibration_method="sigmoid",
        force=True,
        allow_proxy=False,
    )
    assert artifact is None
    assert reason is not None and "skill floor" in reason
    assert common.load_latest_artifact("test_no_skill", "NIFTY", allow_proxy=False) is None


@pytest.mark.contract
def test_legacy_pickle_is_never_loaded(isolated_registry: Path) -> None:
    target = isolated_registry / "NIFTY"
    target.mkdir(parents=True)
    (target / "rogue_2099.pkl").write_bytes(b"legacy pickle must stay inert")
    assert common.load_latest_artifact("rogue", "NIFTY", allow_proxy=False) is None


@pytest.mark.contract
def test_ohlc_artifact_uses_frozen_temporal_validation(isolated_registry: Path) -> None:
    from ml.models.m21_ohlc_forecast import OHLCForecaster

    context = _context()
    forecaster = OHLCForecaster()
    result = forecaster.train("NIFTY", context.ohlcv, force=True)
    assert result["is_trained"] is True
    artifact = common.load_latest_artifact("m21_ohlc", "NIFTY", allow_proxy=False)
    assert artifact is not None
    assert artifact["train_end"] < artifact["validation_start"]
    assert artifact["validation_end"] < artifact["test_start"]
    assert artifact["validation_metric_name"] == "close_mae"
    assert artifact["calibration_metric_name"] == "forecast_range_coverage"
    prediction = forecaster.predict("NIFTY", context.ohlcv, context.current_spot)
    assert prediction["is_trained"] is True
    assert prediction["model_attribution"]["artifact_hash"] == artifact["artifact_hash"]
    assert prediction["model_attribution"]["data_as_of"] is not None
    assert "accuracy_last_10d" not in prediction


@pytest.mark.contract
def test_prediction_outcome_has_attribution_error_and_calibration(
    isolated_app_data: Path,
) -> None:
    from db.write_service import (
        WriteOperation,
        WriterRequestFailed,
        WriterService,
        submit_write,
    )
    from ml.monitor import accuracy_for_model

    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    attribution = {
        "requested_symbol": "NIFTY",
        "data_as_of": "2026-08-21",
        "model_version": "v1",
        "artifact_hash": "a" * 64,
        "feature_schema_version": "features-v1:test",
        "release_version": "0.1.0",
        "validation_status": "VALIDATED",
        "fallback_status": "PRIMARY",
    }
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        prediction_id = submit_write(
            WriteOperation.LOG_PREDICTION,
            {
                "model_id": "test_model",
                "prediction": "UP",
                "probabilities": {"UP": 0.8, "DOWN": 0.2},
                "confidence": 0.8,
                "features": {"spot": 1.0},
                "attribution": attribution,
            },
            database=database,
            queue_path=queue,
        )
        outcome = submit_write(
            WriteOperation.RECORD_PREDICTION_OUTCOME,
            {"prediction_id": prediction_id, "actual": "UP"},
            database=database,
            queue_path=queue,
        )
        report = accuracy_for_model("test_model")
        assert outcome["actual"] == "UP"
        assert outcome["error"] == 0.0
        assert outcome["calibration_metric"]["value"] == pytest.approx(0.08)
        assert report["sample_count"] == 1
        assert report["accuracy"] == 1.0
        assert report["calibration_metric"]["measured"] is True
        with pytest.raises(WriterRequestFailed, match="already recorded"):
            submit_write(
                WriteOperation.RECORD_PREDICTION_OUTCOME,
                {"prediction_id": prediction_id, "actual": "DOWN"},
                database=database,
                queue_path=queue,
            )
    finally:
        service.stop()


@pytest.mark.contract
def test_real_training_never_silently_substitutes_proxy_symbols_artifact(
    isolated_registry: Path,
) -> None:
    """Reproduces a live defect: m06_regime_return._train() (and
    m04_iv_reversion._train()) called train_classifier_artifact() without an
    explicit allow_proxy argument, so once PROXY_SYMBOL ("NIFTY") had its own
    validated artifact, train_classifier_artifact's own "already trained?"
    pre-check (`load_latest_artifact(model_id, context.symbol,
    allow_proxy=True)` -- proxy allowed by default) silently accepted NIFTY's
    artifact as the answer for every OTHER symbol's real (force=False)
    training call: is_trained=True was reported with NIFTY's train_rows, and
    no artifact was ever registered for the requested symbol.

    Reproduced live in production 2026-09-02: the moment NIFTY got its first
    real m06_regime_return artifact, the very next symbol trained
    (FINNIFTY) silently no-opped this way instead of training its own model.
    Fixed by passing allow_proxy=not train_if_missing at the training call
    site (proxy fallback remains available at inference/predict time, where
    it is disclosed via model_attribution.fallback_status -- it must never
    substitute for real per-symbol training)."""
    from sklearn.linear_model import LogisticRegression

    def target_builder(ctx: MLContext) -> pd.Series:
        values = np.where(ctx.feature_frame["feature"].to_numpy() > 100.0, "UP", "DOWN")
        return pd.Series(values, index=pd.to_datetime(ctx.feature_frame["date"]))

    def train_for(symbol: str, *, allow_proxy: bool) -> dict:
        context = dataclasses.replace(_context(), symbol=symbol)
        artifact, reason, _ = common.train_classifier_artifact(
            context,
            model_id="test_regime_proxy_guard",
            classes=["DOWN", "UP"],
            target_builder=target_builder,
            estimator_factory=lambda: LogisticRegression(max_iter=500),
            minimum_rows=100,
            n_splits=3,
            gap=1,
            calibration_method="sigmoid",
            force=False,
            train_if_missing=True,
            allow_proxy=allow_proxy,
        )
        assert reason is None and artifact is not None
        return artifact

    nifty_artifact = train_for("NIFTY", allow_proxy=False)
    assert nifty_artifact["symbol"] == "NIFTY"
    assert nifty_artifact.get("fallback_status", "PRIMARY") != "PROXY"

    # The fixed call pattern (allow_proxy=False during real training): a
    # second symbol with no artifact of its own must get its OWN artifact,
    # not NIFTY's.
    other_artifact = train_for("OTHERSTOCK", allow_proxy=False)
    assert other_artifact["symbol"] == "OTHERSTOCK"
    assert other_artifact["artifact_hash"] != nifty_artifact["artifact_hash"]
    registered = common.load_latest_artifact(
        "test_regime_proxy_guard", "OTHERSTOCK", allow_proxy=False
    )
    assert registered is not None, "OTHERSTOCK must have its own registered artifact"

    # Demonstrates the bug this guards against: the vulnerable call pattern
    # (allow_proxy=True, the old default at every training call site that
    # didn't override it) silently returns NIFTY's artifact for a symbol
    # that has never been trained.
    proxied = train_for("YETANOTHERSTOCK", allow_proxy=True)
    assert proxied["symbol"] == "NIFTY"
    assert proxied["fallback_status"] == "PROXY"
    assert (
        common.load_latest_artifact(
            "test_regime_proxy_guard", "YETANOTHERSTOCK", allow_proxy=False
        )
        is None
    ), "allow_proxy=True at a training call site must never be mistaken for real training"


@pytest.mark.contract
def test_resolve_pending_prediction_outcomes_never_looks_ahead_and_is_idempotent(
    isolated_app_data: Path,
) -> None:
    """Reproduces a live defect: there is no scheduled task anywhere in
    smart_collector.TASK_RUNNERS (or any other caller) that ever invokes
    record_prediction_outcome() -- it exists only as a manual
    POST /ml/predictions/{id}/outcome endpoint. Verified live: 2,790
    ml_predictions rows across every model_id, 0 ever resolved.

    resolve_pending_prediction_outcomes() is the canonical fix: it re-derives
    the actual label using the exact training-time target function
    (direction_target/vol_regime_target) against OHLCV that has, in real
    time, since become available -- never inventing a target definition, and
    never resolving a row whose target date's OHLCV does not yet exist."""
    from db.write_service import WriteOperation, WriterService, submit_write

    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"

    dates = [d.date().isoformat() for d in pd.bdate_range("2026-01-05", periods=6)]
    # Flat then a clean +2% up move on the final day: resolvable_day's next
    # close is known (up move); unresolvable_day is the last date in the
    # frame, so its "next day" does not exist yet.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 102.0]

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        import duckdb as duckdb_module

        with duckdb_module.connect(str(database)) as connection:
            with duckdb_client._writer_transaction_scope(connection):
                for day, close in zip(dates, closes):
                    connection.execute(
                        "INSERT INTO ohlcv VALUES ('TESTSTK', CAST(? AS DATE), ?, ?, ?, ?, 1000, 'NSE_EQUITY_BHAVCOPY')",
                        [day, close, close + 1, close - 1, close],
                    )

        resolvable_day = dates[3]  # next day (dates[4]) is flat -> "FLAT"
        unresolvable_day = dates[5]  # last date, no next-day close yet

        resolvable_id = submit_write(
            WriteOperation.LOG_PREDICTION,
            {
                "model_id": "m02_direction",
                "prediction": "FLAT",
                "probabilities": {"UP": 0.2, "FLAT": 0.6, "DOWN": 0.2},
                "confidence": 0.6,
                "features": {"spot": 1.0},
                "attribution": {
                    "requested_symbol": "TESTSTK",
                    "data_as_of": resolvable_day,
                    "model_version": "v1",
                    "artifact_hash": "a" * 64,
                    "feature_schema_version": "features-v1:test",
                    "release_version": "0.1.0",
                    "validation_status": "VALIDATED",
                    "fallback_status": "PRIMARY",
                },
            },
            database=database,
            queue_path=queue,
        )
        unresolvable_id = submit_write(
            WriteOperation.LOG_PREDICTION,
            {
                "model_id": "m02_direction",
                "prediction": "UP",
                "probabilities": {"UP": 0.7, "FLAT": 0.2, "DOWN": 0.1},
                "confidence": 0.7,
                "features": {"spot": 1.0},
                "attribution": {
                    "requested_symbol": "TESTSTK",
                    "data_as_of": unresolvable_day,
                    "model_version": "v1",
                    "artifact_hash": "a" * 64,
                    "feature_schema_version": "features-v1:test",
                    "release_version": "0.1.0",
                    "validation_status": "VALIDATED",
                    "fallback_status": "PRIMARY",
                },
            },
            database=database,
            queue_path=queue,
        )

        summary = submit_write(
            WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m02_direction"]},
            database=database,
            queue_path=queue,
        )
        assert summary["candidates"] == 2
        assert summary["resolved"] == 1
        assert summary["not_yet_resolvable"] == 1

        with duckdb_module.connect(str(database)) as connection:
            resolved_row = connection.execute(
                "SELECT prediction, outcome, correct FROM ml_predictions WHERE id = ?",
                [resolvable_id],
            ).fetchone()
            unresolved_row = connection.execute(
                "SELECT outcome FROM ml_predictions WHERE id = ?",
                [unresolvable_id],
            ).fetchone()
        assert resolved_row == ("FLAT", "FLAT", 1), "prediction-time value untouched; outcome correctly matches"
        assert unresolved_row == (None,), "no lookahead: last date's next-day close does not exist yet"

        # Idempotent: re-running must not error or double-resolve.
        second = submit_write(
            WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m02_direction"]},
            database=database,
            queue_path=queue,
        )
        assert second["candidates"] == 1  # only the still-unresolvable one
        assert second["resolved"] == 0
        assert second["errors"] == 0
    finally:
        service.stop()
