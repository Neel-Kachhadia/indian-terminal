from __future__ import annotations

import datetime as dt
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from data.exchange_calendar import next_trading_day, today_ist
from data.index_aliases import index_symbol_candidates
from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from ml.models.common import (
    artifact_directory,
    direct_artifact_path,
    dataset_fingerprint,
    load_latest_artifact,
    model_attribution,
    safe_symbol,
    save_artifact,
    runtime_versions,
    temporal_train_validation_test_split,
)
from ml.registry import feature_schema_version
from release_identity import load_release_identity


MODEL_ID = "m21_ohlc"
INPUT_DATASET = "ohlcv"
REQUIRED_INPUTS = ("decision_eligible_ohlcv", "decision_eligible_spot")
MIN_ROWS = 80
MIN_ARTIFACT_BYTES = 1000


class OHLCForecaster:
    """
    Next-day OHLC forecaster.

    Ensemble:
    - ATR realised range model
    - IV-implied range model when ATM IV or VIX exists
    - Overnight gap + intraday range model
    """

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        current_close: float,
        atm_iv: float | None = None,
        vix: float | None = None,
        *,
        log_forecast: bool = False,
        include_accuracy: bool = True,
    ) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        df = self._prepare_frame(ohlcv_df)
        close = self._valid_close(current_close, df)
        atr = self._latest_atr(df, close)

        atr_method = self._atr_method(df, close, atr)
        iv_method = self._iv_method(close, atm_iv, vix)
        gap_method = self._gap_method(df, close)

        methods: dict[str, dict[str, float] | None] = {
            "atr_method": atr_method,
            "iv_method": iv_method,
            "gap_method": gap_method,
        }
        artifact = self._load_artifact(normalized)
        weights = self._weights(normalized, df, methods, artifact=artifact)
        combined = self._combine(methods, weights)
        combined = self._enforce_ohlc(combined)

        expected_range = round(combined["high"] - combined["low"], 2)
        expected_range_pct = round((expected_range / close) * 100.0, 4) if close > 0 else 0.0
        gap_bias = combined["open"] - close
        gap_pct = round((gap_bias / close) * 100.0, 4) if close > 0 else 0.0
        close_mae = self.compute_accuracy(df, n_days=10) if include_accuracy else None
        confidence = self._confidence(len(df), close_mae, close)
        mae_pct = round((close_mae / close) * 100.0, 6) if close_mae is not None and close > 0 else None
        last = df.tail(1).iloc[0] if not df.empty else None
        prev_high = float(last["high"]) if last is not None else combined["high"]
        prev_low = float(last["low"]) if last is not None else combined["low"]
        prev_close = float(last["close"]) if last is not None else close
        pivots = self._compute_pivot_levels(prev_high, prev_low, prev_close)
        key_levels = {
            "resistance_1": round(combined["high"], 2),
            "resistance_2": round(combined["high"] + 0.3 * atr, 2),
            "support_1": round(combined["low"], 2),
            "support_2": round(combined["low"] - 0.3 * atr, 2),
            **pivots,
        }
        forecast_date = self._next_trading_day(df).isoformat()

        result = {
            "symbol": normalized,
            "date": forecast_date,
            "current_close": round(close, 2),
            "forecast_open": round(combined["open"], 2),
            "forecast_high": round(combined["high"], 2),
            "forecast_low": round(combined["low"], 2),
            "forecast_close": round(combined["close"], 2),
            "expected_range": expected_range,
            "expected_range_pct": expected_range_pct,
            "confidence": confidence,
            "confidence_formula": (
                "categorical(rows, measured_close_mae_10d/current_close): "
                "HIGH if rows>=200 and mae_pct<=0.35; MEDIUM if rows>=60 and "
                "mae_pct<=0.8; otherwise LOW"
            ),
            "confidence_inputs": {
                "eligible_rows": len(df),
                "close_mae_last_10d": close_mae,
                "current_close": close,
                "mae_pct": mae_pct,
            },
            "method_outputs": methods,
            "method_weights": weights,
            "key_levels": key_levels,
            "gap_prediction": {
                "direction": self._gap_direction(gap_pct),
                "expected_gap_pct": gap_pct,
                "expected_open": round(combined["open"], 2),
            },
            "is_trained": artifact is not None,
            "close_mae_last_10d": close_mae,
            "model_attribution": (
                model_attribution(artifact, normalized, data_as_of=self._data_as_of(df))
                if artifact
                else self._untrained_attribution(normalized, df)
            ),
        }

        if log_forecast:
            self.log_forecast(result)
        return result

    def train(self, symbol: str, ohlcv_df: pd.DataFrame, force: bool = False) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        df = self._prepare_frame(ohlcv_df)
        if len(df) < MIN_ROWS:
            return {
                "model_id": MODEL_ID,
                "is_trained": False,
                "reason": f"insufficient data: {len(df)}/{MIN_ROWS} rows",
                "train_rows": len(df),
            }

        existing = self._artifact_path(normalized)
        if existing and not force and self._fresh(existing):
            return {
                "model_id": MODEL_ID,
                "is_trained": True,
                "skipped": True,
                "reason": "model fresh under 24h",
                "path": str(existing),
                "train_rows": len(df),
            }

        try:
            split = temporal_train_validation_test_split(df["date"], gap=1)
        except ValueError as error:
            return {
                "model_id": MODEL_ID,
                "is_trained": False,
                "reason": str(error),
                "train_rows": len(df),
            }
        train_slice, validation_slice, test_slice = split["slices"]
        training_frame = df.iloc[train_slice].copy()
        validation_frame = df.iloc[validation_slice].copy()
        test_frame = df.iloc[test_slice].copy()
        train_close = self._valid_close(None, training_frame)
        methods = {
            "atr_method": self._atr_method(
                training_frame,
                train_close,
                self._latest_atr(training_frame, train_close),
            ),
            "iv_method": None,
            "gap_method": self._gap_method(training_frame, train_close),
        }
        weights = self._learn_weights(training_frame)
        validation_metrics = self._evaluate_frozen_weights(df, validation_frame, weights)
        test_metrics = self._evaluate_frozen_weights(df, test_frame, weights)
        if validation_metrics["close_mae"] is None or test_metrics["close_mae"] is None:
            return {
                "model_id": MODEL_ID,
                "is_trained": False,
                "reason": "temporal validation metrics unavailable",
                "train_rows": len(training_frame),
            }
        last_close = self._valid_close(None, df)
        val_score = self._score_from_mae(validation_metrics["close_mae"], last_close)
        now = dt.datetime.now(dt.timezone.utc)
        release = load_release_identity()
        artifact = {
            "model_id": MODEL_ID,
            "model_version": f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}",
            "symbol": safe_symbol(normalized),
            "universe": normalized,
            "weights": weights,
            "method_names": list(methods.keys()),
            "close_mae_test": test_metrics["close_mae"],
            "trained_at": now.isoformat(),
            "training_timestamp": now.isoformat(),
            "training_data_horizon": f"{df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}",
            "feature_schema_version": feature_schema_version(list(methods)),
            "code_version": str(release["python_api_version"]),
            "release_version": str(release["release_version"]),
            "validation_status": "VALIDATED",
            "fallback_status": "PRIMARY",
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "validation_start": split["validation_start"],
            "validation_end": split["validation_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "train_rows": len(training_frame),
            "validation_sample_count": len(validation_frame),
            "test_sample_count": len(test_frame),
            "validation_metric_name": "close_mae",
            "validation_metric_value": validation_metrics["close_mae"],
            "calibration_metric_name": "forecast_range_coverage",
            "calibration_metric_value": test_metrics["range_coverage"],
            "calibration_sample_count": len(test_frame),
            "n_features": len(methods),
            "expected_feature_count": len(methods),
            "dataset_fingerprint": dataset_fingerprint(
                df[["date", "open", "high", "low", "close", "volume"]]
            ),
            "runtime_versions": runtime_versions(self),
            "train_score": val_score,
            "val_score": val_score,
            "feature_importance": weights,
            "feature_importance_top5": dict(list(weights.items())[:5]),
            "bif_rows_used": 0,
            "ohlcv_rows_used": len(df),
            "fit_diagnostics": self._fit_diagnostics(df),
        }
        try:
            path = self._save_artifact(normalized, artifact)
        except Exception as error:
            return {
                "model_id": MODEL_ID,
                "is_trained": False,
                "reason": f"artifact save failed: {error}",
                "train_rows": len(training_frame),
                "n_features": len(methods),
                "close_mae_test": test_metrics["close_mae"],
                "train_score": val_score,
                "val_score": val_score,
                "feature_importance": weights,
                "feature_importance_top5": dict(list(weights.items())[:5]),
                "bif_rows_used": 0,
                "ohlcv_rows_used": len(df),
                "json_path": str(self._artifact_json_path(normalized)),
            }
        return {
            "model_id": MODEL_ID,
            "is_trained": True,
            "train_rows": len(training_frame),
            "n_features": len(methods),
            "close_mae_test": test_metrics["close_mae"],
            "train_score": val_score,
            "val_score": val_score,
            "feature_importance": weights,
            "feature_importance_top5": dict(list(weights.items())[:5]),
            "bif_rows_used": 0,
            "ohlcv_rows_used": len(df),
            "path": str(path),
        }

    def compute_accuracy(self, df: pd.DataFrame, n_days: int = 10) -> float | None:
        frame = self._prepare_frame(df)
        if len(frame) < max(MIN_ROWS + 1, n_days + 5):
            return None
        errors: list[float] = []
        start = max(MIN_ROWS, len(frame) - n_days)
        for index in range(start, len(frame)):
            train_frame = frame.iloc[:index].copy()
            actual = frame.iloc[index]
            close = float(train_frame["close"].iloc[-1])
            forecast = self.predict(
                "ACCURACY",
                train_frame,
                close,
                include_accuracy=False,
                log_forecast=False,
            )
            errors.append(abs(float(forecast["forecast_close"]) - float(actual["close"])))
        if not errors:
            return None
        return round(sum(errors) / len(errors), 2)

    def log_forecast(self, forecast: dict[str, Any]) -> None:
        try:
            ensure_ohlc_forecast_table()
            self._update_actuals(str(forecast["symbol"]))
            self._insert_forecast(forecast)
        except Exception as error:
            print(f"ohlc_forecast: log failed: {error}")

    def _atr_method(self, df: pd.DataFrame, close: float, atr: float) -> dict[str, float]:
        if len(df) < MIN_ROWS:
            k1, k2 = 0.7, 0.7
        else:
            recent = df.tail(60)
            upside = ((recent["high"].shift(-1) - recent["close"]) / recent["atr_14"]).dropna()
            downside = ((recent["close"] - recent["low"].shift(-1)) / recent["atr_14"]).dropna()
            k1 = self._clip_float(upside.mean(), 0.3, 1.5, 0.7)
            k2 = self._clip_float(downside.mean(), 0.3, 1.5, 0.7)

        gap_bias = self._compute_gap_bias(df)
        return {
            "open": round(close + gap_bias, 2),
            "high": round(close + k1 * atr, 2),
            "low": round(close - k2 * atr, 2),
            "close": round(close + gap_bias * 0.3, 2),
        }

    def _iv_method(self, close: float, atm_iv: float | None, vix: float | None = None) -> dict[str, float] | None:
        iv = atm_iv if atm_iv and atm_iv > 0 else vix
        if not iv or iv <= 0:
            return None
        iv_decimal = iv if iv <= 1.5 else iv / 100.0
        daily_move = iv_decimal / math.sqrt(252.0) * close
        return {
            "open": round(close, 2),
            "high": round(close + 0.65 * daily_move, 2),
            "low": round(close - 0.65 * daily_move, 2),
            "close": round(close, 2),
        }

    def _gap_method(self, df: pd.DataFrame, close: float) -> dict[str, float]:
        if len(df) < 5:
            return {
                "open": round(close, 2),
                "high": round(close * 1.005, 2),
                "low": round(close * 0.995, 2),
                "close": round(close, 2),
            }

        gaps = (df["open"] - df["close"].shift(1)).tail(20).dropna()
        intraday_range = (df["high"] - df["low"]).tail(20).dropna()
        avg_gap = float(gaps.mean()) if not gaps.empty else 0.0
        avg_range = float(intraday_range.mean()) if not intraday_range.empty else close * 0.01
        exp_open = close + avg_gap
        return {
            "open": round(exp_open, 2),
            "high": round(exp_open + avg_range * 0.6, 2),
            "low": round(exp_open - avg_range * 0.4, 2),
            "close": round(close + avg_gap * 0.3, 2),
        }

    def _compute_pivot_levels(self, prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2.0 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2.0 * (pivot - prev_low)
        s1 = 2.0 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2.0 * (prev_high - pivot)
        return {
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
        }

    def _compute_gap_bias(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 0.0
        gaps = (df["open"] - df["close"].shift(1)).tail(20).dropna()
        return float(gaps.mean()) if not gaps.empty else 0.0

    def _weights(
        self,
        symbol: str,
        df: pd.DataFrame,
        methods: dict[str, dict[str, float] | None],
        *,
        artifact: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        weights = artifact.get("weights") if artifact else None
        if not isinstance(weights, dict):
            weights = self._learn_weights(df)
        if methods.get("iv_method") and float(weights.get("iv_method", 0.0) or 0.0) <= 0:
            non_iv_total = sum(float(weights.get(name, 0.0) or 0.0) for name in ("atr_method", "gap_method"))
            if non_iv_total > 0:
                weights = {
                    "atr_method": float(weights.get("atr_method", 0.0) or 0.0) / non_iv_total * 0.75,
                    "iv_method": 0.25,
                    "gap_method": float(weights.get("gap_method", 0.0) or 0.0) / non_iv_total * 0.75,
                }
        available = {name: float(weights.get(name, 0.0)) for name, output in methods.items() if output}
        total = sum(available.values())
        if total <= 0:
            available = {name: 1.0 for name, output in methods.items() if output}
            total = sum(available.values())
        return {name: round(value / total, 6) for name, value in available.items()}

    def _learn_weights(self, df: pd.DataFrame) -> dict[str, float]:
        frame = self._prepare_frame(df)
        if len(frame) < MIN_ROWS + 5:
            return {"atr_method": 0.55, "iv_method": 0.0, "gap_method": 0.45}
        errors = {"atr_method": [], "gap_method": []}
        start = max(MIN_ROWS, len(frame) - 40)
        for index in range(start, len(frame)):
            train_frame = frame.iloc[:index].copy()
            actual = frame.iloc[index]
            close = float(train_frame["close"].iloc[-1])
            atr = self._latest_atr(train_frame, close)
            candidates = {
                "atr_method": self._atr_method(train_frame, close, atr),
                "gap_method": self._gap_method(train_frame, close),
            }
            for name, output in candidates.items():
                error = (
                    abs(float(output["high"]) - float(actual["high"]))
                    + abs(float(output["low"]) - float(actual["low"]))
                    + abs(float(output["close"]) - float(actual["close"]))
                ) / 3.0
                errors[name].append(error)
        inverse: dict[str, float] = {}
        for name, values in errors.items():
            mae = sum(values) / len(values) if values else 0.0
            inverse[name] = 1.0 / max(mae, 0.01)
        total = sum(inverse.values())
        if total <= 0:
            return {"atr_method": 0.55, "iv_method": 0.0, "gap_method": 0.45}
        return {
            "atr_method": round(inverse["atr_method"] / total, 6),
            "iv_method": 0.0,
            "gap_method": round(inverse["gap_method"] / total, 6),
        }

    def _evaluate_frozen_weights(
        self,
        full_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        weights: dict[str, float],
    ) -> dict[str, float | None]:
        close_errors: list[float] = []
        range_hits: list[float] = []
        for index in evaluation_frame.index:
            history = full_frame.iloc[: int(index)].copy()
            if history.empty:
                continue
            actual = full_frame.iloc[int(index)]
            close = self._valid_close(None, history)
            methods = {
                "atr_method": self._atr_method(history, close, self._latest_atr(history, close)),
                "iv_method": None,
                "gap_method": self._gap_method(history, close),
            }
            forecast = self._enforce_ohlc(self._combine(methods, weights))
            actual_close = float(actual["close"])
            close_errors.append(abs(float(forecast["close"]) - actual_close))
            range_hits.append(float(float(forecast["low"]) <= actual_close <= float(forecast["high"])))
        return {
            "close_mae": round(sum(close_errors) / len(close_errors), 6) if close_errors else None,
            "range_coverage": round(sum(range_hits) / len(range_hits), 6) if range_hits else None,
        }

    def _combine(
        self,
        methods: dict[str, dict[str, float] | None],
        weights: dict[str, float],
    ) -> dict[str, float]:
        combined = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
        for name, output in methods.items():
            if not output:
                continue
            weight = float(weights.get(name, 0.0))
            for key in combined:
                combined[key] += float(output[key]) * weight
        return {key: round(value, 2) for key, value in combined.items()}

    def _enforce_ohlc(self, values: dict[str, float]) -> dict[str, float]:
        open_price = float(values["open"])
        close = float(values["close"])
        high = max(float(values["high"]), open_price, close)
        low = min(float(values["low"]), open_price, close)
        return {
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
        }

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "atr_14"])
        frame = df.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        required = ["date", "open", "high", "low", "close"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "atr_14"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
        frame = frame.reset_index(drop=True)
        if "volume" not in frame.columns:
            frame["volume"] = 0
        frame["atr_14"] = self._atr(frame)
        return frame

    def _atr(self, frame: pd.DataFrame, period: int = 14) -> pd.Series:
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        prev_close = close.shift(1)
        ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.ewm(alpha=1 / period, adjust=False, min_periods=max(2, period // 2)).mean()

    def _latest_atr(self, df: pd.DataFrame, close: float) -> float:
        if df.empty or "atr_14" not in df.columns:
            return max(close * 0.01, 1.0)
        value = self._clean_float(df["atr_14"].dropna().tail(1).iloc[0] if not df["atr_14"].dropna().empty else None)
        return value if value and value > 0 else max(close * 0.01, 1.0)

    def _valid_close(self, current_close: float | None, df: pd.DataFrame) -> float:
        value = self._clean_float(current_close)
        if value and value > 0:
            return value
        if not df.empty:
            close = self._clean_float(df["close"].tail(1).iloc[0])
            if close and close > 0:
                return close
        return 0.0

    def _next_trading_day(self, df: pd.DataFrame) -> dt.date:
        if not df.empty:
            last = pd.to_datetime(df["date"].tail(1).iloc[0]).date()
        else:
            last = today_ist()
        return next_trading_day(last)

    def _data_as_of(self, df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        return pd.to_datetime(df["date"].iloc[-1]).date().isoformat()

    def _untrained_attribution(self, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
        release = load_release_identity()
        return {
            "model_id": MODEL_ID,
            "model_version": "UNTRAINED-HEURISTIC-v1",
            "trained_symbol": None,
            "requested_symbol": symbol,
            "universe": symbol,
            "training_data_horizon": None,
            "feature_schema_version": feature_schema_version(["atr_method", "iv_method", "gap_method"]),
            "training_timestamp": None,
            "artifact_hash": None,
            "code_version": str(release["python_api_version"]),
            "release_version": str(release["release_version"]),
            "validation_status": "UNVALIDATED",
            "fallback_status": "NONE",
            "data_as_of": self._data_as_of(df),
        }

    def _confidence(self, rows: int, accuracy: float | None, close: float) -> str:
        if rows < MIN_ROWS:
            return "low"
        if accuracy is None:
            return "medium" if rows >= 60 else "low"
        accuracy_pct = (accuracy / close) * 100.0 if close > 0 else 999.0
        if rows >= 200 and accuracy_pct <= 0.35:
            return "high"
        if rows >= 60 and accuracy_pct <= 0.8:
            return "medium"
        return "low"

    def _gap_direction(self, gap_pct: float) -> str:
        if gap_pct > 0.15:
            return "gap_up"
        if gap_pct < -0.15:
            return "gap_down"
        return "flat"

    def _load_artifact(self, symbol: str) -> dict[str, Any] | None:
        return load_latest_artifact(MODEL_ID, symbol, allow_proxy=True)

    def _save_artifact(self, symbol: str, artifact: dict[str, Any]) -> Path:
        try:
            path, record = save_artifact(MODEL_ID, artifact, symbol)
        except Exception as error:
            path = self._artifact_json_path(symbol).with_suffix(".skops")
            self._save_json_fallback(path, artifact, error)
            raise
        artifact["artifact_hash"] = record["artifact_hash"]
        return path

    def _artifact_path(self, symbol: str) -> Path | None:
        return direct_artifact_path(MODEL_ID, symbol)

    def _artifact_json_path(self, symbol: str) -> Path:
        target_dir = artifact_directory(symbol)
        return target_dir / f"{MODEL_ID}_{dt.date.today().isoformat()}.json"

    def _fresh(self, path: Path) -> bool:
        if path.stat().st_size <= MIN_ARTIFACT_BYTES:
            return False
        modified_at = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        return (dt.datetime.now(dt.timezone.utc) - modified_at).total_seconds() <= 24 * 3600

    def _save_json_fallback(self, path: Path, artifact: dict[str, Any], error: Exception) -> Path:
        fallback = path.with_suffix(".json")
        payload = {
            "model_id": MODEL_ID,
            "symbol": artifact.get("symbol"),
            "trained_at": artifact.get("trained_at"),
            "save_error": repr(error),
            "weights": artifact.get("weights", {}),
            "method_names": artifact.get("method_names", []),
            "close_mae_test": artifact.get("close_mae_test"),
            "train_rows": artifact.get("train_rows"),
            "n_features": artifact.get("n_features"),
            "train_score": artifact.get("train_score"),
            "val_score": artifact.get("val_score"),
            "fit_diagnostics": artifact.get("fit_diagnostics", {}),
        }
        fallback.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True), encoding="utf-8")
        print(f"ohlc_forecast: saved JSON fallback for {artifact.get('symbol')} at {fallback}")
        return fallback

    def _fit_diagnostics(self, df: pd.DataFrame) -> dict[str, Any]:
        frame = self._prepare_frame(df)
        if frame.empty:
            return {}
        recent = frame.tail(60)[["date", "open", "high", "low", "close", "volume", "atr_14"]].copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        return {
            "date_start": str(frame["date"].iloc[0].date()),
            "date_end": str(frame["date"].iloc[-1].date()),
            "close_start": round(float(frame["close"].iloc[0]), 4),
            "close_end": round(float(frame["close"].iloc[-1]), 4),
            "avg_volume_20": round(float(frame["volume"].tail(20).mean()), 4) if "volume" in frame else None,
            "recent_window": recent.round(4).to_dict("records"),
        }

    def _score_from_mae(self, mae: float | None, close: float) -> float | None:
        if mae is None or close <= 0:
            return None
        score = 1.0 - (float(mae) / close)
        return round(max(0.0, min(1.0, score)), 6)

    def _insert_forecast(self, forecast: dict[str, Any]) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        row_id = int(uuid.uuid4().int >> 65)
        key_levels = forecast.get("key_levels") or {}
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                # ohlc_forecasts has no UNIQUE(symbol, forecast_date)
                # constraint, and this always did a bare INSERT with a
                # random id -- every scheduled forecast run for a symbol
                # already covered for that target date silently added
                # another full duplicate row (observed live: some symbols
                # had 9+ duplicate rows for the same forecast_date). The
                # earliest forecast for a given target date is also the
                # most genuinely blind one; skip rather than re-insert once
                # any row for this (symbol, forecast_date) exists.
                existing = con.execute(
                    "SELECT COUNT(*) FROM ohlc_forecasts WHERE symbol = ? AND forecast_date = CAST(? AS DATE)",
                    [str(forecast["symbol"]), str(forecast["date"])],
                ).fetchone()[0]
                if existing:
                    return
                con.execute(
                    """
                    INSERT INTO ohlc_forecasts
                        (id, symbol, forecast_date, forecast_open, forecast_high,
                         forecast_low, forecast_close, actual_open, actual_high,
                         actual_low, actual_close, close_mae, high_mae, low_mae,
                         expected_range, expected_range_pct, confidence, key_levels_json,
                         created_at)
                    VALUES (?, ?, CAST(? AS DATE), ?, ?, ?, ?, NULL, NULL, NULL, NULL,
                            NULL, NULL, NULL, ?, ?, ?, ?, ?)
                    """,
                    [
                        row_id,
                        str(forecast["symbol"]),
                        str(forecast["date"]),
                        float(forecast["forecast_open"]),
                        float(forecast["forecast_high"]),
                        float(forecast["forecast_low"]),
                        float(forecast["forecast_close"]),
                        float(forecast["expected_range"]),
                        float(forecast["expected_range_pct"]),
                        str(forecast["confidence"]),
                        json.dumps(key_levels, sort_keys=True),
                        now,
                    ],
                )
            finally:
                con.close()

    def _update_actuals(self, symbol: str) -> None:
        # Single source of truth for index long-form/short-form aliases (see
        # data/index_aliases.py). This used to hand-duplicate its own partial
        # candidate table -- only NIFTY -> "NIFTY 50", never BANKNIFTY ->
        # "NIFTY BANK" -- the same bug class as ml/stock_trainer.py's
        # _load_ohlcv (see its fix for the full incident writeup). Verified
        # actual OHLCV for the index is stored under "NIFTY BANK", so
        # forecast.actual_close could never be resolved for BANKNIFTY: every
        # forecast stayed permanently unresolved and kept re-triggering new
        # duplicate rows on each subsequent collector run.
        candidates = index_symbol_candidates(symbol) or [symbol]
        placeholders = ", ".join("?" for _ in candidates)
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                con.execute(
                    f"""
                    UPDATE ohlc_forecasts AS forecast
                    SET actual_open = actual.open,
                        actual_high = actual.high,
                        actual_low = actual.low,
                        actual_close = actual.close,
                        close_mae = ABS(forecast.forecast_close - actual.close),
                        high_mae = ABS(forecast.forecast_high - actual.high),
                        low_mae = ABS(forecast.forecast_low - actual.low)
                    FROM decision_eligible_ohlcv AS actual
                    WHERE forecast.symbol = ?
                      AND forecast.actual_close IS NULL
                      AND actual.symbol IN ({placeholders})
                      AND actual.date = forecast.forecast_date
                    """,
                    [symbol, *candidates],
                )
            finally:
                con.close()

    def _clean_float(self, value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _clip_float(self, value: Any, low: float, high: float, default: float) -> float:
        number = self._clean_float(value)
        if number is None:
            return default
        return max(low, min(high, number))


def ensure_ohlc_forecast_table() -> None:
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlc_forecasts (
                    id                  BIGINT      PRIMARY KEY,
                    symbol              TEXT        NOT NULL,
                    forecast_date       DATE        NOT NULL,
                    forecast_open       DOUBLE,
                    forecast_high       DOUBLE,
                    forecast_low        DOUBLE,
                    forecast_close      DOUBLE,
                    actual_open         DOUBLE,
                    actual_high         DOUBLE,
                    actual_low          DOUBLE,
                    actual_close        DOUBLE,
                    close_mae           DOUBLE,
                    high_mae            DOUBLE,
                    low_mae             DOUBLE,
                    expected_range      DOUBLE,
                    expected_range_pct  DOUBLE,
                    confidence          TEXT,
                    key_levels_json     TEXT,
                    created_at          TIMESTAMP   NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ohlc_forecasts_symbol_date
                ON ohlc_forecasts(symbol, forecast_date)
                """
            )
        finally:
            con.close()
