from __future__ import annotations

import datetime as dt
import time
from typing import Any, Callable

import pandas as pd

from backtest.spot_quality import verified_spot_sources
from db.write_service import ReadOperation, resolve_read
from ml.feature_engineering import build_context
from ml.models import (
    m01_vol_regime,
    m02_direction,
    m04_iv_reversion,
    m05_anomaly,
    m06_regime_return,
    m07_iv_forecast,
    m08_gp_regime,
    m12_prophet,
    m13_nbeats,
    m14_wavelet,
    m15_kalman,
    m17_conformal,
    m20_bnn,
)
from ml.models.common import (
    artifact_is_fresh,
    load_latest_artifact,
    model_status,
    symbol_has_artifacts,
    trained_model_symbols,
)
from ml.models.m21_ohlc_forecast import MODEL_ID as OHLC_MODEL_ID
from ml.models.m21_ohlc_forecast import OHLCForecaster
from ml.training_log import log_training_result


class StockModelTrainer:
    """Trains symbol-specific ML models for F&O stocks."""

    TRAINABLE_MODELS = [
        "m01_vol_regime",
        "m02_direction",
        "m05_anomaly",
        "m06_regime_return",
        "m08_gp_regime",
        "m12_prophet",
        "m13_nbeats",
        "m14_wavelet",
        "m15_kalman",
        "m17_conformal",
        "m20_bnn",
        "m21_ohlc",
    ]
    MIN_ROWS = 100

    def compute_stock_features(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = self._prepare_ohlcv(ohlcv_df)
        if frame.empty:
            return pd.DataFrame()

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = frame["volume"]
        returns = close.pct_change()
        output = pd.DataFrame({"date": frame["date"]})
        output["returns_1d"] = returns * 100.0
        output["returns_5d"] = close.pct_change(5) * 100.0
        output["returns_20d"] = close.pct_change(20) * 100.0
        output["rsi_14"] = self._rsi(close, 14)
        output["ema_9"] = close.ewm(span=9, adjust=False).mean()
        output["ema_21"] = close.ewm(span=21, adjust=False).mean()
        output["ema_50"] = close.ewm(span=50, adjust=False).mean()
        output["ema_200"] = close.ewm(span=200, adjust=False).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        output["macd_line"] = ema_12 - ema_26
        output["macd_hist"] = output["macd_line"] - output["macd_line"].ewm(span=9, adjust=False).mean()
        bb_mid = close.rolling(20, min_periods=10).mean()
        bb_std = close.rolling(20, min_periods=10).std(ddof=0)
        output["bb_width"] = (4.0 * bb_std) / close.replace(0.0, pd.NA) * 100.0
        output["bb_position"] = (close - (bb_mid - 2.0 * bb_std)) / (4.0 * bb_std).replace(0.0, pd.NA)
        atr = self._atr(frame, 14)
        output["atr_14"] = atr
        output["atr_pct"] = atr / close.replace(0.0, pd.NA) * 100.0
        output["hv_10"] = returns.rolling(10, min_periods=5).std(ddof=0) * (252.0 ** 0.5) * 100.0
        output["hv_20"] = returns.rolling(20, min_periods=10).std(ddof=0) * (252.0 ** 0.5) * 100.0
        output["volume_ratio_20d"] = volume / volume.rolling(20, min_periods=5).mean().replace(0.0, pd.NA)
        output["price_vs_ema50_pct"] = (close - output["ema_50"]) / close.replace(0.0, pd.NA) * 100.0
        output["price_vs_ema200_pct"] = (close - output["ema_200"]) / close.replace(0.0, pd.NA) * 100.0
        output["roc_10"] = close.pct_change(10) * 100.0
        output["roc_20"] = close.pct_change(20) * 100.0
        output["adx_14"] = self._adx(frame, 14)
        return output.replace([float("inf"), float("-inf")], pd.NA)

    def train_symbol(self, symbol: str, force: bool = False) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        start = time.time()
        report: dict[str, Any] = {
            "symbol": normalized,
            "rows_available": 0,
            "models_trained": [],
            "models_skipped": [],
            "model_timings": {},
            "training_time_seconds": 0.0,
            "error": None,
        }
        try:
            ohlcv = self._load_ohlcv(normalized)
            report["rows_available"] = int(len(ohlcv))
            if len(ohlcv) < self.MIN_ROWS:
                report["models_skipped"] = list(self.TRAINABLE_MODELS)
                report["model_timings"] = {model_id: 0.0 for model_id in self.TRAINABLE_MODELS}
                report["error"] = f"insufficient data: {len(ohlcv)}/{self.MIN_ROWS} rows"
                for model_id in self.TRAINABLE_MODELS:
                    try:
                        log_training_result(
                            normalized,
                            model_id,
                            {
                                "model_id": model_id,
                                "is_trained": False,
                                "available_rows": len(ohlcv),
                                "ohlcv_rows_used": len(ohlcv),
                                "reason": report["error"],
                            },
                        )
                    except Exception as error:
                        report[f"{model_id}_log_error"] = str(error)
                return report

            if not force and self._all_fresh(normalized):
                report["models_skipped"] = list(self.TRAINABLE_MODELS)
                report["model_timings"] = {model_id: 0.0 for model_id in self.TRAINABLE_MODELS}
                return report

            try:
                from signals.backfill import backfill_symbol_signals

                report["signals_backfill"] = backfill_symbol_signals(normalized)
            except Exception as error:
                report["signals_backfill"] = {"rows_written": 0, "error": str(error)}

            context = build_context(normalized)
            def run_model(
                model_id: str,
                trainer: Callable[[], dict[str, Any]],
                *,
                available_rows: int | None = None,
            ) -> None:
                if not force and artifact_is_fresh(model_id, normalized):
                    report["models_skipped"].append(model_id)
                    report["model_timings"][model_id] = 0.0
                    return

                model_start = time.time()
                try:
                    result = trainer()
                except Exception as error:
                    result = {
                        "model_id": model_id,
                        "is_trained": False,
                        "reason": str(error),
                        "available_rows": available_rows if available_rows is not None else context.available_rows,
                    }
                if not isinstance(result, dict):
                    result = {
                        "model_id": model_id,
                        "is_trained": False,
                        "reason": f"trainer returned {type(result).__name__}",
                        "available_rows": available_rows if available_rows is not None else context.available_rows,
                    }
                elapsed = round(time.time() - model_start, 2)
                result["training_time_seconds"] = elapsed
                report["model_timings"][model_id] = elapsed

                try:
                    log_training_result(normalized, model_id, result, context=context)
                except Exception as error:
                    result["log_error"] = str(error)
                    report[f"{model_id}_log_error"] = str(error)
                if result.get("is_trained"):
                    report["models_trained"].append(model_id)
                else:
                    report["models_skipped"].append(model_id)
                    report[f"{model_id}_reason"] = result.get("reason")

            def train_m17_conformal() -> dict[str, Any]:
                if load_latest_artifact("m02_direction", normalized, allow_proxy=False) is None:
                    return {
                        "model_id": "m17_conformal",
                        "is_trained": False,
                        "reason": "m02_direction artifact not found; conformal wrapper skipped",
                        "available_rows": context.available_rows,
                    }
                return m17_conformal.train_model(context, force=force)

            for model_id, trainer in (
                ("m01_vol_regime", m01_vol_regime.train_model),
                ("m02_direction", m02_direction.train_model),
                ("m05_anomaly", m05_anomaly.train_model),
                ("m06_regime_return", m06_regime_return.train_model),
                ("m08_gp_regime", m08_gp_regime.train_model),
                ("m12_prophet", m12_prophet.train_model),
                ("m13_nbeats", m13_nbeats.train_model),
                ("m14_wavelet", m14_wavelet.train_model),
                ("m15_kalman", m15_kalman.train_model),
                ("m17_conformal", train_m17_conformal),
                ("m20_bnn", m20_bnn.train_model),
            ):
                if model_id == "m17_conformal":
                    run_model(model_id, trainer)
                else:
                    run_model(model_id, lambda trainer=trainer: trainer(context, force=force))

            run_model(
                OHLC_MODEL_ID,
                lambda: OHLCForecaster().train(normalized, ohlcv, force=force),
                available_rows=len(ohlcv),
            )

            iv_rows = len(context.iv_history) if hasattr(context, "iv_history") else 0
            if iv_rows >= 60:
                for model_id, trainer in [
                    ("m04_iv_reversion", m04_iv_reversion.train_model),
                    ("m07_iv_forecast", m07_iv_forecast.train_model),
                ]:
                    run_model(model_id, lambda trainer=trainer: trainer(context, force=force), available_rows=iv_rows)
            else:
                report["iv_history_rows"] = iv_rows
                report["iv_models_skipped"] = "iv_history insufficient"
        except Exception as error:
            report["error"] = str(error)
            for model_id in self.TRAINABLE_MODELS:
                try:
                    log_training_result(
                        normalized,
                        model_id,
                        {
                            "model_id": model_id,
                            "is_trained": False,
                            "available_rows": report.get("rows_available"),
                            "ohlcv_rows_used": report.get("rows_available"),
                            "reason": str(error),
                        },
                    )
                except Exception as log_error:
                    report[f"{model_id}_log_error"] = str(log_error)
                report["model_timings"].setdefault(model_id, 0.0)
        finally:
            report["training_time_seconds"] = round(time.time() - start, 2)
        return report

    def train_all_fno_stocks(
        self,
        force: bool = False,
        on_progress: Callable[[str, int, int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        symbols = self.fno_symbols()
        summary = {
            "total": len(symbols),
            "trained": 0,
            "skipped_insufficient_data": 0,
            "skipped_already_trained": 0,
            "errors": 0,
            "duration_minutes": 0.0,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_at": None,
        }
        for done, symbol in enumerate(symbols, start=1):
            result = self.train_symbol(symbol, force=force)
            if result.get("error"):
                if "insufficient data" in str(result["error"]):
                    summary["skipped_insufficient_data"] += 1
                else:
                    summary["errors"] += 1
            elif result.get("models_trained"):
                summary["trained"] += 1
            else:
                summary["skipped_already_trained"] += 1
            if on_progress:
                on_progress(symbol, done, len(symbols), result)
        summary["duration_minutes"] = round((time.time() - started) / 60.0, 2)
        summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        return summary

    def load_symbol_models(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        models: dict[str, Any] = {}
        for model_id in self.TRAINABLE_MODELS:
            artifact = load_latest_artifact(model_id, normalized, allow_proxy=False)
            if artifact is not None:
                models[model_id] = artifact
        return models

    def predict_symbol(self, symbol: str, current_data: dict[str, Any] | None = None) -> dict[str, Any]:
        from ml.service import predict_all_models

        payload = current_data if isinstance(current_data, dict) else None
        return predict_all_models(symbol.strip().upper(), payload)

    def trained_symbols(self) -> dict[str, Any]:
        symbols = trained_model_symbols(self.TRAINABLE_MODELS)
        total = len(self.fno_symbols())
        return {
            "count": len(symbols),
            "total_fno": total,
            "symbols": symbols,
        }

    def fno_symbols(self) -> list[str]:
        result = resolve_read(ReadOperation.LOAD_STOCK_TRAINER_FNO_SYMBOLS, {})
        return result["symbols"]

    def _all_fresh(self, symbol: str) -> bool:
        return symbol_has_artifacts(symbol, self.TRAINABLE_MODELS) and all(
            artifact_is_fresh(model_id, symbol) for model_id in self.TRAINABLE_MODELS
        )

    def _load_ohlcv(self, symbol: str) -> pd.DataFrame:
        if symbol == "NIFTY":
            candidates = ["NIFTY"]
        elif symbol == "NIFTY 50":
            candidates = ["NIFTY 50", "NIFTY"]
        elif symbol == "BANKNIFTY":
            candidates = ["BANKNIFTY", "BANK NIFTY", "NIFTY BANK"]
        else:
            candidates = [symbol]
        sources = sorted(verified_spot_sources(symbol))
        result = resolve_read(
            ReadOperation.LOAD_VERIFIED_SOURCE_OHLCV,
            {"candidates": candidates, "sources": sources},
        )
        frame = pd.DataFrame(
            result["rows"],
            columns=["date", "open", "high", "low", "close", "volume"],
        )
        return self._prepare_ohlcv(frame)

    def _prepare_ohlcv(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        output = frame.copy()
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            output[column] = pd.to_numeric(output[column], errors="coerce")
        return output.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)

    def _atr(self, frame: pd.DataFrame, period: int) -> pd.Series:
        prev_close = frame["close"].shift(1)
        ranges = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - prev_close).abs(),
                (frame["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        return ranges.max(axis=1).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    def _adx(self, frame: pd.DataFrame, period: int) -> pd.Series:
        up_move = frame["high"].diff()
        down_move = -frame["low"].diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr = self._atr(frame, period).replace(0.0, pd.NA)
        plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, pd.NA)) * 100.0
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def trained_symbols_summary() -> dict[str, Any]:
    return StockModelTrainer().trained_symbols()


def model_status_for_symbol(symbol: str) -> dict[str, Any]:
    return {
        model_id: model_status(model_id, symbol)
        for model_id in StockModelTrainer.TRAINABLE_MODELS
    }
