from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest.spot_quality import verified_spot_sources
from data.market_data_quality import (
    DecisionContext,
    MarketDataEligibility,
    evaluate_market_bar,
)
from db.write_service import ReadOperation, resolve_read


TRADING_DAYS = 252
LAG_PERIODS = (1, 2, 3, 5, 10)
BIF_FEATURE_COLUMNS = ("cds", "bms", "iad", "ngi", "vrc", "nfve")


@dataclass
class MLContext:
    symbol: str
    signal_payload: dict[str, Any]
    current_features: dict[str, float]
    feature_frame: pd.DataFrame
    ohlcv: pd.DataFrame
    iv_history: pd.DataFrame
    bif_history: pd.DataFrame
    fo_eod_rows: int
    current_spot: float
    spot_provenance: dict[str, Any]
    market_data_eligibility: MarketDataEligibility
    decision_requested_at: dt.datetime

    @property
    def available_rows(self) -> int:
        return int(len(self.feature_frame))

    @property
    def current_vector(self) -> pd.Series:
        if self.feature_frame.empty:
            return pd.Series(dtype="float64")
        row = self.feature_frame.drop(columns=["date"], errors="ignore").tail(1)
        if row.empty:
            return pd.Series(dtype="float64")
        return row.iloc[0]


def build_context(
    symbol: str,
    signal_payload: dict[str, Any] | None = None,
    *,
    decision_requested_at: dt.datetime | None = None,
) -> MLContext:
    normalised = symbol.strip().upper()
    payload = signal_payload or {}
    requested_at = decision_requested_at or dt.datetime.now(dt.timezone.utc)
    if requested_at.tzinfo is None or requested_at.utcoffset() is None:
        raise ValueError("decision_requested_at must be timezone-aware")
    ohlcv = _load_ohlcv(normalised)
    latest_bar = ohlcv.iloc[-1].to_dict() if not ohlcv.empty else None
    market_eligibility = evaluate_market_bar(
        latest_bar,
        context=DecisionContext.CURRENT_DECISION,
        requested_as_of=requested_at,
    )
    signal_history = _load_signal_history(normalised)
    bif_history = _load_bif_history(normalised)
    iv_history = _load_iv_history(normalised)
    current_features = flatten_signal_payload(payload)
    ohlcv_features = _ohlcv_features(ohlcv)
    feature_frame = _join_feature_history(ohlcv_features, signal_history)
    feature_frame = _join_bif_history(feature_frame, bif_history)
    feature_frame = _apply_bif_availability(feature_frame)
    feature_frame = add_lag_and_zscore_features(feature_frame)

    if current_features:
        feature_frame = _overlay_current_features(feature_frame, current_features)

    return MLContext(
        symbol=normalised,
        signal_payload=payload,
        current_features=current_features,
        feature_frame=feature_frame,
        ohlcv=ohlcv,
        iv_history=iv_history,
        bif_history=bif_history,
        fo_eod_rows=_count_fo_eod_rows(normalised),
        current_spot=_extract_spot(payload, ohlcv),
        spot_provenance=(
            dict(payload.get("spot_provenance"))
            if isinstance(payload.get("spot_provenance"), dict)
            else {}
        ),
        market_data_eligibility=market_eligibility,
        decision_requested_at=requested_at,
    )


def flatten_signal_payload(payload: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    categories = payload.get("categories")
    if isinstance(categories, dict):
        for category_name, signals in categories.items():
            if not isinstance(signals, dict):
                continue
            for signal_name, signal in signals.items():
                if isinstance(signal, dict):
                    value = _as_float(signal.get("value"))
                else:
                    value = _as_float(signal)
                if value is None:
                    continue
                features[f"{category_name}.{signal_name}"] = value
                features.setdefault(str(signal_name), value)

    composites = payload.get("composite_scores")
    if isinstance(composites, dict):
        for name, value in composites.items():
            number = _as_float(value)
            if number is not None:
                features[f"composite.{name}"] = number
                features.setdefault(str(name), number)

    options = payload.get("options")
    if isinstance(options, dict):
        for name, value in options.items():
            number = _as_float(value)
            if number is not None:
                features[f"options.{name}"] = number
                features.setdefault(str(name), number)

    market = payload.get("market")
    if isinstance(market, dict):
        vix = _as_float(market.get("india_vix"))
        if vix is not None:
            features["vix.india_vix"] = vix
            features.setdefault("india_vix", vix)
        fii_dii = market.get("fii_dii")
        if isinstance(fii_dii, dict):
            for name, value in fii_dii.items():
                number = _as_float(value)
                if number is not None:
                    features[f"fii_dii.{name}"] = number
                    features.setdefault(str(name), number)

    spot = _as_float(payload.get("spot"))
    if spot is not None:
        features["spot"] = spot
    return features


def add_lag_and_zscore_features(frame: pd.DataFrame, max_columns: int = 140) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    base_cols = [
        col
        for col in output.columns
        if col != "date" and pd.api.types.is_numeric_dtype(output[col])
    ]
    selected = _select_engineering_columns(base_cols, max_columns)
    engineered: dict[str, pd.Series] = {}
    for col in selected:
        series = pd.to_numeric(output[col], errors="coerce")
        for lag in LAG_PERIODS:
            engineered[f"{col}_lag_{lag}"] = series.shift(lag)
        rolling_mean = series.rolling(20, min_periods=5).mean()
        rolling_std = series.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
        engineered[f"{col}_z20"] = (series - rolling_mean) / rolling_std
    if engineered:
        output = pd.concat([output, pd.DataFrame(engineered, index=output.index)], axis=1)

    _add_interaction(output, "options.iv_rank", "options.pcr_total", "iv_rank_x_pcr")
    _add_interaction(output, "iv_rank", "pcr", "iv_rank_x_pcr_current")
    _add_interaction(output, "fii_dii.fii_net_cash", "options.gex", "fii_net_x_gex")
    _add_interaction(output, "india_vix", "rsi_14", "vix_x_rsi")
    return output.replace([np.inf, -np.inf], np.nan)


def training_frame_with_target(
    context: MLContext,
    target_builder: Callable[[MLContext], pd.Series],
) -> pd.DataFrame:
    if context.feature_frame.empty:
        return pd.DataFrame()
    features = context.feature_frame.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.date
    target = target_builder(context)
    if target.empty:
        return pd.DataFrame()
    target = target.copy()
    target.index = pd.to_datetime(target.index).date
    features = features.set_index("date")
    features["target"] = target
    features = features.dropna(subset=["target"]).reset_index().rename(columns={"index": "date"})
    return features.sort_values("date")


def vol_regime_target(context: MLContext) -> pd.Series:
    close = _close_series(context.ohlcv)
    if close.empty:
        return pd.Series(dtype="object")
    returns = close.pct_change()
    future = pd.concat([returns.shift(-i) for i in range(1, 6)], axis=1)
    realised = future.std(axis=1, ddof=0) * math.sqrt(TRADING_DAYS) * 100.0
    labels = pd.Series(index=realised.index, dtype="object")
    labels[realised < 10.0] = "LOW"
    labels[(realised >= 10.0) & (realised <= 18.0)] = "MEDIUM"
    labels[realised > 18.0] = "HIGH"
    return labels


def direction_target(context: MLContext) -> pd.Series:
    close = _close_series(context.ohlcv)
    if close.empty:
        return pd.Series(dtype="object")
    next_return = (close.shift(-1) - close) / close * 100.0
    labels = pd.Series(index=next_return.index, dtype="object")
    labels[next_return > 0.5] = "UP"
    labels[(next_return >= -0.5) & (next_return <= 0.5)] = "FLAT"
    labels[next_return < -0.5] = "DOWN"
    return labels


def iv_reversion_target(context: MLContext) -> pd.Series:
    if context.iv_history.empty:
        return pd.Series(dtype="object")
    series = context.iv_history.set_index("date")["atm_iv"].sort_index()
    change = (series.shift(-3) - series) / series.replace(0.0, np.nan) * 100.0
    labels = pd.Series(index=change.index, dtype="object")
    labels[change > 5.0] = "EXPANDING"
    labels[(change >= -5.0) & (change <= 5.0)] = "STABLE"
    labels[change < -5.0] = "CONTRACTING"
    return labels


def current_value(context: MLContext, candidates: list[str], default: float | None = None) -> float | None:
    for name in candidates:
        value = _as_float(context.current_features.get(name))
        if value is not None:
            return value
    vector = context.current_vector
    for name in candidates:
        if name in vector:
            value = _as_float(vector.get(name))
            if value is not None:
                return value
    for name in candidates:
        matches = [col for col in vector.index if col.endswith(name)]
        for col in matches:
            value = _as_float(vector.get(col))
            if value is not None:
                return value
    return default


def _load_ohlcv(symbol: str) -> pd.DataFrame:
    candidates = _symbol_candidates(symbol)
    sources = sorted(verified_spot_sources(symbol))
    result = resolve_read(
        ReadOperation.LOAD_FEATURE_OHLCV,
        {"candidates": candidates, "sources": sources},
    )
    return pd.DataFrame(result["rows"], columns=result["columns"])


def _load_bif_history(symbol: str) -> pd.DataFrame:
    candidates = _symbol_candidates(symbol)
    result = resolve_read(ReadOperation.LOAD_FEATURE_BIF_HISTORY, {"candidates": candidates})
    rows = pd.DataFrame(result["rows"], columns=result["columns"])
    if rows.empty:
        return pd.DataFrame(columns=["date", *BIF_FEATURE_COLUMNS])
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    for column in BIF_FEATURE_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _load_signal_history(symbol: str) -> pd.DataFrame:
    candidates = _symbol_candidates(symbol)
    result = resolve_read(ReadOperation.LOAD_FEATURE_SIGNAL_HISTORY, {"candidates": candidates})
    rows = pd.DataFrame(result["rows"], columns=result["columns"])
    if rows.empty:
        return pd.DataFrame(columns=["date"])
    pivot = rows.pivot_table(
        index="date",
        columns="signal_name",
        values="value",
        aggfunc="last",
    ).reset_index()
    pivot.columns = [str(col) for col in pivot.columns]
    return pivot


def _load_iv_history(symbol: str) -> pd.DataFrame:
    candidates = _symbol_candidates(symbol)
    result = resolve_read(ReadOperation.LOAD_FEATURE_IV_HISTORY, {"candidates": candidates})
    rows = pd.DataFrame(result["rows"], columns=result["columns"])
    if rows.empty:
        return pd.DataFrame(columns=["date", "atm_iv"])
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    return rows


def _count_fo_eod_rows(symbol: str) -> int:
    candidates = _symbol_candidates(symbol)
    result = resolve_read(ReadOperation.COUNT_FO_EOD_ROWS, {"candidates": candidates})
    return result["count"]


def _ohlcv_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame(columns=["date"])
    frame = ohlcv.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    returns = close.pct_change()
    output = pd.DataFrame({"date": frame["date"]})
    output["close"] = close
    output["return_1d"] = returns * 100.0
    output["return_3d"] = close.pct_change(3) * 100.0
    output["return_5d"] = close.pct_change(5) * 100.0
    output["range_pct"] = (high - low) / close.replace(0.0, np.nan) * 100.0
    output["hv_10"] = returns.rolling(10, min_periods=5).std(ddof=0) * math.sqrt(TRADING_DAYS) * 100.0
    output["hv_20"] = returns.rolling(20, min_periods=10).std(ddof=0) * math.sqrt(TRADING_DAYS) * 100.0
    output["hv_30"] = returns.rolling(30, min_periods=15).std(ddof=0) * math.sqrt(TRADING_DAYS) * 100.0
    output["volume_z20"] = (volume - volume.rolling(20, min_periods=5).mean()) / volume.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    output["rsi_14"] = _rsi(close, 14)
    output["ema_9_21_spread"] = (_ema(close, 9) - _ema(close, 21)) / close.replace(0.0, np.nan) * 100.0
    output["ema_21_50_spread"] = (_ema(close, 21) - _ema(close, 50)) / close.replace(0.0, np.nan) * 100.0
    output["atr_pct"] = _atr_pct(frame, 14)
    output["day_of_week"] = pd.to_datetime(frame["date"]).dt.dayofweek
    output["month_of_year"] = pd.to_datetime(frame["date"]).dt.month
    return output.replace([np.inf, -np.inf], np.nan)


def _symbol_candidates(symbol: str) -> list[str]:
    upper = symbol.strip().upper()
    if upper == "NIFTY":
        return ["NIFTY"]
    if upper == "NIFTY 50":
        return ["NIFTY 50", "NIFTY"]
    if upper == "BANKNIFTY":
        return ["BANKNIFTY", "NIFTY BANK"]
    return [upper]


def _join_feature_history(ohlcv_features: pd.DataFrame, signal_history: pd.DataFrame) -> pd.DataFrame:
    if ohlcv_features.empty and signal_history.empty:
        return pd.DataFrame(columns=["date"])
    if ohlcv_features.empty:
        merged = signal_history.copy()
    elif signal_history.empty:
        merged = ohlcv_features.copy()
    else:
        left = ohlcv_features.copy()
        right = signal_history.copy()
        left["date"] = pd.to_datetime(left["date"]).dt.date
        right["date"] = pd.to_datetime(right["date"]).dt.date
        merged = left.merge(right, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)


def _join_bif_history(feature_frame: pd.DataFrame, bif_history: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty and bif_history.empty:
        return pd.DataFrame(columns=["date", *BIF_FEATURE_COLUMNS])
    if feature_frame.empty:
        merged = bif_history.copy()
    elif bif_history.empty:
        merged = feature_frame.copy()
    else:
        left = feature_frame.copy()
        right = bif_history.copy()
        left["date"] = pd.to_datetime(left["date"]).dt.date
        right["date"] = pd.to_datetime(right["date"]).dt.date
        merged = left.merge(right, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)


def _apply_bif_availability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        output = frame.copy()
        output["bif_available"] = pd.Series(dtype="int64")
        for column in BIF_FEATURE_COLUMNS:
            if column not in output.columns:
                output[column] = pd.Series(dtype="float64")
        return output

    output = frame.copy()
    for column in BIF_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")

    all_missing = output[list(BIF_FEATURE_COLUMNS)].isna().all(axis=1)
    available = ~all_missing
    output["bif_available"] = available.astype(int)
    medians = output.loc[available, list(BIF_FEATURE_COLUMNS)].median(numeric_only=True)
    medians = medians.reindex(BIF_FEATURE_COLUMNS).fillna(0.0)
    for column in BIF_FEATURE_COLUMNS:
        output[column] = output[column].fillna(float(medians[column]))
    return output


def _overlay_current_features(frame: pd.DataFrame, current_features: dict[str, float]) -> pd.DataFrame:
    if frame.empty:
        row = {"date": pd.Timestamp.utcnow().date(), **current_features}
        return pd.DataFrame([row])
    output = frame.copy()
    last_index = output.index[-1]
    for name, value in current_features.items():
        if name in output.columns:
            output.loc[last_index, name] = value
    return output


def _select_engineering_columns(columns: list[str], max_columns: int) -> list[str]:
    priority_terms = (
        "iv",
        "vix",
        "hv",
        "atr",
        "pcr",
        "gex",
        "fii",
        "dii",
        "return",
        "rsi",
        "ema",
        "volume",
        "score",
        "bif",
        "cds",
        "bms",
        "iad",
        "ngi",
        "vrc",
        "nfve",
    )
    priority = [col for col in columns if any(term in col.lower() for term in priority_terms)]
    rest = [col for col in columns if col not in priority]
    return (priority + rest)[:max_columns]


def _add_interaction(frame: pd.DataFrame, left: str, right: str, name: str) -> None:
    left_col = _find_column(frame, left)
    right_col = _find_column(frame, right)
    if left_col is None or right_col is None:
        return
    frame[name] = (
        pd.to_numeric(frame[left_col], errors="coerce")
        * pd.to_numeric(frame[right_col], errors="coerce")
    )


def _find_column(frame: pd.DataFrame, name: str) -> str | None:
    if name in frame.columns:
        return name
    matches = [col for col in frame.columns if col.endswith(name)]
    return matches[0] if matches else None


def _close_series(ohlcv: pd.DataFrame) -> pd.Series:
    if ohlcv.empty or "close" not in ohlcv.columns:
        return pd.Series(dtype="float64")
    dates = pd.to_datetime(ohlcv["date"]).dt.date
    return pd.Series(pd.to_numeric(ohlcv["close"], errors="coerce").to_numpy(), index=dates)


def _extract_spot(payload: dict[str, Any], ohlcv: pd.DataFrame) -> float:
    spot = _as_float(payload.get("spot"))
    if spot is not None and spot > 0:
        return spot
    if not ohlcv.empty:
        value = _as_float(ohlcv["close"].iloc[-1])
        if value is not None:
            return value
    return 0.0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=max(2, period // 2)).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _atr_pct(frame: pd.DataFrame, period: int) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    atr = ranges.max(axis=1).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr / close.replace(0.0, np.nan) * 100.0
