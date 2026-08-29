from __future__ import annotations

import math
from typing import Any

import numpy as np

from .common import (
    current_price,
    historical_vol,
    last,
    max_pain,
    numeric_series,
    option_chain_summary,
    returns,
    safe_div,
    z_score,
)
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "statistical"

SIGNAL_NAMES = [
    "price_z_score_20d",
    "return_z_score_20d",
    "returns_skewness_20d",
    "returns_kurtosis_20d",
    "autocorrelation_lag1",
    "sharpe_60d",
    "sortino_60d",
    "max_drawdown_30d",
    "max_drawdown_90d",
    "calmar_ratio",
    "omega_ratio_60d",
    "hurst_exponent_100d",
    "hurst_regime",
    "volatility_regime",
    "vol_risk_premium",
    "vol_risk_premium_z_score",
    "synthetic_futures_premium",
    "information_ratio_60d",
    "tracking_error_60d",
    "beta_vs_nifty",
    "correlation_vs_vix_30d",
    "tail_risk_cvar_95",
    "realized_vol_5d",
    "realized_vol_20d",
    "downside_deviation_60d",
    "upside_capture_60d",
    "mean_reversion_score",
    "trend_persistence_score",
    "return_entropy_60d",
    "fat_tail_warning",
    "statistical_edge_score_0_100",
]

TYPE_OVERRIDES = {
    "hurst_regime": SignalType.CATEGORICAL,
    "volatility_regime": SignalType.CATEGORICAL,
    "fat_tail_warning": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    "synthetic_futures_premium": DataSource.OPTIONS_CHAIN,
    "vol_risk_premium": DataSource.OPTIONS_CHAIN,
    "vol_risk_premium_z_score": DataSource.OPTIONS_CHAIN,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.OHLCV_DAILY,
    type_overrides=TYPE_OVERRIDES,
    source_overrides=SOURCE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values, histories = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values, histories = {}, {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx), histories=histories)


def _compute_values(ctx: SignalContext) -> tuple[dict[str, Any], dict[str, Any]]:
    df = ctx.ohlcv_df
    if df is None or getattr(df, "empty", True):
        return {}, {}

    close = numeric_series(df, "close")
    ret = returns(close).dropna()
    price = current_price(df, ctx.spot)
    hv_5 = historical_vol(close, 5)
    hv_20 = historical_vol(close, 20)
    hv20_value = last(hv_20)
    hurst = _hurst(close.dropna().tail(100).to_numpy(dtype=float))
    summary = option_chain_summary(ctx.chain_legs, ctx.spot)
    atm_call = summary.get("atm_call") or {}
    atm_put = summary.get("atm_put") or {}
    atm_iv = atm_call.get("iv")
    synthetic = None
    if summary.get("atm_strike") is not None and atm_call and atm_put and ctx.spot > 0:
        synthetic = atm_call["ltp"] - atm_put["ltp"] + summary["atm_strike"] - ctx.spot
    vol_risk = atm_iv * 100.0 - hv20_value if atm_iv is not None and hv20_value is not None else None

    values = {
        "price_z_score_20d": z_score(price, close.dropna().tail(20).tolist()),
        "return_z_score_20d": z_score(ret.iloc[-1] if len(ret) else None, ret.tail(20).tolist()),
        "returns_skewness_20d": float(ret.tail(20).skew()) if len(ret) >= 20 else None,
        "returns_kurtosis_20d": float(ret.tail(20).kurt()) if len(ret) >= 20 else None,
        "autocorrelation_lag1": float(ret.tail(60).autocorr(lag=1)) if len(ret) >= 20 else None,
        "sharpe_60d": _sharpe(ret.tail(60)),
        "sortino_60d": _sortino(ret.tail(60)),
        "max_drawdown_30d": _max_drawdown(close.tail(30)),
        "max_drawdown_90d": _max_drawdown(close.tail(90)),
        "calmar_ratio": _calmar(close.tail(252)),
        "omega_ratio_60d": _omega(ret.tail(60)),
        "hurst_exponent_100d": hurst,
        "hurst_regime": _hurst_regime(hurst),
        "volatility_regime": _vol_regime(hv20_value),
        "vol_risk_premium": vol_risk,
        "vol_risk_premium_z_score": None,
        "synthetic_futures_premium": synthetic,
        "information_ratio_60d": None,
        "tracking_error_60d": None,
        "beta_vs_nifty": 1.0 if ctx.symbol.strip().upper() in {"NIFTY", "NIFTY 50"} else None,
        "correlation_vs_vix_30d": None,
        "tail_risk_cvar_95": _cvar(ret, 0.05),
        "realized_vol_5d": last(hv_5),
        "realized_vol_20d": hv20_value,
        "downside_deviation_60d": _downside_deviation(ret.tail(60)),
        "upside_capture_60d": safe_div(ret.tail(60).where(ret > 0).mean(), abs(ret.tail(60).where(ret < 0).mean())),
        "mean_reversion_score": _mean_reversion_score(ret),
        "trend_persistence_score": _trend_persistence_score(hurst),
        "return_entropy_60d": _entropy(ret.tail(60)),
        "fat_tail_warning": int((float(ret.tail(20).kurt()) if len(ret) >= 20 else 0.0) > 3.0),
    }
    values["statistical_edge_score_0_100"] = _edge_score(values)
    histories = {
        "price_z_score_20d": close.tolist(),
        "realized_vol_20d": hv_20.tolist(),
        "return_z_score_20d": ret.tolist(),
    }
    return values, histories


def _hurst(values: np.ndarray) -> float | None:
    if len(values) < 40:
        return None
    lags = [8, 12, 16, 24, 32]
    rs_values = []
    used_lags = []
    for lag in lags:
        if lag >= len(values):
            continue
        chunks = len(values) // lag
        chunk_rs = []
        for index in range(chunks):
            chunk = values[index * lag:(index + 1) * lag]
            chunk = chunk - np.mean(chunk)
            cumulative = np.cumsum(chunk)
            spread = np.max(cumulative) - np.min(cumulative)
            std = np.std(chunk)
            if std > 1e-12:
                chunk_rs.append(spread / std)
        if chunk_rs:
            used_lags.append(lag)
            rs_values.append(np.mean(chunk_rs))
    if len(rs_values) < 2:
        return 0.5
    slope = np.polyfit(np.log(used_lags), np.log(rs_values), 1)[0]
    return max(0.3, min(0.8, float(slope)))


def _hurst_regime(hurst: float | None) -> str | None:
    if hurst is None:
        return None
    if hurst > 0.6:
        return "trending"
    if hurst < 0.4:
        return "ranging"
    return "random"


def _vol_regime(hv: float | None) -> str | None:
    if hv is None:
        return None
    if hv < 10:
        return "low"
    if hv > 20:
        return "high"
    return "normal"


def _sharpe(ret: Any) -> float | None:
    if len(ret.dropna()) < 10:
        return None
    excess = ret - 0.065 / 252.0
    std = excess.std(ddof=0)
    return float(excess.mean() / std * math.sqrt(252.0)) if std > 1e-12 else None


def _sortino(ret: Any) -> float | None:
    if len(ret.dropna()) < 10:
        return None
    downside = ret[ret < 0]
    std = downside.std(ddof=0)
    return float((ret.mean() - 0.065 / 252.0) / std * math.sqrt(252.0)) if std and std > 1e-12 else None


def _max_drawdown(close: Any) -> float | None:
    clean = close.dropna()
    if len(clean) < 2:
        return None
    peak = clean.cummax()
    drawdown = (clean / peak - 1.0) * 100.0
    return float(drawdown.min())


def _calmar(close: Any) -> float | None:
    clean = close.dropna()
    if len(clean) < 20:
        return None
    annual_return = (clean.iloc[-1] / clean.iloc[0] - 1.0) * 252.0 / len(clean)
    dd = abs(_max_drawdown(clean) or 0.0) / 100.0
    return annual_return / dd if dd > 1e-12 else None


def _omega(ret: Any) -> float | None:
    clean = ret.dropna()
    gains = clean[clean > 0].sum()
    losses = abs(clean[clean < 0].sum())
    return float(gains / losses) if losses > 1e-12 else None


def _cvar(ret: Any, alpha: float) -> float | None:
    clean = ret.dropna()
    if len(clean) < 20:
        return None
    threshold = clean.quantile(alpha)
    tail = clean[clean <= threshold]
    return float(tail.mean() * 100.0) if len(tail) else None


def _downside_deviation(ret: Any) -> float | None:
    downside = ret.dropna()
    downside = downside[downside < 0]
    return float(downside.std(ddof=0) * math.sqrt(252.0) * 100.0) if len(downside) else None


def _mean_reversion_score(ret: Any) -> float | None:
    if len(ret.dropna()) < 20:
        return None
    autocorr = ret.tail(60).autocorr(lag=1)
    return max(0.0, min(100.0, 50.0 - autocorr * 100.0))


def _trend_persistence_score(hurst: float | None) -> float | None:
    if hurst is None:
        return None
    return max(0.0, min(100.0, (hurst - 0.3) / 0.5 * 100.0))


def _entropy(ret: Any) -> float | None:
    clean = ret.dropna()
    if len(clean) < 20:
        return None
    hist, _ = np.histogram(clean, bins=10)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _edge_score(values: dict[str, Any]) -> float | None:
    pieces = []
    sharpe = values.get("sharpe_60d")
    if sharpe is not None:
        pieces.append(max(0.0, min(100.0, 50.0 + float(sharpe) * 15.0)))
    hurst_score = values.get("trend_persistence_score")
    if hurst_score is not None:
        pieces.append(float(hurst_score))
    omega = values.get("omega_ratio_60d")
    if omega is not None:
        pieces.append(max(0.0, min(100.0, float(omega) * 50.0)))
    return sum(pieces) / len(pieces) if pieces else None
