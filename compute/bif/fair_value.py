from __future__ import annotations

import math
import statistics
from typing import Any


NEUTRAL_INDIA_ERP = 2.0
RUPEE_BASELINE = 75.0
DEFAULT_REPO_RATE = 6.25
DEFAULT_GSEC_SPREAD = 0.65
NEUTRAL_PE = 22.5


def compute_nfve(
    symbol: str,
    spot: float,
    ohlcv_rows: list[dict[str, Any]],
    cross_asset: dict[str, Any],
    market_signals: dict[str, Any],
    flow_history: dict[str, list[float]],
    signal_values: dict[str, float],
) -> dict[str, Any]:
    """
    Nifty Fair Value Engine.

    If explicit EPS or India 10Y data is unavailable, derives a low-confidence
    proxy from local OHLCV history and existing repo-rate macro signal.
    """
    del symbol
    spot = _finite_or_zero(spot)
    closes = [_finite(row.get("close")) for row in ohlcv_rows]
    closes = [value for value in closes if value is not None and value > 0]
    eps, eps_source = _earnings_proxy(spot, closes, signal_values)
    bond_yield, bond_source = _bond_yield(cross_asset, signal_values)
    usdinr = _usdinr(cross_asset)
    vix = _india_vix(cross_asset, market_signals)
    fii_net_ytd = _fii_ytd(flow_history, market_signals)

    if spot <= 0 or eps <= 0:
        return {
            "fair_value": None,
            "current_vs_fair": None,
            "valuation": "FAIRLY_VALUED",
            "equity_risk_premium": None,
            "implied_pe": None,
            "currency_impact": None,
            "fii_liquidity_premium": None,
            "confidence": "low",
            "is_valid": False,
        }

    implied_pe = spot / eps
    earnings_yield = (1.0 / implied_pe) * 100.0 if implied_pe > 0 else 0.0
    equity_risk_premium = earnings_yield - bond_yield
    erp_adjustment = (equity_risk_premium - NEUTRAL_INDIA_ERP) / 100.0
    currency_discount = max(0.85, 1.0 - ((usdinr - RUPEE_BASELINE) / RUPEE_BASELINE) * 0.3)
    fii_liquidity_premium = 1.0 + math.tanh(fii_net_ytd / 100_000.0) * 0.1
    vix_discount = 1.0 - max(0.0, (vix - 15.0) / 100.0)
    fair_value = (
        spot
        * (1.0 + erp_adjustment)
        * currency_discount
        * fii_liquidity_premium
        * vix_discount
    )
    current_vs_fair = (spot - fair_value) / fair_value * 100.0 if fair_value > 0 else 0.0

    confidence = _confidence(eps_source, bond_source, usdinr, vix)
    return {
        "fair_value": round(fair_value, 3),
        "current_vs_fair": round(current_vs_fair, 3),
        "valuation": _valuation(current_vs_fair),
        "equity_risk_premium": round(equity_risk_premium, 3),
        "implied_pe": round(implied_pe, 3),
        "currency_impact": round(currency_discount, 4),
        "fii_liquidity_premium": round(fii_liquidity_premium, 4),
        "confidence": confidence,
        "is_valid": True,
    }


def _earnings_proxy(
    spot: float,
    closes: list[float],
    signal_values: dict[str, float],
) -> tuple[float, str]:
    for key in ("nifty_eps", "index_eps", "macro.nifty_eps", "fundamental.eps"):
        value = _finite(signal_values.get(key))
        if value is not None and value > 0:
            return value, "signal"

    if closes:
        window = closes[-252:] if len(closes) >= 30 else closes
        median_close = statistics.median(window)
        if median_close > 0:
            return median_close / NEUTRAL_PE, "ohlcv_pe_proxy"

    return max(spot, 1.0) / NEUTRAL_PE, "spot_pe_proxy"


def _bond_yield(cross_asset: dict[str, Any], signal_values: dict[str, float]) -> tuple[float, str]:
    for key in (
        "bond_yield_10y",
        "india_10y_yield",
        "gsec_10y_yield",
        "macro.bond_yield_10y",
        "macro.gsec_10y_yield",
    ):
        value = _finite(signal_values.get(key))
        if value is not None and value > 0:
            return value, "signal"

    snapshot = _snapshot(cross_asset)
    for key in ("india_10y_yield", "gsec_10y", "bond_yield_10y"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            value = value.get("price") or value.get("yield")
        number = _finite(value)
        if number is not None and number > 0:
            return number, "cross_asset"

    repo = _finite(signal_values.get("macro.repo_rate") or signal_values.get("repo_rate"))
    if repo is None:
        repo = DEFAULT_REPO_RATE
    return repo + DEFAULT_GSEC_SPREAD, "repo_proxy"


def _usdinr(cross_asset: dict[str, Any]) -> float:
    snapshot = _snapshot(cross_asset)
    quote = snapshot.get("usdinr")
    if isinstance(quote, dict):
        value = _finite(quote.get("price"))
        if value is not None and value > 0:
            return value
    return 83.5


def _india_vix(cross_asset: dict[str, Any], market_signals: dict[str, Any]) -> float:
    snapshot = _snapshot(cross_asset)
    value = _finite(snapshot.get("india_vix"))
    if value is not None and value > 0:
        return value
    value = _finite(market_signals.get("india_vix"))
    if value is not None and value > 0:
        return value
    return 15.0


def _fii_ytd(flow_history: dict[str, list[float]], market_signals: dict[str, Any]) -> float:
    values = flow_history.get("fii_net_cash") or []
    clean = [value for value in values if math.isfinite(value)]
    if clean:
        return float(sum(clean[-252:]))
    fii_dii = market_signals.get("fii_dii")
    if isinstance(fii_dii, dict):
        return _finite(fii_dii.get("fii_net_cash")) or 0.0
    return 0.0


def _snapshot(cross_asset: dict[str, Any]) -> dict[str, Any]:
    snapshot = cross_asset.get("snapshot")
    return snapshot if isinstance(snapshot, dict) else cross_asset


def _confidence(eps_source: str, bond_source: str, usdinr: float, vix: float) -> str:
    score = 0
    if eps_source == "signal":
        score += 2
    elif eps_source == "ohlcv_pe_proxy":
        score += 1
    if bond_source in {"signal", "cross_asset"}:
        score += 2
    elif bond_source == "repo_proxy":
        score += 1
    if usdinr > 0:
        score += 1
    if vix > 0:
        score += 1
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _valuation(current_vs_fair: float) -> str:
    if current_vs_fair > 5:
        return "OVERVALUED"
    if current_vs_fair < -5:
        return "UNDERVALUED"
    return "FAIRLY_VALUED"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_or_zero(value: Any) -> float:
    number = _finite(value)
    return number if number is not None else 0.0
