from __future__ import annotations

import math
from typing import Any


NEUTRAL_INDIA_ERP = 2.0
RUPEE_BASELINE = 75.0


def compute_nfve(
    symbol: str,
    spot: float,
    ohlcv_rows: list[dict[str, Any]],
    cross_asset: dict[str, Any],
    market_signals: dict[str, Any],
    flow_history: dict[str, list[float]],
    signal_values: dict[str, float],
) -> dict[str, Any]:
    """Nifty fair value from explicitly observed financial inputs only.

    OHLC-derived EPS, default FX/VIX, and repo-plus-spread substitutions are
    deliberately forbidden: they can be useful scenario assumptions, but are
    not measured inputs and therefore cannot support a live valuation label.
    """
    del symbol, ohlcv_rows
    spot_value = _finite(spot)
    eps, eps_source = _earnings(signal_values)
    bond_yield, bond_source = _bond_yield(cross_asset, signal_values)
    usdinr, fx_source = _usdinr(cross_asset)
    vix, vix_source = _india_vix(cross_asset, market_signals)
    fii_net_ytd, fii_source = _fii_ytd(flow_history, market_signals)

    values = {
        "decision_eligible_spot": spot_value if spot_value and spot_value > 0 else None,
        "index_eps": eps,
        "india_10y_yield": bond_yield,
        "usdinr": usdinr,
        "india_vix": vix,
        "fii_net_ytd": fii_net_ytd,
    }
    missing = [name for name, value in values.items() if value is None]
    sources = {
        "index_eps": eps_source,
        "india_10y_yield": bond_source,
        "usdinr": fx_source,
        "india_vix": vix_source,
        "fii_net_ytd": fii_source,
    }
    if missing:
        return _unavailable(missing, sources)

    assert spot_value is not None
    assert eps is not None
    assert bond_yield is not None
    assert usdinr is not None
    assert vix is not None
    assert fii_net_ytd is not None
    implied_pe = spot_value / eps
    earnings_yield = (1.0 / implied_pe) * 100.0
    equity_risk_premium = earnings_yield - bond_yield
    erp_adjustment = (equity_risk_premium - NEUTRAL_INDIA_ERP) / 100.0
    currency_discount = max(
        0.85,
        1.0 - ((usdinr - RUPEE_BASELINE) / RUPEE_BASELINE) * 0.3,
    )
    fii_liquidity_premium = 1.0 + math.tanh(fii_net_ytd / 100_000.0) * 0.1
    vix_discount = 1.0 - max(0.0, (vix - 15.0) / 100.0)
    fair_value = (
        spot_value
        * (1.0 + erp_adjustment)
        * currency_discount
        * fii_liquidity_premium
        * vix_discount
    )
    current_vs_fair = (
        (spot_value - fair_value) / fair_value * 100.0
        if fair_value > 0
        else None
    )
    if current_vs_fair is None:
        return _unavailable(["positive_fair_value"], sources)

    return {
        "fair_value": round(fair_value, 3),
        "current_vs_fair": round(current_vs_fair, 3),
        "valuation": _valuation(current_vs_fair),
        "equity_risk_premium": round(equity_risk_premium, 3),
        "implied_pe": round(implied_pe, 3),
        "currency_impact": round(currency_discount, 4),
        "fii_liquidity_premium": round(fii_liquidity_premium, 4),
        "confidence": "high",
        "input_sources": sources,
        "missing_inputs": [],
        "reason_code": None,
        "is_valid": True,
    }


def _unavailable(
    missing: list[str],
    sources: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "fair_value": None,
        "current_vs_fair": None,
        "valuation": None,
        "equity_risk_premium": None,
        "implied_pe": None,
        "currency_impact": None,
        "fii_liquidity_premium": None,
        "confidence": None,
        "input_sources": sources,
        "missing_inputs": missing,
        "reason_code": "MISSING_MEASURED_FINANCIAL_INPUTS",
        "is_valid": False,
    }


def _earnings(signal_values: dict[str, float]) -> tuple[float | None, str | None]:
    for key in ("nifty_eps", "index_eps", "macro.nifty_eps", "fundamental.eps"):
        value = _finite(signal_values.get(key))
        if value is not None and value > 0:
            return value, f"signal:{key}"
    return None, None


def _bond_yield(
    cross_asset: dict[str, Any],
    signal_values: dict[str, float],
) -> tuple[float | None, str | None]:
    for key in (
        "bond_yield_10y",
        "india_10y_yield",
        "gsec_10y_yield",
        "macro.bond_yield_10y",
        "macro.gsec_10y_yield",
    ):
        value = _finite(signal_values.get(key))
        if value is not None and value > 0:
            return value, f"signal:{key}"

    snapshot = _snapshot(cross_asset)
    for key in ("india_10y_yield", "gsec_10y", "bond_yield_10y"):
        raw = snapshot.get(key)
        if isinstance(raw, dict):
            raw = raw.get("yield") if raw.get("yield") is not None else raw.get("price")
        value = _finite(raw)
        if value is not None and value > 0:
            return value, f"cross_asset:{key}"
    return None, None


def _usdinr(cross_asset: dict[str, Any]) -> tuple[float | None, str | None]:
    quote = _snapshot(cross_asset).get("usdinr")
    raw = quote.get("price") if isinstance(quote, dict) else quote
    value = _finite(raw)
    return (value, "cross_asset:usdinr") if value is not None and value > 0 else (None, None)


def _india_vix(
    cross_asset: dict[str, Any],
    market_signals: dict[str, Any],
) -> tuple[float | None, str | None]:
    raw = _snapshot(cross_asset).get("india_vix")
    if isinstance(raw, dict):
        raw = raw.get("price")
    value = _finite(raw)
    if value is not None and value > 0:
        return value, "cross_asset:india_vix"
    value = _finite(market_signals.get("india_vix"))
    if value is not None and value > 0:
        return value, "market:india_vix"
    return None, None


def _fii_ytd(
    flow_history: dict[str, list[float]],
    market_signals: dict[str, Any],
) -> tuple[float | None, str | None]:
    values = flow_history.get("fii_net_cash") or []
    clean = [number for value in values[-252:] if (number := _finite(value)) is not None]
    if clean:
        return float(sum(clean)), "flow_history:fii_net_cash"
    fii_dii = market_signals.get("fii_dii")
    if isinstance(fii_dii, dict):
        value = _finite(fii_dii.get("fii_net_cash"))
        if value is not None:
            return value, "market:fii_net_cash"
    return None, None


def _snapshot(cross_asset: dict[str, Any]) -> dict[str, Any]:
    snapshot = cross_asset.get("snapshot")
    return snapshot if isinstance(snapshot, dict) else cross_asset


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
