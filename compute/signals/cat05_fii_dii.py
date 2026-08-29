from __future__ import annotations

from typing import Any

from .common import safe_div
from .framework import (
    DataSource,
    SignalType,
    SignalContext,
    build_signals,
    source_availability,
    specs_from_names,
)


CATEGORY = "fii_dii"

SIGNAL_NAMES = [
    "fii_net_cash_crores",
    "dii_net_cash_crores",
    "fii_gross_buy_crores",
    "fii_gross_sell_crores",
    "dii_gross_buy_crores",
    "dii_gross_sell_crores",
    "fii_dii_net_combined",
    "fii_dii_divergence",
    "fii_5d_cumulative",
    "fii_20d_cumulative",
    "fii_momentum_score",
    "fii_trend_direction",
    "dii_5d_cumulative",
    "dii_20d_cumulative",
    "fii_dii_ratio",
    "fii_buy_sell_ratio",
    "dii_buy_sell_ratio",
    "fii_index_futures_net",
    "fii_index_futures_long_short",
    "fii_stock_futures_net",
    "fii_options_net_premium",
    "prop_desk_net",
    "client_net",
    "institutional_flow_score_0_100",
    "domestic_support_score_0_100",
    "fii_cash_intensity",
    "dii_cash_intensity",
    "flow_conflict_signal",
    "foreign_risk_off_signal",
    "local_absorption_signal",
]

TYPE_OVERRIDES = {
    "fii_dii_divergence": SignalType.DIRECTIONAL,
    "fii_trend_direction": SignalType.DIRECTIONAL,
    "flow_conflict_signal": SignalType.DIRECTIONAL,
    "foreign_risk_off_signal": SignalType.BINARY,
    "local_absorption_signal": SignalType.BINARY,
}

SOURCE_OVERRIDES = {
    "fii_index_futures_net": DataSource.FO_EOD,
    "fii_index_futures_long_short": DataSource.FO_EOD,
    "fii_stock_futures_net": DataSource.FO_EOD,
    "fii_options_net_premium": DataSource.FO_EOD,
    "prop_desk_net": DataSource.FO_EOD,
    "client_net": DataSource.FO_EOD,
}

SPECS = specs_from_names(
    SIGNAL_NAMES,
    data_source=DataSource.FII_DII,
    type_overrides=TYPE_OVERRIDES,
    source_overrides=SOURCE_OVERRIDES,
)


def compute(ctx: SignalContext) -> dict[str, Any]:
    try:
        values = _compute_values(ctx)
    except Exception as error:
        print(f"signals: {CATEGORY} failed: {error}")
        values = {}
    return build_signals(CATEGORY, SPECS, values, source_availability(ctx))


def _compute_values(ctx: SignalContext) -> dict[str, Any]:
    data = ctx.fii_dii_data or {}
    if not data or data.get("error"):
        return {}
    fii = _number(data.get("fii_net_cash"))
    dii = _number(data.get("dii_net_cash"))
    fii_buy = _number(data.get("fii_buy_cash"))
    fii_sell = _number(data.get("fii_sell_cash"))
    dii_buy = _number(data.get("dii_buy_cash"))
    dii_sell = _number(data.get("dii_sell_cash"))
    combined = _add(fii, dii)
    values = {
        "fii_net_cash_crores": fii,
        "dii_net_cash_crores": dii,
        "fii_gross_buy_crores": fii_buy,
        "fii_gross_sell_crores": fii_sell,
        "dii_gross_buy_crores": dii_buy,
        "dii_gross_sell_crores": dii_sell,
        "fii_dii_net_combined": combined,
        "fii_dii_divergence": _divergence(fii, dii),
        "fii_dii_ratio": safe_div(abs(fii or 0.0), abs(dii or 0.0)),
        "fii_buy_sell_ratio": safe_div(fii_buy, fii_sell),
        "dii_buy_sell_ratio": safe_div(dii_buy, dii_sell),
        "institutional_flow_score_0_100": _flow_score(combined),
        "domestic_support_score_0_100": _flow_score(dii),
        "fii_cash_intensity": safe_div(fii, _add(fii_buy, fii_sell)),
        "dii_cash_intensity": safe_div(dii, _add(dii_buy, dii_sell)),
        "flow_conflict_signal": _divergence(fii, dii),
        "foreign_risk_off_signal": int((fii or 0.0) < -1000.0),
        "local_absorption_signal": int((fii or 0.0) < 0.0 and (dii or 0.0) > abs(fii or 0.0) * 0.5),
    }
    values["fii_5d_cumulative"] = None
    values["fii_20d_cumulative"] = None
    values["fii_momentum_score"] = None
    values["fii_trend_direction"] = None
    values["dii_5d_cumulative"] = None
    values["dii_20d_cumulative"] = None
    return values


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _add(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left + right


def _divergence(fii: float | None, dii: float | None) -> int | None:
    if fii is None or dii is None:
        return None
    if fii > 0 and dii < 0:
        return 1
    if fii < 0 and dii > 0:
        return -1
    return 0


def _flow_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, 50.0 + value / 100.0))
