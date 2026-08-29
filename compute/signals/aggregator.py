from __future__ import annotations

import datetime as _dt
import time as _time
from typing import Any, Callable

import pandas as pd

from db.duckdb_client import DB_WRITE_LOCK, writer_connection

from . import (
    cat01_price_action,
    cat02_volume,
    cat03_options,
    cat04_vix,
    cat05_fii_dii,
    cat06_cross_asset,
    cat07_macro,
    cat08_sentiment,
    cat09_patterns,
    cat10_expiry,
    cat11_statistical,
    cat12_microstructure,
    cat13_breadth,
    cat14_derivatives_flow,
    cat15_risk,
)
from .framework import Signal, SignalContext, numeric_history_value


CategoryFn = Callable[[SignalContext], dict[str, Signal]]

CATEGORY_FUNCTIONS: list[tuple[str, CategoryFn]] = [
    ("price_action", cat01_price_action.compute),
    ("volume", cat02_volume.compute),
    ("options", cat03_options.compute),
    ("vix", cat04_vix.compute),
    ("fii_dii", cat05_fii_dii.compute),
    ("cross_asset", cat06_cross_asset.compute),
    ("macro", cat07_macro.compute),
    ("sentiment", cat08_sentiment.compute),
    ("patterns", cat09_patterns.compute),
    ("expiry", cat10_expiry.compute),
    ("statistical", cat11_statistical.compute),
    ("microstructure", cat12_microstructure.compute),
    ("breadth", cat13_breadth.compute),
    ("derivatives_flow", cat14_derivatives_flow.compute),
    ("risk", cat15_risk.compute),
]


def compute_all_signals(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    chain_legs: list[dict[str, Any]],
    spot: float,
    expiry_str: str,
    cross_asset_snapshot: dict[str, Any],
    sentiment_data: dict[str, Any],
    fii_dii_data: dict[str, Any],
    india_vix: float | None,
    *,
    lot_size: int | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    start = _time.time()
    ctx = SignalContext(
        symbol=symbol,
        ohlcv_df=ohlcv_df,
        chain_legs=chain_legs,
        spot=float(spot or 0.0),
        expiry_str=expiry_str,
        cross_asset_snapshot=cross_asset_snapshot or {},
        sentiment_data=sentiment_data or {},
        fii_dii_data=fii_dii_data or {},
        india_vix=india_vix,
        lot_size=lot_size if isinstance(lot_size, int) and lot_size > 0 else None,
    )

    categories: dict[str, dict[str, Signal]] = {}
    for category_name, fn in CATEGORY_FUNCTIONS:
        try:
            categories[category_name] = fn(ctx)
        except Exception as error:
            print(f"signals: category {category_name} crashed: {error}")
            categories[category_name] = {}

    counts = _signal_counts(categories)
    composites = _composite_scores(categories)
    rows_written = _write_signal_history(symbol, categories, ohlcv_df) if persist else 0

    return {
        "symbol": symbol,
        "spot": spot,
        "lot_size": ctx.lot_size,
        "computed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "elapsed_ms": int((_time.time() - start) * 1000),
        "signal_counts": counts,
        "categories": {
            category: {
                name: signal.to_dict()
                for name, signal in signals.items()
            }
            for category, signals in categories.items()
        },
        "composite_scores": composites,
        "duckdb_rows_written": rows_written,
    }


def _signal_counts(
    categories: dict[str, dict[str, Signal]],
) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    total = 0
    non_null = 0
    for category, signals in categories.items():
        category_total = len(signals)
        category_non_null = sum(1 for signal in signals.values() if signal.value is not None)
        by_category[category] = {
            "total": category_total,
            "non_null": category_non_null,
            "null": category_total - category_non_null,
        }
        total += category_total
        non_null += category_non_null

    return {
        "total": total,
        "non_null": non_null,
        "null": total - non_null,
        "by_category": by_category,
    }


def _composite_scores(
    categories: dict[str, dict[str, Signal]],
) -> dict[str, float]:
    trend_score = _score_values([
        _value(categories, "price_action", "trend_composite_0_100"),
        _value(categories, "price_action", "ema_alignment_score", scale=25.0),
        _value(categories, "patterns", "breakout_confirmation_score"),
    ])
    momentum_score = _score_values([
        _value(categories, "price_action", "momentum_composite_0_100"),
        _value(categories, "statistical", "statistical_edge_score_0_100"),
    ])
    volatility_score = _score_values([
        _inverse_score(_value(categories, "vix", "vix_normalised_0_100")),
        _inverse_score(_value(categories, "risk", "volatility_risk_score")),
        _vol_risk_score(_value(categories, "options", "vol_risk_premium")),
    ])
    flow_score = _score_values([
        _value(categories, "fii_dii", "institutional_flow_score_0_100"),
        _pcr_flow_score(_value(categories, "options", "pcr_total")),
        _value(categories, "derivatives_flow", "derivatives_activity_score"),
    ])
    sentiment_score = _score_values([
        _value(categories, "sentiment", "sentiment_score_4h"),
        _value(categories, "sentiment", "fear_greed_composite"),
    ])
    risk_score = _score_values([
        _value(categories, "risk", "market_risk_score_0_100"),
        _value(categories, "risk", "systemic_risk_indicator"),
        _value(categories, "cross_asset", "global_risk_score_0_100"),
    ])

    overall = (
        trend_score * 0.20
        + momentum_score * 0.15
        + volatility_score * 0.15
        + flow_score * 0.25
        + sentiment_score * 0.15
        + (100.0 - risk_score) * 0.10
    )

    return {
        "trend_score": round(_clamp_score(trend_score), 3),
        "momentum_score": round(_clamp_score(momentum_score), 3),
        "volatility_score": round(_clamp_score(volatility_score), 3),
        "flow_score": round(_clamp_score(flow_score), 3),
        "sentiment_score": round(_clamp_score(sentiment_score), 3),
        "risk_score": round(_clamp_score(risk_score), 3),
        "overall_score": round(_clamp_score(overall), 3),
    }


def _write_signal_history(
    symbol: str,
    categories: dict[str, dict[str, Signal]],
    ohlcv_df: pd.DataFrame,
) -> int:
    rows: list[tuple[str, str, str, float]] = []
    if ohlcv_df.empty or "date" not in ohlcv_df.columns:
        raise ValueError("persistent signal computation requires an explicit market-data as-of date")
    as_of = pd.to_datetime(ohlcv_df["date"], errors="coerce").dropna()
    if as_of.empty:
        raise ValueError("persistent signal computation has no valid market-data as-of date")
    market_date = as_of.max().date().isoformat()
    for category, signals in categories.items():
        for name, signal in signals.items():
            value = numeric_history_value(signal)
            if value is None:
                continue
            rows.append((market_date, f"{category}.{name}", symbol, float(value)))

    if not rows:
        return 0

    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS signals_history (
                    date        DATE    NOT NULL,
                    signal_name TEXT    NOT NULL,
                    symbol      TEXT    NOT NULL,
                    value       DOUBLE,
                    PRIMARY KEY (date, signal_name, symbol)
                )
                """
            )
            con.execute("BEGIN TRANSACTION")
            for date, signal_name, row_symbol, value in rows:
                con.execute(
                    """
                    INSERT OR REPLACE INTO signals_history
                        (date, signal_name, symbol, value)
                    VALUES (CAST(? AS DATE), ?, ?, ?)
                    """,
                    [date, signal_name, row_symbol, value],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()
    return len(rows)


def _value(
    categories: dict[str, dict[str, Signal]],
    category: str,
    name: str,
    *,
    scale: float = 1.0,
) -> float | None:
    signal = categories.get(category, {}).get(name)
    if signal is None or signal.value is None:
        return None
    try:
        return float(signal.value) * scale
    except (TypeError, ValueError):
        return None


def _score_values(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 50.0
    return _clamp_score(sum(clean) / len(clean))


def _inverse_score(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 - value


def _pcr_flow_score(pcr: float | None) -> float | None:
    if pcr is None:
        return None
    return _clamp_score(50.0 + (pcr - 1.0) * 50.0)


def _vol_risk_score(value: float | None) -> float | None:
    if value is None:
        return None
    return _clamp_score(50.0 - value * 2.0)


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))
