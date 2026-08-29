from __future__ import annotations

from typing import Any

import pandas as pd


HISTORICAL = [
    ("COVID crash Mar 2020", -38.0),
    ("US rate shock Jun 2022", -17.0),
    ("Adani crisis Jan 2023", -8.0),
    ("IL&FS Sep 2018", -15.0),
    ("2008 global crisis", -60.0),
]

FORWARD = [
    ("RBI surprise 50bps hike", -4.0),
    ("USD/INR to 100", -7.0),
    ("Brent to $120", -6.0),
    ("VIX spike to 30", -5.0),
    ("China Taiwan event", -12.0),
]


def compute_stress_tests(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    spot: float = 0.0,
    lot_size: int | None = None,
) -> dict[str, Any]:
    if lot_size is None or lot_size <= 0:
        return {"symbol": symbol, "historical": [], "forward": [], "error": "validated lot size unavailable"}
    close = pd.to_numeric(ohlcv.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    current = float(spot or (close.iloc[-1] if not close.empty else 0.0))
    vol = float(close.pct_change().tail(20).std(ddof=0) * (252 ** 0.5) * 100.0) if len(close) > 25 else 15.0
    historical = [
        _scenario(name, move, current, lot_size, _similarity(vol, abs(move)))
        for name, move in HISTORICAL
    ]
    forward = [
        _scenario(name, move, current, lot_size, _similarity(vol, abs(move) * 1.6))
        for name, move in FORWARD
    ]
    return {
        "symbol": symbol,
        "historical_scenarios": historical,
        "forward_scenarios": forward,
        "top_risks": sorted(historical + forward, key=lambda item: item["similarity_score"], reverse=True)[:3],
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _scenario(name: str, move_pct: float, spot: float, lot_size: int, similarity: float) -> dict[str, Any]:
    return {
        "scenario": name,
        "index_move_pct": move_pct,
        "estimated_spot": round(spot * (1.0 + move_pct / 100.0), 2) if spot else None,
        "impact_per_lot": round(spot * lot_size * move_pct / 100.0, 2) if spot else None,
        "similarity_score": round(similarity, 3),
    }


def _similarity(current_vol: float, scenario_abs_move: float) -> float:
    return max(0.0, min(100.0, 100.0 - abs(current_vol - scenario_abs_move) * 2.0))
