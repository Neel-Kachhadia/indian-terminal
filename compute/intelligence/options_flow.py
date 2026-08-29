from __future__ import annotations

from typing import Any

import pandas as pd


def compute_options_flow(
    symbol: str,
    legs: list[dict[str, Any]],
    spot: float,
) -> dict[str, Any]:
    if not legs:
        return {
            "symbol": symbol,
            "gamma_walls": [],
            "unusual_oi": [],
            "iv_skew": {"classification": "unavailable"},
            "term_structure": {"shape": "unavailable"},
            "market_maker_inventory": {"signal": "unavailable"},
            "computed_at": pd.Timestamp.utcnow().isoformat(),
        }
    frame = pd.DataFrame(legs)
    for col in ("strike", "oi", "oi_change", "iv", "gamma", "bid", "ask", "ltp"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    gamma = _gamma_walls(frame, spot)
    unusual = _unusual_oi(frame)
    skew = _iv_skew(frame, spot)
    term = _term_structure(frame, spot)
    inventory = _inventory_signal(frame)
    return {
        "symbol": symbol,
        "gamma_walls": gamma,
        "unusual_oi": unusual,
        "iv_skew": skew,
        "term_structure": term,
        "market_maker_inventory": inventory,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _gamma_walls(frame: pd.DataFrame, spot: float) -> list[dict[str, Any]]:
    if "gamma" not in frame.columns:
        frame["gamma"] = 0.0
    grouped = frame.assign(gamma_exposure=frame["gamma"].fillna(0.0) * frame["oi"].fillna(0.0) * spot * spot / 100.0)
    rows = grouped.groupby("strike", as_index=False)["gamma_exposure"].sum()
    rows["abs_gamma"] = rows["gamma_exposure"].abs()
    rows = rows.sort_values("abs_gamma", ascending=False).head(5)
    return [
        {
            "strike": float(row.strike),
            "gamma_exposure": round(float(row.gamma_exposure), 3),
            "distance_from_spot": round(float(row.strike - spot), 3),
        }
        for row in rows.itertuples()
    ]


def _unusual_oi(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "oi_change" not in frame.columns:
        return []
    clean = frame.copy()
    median = clean["oi_change"].abs().median() or 1.0
    clean["score"] = clean["oi_change"].abs() / median
    clean = clean.sort_values("score", ascending=False).head(5)
    return [
        {
            "strike": float(row.strike),
            "option_type": str(row.option_type),
            "oi_change": int(row.oi_change or 0),
            "score": round(float(row.score), 3),
        }
        for row in clean.itertuples()
        if float(row.score) >= 1.0
    ]


def _iv_skew(frame: pd.DataFrame, spot: float) -> dict[str, Any]:
    calls = frame[frame["option_type"].astype(str).str.upper() == "CE"]
    puts = frame[frame["option_type"].astype(str).str.upper() == "PE"]
    call = _nearest_iv(calls, spot * 1.01)
    put = _nearest_iv(puts, spot * 0.99)
    skew = None if call is None or put is None else put - call
    if skew is None:
        classification = "unavailable"
    elif skew > 0.04:
        classification = "steep"
    elif skew < -0.01:
        classification = "inverted"
    elif skew < 0.01:
        classification = "flat"
    else:
        classification = "normal"
    return {
        "put_25d_iv": put,
        "call_25d_iv": call,
        "skew": None if skew is None else round(float(skew), 6),
        "classification": classification,
    }


def _term_structure(frame: pd.DataFrame, spot: float) -> dict[str, Any]:
    if "expiry" not in frame.columns:
        return {"shape": "unavailable", "expiries": []}
    rows = []
    for expiry, group in frame.groupby("expiry"):
        iv = _nearest_iv(group, spot)
        if iv is not None:
            rows.append((str(expiry), iv))
    if len(rows) < 2:
        return {"shape": "single_expiry", "expiries": [{"expiry": exp, "atm_iv": iv} for exp, iv in rows]}
    shape = "backwardation" if rows[0][1] > rows[-1][1] else "contango"
    return {"shape": shape, "expiries": [{"expiry": exp, "atm_iv": iv} for exp, iv in rows]}


def _inventory_signal(frame: pd.DataFrame) -> dict[str, Any]:
    if not {"bid", "ask", "ltp"}.issubset(frame.columns):
        return {"signal": "unavailable"}
    mid = (frame["bid"].fillna(0.0) + frame["ask"].fillna(0.0)) / 2.0
    spread = (frame["ask"].fillna(0.0) - frame["bid"].fillna(0.0)).abs()
    ratio = (spread / mid.replace(0.0, pd.NA)).dropna()
    avg = float(ratio.mean()) if not ratio.empty else None
    if avg is None:
        signal = "unavailable"
    elif avg > 0.08:
        signal = "wide_spread_avoid"
    elif avg < 0.02:
        signal = "tight_liquid"
    else:
        signal = "normal"
    return {"signal": signal, "avg_spread_ratio": None if avg is None else round(avg, 6)}


def _nearest_iv(frame: pd.DataFrame, target: float) -> float | None:
    if frame.empty or "iv" not in frame.columns:
        return None
    local = frame.dropna(subset=["strike", "iv"]).copy()
    if local.empty:
        return None
    local["distance"] = (local["strike"] - target).abs()
    value = local.sort_values("distance").iloc[0]["iv"]
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
