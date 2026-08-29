from __future__ import annotations

from typing import Any

import pandas as pd


TEMPLATES = {
    "atm_straddle": [("CE", 0, 1), ("PE", 0, 1)],
    "iron_condor": [("PE", -2, -1), ("PE", -1, 1), ("CE", 1, 1), ("CE", 2, -1)],
    "bull_spread": [("CE", 0, 1), ("CE", 1, -1)],
    "bear_spread": [("PE", 0, 1), ("PE", -1, -1)],
}


def compute_greeks_dashboard(
    symbol: str,
    legs: list[dict[str, Any]],
    spot: float,
) -> dict[str, Any]:
    if not legs:
        return {"symbol": symbol, "templates": {}, "error": "chain unavailable"}
    frame = pd.DataFrame(legs)
    for col in ("strike", "delta", "gamma", "theta", "vega", "ltp"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    strikes = sorted(float(value) for value in frame["strike"].dropna().unique())
    if not strikes:
        return {"symbol": symbol, "templates": {}, "error": "no strikes"}
    atm_index = min(range(len(strikes)), key=lambda idx: abs(strikes[idx] - spot))
    templates = {
        name: _template_payload(frame, strikes, atm_index, spec, spot)
        for name, spec in TEMPLATES.items()
    }
    return {
        "symbol": symbol,
        "spot": spot,
        "templates": templates,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
    }


def _template_payload(
    frame: pd.DataFrame,
    strikes: list[float],
    atm_index: int,
    spec: list[tuple[str, int, int]],
    spot: float,
) -> dict[str, Any]:
    rows = []
    for option_type, offset, side in spec:
        index = max(0, min(len(strikes) - 1, atm_index + offset))
        strike = strikes[index]
        leg = _find_leg(frame, strike, option_type)
        if leg is None:
            continue
        rows.append((leg, side))
    # A missing per-leg Greek/LTP must never be treated as a zero contribution
    # to the NET figure -- this is a risk dashboard; silently coalescing an
    # unavailable Greek to 0.0 would understate real net exposure instead of
    # disclosing that the net figure can't be trusted. If any included leg is
    # missing a given field, the corresponding net value is reported as
    # explicitly unavailable (None) rather than fabricated.
    net: dict[str, float | None] = {}
    for greek in ("delta", "gamma", "theta", "vega"):
        terms = []
        complete = bool(rows)
        for leg, side in rows:
            raw = leg.get(greek)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                complete = False
                break
            terms.append(float(raw) * side)
        net[greek] = round(sum(terms), 6) if complete else None

    premium: float | None = None
    if rows:
        premium_terms = []
        premium_complete = True
        for leg, side in rows:
            raw = leg.get("ltp")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                premium_complete = False
                break
            premium_terms.append(float(raw) * side)
        premium = sum(premium_terms) if premium_complete else None

    pnl_moves: dict[str, float] = {}
    breakevens: list[float] = []
    if premium is not None:
        for pct in (-3, -2, -1, 1, 2, 3):
            moved = spot * (1.0 + pct / 100.0)
            pnl_moves[f"{pct:+d}%"] = round(_expiry_value(rows, moved) - premium, 2)
        breakevens = _breakeven_range(rows, spot, premium)

    theta = net.get("theta")
    return {
        "net_greeks": net,
        "theta_decay_today": round(theta * 75.0, 2) if theta is not None else None,
        "pnl_if_spot_moves": pnl_moves,
        "break_even_range": breakevens,
        "data_complete": premium is not None and all(value is not None for value in net.values()),
        "legs": [
            {"strike": float(leg.get("strike")), "option_type": leg.get("option_type"), "side": side}
            for leg, side in rows
        ],
    }


def _find_leg(frame: pd.DataFrame, strike: float, option_type: str) -> dict[str, Any] | None:
    match = frame[(frame["strike"] == strike) & (frame["option_type"].astype(str).str.upper() == option_type)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _expiry_value(rows: list[tuple[dict[str, Any], int]], spot: float) -> float:
    total = 0.0
    for leg, side in rows:
        strike = float(leg.get("strike") or 0.0)
        if str(leg.get("option_type")).upper() == "CE":
            value = max(0.0, spot - strike)
        else:
            value = max(0.0, strike - spot)
        total += value * side
    return total


def _breakeven_range(rows: list[tuple[dict[str, Any], int]], spot: float, premium: float) -> list[float]:
    values = []
    for pct in [item / 10 for item in range(-100, 101)]:
        moved = spot * (1.0 + pct / 100.0)
        if abs(_expiry_value(rows, moved) - premium) < spot * 0.002:
            values.append(round(moved, 2))
    if not values:
        return []
    return [min(values), max(values)]
