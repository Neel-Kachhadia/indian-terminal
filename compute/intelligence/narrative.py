from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read

from .groq_client import groq_complete, _template_narrative


def generate_market_brief(
    symbol: str,
    signals: dict[str, Any],
    prediction: dict[str, Any],
    regime: dict[str, Any],
    *,
    persist: bool = False,
) -> dict[str, Any]:
    context = _brief_context(symbol, signals, prediction, regime)
    prompt = _market_brief_prompt(context)
    if _has_enough_context(context):
        brief, model, live = groq_complete(prompt, max_tokens=500, temperature=0.3)
    else:
        brief = _template_narrative(context)
        model = "template_fallback"
        live = False
    brief = _enforce_brief_contract(brief, context)
    payload = {
        "symbol": symbol,
        "brief": brief,
        "brief_type": "market_brief",
        "model": model,
        "groq_live": live,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "word_count": len(brief.split()),
    }
    if persist:
        _log_narrative(symbol, payload, context)
    return payload


def prediction_narrative(
    symbol: str,
    signals: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    context = _brief_context(symbol, signals, prediction, {})
    return _template_narrative(context).replace("NIFTY", symbol, 1)


def _brief_context(
    symbol: str,
    signals: dict[str, Any],
    prediction: dict[str, Any],
    regime: dict[str, Any],
) -> dict[str, Any]:
    options = signals.get("options", {}) if isinstance(signals, dict) else {}
    market = signals.get("market", {}) if isinstance(signals, dict) else {}
    all_signals = signals.get("_all_signals", {}) if isinstance(signals, dict) else {}
    latest = _latest_signal_values(symbol)
    spot = _first_positive(
        signals.get("spot"),
        prediction.get("spot"),
        all_signals.get("spot") if isinstance(all_signals, dict) else None,
    )
    gex = _first_value(options.get("gex"), _signal_value(all_signals, "options", "gex"), latest.get("options.gex"))
    gex_flip = _first_positive(
        options.get("gex_flip"),
        _signal_value(all_signals, "options", "gex_flip"),
        latest.get("options.gex_flip"),
    )
    max_pain = _first_positive(
        options.get("max_pain"),
        _signal_value(all_signals, "options", "max_pain"),
        latest.get("options.max_pain"),
    )
    pcr = _first_positive(
        options.get("pcr"),
        options.get("pcr_total"),
        _signal_value(all_signals, "options", "pcr_total"),
        latest.get("options.pcr_total"),
    )
    pcr_5d = _first_positive(
        options.get("pcr_trend"),
        options.get("pcr_5d_ma"),
        _signal_value(all_signals, "options", "pcr_5d_ma"),
        latest.get("options.pcr_5d_ma"),
    )
    vix = _first_positive(
        market.get("india_vix"),
        _signal_value(all_signals, "vix", "vix_current"),
        latest.get("vix.vix_current"),
    )
    fii_net = _first_value(
        (market.get("fii_dii") or {}).get("fii_net_cash"),
        _signal_value(all_signals, "fii_dii", "fii_net_cash_crores"),
        latest.get("fii_dii.fii_net_cash_crores"),
    )
    iv_rank = _first_value(
        options.get("iv_rank"),
        _signal_value(all_signals, "options", "iv_rank"),
        latest.get("options.iv_rank"),
    )
    atm_iv = _first_positive(
        options.get("atm_iv"),
        _signal_value(all_signals, "options", "atm_iv"),
        latest.get("options.atm_iv"),
    )
    max_pain_distance = None
    if spot is not None and max_pain is not None:
        max_pain_distance = round(max_pain - spot, 2)
    prophet = (prediction.get("component_outputs") or {}).get("m12_prophet") or {}
    tft = (prediction.get("component_outputs") or {}).get("m11_tft") or {}
    return {
        "spot": spot,
        "vol_regime": (prediction.get("vol_regime") or {}).get("prediction"),
        "vix": vix,
        "vix_label": _vix_label(vix),
        "gex": gex,
        "gex_flip": gex_flip,
        "max_pain": max_pain,
        "max_pain_distance": max_pain_distance,
        "pcr": pcr,
        "pcr_5d": pcr_5d,
        "fii_net": fii_net,
        "fii_trend": "positive" if fii_net is not None and fii_net > 0 else "negative" if fii_net is not None else "unknown",
        "ml_direction": prediction.get("composite_direction"),
        "direction": prediction.get("composite_direction"),
        "ml_confidence": round(float(prediction.get("composite_confidence") or 0.0) * 100.0),
        "confidence": round(float(prediction.get("composite_confidence") or 0.0) * 100.0),
        "anomaly_status": "ANOMALY" if ((prediction.get("component_outputs") or {}).get("m05_anomaly") or {}).get("is_anomalous") else "NORMAL",
        "tft_1d": tft.get("return_p50_1d"),
        "prophet_regime": prophet.get("regime"),
        "regime": regime.get("state_label") or prediction.get("hmm_state") or prophet.get("regime"),
        "next_event": prophet.get("next_event_name"),
        "next_event_days": prophet.get("next_event_days"),
        "iv_rank": iv_rank,
        "atm_iv": atm_iv,
        "hurst": None,
        "hurst_regime": "unknown",
    }


def _market_brief_prompt(context: dict[str, Any]) -> str:
    return f"""
You are a senior quant analyst at a top Indian hedge fund.
Generate a concise market brief (150-200 words) for a NIFTY options trader.

Current Data:
- NIFTY Spot: {context.get("spot")}
- Vol Regime: {context.get("vol_regime")} | VIX: {context.get("vix")} ({context.get("vix_label")})
- GEX exposure: {context.get("gex")}Cr
- GEX flip level: ₹{context.get("gex_flip")}
- Max Pain: ₹{context.get("max_pain")} ({context.get("max_pain_distance")} pts from spot)
- PCR: {context.get("pcr")} | 5-day avg: {context.get("pcr_5d")}
- FII Net: ₹{context.get("fii_net")}Cr ({context.get("fii_trend")})
- ML Direction: {context.get("ml_direction")} ({context.get("ml_confidence")}% confidence)
- Anomaly Status: {context.get("anomaly_status")}
- TFT 1-day forecast: {context.get("tft_1d")}
- Prophet regime: {context.get("prophet_regime")}
- Next key event: {context.get("next_event")} in {context.get("next_event_days")} days
- IV Rank: {context.get("iv_rank")} | ATM IV: {context.get("atm_iv")}%
- Hurst exponent: {context.get("hurst")} ({context.get("hurst_regime")})

Rules:
- Be specific about price levels
- Mention the GEX flip level by name
- State the dominant regime clearly
- End with ONE specific observation about what to watch
- Do not use bullet points
- Do not say buy or sell; say consider or watch
- 150-200 words exactly
"""


def _has_enough_context(context: dict[str, Any]) -> bool:
    return context.get("spot") is not None and (
        context.get("gex_flip") is not None or context.get("max_pain") is not None
    )


def _enforce_brief_contract(brief: str, context: dict[str, Any]) -> str:
    words = brief.split()
    has_placeholders = any(
        token in brief
        for token in (
            "nearest GEX flip",
            "nearest max pain",
            "current spot",
            "GEX flip unavailable",
            "max pain unavailable",
            "spot unavailable",
        )
    )
    has_required_levels = _brief_has_level(brief, context.get("gex_flip")) and _brief_has_level(
        brief,
        context.get("max_pain"),
    )
    if 150 <= len(words) <= 200 and "GEX flip" in brief and not has_placeholders and has_required_levels:
        return brief
    return _template_narrative(context)


def _log_narrative(symbol: str, payload: dict[str, Any], context: dict[str, Any]) -> None:
    signals_hash = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()[:16]
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS narratives (
                    id              BIGINT      PRIMARY KEY,
                    generated_at    TIMESTAMP   NOT NULL,
                    symbol          TEXT        NOT NULL,
                    brief           TEXT        NOT NULL,
                    brief_type      TEXT,
                    model           TEXT,
                    signals_hash    TEXT
                )
                """
            )
            row_id = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1_000_000)
            con.execute(
                """
                INSERT INTO narratives
                    (id, generated_at, symbol, brief, brief_type, model, signals_hash)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
                """,
                [row_id, symbol, payload["brief"], payload["brief_type"], payload["model"], signals_hash],
            )
        finally:
            con.close()


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _first_value(*values: Any) -> float | None:
    for value in values:
        number = _num(value)
        if number is not None:
            return number
    return None


def _first_positive(*values: Any) -> float | None:
    for value in values:
        number = _num(value)
        if number is not None and number > 0:
            return number
    return None


def _signal_value(payload: dict[str, Any], category: str, name: str) -> Any:
    if not isinstance(payload, dict):
        return None
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        return None
    signal = (categories.get(category) or {}).get(name)
    if isinstance(signal, dict):
        return signal.get("value")
    return signal


def _latest_signal_values(symbol: str) -> dict[str, float]:
    normalized = symbol.strip().upper()
    try:
        result = resolve_read(ReadOperation.LOAD_LATEST_SIGNAL_VALUES, {"symbol": normalized})
        rows = result["rows"]
    except Exception as error:
        print(f"narrative: latest signal fallback failed: {error!r}")
        return {}
    values: dict[str, float] = {}
    for name, value in rows:
        number = _num(value)
        if number is not None:
            values[str(name)] = number
    return values


def _brief_has_level(brief: str, value: Any) -> bool:
    number = _num(value)
    if number is None:
        return True
    plain = f"{number:.0f}"
    comma = f"{number:,.0f}"
    return plain in brief or comma in brief


def _vix_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 13:
        return "low"
    if value <= 20:
        return "normal"
    return "high"
