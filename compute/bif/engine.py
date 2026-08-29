from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import time
from threading import Lock
from typing import Any, Callable

from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read
from data.exchange_calendar import next_expiry_labels, today_ist
from data.market_data_quality import (
    DecisionContext,
    MarketDataEligibility,
    evaluate_market_bar,
)
from backtest.spot_quality import (
    SpotObservation,
    classify_live_spot,
    unavailable_spot,
)
from ml.recommendation_policy import RecommendationEvidence

from .accumulation import compute_iad
from .composite import compute_cds
from .fair_value import compute_nfve
from .gravity import compute_ngi
from .momentum import compute_bms
from .vol_clock import compute_vrc


Provider = Callable[..., Any]
BIF_CACHE_TTL = 60


class BIFEngine:
    """
    Orchestrates all six BIF modules.

    Reads existing systems only. Writes only bif_history.
    """

    def __init__(self, providers: dict[str, Provider] | None = None) -> None:
        self.providers = providers or {}
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = Lock()

    def compute_all(self, symbol: str, *, persist: bool = False) -> dict[str, Any]:
        normalized = _normalise_symbol(symbol)
        now = time.time()
        with self._cache_lock:
            cached = self._cache.get(normalized)
            if cached and now - float(cached["timestamp"]) < BIF_CACHE_TTL:
                cached_payload = copy.deepcopy(cached["data"])
                if persist:
                    self._write_history(normalized, cached_payload)
                return cached_payload

        context = self._build_context(normalized)
        bms = compute_bms(
            context["ohlcv_rows"],
            context["option_signals"],
            context["market_signals"],
            context["expiry"],
            context["ml_prediction"],
            context["flow_history"],
            context["signal_values"],
        )
        ngi = compute_ngi(context["chain_legs"], context["spot"], context["expiry"])
        iad = compute_iad(
            context["ohlcv_rows"],
            context["chain_legs"],
            context["flow_history"],
            context["market_signals"],
        )
        vrc = compute_vrc(
            context["iv_history"],
            context["option_signals"],
            context["ohlcv_rows"],
            context["signal_values"],
        )
        nfve = compute_nfve(
            normalized,
            context["spot"],
            context["ohlcv_rows"],
            context["cross_asset"],
            context["market_signals"],
            context["flow_history"],
            context["signal_values"],
        )
        cds = compute_cds(
            bms,
            ngi,
            iad,
            vrc,
            nfve,
            context["ml_prediction"],
            decision_evidence=context["decision_evidence"],
        )

        payload = {
            "symbol": normalized,
            "spot": round(context["spot"], 3) if context["spot"] > 0 else None,
            "spot_provenance": context["spot_observation"].as_provenance(),
            "market_data_eligibility": context["market_data_eligibility"].to_dict(),
            "expiry": context["expiry"],
            "computed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "as_of_date": context["ohlcv_rows"][-1]["date"] if context["ohlcv_rows"] else None,
            "bms": bms,
            "ngi": ngi,
            "iad": iad,
            "vrc": vrc,
            "nfve": nfve,
            "cds": cds,
            "history": self.get_history(normalized, days=30),
            "input_status": {
                "ohlcv_rows": len(context["ohlcv_rows"]),
                "chain_legs": len(context["chain_legs"]),
                "iv_history_rows": len(context["iv_history"]),
                "signals_available": bool(context["signal_values"]),
                "ml_available": bool(context["ml_prediction"]),
            },
        }
        if persist:
            self._write_history(normalized, payload)
            payload["history"] = self.get_history(normalized, days=30)

        with self._cache_lock:
            self._cache[normalized] = {"timestamp": now, "data": copy.deepcopy(payload)}
        return payload

    def get_cds_only(self, symbol: str, *, persist: bool = False) -> dict[str, Any]:
        full = self.compute_all(symbol, persist=persist)
        cds = full.get("cds", {})
        return {
            "symbol": full.get("symbol"),
            "spot": full.get("spot"),
            "computed_at": full.get("computed_at"),
            "cds": cds.get("cds"),
            "cds_label": cds.get("cds_label"),
            "cds_confidence": cds.get("cds_confidence"),
            "action": cds.get("action"),
            "action_strength": cds.get("action_strength"),
            "recommendation_state": cds.get("recommendation_state"),
            "recommendation_eligibility": cds.get("recommendation_eligibility"),
            "source_validation_state": cds.get("source_validation_state"),
            "model_validation_state": cds.get("model_validation_state"),
            "data_quality_state": cds.get("data_quality_state"),
            "anomaly_gate_active": cds.get("anomaly_gate_active"),
            "is_valid": cds.get("is_valid"),
        }

    def get_history(self, symbol: str, days: int = 30) -> list[dict[str, Any]]:
        normalized = _normalise_symbol(symbol)
        cutoff = today_ist() - _dt.timedelta(days=max(1, int(days)))
        result = resolve_read(
            ReadOperation.LOAD_BIF_HISTORY,
            {"symbol": normalized, "cutoff_date": cutoff.isoformat()},
        )
        rows = result["rows"]

        return [
            {
                "date": row[0],
                "symbol": row[1],
                "bms": _json_number(row[2]),
                "bms_label": row[3],
                "ngi": _json_number(row[4]),
                "ngi_direction": row[5],
                "gravity_center": _json_number(row[6]),
                "iad": _json_number(row[7]),
                "iad_label": row[8],
                "wyckoff_stage": row[9],
                "vrc_position": _json_number(row[10]),
                "vrc_label": row[11],
                "vrp": _json_number(row[12]),
                "nfve_fair": _json_number(row[13]),
                "nfve_diff": _json_number(row[14]),
                "nfve_valuation": row[15],
                "cds": _json_number(row[16]),
                "cds_label": row[17],
                "cds_confidence": _json_number(row[18]),
                "action": row[19],
                "anomaly_active": bool(row[20]),
                "computed_at": row[21],
            }
            for row in rows
        ]

    def _build_context(self, symbol: str) -> dict[str, Any]:
        requested_at = _dt.datetime.now(_dt.timezone.utc)
        ohlcv_rows, market_data_eligibility = _load_current_ohlcv_rows(symbol, requested_at)
        iv_history = _load_iv_history(symbol)
        signal_history = _load_signal_history(symbol)
        signal_values = _latest_signal_values(signal_history)
        signals_payload = self._provider("signals", symbol) or {}
        signal_values.update(_flatten_signal_payload(signals_payload))

        chain_legs, expiry = self._chain(symbol)
        spot_observation = self._spot(symbol, requested_at)
        spot = (
            float(spot_observation.price)
            if spot_observation.decision_eligible and spot_observation.price is not None
            else 0.0
        )

        option_signals = _option_signals(symbol, chain_legs, spot, expiry)
        market_signals = self._provider("market") or _fallback_market_signals()
        cross_asset = self._provider("cross_asset") or {}
        flow_history = _flow_history(signal_history, market_signals)
        ml_prediction = self._ml_prediction(symbol, signals_payload)
        decision_evidence = _bif_decision_evidence(
            ml_prediction,
            market_data_eligibility,
            spot_observation,
            requested_at,
        )

        return {
            "symbol": symbol,
            "ohlcv_rows": ohlcv_rows,
            "iv_history": iv_history,
            "signal_values": signal_values,
            "signals_payload": signals_payload,
            "chain_legs": chain_legs,
            "expiry": expiry,
            "spot": spot,
            "spot_observation": spot_observation,
            "market_data_eligibility": market_data_eligibility,
            "decision_evidence": decision_evidence,
            "option_signals": option_signals,
            "market_signals": market_signals,
            "cross_asset": cross_asset,
            "flow_history": flow_history,
            "ml_prediction": ml_prediction,
        }

    def _chain(self, symbol: str) -> tuple[list[dict[str, Any]], str]:
        provided = self._provider("chain", symbol)
        if isinstance(provided, tuple) and len(provided) >= 2:
            return _as_leg_list(provided[0]), str(provided[1] or _next_expiries(1)[0])
        if isinstance(provided, dict):
            return (
                _as_leg_list(provided.get("legs") or provided.get("chain_legs")),
                str(provided.get("expiry") or _next_expiries(1)[0]),
            )
        if isinstance(provided, list):
            return _as_leg_list(provided), _next_expiries(1)[0]
        return _read_chain_from_redis(symbol)

    def _spot(self, symbol: str, requested_at: _dt.datetime) -> SpotObservation:
        provided = self._provider("spot", symbol)
        if "spot" in self.providers:
            if isinstance(provided, SpotObservation):
                return provided
            return unavailable_spot(symbol, requested_at, "spot_provider_did_not_return_provenance")
        return _read_spot_from_redis(symbol, requested_at)

    def _ml_prediction(self, symbol: str, signals_payload: dict[str, Any]) -> dict[str, Any]:
        fn = self.providers.get("ml")
        if fn is not None:
            try:
                payload = fn(symbol, signals_payload)
            except TypeError:
                payload = fn(symbol)
            except Exception as error:
                print(f"bif: ML provider failed: {error}")
                payload = {}
            return payload if isinstance(payload, dict) else {}

        try:
            from ml.service import predict_all_models

            payload = predict_all_models(symbol, signals_payload, log=False)
            return payload if isinstance(payload, dict) else {}
        except Exception as error:
            print(f"bif: ML fallback failed: {error}")
            return {}

    def _provider(self, name: str, *args: Any) -> Any:
        fn = self.providers.get(name)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception as error:
            print(f"bif: provider {name} failed: {error}")
            return None

    def _write_history(self, symbol: str, payload: dict[str, Any]) -> None:
        if not payload.get("as_of_date"):
            raise ValueError("persistent BIF computation requires a market-data as-of date")
        bms = payload.get("bms", {})
        ngi = payload.get("ngi", {})
        iad = payload.get("iad", {})
        vrc = payload.get("vrc", {})
        nfve = payload.get("nfve", {})
        cds = payload.get("cds", {})
        row = [
            payload.get("as_of_date"),
            symbol,
            _finite_or_none(bms.get("bms")),
            bms.get("bms_label"),
            _finite_or_none(ngi.get("ngi")),
            ngi.get("ngi_direction"),
            _finite_or_none(ngi.get("gravity_center")),
            _finite_or_none(iad.get("iad_score")),
            iad.get("iad_label"),
            iad.get("wyckoff_stage"),
            _finite_or_none(vrc.get("clock_position")),
            vrc.get("clock_label"),
            _finite_or_none(vrc.get("vrp")),
            _finite_or_none(nfve.get("fair_value")),
            _finite_or_none(nfve.get("current_vs_fair")),
            nfve.get("valuation"),
            _finite_or_none(cds.get("cds")),
            cds.get("cds_label"),
            _finite_or_none(cds.get("cds_confidence")),
            cds.get("action"),
            int(bool(cds.get("anomaly_gate_active"))),
            payload.get("computed_at"),
        ]
        with DB_WRITE_LOCK:
            con = writer_connection()
            try:
                _ensure_bif_schema(con)
                con.execute("BEGIN TRANSACTION")
                con.execute(
                    """
                    INSERT OR REPLACE INTO bif_history
                    (
                        date, symbol, bms, bms_label, ngi, ngi_direction,
                        gravity_center, iad, iad_label, wyckoff_stage,
                        vrc_position, vrc_label, vrp, nfve_fair, nfve_diff,
                        nfve_valuation, cds, cds_label, cds_confidence,
                        action, anomaly_active, computed_at
                    )
                    VALUES (
                        CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMP)
                    )
                    """,
                    row,
                )
                con.execute("COMMIT")
            except Exception as error:
                con.execute("ROLLBACK")
                print(f"bif: history write failed: {error}")
            finally:
                con.close()


def _load_ohlcv_rows(symbol: str) -> list[dict[str, Any]]:
    symbols = _symbol_candidates(symbol)
    from_day = today_ist() - _dt.timedelta(days=430)
    result = resolve_read(
        ReadOperation.LOAD_DECISION_ELIGIBLE_OHLCV,
        {"symbols": symbols, "from_date": from_day.isoformat()},
    )
    rows = result["rows"]

    by_date: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_symbol = str(row[0])
        date = str(row[1])
        if date in by_date and row_symbol != symbols[0]:
            continue
        by_date[date] = {
            "symbol": row_symbol,
            "date": date,
            "observed_at": row[2],
            "open": _json_number(row[3]),
            "high": _json_number(row[4]),
            "low": _json_number(row[5]),
            "close": _json_number(row[6]),
            "volume": _json_number(row[7]),
            "source": str(row[8]),
            "quality": str(row[9]),
            "derivation_type": str(row[10]),
        }
    return [by_date[date] for date in sorted(by_date)]


def _load_current_ohlcv_rows(
    symbol: str,
    requested_at: _dt.datetime,
) -> tuple[list[dict[str, Any]], MarketDataEligibility]:
    rows = _load_ohlcv_rows(symbol)
    eligibility = evaluate_market_bar(
        rows[-1] if rows else None,
        context=DecisionContext.CURRENT_DECISION,
        requested_as_of=requested_at,
    )
    return rows, eligibility


def _load_iv_history(symbol: str) -> list[dict[str, Any]]:
    symbols = _symbol_candidates(symbol)
    from_day = today_ist() - _dt.timedelta(days=370)
    result = resolve_read(
        ReadOperation.LOAD_IV_HISTORY_AVERAGE,
        {"symbols": symbols, "from_date": from_day.isoformat()},
    )
    rows = result["rows"]
    return [
        {"date": str(row[0]), "atm_iv": _json_number(row[1])}
        for row in rows
        if _finite_or_none(row[1]) is not None
    ]


def _load_signal_history(symbol: str) -> dict[str, list[float]]:
    symbols = _symbol_candidates(symbol)
    from_day = today_ist() - _dt.timedelta(days=370)
    result = resolve_read(
        ReadOperation.LOAD_SIGNAL_HISTORY_VALUES,
        {"symbols": symbols, "from_date": from_day.isoformat()},
    )
    rows = result["rows"]

    history: dict[str, list[float]] = {}
    for name, value in rows:
        number = _finite_or_none(value)
        if number is None:
            continue
        history.setdefault(str(name), []).append(number)
    return history


def _read_chain_from_redis(symbol: str) -> tuple[list[dict[str, Any]], str]:
    expiries = _next_expiries(4)
    redis_symbol = _redis_symbol(symbol)
    try:
        import redis

        from security.redis_config import authenticated_redis_client

        client = authenticated_redis_client()
        keys = [f"chain:{redis_symbol}:{expiry}" for expiry in expiries]
        try:
            for raw_key in client.scan_iter(match=f"chain:{redis_symbol}:*"):
                key = _redis_text(raw_key)
                if key not in keys:
                    keys.append(key)
        except redis.RedisError as error:
            print(f"bif: Redis chain scan failed: {error}")

        for key in keys:
            payload = client.get(key)
            if payload is None:
                continue
            legs = _decode_chain_payload(payload)
            if legs:
                expiry = key.split(":", 2)[2] if ":" in key else expiries[0]
                return legs, expiry
    except Exception as error:
        print(f"bif: Redis chain unavailable: {error}")
    return [], expiries[0]


def _read_spot_from_redis(symbol: str, requested_at: _dt.datetime) -> SpotObservation:
    try:
        from live.market_state import evaluate_market_state, parse_quote_envelope
        from security.redis_config import authenticated_redis_client

        client = authenticated_redis_client()
        payload = client.get(f"tick:{_redis_symbol(symbol)}")
        if payload is None:
            return unavailable_spot(symbol, requested_at, "live_spot_missing_from_redis")
        tick = json.loads(_redis_text(payload))
        if not isinstance(tick, dict):
            return unavailable_spot(symbol, requested_at, "live_spot_envelope_invalid")
        quote = parse_quote_envelope(tick)
        if quote.symbol != _redis_symbol(symbol):
            return unavailable_spot(symbol, requested_at, "live_spot_instrument_mismatch")
        evaluated = evaluate_market_state(
            quote,
            now_ms=int(requested_at.timestamp() * 1000),
            expected_feed_live=True,
        )
        return classify_live_spot(
            evaluated.to_dict(),
            symbol=_redis_symbol(symbol),
            requested_at=requested_at,
        )
    except Exception as error:
        return unavailable_spot(
            symbol,
            requested_at,
            f"live_spot_unavailable:{type(error).__name__}",
        )


def _bif_decision_evidence(
    ml_prediction: dict[str, Any],
    market: MarketDataEligibility,
    spot: SpotObservation,
    requested_at: _dt.datetime,
) -> RecommendationEvidence:
    embedded = RecommendationEvidence.from_mapping(
        ml_prediction.get("decision_input_evidence")
        if isinstance(ml_prediction.get("decision_input_evidence"), dict)
        else None
    )
    market_and_spot_valid = market.decision_eligible and spot.decision_eligible
    provenance = None
    if market.source and spot.source and market.source != "NONE" and spot.source != "NONE":
        provenance = f"{market.source}+{spot.source}"
    fallback_state = embedded.fallback_state
    if spot.fallback_used:
        fallback_state = "SPOT_FALLBACK"
    return RecommendationEvidence(
        market_data_quality=(
            "OBSERVED_VERIFIED" if market_and_spot_valid else "INVALID_UNVERIFIED"
        ),
        market_data_freshness=(
            "CURRENT_SESSION_ELIGIBLE" if market_and_spot_valid else "STALE"
        ),
        market_data_provenance=provenance,
        prediction_as_of=embedded.prediction_as_of,
        requested_decision_at=requested_at.isoformat(),
        fallback_state=fallback_state,
        input_feature_schema=embedded.input_feature_schema,
    )


def _option_signals(
    symbol: str,
    chain_legs: list[dict[str, Any]],
    spot: float,
    expiry: str,
) -> dict[str, Any]:
    try:
        from data.instrument_master import master
        from signals.options_signals import compute_options_signals

        payload = compute_options_signals(
            chain_legs,
            spot,
            expiry,
            master.get_lot_size(symbol, expiry),
        )
        return payload if isinstance(payload, dict) else {}
    except Exception as error:
        print(f"bif: options signal read failed: {error}")
        return {}


def _fallback_market_signals() -> dict[str, Any]:
    try:
        from signals.market_signals import fetch_fii_dii, fetch_india_vix

        return {"india_vix": fetch_india_vix(), "fii_dii": fetch_fii_dii()}
    except Exception as error:
        print(f"bif: market signal fallback failed: {error}")
        return {"india_vix": None, "fii_dii": {}}


def _flow_history(
    signal_history: dict[str, list[float]],
    market_signals: dict[str, Any],
) -> dict[str, list[float]]:
    output = {
        "fii_net_cash": _history_for(signal_history, "fii_net_cash"),
        "dii_net_cash": _history_for(signal_history, "dii_net_cash"),
    }
    fii_dii = market_signals.get("fii_dii")
    if isinstance(fii_dii, dict):
        for key in ("fii_net_cash", "dii_net_cash"):
            value = _finite_or_none(fii_dii.get(key))
            if value is not None:
                output.setdefault(key, []).append(value)
    return output


def _history_for(signal_history: dict[str, list[float]], token: str) -> list[float]:
    result: list[float] = []
    for name, values in signal_history.items():
        if token in name:
            result.extend(values)
    return result


def _latest_signal_values(signal_history: dict[str, list[float]]) -> dict[str, float]:
    return {
        name: values[-1]
        for name, values in signal_history.items()
        if values
    }


def _flatten_signal_payload(payload: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    categories = payload.get("categories")
    if isinstance(categories, dict):
        for category, signals in categories.items():
            if not isinstance(signals, dict):
                continue
            for name, signal in signals.items():
                raw = signal.get("value") if isinstance(signal, dict) else signal
                number = _finite_or_none(raw)
                if number is None:
                    continue
                values[f"{category}.{name}"] = number
                values.setdefault(str(name), number)

    for section in ("options", "composite_scores"):
        data = payload.get(section)
        if not isinstance(data, dict):
            continue
        for name, raw in data.items():
            number = _finite_or_none(raw)
            if number is None:
                continue
            values[f"{section}.{name}"] = number
            values.setdefault(str(name), number)

    market = payload.get("market")
    if isinstance(market, dict):
        vix = _finite_or_none(market.get("india_vix"))
        if vix is not None:
            values["india_vix"] = vix
            values["vix.india_vix"] = vix
        fii_dii = market.get("fii_dii")
        if isinstance(fii_dii, dict):
            for name, raw in fii_dii.items():
                number = _finite_or_none(raw)
                if number is not None:
                    values[f"fii_dii.{name}"] = number
                    values.setdefault(str(name), number)
    return values


def _ensure_bif_schema(con: Any) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bif_history (
            date            DATE        NOT NULL,
            symbol          TEXT        NOT NULL,
            bms             DOUBLE,
            bms_label       TEXT,
            ngi             DOUBLE,
            ngi_direction   TEXT,
            gravity_center  DOUBLE,
            iad             DOUBLE,
            iad_label       TEXT,
            wyckoff_stage   TEXT,
            vrc_position    DOUBLE,
            vrc_label       TEXT,
            vrp             DOUBLE,
            nfve_fair       DOUBLE,
            nfve_diff       DOUBLE,
            nfve_valuation  TEXT,
            cds             DOUBLE,
            cds_label       TEXT,
            cds_confidence  DOUBLE,
            action          TEXT,
            anomaly_active  INTEGER     DEFAULT 0,
            computed_at     TIMESTAMP   NOT NULL,
            PRIMARY KEY (date, symbol)
        )
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bif_sym_date
        ON bif_history(symbol, date)
        """
    )


def _decode_chain_payload(payload: Any) -> list[dict[str, Any]]:
    try:
        data = json.loads(_redis_text(payload))
    except (TypeError, ValueError, UnicodeDecodeError):
        return []
    if isinstance(data, list):
        return _as_leg_list(data)
    if isinstance(data, dict):
        nested = data.get("legs") or data.get("data")
        if isinstance(nested, list):
            return _as_leg_list(nested)
    return []


def _as_leg_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _next_expiries(n: int) -> list[str]:
    return next_expiry_labels(n)


def _symbol_candidates(symbol: str) -> list[str]:
    upper = _normalise_symbol(symbol)
    if upper == "NIFTY":
        return ["NIFTY 50", "NIFTY"]
    if upper == "BANKNIFTY":
        return ["NIFTY BANK", "BANKNIFTY"]
    return [upper]


def _redis_symbol(symbol: str) -> str:
    upper = _normalise_symbol(symbol)
    if upper in {"NIFTY", "NIFTY 50"}:
        return "NIFTY"
    return upper


def _normalise_symbol(symbol: str) -> str:
    return str(symbol or "NIFTY").strip().upper() or "NIFTY"


def _redis_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_number(value: Any) -> float | None:
    number = _finite_or_none(value)
    if number is None:
        return None
    return float(number)


def _finite_or_zero(value: Any) -> float:
    number = _finite_or_none(value)
    return number if number is not None else 0.0


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
