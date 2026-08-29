from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import pandas as pd

from db.duckdb_client import open_read_connection
from db.write_service import is_database_owned_by_another_process

from .price_quality import (
    HistoricalPriceObservation,
    HistoricalPriceQuality,
    classify_historical_option_row,
)
from .spot_quality import (
    SpotFallbackPolicy,
    SpotObservation,
    SpotQuality,
    classify_spot_row,
    verified_spot_sources,
)


PRICE_COLUMNS = {"open", "high", "low", "close"}
class HistoricalOptionsPricer:
    """
    Historical option price lookup backed by actual NSE `fo_eod` rows.

    No Black-Scholes modelling is used for backtest entry/exit prices. If an
    expired option has no stored row on the requested exit date, intrinsic
    settlement is derived from the historical spot close for the expiry date.
    """

    def __init__(self, connection: Any | None = None) -> None:
        if connection is None and is_database_owned_by_another_process():
            # Proactive check (writer-queue heartbeat), not exception-driven:
            # this class holds a long-lived connection reused across many ad
            # hoc queries for the lifetime of a backtest run, which has no
            # typed cross-process equivalent (unlike the single bounded
            # ReadOperation reads elsewhere). Production always constructs
            # this with an explicit writer_connection() from inside the
            # owning process's write-dispatch (see backtest/engine.py); this
            # guards the default fallback for any future caller that omits
            # one while a different process holds terminal.db.
            raise PermissionError(
                "HistoricalOptionsPricer requires an explicit connection when a "
                "different process owns DuckDB writer authority; construct it "
                "with writer_connection() from inside the owning process "
                "instead of relying on the default read-only fallback"
            )
        self.con = connection or open_read_connection()
        self._owns_connection = connection is None
        self._price_cache: dict[tuple[Any, ...], HistoricalPriceObservation] = {}
        self._spot_cache: dict[tuple[Any, ...], SpotObservation] = {}
        self.price_observation_events: list[HistoricalPriceObservation] = []
        self.spot_observation_events: list[SpotObservation] = []
        self._lot_cache: dict[str, int] = {}
        self._fo_columns = {
            str(row[0]).lower()
            for row in self.con.execute("DESCRIBE fo_eod").fetchall()
        }

    def close(self) -> None:
        if self._owns_connection:
            self.con.close()

    def get_option_price(
        self,
        symbol: str,
        expiry: _dt.date | str,
        strike: float,
        option_type: str,
        trade_date: _dt.date | str,
        price_type: str = "close",
    ) -> float | None:
        observation = self.get_option_observation(
            symbol,
            expiry,
            strike,
            option_type,
            trade_date,
            price_type,
        )
        return observation.price if observation.executable else None

    def get_option_observation(
        self,
        symbol: str,
        expiry: _dt.date | str,
        strike: float,
        option_type: str,
        trade_date: _dt.date | str,
        price_type: str = "close",
    ) -> HistoricalPriceObservation:
        price_type = price_type.lower().strip()
        if price_type == "settle":
            price_type = "close"
        if price_type not in PRICE_COLUMNS:
            raise ValueError(f"unsupported price_type: {price_type}")

        expiry_day = _parse_date(expiry)
        trade_day = _parse_date(trade_date)
        normalized_type = option_type.strip().upper()
        cache_key = (
            _normalise_symbol(symbol),
            expiry_day,
            round(float(strike), 4),
            normalized_type,
            trade_day,
            price_type,
        )
        cached = self._price_cache.get(cache_key)
        if cached is not None:
            self.price_observation_events.append(cached)
            return cached

        source_expression = "source" if "source" in self._fo_columns else "'LEGACY_UNVERIFIED'"
        quality_expression = "price_quality" if "price_quality" in self._fo_columns else "'UNKNOWN'"
        row = self.con.execute(
            f"""
            SELECT symbol, CAST(expiry AS VARCHAR), strike, option_type,
                   CAST(date AS VARCHAR), open, high, low, close, volume,
                   {source_expression} AS source,
                   {quality_expression} AS price_quality
            FROM fo_eod
            WHERE symbol = ?
              AND expiry = CAST(? AS DATE)
              AND strike = ?
              AND option_type = ?
              AND date = CAST(? AS DATE)
            LIMIT 1
            """,
            [
                _fo_symbol(symbol),
                expiry_day.isoformat(),
                float(strike),
                normalized_type,
                trade_day.isoformat(),
            ],
        ).fetchone()
        if row is None:
            observation = HistoricalPriceObservation(
                None,
                HistoricalPriceQuality.UNKNOWN,
                "NONE",
                trade_day,
                "missing_observation",
            )
        else:
            observation = classify_historical_option_row(
                {
                    "symbol": row[0],
                    "expiry": row[1],
                    "strike": row[2],
                    "option_type": row[3],
                    "date": row[4],
                    "open": row[5],
                    "high": row[6],
                    "low": row[7],
                    "close": row[8],
                    "volume": row[9],
                    "source": row[10],
                    "price_quality": row[11],
                },
                price_type=price_type,
            )
        self._price_cache[cache_key] = observation
        self.price_observation_events.append(observation)
        return observation

    def get_last_option_price(
        self,
        symbol: str,
        expiry: _dt.date | str,
        strike: float,
        option_type: str,
        as_of_date: _dt.date | str,
    ) -> float | None:
        expiry_day = _parse_date(expiry)
        as_of_day = min(_parse_date(as_of_date), expiry_day)
        return self.get_option_price(
            symbol,
            expiry_day,
            strike,
            option_type,
            as_of_day,
        )

    def get_option_row(
        self,
        symbol: str,
        expiry: _dt.date | str,
        strike: float,
        option_type: str,
        trade_date: _dt.date | str,
    ) -> dict[str, Any] | None:
        expiry_day = _parse_date(expiry)
        trade_day = _parse_date(trade_date)
        row = self.con.execute(
            """
            SELECT open, high, low, close, volume, oi, oi_change
            FROM fo_eod
            WHERE symbol = ?
              AND expiry = CAST(? AS DATE)
              AND strike = ?
              AND option_type = ?
              AND date = CAST(? AS DATE)
            LIMIT 1
            """,
            [
                _fo_symbol(symbol),
                expiry_day.isoformat(),
                float(strike),
                option_type.strip().upper(),
                trade_day.isoformat(),
            ],
        ).fetchone()
        if not row:
            return None
        return {
            "open": _finite_or_none(row[0]),
            "high": _finite_or_none(row[1]),
            "low": _finite_or_none(row[2]),
            "close": _finite_or_none(row[3]),
            "volume": int(row[4] or 0),
            "oi": int(row[5] or 0),
            "oi_change": int(row[6] or 0),
        }

    def get_atm_strike(
        self,
        symbol: str,
        spot: float,
        expiry: _dt.date | str,
        trade_date: _dt.date | str,
        strike_interval: float = 50.0,
    ) -> float | None:
        del strike_interval
        return self.nearest_available_strike(symbol, spot, expiry, trade_date)

    def nearest_available_strike(
        self,
        symbol: str,
        target: float,
        expiry: _dt.date | str,
        trade_date: _dt.date | str,
    ) -> float | None:
        rows = self.get_option_chain_snapshot(symbol, trade_date, expiry)
        strikes = {
            float(row["strike"])
            for row in rows
            if row.get("strike") is not None
            and row.get("price_quality") == HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION.value
        }
        return min(strikes, key=lambda value: abs(value - float(target))) if strikes else None

    def get_option_chain_snapshot(
        self,
        symbol: str,
        trade_date: _dt.date | str,
        expiry: _dt.date | str | None = None,
    ) -> list[dict[str, Any]]:
        trade_day = _parse_date(trade_date)
        params: list[Any] = [_fo_symbol(symbol), trade_day.isoformat()]
        expiry_filter = ""
        if expiry is not None:
            expiry_day = _parse_date(expiry)
            expiry_filter = "AND expiry = CAST(? AS DATE)"
            params.append(expiry_day.isoformat())

        source_expression = "source" if "source" in self._fo_columns else "'LEGACY_UNVERIFIED'"
        quality_expression = "price_quality" if "price_quality" in self._fo_columns else "'UNKNOWN'"
        rows = self.con.execute(
            f"""
            SELECT CAST(expiry AS VARCHAR), strike, option_type,
                   open, high, low, close, volume, oi, oi_change,
                   {source_expression} AS source,
                   {quality_expression} AS price_quality
            FROM fo_eod
            WHERE symbol = ?
              AND date = CAST(? AS DATE)
              {expiry_filter}
              AND option_type IN ('CE', 'PE')
            ORDER BY expiry ASC, strike ASC, option_type ASC
            """,
            params,
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            payload = {
                "symbol": _fo_symbol(symbol),
                "expiry": row[0],
                "strike": _finite_or_none(row[1]),
                "option_type": row[2],
                "date": trade_day.isoformat(),
                "open": _finite_or_none(row[3]),
                "high": _finite_or_none(row[4]),
                "low": _finite_or_none(row[5]),
                "close": _finite_or_none(row[6]),
                "volume": int(row[7] or 0),
                "oi": int(row[8] or 0),
                "oi_change": int(row[9] or 0),
                "source": row[10],
                "price_quality": row[11],
            }
            observation = classify_historical_option_row(payload)
            payload["price_quality"] = observation.quality.value
            payload["price_quality_reason"] = observation.reason
            if observation.executable:
                output.append(payload)
        return output

    def compute_historical_iv(
        self,
        option_price: float,
        spot: float,
        strike: float,
        dte: int,
        option_type: str,
        risk_free: float = 0.065,
    ) -> float | None:
        try:
            from py_vollib.black_scholes.implied_volatility import implied_volatility

            flag = "c" if option_type.strip().upper() == "CE" else "p"
            iv = implied_volatility(
                float(option_price),
                float(spot),
                float(strike),
                max(int(dte), 1) / 365.0,
                risk_free,
                flag,
            )
        except Exception:
            return None
        return float(iv) if math.isfinite(iv) and 0.01 <= iv <= 5.0 else None

    def find_nearest_expiry(
        self,
        symbol: str,
        trade_date: _dt.date | str,
        min_dte: int = 1,
    ) -> _dt.date | None:
        trade_day = _parse_date(trade_date)
        minimum = trade_day + _dt.timedelta(days=max(0, int(min_dte)))
        rows = self.con.execute(
            """
            SELECT DISTINCT expiry
            FROM fo_eod
            WHERE symbol = ?
              AND date = CAST(? AS DATE)
              AND expiry >= CAST(? AS DATE)
              AND option_type IN ('CE', 'PE')
            ORDER BY expiry
            LIMIT 16
            """,
            [_fo_symbol(symbol), trade_day.isoformat(), minimum.isoformat()],
        ).fetchall()
        for row in rows:
            expiry_day = _date_or_none(row[0])
            if expiry_day and self.get_option_chain_snapshot(symbol, trade_day, expiry_day):
                return expiry_day
        return None

    def get_strategy_pnl(
        self,
        legs: list[Any],
        entry_date: _dt.date | str,
        exit_date: _dt.date | str,
    ) -> dict[str, Any]:
        gross = 0.0
        leg_rows: list[dict[str, Any]] = []
        for leg in legs:
            entry = self.get_option_price(
                leg.symbol,
                leg.expiry,
                leg.strike,
                leg.option_type,
                entry_date,
            )
            exit_price = self.get_option_price(
                leg.symbol,
                leg.expiry,
                leg.strike,
                leg.option_type,
                exit_date,
            )
            if entry is None or exit_price is None:
                raise MissingExecutablePriceError(
                    f"missing executable price for {leg.symbol} {leg.option_type} "
                    f"{leg.strike:g} {leg.expiry}"
                )
            sign = 1.0 if leg.action == "BUY" else -1.0
            pnl = (exit_price - entry) * sign * leg.lots * leg.lot_size
            gross += pnl
            leg_rows.append(
                {
                    "symbol": leg.symbol,
                    "expiry": _parse_date(leg.expiry).isoformat(),
                    "strike": leg.strike,
                    "option_type": leg.option_type,
                    "action": leg.action,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "gross_pnl": pnl,
                }
            )
        return {"gross_pnl": round(gross, 2), "pnl_by_leg": leg_rows}

    def get_expiry_settlement(
        self,
        symbol: str,
        expiry: _dt.date | str,
        strike: float,
        option_type: str,
    ) -> float | None:
        expiry_day = _parse_date(expiry)
        observation = self.get_option_observation(
            symbol,
            expiry_day,
            strike,
            option_type,
            expiry_day,
            "close",
        )
        return observation.price if observation.executable else None

    def get_spot(
        self,
        symbol: str,
        as_of_date: _dt.date | str,
        *,
        fallback_policy: SpotFallbackPolicy = SpotFallbackPolicy.NO_FALLBACK,
        maximum_staleness_sessions: int = 0,
    ) -> float | None:
        observation = self.get_spot_observation(
            symbol,
            as_of_date,
            fallback_policy=fallback_policy,
            maximum_staleness_sessions=maximum_staleness_sessions,
        )
        return observation.price if observation.valid else None

    def get_spot_observation(
        self,
        symbol: str,
        as_of_date: _dt.date | str,
        *,
        fallback_policy: SpotFallbackPolicy = SpotFallbackPolicy.NO_FALLBACK,
        maximum_staleness_sessions: int = 0,
    ) -> SpotObservation:
        as_of_day = _parse_date(as_of_date)
        normalized = _normalise_symbol(symbol)
        cache_key = (
            normalized,
            as_of_day,
            SpotFallbackPolicy(fallback_policy).value,
            max(int(maximum_staleness_sessions), 0),
        )
        if cache_key in self._spot_cache:
            observation = self._spot_cache[cache_key]
            self.spot_observation_events.append(observation)
            return observation

        candidates = _ohlcv_symbols(symbol)
        placeholders = ", ".join("?" for _ in candidates)
        rows = self.con.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), close, source
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND date <= CAST(? AS DATE)
            ORDER BY date DESC
            LIMIT {max(len(candidates) * 4, 8)}
            """,
            [*candidates, as_of_day.isoformat()],
        ).fetchall()
        candidate_priority = {candidate.upper(): index for index, candidate in enumerate(candidates)}
        rows.sort(key=lambda row: (-_parse_date(row[1]).toordinal(), candidate_priority.get(str(row[0]).upper(), 999)))
        selected = rows[0] if rows else None
        payload = None if selected is None else {
            "symbol": selected[0],
            "date": selected[1],
            "close": selected[2],
            "source": selected[3],
        }
        observation = classify_spot_row(
            payload,
            symbol=normalized,
            requested_as_of=as_of_day,
            fallback_policy=fallback_policy,
            maximum_staleness_sessions=maximum_staleness_sessions,
        )
        self._spot_cache[cache_key] = observation
        self.spot_observation_events.append(observation)
        return observation

    def get_ohlcv_window(
        self,
        symbol: str,
        as_of_date: _dt.date | str,
        lookback: int = 260,
    ) -> pd.DataFrame:
        as_of_day = _parse_date(as_of_date)
        candidates = _ohlcv_symbols(symbol)
        sources = sorted(verified_spot_sources(symbol))
        placeholders = ", ".join("?" for _ in candidates)
        source_placeholders = ", ".join("?" for _ in sources)
        rows = self.con.execute(
            f"""
            SELECT symbol, CAST(date AS VARCHAR), open, high, low, close, volume
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND UPPER(COALESCE(source, '')) IN ({source_placeholders})
              AND date <= CAST(? AS DATE)
            ORDER BY date DESC
            LIMIT {int(max(1, lookback))}
            """,
            [*candidates, *sources, as_of_day.isoformat()],
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        rows = list(reversed(rows))
        return pd.DataFrame(
            [
                {
                    "symbol": row[0],
                    "date": row[1],
                    "open": _finite_or_none(row[2]),
                    "high": _finite_or_none(row[3]),
                    "low": _finite_or_none(row[4]),
                    "close": _finite_or_none(row[5]),
                    "volume": int(row[6] or 0),
                }
                for row in rows
            ]
        )

    def get_trading_days(
        self,
        symbol: str,
        from_date: _dt.date | str,
        to_date: _dt.date | str,
    ) -> list[_dt.date]:
        from_day = _parse_date(from_date)
        to_day = _parse_date(to_date)
        candidates = _ohlcv_symbols(symbol)
        sources = sorted(verified_spot_sources(symbol))
        placeholders = ", ".join("?" for _ in candidates)
        source_placeholders = ", ".join("?" for _ in sources)
        rows = self.con.execute(
            f"""
            SELECT DISTINCT date
            FROM decision_eligible_ohlcv
            WHERE symbol IN ({placeholders})
              AND UPPER(COALESCE(source, '')) IN ({source_placeholders})
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY date ASC
            """,
            [*candidates, *sources, from_day.isoformat(), to_day.isoformat()],
        ).fetchall()
        return [_parse_date(row[0]) for row in rows]

    def get_lot_size(self, symbol: str) -> int:
        normalized = _normalise_symbol(symbol)
        if normalized in self._lot_cache:
            return self._lot_cache[normalized]
        row = self.con.execute(
            """
            SELECT lot_size
            FROM instruments
            WHERE (symbol = ? OR name = ?)
              AND lot_size IS NOT NULL
              AND lot_size > 0
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            [_fo_symbol(symbol), _fo_symbol(symbol)],
        ).fetchone()
        if not row or not row[0] or int(row[0]) <= 0:
            raise ValueError(f"lot size unavailable for {normalized}")
        lot_size = int(row[0])
        self._lot_cache[normalized] = lot_size
        return self._lot_cache[normalized]


def _fo_symbol(symbol: str) -> str:
    normalized = _normalise_symbol(symbol)
    if normalized == "NIFTY 50":
        return "NIFTY"
    if normalized == "NIFTY BANK":
        return "BANKNIFTY"
    return normalized


def _ohlcv_symbols(symbol: str) -> list[str]:
    normalized = _normalise_symbol(symbol)
    if normalized in {"NIFTY", "NIFTY 50"}:
        return ["NIFTY 50", "NIFTY"]
    if normalized in {"BANKNIFTY", "NIFTY BANK"}:
        return ["NIFTY BANK", "BANKNIFTY"]
    return [normalized]


def _normalise_symbol(symbol: str) -> str:
    return str(symbol or "NIFTY").strip().upper() or "NIFTY"


def _parse_date(value: _dt.date | str) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return _dt.date.fromisoformat(text)


def _date_or_none(value: Any) -> _dt.date | None:
    if value is None:
        return None
    try:
        return _parse_date(value)
    except (TypeError, ValueError):
        return None


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class MissingExecutablePriceError(RuntimeError):
    pass
