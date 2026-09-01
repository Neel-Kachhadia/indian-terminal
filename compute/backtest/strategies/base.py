from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np

from ..position import PositionLeg


class BaseStrategy:
    def __init__(self, key: str, definition: dict[str, Any]) -> None:
        self.key = key
        self.definition = definition
        self.name = str(definition.get("name") or key)
        self.description = str(definition.get("description") or "")
        self._last_entry_date: _dt.date | None = None

    @property
    def exit_logic(self) -> dict[str, Any]:
        exit_logic = self.definition.get("exit") or {}
        if isinstance(exit_logic, str):
            return {"type": exit_logic}
        return dict(exit_logic)

    def generate_signals(
        self,
        day: _dt.date,
        signal_date: _dt.date,
        guard: Any,
        pricer: Any,
    ) -> dict[str, Any]:
        ohlcv = guard.get_ohlcv(self.symbol, signal_date, lookback=260)
        spot_observation = pricer.get_spot_observation(self.symbol, signal_date)
        spot = spot_observation.price if spot_observation.valid else None
        expiry = pricer.find_nearest_expiry(
            self.symbol,
            signal_date,
            int(self.definition.get("min_dte", 1)),
        )
        dte = max((expiry - day).days, 0) if expiry else 0
        chain = guard.get_chain_snapshot(self.symbol, signal_date, expiry)

        signals = {
            "date": day.isoformat(),
            "signal_date": signal_date.isoformat(),
            "spot": spot,
            "spot_source": spot_observation.source,
            "spot_quality": spot_observation.quality.value,
            "spot_observed_on": (
                spot_observation.observed_on.isoformat() if spot_observation.observed_on else None
            ),
            "spot_provenance": spot_observation.as_provenance(),
            "expiry": expiry.isoformat() if expiry else None,
            "expiry.dte_weekly": dte,
            "regime.volatility": _vol_regime(ohlcv),
            "price_action.change_pct_today": _change_pct(ohlcv, 1),
            "price_action.return_3d": _change_pct(ohlcv, 3),
            "price_action.return_5d": _change_pct(ohlcv, 5),
            "price_action.return_10d": _change_pct(ohlcv, 10),
            "price_action.return_20d": _change_pct(ohlcv, 20),
            "price_action.pattern_20d_high_breakout": _breakout_20d(ohlcv),
            "price_action.bb_squeeze": _bb_squeeze(ohlcv),
            "price_action.volume_ratio_20d": _volume_ratio(ohlcv),
            "volume.volume_spike_2x": 1.0 if _volume_ratio(ohlcv) >= 2.0 else 0.0,
            "options.pcr_total": _pcr(chain),
            "options.iv_rank": _iv_rank_proxy(chain, float(spot or 0.0)),
            "options.iv_percentile": _iv_rank_proxy(chain, float(spot or 0.0)),
            "options.atm_iv": _atm_iv_proxy(chain, float(spot or 0.0)),
            "options.vol_risk_premium": 0.0,
            "options.gex": _gex_proxy(chain, float(spot or 0.0)),
            "options.max_pain": _max_pain(chain),
            "bif.cds": guard.get_bif_value("cds", self.symbol, signal_date),
            "bif.bms": guard.get_bif_value("bms", self.symbol, signal_date),
            "bif.iad": guard.get_bif_value("iad", self.symbol, signal_date),
            "bif.ngi": guard.get_bif_value("ngi", self.symbol, signal_date),
            "bif.ngi_pin_probability": guard.get_bif_value("ngi_pin_probability", self.symbol, signal_date) or 0.0,
            "bif.vrc_label": guard.get_bif_label("vrc_label", self.symbol, signal_date),
            "bif.wyckoff_stage": guard.get_bif_label("wyckoff_stage", self.symbol, signal_date),
            "bif.nfve_current_vs_fair": guard.get_bif_value("nfve_diff", self.symbol, signal_date),
            "ml.model_agreement": guard.get_signal("ml.model_agreement", self.symbol, signal_date),
            "ml.composite_probabilities.UP": guard.get_signal("ml.composite_probabilities.UP", self.symbol, signal_date),
            "ml.anomaly_score": guard.get_signal("ml.anomaly_score", self.symbol, signal_date) or 0.0,
            "vix.vix_spike_15pct": guard.get_signal("vix.vix_spike_15pct", self.symbol, signal_date) or 0.0,
            "macro.results_season_active": guard.get_signal("macro.results_season_active", self.symbol, signal_date) or 0.0,
            "expiry.rbi_policy_dte": guard.get_signal("expiry.rbi_policy_dte", self.symbol, signal_date) or 999.0,
        }
        return signals

    def should_enter(
        self,
        signals: dict[str, Any],
        portfolio: Any,
        day: _dt.date,
    ) -> bool:
        max_positions = int(self.definition.get("max_open_positions", 1))
        if len(portfolio.open_positions) >= max_positions:
            return False
        if not self._cadence_ok(day, signals):
            return False
        for condition in self.definition.get("entry_conditions", []):
            if not _condition_ok(condition, signals):
                return False
        return True

    def build_legs(
        self,
        signals: dict[str, Any],
        day: _dt.date,
        lots: int,
        pricer: Any,
        guard: Any,
    ) -> list[PositionLeg]:
        del guard
        expiry_text = signals.get("expiry")
        if not expiry_text:
            return []
        expiry = _dt.date.fromisoformat(str(expiry_text))
        spot = float(signals.get("spot") or 0.0)
        if spot <= 0:
            return []
        selection_date = _dt.date.fromisoformat(str(signals["signal_date"]))
        lot_size = pricer.get_lot_size(self.symbol)
        legs = []
        for raw_leg in self.definition.get("legs", []):
            option_type = str(raw_leg.get("type") or raw_leg.get("option_type") or "CE").upper()
            action = str(raw_leg.get("action") or "BUY").upper()
            strike = self._resolve_strike(
                raw_leg.get("strike", "ATM"), spot, expiry, selection_date, pricer, signals
            )
            if strike is None:
                return []
            observation = pricer.get_option_observation(
                self.symbol, expiry, strike, option_type, day, "open"
            )
            if not observation.executable or observation.price is None or observation.price <= 0.5:
                return []
            legs.append(
                PositionLeg(
                    symbol=self.symbol,
                    expiry=expiry,
                    strike=float(strike),
                    option_type=option_type,
                    action=action,
                    lots=max(int(lots), 1),
                    lot_size=lot_size,
                    entry_price=float(observation.price),
                    entry_date=day,
                    entry_price_source=observation.source,
                    entry_price_quality=observation.quality.value,
                    entry_execution_point="NEXT_SESSION_OPEN",
                )
            )
        if legs:
            self._last_entry_date = day
        return legs

    def should_exit(
        self,
        position: Any,
        day: _dt.date,
        portfolio: Any,
        pricer: Any,
    ) -> tuple[bool, str]:
        exit_logic = self.exit_logic
        days_held = max((day - position.entry_date).days, 0)
        if exit_logic.get("type") == "expiry" or exit_logic == "expiry":
            if any(day >= leg.expiry for leg in position.legs):
                return True, "expiry"
        max_days = int(exit_logic.get("days") or self.definition.get("max_days") or 0)
        if max_days and days_held >= max_days:
            return True, "time_exit"
        if any(day >= leg.expiry for leg in position.legs):
            return True, "expiry"

        gross = portfolio.position_gross_pnl(position, day, pricer)
        premium = sum(leg.entry_price * leg.lots * leg.lot_size for leg in position.legs)
        if premium <= 0:
            return False, ""
        profit_pct = float(exit_logic.get("profit_pct") or 0.0)
        loss_pct = float(exit_logic.get("loss_pct") or self.definition.get("stop_loss_pct") or 0.0)
        if profit_pct and gross >= premium * profit_pct / 100.0:
            return True, "profit_target"
        if loss_pct and gross <= -premium * loss_pct / 100.0:
            return True, "stop_loss"
        return False, ""

    def _cadence_ok(self, day: _dt.date, signals: dict[str, Any]) -> bool:
        cadence = str(self.definition.get("cadence") or "daily").lower()
        dte = int(float(signals.get("expiry.dte_weekly") or 0.0))
        if cadence == "daily":
            return True
        if cadence == "weekly":
            if self._last_entry_date and (day - self._last_entry_date).days < 5:
                return False
            return dte >= int(self.definition.get("min_dte", 1))
        if cadence == "expiry":
            return dte <= int(self.definition.get("entry_dte_max", 1))
        if cadence == "monthly":
            return self._last_entry_date is None or day.month != self._last_entry_date.month
        return True

    def _resolve_strike(
        self,
        spec: Any,
        spot: float,
        expiry: _dt.date,
        day: _dt.date,
        pricer: Any,
        signals: dict[str, Any],
    ) -> float | None:
        if isinstance(spec, (int, float)):
            return pricer.nearest_available_strike(self.symbol, float(spec), expiry, day)
        text = str(spec or "ATM").strip().upper()
        if text == "ATM":
            return pricer.get_atm_strike(self.symbol, spot, expiry, day)
        if text == "GRAVITY_CENTER":
            center = float(signals.get("options.max_pain") or spot)
            return pricer.nearest_available_strike(self.symbol, center, expiry, day)
        if text.startswith("ATM+") or text.startswith("ATM-"):
            sign = 1.0 if "+" in text else -1.0
            offset = float(text.split("+" if sign > 0 else "-", 1)[1])
            return pricer.nearest_available_strike(self.symbol, spot + sign * offset, expiry, day)
        if text == "OTM_CALL_1SD":
            return pricer.nearest_available_strike(self.symbol, spot * 1.01, expiry, day)
        if text == "OTM_PUT_1SD":
            return pricer.nearest_available_strike(self.symbol, spot * 0.99, expiry, day)
        return pricer.nearest_available_strike(self.symbol, spot, expiry, day)


    @property
    def symbol(self) -> str:
        return str(self.definition.get("symbol") or "NIFTY").upper()


def _condition_ok(condition: dict[str, Any], signals: dict[str, Any]) -> bool:
    name = condition.get("signal")
    if not name:
        return True
    actual = signals.get(str(name))
    expected = condition.get("value")
    op = str(condition.get("operator") or "==").lower()
    if actual is None:
        return bool(condition.get("allow_missing"))
    if op in {"==", "="}:
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return actual in (expected or [])
    if op == "contains":
        return str(expected).upper() in str(actual).upper()
    try:
        a = float(actual)
        b = float(expected)
    except (TypeError, ValueError):
        return False
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    return False


def _last_numeric(frame: Any, column: str) -> float:
    if frame is None or frame.empty or column not in frame:
        return 0.0
    try:
        value = float(frame[column].iloc[-1])
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _change_pct(frame: Any, days: int) -> float:
    if frame is None or frame.empty or len(frame) <= days or "close" not in frame:
        return 0.0
    current = float(frame["close"].iloc[-1])
    previous = float(frame["close"].iloc[-1 - days])
    return (current / previous - 1.0) * 100.0 if previous > 0 else 0.0


def _volume_ratio(frame: Any) -> float:
    if frame is None or frame.empty or len(frame) < 21 or "volume" not in frame:
        return 1.0
    current = float(frame["volume"].iloc[-1])
    avg = float(frame["volume"].iloc[-21:-1].mean())
    return current / avg if avg > 0 else 1.0


def _breakout_20d(frame: Any) -> float:
    if frame is None or frame.empty or len(frame) < 21:
        return 0.0
    close = float(frame["close"].iloc[-1])
    high = float(frame["high"].iloc[-21:-1].max())
    return 1.0 if close > high else 0.0


def _bb_squeeze(frame: Any) -> float:
    if frame is None or frame.empty or len(frame) < 40:
        return 0.0
    close = frame["close"].astype(float)
    std = close.rolling(20).std()
    mean = close.rolling(20).mean()
    width = (4.0 * std / mean).dropna()
    if len(width) < 20:
        return 0.0
    return 1.0 if float(width.iloc[-1]) <= float(width.quantile(0.25)) else 0.0


def _vol_regime(frame: Any) -> str:
    if frame is None or frame.empty or len(frame) < 25:
        return "MED_VOL"
    returns = frame["close"].astype(float).pct_change().dropna().tail(20)
    vol = float(returns.std() * np.sqrt(252.0)) if len(returns) else 0.0
    if vol < 0.12:
        return "LOW_VOL"
    if vol > 0.22:
        return "HIGH_VOL"
    return "MED_VOL"


def _pcr(chain: list[dict[str, Any]]) -> float:
    call_oi = sum(float(row.get("oi") or 0.0) for row in chain if str(row.get("option_type")).upper() == "CE")
    put_oi = sum(float(row.get("oi") or 0.0) for row in chain if str(row.get("option_type")).upper() == "PE")
    return put_oi / call_oi if call_oi > 0 else 1.0


def _atm_iv_proxy(chain: list[dict[str, Any]], spot: float) -> float:
    if not chain or spot <= 0:
        return 0.15
    best = min(chain, key=lambda row: abs(float(row.get("strike") or 0.0) - spot))
    premium = float(best.get("close") or 0.0)
    return min(max(premium / spot * 8.0, 0.05), 0.8) if spot > 0 else 0.15


def _iv_rank_proxy(chain: list[dict[str, Any]], spot: float) -> float:
    iv = _atm_iv_proxy(chain, spot)
    return max(0.0, min(100.0, (iv - 0.08) / 0.35 * 100.0))


def _gex_proxy(chain: list[dict[str, Any]], spot: float) -> float:
    if spot <= 0:
        return 0.0
    total = 0.0
    for row in chain:
        strike = float(row.get("strike") or 0.0)
        oi = float(row.get("oi") or 0.0)
        option_type = str(row.get("option_type") or "").upper()
        distance = max(abs(strike - spot), 50.0)
        sign = 1.0 if option_type == "CE" and strike >= spot else -1.0 if option_type == "PE" and strike <= spot else 0.3
        total += sign * oi / (distance * distance)
    return math.tanh(total / 1000.0) * 100.0


def _max_pain(chain: list[dict[str, Any]]) -> float:
    strikes = sorted({float(row.get("strike") or 0.0) for row in chain if float(row.get("strike") or 0.0) > 0})
    if not strikes:
        return 0.0
    best_strike = strikes[0]
    best_pain: float | None = None
    for settlement in strikes:
        pain = 0.0
        for row in chain:
            strike = float(row.get("strike") or 0.0)
            oi = float(row.get("oi") or 0.0)
            if str(row.get("option_type")).upper() == "CE":
                pain += max(settlement - strike, 0.0) * oi
            else:
                pain += max(strike - settlement, 0.0) * oi
        if best_pain is None or pain < best_pain:
            best_pain = pain
            best_strike = settlement
    return best_strike
