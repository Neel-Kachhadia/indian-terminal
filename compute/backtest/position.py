from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .costs import CostBreakdown, IndianMarketCosts
from .options_pricer import MissingExecutablePriceError


@dataclass
class PositionLeg:
    symbol: str
    expiry: _dt.date
    strike: float
    option_type: str
    action: str
    lots: int
    lot_size: int
    entry_price: float
    entry_date: _dt.date
    entry_price_source: str
    entry_price_quality: str
    entry_execution_point: str

    def close_action(self) -> str:
        return "SELL" if self.action == "BUY" else "BUY"

    def signed_quantity(self) -> float:
        sign = 1.0 if self.action == "BUY" else -1.0
        return sign * float(self.lots) * float(self.lot_size)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expiry"] = self.expiry.isoformat()
        payload["entry_date"] = self.entry_date.isoformat()
        return payload


@dataclass
class OpenPosition:
    trade_id: str
    strategy: str
    symbol: str
    entry_date: _dt.date
    legs: list[PositionLeg]
    entry_costs: CostBreakdown
    regime_at_entry: str = "UNKNOWN"
    bif_cds_entry: float | None = None
    iv_at_entry: float | None = None
    signals_at_entry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat(),
            "legs": [leg.to_dict() for leg in self.legs],
            "entry_costs": self.entry_costs.to_dict(),
            "regime_at_entry": self.regime_at_entry,
            "bif_cds_entry": self.bif_cds_entry,
            "iv_at_entry": self.iv_at_entry,
            "signals_at_entry": self.signals_at_entry,
        }


class Portfolio:
    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = float(initial_capital)
        self.open_positions: list[OpenPosition] = []
        self.closed_trades: list[dict[str, Any]] = []
        self.realized_pnl = 0.0
        self.realized_costs = 0.0

    def enter_position(
        self,
        strategy: str,
        symbol: str,
        legs: list[PositionLeg],
        day: _dt.date,
        pricer: Any,
        cost_engine: IndianMarketCosts,
        signals: dict[str, Any],
        *,
        cost_model: str = "full",
        slippage_model: str = "realistic",
    ) -> OpenPosition | None:
        if not legs:
            return None
        spot = float(signals.get("spot") or 0.0)
        if spot <= 0:
            return None
        costs = CostBreakdown()
        for leg in legs:
            costs += cost_engine.compute_leg_cost(
                leg,
                leg.entry_price,
                leg.action,
                spot,
                cost_model=cost_model,
                slippage_model=slippage_model,
            )
        position = OpenPosition(
            trade_id=str(uuid4()),
            strategy=strategy,
            symbol=symbol,
            entry_date=day,
            legs=legs,
            entry_costs=costs,
            regime_at_entry=str(signals.get("regime.volatility") or "UNKNOWN"),
            bif_cds_entry=_number_or_none(signals.get("bif.cds")),
            iv_at_entry=_number_or_none(signals.get("options.atm_iv")),
            signals_at_entry=_serialisable_signals(signals),
        )
        self.open_positions.append(position)
        return position

    def close_position(
        self,
        position: OpenPosition,
        day: _dt.date,
        pricer: Any,
        cost_engine: IndianMarketCosts,
        *,
        reason: str,
        price_type: str = "close",
        execution_point: str = "SESSION_CLOSE",
        spot_as_of: _dt.date | None = None,
        cost_model: str = "full",
        slippage_model: str = "realistic",
    ) -> dict[str, Any]:
        spot_date = spot_as_of or day
        spot = pricer.get_spot(position.symbol, spot_date)
        if spot is None:
            raise MissingExecutablePriceError(
                f"cannot close position without verified same-session spot: "
                f"{position.symbol} on {spot_date}"
            )
        gross = 0.0
        exit_costs = CostBreakdown()
        leg_results: list[dict[str, Any]] = []
        missing_prices: list[str] = []

        for leg in position.legs:
            observation = pricer.get_option_observation(
                leg.symbol,
                leg.expiry,
                leg.strike,
                leg.option_type,
                day,
                price_type,
            )
            if not observation.executable or observation.price is None:
                missing_prices.append(f"{leg.option_type} {leg.strike:g} {leg.expiry}")
                raise MissingExecutablePriceError(
                    f"cannot close position without executable price: {missing_prices[-1]} on {day}"
                )
            exit_price = observation.price

            leg_pnl = (exit_price - leg.entry_price) * leg.signed_quantity()
            gross += leg_pnl
            exit_costs += cost_engine.compute_leg_cost(
                leg,
                exit_price,
                leg.close_action(),
                spot,
                cost_model=cost_model,
                slippage_model=slippage_model,
            )
            leg_results.append(
                {
                    **leg.to_dict(),
                    "exit_price": round(exit_price, 4),
                    "exit_price_source": observation.source,
                    "exit_price_quality": observation.quality.value,
                    "exit_execution_point": execution_point,
                    "gross_pnl": round(leg_pnl, 2),
                }
            )

        total_costs = position.entry_costs.total + exit_costs.total
        net = gross - total_costs
        self.realized_pnl += net
        self.realized_costs += total_costs
        if position in self.open_positions:
            self.open_positions.remove(position)

        trade = {
            "trade_id": position.trade_id,
            "symbol": position.symbol,
            "strategy": position.strategy,
            "entry_date": position.entry_date.isoformat(),
            "exit_date": day.isoformat(),
            "days_held": max((day - position.entry_date).days, 0),
            "legs": leg_results,
            "entry_prices": {
                f"{leg.option_type}:{leg.strike:g}:{leg.expiry.isoformat()}": leg.entry_price
                for leg in position.legs
            },
            "exit_prices": {
                f"{leg['option_type']}:{leg['strike']:g}:{leg['expiry']}": leg["exit_price"]
                for leg in leg_results
            },
            "gross_pnl": round(gross, 2),
            "costs": round(total_costs, 2),
            "entry_costs": position.entry_costs.to_dict(),
            "exit_costs": exit_costs.to_dict(),
            "net_pnl": round(net, 2),
            "return_on_capital_pct": 0.0,
            "reason": reason,
            "regime_at_entry": position.regime_at_entry,
            "bif_cds_entry": position.bif_cds_entry,
            "iv_at_entry": position.iv_at_entry,
            "signals_at_entry": position.signals_at_entry,
            "missing_prices": missing_prices,
        }
        self.closed_trades.append(trade)
        return trade

    def close_all(
        self,
        day: _dt.date,
        pricer: Any,
        cost_engine: IndianMarketCosts,
        *,
        price_type: str = "close",
        execution_point: str = "SESSION_CLOSE",
        spot_as_of: _dt.date | None = None,
        cost_model: str = "full",
        slippage_model: str = "realistic",
    ) -> list[dict[str, Any]]:
        trades = []
        for position in list(self.open_positions):
            trades.append(
                self.close_position(
                    position,
                    day,
                    pricer,
                    cost_engine,
                    reason="end_of_backtest",
                    price_type=price_type,
                    execution_point=execution_point,
                    spot_as_of=spot_as_of,
                    cost_model=cost_model,
                    slippage_model=slippage_model,
                )
            )
        return trades

    def position_gross_pnl(self, position: OpenPosition, day: _dt.date, pricer: Any) -> float:
        gross = 0.0
        for leg in position.legs:
            price = pricer.get_option_price(
                leg.symbol,
                leg.expiry,
                leg.strike,
                leg.option_type,
                day,
            )
            if price is None:
                raise MissingExecutablePriceError(
                    f"cannot value position without executable price: "
                    f"{leg.option_type} {leg.strike:g} {leg.expiry} on {day}"
                )
            gross += (price - leg.entry_price) * leg.signed_quantity()
        return gross

    def total_value(self, day: _dt.date, pricer: Any) -> float:
        unrealized = sum(self.position_gross_pnl(position, day, pricer) - position.entry_costs.total for position in self.open_positions)
        return self.initial_capital + self.realized_pnl + unrealized

    def unrealized_pnl(self, day: _dt.date, pricer: Any) -> float:
        return sum(self.position_gross_pnl(position, day, pricer) - position.entry_costs.total for position in self.open_positions)


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _serialisable_signals(signals: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in signals.items():
        if isinstance(value, (_dt.date, _dt.datetime)):
            output[key] = value.isoformat()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
    return output
