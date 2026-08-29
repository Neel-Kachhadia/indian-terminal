from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    gst: float = 0.0
    stamp: float = 0.0
    slippage: float = 0.0
    total: float = 0.0
    total_pct: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {key: round(float(value), 6) for key, value in asdict(self).items()}

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        total_turnover_pct = self.total_pct + other.total_pct
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange=self.exchange + other.exchange,
            sebi=self.sebi + other.sebi,
            gst=self.gst + other.gst,
            stamp=self.stamp + other.stamp,
            slippage=self.slippage + other.slippage,
            total=self.total + other.total,
            total_pct=total_turnover_pct,
        )


@dataclass
class IndianMarketCosts:
    """
    Indian derivatives transaction cost model.

    Rates follow the project contract. Values are rupees except percentage
    rates, which are stored as decimal fractions.
    """

    brokerage_per_order: float = 20.0
    brokerage_pct_max: float = 0.0003

    stt_options_sell: float = 0.0005
    stt_futures_sell: float = 0.0001
    stt_equity_delivery_buy: float = 0.001
    stt_equity_delivery_sell: float = 0.001
    stt_equity_intraday: float = 0.00025

    nse_options: float = 0.000053
    nse_futures: float = 0.0000019
    nse_equity: float = 0.0000322

    sebi_fee: float = 0.000001
    gst_rate: float = 0.18

    stamp_duty_options: float = 0.00003
    stamp_duty_futures: float = 0.00002
    stamp_duty_equity_delivery: float = 0.00015

    slippage_atm_options: float = 0.001
    slippage_otm_options: float = 0.003
    slippage_deep_otm: float = 0.008
    slippage_futures: float = 0.00005
    slippage_equity: float = 0.0002

    def compute_options_cost(
        self,
        premium: float,
        lot_size: int,
        num_lots: int,
        action: str,
        moneyness: str = "ATM",
        is_entry: bool = True,
        *,
        cost_model: str = "full",
        slippage_model: str = "realistic",
    ) -> CostBreakdown:
        del is_entry
        if cost_model == "zero":
            return CostBreakdown()

        action = action.strip().upper()
        turnover = max(float(premium), 0.0) * max(int(lot_size), 0) * max(int(num_lots), 0)
        if turnover <= 0:
            return CostBreakdown()

        brokerage = self.brokerage_per_order
        stt = turnover * self.stt_options_sell if action == "SELL" else 0.0
        exchange = turnover * self.nse_options
        sebi = turnover * self.sebi_fee
        gst = (brokerage + exchange) * self.gst_rate
        stamp = turnover * self.stamp_duty_options if action == "BUY" else 0.0
        slippage = turnover * self._option_slippage_rate(moneyness, slippage_model)

        if cost_model == "brokerage_only":
            total = brokerage
            return CostBreakdown(brokerage=brokerage, total=total, total_pct=total / turnover * 100.0)

        total = brokerage + stt + exchange + sebi + gst + stamp + slippage
        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange=exchange,
            sebi=sebi,
            gst=gst,
            stamp=stamp,
            slippage=slippage,
            total=total,
            total_pct=total / turnover * 100.0,
        )

    def compute_futures_cost(
        self,
        price: float,
        lot_size: int,
        num_lots: int,
        action: str,
        *,
        cost_model: str = "full",
        slippage_model: str = "realistic",
    ) -> CostBreakdown:
        if cost_model == "zero":
            return CostBreakdown()

        turnover = max(float(price), 0.0) * max(int(lot_size), 0) * max(int(num_lots), 0)
        if turnover <= 0:
            return CostBreakdown()

        brokerage = self.brokerage_per_order
        stt = turnover * self.stt_futures_sell if action.strip().upper() == "SELL" else 0.0
        exchange = turnover * self.nse_futures
        sebi = turnover * self.sebi_fee
        gst = (brokerage + exchange) * self.gst_rate
        stamp = turnover * self.stamp_duty_futures if action.strip().upper() == "BUY" else 0.0
        slippage = 0.0 if slippage_model == "zero" else turnover * self.slippage_futures

        if cost_model == "brokerage_only":
            return CostBreakdown(brokerage=brokerage, total=brokerage, total_pct=brokerage / turnover * 100.0)

        total = brokerage + stt + exchange + sebi + gst + stamp + slippage
        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange=exchange,
            sebi=sebi,
            gst=gst,
            stamp=stamp,
            slippage=slippage,
            total=total,
            total_pct=total / turnover * 100.0,
        )

    def compute_leg_cost(
        self,
        leg: Any,
        price: float,
        action: str,
        spot: float,
        *,
        cost_model: str = "full",
        slippage_model: str = "realistic",
    ) -> CostBreakdown:
        option_type = str(getattr(leg, "option_type", "")).upper()
        lot_size = getattr(leg, "lot_size", None)
        if not isinstance(lot_size, int) or lot_size <= 0:
            raise ValueError(
                f"cannot cost a position leg without a validated positive lot_size "
                f"(got {lot_size!r})"
            )
        if option_type == "FUT":
            return self.compute_futures_cost(
                price,
                lot_size,
                int(getattr(leg, "lots", 1)),
                action,
                cost_model=cost_model,
                slippage_model=slippage_model,
            )

        return self.compute_options_cost(
            price,
            lot_size,
            int(getattr(leg, "lots", 1)),
            action,
            _moneyness(option_type, float(getattr(leg, "strike", 0.0)), spot),
            cost_model=cost_model,
            slippage_model=slippage_model,
        )

    def round_trip_cost_pct(
        self,
        premium: float,
        instrument: str = "OPTIONS_ATM",
    ) -> float:
        moneyness = "ATM"
        if "DEEP" in instrument.upper():
            moneyness = "DEEP_OTM"
        elif "OTM" in instrument.upper():
            moneyness = "OTM"
        buy = self.compute_options_cost(premium, 1, 1, "BUY", moneyness)
        sell = self.compute_options_cost(premium, 1, 1, "SELL", moneyness)
        turnover = max(float(premium), 0.0) * 2.0
        return (buy.total + sell.total) / turnover * 100.0 if turnover > 0 else 0.0

    def _option_slippage_rate(self, moneyness: str, slippage_model: str) -> float:
        if slippage_model == "zero":
            return 0.0
        multiplier = 1.8 if slippage_model == "pessimistic" else 1.0
        rate = {
            "ATM": self.slippage_atm_options,
            "OTM": self.slippage_otm_options,
            "DEEP_OTM": self.slippage_deep_otm,
        }.get(moneyness.upper(), self.slippage_atm_options)
        return rate * multiplier


def _moneyness(option_type: str, strike: float, spot: float) -> str:
    if spot <= 0 or strike <= 0:
        return "ATM"
    distance_pct = abs(strike - spot) / spot
    if distance_pct <= 0.0075:
        return "ATM"
    if distance_pct <= 0.03:
        return "OTM"
    return "DEEP_OTM"
