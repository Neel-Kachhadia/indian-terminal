from __future__ import annotations

import copy
from typing import Any

from .base import BaseStrategy


PREBUILT_STRATEGIES: dict[str, dict[str, Any]] = {
    "sell_weekly_atm_straddle": {
        "name": "Weekly ATM Straddle Sell",
        "description": "Sell weekly ATM call and put using actual fo_eod option prices.",
        "cadence": "weekly",
        "min_dte": 2,
        "max_open_positions": 1,
        "entry_conditions": [],
        "legs": [
            {"action": "SELL", "type": "CE", "strike": "ATM"},
            {"action": "SELL", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 5, "loss_pct": 80, "profit_pct": 55},
        "stop_loss_pct": 80,
    },
    "sell_iron_condor_low_vol": {
        "name": "Low Vol Iron Condor",
        "description": "Sell OTM put/call spread wings when IV rank is high but realised vol is low.",
        "cadence": "weekly",
        "min_dte": 5,
        "entry_conditions": [
            {"signal": "options.iv_rank", "operator": ">", "value": 35},
            {"signal": "regime.volatility", "operator": "in", "value": ["LOW_VOL", "MED_VOL"]},
        ],
        "legs": [
            {"action": "SELL", "type": "PE", "strike": "ATM-200"},
            {"action": "BUY", "type": "PE", "strike": "ATM-400"},
            {"action": "SELL", "type": "CE", "strike": "ATM+200"},
            {"action": "BUY", "type": "CE", "strike": "ATM+400"},
        ],
        "exit": {"days": 7, "profit_pct": 60, "loss_pct": 70},
    },
    "sell_put_support": {
        "name": "Put Support Premium Sell",
        "description": "Sell put below ATM when flow and trend are not bearish.",
        "cadence": "weekly",
        "min_dte": 3,
        "entry_conditions": [{"signal": "bif.cds", "operator": ">", "value": 45, "allow_missing": True}],
        "legs": [{"action": "SELL", "type": "PE", "strike": "ATM-200"}],
        "exit": {"days": 5, "profit_pct": 60, "loss_pct": 70},
    },
    "sell_call_resistance": {
        "name": "Call Resistance Premium Sell",
        "description": "Sell call above ATM when flow and trend are not bullish.",
        "cadence": "weekly",
        "min_dte": 3,
        "entry_conditions": [{"signal": "bif.cds", "operator": "<", "value": 55, "allow_missing": True}],
        "legs": [{"action": "SELL", "type": "CE", "strike": "ATM+200"}],
        "exit": {"days": 5, "profit_pct": 60, "loss_pct": 70},
    },
    "sell_on_vix_spike": {
        "name": "Sell Vol on VIX Spike",
        "description": "Sell ATM straddle when VIX spike mean reversion signal appears.",
        "cadence": "daily",
        "min_dte": 3,
        "entry_conditions": [{"signal": "vix.vix_spike_15pct", "value": 1}],
        "legs": [
            {"action": "SELL", "type": "CE", "strike": "ATM"},
            {"action": "SELL", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 5, "loss_pct": 80},
    },
    "buy_breakout_straddle": {
        "name": "Breakout Straddle Buy",
        "description": "Buy ATM straddle after squeeze, breakout, and volume expansion.",
        "cadence": "daily",
        "min_dte": 5,
        "entry_conditions": [
            {"signal": "price_action.bb_squeeze", "value": 1},
            {"signal": "price_action.pattern_20d_high_breakout", "value": 1},
            {"signal": "volume.volume_spike_2x", "value": 1},
        ],
        "legs": [
            {"action": "BUY", "type": "CE", "strike": "ATM"},
            {"action": "BUY", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 5, "profit_pct": 100, "loss_pct": 50},
    },
    "buy_call_on_fii_buying": {
        "name": "FII Accumulation Call Buy",
        "description": "Buy ATM call when institutional and BIF momentum are constructive.",
        "cadence": "daily",
        "min_dte": 5,
        "entry_conditions": [{"signal": "bif.bms", "operator": ">", "value": 30}],
        "legs": [{"action": "BUY", "type": "CE", "strike": "ATM"}],
        "exit": {"days": 5, "profit_pct": 80, "loss_pct": 50},
    },
    "ml_consensus_bull_spread": {
        "name": "ML Consensus Bull Call Spread",
        "description": "Buy ATM call and sell higher call when ML consensus is bullish.",
        "cadence": "daily",
        "min_dte": 5,
        "entry_conditions": [
            {"signal": "ml.model_agreement", "operator": ">", "value": 0.65},
            {"signal": "ml.composite_probabilities.UP", "operator": ">", "value": 0.55},
        ],
        "legs": [
            {"action": "BUY", "type": "CE", "strike": "ATM"},
            {"action": "SELL", "type": "CE", "strike": "ATM+200"},
        ],
        "exit": {"days": 5, "profit_pct": 70, "loss_pct": 50},
    },
    "bif_strong_bull": {
        "name": "BIF Strong Bull Call Buy",
        "description": "Buy ATM call when BIF CDS is strongly bullish and anomaly score is quiet.",
        "cadence": "daily",
        "min_dte": 5,
        "entry_conditions": [
            {"signal": "bif.cds", "operator": ">", "value": 70},
            {"signal": "ml.anomaly_score", "operator": "<", "value": 0.5},
        ],
        "legs": [{"action": "BUY", "type": "CE", "strike": "ATM"}],
        "exit": {"days": 5, "profit_pct": 80, "loss_pct": 50},
    },
    "gravity_pin_trade": {
        "name": "NGI Gravity Pin Trade",
        "description": "Sell at gravity center when pin probability is high near expiry.",
        "cadence": "expiry",
        "entry_dte_max": 3,
        "entry_conditions": [{"signal": "bif.ngi_pin_probability", "operator": ">", "value": 0.70}],
        "legs": [
            {"action": "SELL", "type": "CE", "strike": "GRAVITY_CENTER"},
            {"action": "SELL", "type": "PE", "strike": "GRAVITY_CENTER"},
        ],
        "exit": {"days": 3, "loss_pct": 80},
    },
    "vol_clock_compression_buy": {
        "name": "VRC Compression Straddle Buy",
        "description": "Buy straddle when VRC marks compression or early expansion.",
        "cadence": "daily",
        "min_dte": 7,
        "entry_conditions": [
            {"signal": "bif.vrc_label", "operator": "in", "value": ["COMPRESSION", "EARLY EXPANSION"]},
            {"signal": "options.iv_rank", "operator": "<", "value": 30},
        ],
        "legs": [
            {"action": "BUY", "type": "CE", "strike": "ATM"},
            {"action": "BUY", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 10, "profit_pct": 80, "loss_pct": 40},
    },
    "iad_accumulation_call": {
        "name": "IAD Accumulation Call Buy",
        "description": "Buy call during accumulation or markup.",
        "cadence": "daily",
        "min_dte": 7,
        "entry_conditions": [{"signal": "bif.iad", "operator": ">", "value": 50}],
        "legs": [{"action": "BUY", "type": "CE", "strike": "ATM"}],
        "exit": {"days": 10, "profit_pct": 100, "loss_pct": 50},
    },
    "rbi_event_sell_vol": {
        "name": "RBI Event IV Crush",
        "description": "Sell straddle two days before RBI policy when IV rank is elevated.",
        "cadence": "daily",
        "min_dte": 3,
        "entry_conditions": [
            {"signal": "expiry.rbi_policy_dte", "operator": "==", "value": 2},
            {"signal": "options.iv_rank", "operator": ">", "value": 50},
        ],
        "legs": [
            {"action": "SELL", "type": "CE", "strike": "ATM"},
            {"action": "SELL", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 3, "loss_pct": 70},
    },
    "results_season_momentum": {
        "name": "Results Season Momentum",
        "description": "Buy ATM call during results season with strong volume and price response.",
        "cadence": "daily",
        "min_dte": 5,
        "universe": "FNO_STOCKS",
        "entry_conditions": [
            {"signal": "macro.results_season_active", "value": 1},
            {"signal": "price_action.volume_ratio_20d", "operator": ">", "value": 2.0},
            {"signal": "price_action.change_pct_today", "operator": ">", "value": 2.0},
        ],
        "legs": [{"action": "BUY", "type": "CE", "strike": "ATM"}],
        "exit": {"days": 3, "profit_pct": 70, "loss_pct": 50},
    },
    "delta_neutral_gamma_scalp": {
        "name": "Delta Neutral Gamma Scalping",
        "description": "Buy ATM straddle when vol risk premium is cheap; daily hedge approximated by short holding.",
        "cadence": "weekly",
        "min_dte": 7,
        "entry_conditions": [
            {"signal": "options.vol_risk_premium", "operator": "<", "value": 0},
            {"signal": "options.iv_rank", "operator": "<", "value": 35},
        ],
        "legs": [
            {"action": "BUY", "type": "CE", "strike": "ATM"},
            {"action": "BUY", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 7, "profit_pct": 60, "loss_pct": 40},
    },
    "nfve_mean_reversion": {
        "name": "NFVE Mean Reversion",
        "description": "Use fair-value premium or discount to choose directional option exposure.",
        "cadence": "weekly",
        "min_dte": 10,
        "entry_conditions": [{"signal": "bif.nfve_current_vs_fair", "operator": ">", "value": 5}],
        "legs": [{"action": "BUY", "type": "PE", "strike": "ATM"}],
        "exit": {"days": 15, "profit_pct": 60, "loss_pct": 40},
    },
    "protective_put_after_rally": {
        "name": "Protective Put After Rally",
        "description": "Buy put after a fast 10-day rally when IV rank remains affordable.",
        "cadence": "daily",
        "min_dte": 7,
        "entry_conditions": [
            {"signal": "price_action.return_10d", "operator": ">", "value": 3.0},
            {"signal": "options.iv_rank", "operator": "<", "value": 45},
        ],
        "legs": [{"action": "BUY", "type": "PE", "strike": "ATM"}],
        "exit": {"days": 7, "profit_pct": 80, "loss_pct": 45},
    },
    "banknifty_weekly_short_strangle": {
        "name": "BANKNIFTY Weekly Short Strangle",
        "description": "Sell one-standard-deviation weekly call and put on BANKNIFTY.",
        "symbol": "BANKNIFTY",
        "cadence": "weekly",
        "min_dte": 2,
        "entry_conditions": [{"signal": "regime.volatility", "operator": "in", "value": ["LOW_VOL", "MED_VOL"]}],
        "legs": [
            {"action": "SELL", "type": "PE", "strike": "OTM_PUT_1SD"},
            {"action": "SELL", "type": "CE", "strike": "OTM_CALL_1SD"},
        ],
        "exit": {"days": 5, "profit_pct": 55, "loss_pct": 80},
    },
    "expiry_day_short_straddle": {
        "name": "Expiry Day Short Straddle",
        "description": "Sell ATM straddle on last trading day before expiry with tight stop.",
        "cadence": "expiry",
        "entry_dte_max": 1,
        "min_dte": 0,
        "entry_conditions": [{"signal": "regime.volatility", "operator": "!=", "value": "HIGH_VOL", "allow_missing": True}],
        "legs": [
            {"action": "SELL", "type": "CE", "strike": "ATM"},
            {"action": "SELL", "type": "PE", "strike": "ATM"},
        ],
        "exit": {"days": 1, "profit_pct": 45, "loss_pct": 50},
    },
    "momentum_put_buy": {
        "name": "Breakdown Momentum Put Buy",
        "description": "Buy ATM put after negative 5-day momentum with high-volume confirmation.",
        "cadence": "daily",
        "min_dte": 5,
        "entry_conditions": [
            {"signal": "price_action.return_5d", "operator": "<", "value": -2.0},
            {"signal": "price_action.volume_ratio_20d", "operator": ">", "value": 1.2},
        ],
        "legs": [{"action": "BUY", "type": "PE", "strike": "ATM"}],
        "exit": {"days": 5, "profit_pct": 80, "loss_pct": 45},
    },
}


def get_strategy(key: str, symbol: str | None = None) -> BaseStrategy:
    normalized = str(key or "sell_weekly_atm_straddle").strip()
    if normalized not in PREBUILT_STRATEGIES:
        raise KeyError(f"unknown backtest strategy: {normalized}")
    definition = copy.deepcopy(PREBUILT_STRATEGIES[normalized])
    if symbol:
        definition["symbol"] = symbol.strip().upper()
    return BaseStrategy(normalized, definition)


def list_strategies() -> list[dict[str, Any]]:
    rows = []
    for key, definition in PREBUILT_STRATEGIES.items():
        rows.append(
            {
                "key": key,
                "name": definition.get("name") or key,
                "description": definition.get("description") or "",
                "symbol": definition.get("symbol", "NIFTY"),
                "legs": definition.get("legs", []),
                "entry_conditions": definition.get("entry_conditions", []),
                "exit": definition.get("exit", {}),
                "cadence": definition.get("cadence", "daily"),
            }
        )
    return rows
