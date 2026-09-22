from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
import pandas as pd


CONFIGURED_ANALYTICAL_RISK_FREE_RATE = 0.065
RISK_FREE_RATE_SOURCE = "CONFIGURED_ASSUMPTION"


def compute_metrics(
    *,
    strategy_name: str,
    symbol: str,
    from_date: _dt.date,
    to_date: _dt.date,
    initial_capital: float,
    daily_records: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    benchmark_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    equity = np.array([float(row["portfolio_value"]) for row in daily_records], dtype=np.float64)
    dates = [str(row["date"]) for row in daily_records]
    if equity.size == 0:
        equity = np.array([float(initial_capital)], dtype=np.float64)
        dates = [from_date.isoformat()]

    final_capital = float(equity[-1])
    daily_returns = _pct_change(equity)
    total_return_pct = (final_capital / float(initial_capital) - 1.0) * 100.0 if initial_capital else 0.0
    years = max((to_date - from_date).days / 365.25, 1.0 / 252.0)
    annualised_return_pct = ((final_capital / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if initial_capital > 0 and final_capital > 0 else 0.0

    drawdowns, max_dd, max_dd_start, max_dd_end, max_dd_duration = _drawdowns(equity, dates)
    trade_stats = _trade_stats(trades)
    benchmark = _benchmark_stats(benchmark_records, from_date, to_date)
    total_costs = sum(float(trade.get("costs") or 0.0) for trade in trades)
    gross_profit = sum(max(float(trade.get("gross_pnl") or 0.0), 0.0) for trade in trades)
    gross_loss = sum(min(float(trade.get("gross_pnl") or 0.0), 0.0) for trade in trades)
    gross_returns = abs(gross_profit) + abs(gross_loss)

    sharpe = _sharpe(daily_returns)
    downside = daily_returns[daily_returns < 0]
    sortino = _sortino(daily_returns, downside)
    calmar = annualised_return_pct / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
    avg_drawdown_pct = float(np.mean(drawdowns) * 100.0) if drawdowns.size else 0.0
    ulcer_index = float(math.sqrt(np.mean(np.square(drawdowns * 100.0)))) if drawdowns.size else 0.0
    pain_index = float(np.mean(np.abs(drawdowns * 100.0))) if drawdowns.size else 0.0
    monthly = _period_returns(dates, equity, "ME")
    weekly = _period_returns(dates, equity, "W")

    result = {
        "strategy_name": strategy_name,
        "symbol": symbol,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "initial_capital": round(float(initial_capital), 2),
        "final_capital": round(final_capital, 2),
        **trade_stats,
        "total_return_pct": round(total_return_pct, 3),
        "net_return_pct": round(total_return_pct, 3),
        "annualised_return_pct": round(annualised_return_pct, 3),
        "risk_free_rate_annual": CONFIGURED_ANALYTICAL_RISK_FREE_RATE,
        "risk_free_rate_source": RISK_FREE_RATE_SOURCE,
        "risk_free_rate_is_observed_market_data": False,
        "monthly_returns": monthly,
        "weekly_returns": weekly,
        "daily_returns": [round(float(value * 100.0), 5) for value in daily_returns],
        "equity_curve": [round(float(value), 2) for value in equity],
        "equity_curve_dates": dates,
        "equity_points": [
            {"date": date, "value": round(float(value), 2)}
            for date, value in zip(dates, equity, strict=False)
        ],
        "drawdown_curve": [
            {"date": date, "drawdown_pct": round(float(drawdown * 100.0), 3)}
            for date, drawdown in zip(dates, drawdowns, strict=False)
        ],
        "max_drawdown_pct": round(max_dd, 3),
        "max_drawdown_duration_days": max_dd_duration,
        "max_drawdown_start": max_dd_start,
        "max_drawdown_end": max_dd_end,
        "avg_drawdown_pct": round(avg_drawdown_pct, 3),
        "recovery_factor": round(total_return_pct / abs(max_dd), 3) if abs(max_dd) > 1e-9 else 0.0,
        "ulcer_index": round(ulcer_index, 3),
        "pain_index": round(pain_index, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "omega_ratio": round(_omega(daily_returns), 3),
        "information_ratio": round(_information_ratio(daily_returns, benchmark.get("daily_returns")), 3),
        "treynor_ratio": round(_treynor(daily_returns, benchmark.get("daily_returns"), annualised_return_pct), 3),
        "sterling_ratio": round(annualised_return_pct / abs(avg_drawdown_pct), 3) if abs(avg_drawdown_pct) > 1e-9 else 0.0,
        "kappa_3": round(_kappa_3(daily_returns), 3),
        "gain_to_pain_ratio": round(_gain_to_pain(daily_returns), 3),
        "return_skewness": round(_moment(daily_returns, 3), 3),
        "return_kurtosis": round(_moment(daily_returns, 4) - 3.0, 3),
        "var_95_daily": round(_var(daily_returns, 0.05) * 100.0, 3),
        "var_99_daily": round(_var(daily_returns, 0.01) * 100.0, 3),
        "cvar_95_daily": round(_cvar(daily_returns, 0.05) * 100.0, 3),
        "cvar_99_daily": round(_cvar(daily_returns, 0.01) * 100.0, 3),
        "best_daily_return_pct": round(float(np.max(daily_returns) * 100.0), 3) if daily_returns.size else 0.0,
        "worst_daily_return_pct": round(float(np.min(daily_returns) * 100.0), 3) if daily_returns.size else 0.0,
        "positive_days": int(np.sum(daily_returns > 0)),
        "negative_days": int(np.sum(daily_returns < 0)),
        "flat_days": int(np.sum(daily_returns == 0)),
        "total_costs": round(total_costs, 2),
        "total_costs_pct": round(total_costs / gross_returns * 100.0, 3) if gross_returns > 0 else 0.0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "nifty_return_same_period": benchmark.get("return_pct"),
        "alpha_pct": round(total_return_pct - float(benchmark.get("return_pct") or 0.0), 3),
        "benchmark": benchmark,
        "warnings": _warnings(trades, daily_records),
    }
    return result


def _trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = np.array([float(trade.get("net_pnl") or 0.0) for trade in trades], dtype=np.float64)
    gross = np.array([float(trade.get("gross_pnl") or 0.0) for trade in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    breakeven = pnls[pnls == 0]
    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0
    gross_profit = float(np.sum(gross[gross > 0])) if gross.size else 0.0
    gross_loss = float(np.sum(gross[gross < 0])) if gross.size else 0.0
    net_profit = float(np.sum(wins)) if wins.size else 0.0
    net_loss = float(np.sum(losses)) if losses.size else 0.0
    gross_profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else 0.0
    net_profit_factor = net_profit / abs(net_loss) if net_loss < 0 else 0.0
    largest_win_index = int(np.argmax(pnls)) if pnls.size else -1
    largest_loss_index = int(np.argmin(pnls)) if pnls.size else -1
    streaks = _streaks(pnls)
    return {
        "total_trades": int(pnls.size),
        "winning_trades": int(wins.size),
        "losing_trades": int(losses.size),
        "breakeven_trades": int(breakeven.size),
        "win_rate": round(float(wins.size / pnls.size), 4) if pnls.size else 0.0,
        "net_hit_rate": round(float(wins.size / pnls.size), 4) if pnls.size else 0.0,
        "avg_win_pnl": round(avg_win, 2),
        "avg_loss_pnl": round(avg_loss, 2),
        "largest_win": round(float(np.max(pnls)), 2) if pnls.size else 0.0,
        "largest_loss": round(float(np.min(pnls)), 2) if pnls.size else 0.0,
        "largest_win_date": trades[largest_win_index]["exit_date"] if largest_win_index >= 0 else None,
        "largest_loss_date": trades[largest_loss_index]["exit_date"] if largest_loss_index >= 0 else None,
        "profit_factor": round(net_profit_factor, 3),
        "gross_profit_factor": round(gross_profit_factor, 3),
        "net_profit_factor": round(net_profit_factor, 3),
        "expectancy": round(float(np.mean(pnls)), 2) if pnls.size else 0.0,
        "net_expectancy": round(float(np.mean(pnls)), 2) if pnls.size else 0.0,
        "payoff_ratio": round(avg_win / abs(avg_loss), 3) if avg_loss < 0 else 0.0,
        "consecutive_wins_max": streaks["wins"],
        "consecutive_losses_max": streaks["losses"],
        "avg_days_held": round(float(np.mean([float(t.get("days_held") or 0.0) for t in trades])), 2) if trades else 0.0,
        "avg_options_dte_at_entry": round(float(np.mean([_avg_dte(t) for t in trades])), 2) if trades else 0.0,
    }


def _pct_change(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([], dtype=np.float64)
    prev = equity[:-1]
    current = equity[1:]
    mask = prev != 0
    returns = np.zeros_like(current)
    returns[mask] = current[mask] / prev[mask] - 1.0
    return returns


def _drawdowns(equity: np.ndarray, dates: list[str]) -> tuple[np.ndarray, float, str | None, str | None, int]:
    high_water = np.maximum.accumulate(equity)
    drawdowns = np.divide(equity - high_water, high_water, out=np.zeros_like(equity), where=high_water != 0)
    if drawdowns.size == 0:
        return drawdowns, 0.0, None, None, 0
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(equity[: trough + 1])) if trough >= 0 else 0
    max_dd = float(drawdowns[trough] * 100.0)
    duration = 0
    current = 0
    for dd in drawdowns:
        if dd < 0:
            current += 1
            duration = max(duration, current)
        else:
            current = 0
    return drawdowns, max_dd, dates[peak] if dates else None, dates[trough] if dates else None, duration


def _period_returns(dates: list[str], equity: np.ndarray, freq: str) -> list[dict[str, Any]]:
    if len(dates) != len(equity) or len(dates) < 2:
        return []
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity}).set_index("date")
    grouped = frame["equity"].resample(freq)
    rows = []
    for label, series in grouped:
        if series.empty:
            continue
        first = float(series.iloc[0])
        last = float(series.iloc[-1])
        start_date = series.index[0].date()
        end_date = series.index[-1].date()
        if freq == "W":
            iso_year, iso_week, _ = label.date().isocalendar()
            period = f"{iso_year:04d}-W{iso_week:02d}"
            period_type = "iso_week"
        else:
            period = label.strftime("%Y-%m")
            period_type = "calendar_month"
        rows.append(
            {
                "period": period,
                "period_type": period_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "return_pct": round((last / first - 1.0) * 100.0, 3) if first else 0.0,
            }
        )
    return rows


def _benchmark_stats(
    records: list[dict[str, Any]] | None,
    from_date: _dt.date,
    to_date: _dt.date,
) -> dict[str, Any]:
    if not records:
        return {"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "return_pct": 0.0, "daily_returns": None}
    closes = np.array([float(row["close"]) for row in records if float(row.get("close") or 0.0) > 0], dtype=np.float64)
    if closes.size < 2:
        return {"return_pct": 0.0, "daily_returns": None}
    daily_returns = _pct_change(closes)
    return {
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "start_close": round(float(closes[0]), 2),
        "end_close": round(float(closes[-1]), 2),
        "return_pct": round((float(closes[-1] / closes[0]) - 1.0) * 100.0, 3),
        "daily_returns": [float(value) for value in daily_returns],
    }


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - CONFIGURED_ANALYTICAL_RISK_FREE_RATE / 252.0
    std = float(np.std(returns, ddof=1))
    return float(np.mean(excess) / std * math.sqrt(252.0)) if std > 1e-12 else 0.0


def _sortino(returns: np.ndarray, downside: np.ndarray) -> float:
    if returns.size < 2 or downside.size < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    return float((np.mean(returns) - CONFIGURED_ANALYTICAL_RISK_FREE_RATE / 252.0) / downside_std * math.sqrt(252.0)) if downside_std > 1e-12 else 0.0


def _omega(returns: np.ndarray) -> float:
    gains = float(np.sum(returns[returns > 0]))
    losses = abs(float(np.sum(returns[returns < 0])))
    return gains / losses if losses > 1e-12 else 0.0


def _information_ratio(returns: np.ndarray, benchmark_returns: Any) -> float:
    if benchmark_returns is None:
        return 0.0
    bench = np.array(benchmark_returns, dtype=np.float64)
    size = min(returns.size, bench.size)
    if size < 2:
        return 0.0
    diff = returns[-size:] - bench[-size:]
    std = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / std * math.sqrt(252.0)) if std > 1e-12 else 0.0


def _treynor(returns: np.ndarray, benchmark_returns: Any, annualised_return_pct: float) -> float:
    if benchmark_returns is None:
        return 0.0
    bench = np.array(benchmark_returns, dtype=np.float64)
    size = min(returns.size, bench.size)
    if size < 2:
        return 0.0
    covariance = float(np.cov(returns[-size:], bench[-size:])[0, 1])
    variance = float(np.var(bench[-size:], ddof=1))
    beta = covariance / variance if variance > 1e-12 else 0.0
    return (annualised_return_pct / 100.0 - CONFIGURED_ANALYTICAL_RISK_FREE_RATE) / beta if abs(beta) > 1e-12 else 0.0


def _kappa_3(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    downside = np.minimum(returns, 0.0)
    lpm3 = float(np.mean(np.abs(downside) ** 3)) ** (1.0 / 3.0)
    return float((np.mean(returns) - CONFIGURED_ANALYTICAL_RISK_FREE_RATE / 252.0) / lpm3 * math.sqrt(252.0)) if lpm3 > 1e-12 else 0.0


def _gain_to_pain(returns: np.ndarray) -> float:
    pain = abs(float(np.sum(returns[returns < 0])))
    return float(np.sum(returns)) / pain if pain > 1e-12 else 0.0


def _moment(returns: np.ndarray, order: int) -> float:
    if returns.size < 3:
        return 0.0
    std = float(np.std(returns))
    if std <= 1e-12:
        return 0.0
    mean = float(np.mean(returns))
    return float(np.mean(((returns - mean) / std) ** order))


def _var(returns: np.ndarray, q: float) -> float:
    return float(np.quantile(returns, q)) if returns.size else 0.0


def _cvar(returns: np.ndarray, q: float) -> float:
    if returns.size == 0:
        return 0.0
    threshold = np.quantile(returns, q)
    tail = returns[returns <= threshold]
    return float(np.mean(tail)) if tail.size else float(threshold)


def _streaks(pnls: np.ndarray) -> dict[str, int]:
    max_wins = max_losses = wins = losses = 0
    for pnl in pnls:
        if pnl > 0:
            wins += 1
            losses = 0
        elif pnl < 0:
            losses += 1
            wins = 0
        else:
            wins = losses = 0
        max_wins = max(max_wins, wins)
        max_losses = max(max_losses, losses)
    return {"wins": max_wins, "losses": max_losses}


def _avg_dte(trade: dict[str, Any]) -> float:
    entry = _parse_date_or_none(trade.get("entry_date"))
    if entry is None:
        return 0.0
    dtes = []
    for leg in trade.get("legs", []):
        expiry = _parse_date_or_none(leg.get("expiry"))
        if expiry is not None:
            dtes.append(max((expiry - entry).days, 0))
    return float(np.mean(dtes)) if dtes else 0.0


def _parse_date_or_none(value: Any) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _warnings(trades: list[dict[str, Any]], daily_records: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if len(trades) < 30:
        warnings.append("Insufficient sample: fewer than 30 trades.")
    if not daily_records:
        warnings.append("No equity curve records produced.")
    missing_price_trades = sum(1 for trade in trades if trade.get("missing_prices"))
    if missing_price_trades:
        warnings.append(f"{missing_price_trades} trades used fallback prices for one or more legs.")
    return warnings
