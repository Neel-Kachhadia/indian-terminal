from __future__ import annotations

import copy
import datetime as _dt
from enum import StrEnum
import json
import math
import traceback
from typing import Any, Callable

from fastapi import HTTPException

from data.exchange_calendar import exchange_calendar, today_ist
from db.duckdb_client import DB_WRITE_LOCK, writer_connection
from db.write_service import ReadOperation, resolve_read

from .chronological_segments import chronological_segment_summary
from .costs import IndianMarketCosts
from .metrics import compute_metrics
from .monte_carlo import monte_carlo_summary
from .options_pricer import HistoricalOptionsPricer, _fo_symbol, _ohlcv_symbols, _parse_date
from .price_quality import PRICE_POLICY_VERSION
from .position import Portfolio
from .regime_analyzer import regime_attribution
from .reporter import BacktestReporter
from .signal_validator import validate_signals
from .stress_tester import stress_test
from .strategies import get_strategy


ProgressCallback = Callable[[float, str], None]

EXECUTION_TIMING_POLICY_VERSION = "daily-t-close-next-session-open-v1"


class BacktestValidity(StrEnum):
    VALIDATED = "VALIDATED_EXECUTABLE_PRICE_POLICY"
    INVALID_PRICE_DATA = "INVALID_PRICE_DATA"
    INVALID_SPOT_DATA = "INVALID_SPOT_DATA"
    INVALID_TIMING = "INVALID_TIMING"
    NO_EXECUTIONS = "NO_EXECUTIONS"
    UNVERIFIED = "UNVERIFIED"
    INVALID_LEGACY = "INVALID_LEGACY_PRICE_SEMANTICS"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class LookaheadBiasError(RuntimeError):
    pass


class LookaheadGuard:
    """
    All historical data access used by strategy decisions passes here.
    """

    def __init__(self, connection: Any, pricer: HistoricalOptionsPricer, debug: bool = True) -> None:
        self.con = connection
        self.pricer = pricer
        self.debug = debug
        self.simulation_date: _dt.date | None = None
        self.violations: list[str] = []

    def set_simulation_date(self, day: _dt.date) -> None:
        self.simulation_date = day

    def get_ohlcv(self, symbol: str, as_of_date: _dt.date, lookback: int = 260) -> Any:
        self._check(as_of_date, "ohlcv")
        return self.pricer.get_ohlcv_window(symbol, as_of_date, lookback)

    def get_fo_eod(
        self,
        symbol: str,
        as_of_date: _dt.date,
        expiry: _dt.date | None = None,
    ) -> list[dict[str, Any]]:
        self._check(as_of_date, "fo_eod")
        params: list[Any] = [_fo_symbol(symbol), as_of_date.isoformat()]
        expiry_clause = ""
        if expiry is not None:
            expiry_clause = "AND expiry <= CAST(? AS DATE)"
            params.append(expiry.isoformat())
        rows = self.con.execute(
            f"""
            SELECT CAST(date AS VARCHAR), CAST(expiry AS VARCHAR), strike,
                   option_type, open, high, low, close, volume, oi, oi_change
            FROM fo_eod
            WHERE symbol = ?
              AND date <= CAST(? AS DATE)
              {expiry_clause}
            ORDER BY date ASC
            """,
            params,
        ).fetchall()
        return [
            {
                "date": row[0],
                "expiry": row[1],
                "strike": _number_or_none(row[2]),
                "option_type": row[3],
                "open": _number_or_none(row[4]),
                "high": _number_or_none(row[5]),
                "low": _number_or_none(row[6]),
                "close": _number_or_none(row[7]),
                "volume": int(row[8] or 0),
                "oi": int(row[9] or 0),
                "oi_change": int(row[10] or 0),
            }
            for row in rows
        ]

    def get_signal(self, signal_name: str, symbol: str, as_of_date: _dt.date) -> float | None:
        self._check(as_of_date, f"signal:{signal_name}")
        names = _signal_name_candidates(signal_name)
        symbols = _symbol_candidates(symbol)
        placeholders_names = ", ".join("?" for _ in names)
        placeholders_symbols = ", ".join("?" for _ in symbols)
        row = self.con.execute(
            f"""
            SELECT value
            FROM signals_history
            WHERE signal_name IN ({placeholders_names})
              AND symbol IN ({placeholders_symbols})
              AND date <= CAST(? AS DATE)
            ORDER BY date DESC
            LIMIT 1
            """,
            [*names, *symbols, as_of_date.isoformat()],
        ).fetchone()
        return _number_or_none(row[0] if row else None)

    def get_bif_value(self, column: str, symbol: str, as_of_date: _dt.date) -> float | None:
        self._check(as_of_date, f"bif:{column}")
        safe_column = _bif_column(column)
        symbols = _symbol_candidates(symbol)
        placeholders = ", ".join("?" for _ in symbols)
        row = self.con.execute(
            f"""
            SELECT {safe_column}
            FROM bif_history
            WHERE symbol IN ({placeholders})
              AND date <= CAST(? AS DATE)
            ORDER BY date DESC
            LIMIT 1
            """,
            [*symbols, as_of_date.isoformat()],
        ).fetchone()
        return _number_or_none(row[0] if row else None)

    def get_bif_label(self, column: str, symbol: str, as_of_date: _dt.date) -> str | None:
        self._check(as_of_date, f"bif:{column}")
        safe_column = _bif_column(column)
        symbols = _symbol_candidates(symbol)
        placeholders = ", ".join("?" for _ in symbols)
        row = self.con.execute(
            f"""
            SELECT {safe_column}
            FROM bif_history
            WHERE symbol IN ({placeholders})
              AND date <= CAST(? AS DATE)
            ORDER BY date DESC
            LIMIT 1
            """,
            [*symbols, as_of_date.isoformat()],
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None

    def get_chain_snapshot(
        self,
        symbol: str,
        as_of_date: _dt.date,
        expiry: _dt.date | None,
    ) -> list[dict[str, Any]]:
        self._check(as_of_date, "chain_snapshot")
        if expiry is None:
            return []
        return self.pricer.get_option_chain_snapshot(symbol, as_of_date, expiry)

    def _check(self, as_of_date: _dt.date, label: str) -> bool:
        if self.simulation_date is None or as_of_date <= self.simulation_date:
            return True
        message = f"lookahead access blocked: {label} as_of={as_of_date} simulation={self.simulation_date}"
        self.violations.append(message)
        if self.debug:
            raise LookaheadBiasError(message)
        print(message)
        return False


class BacktestEngine:
    def __init__(
        self,
        strategy: str,
        from_date: _dt.date | str,
        to_date: _dt.date | str,
        initial_capital: float = 500000.0,
        position_sizer: str = "quarter_kelly",
        cost_model: str = "full",
        slippage_model: str = "realistic",
        symbol: str = "NIFTY",
        chronological_segments: bool = True,
        monte_carlo: bool = True,
        n_mc_simulations: int = 10000,
    ) -> None:
        self.strategy_key = str(strategy or "sell_weekly_atm_straddle")
        self.symbol = str(symbol or "NIFTY").strip().upper()
        self.from_date = _parse_date(from_date)
        self.to_date = _parse_date(to_date)
        self.initial_capital = float(initial_capital)
        self.position_sizer = str(position_sizer or "quarter_kelly")
        self.cost_model = str(cost_model or "full")
        self.slippage_model = str(slippage_model or "realistic")
        self.chronological_segments_enabled = bool(chronological_segments)
        self.monte_carlo_enabled = bool(monte_carlo)
        self.n_mc_simulations = int(n_mc_simulations)

    def run(self, progress: ProgressCallback | None = None) -> dict[str, Any]:
        if self.from_date > self.to_date:
            raise ValueError("from_date must be before to_date")
        progress = progress or (lambda _pct, _stage: None)
        progress(0.01, "opening_duckdb")
        con = writer_connection()
        try:
            ensure_backtest_schema(con)
            pricer = HistoricalOptionsPricer(con)
            guard = LookaheadGuard(con, pricer, debug=True)
            strategy = get_strategy(self.strategy_key, self.symbol)
            cost_engine = IndianMarketCosts()
            portfolio = Portfolio(self.initial_capital)
            trading_days = pricer.get_trading_days(self.symbol, self.from_date, self.to_date)
            if not trading_days:
                raise ValueError(f"no OHLCV trading days for {self.symbol} in requested period")

            daily_records: list[dict[str, Any]] = []
            progress(0.05, "replaying_market_history")
            for index, day in enumerate(trading_days):
                if index == 0:
                    guard.set_simulation_date(day)
                    signals = {
                        "date": day.isoformat(),
                        "signal_date": None,
                        "execution_skipped": "no_prior_completed_session",
                    }
                else:
                    previous_day = trading_days[index - 1]
                    guard.set_simulation_date(previous_day)
                    expected_execution_day = expected_execution_session(previous_day)
                    if day != expected_execution_day:
                        violation = (
                            "MISSING_EXECUTION_SESSION: "
                            f"signal={previous_day.isoformat()} "
                            f"expected={expected_execution_day.isoformat()} "
                            f"observed={day.isoformat()}"
                        )
                        guard.violations.append(violation)
                        signals = {
                            "date": day.isoformat(),
                            "signal_date": previous_day.isoformat(),
                            "expected_execution_session": expected_execution_day.isoformat(),
                            "execution_skipped": "MISSING_EXECUTION_SESSION",
                            "validity_status": BacktestValidity.INVALID_TIMING.value,
                        }
                        daily_records.append(
                            {
                                "date": day.isoformat(),
                                "portfolio_value": round(portfolio.total_value(day, pricer), 2),
                                "open_positions": len(portfolio.open_positions),
                                "realized_pnl": round(portfolio.realized_pnl, 2),
                                "unrealized_pnl": round(portfolio.unrealized_pnl(day, pricer), 2),
                                "signals": signals,
                            }
                        )
                        continue

                    for position in list(portfolio.open_positions):
                        expiry_exit = any(day >= leg.expiry for leg in position.legs)
                        should_exit, reason = (
                            (True, "expiry_open")
                            if expiry_exit
                            else strategy.should_exit(position, previous_day, portfolio, pricer)
                        )
                        if should_exit:
                            portfolio.close_position(
                                position,
                                day,
                                pricer,
                                cost_engine,
                                reason=reason,
                                price_type="open",
                                execution_point="NEXT_SESSION_OPEN",
                                spot_as_of=previous_day,
                                cost_model=self.cost_model,
                                slippage_model=self.slippage_model,
                            )

                    signals = strategy.generate_signals(day, previous_day, guard, pricer)
                    if strategy.should_enter(signals, portfolio, day):
                        lots = self._compute_position_size(signals, day, portfolio, pricer)
                        legs = strategy.build_legs(signals, day, lots, pricer, guard)
                        if legs:
                            portfolio.enter_position(
                                strategy.name,
                                self.symbol,
                                legs,
                                day,
                                pricer,
                                cost_engine,
                                signals,
                                cost_model=self.cost_model,
                                slippage_model=self.slippage_model,
                            )

                daily_records.append(
                    {
                        "date": day.isoformat(),
                        "portfolio_value": round(portfolio.total_value(day, pricer), 2),
                        "open_positions": len(portfolio.open_positions),
                        "realized_pnl": round(portfolio.realized_pnl, 2),
                        "unrealized_pnl": round(portfolio.unrealized_pnl(day, pricer), 2),
                        "signals": signals,
                    }
                )
                if index % 10 == 0:
                    progress(0.05 + 0.70 * (index + 1) / len(trading_days), "replaying_market_history")

            portfolio.close_all(
                trading_days[-1],
                pricer,
                cost_engine,
                price_type="close",
                execution_point="FORCED_HORIZON_CLOSE",
                spot_as_of=trading_days[-1],
                cost_model=self.cost_model,
                slippage_model=self.slippage_model,
            )
            final_value = portfolio.total_value(trading_days[-1], pricer)
            if daily_records:
                daily_records[-1]["portfolio_value"] = round(final_value, 2)
                daily_records[-1]["realized_pnl"] = round(portfolio.realized_pnl, 2)
                daily_records[-1]["unrealized_pnl"] = 0.0

            for trade in portfolio.closed_trades:
                trade["return_on_capital_pct"] = round(float(trade["net_pnl"]) / self.initial_capital * 100.0, 5)

            progress(0.78, "computing_metrics")
            benchmark = _benchmark_records(con, self.from_date, self.to_date)
            metrics = compute_metrics(
                strategy_name=strategy.name,
                symbol=self.symbol,
                from_date=self.from_date,
                to_date=self.to_date,
                initial_capital=self.initial_capital,
                daily_records=daily_records,
                trades=portfolio.closed_trades,
                benchmark_records=benchmark,
            )
            regimes = regime_attribution(daily_records, portfolio.closed_trades)
            segments = (
                chronological_segment_summary(daily_records)
                if self.chronological_segments_enabled
                else None
            )
            mc = (
                monte_carlo_summary(
                    portfolio.closed_trades,
                    self.initial_capital,
                    self.n_mc_simulations,
                )
                if self.monte_carlo_enabled
                else None
            )
            signal_validation = validate_signals(daily_records)
            stress = stress_test(daily_records, portfolio.closed_trades)

            progress(0.90, "generating_report")
            report = BacktestReporter().generate_full_report(metrics, regimes, segments, mc)

            execution_evidence = _execution_evidence(
                portfolio.closed_trades,
                pricer,
                guard.violations,
            )

            result = {
                **metrics,
                "job_status": "completed",
                "trades": portfolio.closed_trades,
                "daily_records": daily_records,
                "regime_analysis": regimes,
                "chronological_segments": segments,
                "monte_carlo": mc,
                "signal_validation": signal_validation,
                "stress_test": stress,
                "groq_report": report["text"],
                "groq_model": report["model"],
                "groq_live": report["live"],
                "lookahead_violations": guard.violations,
                "validity_status": BacktestValidity.UNVERIFIED.value,
                "price_policy": {
                    "version": PRICE_POLICY_VERSION,
                    "price_source": "NSE_FO_BHAVCOPY",
                    "quality": "VALID_EXECUTABLE_OBSERVATION",
                    "fill_model": "exact-session stored OHLC field",
                    "missing_price_policy": "reject_backtest",
                    "liquidity_constraint": "positive traded volume",
                    "contract_expiry_behavior": "no synthetic or stale settlement fallback",
                    "timing_policy_version": EXECUTION_TIMING_POLICY_VERSION,
                    "decision_information_cutoff": "prior completed session close",
                    "entry_execution_point": "next session open",
                    "risk_exit_execution_point": "next session open after prior-close decision",
                },
                "execution_evidence": execution_evidence,
                "parameters": {
                    "strategy": self.strategy_key,
                    "symbol": self.symbol,
                    "from_date": self.from_date.isoformat(),
                    "to_date": self.to_date.isoformat(),
                    "initial_capital": self.initial_capital,
                    "position_sizer": self.position_sizer,
                    "cost_model": self.cost_model,
                    "slippage_model": self.slippage_model,
                    "chronological_segments": self.chronological_segments_enabled,
                    "monte_carlo": self.monte_carlo_enabled,
                    "n_mc_simulations": self.n_mc_simulations,
                },
            }
            validity, invalid_reason = classify_result_validity(result, "completed")
            result["validity_status"] = validity
            if invalid_reason:
                result["invalid_reason"] = invalid_reason
            progress(1.0, "complete")
            return result
        finally:
            con.close()

    def _compute_position_size(
        self,
        signals: dict[str, Any],
        trade_date: _dt.date,
        portfolio: Portfolio,
        pricer: HistoricalOptionsPricer,
    ) -> int:
        if self.position_sizer == "fixed":
            return 1
        spot = float(signals.get("spot") or pricer.get_spot(self.symbol, trade_date) or 0.0)
        lot_size = pricer.get_lot_size(self.symbol)
        risk_per_lot = max(spot * lot_size * 0.12, 1.0)
        win_prob = max(0.05, min(float(signals.get("ml.composite_probabilities.UP") or 0.55), 0.95))
        closed = portfolio.closed_trades[-30:]
        wins = [float(trade.get("net_pnl") or 0.0) for trade in closed if float(trade.get("net_pnl") or 0.0) > 0]
        losses = [abs(float(trade.get("net_pnl") or 0.0)) for trade in closed if float(trade.get("net_pnl") or 0.0) < 0]
        avg_win = sum(wins) / len(wins) if wins else risk_per_lot * 0.25
        avg_loss = sum(losses) / len(losses) if losses else risk_per_lot * 0.20
        b = max(avg_win / max(avg_loss, 1.0), 0.1)
        kelly = max((win_prob * b - (1.0 - win_prob)) / b, 0.0)
        fraction = min(kelly * 0.25, 0.08)
        lots = math.floor((self.initial_capital + portfolio.realized_pnl) * max(fraction, 0.01) / risk_per_lot)
        return max(1, min(lots, 5))


def expected_execution_session(signal_date: _dt.date) -> _dt.date:
    """The only permitted default fill date for a prior-close decision."""
    return exchange_calendar().next_session(signal_date).date


def ensure_backtest_schema(con: Any | None = None) -> None:
    owned = con is None
    connection = con or writer_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                job_id          TEXT        PRIMARY KEY,
                strategy_name   TEXT        NOT NULL,
                symbol          TEXT        NOT NULL,
                from_date       DATE        NOT NULL,
                to_date         DATE        NOT NULL,
                initial_capital DOUBLE      NOT NULL,
                status          TEXT        NOT NULL,
                progress        DOUBLE      DEFAULT 0,
                stage           TEXT,
                started_at      TIMESTAMP   NOT NULL,
                completed_at    TIMESTAMP,
                error           TEXT,
                result_json     TEXT,
                groq_report     TEXT,
                net_return_pct  DOUBLE,
                sharpe_ratio    DOUBLE,
                max_drawdown    DOUBLE,
                win_rate        DOUBLE,
                total_trades    INTEGER,
                chronological_segment_return DOUBLE,
                validity_status TEXT NOT NULL DEFAULT 'INVALID_LEGACY_PRICE_SEMANTICS',
                price_policy_version TEXT,
                invalid_reason TEXT,
                mc_p50_return   DOUBLE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_trades (
                job_id          TEXT        NOT NULL,
                trade_id        TEXT        NOT NULL,
                symbol          TEXT        NOT NULL,
                strategy        TEXT        NOT NULL,
                entry_date      DATE        NOT NULL,
                exit_date       DATE,
                legs_json       TEXT        NOT NULL,
                entry_prices    TEXT,
                exit_prices     TEXT,
                gross_pnl       DOUBLE,
                net_pnl         DOUBLE,
                costs           DOUBLE,
                regime_at_entry TEXT,
                bif_cds_entry   DOUBLE,
                iv_at_entry     DOUBLE,
                PRIMARY KEY (job_id, trade_id)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status, started_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_job ON backtest_trades(job_id)")
    finally:
        if owned:
            connection.close()


def save_run_started(job_id: str, params: dict[str, Any]) -> None:
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            ensure_backtest_schema(con)
            con.execute("DELETE FROM backtest_runs WHERE job_id = ?", [job_id])
            con.execute(
                """
                INSERT INTO backtest_runs
                (job_id, strategy_name, symbol, from_date, to_date, initial_capital,
                 status, progress, stage, started_at, validity_status,
                 price_policy_version, invalid_reason)
                VALUES (?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), ?, 'queued', 0,
                        'queued', CURRENT_TIMESTAMP, 'PENDING_VALIDATION', ?, NULL)
                """,
                [
                    job_id,
                    params["strategy"],
                    params["symbol"],
                    params["from_date"],
                    params["to_date"],
                    float(params["initial_capital"]),
                    PRICE_POLICY_VERSION,
                ],
            )
        finally:
            con.close()


def recover_interrupted_runs() -> int:
    """Fail jobs whose worker vanished during a prior application lifetime."""
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            ensure_backtest_schema(con)
            interrupted = int(
                con.execute(
                    "SELECT COUNT(*) FROM backtest_runs WHERE status IN ('queued', 'running')"
                ).fetchone()[0]
            )
            if interrupted:
                con.execute(
                    """
                    UPDATE backtest_runs
                    SET status = 'failed',
                        progress = 1,
                        stage = 'interrupted',
                        completed_at = CURRENT_TIMESTAMP,
                        error = COALESCE(error, 'application restarted before backtest completion'),
                        validity_status = 'INVALID_INTERRUPTED_RUN',
                        invalid_reason = COALESCE(
                            invalid_reason,
                            'application restarted before backtest completion'
                        )
                    WHERE status IN ('queued', 'running')
                    """
                )
            return interrupted
        finally:
            con.close()


def save_run_progress(job_id: str, progress: float, stage: str, status: str = "running") -> None:
    bounded = max(0.0, min(float(progress), 1.0))
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            con.execute(
                """
                UPDATE backtest_runs
                SET progress = ?, stage = ?, status = ?
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                [bounded, str(stage), str(status), job_id],
            )
        finally:
            con.close()


def save_run_completed(job_id: str, result: dict[str, Any]) -> None:
    result_json = json.dumps(result, default=str)
    trades = result.get("trades") or []
    validity_status, invalid_reason = classify_result_validity(result, "completed")
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            ensure_backtest_schema(con)
            con.execute("BEGIN")
            con.execute("DELETE FROM backtest_trades WHERE job_id = ?", [job_id])
            con.execute("DELETE FROM backtest_runs WHERE job_id = ?", [job_id])
            con.execute(
                """
                INSERT INTO backtest_runs
                (
                    job_id, strategy_name, symbol, from_date, to_date,
                    initial_capital, status, progress, stage, started_at,
                    completed_at, error, result_json, groq_report,
                    net_return_pct, sharpe_ratio, max_drawdown, win_rate,
                    total_trades, chronological_segment_return, mc_p50_return,
                    validity_status, price_policy_version, invalid_reason
                )
                VALUES (
                    ?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), ?, 'completed',
                    1, 'complete', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                [
                    job_id,
                    result.get("parameters", {}).get("strategy") or result.get("strategy_name") or "backtest",
                    result.get("symbol") or "NIFTY",
                    result.get("from_date"),
                    result.get("to_date"),
                    result.get("initial_capital"),
                    result_json,
                    result.get("groq_report"),
                    result.get("net_return_pct"),
                    result.get("sharpe_ratio"),
                    result.get("max_drawdown_pct"),
                    result.get("win_rate"),
                    result.get("total_trades"),
                    (result.get("chronological_segments") or {}).get("subsequent_avg_return")
                    if result.get("chronological_segments")
                    else None,
                    (result.get("monte_carlo") or {}).get("return_p50") if result.get("monte_carlo") else None,
                    validity_status,
                    PRICE_POLICY_VERSION,
                    invalid_reason,
                ],
            )
            for trade in trades:
                con.execute(
                    """
                    INSERT OR REPLACE INTO backtest_trades
                    (job_id, trade_id, symbol, strategy, entry_date, exit_date,
                     legs_json, entry_prices, exit_prices, gross_pnl, net_pnl,
                     costs, regime_at_entry, bif_cds_entry, iv_at_entry)
                    VALUES (?, ?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        job_id,
                        trade.get("trade_id"),
                        trade.get("symbol"),
                        trade.get("strategy"),
                        trade.get("entry_date"),
                        trade.get("exit_date"),
                        json.dumps(trade.get("legs") or [], default=str),
                        json.dumps(trade.get("entry_prices") or {}, default=str),
                        json.dumps(trade.get("exit_prices") or {}, default=str),
                        trade.get("gross_pnl"),
                        trade.get("net_pnl"),
                        trade.get("costs"),
                        trade.get("regime_at_entry"),
                        trade.get("bif_cds_entry"),
                        trade.get("iv_at_entry"),
                    ],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()


def save_run_failed(job_id: str, error: Exception) -> None:
    with DB_WRITE_LOCK:
        con = writer_connection()
        try:
            ensure_backtest_schema(con)
            row = con.execute(
                """
                SELECT strategy_name, symbol, CAST(from_date AS VARCHAR),
                       CAST(to_date AS VARCHAR), initial_capital
                FROM backtest_runs
                WHERE job_id = ?
                """,
                [job_id],
            ).fetchone()
            strategy, symbol, from_date, to_date, capital = row if row else ("backtest", "NIFTY", "2024-01-01", today_ist().isoformat(), 500000.0)
            con.execute("DELETE FROM backtest_runs WHERE job_id = ?", [job_id])
            con.execute(
                """
                INSERT INTO backtest_runs
                (job_id, strategy_name, symbol, from_date, to_date, initial_capital,
                 status, progress, stage, started_at, completed_at, error)
                VALUES (?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), ?, 'failed',
                        1, 'failed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """,
                [
                    job_id,
                    strategy,
                    symbol,
                    from_date,
                    to_date,
                    capital,
                    f"{error}\n{traceback.format_exc()}",
                ],
            )
        finally:
            con.close()


def load_run(job_id: str) -> dict[str, Any]:
    result = resolve_read(ReadOperation.LOAD_BACKTEST_RUN, {"job_id": job_id})
    row = result["row"]
    if not row:
        raise HTTPException(status_code=404, detail=f"backtest job not found: {job_id}")
    payload = {
        "job_id": row[0],
        "strategy": row[1],
        "symbol": row[2],
        "from_date": row[3],
        "to_date": row[4],
        "initial_capital": row[5],
        "status": row[6],
        "progress": row[7],
        "stage": row[8],
        "started_at": row[9],
        "completed_at": row[10],
        "error": row[11],
    }
    if row[12]:
        try:
            payload["result"] = json.loads(row[12])
        except ValueError:
            payload["result_json"] = row[12]
    validity, reason = classify_result_validity(payload.get("result"), payload["status"])
    payload["validity_status"] = validity
    if reason:
        payload["invalid_reason"] = reason
        if isinstance(payload.get("result"), dict):
            payload["result"]["validity_status"] = validity
            warnings = list(payload["result"].get("warnings") or [])
            payload["result"]["warnings"] = [reason, *warnings]
    return payload


def load_history(limit: int = 25) -> list[dict[str, Any]]:
    result = resolve_read(
        ReadOperation.LOAD_BACKTEST_HISTORY,
        {"limit": int(max(1, min(limit, 100)))},
    )
    rows = result["rows"]
    history: list[dict[str, Any]] = []
    for row in rows:
        result = None
        if row[12]:
            try:
                result = json.loads(row[12])
            except ValueError:
                pass
        validity, reason = classify_result_validity(result, row[5])
        history.append({
            "job_id": row[0],
            "strategy": row[1],
            "symbol": row[2],
            "from_date": row[3],
            "to_date": row[4],
            "status": row[5],
            "net_return_pct": _number_or_none(row[6]),
            "sharpe_ratio": _number_or_none(row[7]),
            "max_drawdown": _number_or_none(row[8]),
            "win_rate": _number_or_none(row[9]),
            "total_trades": int(row[10] or 0),
            "started_at": row[11],
            "validity_status": validity,
            "invalid_reason": reason,
        })
    return history


def classify_result_validity(
    result: dict[str, Any] | None,
    status: str,
) -> tuple[str, str | None]:
    if status != "completed":
        return (BacktestValidity.NOT_APPLICABLE.value, None)
    policy = result.get("price_policy") if isinstance(result, dict) else None
    if not isinstance(policy, dict) or policy.get("version") != PRICE_POLICY_VERSION:
        return (
            BacktestValidity.INVALID_LEGACY.value,
            "Legacy result quarantined: executable historical-price provenance was not enforced.",
        )
    evidence = result.get("execution_evidence") if isinstance(result, dict) else None
    if not isinstance(evidence, dict):
        return (BacktestValidity.UNVERIFIED.value, "Run has no structured execution evidence.")
    if not evidence.get("timing_contract_satisfied"):
        return (BacktestValidity.INVALID_TIMING.value, "Execution timing contract was not satisfied.")
    if int(evidence.get("invalid_spot_observation_count") or 0) > 0:
        return (BacktestValidity.INVALID_SPOT_DATA.value, "Run observed invalid spot/reference data.")
    if int(result.get("total_trades") or 0) <= 0:
        return (BacktestValidity.NO_EXECUTIONS.value, "Run executed no trades; fill policy is unproven.")
    if int(evidence.get("invalid_price_observation_count") or 0) > 0:
        return (BacktestValidity.INVALID_PRICE_DATA.value, "Run observed invalid option price data.")
    if not evidence.get("all_executed_option_fills_valid"):
        return (BacktestValidity.INVALID_PRICE_DATA.value, "One or more executed fills lack valid provenance.")
    if not evidence.get("all_required_spot_observations_valid"):
        return (BacktestValidity.INVALID_SPOT_DATA.value, "Required spot provenance is unverified.")
    if evidence.get("invalid_fallback_used"):
        return (BacktestValidity.UNVERIFIED.value, "Run used an invalid fallback.")
    return (BacktestValidity.VALIDATED.value, None)


def _execution_evidence(
    trades: list[dict[str, Any]],
    pricer: HistoricalOptionsPricer,
    lookahead_violations: list[str],
) -> dict[str, Any]:
    executed_legs = [leg for trade in trades for leg in trade.get("legs", [])]
    all_fills_valid = bool(executed_legs) and all(
        leg.get("entry_price_quality") == "VALID_EXECUTABLE_OBSERVATION"
        and leg.get("exit_price_quality") == "VALID_EXECUTABLE_OBSERVATION"
        and leg.get("entry_price_source") == "NSE_FO_BHAVCOPY"
        and leg.get("exit_price_source") == "NSE_FO_BHAVCOPY"
        for leg in executed_legs
    )
    timing_valid = not lookahead_violations and all(
        leg.get("entry_execution_point") == "NEXT_SESSION_OPEN"
        and leg.get("exit_execution_point") in {"NEXT_SESSION_OPEN", "FORCED_HORIZON_CLOSE"}
        for leg in executed_legs
    )
    invalid_prices = sum(1 for observation in pricer.price_observation_events if not observation.executable)
    invalid_spots = sum(1 for observation in pricer.spot_observation_events if not observation.valid)
    return {
        "executed_option_fill_count": len(executed_legs) * 2,
        "all_executed_option_fills_valid": all_fills_valid,
        "required_spot_observation_count": len(pricer.spot_observation_events),
        "all_required_spot_observations_valid": bool(pricer.spot_observation_events)
        and invalid_spots == 0,
        "invalid_price_observation_count": invalid_prices,
        "invalid_spot_observation_count": invalid_spots,
        "spot_observations": [
            observation.as_provenance() for observation in pricer.spot_observation_events
        ],
        "timing_contract_satisfied": timing_valid,
        "timing_policy_version": EXECUTION_TIMING_POLICY_VERSION,
        "invalid_fallback_used": any(trade.get("missing_prices") for trade in trades),
    }


def _benchmark_records(con: Any, from_date: _dt.date, to_date: _dt.date) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT CAST(date AS VARCHAR), close
        FROM decision_eligible_ohlcv
        WHERE symbol = 'NIFTY 50'
          AND UPPER(COALESCE(source, '')) = 'NSE_INDEX_BHAVCOPY'
          AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        ORDER BY date ASC
        """,
        [from_date.isoformat(), to_date.isoformat()],
    ).fetchall()
    return [{"date": row[0], "close": row[1]} for row in rows]


def _symbol_candidates(symbol: str) -> list[str]:
    return [_fo_symbol(symbol), *_ohlcv_symbols(symbol)]


def _signal_name_candidates(signal_name: str) -> list[str]:
    name = str(signal_name)
    candidates = [name]
    if "." in name:
        candidates.append(name.split(".", 1)[1])
    return list(dict.fromkeys(candidates))


def _bif_column(column: str) -> str:
    allowed = {
        "bms",
        "bms_label",
        "ngi",
        "ngi_direction",
        "gravity_center",
        "iad",
        "iad_label",
        "wyckoff_stage",
        "vrc_position",
        "vrc_label",
        "vrp",
        "nfve_fair",
        "nfve_diff",
        "nfve_valuation",
        "cds",
        "cds_label",
        "cds_confidence",
        "action",
        "anomaly_active",
    }
    if column not in allowed:
        if column == "ngi_pin_probability":
            return "ngi"
        raise ValueError(f"unsupported bif column: {column}")
    return column


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
