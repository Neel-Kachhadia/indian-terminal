from __future__ import annotations

import os
from typing import Any


class BacktestReporter:
    def generate_full_report(
        self,
        result: dict[str, Any],
        regime_analysis: dict[str, Any],
        chronological_segments: dict[str, Any] | None,
        monte_carlo: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if os.environ.get("BHARAT_BACKTEST_REPORT_MODE", "").lower() == "template":
            return {
                "text": self._template(result, regime_analysis, chronological_segments, monte_carlo),
                "model": "template_fallback",
                "live": False,
            }

        prompt = self._prompt(result, regime_analysis, chronological_segments, monte_carlo)
        try:
            from intelligence.groq_client import groq_complete

            text, model, live = groq_complete(
                prompt,
                system="You are a senior portfolio manager at a top Indian options desk.",
                max_tokens=800,
                temperature=0.2,
            )
        except Exception as error:
            text = self._template(result, regime_analysis, chronological_segments, monte_carlo)
            model = f"template_fallback: {error}"
            live = False
        if len(text.split()) < 120:
            text = self._template(result, regime_analysis, chronological_segments, monte_carlo)
            model = "template_fallback"
            live = False
        return {"text": text, "model": model, "live": live}

    def _prompt(
        self,
        result: dict[str, Any],
        regime_analysis: dict[str, Any],
        chronological_segments: dict[str, Any] | None,
        monte_carlo: dict[str, Any] | None,
    ) -> str:
        return f"""
Analyze this Indian options backtest as an institutional strategy review.

Strategy: {result.get("strategy_name")}
Period: {result.get("from_date")} to {result.get("to_date")}
Symbol: {result.get("symbol")}

Net Return: {result.get("net_return_pct")}% vs NIFTY {result.get("nifty_return_same_period")}%
Sharpe: {result.get("sharpe_ratio")}
Max Drawdown: {result.get("max_drawdown_pct")}%
Win Rate: {result.get("win_rate")}
Profit Factor: {result.get("profit_factor")}
Trades: {result.get("total_trades")}
Total Costs: {result.get("total_costs")} rupees, {result.get("total_costs_pct")}% of gross movement

Best regime: {regime_analysis.get("best_regime")}
Worst regime: {regime_analysis.get("worst_regime")}
Fixed-strategy chronological segments (not walk-forward validation): {chronological_segments}
Monte Carlo: {monte_carlo}

Write 500-800 words with:
1. executive summary
2. why strategy performed as it did
3. risk profile
4. robustness and data-mining risk
5. three improvements
6. research recommendation: CONTINUE VALIDATION or REJECT EVIDENCE.
Use the numbers. Be direct.
"""

    def _template(
        self,
        result: dict[str, Any],
        regime_analysis: dict[str, Any],
        chronological_segments: dict[str, Any] | None,
        monte_carlo: dict[str, Any] | None,
    ) -> str:
        trades = int(result.get("total_trades") or 0)
        sharpe = float(result.get("sharpe_ratio") or 0.0)
        max_dd = float(result.get("max_drawdown_pct") or 0.0)
        net = float(result.get("net_return_pct") or 0.0)
        win_rate = float(result.get("win_rate") or 0.0) * 100.0
        segments = chronological_segments or {}
        mc = monte_carlo or {}
        best = regime_analysis.get("best_regime") or "unknown"
        worst = regime_analysis.get("worst_regime") or "unknown"
        recommendation = "CONTINUE VALIDATION"
        if trades < 15 or sharpe <= 0 or max_dd < -20:
            recommendation = "REJECT EVIDENCE"
        return (
            f"Executive summary: {result.get('strategy_name')} on {result.get('symbol')} "
            f"finished with {net:.1f}% net return, {sharpe:.2f} Sharpe, {max_dd:.1f}% max "
            f"drawdown, {win_rate:.1f}% win rate, and {trades} closed trades. Costs were "
            f"Rs {float(result.get('total_costs') or 0.0):,.0f}, which is material because "
            f"Indian options costs and slippage can turn a visually strong premium strategy "
            f"into a weaker live result.\n\n"
            f"Regime attribution says best behavior came in {best}, while the weakest pocket "
            f"was {worst}. That matters more than headline return: a strategy with strong "
            f"low-volatility carry and weak high-volatility behavior should be deployed only "
            f"when the regime filter agrees. If the current BIF or volatility regime moves "
            f"away from the profitable bucket, exposure should compress quickly.\n\n"
            f"Risk profile is acceptable only if the drawdown is operationally tolerable. "
            f"The observed max drawdown of {max_dd:.1f}% and profit factor "
            f"{float(result.get('profit_factor') or 0.0):.2f} suggest the edge is "
            f"{'stable' if sharpe > 1 else 'not yet stable'}, but sample size remains a "
            f"constraint whenever trades are below 30. Fixed-strategy subsequent-segment Sharpe is "
            f"{float(segments.get('subsequent_sharpe') or 0.0):.2f}; this descriptive split does not "
            f"retrain or freeze parameters. Monte Carlo median return is {float(mc.get('return_p50') or 0.0):.1f}% "
            f"with ruin probability {float(mc.get('probability_of_ruin') or 0.0):.1f}%.\n\n"
            "Specific improvements: add a hard high-volatility disable rule, size down after "
            "two consecutive losses, and require BIF/CDS plus options-flow agreement before "
            "opening short-volatility structures. Also review entry timing around weekly "
            "expiry because NIFTY expiry mechanics changed to Tuesday and older Thursday-era "
            "data can mix different microstructure regimes.\n\n"
            f"Deployment recommendation: {recommendation}. Live use requires small lot size, "
            "monthly review, strict stop loss, and rejection of any run with fewer than 30 "
            "historical trades or materially negative subsequent-segment performance."
        )
