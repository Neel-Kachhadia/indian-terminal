import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";

type StrategyRow = {
  key: string;
  name: string;
  description?: string;
  symbol?: string;
  cadence?: string;
};

type StrategiesPayload = {
  strategies: StrategyRow[];
  count: number;
};

type BacktestRunResponse = {
  job_id: string;
  status: string;
  status_url?: string;
  result_url?: string;
};

type BacktestStatus = {
  job_id: string;
  status: string;
  progress?: number;
  stage?: string;
  error?: string | null;
  result?: BacktestResult;
  validity_status?: string;
};

type BacktestResult = {
  strategy_name?: string;
  symbol?: string;
  from_date?: string;
  to_date?: string;
  net_return_pct?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  win_rate?: number;
  total_trades?: number;
  profit_factor?: number;
  final_capital?: number;
  initial_capital?: number;
  total_costs?: number;
  total_costs_pct?: number;
  nifty_return_same_period?: number;
  alpha_pct?: number;
  equity_points?: Array<{ date: string; value: number }>;
  drawdown_curve?: Array<{ date: string; drawdown_pct: number }>;
  monthly_returns?: Array<{ period: string; return_pct: number }>;
  trades?: TradeRow[];
  regime_analysis?: {
    by_vol_regime?: Record<string, RegimeRow>;
    best_regime?: string | null;
    worst_regime?: string | null;
  };
  chronological_segments?: {
    historical_sharpe?: number;
    subsequent_sharpe?: number;
    difference_pct?: number;
    verdict?: string;
  } | null;
  monte_carlo?: {
    return_p5?: number;
    return_p50?: number;
    return_p95?: number;
    probability_of_ruin?: number;
    verdict?: string;
  } | null;
  signal_validation?: {
    signals_tested?: number;
    mean_abs_ic?: number;
    verdict?: string;
  };
  stress_test?: {
    verdict?: string;
    worst_scenario?: { scenario?: string; return_pct?: number | null } | null;
  };
  groq_report?: string;
  warnings?: string[];
  validity_status?: string;
};

type TradeRow = {
  trade_id?: string;
  entry_date?: string;
  exit_date?: string;
  net_pnl?: number;
  gross_pnl?: number;
  costs?: number;
  reason?: string;
  regime_at_entry?: string;
};

type RegimeRow = {
  trades?: number;
  win_rate?: number;
  avg_pnl?: number;
  total_pnl?: number;
};

type View = "equity" | "heatmap" | "drawdown" | "trades";

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function BacktestPanel({ symbol = "NIFTY" }: { symbol?: string }) {
  const [strategies, setStrategies] = useState<StrategyRow[]>([]);
  const [strategy, setStrategy] = useState("sell_weekly_atm_straddle");
  const [selectedSymbol, setSelectedSymbol] = useState(symbol);
  const [fromDate, setFromDate] = useState("2024-01-01");
  const [toDate, setToDate] = useState("2026-06-19");
  const [capital, setCapital] = useState(500000);
  const [costModel, setCostModel] = useState("full");
  const [sizer, setSizer] = useState("quarter_kelly");
  const [walkForward, setWalkForward] = useState(true);
  const [monteCarlo, setMonteCarlo] = useState(true);
  const [mcSims, setMcSims] = useState(10000);
  const [job, setJob] = useState<BacktestStatus | null>(null);
  const [message, setMessage] = useState("");
  const [running, setRunning] = useState(false);
  const [view, setView] = useState<View>("equity");

  useEffect(() => {
    setSelectedSymbol(symbol);
  }, [symbol]);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const payload = await invoke<StrategiesPayload>("list_strategies");
        if (active) {
          setStrategies(payload.strategies ?? []);
        }
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };
    load();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!job?.job_id || job.status === "completed" || job.status === "failed") {
      setRunning(false);
      return undefined;
    }

    setRunning(true);
    let active = true;
    const poll = async () => {
      try {
        const payload = await invoke<BacktestStatus>("get_backtest_status", {
          jobId: job.job_id,
        });
        if (active) {
          setJob(payload);
          setMessage(payload.error ?? "");
        }
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };
    const timer = window.setInterval(poll, 1500);
    poll();
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  const result = job?.result;
  const selectedStrategy = useMemo(
    () => strategies.find((item) => item.key === strategy),
    [strategies, strategy],
  );

  const run = async () => {
    setMessage("");
    setRunning(true);
    setJob(null);
    try {
      const response = await invoke<BacktestRunResponse>("run_backtest", {
        params: JSON.stringify({
          strategy,
          symbol: selectedSymbol,
          from_date: fromDate,
          to_date: toDate,
          initial_capital: capital,
          chronological_segments: walkForward,
          monte_carlo: monteCarlo,
          n_mc_simulations: mcSims,
          cost_model: costModel,
          position_sizer: sizer,
        }),
      });
      setJob({
        job_id: response.job_id,
        status: response.status,
        progress: 0,
        stage: "queued",
      });
    } catch (error) {
      setRunning(false);
      setMessage(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <section style={wrapperStyle} aria-label="Backtest engine">
      <div style={controlBandStyle}>
        <div style={titleBlockStyle}>
          <span style={eyebrowStyle}>BACKTEST ENGINE</span>
          <strong style={titleStyle}>{selectedStrategy?.name ?? "Strategy"}</strong>
        </div>
        <Field label="Strategy">
          <select value={strategy} onChange={(event) => setStrategy(event.target.value)} style={inputStyle}>
            {strategies.map((row) => (
              <option key={row.key} value={row.key}>
                {row.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Symbol">
          <select value={selectedSymbol} onChange={(event) => setSelectedSymbol(event.target.value)} style={inputStyle}>
            <option value="NIFTY">NIFTY</option>
            <option value="BANKNIFTY">BANKNIFTY</option>
            <option value="RELIANCE">RELIANCE</option>
            <option value="HDFCBANK">HDFCBANK</option>
          </select>
        </Field>
        <Field label="From">
          <input style={inputStyle} type="date" value={fromDate} onChange={(event) => setFromDate(event.target.value)} />
        </Field>
        <Field label="To">
          <input style={inputStyle} type="date" value={toDate} onChange={(event) => setToDate(event.target.value)} />
        </Field>
        <Field label="Capital">
          <input
            min={10000}
            step={50000}
            style={inputStyle}
            type="number"
            value={capital}
            onChange={(event) => setCapital(Number(event.target.value))}
          />
        </Field>
        <Field label="Costs">
          <select value={costModel} onChange={(event) => setCostModel(event.target.value)} style={inputStyle}>
            <option value="full">Full Indian Costs</option>
            <option value="brokerage_only">Brokerage Only</option>
            <option value="zero">Zero Cost</option>
          </select>
        </Field>
        <Field label="Sizer">
          <select value={sizer} onChange={(event) => setSizer(event.target.value)} style={inputStyle}>
            <option value="quarter_kelly">Quarter Kelly</option>
            <option value="fixed">Fixed 1 Lot</option>
          </select>
        </Field>
        <Toggle label="Segments" checked={walkForward} setChecked={setWalkForward} />
        <Toggle label="MC" checked={monteCarlo} setChecked={setMonteCarlo} />
        <Field label="Sims">
          <input
            min={100}
            step={1000}
            style={inputStyle}
            type="number"
            value={mcSims}
            onChange={(event) => setMcSims(Number(event.target.value))}
          />
        </Field>
        <button disabled={running} onClick={run} style={runButtonStyle(running)} type="button">
          {running ? "RUNNING" : "RUN BACKTEST"}
        </button>
      </div>

      <div style={progressStyle}>
        <div style={progressTrackStyle}>
          <div style={{ ...progressFillStyle, width: `${Math.round((job?.progress ?? 0) * 100)}%` }} />
        </div>
        <span>{job?.stage ?? "idle"}</span>
        {message ? <span style={messageStyle}>{message}</span> : null}
      </div>

      {result ? (
        <>
          {result.validity_status === "INVALID_LEGACY_PRICE_SEMANTICS" ? (
            <div role="alert" style={invalidResultStyle}>
              INVALID LEGACY RESULT — historical fill executability was not proven. Do not use as performance evidence.
            </div>
          ) : null}
          <MetricStrip result={result} />
          <div style={tabBarStyle}>
            {(["equity", "heatmap", "drawdown", "trades"] as View[]).map((item) => (
              <button key={item} onClick={() => setView(item)} style={tabStyle(view === item)} type="button">
                {labelView(item)}
              </button>
            ))}
          </div>
          <div style={resultGridStyle}>
            <div style={chartShellStyle}>
              {view === "equity" ? <EquityChart points={result.equity_points ?? []} /> : null}
              {view === "heatmap" ? <MonthlyHeatmap rows={result.monthly_returns ?? []} /> : null}
              {view === "drawdown" ? <DrawdownChart points={result.drawdown_curve ?? []} /> : null}
              {view === "trades" ? <TradeLog trades={result.trades ?? []} /> : null}
            </div>
            <AnalysisColumn result={result} />
          </div>
        </>
      ) : (
        <div style={emptyStyle}>{selectedStrategy?.description ?? "Select strategy."}</div>
      )}
    </section>
  );
}

function Field({ children, label }: { children: ReactNode; label: string }) {
  return (
    <label style={fieldStyle}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function Toggle({
  checked,
  label,
  setChecked,
}: {
  checked: boolean;
  label: string;
  setChecked: (value: boolean) => void;
}) {
  return (
    <label style={toggleStyle}>
      <span>{label}</span>
      <input checked={checked} onChange={(event) => setChecked(event.target.checked)} type="checkbox" />
    </label>
  );
}

function MetricStrip({ result }: { result: BacktestResult }) {
  return (
    <div style={metricStripStyle}>
      <Metric label="Net" value={formatSignedPct(result.net_return_pct)} color={signedColor(result.net_return_pct)} />
      <Metric label="Sharpe" value={formatNumber(result.sharpe_ratio)} color={scoreColor(result.sharpe_ratio, 1)} />
      <Metric label="Max DD" value={formatSignedPct(result.max_drawdown_pct)} color="#ff6b6b" />
      <Metric label="Win" value={formatUnitPct(result.win_rate)} />
      <Metric label="Trades" value={String(result.total_trades ?? 0)} />
      <Metric label="Alpha" value={formatSignedPct(result.alpha_pct)} color={signedColor(result.alpha_pct)} />
      <Metric label="Costs" value={formatCurrency(result.total_costs)} />
      <Metric label="PF" value={formatNumber(result.profit_factor)} />
    </div>
  );
}

function Metric({ color = "#e8e8f0", label, value }: { color?: string; label: string; value: string }) {
  return (
    <div style={metricStyle}>
      <span>{label}</span>
      <strong style={{ color }}>{value}</strong>
    </div>
  );
}

function EquityChart({ points }: { points: Array<{ date: string; value: number }> }) {
  return (
    <svg style={svgStyle} viewBox="0 0 640 180" role="img" aria-label="Equity curve">
      <ChartGrid />
      <polyline fill="none" points={polyline(points.map((point) => point.value), 640, 180)} stroke="#00d4aa" strokeWidth="2.5" />
      <text fill="#8a94a6" fontSize="10" x="12" y="18">
        Equity Curve
      </text>
    </svg>
  );
}

function DrawdownChart({ points }: { points: Array<{ date: string; drawdown_pct: number }> }) {
  return (
    <svg style={svgStyle} viewBox="0 0 640 180" role="img" aria-label="Drawdown curve">
      <ChartGrid />
      <polyline fill="none" points={polyline(points.map((point) => point.drawdown_pct), 640, 180)} stroke="#ff4444" strokeWidth="2.5" />
      <text fill="#8a94a6" fontSize="10" x="12" y="18">
        Drawdown
      </text>
    </svg>
  );
}

function ChartGrid() {
  return (
    <>
      <rect fill="#101827" height="180" width="640" />
      {[0, 1, 2, 3].map((index) => (
        <line key={index} stroke="#243044" strokeWidth="1" x1="0" x2="640" y1={30 + index * 40} y2={30 + index * 40} />
      ))}
    </>
  );
}

function MonthlyHeatmap({ rows }: { rows: Array<{ period: string; return_pct: number }> }) {
  return (
    <div style={heatmapStyle}>
      {rows.slice(-36).map((row) => (
        <div key={row.period} style={{ ...monthCellStyle, borderColor: signedColor(row.return_pct), color: signedColor(row.return_pct) }}>
          <span>{row.period}</span>
          <strong>{formatSignedPct(row.return_pct)}</strong>
        </div>
      ))}
      {rows.length === 0 ? <div style={emptyStyle}>No monthly rows.</div> : null}
    </div>
  );
}

function TradeLog({ trades }: { trades: TradeRow[] }) {
  return (
    <div style={tradeTableStyle}>
      <div style={tradeHeaderStyle}>
        <span>Entry</span>
        <span>Exit</span>
        <span>Net</span>
        <span>Costs</span>
        <span>Reason</span>
        <span>Regime</span>
      </div>
      {trades.slice(-80).reverse().map((trade) => (
        <div key={trade.trade_id} style={tradeRowStyle}>
          <span>{shortDate(trade.entry_date)}</span>
          <span>{shortDate(trade.exit_date)}</span>
          <span style={{ color: signedColor(trade.net_pnl) }}>{formatCurrency(trade.net_pnl)}</span>
          <span>{formatCurrency(trade.costs)}</span>
          <span>{trade.reason ?? "--"}</span>
          <span>{trade.regime_at_entry ?? "--"}</span>
        </div>
      ))}
      {trades.length === 0 ? <div style={emptyStyle}>No trades.</div> : null}
    </div>
  );
}

function AnalysisColumn({ result }: { result: BacktestResult }) {
  const regimes = result.regime_analysis?.by_vol_regime ?? {};
  return (
    <aside style={analysisStyle}>
      <Section title="Regime">
        {Object.entries(regimes).map(([name, row]) => (
          <DataLine key={name} label={name} value={`${formatUnitPct(row.win_rate)} | ${formatCurrency(row.avg_pnl)}`} />
        ))}
        <DataLine label="Best" value={result.regime_analysis?.best_regime ?? "--"} />
        <DataLine label="Worst" value={result.regime_analysis?.worst_regime ?? "--"} />
      </Section>
      <Section title="Chronological Segments">
        <DataLine label="History Sharpe" value={formatNumber(result.chronological_segments?.historical_sharpe)} />
        <DataLine label="Subsequent Sharpe" value={formatNumber(result.chronological_segments?.subsequent_sharpe)} />
        <DataLine label="Difference" value={formatNumber(result.chronological_segments?.difference_pct)} />
        <DataLine label="Verdict" value={result.chronological_segments?.verdict ?? "--"} />
      </Section>
      <Section title="Monte Carlo">
        <DataLine label="P5" value={formatSignedPct(result.monte_carlo?.return_p5)} />
        <DataLine label="P50" value={formatSignedPct(result.monte_carlo?.return_p50)} />
        <DataLine label="P95" value={formatSignedPct(result.monte_carlo?.return_p95)} />
        <DataLine label="Ruin" value={formatSignedPct(result.monte_carlo?.probability_of_ruin)} />
      </Section>
      <Section title="Groq Analysis">
        <p style={reportStyle}>{result.groq_report ?? "Report pending."}</p>
      </Section>
      {(result.warnings ?? []).length ? (
        <Section title="Warnings">
          {(result.warnings ?? []).map((warning) => (
            <div key={warning} style={warningStyle}>{warning}</div>
          ))}
        </Section>
      ) : null}
    </aside>
  );
}

function Section({ children, title }: { children: ReactNode; title: string }) {
  return (
    <div style={sectionStyle}>
      <span style={sectionTitleStyle}>{title}</span>
      {children}
    </div>
  );
}

function DataLine({ label, value }: { label: string; value: string }) {
  return (
    <div style={dataLineStyle}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function polyline(values: number[], width: number, height: number) {
  if (values.length === 0) {
    return "";
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1e-9);
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * (width - 16) + 8;
      const y = height - 12 - ((value - min) / range) * (height - 28);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function labelView(view: View) {
  return view === "equity" ? "Equity Curve" : view === "heatmap" ? "Monthly Heatmap" : view === "drawdown" ? "Drawdown" : "Trade Log";
}

function formatNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? numberFormatter.format(value) : "--";
}

function formatSignedPct(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(2)}%` : "--";
}

function formatUnitPct(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";
}

function formatCurrency(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `Rs ${integerFormatter.format(value)}` : "--";
}

function shortDate(value: string | undefined) {
  return value ? value.slice(5) : "--";
}

function signedColor(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) < 1e-9) {
    return "#ffd700";
  }
  return value > 0 ? "#00d4aa" : "#ff4444";
}

function scoreColor(value: number | null | undefined, threshold: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "#8a94a6";
  }
  return value >= threshold ? "#00d4aa" : "#ffd700";
}

const wrapperStyle = {
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 10,
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const controlBandStyle = {
  alignItems: "end",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "180px minmax(180px, 1.3fr) repeat(9, minmax(72px, 0.65fr)) 120px",
  minWidth: 1180,
} satisfies CSSProperties;

const titleBlockStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  minWidth: 0,
} satisfies CSSProperties;

const eyebrowStyle = {
  color: "#6666aa",
  fontSize: 10,
  lineHeight: 1,
} satisfies CSSProperties;

const titleStyle = {
  color: "#f8fafc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 13,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const fieldStyle = {
  color: "#8a94a6",
  display: "flex",
  flexDirection: "column",
  fontSize: 10,
  gap: 4,
  minWidth: 0,
  textTransform: "uppercase",
} satisfies CSSProperties;

const inputStyle = {
  background: "#101827",
  border: "1px solid #263044",
  borderRadius: 6,
  color: "#e8e8f0",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  height: 30,
  minWidth: 0,
  padding: "0 8px",
} satisfies CSSProperties;

const toggleStyle = {
  alignItems: "center",
  background: "#101827",
  border: "1px solid #263044",
  borderRadius: 6,
  color: "#8a94a6",
  display: "flex",
  fontSize: 10,
  gap: 6,
  height: 30,
  justifyContent: "center",
} satisfies CSSProperties;

function runButtonStyle(running: boolean): CSSProperties {
  return {
    background: running ? "#263044" : "#00d4aa",
    border: 0,
    borderRadius: 6,
    color: running ? "#8a94a6" : "#041110",
    cursor: running ? "default" : "pointer",
    fontSize: 11,
    fontWeight: 800,
    height: 30,
  };
}

const progressStyle = {
  alignItems: "center",
  color: "#8a94a6",
  display: "grid",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "180px 140px minmax(0, 1fr)",
  minWidth: 0,
} satisfies CSSProperties;

const progressTrackStyle = {
  background: "#101827",
  borderRadius: 4,
  height: 7,
  overflow: "hidden",
} satisfies CSSProperties;

const progressFillStyle = {
  background: "#00d4aa",
  height: "100%",
  transition: "width 160ms ease",
} satisfies CSSProperties;

const messageStyle = {
  color: "#ffd700",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const metricStripStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(8, minmax(90px, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const metricStyle = {
  background: "#12121f",
  border: "1px solid #263044",
  borderRadius: 6,
  display: "flex",
  flexDirection: "column",
  gap: 4,
  minHeight: 42,
  minWidth: 0,
  padding: "7px 8px",
} satisfies CSSProperties;

const tabBarStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(4, minmax(100px, 150px)) minmax(0, 1fr)",
} satisfies CSSProperties;

function tabStyle(active: boolean): CSSProperties {
  return {
    background: active ? "#1a1a2e" : "#101827",
    border: "1px solid #263044",
    color: active ? "#f8fafc" : "#8a94a6",
    cursor: "pointer",
    fontSize: 11,
    height: 30,
  };
}

const resultGridStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "minmax(520px, 1fr) 360px",
  minWidth: 0,
} satisfies CSSProperties;

const chartShellStyle = {
  background: "#101827",
  border: "1px solid #263044",
  borderRadius: 8,
  minHeight: 210,
  minWidth: 0,
  overflow: "hidden",
} satisfies CSSProperties;

const svgStyle = {
  display: "block",
  height: 220,
  width: "100%",
} satisfies CSSProperties;

const analysisStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  minWidth: 0,
} satisfies CSSProperties;

const sectionStyle = {
  background: "#12121f",
  border: "1px solid #263044",
  borderRadius: 8,
  display: "flex",
  flexDirection: "column",
  gap: 5,
  padding: 9,
} satisfies CSSProperties;

const sectionTitleStyle = {
  color: "#8a94a6",
  fontSize: 10,
  textTransform: "uppercase",
} satisfies CSSProperties;

const dataLineStyle = {
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "minmax(0, 1fr) auto",
} satisfies CSSProperties;

const reportStyle = {
  color: "#d1d4dc",
  fontSize: 11,
  lineHeight: 1.45,
  margin: 0,
  maxHeight: 160,
  overflowY: "auto",
} satisfies CSSProperties;

const warningStyle = {
  color: "#ffd700",
  fontSize: 11,
} satisfies CSSProperties;

const invalidResultStyle = {
  background: "#3a1010",
  border: "1px solid #ff4444",
  borderRadius: 6,
  color: "#ffb3b3",
  fontSize: 12,
  fontWeight: 800,
  padding: "10px 12px",
} satisfies CSSProperties;

const heatmapStyle = {
  display: "grid",
  gap: 7,
  gridTemplateColumns: "repeat(auto-fill, minmax(88px, 1fr))",
  padding: 10,
} satisfies CSSProperties;

const monthCellStyle = {
  background: "#12121f",
  border: "1px solid",
  borderRadius: 6,
  display: "flex",
  flexDirection: "column",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  gap: 4,
  minHeight: 42,
  padding: 7,
} satisfies CSSProperties;

const tradeTableStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  maxHeight: 220,
  overflowY: "auto",
  padding: 10,
} satisfies CSSProperties;

const tradeHeaderStyle = {
  color: "#8a94a6",
  display: "grid",
  fontSize: 10,
  gridTemplateColumns: "80px 80px 95px 85px 1fr 95px",
  padding: "0 8px",
  textTransform: "uppercase",
} satisfies CSSProperties;

const tradeRowStyle = {
  background: "#12121f",
  border: "1px solid #263044",
  borderRadius: 6,
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "80px 80px 95px 85px 1fr 95px",
  minHeight: 28,
  padding: "6px 8px",
} satisfies CSSProperties;

const emptyStyle = {
  color: "#8a94a6",
  fontSize: 12,
  padding: 12,
} satisfies CSSProperties;
