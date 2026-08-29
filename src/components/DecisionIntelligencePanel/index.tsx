import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";

import { MLPanel } from "../MLPanel";

type DecisionIntelligence = {
  symbol?: string;
  spot?: number;
  computed_at?: number;
  apex_strip?: {
    spot?: number;
    regime?: string | null;
    regime_probability?: number | null;
    anomaly?: string | null;
    ml_direction?: string | null;
    ml_confidence?: number | null;
    tft_1d?: number | null;
    var_per_lot?: number | null;
    next_event?: string | null;
    next_event_days?: number | null;
    observation?: string | null;
  };
  ml_consensus?: {
    direction?: string | null;
    confidence?: number | null;
    uncertainty_level?: string | null;
    model_agreement?: number | null;
    conformal_interval_90?: Array<number | null>;
    tft?: Record<string, number | null | undefined>;
    prophet_regime?: string | null;
    prophet_trend_strength?: number | null;
    wavelet_dominant_frequency?: string | null;
    bayesian_uncertainty?: {
      epistemic?: number | null;
      aleatoric?: number | null;
    };
  };
  patterns?: {
    active_patterns?: PatternRow[];
    patterns?: PatternRow[];
  };
  action_intelligence?: {
    trade?: boolean;
    action?: string | null;
    strategy?: string | null;
    size?: number | null;
    entry_level?: number | null;
    stop_loss_level?: number | null;
    target_level?: number | null;
    risk_reward_ratio?: number | null;
    maximum_notional_loss?: number | null;
    kelly_fraction?: number | null;
    recommended_fraction?: number | null;
    reasoning?: string[];
    eligibility?: {
      eligible?: boolean;
      state?: string;
      reasons?: string[];
    };
  };
  market_brief?: {
    brief?: string;
    generated_at?: string;
    model?: string;
    groq_live?: boolean;
  };
  risk_dashboard?: {
    var?: {
      methods?: RiskVarMethod[];
    };
    greeks?: {
      templates?: Record<string, GreekTemplate>;
    };
    drawdown?: Record<string, number | string | null | undefined>;
    correlations?: {
      contagion_signal?: boolean;
      correlation_regime_change?: boolean;
      reason?: string;
    };
    stress_test?: {
      top_risks?: StressRow[];
    };
  };
  model_grid?: ModelGridRow[];
  error?: string;
};

type PatternRow = {
  pattern_name?: string;
  match_score?: number | null;
  historical_win_rate?: number | null;
  average_move_5d?: number | null;
  optimal_strategy?: string | null;
};

type RiskVarMethod = {
  method?: string;
  var_95_1d?: number | null;
  var_99_1d?: number | null;
  cvar_95_1d?: number | null;
  var_95_5d?: number | null;
};

type GreekTemplate = {
  net_greeks?: Record<string, number | null>;
  theta_decay_today?: number | null;
  break_even_range?: number[];
};

type StressRow = {
  scenario?: string;
  impact_per_lot?: number | null;
  similarity_score?: number | null;
};

type ModelGridRow = {
  model_id?: string;
  model?: string;
  prediction?: string | null;
  confidence?: number | null;
  accuracy_7d?: number | null;
  accuracy_30d?: number | null;
  status?: string;
  reason?: string | null;
};

type Props = {
  isLive: boolean;
  priceText: string;
};

const percentFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
  style: "percent",
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function DecisionIntelligencePanel({ isLive, priceText }: Props) {
  const [data, setData] = useState<DecisionIntelligence | null>(null);
  const [message, setMessage] = useState("");
  const [riskOpen, setRiskOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const result = await invoke<DecisionIntelligence>("get_decision_intelligence");
        if (!active) {
          return;
        }
        setData(result);
        setMessage(result.error ?? "");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };

    load();
    const timer = window.setInterval(load, 60_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const strip = data?.apex_strip;
  const consensus = data?.ml_consensus;
  const action = data?.action_intelligence;
  const actionEligible = action?.trade === true && action?.eligibility?.eligible === true;
  const displayedAction = actionEligible ? action?.action ?? "NON-ACTIONABLE" : "NON-ACTIONABLE";
  const activePatterns = useMemo(
    () => (data?.patterns?.active_patterns?.length ? data.patterns.active_patterns : data?.patterns?.patterns ?? []).slice(0, 4),
    [data],
  );

  if (!data && message) {
    return (
      <section style={wrapperStyle} aria-label="Decision intelligence fallback">
        <ApexStrip isLive={isLive} message={message} priceText={priceText} strip={strip} />
        <MLPanel />
      </section>
    );
  }

  return (
    <section style={wrapperStyle} aria-label="Decision intelligence">
      <ApexStrip isLive={isLive} message={message} priceText={priceText} strip={strip} />

      <div style={decisionGridStyle}>
        <Panel title="ML Consensus">
          <div style={directionRowStyle}>
            <strong style={{ ...directionStyle, color: directionColor(consensus?.direction) }}>
              {consensus?.direction ?? "--"}
            </strong>
            <Metric label="Confidence" value={formatPercent(consensus?.confidence)} color={confidenceColor(consensus?.confidence)} />
            <Metric label="Uncertainty" value={consensus?.uncertainty_level ?? "--"} color={uncertaintyColor(consensus?.uncertainty_level)} />
          </div>
          <div style={metricGridStyle}>
            <Metric label="Agreement" value={formatPercent(consensus?.model_agreement)} />
            <Metric label="90% CI" value={formatInterval(consensus?.conformal_interval_90)} />
            <Metric label="TFT 1D" value={formatSignedPct(consensus?.tft?.return_p50_1d)} />
            <Metric label="TFT 3D" value={formatSignedPct(consensus?.tft?.return_p50_3d)} />
            <Metric label="Prophet" value={consensus?.prophet_regime ?? "--"} />
            <Metric label="Wavelet" value={shortText(consensus?.wavelet_dominant_frequency)} />
          </div>
          <div style={splitLineStyle}>
            <span>Epistemic {formatDecimal(consensus?.bayesian_uncertainty?.epistemic, 2)}</span>
            <span>Aleatoric {formatDecimal(consensus?.bayesian_uncertainty?.aleatoric, 2)}</span>
          </div>
        </Panel>

        <Panel title="Active Patterns">
          <div style={patternListStyle}>
            {activePatterns.length ? (
              activePatterns.map((pattern) => (
                <div key={pattern.pattern_name} style={patternRowStyle}>
                  <div style={patternNameStyle}>{pattern.pattern_name ?? "--"}</div>
                  <div style={patternMetaStyle}>
                    <span>{formatScore(pattern.match_score)}</span>
                    <span>{pattern.optimal_strategy ?? "STAY_FLAT"}</span>
                  </div>
                </div>
              ))
            ) : (
              <EmptyLine value="No high-score pattern" />
            )}
          </div>
          <Metric label="Regime" value={strip?.regime ?? "--"} color={regimeColor(strip?.regime)} />
        </Panel>

        <Panel title="Action Intelligence">
          <div style={{ ...actionStyle, color: actionColor(displayedAction) }}>
            {displayedAction}
          </div>
          <div style={metricGridStyle}>
            <Metric label="Strategy" value={actionEligible ? action?.strategy ?? "--" : "--"} />
            <Metric label="Validation" value={action?.eligibility?.state ?? "UNVALIDATED"} />
            <Metric label="Size" value={actionEligible ? formatSize(action?.size) : "--"} />
            <Metric label="Entry" value={formatPrice(action?.entry_level)} />
            <Metric label="Stop" value={formatPrice(action?.stop_loss_level)} />
            <Metric label="Target" value={formatPrice(action?.target_level)} />
            <Metric label="Max Loss" value={formatCurrency(action?.maximum_notional_loss)} color="#ef5350" />
          </div>
          <div style={reasonListStyle}>
            {(action?.reasoning ?? []).slice(0, 3).map((reason) => (
              <div key={reason} style={reasonStyle}>{reason}</div>
            ))}
          </div>
        </Panel>
      </div>

      <section style={briefStyle} aria-label="Groq market brief">
        <div style={briefHeaderStyle}>
          <span style={sectionTitleStyle}>Groq Market Brief</span>
          <span style={timestampStyle}>
            {data?.market_brief?.model ?? "template"} · {formatBriefTime(data?.market_brief?.generated_at)}
          </span>
        </div>
        <p style={briefTextStyle}>{data?.market_brief?.brief ?? "Market brief loading."}</p>
      </section>

      <Collapsible open={riskOpen} setOpen={setRiskOpen} title="Risk Dashboard">
        <RiskDashboard data={data?.risk_dashboard} />
      </Collapsible>

      <Collapsible open={modelsOpen} setOpen={setModelsOpen} title="Full Model Grid">
        <ModelGrid rows={data?.model_grid ?? []} />
      </Collapsible>
    </section>
  );
}

function ApexStrip({
  isLive,
  message,
  priceText,
  strip,
}: {
  isLive: boolean;
  message: string;
  priceText: string;
  strip?: DecisionIntelligence["apex_strip"];
}) {
  return (
    <div style={stripStyle}>
      <div style={priceBlockStyle}>
        <div style={priceStyle}>{formatPrice(strip?.spot) || priceText}</div>
        <div style={liveStyle}>
          <span
            aria-label={isLive ? "live data" : "no data"}
            style={{ ...dotStyle, background: isLive ? "#22c55e" : "#ef4444" }}
          />
          NIFTY 50
        </div>
      </div>
      <div style={stripMetricsStyle}>
        <StripItem label="REGIME" value={`${strip?.regime ?? "--"} ${formatParenPercent(strip?.regime_probability)}`} color={regimeColor(strip?.regime)} />
        <StripItem label="ANOMALY" value={strip?.anomaly ?? "--"} color={strip?.anomaly === "NORMAL" ? "#22c55e" : "#ef5350"} />
        <StripItem label="ML" value={`${strip?.ml_direction ?? "--"} ${formatPercent(strip?.ml_confidence)}`} color={directionColor(strip?.ml_direction)} />
        <StripItem label="TFT" value={formatSignedPct(strip?.tft_1d)} />
        <StripItem label="VaR/lot" value={formatCurrency(strip?.var_per_lot)} color="#facc15" />
        <StripItem label="Event" value={formatEvent(strip?.next_event, strip?.next_event_days)} />
      </div>
      <div style={observationStyle}>{message || strip?.observation || "Apex intelligence loading."}</div>
    </div>
  );
}

function StripItem({ color = "#d1d4dc", label, value }: { color?: string; label: string; value: string }) {
  return (
    <div style={stripItemStyle}>
      <span style={stripLabelStyle}>{label}</span>
      <strong style={{ color }}>{value}</strong>
    </div>
  );
}

function Panel({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section style={panelStyle}>
      <div style={sectionTitleStyle}>{title}</div>
      {children}
    </section>
  );
}

function Metric({ color = "#f8fafc", label, value }: { color?: string; label: string; value: string }) {
  return (
    <div style={metricStyle}>
      <div style={metricLabelStyle}>{label}</div>
      <div style={{ ...metricValueStyle, color }}>{value}</div>
    </div>
  );
}

function Collapsible({
  children,
  open,
  setOpen,
  title,
}: {
  children: ReactNode;
  open: boolean;
  setOpen: (open: boolean) => void;
  title: string;
}) {
  return (
    <section style={collapsibleStyle}>
      <button style={collapseButtonStyle} type="button" onClick={() => setOpen(!open)}>
        <span>{open ? "-" : "+"}</span>
        <span>{title}</span>
      </button>
      {open ? <div style={collapseBodyStyle}>{children}</div> : null}
    </section>
  );
}

function RiskDashboard({ data }: { data?: DecisionIntelligence["risk_dashboard"] }) {
  const methods = data?.var?.methods ?? [];
  const templates = data?.greeks?.templates ?? {};
  const topRisks = data?.stress_test?.top_risks ?? [];
  return (
    <div style={riskGridStyle}>
      <div>
        <div style={subheadStyle}>VaR</div>
        <div style={tableStyle}>
          {methods.map((method) => (
            <div key={method.method} style={tableRowStyle}>
              <span>{method.method}</span>
              <span>{formatCurrency(method.var_95_1d)}</span>
              <span>{formatCurrency(method.cvar_95_1d)}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <div style={subheadStyle}>Greeks</div>
        <div style={tableStyle}>
          {Object.entries(templates).slice(0, 4).map(([name, template]) => (
            <div key={name} style={tableRowStyle}>
              <span>{labelCase(name)}</span>
              <span>Δ {formatDecimal(template.net_greeks?.delta, 2)}</span>
              <span>Θ {formatCurrency(template.theta_decay_today)}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <div style={subheadStyle}>Stress</div>
        <div style={tableStyle}>
          {topRisks.map((risk) => (
            <div key={risk.scenario} style={tableRowStyle}>
              <span>{risk.scenario}</span>
              <span>{formatCurrency(risk.impact_per_lot)}</span>
              <span>{formatScore(risk.similarity_score)}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <div style={subheadStyle}>Drawdown / Corr</div>
        <div style={miniStackStyle}>
          <Metric label="Current DD" value={formatSignedPct(data?.drawdown?.current_drawdown_pct as number | null | undefined)} />
          <Metric label="Max DD" value={formatSignedPct(data?.drawdown?.max_drawdown_pct as number | null | undefined)} />
          <Metric label="Contagion" value={data?.correlations?.contagion_signal ? "ON" : "OFF"} color={data?.correlations?.contagion_signal ? "#ef5350" : "#22c55e"} />
        </div>
      </div>
    </div>
  );
}

function ModelGrid({ rows }: { rows: ModelGridRow[] }) {
  return (
    <div style={modelTableStyle}>
      <div style={modelHeaderStyle}>
        <span>Model</span>
        <span>Prediction</span>
        <span>Conf</span>
        <span>7d Acc</span>
        <span>30d Acc</span>
        <span>Status</span>
      </div>
      {rows.map((row) => (
        <div key={row.model_id ?? row.model} style={modelRowStyle} title={row.reason ?? undefined}>
          <span style={modelNameStyle}>{row.model}</span>
          <span style={{ color: directionColor(row.prediction) }}>{row.prediction ?? "--"}</span>
          <span>{formatPercent(row.confidence)}</span>
          <span>{formatPercent(row.accuracy_7d)}</span>
          <span>{formatPercent(row.accuracy_30d)}</span>
          <span style={{ color: row.status === "Trained" ? "#22c55e" : "#facc15" }}>{row.status ?? "Fallback"}</span>
        </div>
      ))}
    </div>
  );
}

function EmptyLine({ value }: { value: string }) {
  return <div style={emptyStyle}>{value}</div>;
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatPercent(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : percentFormatter.format(Math.max(0, Math.min(1, number)));
}

function formatParenPercent(value: number | null | undefined) {
  const formatted = formatPercent(value);
  return formatted === "--" ? "" : `(${formatted})`;
}

function formatSignedPct(value: number | string | null | undefined) {
  const number = typeof value === "string" ? Number(value) : asNumber(value);
  if (number === null || !Number.isFinite(number)) {
    return "--";
  }
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}%`;
}

function formatDecimal(value: number | null | undefined, digits: number) {
  const number = asNumber(value);
  return number === null ? "--" : number.toFixed(digits);
}

function formatCurrency(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return `₹${integerFormatter.format(Math.abs(number))}`;
}

function formatPrice(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null || number <= 0) {
    return "";
  }
  return `₹${integerFormatter.format(number)}`;
}

function formatInterval(values: Array<number | null> | undefined) {
  if (!values || values.length < 2) {
    return "--";
  }
  return `${formatSignedPct(values[0])} to ${formatSignedPct(values[1])}`;
}

function formatSize(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `${(number * 100).toFixed(1)}%`;
}

function formatScore(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `${Math.round(number)}%`;
}

function formatBriefTime(value: string | undefined) {
  if (!value) {
    return "--";
  }
  return new Date(value).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

function formatEvent(name: string | null | undefined, days: number | null | undefined) {
  if (!name) {
    return "--";
  }
  const suffix = typeof days === "number" ? ` in ${days}d` : "";
  return `${name}${suffix}`;
}

function shortText(value: string | null | undefined) {
  if (!value) {
    return "--";
  }
  return value.replace(/_/g, " ");
}

function labelCase(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function directionColor(value: string | null | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("UP") || text.includes("LONG") || text.includes("CALM") || text.includes("NORMAL")) {
    return "#22c55e";
  }
  if (text.includes("DOWN") || text.includes("SHORT") || text.includes("PANIC") || text.includes("FEAR")) {
    return "#ef5350";
  }
  return "#facc15";
}

function confidenceColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number >= 0.6) {
    return "#22c55e";
  }
  if (number >= 0.4) {
    return "#facc15";
  }
  return "#ef5350";
}

function uncertaintyColor(value: string | null | undefined) {
  if (value === "LOW") {
    return "#22c55e";
  }
  if (value === "HIGH") {
    return "#ef5350";
  }
  return "#facc15";
}

function regimeColor(value: string | null | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("CALM") || text.includes("GREED")) {
    return "#22c55e";
  }
  if (text.includes("PANIC") || text.includes("FEAR")) {
    return "#ef5350";
  }
  return "#facc15";
}

function actionColor(value: string | null | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("LONG")) {
    return "#22c55e";
  }
  if (text.includes("SHORT")) {
    return "#ef5350";
  }
  return "#facc15";
}

const wrapperStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const stripStyle = {
  alignItems: "stretch",
  background: "#111827",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  display: "grid",
  gap: 12,
  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
  minHeight: 96,
  minWidth: 0,
  padding: 14,
} satisfies CSSProperties;

const priceBlockStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  justifyContent: "center",
  minWidth: 0,
} satisfies CSSProperties;

const priceStyle = {
  color: "#f8fafc",
  fontSize: 34,
  fontWeight: 800,
  letterSpacing: 0,
  lineHeight: 1,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const liveStyle = {
  alignItems: "center",
  color: "#9ca3af",
  display: "flex",
  fontSize: 12,
  gap: 8,
  lineHeight: 1.2,
} satisfies CSSProperties;

const dotStyle = {
  borderRadius: "50%",
  display: "inline-block",
  height: 9,
  width: 9,
} satisfies CSSProperties;

const stripMetricsStyle = {
  alignItems: "stretch",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(auto-fit, minmax(118px, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const stripItemStyle = {
  background: "#0f172a",
  border: "1px solid #263044",
  borderRadius: 6,
  display: "flex",
  flexDirection: "column",
  gap: 4,
  justifyContent: "center",
  minHeight: 32,
  minWidth: 0,
  padding: "7px 9px",
} satisfies CSSProperties;

const stripLabelStyle = {
  color: "#8a94a6",
  fontSize: 10,
  lineHeight: 1,
} satisfies CSSProperties;

const observationStyle = {
  alignItems: "center",
  color: "#d1d4dc",
  display: "flex",
  fontSize: 13,
  lineHeight: 1.45,
  minWidth: 0,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const decisionGridStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const panelStyle = {
  background: "#171b2a",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 10,
  minHeight: 255,
  minWidth: 0,
  padding: 12,
} satisfies CSSProperties;

const sectionTitleStyle = {
  color: "#8a94a6",
  fontSize: 11,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const directionRowStyle = {
  alignItems: "stretch",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "1.2fr 1fr 1fr",
} satisfies CSSProperties;

const directionStyle = {
  alignItems: "center",
  display: "flex",
  fontSize: 34,
  lineHeight: 1,
  minHeight: 58,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const metricGridStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
} satisfies CSSProperties;

const metricStyle = {
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  minHeight: 48,
  minWidth: 0,
  padding: "8px",
} satisfies CSSProperties;

const metricLabelStyle = {
  color: "#8a94a6",
  fontSize: 10,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const metricValueStyle = {
  fontSize: 14,
  fontWeight: 700,
  lineHeight: 1.25,
  minHeight: 18,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const splitLineStyle = {
  color: "#9ca3af",
  display: "flex",
  fontSize: 12,
  gap: 12,
  justifyContent: "space-between",
} satisfies CSSProperties;

const patternListStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 7,
  minHeight: 164,
} satisfies CSSProperties;

const patternRowStyle = {
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  display: "grid",
  gap: 6,
  gridTemplateColumns: "minmax(0, 1fr) auto",
  minHeight: 34,
  padding: "7px 8px",
} satisfies CSSProperties;

const patternNameStyle = {
  color: "#f8fafc",
  fontSize: 12,
  fontWeight: 700,
  lineHeight: 1.25,
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const patternMetaStyle = {
  color: "#facc15",
  display: "flex",
  fontSize: 11,
  gap: 8,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const actionStyle = {
  fontSize: 28,
  fontWeight: 800,
  lineHeight: 1,
  minHeight: 34,
} satisfies CSSProperties;

const reasonListStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
} satisfies CSSProperties;

const reasonStyle = {
  color: "#cbd5e1",
  fontSize: 11,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const briefStyle = {
  background: "#101827",
  border: "1px solid #263044",
  borderRadius: 8,
  padding: "12px 14px",
} satisfies CSSProperties;

const briefHeaderStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
  gap: 12,
} satisfies CSSProperties;

const timestampStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.2,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const briefTextStyle = {
  color: "#d1d4dc",
  fontSize: 13,
  lineHeight: 1.55,
  margin: "8px 0 0",
} satisfies CSSProperties;

const collapsibleStyle = {
  background: "#151926",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  overflow: "hidden",
} satisfies CSSProperties;

const collapseButtonStyle = {
  alignItems: "center",
  background: "transparent",
  border: 0,
  color: "#d1d4dc",
  cursor: "pointer",
  display: "flex",
  fontFamily: "monospace",
  fontSize: 12,
  fontWeight: 700,
  gap: 8,
  minHeight: 38,
  padding: "0 12px",
  width: "100%",
} satisfies CSSProperties;

const collapseBodyStyle = {
  borderTop: "1px solid #2a2e39",
  padding: 12,
} satisfies CSSProperties;

const riskGridStyle = {
  display: "grid",
  gap: 12,
  gridTemplateColumns: "repeat(4, minmax(200px, 1fr))",
  overflowX: "auto",
} satisfies CSSProperties;

const subheadStyle = {
  color: "#8a94a6",
  fontSize: 11,
  marginBottom: 8,
  textTransform: "uppercase",
} satisfies CSSProperties;

const tableStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
} satisfies CSSProperties;

const tableRowStyle = {
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  color: "#d1d4dc",
  display: "grid",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "minmax(0, 1fr) auto auto",
  minHeight: 30,
  padding: "7px 8px",
} satisfies CSSProperties;

const miniStackStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "1fr",
} satisfies CSSProperties;

const modelTableStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
  minWidth: 720,
  overflowX: "auto",
} satisfies CSSProperties;

const modelHeaderStyle = {
  color: "#8a94a6",
  display: "grid",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "1.2fr 1fr 0.55fr 0.55fr 0.65fr 0.65fr",
  padding: "0 8px 4px",
  textTransform: "uppercase",
} satisfies CSSProperties;

const modelRowStyle = {
  alignItems: "center",
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  color: "#d1d4dc",
  display: "grid",
  fontSize: 12,
  gap: 8,
  gridTemplateColumns: "1.2fr 1fr 0.55fr 0.55fr 0.65fr 0.65fr",
  minHeight: 30,
  padding: "6px 8px",
} satisfies CSSProperties;

const modelNameStyle = {
  color: "#f8fafc",
  fontWeight: 700,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const emptyStyle = {
  color: "#8a94a6",
  fontSize: 12,
  padding: "8px 0",
} satisfies CSSProperties;
