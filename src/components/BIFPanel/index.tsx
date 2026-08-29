import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";

type BMS = {
  bms?: number | null;
  bms_label?: string;
  price_momentum?: number | null;
  options_flow_momentum?: number | null;
  institutional_momentum?: number | null;
  expiry_adjustment?: number | null;
  regime_multiplier?: number | null;
};

type NGI = {
  ngi?: number | null;
  ngi_direction?: string;
  gravity_center?: number | null;
  gravity_strength?: number | null;
  escape_velocity_up?: number | null;
  escape_velocity_down?: number | null;
  pin_probability?: number | null;
};

type IAD = {
  iad_score?: number | null;
  iad_label?: string;
  wyckoff_stage?: string;
  price_volume_score?: number | null;
  fii_flow_score?: number | null;
  options_hedge_score?: number | null;
};

type VRC = {
  clock_position?: number | null;
  clock_label?: string;
  iv_percentile?: number | null;
  iv_trend?: string;
  vrp?: number | null;
  vrp_signal?: string;
  optimal_strategy?: string;
  expected_iv_7d?: number | null;
};

type NFVE = {
  fair_value?: number | null;
  current_vs_fair?: number | null;
  valuation?: string;
  equity_risk_premium?: number | null;
  implied_pe?: number | null;
  currency_impact?: number | null;
  fii_liquidity_premium?: number | null;
  confidence?: string;
};

type CDS = {
  cds?: number | null;
  cds_label?: string;
  cds_confidence?: number | null;
  action?: string;
  action_strength?: string;
  anomaly_gate_active?: boolean;
  component_scores?: Record<string, number>;
  recommendation_state?: "ANALYTICAL_OUTPUT" | "INFORMATIONAL" | "UNVALIDATED" | "VALIDATED_NON_ACTIONABLE" | "ACTIONABLE";
  model_validation_state?: string;
  data_quality_state?: string;
};

type BIFHistoryRow = {
  date?: string;
  cds?: number | null;
  cds_label?: string;
};

type BIFPayload = {
  symbol?: string;
  spot?: number | null;
  expiry?: string;
  computed_at?: string;
  bms?: BMS;
  ngi?: NGI;
  iad?: IAD;
  vrc?: VRC;
  nfve?: NFVE;
  cds?: CDS;
  history?: BIFHistoryRow[];
  error?: string;
};

type BIFPanelProps = {
  symbol?: string;
};

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function BIFPanel({ symbol = "NIFTY" }: BIFPanelProps) {
  const [data, setData] = useState<BIFPayload | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const loadBif = async () => {
      try {
        const result = await invoke<BIFPayload>("get_bif_all", { symbol });
        if (!active) {
          return;
        }
        setData(result);
        setMessage(result.error ?? "");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadBif();
    const timer = window.setInterval(loadBif, 60_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [symbol]);

  const cds = data?.cds;
  const history = data?.history ?? [];
  const trend = useMemo(() => historyTrend(history), [history]);

  if (loading && data === null) {
    return (
      <section style={wrapperStyle} aria-label="BIF loading">
        <div style={loadingStyle}>Loading BIF...</div>
      </section>
    );
  }

  return (
    <section style={wrapperStyle} aria-label="Bharat Intelligence Framework">
      <div style={headerStyle}>
        <span style={titleStyle}>BIF PANEL</span>
        <span style={timestampStyle}>Updated: {formatTime(data?.computed_at)}</span>
      </div>

      <div style={cdsCardStyle(cds?.anomaly_gate_active)}>
        <div style={cdsTopStyle}>
          <div>
            <span style={labelStyle}>Composite Decision Score</span>
            <strong style={{ ...cdsValueStyle, color: scoreColor(cds?.cds) }}>
              {formatScore(cds?.cds)}
            </strong>
          </div>
          <div style={actionBlockStyle}>
            <span style={{ ...labelStyle, color: scoreColor(cds?.cds) }}>
              {cds?.cds_label ?? "--"}
            </span>
            <strong style={actionStyle}>
              {cds?.recommendation_state === "ACTIONABLE" ? cds.action : "NON-ACTIONABLE"}
            </strong>
            <span style={subvalueStyle}>{cds?.action_strength ?? "NONE"}</span>
            <span style={subvalueStyle}>{cds?.recommendation_state ?? "UNVALIDATED"}</span>
          </div>
        </div>
        <div style={barTrackStyle}>
          <div
            style={{
              ...barFillStyle,
              background: scoreColor(cds?.cds),
              width: `${clamp(asNumber(cds?.cds) ?? 0, 0, 100)}%`,
            }}
          />
        </div>
        <div style={cdsMetaStyle}>
          <span>Confidence: {formatUnitPct(cds?.cds_confidence)}</span>
          <span>Anomaly Gate: {cds?.anomaly_gate_active ? "ON" : "OFF"}</span>
          <span>Spot: {formatPrice(data?.spot)}</span>
          <span>Expiry: {data?.expiry ?? "--"}</span>
        </div>
      </div>

      <div style={moduleGridStyle}>
        <ModuleCard
          label="BMS"
          title={data?.bms?.bms_label ?? "--"}
          value={formatSigned(data?.bms?.bms)}
          color={signedColor(data?.bms?.bms)}
          rows={[
            ["Price", formatSigned(data?.bms?.price_momentum)],
            ["Options", formatSigned(data?.bms?.options_flow_momentum)],
            ["FII", formatSigned(data?.bms?.institutional_momentum)],
            ["Expiry", formatDecimal(data?.bms?.expiry_adjustment, 2)],
          ]}
        />
        <ModuleCard
          label="NGI"
          title={gravityTitle(data?.ngi)}
          value={formatSigned(data?.ngi?.ngi)}
          color={signedColor(data?.ngi?.ngi)}
          rows={[
            ["Center", formatStrike(data?.ngi?.gravity_center)],
            ["Escape Up", formatPoints(data?.ngi?.escape_velocity_up)],
            ["Escape Dn", formatPoints(data?.ngi?.escape_velocity_down)],
            ["Pin", formatPctValue(data?.ngi?.pin_probability)],
          ]}
        />
        <ModuleCard
          label="IAD"
          title={data?.iad?.iad_label ?? "--"}
          value={formatSigned(data?.iad?.iad_score)}
          color={signedColor(data?.iad?.iad_score)}
          rows={[
            ["PV", formatSigned(data?.iad?.price_volume_score)],
            ["FII", formatSigned(data?.iad?.fii_flow_score)],
            ["Options", formatSigned(data?.iad?.options_hedge_score)],
            ["Stage", data?.iad?.wyckoff_stage ?? "--"],
          ]}
        />
        <div style={cardStyle}>
          <div style={cardTopStyle}>
            <span style={labelStyle}>VRC</span>
            <Clock position={data?.vrc?.clock_position} />
          </div>
          <strong style={{ ...moduleValueStyle, color: vrcColor(data?.vrc?.clock_label) }}>
            {data?.vrc?.clock_label ?? "--"}
          </strong>
          <div style={rowListStyle}>
            <DataRow label="IV Pct" value={formatPctValue(data?.vrc?.iv_percentile)} />
            <DataRow label="Trend" value={data?.vrc?.iv_trend ?? "--"} />
            <DataRow label="VRP" value={formatPctPoints(data?.vrc?.vrp)} />
            <DataRow label="Signal" value={volSignalLabel(data?.vrc?.vrp_signal)} />
          </div>
        </div>
      </div>

      <div style={bottomGridStyle}>
        <div style={cardStyle}>
          <span style={labelStyle}>NFVE</span>
          <div style={nfveGridStyle}>
            <DataRow label="Fair Value" value={formatPrice(data?.nfve?.fair_value)} />
            <DataRow label="Premium" value={formatPctPoints(data?.nfve?.current_vs_fair)} />
            <DataRow label="Valuation" value={data?.nfve?.valuation ?? "--"} />
            <DataRow label="ERP" value={formatPctPoints(data?.nfve?.equity_risk_premium)} />
            <DataRow label="PE" value={formatDecimal(data?.nfve?.implied_pe, 1)} />
            <DataRow label="FX" value={formatDecimal(data?.nfve?.currency_impact, 2)} />
          </div>
        </div>

        <div style={cardStyle}>
          <div style={historyHeaderStyle}>
            <span style={labelStyle}>CDS History</span>
            <span style={{ ...subvalueStyle, color: trend.color }}>{trend.label}</span>
          </div>
          <div style={historyListStyle}>
            {history.slice(0, 5).map((row) => (
              <div key={row.date} style={historyRowStyle}>
                <span>{shortDate(row.date)}</span>
                <strong style={{ color: scoreColor(row.cds) }}>
                  {formatScore(row.cds)}
                </strong>
                <span>{row.cds_label ?? "--"}</span>
              </div>
            ))}
            {history.length === 0 ? <div style={emptyStyle}>No history.</div> : null}
          </div>
        </div>
      </div>

      {message ? <div style={messageStyle}>{message}</div> : null}
    </section>
  );
}

function ModuleCard({
  color,
  label,
  rows,
  title,
  value,
}: {
  color: string;
  label: string;
  rows: Array<[string, string]>;
  title: string;
  value: string;
}) {
  return (
    <div style={cardStyle}>
      <span style={labelStyle}>{label}</span>
      <strong style={{ ...moduleValueStyle, color }}>{value}</strong>
      <span style={moduleTitleStyle}>{title}</span>
      <div style={rowListStyle}>
        {rows.map(([rowLabel, rowValue]) => (
          <DataRow key={rowLabel} label={rowLabel} value={rowValue} />
        ))}
      </div>
    </div>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={dataRowStyle}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Clock({ position }: { position?: number | null }) {
  const degrees = asNumber(position) ?? 0;
  return (
    <svg aria-hidden style={clockStyle} viewBox="0 0 40 40">
      <circle cx="20" cy="20" fill="#101827" r="17" stroke="#35405f" strokeWidth="2" />
      <line
        stroke="#facc15"
        strokeLinecap="round"
        strokeWidth="3"
        transform={`rotate(${degrees} 20 20)`}
        x1="20"
        x2="20"
        y1="20"
        y2="7"
      />
      <circle cx="20" cy="20" fill="#facc15" r="2.5" />
    </svg>
  );
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatScore(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : number.toFixed(1);
}

function formatSigned(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return `${number > 0 ? "+" : ""}${number.toFixed(1)}`;
}

function formatDecimal(value: number | null | undefined, digits: number) {
  const number = asNumber(value);
  return number === null ? "--" : number.toFixed(digits);
}

function formatUnitPct(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `${Math.round(clamp(number, 0, 1) * 100)}%`;
}

function formatPctValue(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `${number.toFixed(0)}%`;
}

function formatPctPoints(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `${number > 0 ? "+" : ""}${number.toFixed(2)}%`;
}

function formatPrice(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `Rs ${integerFormatter.format(number)}`;
}

function formatStrike(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : integerFormatter.format(number);
}

function formatPoints(value: number | null | undefined) {
  const number = asNumber(value);
  return number === null ? "--" : `${numberFormatter.format(number)} pts`;
}

function formatTime(value: string | undefined) {
  if (!value) {
    return "--";
  }
  return new Date(value).toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortDate(value: string | undefined) {
  if (!value) {
    return "--";
  }
  return new Date(value).toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
  });
}

function historyTrend(history: BIFHistoryRow[]) {
  const latest = asNumber(history[0]?.cds);
  const oldest = asNumber(history[Math.min(history.length - 1, 4)]?.cds);
  if (latest === null || oldest === null || history.length < 2) {
    return { color: "#8a94a6", label: "Trend: --" };
  }
  if (latest > oldest) {
    return { color: "#22c55e", label: "Trend: Rising" };
  }
  if (latest < oldest) {
    return { color: "#ef5350", label: "Trend: Falling" };
  }
  return { color: "#facc15", label: "Trend: Flat" };
}

function gravityTitle(ngi: NGI | undefined) {
  const direction = ngi?.ngi_direction ?? "NEUTRAL";
  return direction === "UP"
    ? "UP GRAVITY"
    : direction === "DOWN"
      ? "DOWN GRAVITY"
      : "BALANCED";
}

function volSignalLabel(value: string | undefined) {
  if (value === "BUY_VOL") {
    return "LONG VOL";
  }
  if (value === "SELL_VOL") {
    return "SHORT VOL";
  }
  return value ?? "--";
}

function scoreColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 65) {
    return "#00d4aa";
  }
  if (number >= 45) {
    return "#ffd700";
  }
  return "#ff4444";
}

function signedColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null || Math.abs(number) < 1e-9) {
    return "#ffd700";
  }
  return number > 0 ? "#00d4aa" : "#ff4444";
}

function vrcColor(value: string | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("COMPRESSION") || text.includes("EARLY")) {
    return "#00d4aa";
  }
  if (text.includes("PEAK")) {
    return "#ff4444";
  }
  return "#ffd700";
}

function clamp(value: number, low: number, high: number) {
  return Math.max(low, Math.min(high, value));
}

const wrapperStyle = {
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 10,
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const loadingStyle = {
  color: "#8a94a6",
  fontSize: 12,
} satisfies CSSProperties;

const headerStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
  minWidth: 0,
} satisfies CSSProperties;

const titleStyle = {
  color: "#f8fafc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 14,
  fontWeight: 700,
  lineHeight: 1.2,
} satisfies CSSProperties;

const timestampStyle = {
  color: "#8a94a6",
  fontSize: 11,
  lineHeight: 1.2,
} satisfies CSSProperties;

function cdsCardStyle(anomaly?: boolean): CSSProperties {
  return {
    background: "#171b2a",
    border: `1px solid ${anomaly ? "#f97316" : "#2a2e39"}`,
    borderRadius: 8,
    boxSizing: "border-box",
    display: "grid",
    gap: 10,
    minWidth: 0,
    padding: "12px 14px",
  };
}

const cdsTopStyle = {
  alignItems: "center",
  display: "flex",
  gap: 12,
  justifyContent: "space-between",
  minWidth: 0,
} satisfies CSSProperties;

const labelStyle = {
  color: "#8a94a6",
  fontSize: 11,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const cdsValueStyle = {
  display: "block",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 36,
  fontWeight: 800,
  lineHeight: 1,
  marginTop: 4,
} satisfies CSSProperties;

const actionBlockStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  minWidth: 150,
  textAlign: "right",
} satisfies CSSProperties;

const actionStyle = {
  color: "#f8fafc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 14,
  lineHeight: 1.2,
} satisfies CSSProperties;

const subvalueStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.3,
} satisfies CSSProperties;

const barTrackStyle = {
  background: "#0f172a",
  borderRadius: 5,
  height: 10,
  overflow: "hidden",
} satisfies CSSProperties;

const barFillStyle = {
  height: "100%",
  transition: "width 180ms ease",
} satisfies CSSProperties;

const cdsMetaStyle = {
  color: "#cbd5e1",
  display: "grid",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
  lineHeight: 1.25,
  minWidth: 0,
} satisfies CSSProperties;

const moduleGridStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "repeat(4, minmax(170px, 1fr))",
  minWidth: 0,
  overflowX: "auto",
} satisfies CSSProperties;

const cardStyle = {
  background: "#1a1a2e",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  display: "flex",
  flexDirection: "column",
  gap: 8,
  minHeight: 160,
  minWidth: 170,
  padding: "12px",
} satisfies CSSProperties;

const cardTopStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
} satisfies CSSProperties;

const moduleValueStyle = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 24,
  fontWeight: 800,
  lineHeight: 1,
} satisfies CSSProperties;

const moduleTitleStyle = {
  color: "#f8fafc",
  fontSize: 12,
  fontWeight: 700,
  lineHeight: 1.2,
  minHeight: 15,
} satisfies CSSProperties;

const rowListStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
} satisfies CSSProperties;

const dataRowStyle = {
  color: "#cbd5e1",
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "minmax(0, 0.78fr) minmax(0, 1fr)",
  lineHeight: 1.25,
  minWidth: 0,
} satisfies CSSProperties;

const clockStyle = {
  height: 40,
  width: 40,
} satisfies CSSProperties;

const bottomGridStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "minmax(280px, 0.8fr) minmax(320px, 1.2fr)",
  minWidth: 0,
} satisfies CSSProperties;

const nfveGridStyle = {
  display: "grid",
  gap: 6,
  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const historyHeaderStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
  gap: 10,
} satisfies CSSProperties;

const historyListStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
} satisfies CSSProperties;

const historyRowStyle = {
  alignItems: "center",
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "80px 70px minmax(0, 1fr)",
  minHeight: 28,
  padding: "6px 8px",
} satisfies CSSProperties;

const emptyStyle = {
  color: "#8a94a6",
  fontSize: 12,
} satisfies CSSProperties;

const messageStyle = {
  color: "#facc15",
  fontSize: 12,
  lineHeight: 1.3,
} satisfies CSSProperties;
