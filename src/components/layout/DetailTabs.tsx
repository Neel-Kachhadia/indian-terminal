import {
  lazy,
  Suspense,
  useEffect,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";

import type { StockSnapshot } from "./types";

const BacktestPanel = lazy(() =>
  import("../BacktestPanel").then((module) => ({ default: module.BacktestPanel })),
);
const BIFPanel = lazy(() =>
  import("../BIFPanel").then((module) => ({ default: module.BIFPanel })),
);
const IVSurface = lazy(() =>
  import("../IVSurface").then((module) => ({ default: module.IVSurface })),
);
const MLPanel = lazy(() =>
  import("../MLPanel").then((module) => ({ default: module.MLPanel })),
);
const OptionsChain = lazy(() =>
  import("../OptionsChain").then((module) => ({ default: module.OptionsChain })),
);
const SentimentPanel = lazy(() =>
  import("../SentimentPanel").then((module) => ({ default: module.SentimentPanel })),
);
const SignalsPanel = lazy(() =>
  import("../SignalsPanel").then((module) => ({ default: module.SignalsPanel })),
);

type Tab = "chain" | "iv" | "signals" | "bif" | "backtest" | "risk" | "ml" | "sentiment";

type DetailTabsProps = {
  selectedSymbol: string;
  snapshot: StockSnapshot | null;
};

type RiskPayload = {
  error?: string;
  methods?: Array<{
    method?: string;
    var_95_1d?: number | null;
    var_99_1d?: number | null;
    cvar_95_1d?: number | null;
  }>;
  spot?: number | null;
};

const tabs: Array<{ label: string; value: Tab }> = [
  { label: "Chain", value: "chain" },
  { label: "IV Surface", value: "iv" },
  { label: "Signals", value: "signals" },
  { label: "BIF", value: "bif" },
  { label: "Backtest", value: "backtest" },
  { label: "Risk", value: "risk" },
  { label: "ML Models", value: "ml" },
  { label: "Sentiment", value: "sentiment" },
];

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

export function DetailTabs({ selectedSymbol, snapshot }: DetailTabsProps) {
  const [activeTab, setActiveTab] = useState<Tab>("chain");
  const spot = snapshot?.close ?? undefined;

  return (
    <section style={zoneStyle} aria-label="Detail tabs">
      <div style={tabBarStyle}>
        <div style={labelStyle}>ZONE C</div>
        {tabs.map((tab) => (
          <button
            key={tab.value}
            onClick={() => setActiveTab(tab.value)}
            style={tabStyle(activeTab === tab.value)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={bodyStyle}>
        <Suspense fallback={<div style={messageStyle}>Loading panel...</div>}>
          {activeTab === "chain" ? (
            <OptionsChain symbol={selectedSymbol} spot={spot} />
          ) : null}
          {activeTab === "iv" ? <IVSurface symbol={selectedSymbol} /> : null}
          {activeTab === "signals" ? (
            <SignalsPanel symbol={selectedSymbol} spot={spot} />
          ) : null}
          {activeTab === "bif" ? <BIFPanel symbol={selectedSymbol} /> : null}
          {activeTab === "backtest" ? (
            <BacktestPanel symbol={selectedSymbol} />
          ) : null}
          {activeTab === "risk" ? (
            <RiskPanel selectedSymbol={selectedSymbol} snapshot={snapshot} />
          ) : null}
          {activeTab === "ml" ? <MLPanel symbol={selectedSymbol} /> : null}
          {activeTab === "sentiment" ? <SentimentPanel /> : null}
        </Suspense>
      </div>
    </section>
  );
}

function RiskPanel({
  selectedSymbol,
  snapshot,
}: {
  selectedSymbol: string;
  snapshot: StockSnapshot | null;
}) {
  const [risk, setRisk] = useState<RiskPayload | null>(null);
  const [message, setMessage] = useState("Loading risk...");

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const result = await invoke<RiskPayload>("get_risk_var");
        if (!active) {
          return;
        }
        setRisk(result);
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

  return (
    <div style={riskStyle}>
      <PanelTitle>{selectedSymbol} Risk</PanelTitle>
      <div style={riskGridStyle}>
        <RiskMetric label="Close" value={formatNumber(snapshot?.close)} />
        <RiskMetric label="Day Change" value={formatPercent(snapshot?.change_pct)} />
        <RiskMetric label="Volume Ratio" value={formatRatio(snapshot?.volume_ratio)} />
        <RiskMetric label="52W High Gap" value={formatPercent(snapshot?.week_52_high_pct)} />
      </div>
      <div style={riskTableStyle}>
        <div style={riskHeaderStyle}>
          <span>Method</span>
          <span>VaR 95</span>
          <span>VaR 99</span>
          <span>CVaR 95</span>
        </div>
        {(risk?.methods ?? []).slice(0, 6).map((method) => (
          <div key={method.method} style={riskRowStyle}>
            <span>{method.method ?? "--"}</span>
            <span>{formatCurrency(method.var_95_1d)}</span>
            <span>{formatCurrency(method.var_99_1d)}</span>
            <span>{formatCurrency(method.cvar_95_1d)}</span>
          </div>
        ))}
        {message ? <div style={messageStyle}>{message}</div> : null}
      </div>
    </div>
  );
}

function RiskMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div style={metricStyle}>
      <span style={metricLabelStyle}>{label}</span>
      <strong style={metricValueStyle}>{value}</strong>
    </div>
  );
}

function PanelTitle({ children }: { children: ReactNode }) {
  return <div style={panelTitleStyle}>{children}</div>;
}

function formatNumber(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return numberFormatter.format(value);
}

function formatCurrency(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return `Rs ${numberFormatter.format(Math.abs(value))}`;
}

function formatPercent(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function formatRatio(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return `${value.toFixed(2)}x`;
}

const zoneStyle = {
  background: "#0f0f1a",
  borderTop: "1px solid #1e1e3a",
  boxSizing: "border-box",
  display: "grid",
  gridTemplateRows: "36px minmax(0, 1fr)",
  height: 300,
  minHeight: 0,
  minWidth: 0,
  overflow: "hidden",
} satisfies CSSProperties;

const tabBarStyle = {
  alignItems: "stretch",
  display: "grid",
  gridTemplateColumns: "86px repeat(8, minmax(82px, 122px)) minmax(0, 1fr)",
} satisfies CSSProperties;

const labelStyle = {
  alignItems: "center",
  borderRight: "1px solid #1e1e3a",
  color: "#6666aa",
  display: "flex",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  paddingLeft: 12,
} satisfies CSSProperties;

function tabStyle(active: boolean): CSSProperties {
  return {
    background: active ? "#1a1a2e" : "#0f0f1a",
    border: 0,
    borderRight: "1px solid #1e1e3a",
    color: active ? "#e8e8f0" : "#6666aa",
    cursor: "pointer",
    fontFamily: "Inter, system-ui, sans-serif",
    fontSize: 11,
  };
}

const bodyStyle = {
  minHeight: 0,
  overflow: "auto",
  padding: 10,
} satisfies CSSProperties;

const riskStyle = {
  color: "#e8e8f0",
  display: "grid",
  gap: 10,
  gridTemplateColumns: "220px minmax(440px, 1fr)",
  minWidth: 0,
} satisfies CSSProperties;

const panelTitleStyle = {
  color: "#ffd700",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 14,
  fontWeight: 600,
  gridColumn: "1 / -1",
} satisfies CSSProperties;

const riskGridStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
} satisfies CSSProperties;

const metricStyle = {
  background: "#12121f",
  border: "1px solid #1e1e3a",
  minWidth: 0,
  padding: "8px",
} satisfies CSSProperties;

const metricLabelStyle = {
  color: "#6666aa",
  display: "block",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
} satisfies CSSProperties;

const metricValueStyle = {
  color: "#e8e8f0",
  display: "block",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 13,
  marginTop: 6,
} satisfies CSSProperties;

const riskTableStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  minWidth: 0,
} satisfies CSSProperties;

const riskHeaderStyle = {
  color: "#6666aa",
  display: "grid",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  gridTemplateColumns: "1fr repeat(3, 0.8fr)",
  padding: "0 8px",
} satisfies CSSProperties;

const riskRowStyle = {
  background: "#12121f",
  border: "1px solid #1e1e3a",
  color: "#e8e8f0",
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  gridTemplateColumns: "1fr repeat(3, 0.8fr)",
  minHeight: 28,
  padding: "6px 8px",
} satisfies CSSProperties;

const messageStyle = {
  color: "#ffd700",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
  padding: 8,
} satisfies CSSProperties;
