import {
  lazy,
  Suspense,
  useEffect,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";

import type { StockSnapshot, Tick } from "./types";
import {
  displayQuotePrice,
  quoteDisplayStatus,
  type QuoteDisplayStatus,
} from "../../market/quote";
import { useMarketSession } from "../../market/session";

const ChartPanel = lazy(() =>
  import("../ChartPanel").then((module) => ({ default: module.ChartPanel })),
);
const EquityPanel = lazy(() =>
  import("../EquityPanel").then((module) => ({ default: module.EquityPanel })),
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
const SignalsPanel = lazy(() =>
  import("../SignalsPanel").then((module) => ({ default: module.SignalsPanel })),
);

type DetailTab = "chart" | "ml" | "options" | "brief";

type NiftyDetailProps = {
  selectedSymbol: string;
  snapshot: StockSnapshot | null;
};

const tabs: Array<{ label: string; value: DetailTab }> = [
  { label: "CHART", value: "chart" },
  { label: "ML + SIGNALS", value: "ml" },
  { label: "OPTIONS", value: "options" },
  { label: "GROQ BRIEF", value: "brief" },
];

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});
const liveDotKeyframes = `
@keyframes live-dot-pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.78); }
  72% { box-shadow: 0 0 0 7px rgba(0, 212, 170, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
}`;

export function NiftyDetail({ selectedSymbol, snapshot }: NiftyDetailProps) {
  const [activeTab, setActiveTab] = useState<DetailTab>("chart");
  const [liveQuote, setLiveQuote] = useState<Tick | null>(null);
  const [nowMs, setNowMs] = useState(Date.now());
  const marketSession = useMarketSession();
  const close = snapshot?.close ?? null;
  const displayClose = displayQuotePrice(close, liveQuote, nowMs);
  const closeStatus = quoteDisplayStatus(liveQuote, nowMs);

  useEffect(() => {
    let active = true;
    setLiveQuote(null);

    const loadLiveQuote = async () => {
      setNowMs(Date.now());
      if (marketSession?.current_state !== "OPEN") {
        return;
      }
      try {
        const tick = await invoke<Tick>("get_quote", { symbol: selectedSymbol });
        if (active && Number.isFinite(tick.ltp)) {
          setLiveQuote(tick);
        }
      } catch {
        if (active) {
          setLiveQuote(null);
        }
      }
    };

    loadLiveQuote();
    const timer = window.setInterval(loadLiveQuote, 2_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [marketSession?.current_state, selectedSymbol]);

  return (
    <section style={zoneStyle} aria-label={`${selectedSymbol} detail`}>
      <style>{liveDotKeyframes}</style>
      <div style={headerStyle}>
        <div style={titleBlockStyle}>
          <span style={eyebrowStyle}>ZONE B LEFT</span>
          <strong style={titleStyle}>{selectedSymbol}</strong>
        </div>
        <div style={statStripStyle}>
          <MiniStat label="CLOSE" status={closeStatus} value={formatPrice(displayClose)} />
          <MiniStat
            label="CHG"
            value={formatPercent(snapshot?.change_pct)}
            color={signedColor(snapshot?.change_pct)}
          />
          <MiniStat label="VOL" value={formatRatio(snapshot?.volume_ratio)} />
          <MiniStat label="52W" value={formatPercent(snapshot?.week_52_high_pct)} />
        </div>
      </div>

      <div style={tabBarStyle}>
        {tabs.map((tab) => (
          <button
            key={tab.value}
            onClick={() => setActiveTab(tab.value)}
            style={tabButtonStyle(activeTab === tab.value)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div style={bodyStyle}>
        <Suspense fallback={<LazyPanelFallback />}>
          {activeTab === "chart" ? (
            <ChartPanel compact symbol={selectedSymbol} />
          ) : null}

          {activeTab === "ml" ? (
            <Stack>
              <MLPanel symbol={selectedSymbol} />
              <SignalsPanel symbol={selectedSymbol} spot={close ?? undefined} />
            </Stack>
          ) : null}

          {activeTab === "options" ? (
            <Stack>
              <OptionsChain symbol={selectedSymbol} spot={close ?? undefined} />
              <IVSurface symbol={selectedSymbol} />
            </Stack>
          ) : null}
        </Suspense>

        {activeTab === "brief" ? (
          <Brief snapshot={snapshot} symbol={selectedSymbol} />
        ) : null}
      </div>
      <Suspense fallback={<LazyPanelFallback />}>
        <EquityPanel symbol={selectedSymbol} />
      </Suspense>
    </section>
  );
}

function LazyPanelFallback() {
  return <div style={emptyStyle}>Loading panel...</div>;
}

function Stack({ children }: { children: ReactNode }) {
  return <div style={stackStyle}>{children}</div>;
}

function Brief({
  snapshot,
  symbol,
}: {
  snapshot: StockSnapshot | null;
  symbol: string;
}) {
  if (!snapshot) {
    return <div style={emptyStyle}>Snapshot loading.</div>;
  }

  return (
    <div style={briefStyle}>
      <div style={briefTitleStyle}>{symbol} EOD Brief</div>
      <p style={briefTextStyle}>
        {symbol} closed at {formatPrice(snapshot.close)} on {snapshot.date}. Day
        change {formatPercent(snapshot.change_pct)}, volume ratio{" "}
        {formatRatio(snapshot.volume_ratio)}, and 52-week high distance{" "}
        {formatPercent(snapshot.week_52_high_pct)}. Current table signal:{" "}
        {snapshot.signal ?? "NEUTRAL"}.
      </p>
      <div style={briefGridStyle}>
        <MiniStat label="OPEN" value={formatPrice(snapshot.open)} />
        <MiniStat label="HIGH" value={formatPrice(snapshot.high)} />
        <MiniStat label="LOW" value={formatPrice(snapshot.low)} />
        <MiniStat label="VOLUME" value={integerFormatter.format(snapshot.volume)} />
        <MiniStat label="52W HIGH" value={formatPrice(snapshot.week_52_high)} />
        <MiniStat label="52W LOW" value={formatPrice(snapshot.week_52_low)} />
      </div>
    </div>
  );
}

function MiniStat({
  color = "#e8e8f0",
  label,
  status,
  value,
}: {
  color?: string;
  label: string;
  status?: QuoteDisplayStatus;
  value: string;
}) {
  return (
    <div style={miniStatStyle}>
      <span style={miniLabelStyle}>{label}</span>
      <strong style={{ ...miniValueStyle, ...(status ? miniLiveValueStyle : null), color }}>
        {status ? <span aria-hidden="true" style={liveDotStyle(status)} /> : null}
        {value}
      </strong>
    </div>
  );
}

function formatPrice(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return numberFormatter.format(value);
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

function signedColor(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) {
    return "#ffd700";
  }
  return value > 0 ? "#00d4aa" : "#ff4444";
}

const zoneStyle = {
  background: "#0f0f1a",
  borderRight: "1px solid #1e1e3a",
  boxSizing: "border-box",
  color: "#e8e8f0",
  display: "grid",
  gridTemplateRows: "74px 34px minmax(0, 1fr) auto",
  minHeight: 0,
  minWidth: 0,
  overflow: "hidden",
} satisfies CSSProperties;

const headerStyle = {
  alignItems: "center",
  borderBottom: "1px solid #1a1a2e",
  display: "grid",
  gap: 10,
  gridTemplateColumns: "132px minmax(0, 1fr)",
  minWidth: 0,
  padding: "10px 12px",
} satisfies CSSProperties;

const titleBlockStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
  minWidth: 0,
} satisfies CSSProperties;

const eyebrowStyle = {
  color: "#6666aa",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
} satisfies CSSProperties;

const titleStyle = {
  color: "#e8e8f0",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 22,
  lineHeight: 1,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const statStripStyle = {
  display: "grid",
  gap: 7,
  gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const miniStatStyle = {
  background: "#12121f",
  border: "1px solid #1e1e3a",
  boxSizing: "border-box",
  minWidth: 0,
  padding: "6px 7px",
} satisfies CSSProperties;

const miniLabelStyle = {
  color: "#6666aa",
  display: "block",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  lineHeight: 1,
} satisfies CSSProperties;

const miniValueStyle = {
  display: "block",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  lineHeight: 1.2,
  marginTop: 5,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const miniLiveValueStyle = {
  alignItems: "center",
  display: "flex",
  gap: 6,
} satisfies CSSProperties;

function liveDotStyle(status: QuoteDisplayStatus): CSSProperties {
  const color =
    status === "live" ? "#00d4aa" : status === "degraded" ? "#a78bfa" : status === "stale" ? "#f59e0b" : "#4b5563";
  return {
    animation: status === "live" ? "live-dot-pulse 1.2s ease-out infinite" : undefined,
    background: color,
    borderRadius: "50%",
    flex: "0 0 auto",
    height: 7,
    width: 7,
  };
}

const tabBarStyle = {
  alignItems: "stretch",
  borderBottom: "1px solid #1a1a2e",
  display: "grid",
  gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
} satisfies CSSProperties;

function tabButtonStyle(active: boolean): CSSProperties {
  return {
    background: active ? "#1a1a2e" : "#0f0f1a",
    border: 0,
    borderRight: "1px solid #1a1a2e",
    color: active ? "#e8e8f0" : "#6666aa",
    cursor: "pointer",
    fontFamily: "Inter, system-ui, sans-serif",
    fontSize: 10,
    fontWeight: active ? 600 : 400,
    minWidth: 0,
  };
}

const bodyStyle = {
  minHeight: 0,
  overflowY: "auto",
  padding: 12,
} satisfies CSSProperties;

const stackStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
  minHeight: 0,
  minWidth: 0,
} satisfies CSSProperties;

const briefStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
  minWidth: 0,
} satisfies CSSProperties;

const briefTitleStyle = {
  color: "#ffd700",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 14,
  fontWeight: 600,
} satisfies CSSProperties;

const briefTextStyle = {
  color: "#d7d7e2",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 13,
  lineHeight: 1.5,
  margin: 0,
} satisfies CSSProperties;

const briefGridStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
} satisfies CSSProperties;

const emptyStyle = {
  color: "#6666aa",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
} satisfies CSSProperties;
