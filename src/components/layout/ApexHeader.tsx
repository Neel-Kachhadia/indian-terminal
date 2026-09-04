import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";

import type { Tick } from "./types";
import { marketStateColor, type MarketDataState } from "../../market/quote";

type DecisionPayload = {
  apex_strip?: {
    anomaly?: string | null;
    ml_confidence?: number | null;
    ml_direction?: string | null;
    next_event?: string | null;
    next_event_days?: number | null;
    observation?: string | null;
    regime?: string | null;
    regime_probability?: number | null;
    tft_1d?: number | null;
    var_per_lot?: number | null;
  };
  market_brief?: {
    brief?: string;
    generated_at?: string;
    model?: string;
  };
};

type OHLCForecast = {
  forecast_open?: number;
  forecast_high?: number;
  forecast_low?: number;
  forecast_close?: number;
};

type BIFCds = {
  cds?: number | null;
  cds_label?: string | null;
  action?: string | null;
  anomaly_gate_active?: boolean;
};

type ApexHeaderProps = {
  brokerAuth: "CONNECTED" | "AUTHENTICATING" | "DISCONNECTED" | "ERROR";
  integrityWarning: boolean;
  marketState: MarketDataState;
  onOpenScreener: () => void;
  onOpenSettings: () => void;
  onSelectNifty: () => void;
  selectedSymbol: string;
  tick: Tick | null;
};

const priceFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function ApexHeader({
  brokerAuth,
  integrityWarning,
  marketState,
  onOpenScreener,
  onOpenSettings,
  onSelectNifty,
  selectedSymbol,
  tick,
}: ApexHeaderProps) {
  const [decision, setDecision] = useState<DecisionPayload | null>(null);
  const [tomorrow, setTomorrow] = useState<OHLCForecast | null>(null);
  const [bifCds, setBifCds] = useState<BIFCds | null>(null);
  const [briefOpen, setBriefOpen] = useState(false);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const result = await invoke<DecisionPayload>("get_decision_intelligence");
        if (active) {
          setDecision(result);
        }
      } catch {
        if (active) {
          setDecision(null);
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

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const result = await invoke<OHLCForecast>("get_ohlc_forecast", { symbol: "NIFTY" });
        if (active) {
          setTomorrow(result);
        }
      } catch {
        if (active) {
          setTomorrow(null);
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

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const result = await invoke<BIFCds>("get_bif_cds", { symbol: "NIFTY" });
        if (active) {
          setBifCds(result);
        }
      } catch {
        if (active) {
          setBifCds(null);
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

  const strip = decision?.apex_strip;
  const priceText = useMemo(() => {
    if (!tick) {
      return "CONNECTING";
    }
    return priceFormatter.format(tick.ltp);
  }, [tick]);
  const brief =
    decision?.market_brief?.brief ||
    strip?.observation ||
    "Apex intelligence loading";

  return (
    <header style={headerStyle}>
      <button
        aria-label="Select NIFTY"
        onClick={onSelectNifty}
        style={priceButtonStyle}
        title="Return to NIFTY home"
        type="button"
      >
        <span style={priceStyle}>Rs {priceText}</span>
        <span style={symbolStyle}>
          <span
            aria-hidden
            style={{ ...liveDotStyle, background: marketStateColor(marketState) }}
            title={`Market data: ${marketState}`}
          />
          NIFTY 50
        </span>
      </button>

      <div style={metricsStyle}>
        <Metric
          label="BROKER"
          value={brokerAuth}
          color={brokerAuthColor(brokerAuth)}
        />
        <Metric label="REGIME" value={regimeText(strip)} color={regimeColor(strip?.regime)} />
        <Metric label="ML" value={mlText(strip)} color={directionColor(strip?.ml_direction)} />
        <Metric label="CDS" value={bifText(bifCds)} color={bifColor(bifCds)} />
        <Metric label="ANOMALY" value={strip?.anomaly ?? "--"} color={strip?.anomaly === "NORMAL" ? "#00d4aa" : "#ff4444"} />
        <Metric label="TFT" value={formatSignedPct(strip?.tft_1d)} />
        <Metric label="TOMORROW" value={tomorrowText(tomorrow)} color="#e8e8f0" />
        <Metric label={integrityWarning ? "DATA" : "VaR"} value={integrityWarning ? "WARNING" : formatCurrency(strip?.var_per_lot)} color="#ffd700" />
      </div>

      <div style={rightRailStyle}>
        {integrityWarning ? <div style={warningBadgeStyle}>Data integrity warning</div> : null}
        <button
          aria-expanded={briefOpen}
          onClick={() => setBriefOpen((open) => !open)}
          style={briefButtonStyle}
          title={brief}
          type="button"
        >
          <span style={briefLabelStyle}>
            {selectedSymbol === "NIFTY" ? "BRIEF" : `VIEW ${selectedSymbol}`}
          </span>
          <span style={briefTextStyle}>{brief}</span>
        </button>
        <button
          aria-label="Screener"
          onClick={onOpenScreener}
          style={screenerButtonStyle}
          title="Screener"
          type="button"
        >
          Screener
        </button>
        <button
          aria-label="Settings"
          onClick={onOpenSettings}
          style={settingsButtonStyle}
          title="Settings"
          type="button"
        >
          Settings
        </button>
      </div>

      {briefOpen ? (
        <div style={briefOverlayStyle}>
          <div style={briefOverlayMetaStyle}>
            {decision?.market_brief?.model ?? "template"} |{" "}
            {formatTime(decision?.market_brief?.generated_at)}
          </div>
          {brief}
        </div>
      ) : null}
    </header>
  );
}

function Metric({
  color = "#e8e8f0",
  label,
  value,
}: {
  color?: string;
  label: string;
  value: string;
}) {
  return (
    <div style={metricStyle}>
      <span style={metricLabelStyle}>{label}</span>
      <strong style={{ ...metricValueStyle, color }}>{value}</strong>
    </div>
  );
}

function regimeText(strip?: DecisionPayload["apex_strip"]) {
  const regime = strip?.regime ?? "--";
  const probability = asNumber(strip?.regime_probability);
  return probability === null ? regime : `${regime} ${Math.round(probability * 100)}%`;
}

function mlText(strip?: DecisionPayload["apex_strip"]) {
  const direction = strip?.ml_direction ?? "--";
  const confidence = asNumber(strip?.ml_confidence);
  return confidence === null ? direction : `${direction} ${Math.round(confidence * 100)}%`;
}

function bifText(value: BIFCds | null) {
  const cds = asNumber(value?.cds);
  if (cds === null) {
    return "--";
  }
  return `${cds.toFixed(1)} ${value?.cds_label ?? ""}`.trim();
}

function eventText(strip?: DecisionPayload["apex_strip"]) {
  if (!strip?.next_event) {
    return "--";
  }
  return `${strip.next_event}${typeof strip.next_event_days === "number" ? ` ${strip.next_event_days}d` : ""}`;
}

function formatCurrency(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return `Rs ${integerFormatter.format(Math.abs(number))}`;
}

function formatSignedPct(value: number | string | null | undefined) {
  const number = typeof value === "string" ? Number(value) : asNumber(value);
  if (number === null || !Number.isFinite(number)) {
    return "--";
  }
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}%`;
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

function tomorrowText(value: OHLCForecast | null) {
  if (!value) {
    return "--";
  }
  return `O:${formatCompact(value.forecast_open)} H:${formatCompact(value.forecast_high)} L:${formatCompact(value.forecast_low)} C:${formatCompact(value.forecast_close)}`;
}

function formatCompact(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return integerFormatter.format(number);
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function directionColor(value: string | null | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("UP") || text.includes("BULL")) {
    return "#00d4aa";
  }
  if (text.includes("DOWN") || text.includes("BEAR")) {
    return "#ff4444";
  }
  return "#ffd700";
}

function brokerAuthColor(value: ApexHeaderProps["brokerAuth"]) {
  if (value === "CONNECTED") {
    return "#00d4aa";
  }
  if (value === "AUTHENTICATING") {
    return "#ffd700";
  }
  return "#ff4444";
}

function bifColor(value: BIFCds | null) {
  if (value?.anomaly_gate_active) {
    return "#f97316";
  }
  const cds = asNumber(value?.cds);
  if (cds === null) {
    return "#e8e8f0";
  }
  if (cds > 65) {
    return "#00d4aa";
  }
  if (cds >= 45) {
    return "#ffd700";
  }
  return "#ff4444";
}

function regimeColor(value: string | null | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("CALM") || text.includes("GREED")) {
    return "#00d4aa";
  }
  if (text.includes("PANIC") || text.includes("FEAR")) {
    return "#ff4444";
  }
  return "#ffd700";
}

const headerStyle = {
  alignItems: "stretch",
  background: "#0a0a0f",
  borderBottom: "1px solid #1e1e3a",
  boxSizing: "border-box",
  color: "#e8e8f0",
  display: "grid",
  gridTemplateColumns: "230px minmax(460px, 1fr) minmax(260px, 0.72fr)",
  height: 80,
  minWidth: 0,
  position: "relative",
  width: "100%",
} satisfies CSSProperties;

const priceButtonStyle = {
  background: "#0f0f1a",
  border: 0,
  borderRight: "1px solid #1e1e3a",
  color: "#e8e8f0",
  cursor: "pointer",
  display: "flex",
  flexDirection: "column",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  justifyContent: "center",
  minWidth: 0,
  padding: "0 14px",
  textAlign: "left",
} satisfies CSSProperties;

const priceStyle = {
  color: "#e8e8f0",
  fontSize: 28,
  fontWeight: 600,
  lineHeight: 1,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const symbolStyle = {
  alignItems: "center",
  color: "#6666aa",
  display: "flex",
  fontSize: 11,
  gap: 7,
  marginTop: 8,
} satisfies CSSProperties;

const liveDotStyle = {
  borderRadius: "50%",
  height: 7,
  width: 7,
} satisfies CSSProperties;

const metricsStyle = {
  display: "grid",
  gap: 0,
  gridTemplateColumns: "repeat(8, minmax(76px, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const metricStyle = {
  borderRight: "1px solid #1a1a2e",
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  minWidth: 0,
  padding: "0 10px",
} satisfies CSSProperties;

const metricLabelStyle = {
  color: "#6666aa",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  lineHeight: 1.2,
} satisfies CSSProperties;

const metricValueStyle = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  lineHeight: 1.25,
  marginTop: 5,
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const briefButtonStyle = {
  background: "#0f0f1a",
  border: 0,
  color: "#e8e8f0",
  cursor: "pointer",
  display: "grid",
  fontFamily: "Inter, system-ui, sans-serif",
  gridTemplateRows: "18px 1fr",
  minWidth: 0,
  padding: "11px 14px",
  textAlign: "left",
} satisfies CSSProperties;

const rightRailStyle = {
  borderLeft: "1px solid #1a1a2e",
  display: "grid",
  gridTemplateColumns: "minmax(0, 1fr) 78px 78px",
  gridTemplateRows: "22px minmax(0, 1fr)",
  minWidth: 0,
} satisfies CSSProperties;

const warningBadgeStyle = {
  alignSelf: "center",
  color: "#ffb4b4",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  gridColumn: "1 / 2",
  paddingLeft: 14,
} satisfies CSSProperties;

const screenerButtonStyle = {
  background: "#12121f",
  border: 0,
  borderLeft: "1px solid #1a1a2e",
  color: "#9ca3af",
  cursor: "pointer",
  fontSize: 11,
  gridColumn: "2 / 3",
  gridRow: "1 / 3",
  padding: "0 8px",
} satisfies CSSProperties;

const settingsButtonStyle = {
  background: "#12121f",
  border: 0,
  borderLeft: "1px solid #1a1a2e",
  color: "#9ca3af",
  cursor: "pointer",
  fontSize: 11,
  gridColumn: "3 / 4",
  gridRow: "1 / 3",
  padding: "0 8px",
} satisfies CSSProperties;

const briefLabelStyle = {
  color: "#6666aa",
  fontSize: 10,
  lineHeight: 1.2,
} satisfies CSSProperties;

const briefTextStyle = {
  alignSelf: "center",
  color: "#e8e8f0",
  fontSize: 12,
  lineHeight: 1.3,
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const briefOverlayStyle = {
  background: "#12121f",
  border: "1px solid #2a2a4a",
  boxShadow: "0 20px 60px rgba(0,0,0,0.45)",
  color: "#e8e8f0",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 13,
  lineHeight: 1.45,
  maxWidth: 720,
  padding: "12px 14px",
  position: "absolute",
  right: 12,
  top: 86,
  zIndex: 20,
} satisfies CSSProperties;

const briefOverlayMetaStyle = {
  color: "#6666aa",
  fontSize: 10,
  marginBottom: 6,
  textTransform: "uppercase",
} satisfies CSSProperties;
