import { useState } from "react";

export function FinancialsTab({ financials, symbol }) {
  const [view, setView] = useState("quarterly");
  const quarters = financials?.quarters ?? [];
  const ratios = financials?.ratios ?? {};

  if (!quarters.length && view === "quarterly") {
    return (
      <div style={emptyStyle}>
        Financial data unavailable — {symbol} may not have NSE listed financial results
      </div>
    );
  }

  return (
    <div style={wrapStyle}>
      <div style={subTabStyle}>
        <button onClick={() => setView("quarterly")} style={subButtonStyle(view === "quarterly")} type="button">
          QUARTERLY
        </button>
        <button onClick={() => setView("annual")} style={subButtonStyle(view === "annual")} type="button">
          ANNUAL
        </button>
      </div>
      {view === "quarterly" ? (
        <>
          <div style={tableStyle}>
            <div style={headerStyle}>
              <span>Period</span>
              <span>Revenue (Cr)</span>
              <span>PAT (Cr)</span>
              <span>EPS</span>
              <span>YoY%</span>
            </div>
            {quarters.map((quarter, index) => (
              <div key={`${quarter.period}-${quarter.report_date}-${index}`} style={rowStyle(index === 0)}>
                <span>{quarter.period ?? "--"}</span>
                <span>{formatNumber(quarter.revenue)}</span>
                <span>{formatNumber(quarter.pat)}</span>
                <span>{formatNumber(quarter.eps)}</span>
                <span style={{ color: yoyColor(quarter.yoy_growth_pct) }}>
                  {formatPercent(quarter.yoy_growth_pct)}
                </span>
              </div>
            ))}
          </div>
          <div style={ratiosGridStyle}>
            <Ratio label="P/E" value={formatNumber(ratios.pe)} />
            <Ratio label="P/B" value={formatNumber(ratios.pb)} />
            <Ratio label="ROE" value={formatPercent(ratios.roe)} />
            <Ratio label="D/E" value={formatNumber(ratios.debt_to_equity)} />
          </div>
        </>
      ) : (
        <div style={emptyStyle}>Annual financials unavailable for {symbol}</div>
      )}
    </div>
  );
}

function Ratio({ label, value }) {
  return (
    <div style={ratioStyle}>
      <span style={ratioLabelStyle}>{label}</span>
      <strong style={ratioValueStyle}>{value}</strong>
    </div>
  );
}

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function yoyColor(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) {
    return "#888888";
  }
  return value > 0 ? "#00d4aa" : "#ff4444";
}

const wrapStyle = {
  display: "grid",
  gap: 10,
};

const subTabStyle = {
  display: "grid",
  gridTemplateColumns: "110px 110px minmax(0, 1fr)",
};

function subButtonStyle(active) {
  return {
    background: active ? "#1a1a1a" : "#0d0d0d",
    border: "1px solid #242424",
    color: active ? "#ffffff" : "#666666",
    cursor: "pointer",
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    fontSize: 10,
    height: 28,
  };
}

const tableStyle = {
  border: "1px solid #242424",
  display: "grid",
};

const headerStyle = {
  background: "#141414",
  color: "#888888",
  display: "grid",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  gridTemplateColumns: "1fr repeat(4, minmax(74px, 0.8fr))",
  minHeight: 28,
  padding: "7px 9px",
};

function rowStyle(active) {
  return {
    background: active ? "#1a1a1a" : "#101010",
    borderTop: "1px solid #242424",
    color: "#dddddd",
    display: "grid",
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    fontSize: 11,
    gridTemplateColumns: "1fr repeat(4, minmax(74px, 0.8fr))",
    minHeight: 30,
    padding: "8px 9px",
  };
}

const ratiosGridStyle = {
  border: "1px solid #242424",
  display: "grid",
  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
};

const ratioStyle = {
  alignItems: "center",
  borderBottom: "1px solid #242424",
  borderRight: "1px solid #242424",
  display: "flex",
  gap: 8,
  minHeight: 34,
  padding: "0 10px",
};

const ratioLabelStyle = {
  color: "#888888",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 11,
};

const ratioValueStyle = {
  color: "#ffffff",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
};

const emptyStyle = {
  color: "#888888",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
  lineHeight: 1.4,
};
