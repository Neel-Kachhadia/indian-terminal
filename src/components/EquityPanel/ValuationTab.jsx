function money(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `Rs ${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`
    : "--";
}

function percent(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value > 0 ? "+" : ""}${value.toFixed(2)}%`
    : "--";
}

export function ValuationTab({ valuation, symbol, status, message }) {
  const methods = Array.isArray(valuation?.methods) ? valuation.methods : [];
  if (status === "ERROR") {
    return <div style={messageStyle}>{message || `Valuation unavailable for ${symbol}.`}</div>;
  }
  return (
    <div style={wrapStyle}>
      <div style={summaryStyle}>
        <Metric label="CURRENT PRICE" value={money(valuation?.market_price)} />
        <Metric label="FAIR VALUE" value={money(valuation?.fair_value)} />
        <Metric
          label="FAIR VALUE RANGE"
          value={`${money(valuation?.fair_value_low)} - ${money(valuation?.fair_value_high)}`}
        />
        <Metric label="DISCOUNT / PREMIUM" value={percent(valuation?.discount_premium_pct)} />
        <Metric label="CLASSIFICATION" value={valuation?.classification ?? "INDETERMINATE"} />
        <Metric label="EVIDENCE" value={valuation?.evidence_quality ?? "INSUFFICIENT_EVIDENCE"} />
        <Metric label="DATA AS OF" value={valuation?.data_as_of ?? "--"} />
        <Metric label="PRICE QUALITY" value={valuation?.quote_quality ?? "NO_DATA"} />
      </div>

      <div style={methodHeaderStyle}>METHOD EVIDENCE</div>
      <div style={methodsStyle}>
        {methods.length ? methods.map((method) => (
          <div key={method.method} style={methodStyle}>
            <strong style={methodNameStyle}>{method.method}</strong>
            <span style={methodStatusStyle(method.status)}>{method.status}</span>
            <span>{money(method.estimate)}</span>
            <span style={reasonStyle}>{method.reason ?? `${method.peer_count ?? 0} peers`}</span>
          </div>
        )) : <div style={emptyStyle}>No defensible valuation method available for {symbol}.</div>}
      </div>

      {valuation?.reason ? <div style={warningStyle}>{valuation.reason}</div> : null}
      {(valuation?.warnings ?? []).map((warning) => (
        <div key={warning} style={noteStyle}>{warning}</div>
      ))}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div style={metricStyle}>
      <span style={labelStyle}>{label}</span>
      <strong style={valueStyle}>{value}</strong>
    </div>
  );
}

const wrapStyle = { display: "grid", gap: 10 };
const summaryStyle = {
  display: "grid",
  gap: 1,
  gridTemplateColumns: "repeat(4, minmax(120px, 1fr))",
};
const metricStyle = { background: "#111", border: "1px solid #1f1f1f", display: "grid", gap: 5, padding: 9 };
const labelStyle = { color: "#666", fontSize: 9, letterSpacing: "0.08em" };
const valueStyle = { color: "#e8e8f0", fontFamily: "'JetBrains Mono', monospace", fontSize: 12 };
const methodHeaderStyle = { color: "#777", fontSize: 9, letterSpacing: "0.08em" };
const methodsStyle = { display: "grid", gap: 4 };
const methodStyle = {
  alignItems: "center",
  background: "#101010",
  border: "1px solid #1f1f1f",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "160px 130px 120px minmax(0, 1fr)",
  padding: "7px 9px",
};
const methodNameStyle = { color: "#ddd", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 };
const methodStatusStyle = (status) => ({ color: status === "VALID" ? "#00d4aa" : "#f59e0b", fontSize: 10 });
const reasonStyle = { color: "#777", fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };
const warningStyle = { color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 };
const noteStyle = { color: "#777", fontSize: 10 };
const emptyStyle = { color: "#777", fontSize: 11 };
const messageStyle = { color: "#d4aa00", fontFamily: "Inter, system-ui, sans-serif", fontSize: 12 };
