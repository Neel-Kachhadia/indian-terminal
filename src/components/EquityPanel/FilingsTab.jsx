const typeColors = {
  annual_report: { background: "#2a1a3a", color: "#9d4de0", label: "Annual" },
  other: { background: "#1a1a1a", color: "#888888", label: "Other" },
  quarterly_results: { background: "#1a3a2a", color: "#00d4aa", label: "Results" },
  shareholding: { background: "#3a2a1a", color: "#d47700", label: "Shareholding" },
};

export function FilingsTab({ filings, onItemClick, symbol }) {
  if (!filings || filings.length === 0) {
    return <div style={emptyStyle}>No filings found for {symbol}</div>;
  }

  return (
    <div style={listStyle}>
      {filings.map((filing, index) => (
        <div key={`${filing.title}-${filing.date}-${index}`} style={itemStyle}>
          <div style={topStyle}>
            <span style={iconStyle}>📄</span>
            <strong style={titleStyle}>{filing.title}</strong>
            <span style={dateStyle}>{formatDate(filing.date)}</span>
          </div>
          <div style={bottomStyle}>
            <span style={badgeStyle(filing.type)}>{typeLabel(filing.type)}</span>
            <button onClick={() => onItemClick(filing)} style={openStyle} type="button">
              Open →
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function formatDate(dateValue) {
  const timestamp = Date.parse(dateValue ?? "");
  if (!Number.isFinite(timestamp)) {
    return "--";
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(timestamp));
}

function typeLabel(type) {
  return typeColors[type]?.label ?? "Other";
}

function badgeStyle(type) {
  const colors = typeColors[type] ?? typeColors.other;
  return {
    background: colors.background,
    color: colors.color,
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    fontSize: 10,
    padding: "4px 7px",
  };
}

const listStyle = {
  display: "grid",
  gap: 8,
};

const itemStyle = {
  background: "#111111",
  border: "1px solid #242424",
  display: "grid",
  gap: 8,
  minHeight: 70,
  padding: "9px 10px",
};

const topStyle = {
  alignItems: "center",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "20px minmax(0, 1fr) 104px",
};

const iconStyle = {
  fontSize: 14,
};

const titleStyle = {
  color: "#f1f1f1",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

const dateStyle = {
  color: "#888888",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  textAlign: "right",
};

const bottomStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
};

const openStyle = {
  background: "transparent",
  border: "1px solid #333333",
  color: "#ffffff",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  height: 26,
  padding: "0 9px",
};

const emptyStyle = {
  color: "#888888",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
};
