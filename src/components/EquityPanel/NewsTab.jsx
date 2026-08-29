const sourceStyles = {
  BSE: { background: "#3a1a1a", color: "#e04d4d" },
  ET: { background: "#2a2a1a", color: "#d4aa00" },
  NSE: { background: "#1a3a5c", color: "#4d9de0" },
};

const typeIcons = {
  announcement: "📢",
  filing: "📄",
  news: "📰",
};

export function NewsTab({ news, onItemClick, symbol }) {
  if (!news || news.length === 0) {
    return <div style={emptyStyle}>No recent news for {symbol}</div>;
  }

  return (
    <div style={listStyle}>
      {news.map((item) => (
        <button
          key={item.id ?? `${item.title}-${item.date}`}
          onClick={() => onItemClick(item)}
          style={itemStyle}
          type="button"
        >
          <div style={headlineStyle}>
            <span style={badgeStyle(item.source)}>{item.source ?? "NSE"}</span>
            <span style={iconStyle}>{typeIcons[item.type] ?? "📰"}</span>
            <strong style={titleStyle}>{item.title}</strong>
            {isLive(item.date) ? <LiveBadge /> : null}
          </div>
          <div style={metaStyle}>{formatAge(item.date)}</div>
          <div style={hintStyle}>Click to read ▼</div>
        </button>
      ))}
    </div>
  );
}

function LiveBadge() {
  return (
    <span style={liveStyle}>
      <span style={liveDotWrapStyle}>
        <span style={liveDotPulseStyle} />
        <span style={liveDotStyle} />
      </span>
      LIVE
    </span>
  );
}

function isLive(dateValue) {
  const timestamp = Date.parse(dateValue ?? "");
  if (!Number.isFinite(timestamp)) {
    return false;
  }
  return Date.now() - timestamp <= 2 * 60 * 60 * 1000;
}

function formatAge(dateValue) {
  const timestamp = Date.parse(dateValue ?? "");
  if (!Number.isFinite(timestamp)) {
    return "time unavailable";
  }
  const diff = Math.max(0, Date.now() - timestamp);
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) {
    return "just now";
  }
  if (minutes < 60) {
    return `${minutes} minutes ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours} hours ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days} days ago`;
}

function badgeStyle(source) {
  const colors = sourceStyles[source] ?? sourceStyles.NSE;
  return {
    ...baseBadgeStyle,
    background: colors.background,
    color: colors.color,
  };
}

const listStyle = {
  display: "grid",
  gap: 8,
};

const itemStyle = {
  background: "#111111",
  border: "1px solid #242424",
  color: "#cccccc",
  cursor: "pointer",
  display: "grid",
  gap: 5,
  minHeight: 72,
  padding: "9px 10px",
  textAlign: "left",
  width: "100%",
};

const headlineStyle = {
  alignItems: "center",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "42px 18px minmax(0, 1fr) auto",
  minWidth: 0,
};

const baseBadgeStyle = {
  borderRadius: 3,
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  fontWeight: 700,
  padding: "3px 5px",
  textAlign: "center",
};

const iconStyle = {
  fontSize: 14,
};

const titleStyle = {
  color: "#f1f1f1",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
  fontWeight: 600,
  lineHeight: 1.25,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

const metaStyle = {
  color: "#888888",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  paddingLeft: 50,
};

const hintStyle = {
  color: "#777777",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 11,
  paddingLeft: 50,
};

const liveStyle = {
  alignItems: "center",
  color: "#ff4444",
  display: "inline-flex",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  fontWeight: 700,
  gap: 6,
};

const liveDotWrapStyle = {
  display: "inline-flex",
  height: 9,
  position: "relative",
  width: 9,
};

const liveDotStyle = {
  background: "#ff4444",
  borderRadius: "50%",
  height: 7,
  left: 1,
  position: "absolute",
  top: 1,
  width: 7,
};

const liveDotPulseStyle = {
  animation: "equity-live-pulse 1.1s ease-out infinite",
  background: "#ff4444",
  borderRadius: "50%",
  height: 9,
  left: 0,
  position: "absolute",
  top: 0,
  width: 9,
};

const emptyStyle = {
  color: "#777777",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
};
