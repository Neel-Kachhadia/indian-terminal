import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";
import { List } from "react-window";

import type { StockSnapshot, Tick } from "./types";
import {
  displayQuotePrice,
  quoteDisplayStatus,
  type QuoteDisplayStatus,
} from "../../market/quote";
import { useMarketSession } from "../../market/session";

type Universe = "FNO" | "NIFTY50" | "ALL";
type SortKey = "volume_ratio" | "change_pct" | "relative_strength" | "volume";
type ScreenerField =
  | "close"
  | "change_pct"
  | "volume_ratio"
  | "week_52_high_pct"
  | "week_52_low_pct"
  | "volume";
type Operator = ">" | "<" | ">=" | "<=";

type Condition = {
  id: number;
  field: ScreenerField;
  operator: Operator;
  value: number;
};

type ScreenerResponse = {
  computed_at?: number;
  error?: string;
  results?: StockSnapshot[];
  total_scanned?: number;
};

type TrainedSummary = {
  count?: number;
  total_fno?: number;
  symbols?: string[];
};

type StockUniverseProps = {
  onSelectSymbol: (symbol: string, snapshot: StockSnapshot) => void;
  selectedSymbol: string;
};

type VirtualRowData = {
  liveQuotes: Record<string, Tick>;
  nowMs: number;
  onSelectSymbol: (symbol: string, snapshot: StockSnapshot) => void;
  rows: StockSnapshot[];
  selectedSymbol: string;
};

const fieldOptions: Array<{ label: string; value: ScreenerField }> = [
  { label: "Close", value: "close" },
  { label: "Chg %", value: "change_pct" },
  { label: "Volume x", value: "volume_ratio" },
  { label: "52W High %", value: "week_52_high_pct" },
  { label: "52W Low %", value: "week_52_low_pct" },
  { label: "Volume", value: "volume" },
];

const sortOptions: Array<{ label: string; value: SortKey }> = [
  { label: "Volume Ratio", value: "volume_ratio" },
  { label: "Change %", value: "change_pct" },
  { label: "Relative Strength", value: "relative_strength" },
  { label: "Volume", value: "volume" },
];

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

const priceFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});
const liveDotKeyframes = `
@keyframes live-dot-pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.78); }
  72% { box-shadow: 0 0 0 7px rgba(0, 212, 170, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
}`;

export function StockUniverse({
  onSelectSymbol,
  selectedSymbol,
}: StockUniverseProps) {
  const [universe, setUniverse] = useState<Universe>("FNO");
  const [limit, setLimit] = useState(250);
  const [rows, setRows] = useState<StockSnapshot[]>([]);
  const [screenerRows, setScreenerRows] = useState<StockSnapshot[] | null>(null);
  const [field, setField] = useState<ScreenerField>("change_pct");
  const [operator, setOperator] = useState<Operator>(">");
  const [value, setValue] = useState(1);
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("volume_ratio");
  const [search, setSearch] = useState("");
  const [message, setMessage] = useState("Loading FNO universe...");
  const [loading, setLoading] = useState(false);
  const [trainedSummary, setTrainedSummary] = useState<TrainedSummary | null>(null);
  const [trainOpen, setTrainOpen] = useState(false);
  const [trainMessage, setTrainMessage] = useState("");
  const [liveQuotes, setLiveQuotes] = useState<Record<string, Tick>>({});
  const [nowMs, setNowMs] = useState(Date.now());
  const marketSession = useMarketSession();

  useEffect(() => {
    let active = true;

    const loadUniverse = async () => {
      setLoading(true);
      try {
        const result = await invoke<StockSnapshot[] | { error?: string; results?: StockSnapshot[] }>(
          "get_stock_universe",
          { limit, universe },
        );
        if (!active) {
          return;
        }

        const nextRows = Array.isArray(result) ? result : result.results ?? [];
        setRows(nextRows);
        setScreenerRows(null);
        setMessage(`${nextRows.length} ${universe} symbols`);
      } catch (error) {
        if (active) {
          setRows([]);
          setMessage(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadUniverse();
    const timer = window.setInterval(loadUniverse, 300_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [limit, universe]);

  useEffect(() => {
    let active = true;

    const loadTrained = async () => {
      try {
        const result = await invoke<TrainedSummary>("get_trained_symbols");
        if (active) {
          setTrainedSummary(result);
        }
      } catch {
        if (active) {
          setTrainedSummary(null);
        }
      }
    };

    loadTrained();
    const timer = window.setInterval(loadTrained, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const sourceRows = screenerRows ?? rows;

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 3_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let active = true;
    const symbols = Array.from(new Set(sourceRows.map((row) => row.symbol))).slice(0, 500);

    const loadLiveQuotes = async () => {
      setNowMs(Date.now());
      if (marketSession?.current_state !== "OPEN" || symbols.length === 0) {
        return;
      }

      let updates: Record<string, Tick> = {};
      try {
        updates = await invoke<Record<string, Tick>>("get_quotes", { symbols });
      } catch {
        return;
      }

      if (active && Object.keys(updates).length > 0) {
        setLiveQuotes((current) => ({ ...current, ...updates }));
      }
    };

    loadLiveQuotes();
    const timer = window.setInterval(loadLiveQuotes, 3_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [marketSession?.current_state, sourceRows]);

  const displayRows = useMemo(() => {
    const needle = search.trim().toUpperCase();
    return [...sourceRows]
      .filter((row) => !needle || row.symbol.includes(needle))
      .sort((left, right) => numeric(right, sortKey) - numeric(left, sortKey));
  }, [search, sortKey, sourceRows]);

  const addCondition = () => {
    if (conditions.length >= 5) {
      return;
    }
    setConditions((current) => [
      ...current,
      { field, id: Date.now(), operator, value },
    ]);
  };

  const removeCondition = (id: number) => {
    setConditions((current) => current.filter((condition) => condition.id !== id));
  };

  const runScreener = async () => {
    setLoading(true);
    setMessage("Running screener...");
    try {
      const response = await invoke<ScreenerResponse>("run_screener", {
        body: JSON.stringify({
          conditions: conditions.map(({ id: _id, ...condition }) => condition),
          limit,
          logic: "AND",
          universe,
        }),
      });
      if (response.error) {
        throw new Error(response.error);
      }
      const nextRows = response.results ?? [];
      setScreenerRows(nextRows);
      setMessage(`${nextRows.length} stocks match`);
    } catch (error) {
      setScreenerRows([]);
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  };

  const clearScreener = () => {
    setScreenerRows(null);
    setConditions([]);
    setMessage(`${rows.length} ${universe} symbols`);
  };

  const trainAllStocks = async () => {
    setTrainMessage("Training all FNO stocks...");
    try {
      const result = await invoke<{ status?: string }>("train_all_stocks", { force: false });
      setTrainMessage(result.status === "already_running" ? "Training already running." : "Training started.");
    } catch (error) {
      setTrainMessage(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <section style={zoneStyle} aria-label="Stock universe and screener">
      <style>{liveDotKeyframes}</style>
      <div style={topStyle}>
        <div style={titleRowStyle}>
          <div>
            <span style={eyebrowStyle}>ZONE B RIGHT</span>
            <strong style={titleStyle}>STOCK UNIVERSE / SCREENER</strong>
          </div>
          <div style={headerStatusStyle}>
            <button onClick={() => setTrainOpen((open) => !open)} style={trainedButtonStyle} type="button">
              Trained: {trainedSummary?.count ?? 0}/{trainedSummary?.total_fno ?? "--"} stocks
            </button>
            <span style={statusStyle}>{loading ? "Loading" : message}</span>
          </div>
        </div>

        {trainOpen ? (
          <div style={trainPanelStyle}>
            <span>Full FNO training runs in background and may take 30-60 minutes.</span>
            <button onClick={trainAllStocks} style={smallButtonStyle} type="button">
              Train All 212 FNO Stocks
            </button>
            <span style={statusStyle}>{trainMessage}</span>
          </div>
        ) : null}

        <div style={controlsStyle}>
          <select
            aria-label="Universe"
            onChange={(event) => setUniverse(event.target.value as Universe)}
            style={controlStyle}
            value={universe}
          >
            <option value="FNO">FNO</option>
            <option value="NIFTY50">NIFTY50</option>
            <option value="ALL">ALL</option>
          </select>
          <select
            aria-label="Screener field"
            onChange={(event) => setField(event.target.value as ScreenerField)}
            style={controlStyle}
            value={field}
          >
            {fieldOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <select
            aria-label="Screener operator"
            onChange={(event) => setOperator(event.target.value as Operator)}
            style={controlStyle}
            value={operator}
          >
            {([">", "<", ">=", "<="] as Operator[]).map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <input
            aria-label="Screener value"
            onChange={(event) => setValue(Number(event.target.value))}
            style={controlStyle}
            type="number"
            value={value}
          />
          <button onClick={addCondition} style={buttonStyle} type="button">
            + Add
          </button>
          <button onClick={runScreener} style={buttonStyle} type="button">
            Run
          </button>
          <button onClick={clearScreener} style={buttonStyle} type="button">
            Clear
          </button>
        </div>

        <div style={filterStyle}>
          <select
            aria-label="Sort by"
            onChange={(event) => setSortKey(event.target.value as SortKey)}
            style={controlStyle}
            value={sortKey}
          >
            {sortOptions.map((option) => (
              <option key={option.value} value={option.value}>
                Sort: {option.label}
              </option>
            ))}
          </select>
          <select
            aria-label="Show count"
            onChange={(event) => setLimit(Number(event.target.value))}
            style={controlStyle}
            value={limit}
          >
            {[50, 100, 200, 250, 500].map((count) => (
              <option key={count} value={count}>
                Show: {count}
              </option>
            ))}
          </select>
          <input
            aria-label="Search symbols"
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search symbol"
            style={controlStyle}
            value={search}
          />
        </div>

        {conditions.length > 0 ? (
          <div style={chipsStyle}>
            {conditions.map((condition) => (
              <button
                key={condition.id}
                onClick={() => removeCondition(condition.id)}
                style={chipStyle}
                type="button"
              >
                {conditionLabel(condition)} x
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div style={tableStyle}>
        <div style={headerGridStyle}>
          <span>SYMBOL</span>
          <span>CLOSE</span>
          <span>CHG%</span>
          <span>VOL RATIO</span>
          <span>52W</span>
          <span>RS</span>
          <span>SIGNAL</span>
        </div>
        <div style={bodyStyle}>
          {displayRows.length > 200 ? (
            <List<VirtualRowData>
              defaultHeight={500}
              overscanCount={8}
              rowComponent={VirtualRow}
              rowCount={displayRows.length}
              rowHeight={34}
              rowProps={{
                liveQuotes,
                nowMs,
                onSelectSymbol,
                rows: displayRows,
                selectedSymbol,
              }}
              style={{ height: "100%", width: "100%" }}
            />
          ) : (
            <div style={plainRowsStyle}>
              {displayRows.map((row) => (
                <StockRow
                  key={`${row.symbol}-${row.date}`}
                  liveQuote={liveQuotes[row.symbol]}
                  nowMs={nowMs}
                  onSelectSymbol={onSelectSymbol}
                  row={row}
                  selected={row.symbol === selectedSymbol}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function VirtualRow({
  ariaAttributes,
  index,
  liveQuotes,
  nowMs,
  onSelectSymbol,
  rows,
  selectedSymbol,
  style,
}: {
  ariaAttributes: {
    "aria-posinset": number;
    "aria-setsize": number;
    role: "listitem";
  };
  index: number;
  style: CSSProperties;
} & VirtualRowData) {
  const row = rows[index];
  return (
    <StockRow
      ariaAttributes={ariaAttributes}
      liveQuote={liveQuotes[row.symbol]}
      nowMs={nowMs}
      onSelectSymbol={onSelectSymbol}
      row={row}
      selected={row.symbol === selectedSymbol}
      style={style}
    />
  );
}

function StockRow({
  ariaAttributes,
  liveQuote,
  nowMs,
  onSelectSymbol,
  row,
  selected,
  style,
}: {
  ariaAttributes?: {
    "aria-posinset": number;
    "aria-setsize": number;
    role: "listitem";
  };
  liveQuote?: Tick;
  nowMs: number;
  onSelectSymbol: (symbol: string, snapshot: StockSnapshot) => void;
  row: StockSnapshot;
  selected: boolean;
  style?: CSSProperties;
}) {
  const status = quoteDisplayStatus(liveQuote, nowMs);
  const close = displayQuotePrice(row.close, liveQuote, nowMs);

  return (
    <button
      {...ariaAttributes}
      onClick={() => onSelectSymbol(row.symbol, row)}
      style={{
        ...rowStyle,
        ...(selected ? selectedRowStyle : null),
        ...style,
      }}
      type="button"
    >
      <span style={symbolCellStyle}>{row.symbol}</span>
      <span style={priceCellStyle}>
        <span aria-hidden="true" style={liveDotStyle(status)} />
        {formatPrice(close)}
      </span>
      <span style={{ color: signedColor(row.change_pct) }}>
        {formatPercent(row.change_pct)}
      </span>
      <span style={{ color: volumeColor(row.volume_ratio) }}>
        {formatRatio(row.volume_ratio)}
      </span>
      <span>{formatPercent(row.week_52_high_pct)}</span>
      <span style={{ color: rsColor(row.relative_strength) }}>
        {rsArrow(row.relative_strength)}
      </span>
      <span style={{ color: signalColor(row.signal) }}>{row.signal ?? "NEUTRAL"}</span>
    </button>
  );
}

function conditionLabel(condition: Condition) {
  const label =
    fieldOptions.find((option) => option.value === condition.field)?.label ??
    condition.field;
  return `${label} ${condition.operator} ${condition.value}`;
}

function numeric(row: StockSnapshot, key: SortKey) {
  const value = row[key];
  return typeof value === "number" && Number.isFinite(value) ? value : -Infinity;
}

function formatPrice(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return priceFormatter.format(value);
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

function volumeColor(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "#6666aa";
  }
  if (value > 2) {
    return "#00d4aa";
  }
  if (value < 0.5) {
    return "#ff4444";
  }
  return "#e8e8f0";
}

function rsArrow(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "->";
  }
  if (value > 1) {
    return "↑";
  }
  if (value < -1) {
    return "↓";
  }
  return "→";
}

function rsColor(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "#6666aa";
  }
  if (value > 1) {
    return "#00d4aa";
  }
  if (value < -1) {
    return "#ff4444";
  }
  return "#ffd700";
}

function signalColor(value: string | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (text.includes("BULL") || text.includes("VOL")) {
    return "#00d4aa";
  }
  if (text.includes("BEAR") || text.includes("SELL")) {
    return "#ff4444";
  }
  return "#ffd700";
}

const zoneStyle = {
  background: "#0a0a0f",
  boxSizing: "border-box",
  color: "#e8e8f0",
  display: "grid",
  gridTemplateRows: "auto minmax(0, 1fr)",
  minHeight: 0,
  minWidth: 0,
  overflow: "hidden",
} satisfies CSSProperties;

const topStyle = {
  borderBottom: "1px solid #1e1e3a",
  display: "flex",
  flexDirection: "column",
  gap: 8,
  padding: "10px 12px",
} satisfies CSSProperties;

const titleRowStyle = {
  alignItems: "center",
  display: "flex",
  gap: 12,
  justifyContent: "space-between",
} satisfies CSSProperties;

const headerStatusStyle = {
  alignItems: "center",
  display: "flex",
  gap: 10,
  minWidth: 0,
} satisfies CSSProperties;

const eyebrowStyle = {
  color: "#6666aa",
  display: "block",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  lineHeight: 1,
  marginBottom: 5,
} satisfies CSSProperties;

const titleStyle = {
  color: "#e8e8f0",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 14,
  lineHeight: 1,
} satisfies CSSProperties;

const statusStyle = {
  color: "#6666aa",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 11,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const trainedButtonStyle = {
  background: "#12121f",
  border: "1px solid #2a2a4a",
  borderRadius: 4,
  color: "#ffd700",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  height: 24,
  padding: "0 8px",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const trainPanelStyle = {
  alignItems: "center",
  background: "#12121f",
  border: "1px solid #2a2a4a",
  color: "#d7d7e2",
  display: "grid",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "minmax(220px, 1fr) 178px minmax(120px, 0.6fr)",
  minHeight: 32,
  padding: "6px 8px",
} satisfies CSSProperties;

const controlsStyle = {
  display: "grid",
  gap: 7,
  gridTemplateColumns: "78px 104px 66px 82px 62px 52px 56px",
} satisfies CSSProperties;

const filterStyle = {
  display: "grid",
  gap: 7,
  gridTemplateColumns: "minmax(150px, 1fr) 112px minmax(160px, 1fr)",
} satisfies CSSProperties;

const controlStyle = {
  background: "#12121f",
  border: "1px solid #2a2a4a",
  borderRadius: 4,
  boxSizing: "border-box",
  color: "#e8e8f0",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  height: 28,
  minWidth: 0,
  padding: "0 8px",
  width: "100%",
} satisfies CSSProperties;

const buttonStyle = {
  ...controlStyle,
  cursor: "pointer",
  fontFamily: "Inter, system-ui, sans-serif",
  padding: "0 7px",
} satisfies CSSProperties;

const smallButtonStyle = {
  ...buttonStyle,
  color: "#ffd700",
  height: 26,
} satisfies CSSProperties;

const chipsStyle = {
  display: "flex",
  flexWrap: "wrap",
  gap: 6,
} satisfies CSSProperties;

const chipStyle = {
  background: "#1a1a2e",
  border: "1px solid #2a2a4a",
  borderRadius: 4,
  color: "#ffd700",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10,
  minHeight: 24,
  padding: "0 7px",
} satisfies CSSProperties;

const tableStyle = {
  display: "grid",
  gridTemplateRows: "28px minmax(0, 1fr)",
  minHeight: 0,
  minWidth: 0,
  overflow: "hidden",
} satisfies CSSProperties;

const headerGridStyle = {
  alignItems: "center",
  background: "#12121f",
  borderBottom: "1px solid #1e1e3a",
  color: "#6666aa",
  display: "grid",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 10,
  gap: 8,
  gridTemplateColumns: "1.15fr 0.9fr 0.72fr 0.86fr 0.62fr 0.42fr 0.92fr",
  padding: "0 12px",
} satisfies CSSProperties;

const bodyStyle = {
  minHeight: 0,
  overflowY: "auto",
} satisfies CSSProperties;

const plainRowsStyle = {
  display: "flex",
  flexDirection: "column",
} satisfies CSSProperties;

const rowStyle = {
  alignItems: "center",
  background: "#0a0a0f",
  border: 0,
  borderBottom: "1px solid #14142a",
  color: "#e8e8f0",
  cursor: "pointer",
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  gap: 8,
  gridTemplateColumns: "1.15fr 0.9fr 0.72fr 0.86fr 0.62fr 0.42fr 0.92fr",
  height: 34,
  minWidth: 0,
  padding: "0 12px",
  textAlign: "left",
  width: "100%",
} satisfies CSSProperties;

const priceCellStyle = {
  alignItems: "center",
  display: "inline-flex",
  gap: 6,
  minWidth: 0,
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

const selectedRowStyle = {
  background: "#1a1a2e",
  boxShadow: "inset 3px 0 0 #4488ff",
} satisfies CSSProperties;

const symbolCellStyle = {
  color: "#e8e8f0",
  fontWeight: 600,
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;
