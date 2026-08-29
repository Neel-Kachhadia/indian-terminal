import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  displayQuotePrice,
  quoteDisplayStatus,
  type QuoteDisplayStatus,
  type Tick,
} from "../../market/quote";
import { useMarketSession } from "../../market/session";

type ScreenerField = "close" | "rsi_14" | "ema_20" | "volume_spike";
type ScreenerOperator = ">" | "<" | ">=" | "<=";
type ScreenerLogic = "AND" | "OR";

type Condition = {
  id: number;
  field: ScreenerField;
  operator: ScreenerOperator;
  value: number;
  timeframe: "1d";
};

type ScreenerResult = {
  symbol: string;
  ltp: number;
  matched_conditions: string[];
  score: number;
};

type ScreenerResponse = {
  results: ScreenerResult[];
  total_scanned: number;
  computed_at: number;
  error?: string;
};

const FIELD_OPTIONS: { label: string; value: ScreenerField }[] = [
  { label: "Close", value: "close" },
  { label: "RSI 14", value: "rsi_14" },
  { label: "EMA 20", value: "ema_20" },
  { label: "Volume Spike", value: "volume_spike" },
];

const OPERATOR_OPTIONS: ScreenerOperator[] = [">", "<", ">=", "<="];
const liveDotKeyframes = `
@keyframes live-dot-pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.78); }
  72% { box-shadow: 0 0 0 7px rgba(0, 212, 170, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
}`;

export function Screener() {
  const [field, setField] = useState<ScreenerField>("close");
  const [operator, setOperator] = useState<ScreenerOperator>(">");
  const [value, setValue] = useState(1000);
  const [logic, setLogic] = useState<ScreenerLogic>("AND");
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [results, setResults] = useState<ScreenerResult[]>([]);
  const [message, setMessage] = useState("No screener run yet");
  const [isLoading, setIsLoading] = useState(false);
  const [liveQuotes, setLiveQuotes] = useState<Record<string, Tick>>({});
  const [nowMs, setNowMs] = useState(Date.now());
  const marketSession = useMarketSession();

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 3_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let active = true;
    const symbols = Array.from(new Set(results.map((result) => result.symbol))).slice(0, 500);

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
  }, [marketSession?.current_state, results]);

  const addCondition = () => {
    if (conditions.length >= 5) {
      return;
    }

    setConditions((current) => [
      ...current,
      {
        field,
        id: Date.now(),
        operator,
        timeframe: "1d",
        value: field === "volume_spike" ? 0 : value,
      },
    ]);
  };

  const removeCondition = (id: number) => {
    setConditions((current) => current.filter((item) => item.id !== id));
  };

  const runScreener = async () => {
    setIsLoading(true);
    setMessage("Running screener...");
    try {
      const response = await invoke<ScreenerResponse>("get_screener_results", {
        conditions: conditions.map(({ id: _id, ...condition }) => condition),
        limit: 50,
        logic,
        universe: "NIFTY50",
      });
      if (response.error) {
        throw new Error(response.error);
      }

      setResults(response.results);
      setMessage(
        `Scanned ${response.total_scanned} symbols, matched ${response.results.length}`,
      );
    } catch (error) {
      setResults([]);
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section
      style={{
        borderTop: "1px solid #1f2933",
        paddingTop: 20,
      }}
    >
      <style>{liveDotKeyframes}</style>
      <div
        style={{
          alignItems: "center",
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <h2
          style={{
            fontSize: 18,
            letterSpacing: 0,
            margin: 0,
          }}
        >
          Screener
        </h2>
        <div
          style={{
            color: "#8a94a6",
            fontSize: 12,
          }}
        >
          NIFTY50 · 1D
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gap: 10,
          gridTemplateColumns: "repeat(auto-fit, minmax(132px, 1fr))",
          marginBottom: 12,
        }}
      >
        <select
          onChange={(event) => setField(event.target.value as ScreenerField)}
          style={controlStyle}
          value={field}
        >
          {FIELD_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <select
          disabled={field === "volume_spike"}
          onChange={(event) =>
            setOperator(event.target.value as ScreenerOperator)
          }
          style={controlStyle}
          value={operator}
        >
          {OPERATOR_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <input
          disabled={field === "volume_spike"}
          onChange={(event) => setValue(Number(event.target.value))}
          style={controlStyle}
          type="number"
          value={field === "volume_spike" ? 0 : value}
        />
        <button
          disabled={conditions.length >= 5}
          onClick={addCondition}
          style={buttonStyle(conditions.length < 5)}
          type="button"
        >
          Add Condition
        </button>
      </div>

      {conditions.length > 0 ? (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 8,
            marginBottom: 12,
          }}
        >
          {conditions.map((condition) => (
            <button
              key={condition.id}
              onClick={() => removeCondition(condition.id)}
              style={{
                ...buttonStyle(true),
                background: "#101820",
                color: "#d1d4dc",
              }}
              title="Remove condition"
              type="button"
            >
              {conditionLabel(condition)} ×
            </button>
          ))}
        </div>
      ) : null}

      <div
        style={{
          alignItems: "center",
          display: "flex",
          flexWrap: "wrap",
          gap: 10,
          marginBottom: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            gap: 6,
          }}
        >
          {(["AND", "OR"] as ScreenerLogic[]).map((item) => (
            <button
              key={item}
              onClick={() => setLogic(item)}
              style={{
                ...buttonStyle(true),
                background: logic === item ? "#1f2937" : "#101010",
                color: logic === item ? "#ffffff" : "#8a94a6",
              }}
              type="button"
            >
              {item}
            </button>
          ))}
        </div>
        <button
          disabled={isLoading}
          onClick={runScreener}
          style={buttonStyle(!isLoading)}
          type="button"
        >
          Run Screener
        </button>
        <div
          style={{
            color: "#8a94a6",
            fontSize: 12,
          }}
        >
          {message}
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            borderCollapse: "collapse",
            fontSize: 12,
            minWidth: 560,
            width: "100%",
          }}
        >
          <thead>
            <tr style={{ color: "#8a94a6", textAlign: "left" }}>
              <th style={cellStyle}>Symbol</th>
              <th style={cellStyle}>LTP</th>
              <th style={cellStyle}>Matched Conditions</th>
            </tr>
          </thead>
          <tbody>
            {results.length > 0 ? (
              results.map((result) => (
                <tr key={result.symbol}>
                  <td style={cellStyle}>{result.symbol}</td>
                  <td style={cellStyle}>
                    <span style={priceCellStyle}>
                      <span
                        aria-hidden="true"
                        style={liveDotStyle(quoteDisplayStatus(liveQuotes[result.symbol], nowMs))}
                      />
                      {formatPrice(displayQuotePrice(result.ltp, liveQuotes[result.symbol], nowMs) ?? result.ltp)}
                    </span>
                  </td>
                  <td style={cellStyle}>
                    {result.matched_conditions.join(", ")}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} style={cellStyle}>
                  No results
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatPrice(value: number) {
  return value.toFixed(2);
}

const controlStyle = {
  background: "#101010",
  border: "1px solid #262626",
  borderRadius: 6,
  boxSizing: "border-box",
  color: "#ffffff",
  fontFamily: "monospace",
  fontSize: 12,
  height: 34,
  padding: "0 10px",
  width: "100%",
} satisfies React.CSSProperties;

const cellStyle = {
  borderBottom: "1px solid #1f2933",
  padding: "9px 8px",
  verticalAlign: "top",
} satisfies React.CSSProperties;

const priceCellStyle = {
  alignItems: "center",
  display: "inline-flex",
  gap: 6,
} satisfies React.CSSProperties;

function liveDotStyle(status: QuoteDisplayStatus): React.CSSProperties {
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

function buttonStyle(enabled: boolean): React.CSSProperties {
  return {
    background: enabled ? "#1f2937" : "#111111",
    border: "1px solid #333333",
    borderRadius: 6,
    color: enabled ? "#ffffff" : "#5f6673",
    cursor: enabled ? "pointer" : "not-allowed",
    fontFamily: "monospace",
    fontSize: 12,
    height: 34,
    padding: "0 12px",
  };
}

function conditionLabel(condition: Condition) {
  const label =
    FIELD_OPTIONS.find((option) => option.value === condition.field)?.label ??
    condition.field;
  if (condition.field === "volume_spike") {
    return label;
  }

  return `${label} ${condition.operator} ${condition.value}`;
}
