import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { Config, Data, Layout } from "plotly.js";

import { loadPlotly } from "../../lib/plotly";
import { useQuotesStore } from "../../store/quotes";
import { executableQuotePrice } from "../../market/quote";

type LegAction = "BUY" | "SELL";
type LegType = "CE" | "PE" | "FUT";
type Preset = "custom" | "long-straddle" | "short-straddle" | "bull-call" | "bear-put" | "iron-condor";

type OptionLeg = {
  symbol: string;
  expiry: string;
  strike: number;
  option_type: "CE" | "PE";
  ltp: number;
};

type StrategyLeg = {
  id: string;
  symbol: string;
  strike: number;
  expiry: string;
  option_type: LegType;
  action: LegAction;
  quantity: number;
  entry_price: number;
  lot_size: number;
};

type PayoffResult = {
  spot_range: number[];
  pnl_at_expiry: number[];
  pnl_today: number[] | null;
  analytical_status: string;
  risk_free_rate_source: string | null;
  dividend_yield_source: string | null;
  breakevens: number[];
  max_profit: number | null;
  max_loss: number | null;
  net_greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  } | null;
};

type FuturesBasisData = {
  error?: string;
  dates: string[];
  spot: number[];
  futures: number[];
  basis: number[];
  basis_pct: number[];
  annualised_carry: number[];
  near_expiry: string;
  current_basis: number | null;
  current_carry: number | null;
};

const plotConfig: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};

const baseLayout: Partial<Layout> = {
  autosize: true,
  font: { color: "#d1d4dc", family: "monospace" },
  margin: { b: 48, l: 64, r: 44, t: 18 },
  paper_bgcolor: "#0d0d0d",
  plot_bgcolor: "#0d0d0d",
};

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const PRESETS: { label: string; value: Preset }[] = [
  { label: "Custom", value: "custom" },
  { label: "Long Straddle", value: "long-straddle" },
  { label: "Short Straddle", value: "short-straddle" },
  { label: "Bull Call Spread", value: "bull-call" },
  { label: "Bear Put Spread", value: "bear-put" },
  { label: "Iron Condor", value: "iron-condor" },
];

export function StrategyLab() {
  const spot = useQuotesStore(
    (state) => executableQuotePrice(state.quotes.NIFTY ?? state.quotes["NIFTY 50"]),
  );
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState("");
  const [chain, setChain] = useState<OptionLeg[]>([]);
  const [lotSize, setLotSize] = useState<number | null>(null);
  const [legs, setLegs] = useState<StrategyLeg[]>([]);
  const [preset, setPreset] = useState<Preset>("custom");
  const [payoff, setPayoff] = useState<PayoffResult | null>(null);
  const [payoffMessage, setPayoffMessage] = useState("Add legs to plot payoff");
  const [basis, setBasis] = useState<FuturesBasisData | null>(null);
  const [basisMessage, setBasisMessage] = useState("Loading futures basis...");

  useEffect(() => {
    let active = true;

    const loadExpiries = async () => {
      try {
        const nextExpiries = await invoke<string[]>("get_expiries", {
          symbol: "NIFTY",
        });
        if (!active) {
          return;
        }

        setExpiries(nextExpiries);
        setSelectedExpiry((current) => current || nextExpiries[0] || "");
      } catch {
        if (active) {
          setExpiries([]);
        }
      }
    };

    loadExpiries();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedExpiry) {
      setLotSize(null);
      return;
    }

    let active = true;

    const loadLotSize = async () => {
      setLotSize(null);
      try {
        const nextLotSize = await invoke<number>("get_lot_size", {
          expiry: selectedExpiry,
          symbol: "NIFTY",
        });
        if (active) {
          setLotSize(nextLotSize);
        }
      } catch (error) {
        if (active) {
          setLotSize(null);
          setPayoffMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };

    const loadChain = async () => {
      try {
        const nextChain = await invoke<OptionLeg[]>("get_chain", {
          expiry: selectedExpiry,
          symbol: "NIFTY",
        });
        if (active) {
          setChain(nextChain);
        }
      } catch {
        if (active) {
          setChain([]);
        }
      }
    };

    loadLotSize();
    loadChain();
    const timer = window.setInterval(loadChain, 3000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [selectedExpiry]);

  useEffect(() => {
    let active = true;

    const loadBasis = async () => {
      try {
        const result = await invoke<FuturesBasisData>("get_futures_basis", {
          symbol: "NIFTY",
        });
        if (!active) {
          return;
        }

        setBasis(result);
        setBasisMessage(
          result.dates.length > 0 ? "" : result.error || "No historical futures data",
        );
      } catch (error) {
        if (active) {
          setBasis(null);
          setBasisMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };

    loadBasis();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (legs.length === 0) {
      setPayoff(null);
      setPayoffMessage("Add legs to plot payoff");
      return;
    }

    if (spot <= 0) {
      setPayoff(null);
      setPayoffMessage("Waiting for NIFTY spot");
      return;
    }

    let active = true;
    setPayoffMessage("Computing payoff...");

    const timer = window.setTimeout(async () => {
      try {
        const result = await invoke<PayoffResult>("get_payoff", {
          legs: legs.map(({ id: _id, ...leg }) => leg),
          spot,
          spotRangePct: 0.2,
        });
        if (!active) {
          return;
        }

        setPayoff(result);
        setPayoffMessage("");
      } catch (error) {
        if (active) {
          setPayoff(null);
          setPayoffMessage(error instanceof Error ? error.message : String(error));
        }
      }
    }, 500);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [legs, spot]);

  const strikes = useMemo(() => {
    return Array.from(new Set(chain.map((leg) => leg.strike))).sort(
      (left, right) => left - right,
    );
  }, [chain]);

  const atmStrike = useMemo(() => {
    if (spot > 0 && strikes.length > 0) {
      return strikes.reduce((best, strike) =>
        Math.abs(strike - spot) < Math.abs(best - spot) ? strike : best,
      );
    }

    if (spot > 0) {
      return Math.round(spot / 50) * 50;
    }

    return strikes[0] ?? 23400;
  }, [spot, strikes]);

  const payoffTraces = useMemo<Data[]>(() => {
    if (!payoff) {
      return [];
    }

    return [
      {
        fill: "tozeroy",
        fillcolor: "rgba(38, 166, 154, 0.18)",
        hoverinfo: "skip",
        line: { color: "rgba(38, 166, 154, 0)", width: 0 },
        mode: "lines",
        name: "Profit Zone",
        showlegend: false,
        type: "scatter",
        x: payoff.spot_range,
        y: payoff.pnl_at_expiry.map((value) => Math.max(value, 0)),
      } as Data,
      {
        fill: "tozeroy",
        fillcolor: "rgba(239, 83, 80, 0.18)",
        hoverinfo: "skip",
        line: { color: "rgba(239, 83, 80, 0)", width: 0 },
        mode: "lines",
        name: "Loss Zone",
        showlegend: false,
        type: "scatter",
        x: payoff.spot_range,
        y: payoff.pnl_at_expiry.map((value) => Math.min(value, 0)),
      } as Data,
      {
        line: { color: "#d1d4dc", width: 2 },
        mode: "lines",
        name: "P&L at Expiry",
        type: "scatter",
        x: payoff.spot_range,
        y: payoff.pnl_at_expiry,
      } as Data,
      ...(payoff.pnl_today ? [{
        line: { color: "#546e7a", dash: "dash", width: 2 },
        mode: "lines",
        name: "P&L Today",
        type: "scatter",
        x: payoff.spot_range,
        y: payoff.pnl_today,
      } as Data] : []),
    ];
  }, [payoff]);

  const payoffLayout = useMemo<Partial<Layout>>(() => {
    const x0 = payoff?.spot_range[0] ?? spot * 0.8;
    const x1 = payoff?.spot_range[payoff.spot_range.length - 1] ?? spot * 1.2;
    const shapes: Partial<Layout>["shapes"] = [
      {
        line: { color: "#546e7a", width: 1 },
        type: "line",
        x0,
        x1,
        y0: 0,
        y1: 0,
      },
      {
        line: { color: "#2196f3", width: 1.5 },
        type: "line",
        x0: spot,
        x1: spot,
        y0: 0,
        y1: 1,
        yref: "paper",
      },
      ...((payoff?.breakevens ?? []).map((breakeven) => ({
        line: { color: "#facc15", dash: "dot" as const, width: 1 },
        type: "line" as const,
        x0: breakeven,
        x1: breakeven,
        y0: 0,
        y1: 1,
        yref: "paper" as const,
      })) ?? []),
    ];

    return {
      ...baseLayout,
      height: 420,
      legend: { orientation: "h", x: 0, y: 1.08 },
      shapes,
      xaxis: axis2d("Spot at Expiry"),
      yaxis: axis2d("P&L (INR)"),
    };
  }, [payoff, spot]);

  const basisTraces = useMemo<Data[]>(() => {
    if (!basis || basis.dates.length === 0) {
      return [];
    }

    return [
      {
        marker: {
          color: basis.basis.map((value) => (value >= 0 ? "#26a69a" : "#ef5350")),
        },
        name: "Basis",
        type: "bar",
        x: basis.dates,
        y: basis.basis,
      } as Data,
      {
        line: { color: "#2196f3", width: 2 },
        mode: "lines",
        name: "Annualised Carry %",
        type: "scatter",
        x: basis.dates,
        y: basis.annualised_carry,
        yaxis: "y2",
      } as Data,
    ];
  }, [basis]);

  const basisLayout = useMemo<Partial<Layout>>(
    () => ({
      ...baseLayout,
      height: 340,
      legend: { orientation: "h", x: 0, y: 1.1 },
      title: { font: { color: "#f8fafc", size: 15 }, text: "NIFTY Futures Basis" },
      xaxis: axis2d("Date"),
      yaxis: axis2d("Basis"),
      yaxis2: {
        ...axis2d("Carry %"),
        overlaying: "y",
        side: "right",
      },
    }),
    [],
  );

  const addLeg = () => {
    if (!selectedExpiry || lotSize === null) {
      setPayoffMessage("Validated lot size unavailable");
      return;
    }

    const leg = makeLeg("BUY", "CE", atmStrike);
    if (leg === null) {
      setPayoffMessage("No observed market price for the selected leg");
      return;
    }
    setPreset("custom");
    setLegs((current) => [...current, leg]);
  };

  const applyPreset = (nextPreset: Preset) => {
    setPreset(nextPreset);
    if (nextPreset === "custom" || !selectedExpiry) {
      return;
    }

    const specs: Array<[LegAction, LegType, number]> =
      nextPreset === "long-straddle"
        ? [
            ["BUY", "CE", atmStrike],
            ["BUY", "PE", atmStrike],
          ]
        : nextPreset === "short-straddle"
          ? [
              ["SELL", "CE", atmStrike],
              ["SELL", "PE", atmStrike],
            ]
          : nextPreset === "bull-call"
            ? [
                ["BUY", "CE", atmStrike],
                ["SELL", "CE", atmStrike + 200],
              ]
            : nextPreset === "bear-put"
              ? [
                  ["BUY", "PE", atmStrike],
                  ["SELL", "PE", atmStrike - 200],
                ]
              : [
                  ["SELL", "PE", atmStrike - 100],
                  ["BUY", "PE", atmStrike - 200],
                  ["SELL", "CE", atmStrike + 100],
                  ["BUY", "CE", atmStrike + 200],
                ];

    const nextLegs = specs
      .map(([action, optionType, strike]) => makeLeg(action, optionType, strike))
      .filter((leg): leg is StrategyLeg => leg !== null);
    if (nextLegs.length !== specs.length) {
      setLegs([]);
      setPayoffMessage("Preset unavailable: one or more observed market prices are missing");
      return;
    }
    setLegs(nextLegs);
  };

  const makeLeg = (
    action: LegAction,
    optionType: LegType,
    strike: number,
  ): StrategyLeg | null => {
    const entryPrice = marketPrice(optionType, strike);
    if (entryPrice === null || lotSize === null) {
      return null;
    }
    return {
      action,
      entry_price: entryPrice,
      expiry: selectedExpiry,
      id: createId(),
      lot_size: lotSize,
      option_type: optionType,
      quantity: 1,
      strike,
      symbol: "NIFTY",
    };
  };

  const updateLeg = (id: string, updates: Partial<StrategyLeg>) => {
    setPreset("custom");
    setLegs((current) =>
      current.map((leg) => {
        if (leg.id !== id) {
          return leg;
        }

        const next = { ...leg, ...updates };
        if (
          updates.option_type !== undefined ||
          updates.strike !== undefined ||
          updates.expiry !== undefined
        ) {
          const observedPrice = marketPrice(next.option_type, next.strike);
          if (observedPrice !== null) {
            next.entry_price = observedPrice;
          }
        }
        return next;
      }),
    );
  };

  const removeLeg = (id: string) => {
    setPreset("custom");
    setLegs((current) => current.filter((leg) => leg.id !== id));
  };

  const marketPrice = (optionType: LegType, strike: number) => {
    if (optionType === "FUT") {
      return spot > 0 ? Number(spot.toFixed(2)) : null;
    }

    const leg = chain.find(
      (item) => item.option_type === optionType && Math.abs(item.strike - strike) < 0.01,
    );
    return leg?.ltp !== undefined && leg.ltp >= 0 ? Number(leg.ltp.toFixed(2)) : null;
  };

  return (
    <section
      style={{
        borderTop: "1px solid #1f2933",
        color: "#d1d4dc",
        display: "flex",
        flexDirection: "column",
        gap: 16,
        minWidth: 0,
        paddingTop: 20,
        width: "100%",
      }}
    >
      <div
        style={{
          alignItems: "center",
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          justifyContent: "space-between",
        }}
      >
        <h2 style={{ color: "#f8fafc", fontSize: 18, letterSpacing: 0, margin: 0 }}>
          Strategy Lab
        </h2>
        <div style={{ color: "#8a94a6", fontSize: 12 }}>
          Spot {spot > 0 ? numberFormatter.format(spot) : "waiting"}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gap: 10,
          gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
        }}
      >
        <select
          aria-label="Preset strategy"
          onChange={(event) => applyPreset(event.target.value as Preset)}
          style={controlStyle}
          value={preset}
        >
          {PRESETS.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
        <select
          aria-label="Strategy expiry"
          onChange={(event) => setSelectedExpiry(event.target.value)}
          style={controlStyle}
          value={selectedExpiry}
        >
          {expiries.map((expiry) => (
            <option key={expiry} value={expiry}>
              {expiry}
            </option>
          ))}
        </select>
        <button
          disabled={!selectedExpiry || lotSize === null}
          onClick={addLeg}
          style={buttonStyle(Boolean(selectedExpiry && lotSize !== null))}
          type="button"
        >
          Add Leg
        </button>
      </div>

      <div style={{ border: "1px solid #2a2e39", overflowX: "auto" }}>
        <table
          style={{
            borderCollapse: "collapse",
            fontSize: 12,
            minWidth: 860,
            tableLayout: "fixed",
            width: "100%",
          }}
        >
          <thead>
            <tr>
              {["Action", "Type", "Strike", "Expiry", "Qty", "Entry Price", "Remove"].map(
                (heading) => (
                  <th key={heading} style={headerCellStyle}>
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {legs.length > 0 ? (
              legs.map((leg) => (
                <tr key={leg.id}>
                  <td style={cellStyle}>
                    <div style={{ display: "flex", gap: 6 }}>
                      {(["BUY", "SELL"] as LegAction[]).map((action) => (
                        <button
                          key={action}
                          onClick={() => updateLeg(leg.id, { action })}
                          style={toggleStyle(leg.action === action)}
                          type="button"
                        >
                          {action}
                        </button>
                      ))}
                    </div>
                  </td>
                  <td style={cellStyle}>
                    <select
                      onChange={(event) =>
                        updateLeg(leg.id, { option_type: event.target.value as LegType })
                      }
                      style={tableControlStyle}
                      value={leg.option_type}
                    >
                      {(["CE", "PE", "FUT"] as LegType[]).map((optionType) => (
                        <option key={optionType} value={optionType}>
                          {optionType}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td style={cellStyle}>
                    <input
                      disabled={leg.option_type === "FUT"}
                      onChange={(event) =>
                        updateLeg(leg.id, { strike: Number(event.target.value) })
                      }
                      style={tableControlStyle}
                      type="number"
                      value={leg.strike}
                    />
                  </td>
                  <td style={cellStyle}>
                    <select
                      onChange={(event) => updateLeg(leg.id, { expiry: event.target.value })}
                      style={tableControlStyle}
                      value={leg.expiry}
                    >
                      {expiries.map((expiry) => (
                        <option key={expiry} value={expiry}>
                          {expiry}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td style={cellStyle}>
                    <input
                      min={1}
                      onChange={(event) =>
                        updateLeg(leg.id, { quantity: Math.max(1, Number(event.target.value)) })
                      }
                      style={tableControlStyle}
                      type="number"
                      value={leg.quantity}
                    />
                  </td>
                  <td style={cellStyle}>
                    <input
                      min={0}
                      onChange={(event) =>
                        updateLeg(leg.id, { entry_price: Math.max(0, Number(event.target.value)) })
                      }
                      style={tableControlStyle}
                      type="number"
                      value={leg.entry_price}
                    />
                  </td>
                  <td style={{ ...cellStyle, textAlign: "center" }}>
                    <button
                      aria-label="Remove leg"
                      onClick={() => removeLeg(leg.id)}
                      style={removeButtonStyle}
                      type="button"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={7} style={{ ...cellStyle, color: "#8a94a6" }}>
                  No legs yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <ChartBlock title="Payoff Diagram">
        {payoff ? (
          <>
            <PlotlyChart
              data={payoffTraces}
              layout={payoffLayout}
              style={{ height: 420 }}
            />
            <div
              style={{
                color: "#d1d4dc",
                display: "flex",
                flexWrap: "wrap",
                fontSize: 12,
                gap: "8px 16px",
                padding: "0 4px 8px",
              }}
            >
              {payoff.net_greeks ? (
                <>
                  <span>Delta: {payoff.net_greeks.delta.toFixed(3)}</span>
                  <span>Gamma: {payoff.net_greeks.gamma.toFixed(4)}</span>
                  <span>Theta: {payoff.net_greeks.theta.toFixed(2)}</span>
                  <span>Vega: {payoff.net_greeks.vega.toFixed(2)}</span>
                  <span style={{ color: "#7d8590" }} title="Never observed market data for dividend yield">
                    rate: {payoff.risk_free_rate_source ?? "n/a"} · dividend: {payoff.dividend_yield_source ?? "n/a"}
                  </span>
                </>
              ) : (
                <span style={{ color: "#facc15" }}>
                  Greeks/P&amp;L Today: unavailable ({payoff.analytical_status})
                </span>
              )}
              <span>Max Profit: {formatLimit(payoff.max_profit)}</span>
              <span>Max Loss: {formatLimit(payoff.max_loss)}</span>
              <span>
                Breakevens:{" "}
                {payoff.breakevens.length > 0
                  ? payoff.breakevens.map((item) => numberFormatter.format(item)).join(", ")
                  : "none"}
              </span>
            </div>
          </>
        ) : (
          <InlineMessage message={payoffMessage} />
        )}
      </ChartBlock>

      <ChartBlock title="NIFTY Futures Basis">
        {basis && basis.dates.length > 0 ? (
          <>
            <PlotlyChart
              data={basisTraces}
              layout={basisLayout}
              style={{ height: 340 }}
            />
            <div style={{ color: "#8a94a6", fontSize: 12, padding: "0 4px 8px" }}>
              Near expiry: {basis.near_expiry || "unknown"} | Current basis:{" "}
              {basis.current_basis === null ? "n/a" : numberFormatter.format(basis.current_basis)} |
              Current carry:{" "}
              {basis.current_carry === null ? "n/a" : `${basis.current_carry.toFixed(2)}%`}
            </div>
          </>
        ) : (
          <InlineMessage message={basisMessage || "No historical futures data"} />
        )}
      </ChartBlock>
    </section>
  );
}

function PlotlyChart({
  data,
  layout,
  style,
}: {
  data: Data[];
  layout: Partial<Layout>;
  style: React.CSSProperties;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let active = true;
    let resizeObserver: ResizeObserver | null = null;
    let plotly: Awaited<ReturnType<typeof loadPlotly>> | null = null;
    void loadPlotly().then((loaded) => {
      if (!active) {
        return;
      }
      plotly = loaded;
      loaded.react(container, data, layout, plotConfig);
      resizeObserver = new ResizeObserver(() => loaded.Plots.resize(container));
      resizeObserver.observe(container);
    });

    return () => {
      active = false;
      resizeObserver?.disconnect();
      plotly?.purge(container);
    };
  }, [data, layout]);

  return (
    <div
      ref={containerRef}
      style={{
        minWidth: 0,
        overflow: "hidden",
        width: "100%",
        ...style,
      }}
    />
  );
}

function ChartBlock({
  children,
  title,
}: {
  children: React.ReactNode;
  title: string;
}) {
  return (
    <div
      style={{
        background: "#0d0d0d",
        border: "1px solid #2a2e39",
        minWidth: 0,
        padding: "12px 12px 8px",
        width: "100%",
      }}
    >
      <h3
        style={{
          color: "#f8fafc",
          fontSize: 15,
          fontWeight: 700,
          letterSpacing: 0,
          margin: "0 0 8px",
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}

function InlineMessage({ message }: { message: string }) {
  return (
    <div
      style={{
        alignItems: "center",
        color: "#8a94a6",
        display: "flex",
        fontSize: 12,
        minHeight: 120,
      }}
    >
      {message}
    </div>
  );
}

function axis2d(title: string) {
  return {
    gridcolor: "#1f2933",
    linecolor: "#2a2e39",
    tickfont: { color: "#d1d4dc" },
    title: { font: { color: "#d1d4dc" }, text: title },
    zerolinecolor: "#2a2e39",
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

function toggleStyle(active: boolean): React.CSSProperties {
  return {
    ...buttonStyle(true),
    background: active ? "#1f2937" : "#101010",
    color: active ? "#ffffff" : "#8a94a6",
    height: 28,
    padding: "0 8px",
  };
}

function formatLimit(value: number | null) {
  if (value === null) {
    return "unlimited";
  }
  return numberFormatter.format(value);
}

function createId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
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

const tableControlStyle = {
  ...controlStyle,
  height: 30,
  padding: "0 8px",
} satisfies React.CSSProperties;

const headerCellStyle = {
  background: "#111827",
  borderBottom: "1px solid #2a2e39",
  color: "#9ca3af",
  fontWeight: 700,
  padding: "9px 8px",
  textAlign: "left",
} satisfies React.CSSProperties;

const cellStyle = {
  borderBottom: "1px solid #1f2933",
  padding: "8px",
  verticalAlign: "middle",
} satisfies React.CSSProperties;

const removeButtonStyle = {
  ...buttonStyle(true),
  background: "#1a0a0a",
  borderColor: "#3a1a1a",
  color: "#ef5350",
  height: 28,
  padding: "0 10px",
} satisfies React.CSSProperties;
