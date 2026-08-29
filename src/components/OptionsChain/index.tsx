import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import { useQuotesStore } from "../../store/quotes";
import { executableQuotePrice } from "../../market/quote";

type OptionType = "CE" | "PE";

type OptionLeg = {
  symbol: string;
  expiry: string;
  strike: number;
  option_type: OptionType;
  ltp: number;
  bid: number | null;
  ask: number | null;
  volume: number | null;
  oi: number | null;
  oi_change: number | null;
  iv: number | null;
  delta: number | null;
  gamma: number | null;
  theta: number | null;
  vega: number | null;
};

type ChainRow = {
  strike: number;
  call?: OptionLeg;
  put?: OptionLeg;
};

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

const priceFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

type OptionsChainProps = {
  spot?: number;
  symbol?: string;
};

export function OptionsChain({ spot: spotOverride, symbol = "NIFTY" }: OptionsChainProps) {
  const quoteSpot = useQuotesStore((state) => executableQuotePrice(state.quotes[symbol]));
  const spot = spotOverride ?? quoteSpot;
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState("");
  const [chain, setChain] = useState<OptionLeg[]>([]);
  const [pcr, setPcr] = useState<number | null>(null);
  const [message, setMessage] = useState("Loading expiries...");

  useEffect(() => {
    let active = true;

    const loadExpiries = async () => {
      try {
        const nextExpiries = await invoke<string[]>("get_expiries", {
          symbol,
        });
        if (!active) {
          return;
        }

        setExpiries(nextExpiries);
        setSelectedExpiry(nextExpiries[0] || "");
        setMessage(nextExpiries.length > 0 ? "" : "No expiries available");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };

    loadExpiries();

    return () => {
      active = false;
    };
  }, [symbol]);

  useEffect(() => {
    if (!selectedExpiry) {
      return;
    }

    let active = true;

    const loadChain = async () => {
      try {
        const [nextChain, nextPcr] = await Promise.all([
          invoke<OptionLeg[]>("get_chain", {
            expiry: selectedExpiry,
            symbol,
          }),
          invoke<number | null>("get_pcr", {
            expiry: selectedExpiry,
            symbol,
          }),
        ]);

        if (!active) {
          return;
        }

        setChain(nextChain);
        setPcr(nextPcr);
        setMessage(nextChain.length > 0 ? "" : "Waiting for chain data");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };

    loadChain();
    const timer = window.setInterval(loadChain, 3000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [selectedExpiry, symbol]);

  const rows = useMemo(() => {
    const byStrike = new Map<number, ChainRow>();

    for (const leg of chain) {
      const existing = byStrike.get(leg.strike) ?? { strike: leg.strike };
      if (leg.option_type === "CE") {
        existing.call = leg;
      } else {
        existing.put = leg;
      }
      byStrike.set(leg.strike, existing);
    }

    return Array.from(byStrike.values()).sort((left, right) => left.strike - right.strike);
  }, [chain]);

  const atmStrike = useMemo(() => {
    if (spot <= 0 || rows.length === 0) {
      return null;
    }

    return rows.reduce((best, row) => {
      const bestDistance = Math.abs(best - spot);
      const rowDistance = Math.abs(row.strike - spot);
      return rowDistance < bestDistance ? row.strike : best;
    }, rows[0].strike);
  }, [rows, spot]);

  return (
    <section
      style={{
        background: "#0d0d0d",
        color: "#d1d4dc",
        minWidth: 0,
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
          marginBottom: 10,
        }}
      >
        <select
          aria-label="Expiry"
          value={selectedExpiry}
          onChange={(event) => setSelectedExpiry(event.target.value)}
          style={{
            background: "#111827",
            border: "1px solid #2a2e39",
            borderRadius: 6,
            color: "#d1d4dc",
            fontFamily: "monospace",
            fontSize: 13,
            height: 34,
            minWidth: 150,
            padding: "0 10px",
          }}
        >
          {expiries.map((expiry) => (
            <option key={expiry} value={expiry}>
              {expiry}
            </option>
          ))}
        </select>
        <div
          style={{
            color: "#f8fafc",
            fontFamily: "monospace",
            fontSize: 16,
            fontWeight: 700,
          }}
        >
          PCR: {pcr === null ? "—" : priceFormatter.format(pcr)}
        </div>
      </div>

      <div
        style={{
          border: "1px solid #2a2e39",
          maxHeight: 600,
          overflowY: "auto",
        }}
      >
        <table
          style={{
            borderCollapse: "collapse",
            fontFamily: "monospace",
            fontSize: 13,
            minWidth: 1040,
            tableLayout: "fixed",
            width: "100%",
          }}
        >
          <thead>
            <tr>
              {["OI", "Volume", "IV", "Delta", "LTP", "Strike Price", "LTP", "Delta", "IV", "Volume", "OI"].map(
                (heading, index) => (
                  <th
                    key={`${heading}-${index}`}
                    style={{
                      background: "#111827",
                      borderBottom: "1px solid #2a2e39",
                      color: "#9ca3af",
                      fontWeight: 700,
                      padding: "9px 10px",
                      position: "sticky",
                      textAlign: index === 5 ? "center" : "right",
                      top: 0,
                      zIndex: 1,
                    }}
                  >
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isAtm = atmStrike !== null && row.strike === atmStrike;
              const callBackground = cellBackground(isAtm, spot > 0 && row.strike < spot, "call");
              const putBackground = cellBackground(isAtm, spot > 0 && row.strike > spot, "put");
              const strikeBackground = isAtm ? "#2a2a00" : "#101010";
              const callTooltip = greekTooltip(row.call);
              const putTooltip = greekTooltip(row.put);
              const rowTooltip = callTooltip ?? putTooltip;

              return (
                <tr key={row.strike} title={rowTooltip}>
                  <NumberCell background={callBackground} value={formatInteger(row.call?.oi)} />
                  <NumberCell background={callBackground} value={formatInteger(row.call?.volume)} />
                  <NumberCell background={callBackground} title={callTooltip} value={formatIv(row.call?.iv)} />
                  <NumberCell background={callBackground} title={callTooltip} value={formatDelta(row.call?.delta)} />
                  <NumberCell background={callBackground} value={formatPrice(row.call?.ltp)} />
                  <td
                    title={rowTooltip}
                    style={{
                      background: strikeBackground,
                      borderBottom: "1px solid #1f2933",
                      color: "#f8fafc",
                      fontWeight: 700,
                      padding: "8px 10px",
                      textAlign: "center",
                    }}
                  >
                    {integerFormatter.format(row.strike)}
                  </td>
                  <NumberCell background={putBackground} value={formatPrice(row.put?.ltp)} />
                  <NumberCell background={putBackground} title={putTooltip} value={formatDelta(row.put?.delta)} />
                  <NumberCell background={putBackground} title={putTooltip} value={formatIv(row.put?.iv)} />
                  <NumberCell background={putBackground} value={formatInteger(row.put?.volume)} />
                  <NumberCell background={putBackground} value={formatInteger(row.put?.oi)} />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {message ? (
        <div
          style={{
            color: "#8a94a6",
            fontSize: 12,
            marginTop: 8,
            minHeight: 16,
          }}
        >
          {message}
        </div>
      ) : null}
    </section>
  );
}

function NumberCell({
  background,
  title,
  value,
}: {
  background: string;
  title?: string;
  value: string;
}) {
  return (
    <td
      title={title}
      style={{
        background,
        borderBottom: "1px solid #1f2933",
        color: "#d1d4dc",
        padding: "8px 10px",
        textAlign: "right",
        whiteSpace: "nowrap",
      }}
    >
      {value}
    </td>
  );
}

function cellBackground(isAtm: boolean, isItm: boolean, side: "call" | "put") {
  if (isAtm) {
    return "#2a2a00";
  }

  if (!isItm) {
    return "#0d0d0d";
  }

  return side === "call" ? "#0a1a0a" : "#1a0a0a";
}

function formatInteger(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return integerFormatter.format(value);
}

function formatPrice(value: number | undefined) {
  if (value === undefined) {
    return "—";
  }

  return priceFormatter.format(value);
}

function formatIv(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatDelta(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return value.toFixed(3);
}

function greekTooltip(leg?: OptionLeg) {
  if (!leg) {
    return undefined;
  }

  return [
    `Gamma: ${formatTooltipNumber(leg.gamma, 5)}`,
    `Theta: ${formatTooltipNumber(leg.theta, 2)}`,
    `Vega: ${formatTooltipNumber(leg.vega, 2)}`,
  ].join(" | ");
}

function formatTooltipNumber(value: number | null | undefined, digits: number) {
  if (value === null || value === undefined) {
    return "—";
  }

  return value.toFixed(digits);
}
