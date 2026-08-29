import {
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import { useQuotesStore } from "../../store/quotes";
import { executableQuotePrice } from "../../market/quote";

type OptionsSignals = {
  gex: number | null;
  gex_flip: number | null;
  max_pain: number | null;
  iv_rank: number | null;
  iv_percentile: number | null;
  atm_iv: number | null;
  expected_move: number | null;
  expected_move_pct: number | null;
  pcr: number | null;
  pcr_trend: number | null;
  call_oi_peak: number | null;
  put_oi_peak: number | null;
  total_call_oi: number;
  total_put_oi: number;
};

type FiiDiiSignals = {
  date?: string;
  error?: string;
  fii_net_cash?: number | null;
  dii_net_cash?: number | null;
  fii_net_futures?: number | null;
  fii_buy_cash?: number | null;
  fii_sell_cash?: number | null;
  dii_buy_cash?: number | null;
  dii_sell_cash?: number | null;
};

type SignalsResponse = {
  options?: OptionsSignals;
  market?: {
    india_vix?: number | null;
    fii_dii?: FiiDiiSignals;
  };
  expiry?: string;
  computed_at?: number;
};

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

type SignalsPanelProps = {
  spot?: number;
  symbol?: string;
};

export function SignalsPanel({ spot: spotOverride, symbol = "NIFTY" }: SignalsPanelProps) {
  const quoteSpot = useQuotesStore((state) => executableQuotePrice(state.quotes[symbol]));
  const spot = spotOverride ?? quoteSpot;
  const [signals, setSignals] = useState<SignalsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;

    const loadSignals = async () => {
      try {
        const result = await invoke<SignalsResponse>("get_signals", {
          symbol,
        });
        if (!active) {
          return;
        }

        setSignals(result);
        setMessage("");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadSignals();
    const timer = window.setInterval(loadSignals, 10_000);
    const unlisten = listen("chain_update", () => {
      loadSignals();
    });

    return () => {
      active = false;
      window.clearInterval(timer);
      unlisten.then((dispose) => dispose()).catch(() => undefined);
    };
  }, [symbol]);

  const options = signals?.options;
  const hasOptionData =
    ((options?.total_call_oi ?? 0) + (options?.total_put_oi ?? 0)) > 0;
  const gex = hasOptionData ? asNumber(options?.gex) : null;
  const ivRank = hasOptionData ? asNumber(options?.iv_rank) : null;
  const maxPain = hasOptionData ? asNumber(options?.max_pain) : null;
  const expectedMove = hasOptionData ? asNumber(options?.expected_move) : null;
  const expectedMovePct = hasOptionData
    ? asNumber(options?.expected_move_pct)
    : null;
  const pcr = hasOptionData ? asNumber(options?.pcr) : null;
  const pcrTrend = hasOptionData ? asNumber(options?.pcr_trend) : null;
  const indiaVix = asNumber(signals?.market?.india_vix);
  const fiiDii = signals?.market?.fii_dii;
  const fiiNet = asNumber(fiiDii?.fii_net_cash);
  const diiNet = asNumber(fiiDii?.dii_net_cash);

  const maxPainDistance = useMemo(() => {
    if (maxPain === null || spot <= 0) {
      return null;
    }
    return maxPain - spot;
  }, [maxPain, spot]);

  if (loading && signals === null) {
    return (
      <section style={rowStyle} aria-label="Signals loading">
        {["GEX", "IV Rank", "Max Pain", "Expected Move", "PCR", "India VIX", "FII/DII"].map(
          (label) => (
            <SkeletonCard key={label} label={label} />
          ),
        )}
      </section>
    );
  }

  return (
    <section style={wrapperStyle}>
      <div style={rowStyle} aria-label="Market signals">
        <SignalCard
          label="GEX"
          title="Positive GEX: market tends to mean-revert. Negative GEX: moves get amplified."
          value={formatCrores(gex)}
          valueColor={signedColor(gex)}
          subvalue={`Flip: ${formatStrike(hasOptionData ? options?.gex_flip : null)}`}
        />

        <SignalCard
          label="IV Rank"
          value={formatRank(ivRank)}
          valueColor={ivRankColor(ivRank)}
          subvalue={ivRankLabel(ivRank)}
        >
          <div style={barTrackStyle}>
            <div
              style={{
                ...barFillStyle,
                background: ivRankColor(ivRank),
                width: `${ivRank === null ? 0 : Math.max(0, Math.min(100, ivRank))}%`,
              }}
            />
          </div>
        </SignalCard>

        <SignalCard
          label="Max Pain"
          title="Strike where option sellers lose least at expiry"
          value={formatStrike(maxPain)}
          subvalue={formatDistance(maxPainDistance)}
        />

        <SignalCard
          label="Expected Move"
          title={`Market-implied move by ${signals?.expiry ?? "nearest expiry"} based on ATM IV`}
          value={formatMove(expectedMove)}
          subvalue={formatMovePct(expectedMovePct)}
        />

        <SignalCard
          label="PCR"
          value={formatDecimal(pcr, 2)}
          valueColor={pcrColor(pcr)}
          subvalue={`5p avg: ${formatDecimal(pcrTrend, 2)}`}
        >
          <span style={{ color: pcrArrowColor(pcr, pcrTrend), fontSize: 18 }}>
            {pcrArrow(pcr, pcrTrend)}
          </span>
        </SignalCard>

        <SignalCard
          label="India VIX"
          value={formatDecimal(indiaVix, 2)}
          valueColor={vixColor(indiaVix)}
          subvalue={vixLabel(indiaVix)}
        />

        <div style={cardStyle} title={fiiDii?.error}>
          <div style={labelStyle}>FII/DII</div>
          <div style={flowRowStyle}>
            <span>FII:</span>
            <strong style={{ color: signedColor(fiiNet) }}>
              {formatFlow(fiiNet)}
            </strong>
          </div>
          <div style={flowRowStyle}>
            <span>DII:</span>
            <strong style={{ color: signedColor(diiNet) }}>
              {formatFlow(diiNet)}
            </strong>
          </div>
          <div style={subvalueStyle}>{fiiDii?.date || "—"}</div>
        </div>
      </div>

      {message ? <div style={messageStyle}>{message}</div> : null}
    </section>
  );
}

function SignalCard({
  children,
  label,
  subvalue,
  title,
  value,
  valueColor = "#f8fafc",
}: {
  children?: ReactNode;
  label: string;
  subvalue?: string;
  title?: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div style={cardStyle} title={title}>
      <div style={labelStyle}>{label}</div>
      <div style={valueRowStyle}>
        <div style={{ ...valueStyle, color: valueColor }}>{value}</div>
        {children}
      </div>
      {subvalue ? <div style={subvalueStyle}>{subvalue}</div> : null}
    </div>
  );
}

function SkeletonCard({ label }: { label: string }) {
  return (
    <div style={cardStyle}>
      <div style={labelStyle}>{label}</div>
      <div style={skeletonLineStyle} />
      <div style={{ ...skeletonLineStyle, height: 8, width: "54%" }} />
    </div>
  );
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatCrores(value: number | null) {
  if (value === null) {
    return "—";
  }
  const sign = value < 0 ? "-" : "";
  return `${sign}₹${Math.abs(value).toFixed(1)}Cr`;
}

function formatStrike(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null || number <= 0) {
    return "—";
  }
  return `₹${integerFormatter.format(number)}`;
}

function formatRank(value: number | null) {
  if (value === null) {
    return "—";
  }
  return integerFormatter.format(Math.round(value));
}

function formatMove(value: number | null) {
  if (value === null || value <= 0) {
    return "—";
  }
  return `±₹${integerFormatter.format(value)}`;
}

function formatMovePct(value: number | null) {
  if (value === null || value <= 0) {
    return "—";
  }
  return `±${value.toFixed(1)}%`;
}

function formatDecimal(value: number | null, digits: number) {
  if (value === null) {
    return "—";
  }
  return value.toFixed(digits);
}

function formatDistance(value: number | null) {
  if (value === null) {
    return "—";
  }
  const rounded = Math.round(value);
  if (rounded === 0) {
    return "(0 pts)";
  }
  return `(${rounded > 0 ? "+" : ""}${integerFormatter.format(rounded)} pts)`;
}

function formatFlow(value: number | null) {
  if (value === null) {
    return "—";
  }
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}₹${integerFormatter.format(Math.abs(value))}Cr`;
}

function signedColor(value: number | null) {
  if (value === null || value === 0) {
    return "#d1d4dc";
  }
  return value > 0 ? "#22c55e" : "#ef5350";
}

function ivRankColor(value: number | null) {
  if (value === null) {
    return "#8a94a6";
  }
  if (value < 30) {
    return "#22c55e";
  }
  if (value <= 70) {
    return "#facc15";
  }
  return "#ef5350";
}

function ivRankLabel(value: number | null) {
  if (value === null) {
    return "—";
  }
  if (value < 30) {
    return "Cheap";
  }
  if (value <= 70) {
    return "Normal";
  }
  return "Expensive";
}

function pcrColor(value: number | null) {
  if (value === null) {
    return "#8a94a6";
  }
  if (value > 1.0) {
    return "#22c55e";
  }
  if (value >= 0.7) {
    return "#facc15";
  }
  return "#ef5350";
}

function pcrArrow(value: number | null, trend: number | null) {
  if (value === null || trend === null) {
    return "—";
  }
  return value >= trend ? "↑" : "↓";
}

function pcrArrowColor(value: number | null, trend: number | null) {
  if (value === null || trend === null) {
    return "#8a94a6";
  }
  return value >= trend ? "#22c55e" : "#ef5350";
}

function vixColor(value: number | null) {
  if (value === null) {
    return "#8a94a6";
  }
  if (value < 13) {
    return "#22c55e";
  }
  if (value <= 20) {
    return "#facc15";
  }
  return "#ef5350";
}

function vixLabel(value: number | null) {
  if (value === null) {
    return "—";
  }
  if (value < 13) {
    return "Low";
  }
  if (value <= 20) {
    return "Normal";
  }
  return "High";
}

const wrapperStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const rowStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "repeat(7, minmax(150px, 1fr))",
  minWidth: 0,
  overflowX: "auto",
  paddingBottom: 2,
  width: "100%",
} satisfies CSSProperties;

const cardStyle = {
  background: "#1a1a2e",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 8,
  justifyContent: "space-between",
  minHeight: 118,
  minWidth: 150,
  padding: "12px",
} satisfies CSSProperties;

const labelStyle = {
  color: "#8a94a6",
  fontSize: 11,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const valueRowStyle = {
  alignItems: "center",
  display: "flex",
  gap: 8,
  justifyContent: "space-between",
  minHeight: 32,
} satisfies CSSProperties;

const valueStyle = {
  color: "#f8fafc",
  fontSize: 23,
  fontWeight: 700,
  letterSpacing: 0,
  lineHeight: 1.1,
  minWidth: 0,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const subvalueStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.3,
  minHeight: 14,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const barTrackStyle = {
  background: "#0f172a",
  borderRadius: 6,
  height: 7,
  overflow: "hidden",
  width: "100%",
} satisfies CSSProperties;

const barFillStyle = {
  borderRadius: 6,
  height: "100%",
  transition: "width 160ms ease",
} satisfies CSSProperties;

const flowRowStyle = {
  alignItems: "baseline",
  display: "flex",
  fontSize: 13,
  gap: 6,
  justifyContent: "space-between",
  lineHeight: 1.2,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const skeletonLineStyle = {
  background: "#2a2e39",
  borderRadius: 4,
  height: 18,
  width: "76%",
} satisfies CSSProperties;

const messageStyle = {
  color: "#facc15",
  fontSize: 12,
} satisfies CSSProperties;
