import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";

type Direction = "up" | "down" | "flat";

type AssetQuote = {
  price?: number | null;
  change?: number | null;
  change_pct?: number | null;
  direction?: Direction;
  stale?: boolean;
  stale_minutes?: number | null;
  last_updated?: string | null;
  premium?: number | null;
  premium_pct?: number | null;
};

type CrossAssetSnapshot = {
  usdinr?: AssetQuote;
  brent?: AssetQuote;
  dow_futures?: AssetQuote;
  sp500_futures?: AssetQuote;
  us_10y_yield?: AssetQuote;
  gold?: AssetQuote;
  nikkei?: AssetQuote;
  sgx_nifty?: AssetQuote;
  gift_nifty?: AssetQuote;
  fetched_at?: string;
  staleness_warning?: boolean;
};

type PremarketBias = {
  bias?: "bullish" | "bearish" | "neutral";
  score?: number;
  signals?: string[];
  summary?: string;
};

type CrossAssetResponse = {
  snapshot?: CrossAssetSnapshot;
  bias?: PremarketBias;
  computed_at?: number;
  error?: string;
};

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function CrossAssetPanel() {
  const [data, setData] = useState<CrossAssetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;

    const loadCrossAsset = async () => {
      try {
        const result = await invoke<CrossAssetResponse>("get_cross_asset");
        if (!active) {
          return;
        }
        setData(result);
        setMessage(result.error ?? "");
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

    loadCrossAsset();
    const timer = window.setInterval(loadCrossAsset, 60_000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const snapshot = data?.snapshot;
  const bias = data?.bias;
  const banner = useMemo(() => biasBannerText(bias), [bias]);
  const niftyContext = useMemo(
    () => formatNiftyContext(snapshot?.gift_nifty, snapshot?.sgx_nifty),
    [snapshot?.gift_nifty, snapshot?.sgx_nifty],
  );

  if (loading && data === null) {
    return (
      <section style={wrapperStyle} aria-label="Cross asset loading">
        <div style={loadingTextStyle}>Loading global data...</div>
        <div style={rowStyle}>
          {["USD/INR", "Brent", "Dow Fut", "S&P Fut", "US 10Y", "Gold", "Nikkei"].map(
            (label) => (
              <SkeletonCard key={label} label={label} />
            ),
          )}
        </div>
      </section>
    );
  }

  return (
    <section style={wrapperStyle} aria-label="Cross asset dashboard">
      <div style={rowStyle}>
        <AssetCard
          label="USD/INR"
          quote={snapshot?.usdinr}
          value={formatNumber(snapshot?.usdinr?.price, 2)}
          valueColor={usdinrColor(snapshot?.usdinr?.price)}
          subvalue={formatChangeLine(snapshot?.usdinr, 2)}
        />
        <AssetCard
          label="Brent Crude"
          quote={snapshot?.brent}
          value={formatCurrency(snapshot?.brent?.price)}
          valueColor={brentColor(snapshot?.brent?.change_pct)}
          subvalue={formatPctLine(snapshot?.brent)}
        />
        <AssetCard
          label="Dow Futures"
          quote={snapshot?.dow_futures}
          value={formatIndex(snapshot?.dow_futures?.price)}
          valueColor={futuresColor(snapshot?.dow_futures?.change_pct)}
          subvalue={formatPctLine(snapshot?.dow_futures)}
        />
        <AssetCard
          label="S&P Futures"
          quote={snapshot?.sp500_futures}
          value={formatIndex(snapshot?.sp500_futures?.price)}
          valueColor={futuresColor(snapshot?.sp500_futures?.change_pct)}
          subvalue={formatPctLine(snapshot?.sp500_futures)}
        />
        <AssetCard
          label="US 10Y Yield"
          quote={snapshot?.us_10y_yield}
          value={formatYield(snapshot?.us_10y_yield?.price)}
          valueColor={yieldColor(snapshot?.us_10y_yield?.price)}
          subvalue={formatBpLine(snapshot?.us_10y_yield)}
        />
        <AssetCard
          label="Gold"
          quote={snapshot?.gold}
          value={formatCurrency(snapshot?.gold?.price)}
          valueColor={signedColor(snapshot?.gold?.change_pct)}
          subvalue={formatPctLine(snapshot?.gold)}
        />
        <AssetCard
          label="Nikkei"
          quote={snapshot?.nikkei}
          value={formatIndex(snapshot?.nikkei?.price)}
          valueColor={asiaColor(snapshot?.nikkei?.change_pct)}
          subvalue={formatPctLine(snapshot?.nikkei)}
        />
      </div>

      <div style={biasBannerStyle(bias?.bias)}>
        <div style={biasTextStyle}>{banner}</div>
        <div style={contextTextStyle}>{niftyContext}</div>
      </div>

      {message ? <div style={messageStyle}>{message}</div> : null}
    </section>
  );
}

function AssetCard({
  label,
  quote,
  subvalue,
  value,
  valueColor,
}: {
  label: string;
  quote?: AssetQuote;
  subvalue: string;
  value: string;
  valueColor: string;
}) {
  return (
    <div style={cardStyle(quote)} title={staleTitle(quote)}>
      <div style={labelRowStyle}>
        <span style={labelStyle}>{label}</span>
        <span style={{ ...directionStyle, color: directionColor(quote?.direction) }}>
          {directionSymbol(quote?.direction)}
        </span>
      </div>
      <div style={{ ...valueStyle, color: valueColor }}>{value}</div>
      <div style={subvalueStyle}>{subvalue}</div>
    </div>
  );
}

function SkeletonCard({ label }: { label: string }) {
  return (
    <div style={cardStyle()}>
      <div style={labelStyle}>{label}</div>
      <div style={skeletonLineStyle} />
      <div style={{ ...skeletonLineStyle, height: 8, width: "52%" }} />
    </div>
  );
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatNumber(value: number | null | undefined, digits: number) {
  const number = asNumber(value);
  if (number === null) {
    return "—";
  }
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(number);
}

function formatCurrency(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "—";
  }
  return `$${numberFormatter.format(number)}`;
}

function formatIndex(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "—";
  }
  return integerFormatter.format(Math.round(number));
}

function formatYield(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "—";
  }
  return `${number.toFixed(2)}%`;
}

function formatChangeLine(quote?: AssetQuote, digits = 2) {
  const change = asNumber(quote?.change);
  const pct = asNumber(quote?.change_pct);
  if (change === null && pct === null) {
    return "—";
  }
  const changeText =
    change === null
      ? "—"
      : `${change > 0 ? "+" : ""}${change.toFixed(digits)}`;
  return `${directionSymbol(quote?.direction)} ${changeText} (${formatPct(pct)})`;
}

function formatPctLine(quote?: AssetQuote) {
  return `${directionSymbol(quote?.direction)} ${formatPct(quote?.change_pct)}`;
}

function formatBpLine(quote?: AssetQuote) {
  const bp = asNumber(quote?.change);
  if (bp === null) {
    return "—";
  }
  return `${directionSymbol(quote?.direction)} ${bp > 0 ? "+" : ""}${bp.toFixed(1)} bp`;
}

function formatPct(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "—";
  }
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}%`;
}

function formatPremium(quote?: AssetQuote) {
  const premium = asNumber(quote?.premium);
  const premiumPct = asNumber(quote?.premium_pct);
  if (premium === null) {
    return "—";
  }
  return `${premium > 0 ? "+" : ""}${premium.toFixed(0)} pts (${formatPct(premiumPct)})`;
}

function formatNiftyContext(gift?: AssetQuote, sgx?: AssetQuote) {
  const giftPrice = asNumber(gift?.price);
  const sgxPrice = asNumber(sgx?.price);
  const giftText =
    giftPrice === null
      ? "GIFT NIFTY —"
      : `GIFT NIFTY ${formatIndex(giftPrice)} ${formatPremium(gift)}`;
  const sgxText =
    sgxPrice === null
      ? "SGX NIFTY —"
      : `SGX NIFTY ${formatIndex(sgxPrice)} ${formatPremium(sgx)}`;
  return `${giftText} | ${sgxText}`;
}

function biasBannerText(bias?: PremarketBias) {
  const rawBias = bias?.bias ?? "neutral";
  const label = `${rawBias.toUpperCase()} OPEN`;
  const rawSummary = bias?.summary?.trim() || "mixed global signals";
  const summary = rawSummary.replace(/^(bullish|bearish|neutral)\s+open:\s*/i, "");
  return `${label}: ${summary}`;
}

function usdinrColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number < 83.5) {
    return "#22c55e";
  }
  if (number > 84.5) {
    return "#ef5350";
  }
  return "#facc15";
}

function brentColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 2) {
    return "#ef5350";
  }
  if (number < -2) {
    return "#22c55e";
  }
  return "#facc15";
}

function futuresColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 0.3) {
    return "#22c55e";
  }
  if (number < -0.3) {
    return "#ef5350";
  }
  return "#facc15";
}

function yieldColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 4.5) {
    return "#ef5350";
  }
  if (number < 4.0) {
    return "#22c55e";
  }
  return "#facc15";
}

function asiaColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 0.5) {
    return "#22c55e";
  }
  if (number < -0.5) {
    return "#ef5350";
  }
  return "#facc15";
}

function signedColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null || number === 0) {
    return "#d1d4dc";
  }
  return number > 0 ? "#22c55e" : "#ef5350";
}

function directionSymbol(direction?: Direction) {
  if (direction === "up") {
    return "↑";
  }
  if (direction === "down") {
    return "↓";
  }
  return "→";
}

function directionColor(direction?: Direction) {
  if (direction === "up") {
    return "#22c55e";
  }
  if (direction === "down") {
    return "#ef5350";
  }
  return "#facc15";
}

function staleTitle(quote?: AssetQuote) {
  if (!quote?.stale) {
    return quote?.last_updated ? `Last updated ${quote.last_updated}` : undefined;
  }
  return quote.last_updated
    ? `Stale data: last updated ${quote.last_updated}`
    : "Stale data";
}

const wrapperStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const rowStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "repeat(7, minmax(145px, 1fr))",
  minWidth: 0,
  overflowX: "auto",
  paddingBottom: 2,
  width: "100%",
} satisfies CSSProperties;

function cardStyle(quote?: AssetQuote) {
  return {
    background: "#1a1a2e",
    border: `1px solid ${quote?.stale ? "#f97316" : "#2a2e39"}`,
    borderRadius: 8,
    boxSizing: "border-box",
    color: "#d1d4dc",
    display: "flex",
    flexDirection: "column",
    gap: 8,
    justifyContent: "space-between",
    minHeight: 108,
    minWidth: 145,
    padding: "12px",
  } satisfies CSSProperties;
}

const labelRowStyle = {
  alignItems: "center",
  display: "flex",
  gap: 8,
  justifyContent: "space-between",
} satisfies CSSProperties;

const labelStyle = {
  color: "#8a94a6",
  fontSize: 11,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const directionStyle = {
  fontSize: 16,
  fontWeight: 700,
  lineHeight: 1,
} satisfies CSSProperties;

const valueStyle = {
  color: "#f8fafc",
  fontSize: 22,
  fontWeight: 700,
  letterSpacing: 0,
  lineHeight: 1.1,
  minHeight: 25,
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

function biasBannerStyle(bias?: PremarketBias["bias"]) {
  const color =
    bias === "bullish" ? "#22c55e" : bias === "bearish" ? "#ef5350" : "#facc15";
  return {
    alignItems: "center",
    background: "#101827",
    border: `1px solid ${color}`,
    borderRadius: 8,
    boxSizing: "border-box",
    color: "#f8fafc",
    display: "flex",
    flexWrap: "wrap",
    gap: "8px 14px",
    justifyContent: "space-between",
    minHeight: 44,
    padding: "10px 12px",
    width: "100%",
  } satisfies CSSProperties;
}

const biasTextStyle = {
  fontSize: 13,
  fontWeight: 700,
  lineHeight: 1.35,
  minWidth: 220,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const contextTextStyle = {
  color: "#cbd5e1",
  fontSize: 11,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const loadingTextStyle = {
  color: "#8a94a6",
  fontSize: 12,
  lineHeight: 1.3,
} satisfies CSSProperties;

const skeletonLineStyle = {
  background: "#2a2e39",
  borderRadius: 4,
  height: 18,
  width: "74%",
} satisfies CSSProperties;

const messageStyle = {
  color: "#facc15",
  fontSize: 12,
  lineHeight: 1.3,
} satisfies CSSProperties;
