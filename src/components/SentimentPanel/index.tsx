import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";

type SentimentScore = {
  score: number | null;
  label: string | null;
  article_count: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  high_impact_count: number;
  breakdown: {
    equity?: number | null;
    rates?: number | null;
    currency?: number | null;
    macro?: number | null;
  };
};

type SentimentHeadline = {
  source: string;
  title: string;
  sentiment: "bullish" | "bearish" | "neutral";
  confidence: number;
  impact: "high" | "medium" | "low";
  age_minutes: number;
  url: string;
  published_at?: string;
};

type SentimentResponse = {
  score?: SentimentScore;
  headlines?: SentimentHeadline[];
  announcements?: {
    symbol: string;
    subject: string;
    impact: "high" | "medium" | "low";
    published_at: string;
  }[];
  computed_at?: number;
};

const numberFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function SentimentPanel() {
  const [data, setData] = useState<SentimentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;

    const loadSentiment = async () => {
      try {
        const result = await invoke<SentimentResponse>("get_sentiment");
        if (!active) {
          return;
        }
        setData(result);
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

    loadSentiment();
    const timer = window.setInterval(loadSentiment, 120_000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const score = data?.score;
  const scoreValue = asNumber(score?.score);
  const headlines = data?.headlines ?? [];
  const scoreColor = scoreValue === null ? "#8a94a6" : sentimentScoreColor(scoreValue);

  const subLabels = useMemo(() => {
    if (!score) {
      return "↑ 0 bullish  ↓ 0 bearish  → 0 neutral";
    }
    return `↑ ${score.bullish_count} bullish  ↓ ${score.bearish_count} bearish  → ${score.neutral_count} neutral`;
  }, [score]);

  return (
    <section style={wrapperStyle} aria-label="Market sentiment">
      <div style={scoreCardStyle}>
        <div style={labelStyle}>Sentiment</div>
        <div style={{ ...scoreValueStyle, color: scoreColor }}>
          {scoreValue === null ? "--" : numberFormatter.format(Math.round(scoreValue))}
        </div>
        <div style={{ ...scoreLabelStyle, color: scoreColor }}>
          {score?.label ?? "Unavailable"}
        </div>
        <div style={subvalueStyle}>{subLabels}</div>
        <div style={breakdownStyle}>
          Equity: {formatBreakdown(score?.breakdown?.equity)} | Rates:{" "}
          {formatBreakdown(score?.breakdown?.rates)} | FX:{" "}
          {formatBreakdown(score?.breakdown?.currency)} | Macro:{" "}
          {formatBreakdown(score?.breakdown?.macro)}
        </div>
      </div>

      <div style={tickerCardStyle}>
        <div style={tickerHeaderStyle}>
          <span style={labelStyle}>News</span>
          <span style={tickerMetaStyle}>
            {loading && data === null ? "Fetching news..." : `${headlines.length} headlines`}
          </span>
        </div>

        <div style={tickerListStyle}>
          {headlines.length === 0 ? (
            <div style={emptyStyle}>
              {loading ? "Fetching news..." : "No recent headlines"}
            </div>
          ) : (
            headlines.map((headline, index) => (
              <button
                key={`${headline.url}-${index}`}
                style={headlineRowStyle(headline)}
                title={headline.title}
                type="button"
                onClick={() => openHeadline(headline.url, setMessage)}
              >
                <span style={dotStyle(headline)} />
                <span style={headlineTextStyle}>
                  {truncateHeadline(headline.title)}
                </span>
                <span style={sourceStyle}>
                  {sourceLabel(headline.source)} {formatAge(headline.age_minutes)}
                </span>
              </button>
            ))
          )}
        </div>

        {message ? <div style={messageStyle}>{message}</div> : null}
      </div>
    </section>
  );
}

async function openHeadline(
  url: string,
  setMessage: (message: string) => void,
) {
  if (!url) {
    return;
  }
  try {
    await open(url);
    setMessage("");
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error));
  }
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sentimentScoreColor(value: number) {
  if (value < 35) {
    return "#ef5350";
  }
  if (value <= 65) {
    return "#facc15";
  }
  return "#22c55e";
}

function sentimentDotColor(sentiment: SentimentHeadline["sentiment"]) {
  if (sentiment === "bullish") {
    return "#22c55e";
  }
  if (sentiment === "bearish") {
    return "#ef5350";
  }
  return "#8a94a6";
}

function formatBreakdown(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return numberFormatter.format(Math.round(number));
}

function formatAge(ageMinutes: number) {
  if (!Number.isFinite(ageMinutes) || ageMinutes < 0) {
    return "--";
  }
  if (ageMinutes < 60) {
    return `${Math.round(ageMinutes)}m`;
  }
  return `${Math.floor(ageMinutes / 60)}h`;
}

function sourceLabel(source: string) {
  const labels: Record<string, string> = {
    business_standard: "BS",
    cnbc_tv18: "CNBC",
    et_markets: "ET",
    hindu_business: "HBL",
    livemint: "Mint",
    moneycontrol: "MC",
    news18_markets: "N18",
  };
  return labels[source] ?? source.slice(0, 4).toUpperCase();
}

function truncateHeadline(value: string) {
  if (value.length <= 80) {
    return value;
  }
  return `${value.slice(0, 77)}...`;
}

function headlineRowStyle(headline: SentimentHeadline) {
  const stale = headline.age_minutes > 240;
  return {
    ...headlineButtonStyle,
    borderLeftColor: headline.impact === "high" ? "#facc15" : "transparent",
    opacity: stale ? 0.55 : 1,
  } satisfies CSSProperties;
}

function dotStyle(headline: SentimentHeadline) {
  return {
    background: headline.age_minutes > 240 ? "#6b7280" : sentimentDotColor(headline.sentiment),
    borderRadius: "50%",
    flex: "0 0 auto",
    height: 8,
    width: 8,
  } satisfies CSSProperties;
}

const wrapperStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "minmax(260px, 0.72fr) minmax(420px, 1.28fr)",
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const scoreCardStyle = {
  background: "#1a1a2e",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 8,
  minHeight: 154,
  minWidth: 0,
  padding: "14px",
} satisfies CSSProperties;

const tickerCardStyle = {
  background: "#1a1a2e",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 8,
  minHeight: 154,
  minWidth: 0,
  padding: "12px",
} satisfies CSSProperties;

const labelStyle = {
  color: "#8a94a6",
  fontSize: 11,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const scoreValueStyle = {
  fontSize: 40,
  fontWeight: 700,
  letterSpacing: 0,
  lineHeight: 1,
} satisfies CSSProperties;

const scoreLabelStyle = {
  fontSize: 15,
  fontWeight: 700,
  lineHeight: 1.2,
} satisfies CSSProperties;

const subvalueStyle = {
  color: "#d1d4dc",
  fontSize: 12,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const breakdownStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const tickerHeaderStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
  gap: 12,
} satisfies CSSProperties;

const tickerMetaStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.2,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const tickerListStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
  height: 120,
  overflowY: "auto",
  paddingRight: 4,
} satisfies CSSProperties;

const headlineButtonStyle = {
  alignItems: "center",
  background: "#101827",
  border: "1px solid #23283a",
  borderLeft: "3px solid transparent",
  borderRadius: 6,
  color: "#d1d4dc",
  cursor: "pointer",
  display: "grid",
  fontFamily: "monospace",
  fontSize: 12,
  gap: 8,
  gridTemplateColumns: "8px minmax(0, 1fr) auto",
  lineHeight: 1.25,
  minHeight: 30,
  padding: "6px 8px",
  textAlign: "left",
  width: "100%",
} satisfies CSSProperties;

const headlineTextStyle = {
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const sourceStyle = {
  color: "#8a94a6",
  fontSize: 11,
  lineHeight: 1.2,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const emptyStyle = {
  alignItems: "center",
  color: "#8a94a6",
  display: "flex",
  fontSize: 12,
  height: "100%",
  justifyContent: "center",
} satisfies CSSProperties;

const messageStyle = {
  color: "#facc15",
  fontSize: 12,
  lineHeight: 1.3,
} satisfies CSSProperties;
