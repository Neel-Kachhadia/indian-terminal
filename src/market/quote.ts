export type QuoteSource =
  | "ANGEL_STREAM"
  | "NSE_CHAIN"
  | "CACHE"
  | "HISTORICAL"
  | "DERIVED";

export type QuoteFreshness = "FRESH" | "STALE" | "UNKNOWN";
export type QuoteQuality =
  | "EXECUTABLE_STREAM"
  | "REFERENCE_ONLY"
  | "INCOMPLETE"
  | "INVALID";
export type MarketDataState =
  | "LIVE"
  | "DEGRADED"
  | "STALE"
  | "DISCONNECTED"
  | "STARTING";
export type QuoteDisplayStatus = "live" | "degraded" | "stale" | "none";

export type Tick = {
  schema_version: 1;
  symbol: string;
  token: string;
  ltp: number;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  oi: number | null;
  bid: number | null;
  ask: number | null;
  exchange: "NSE" | "BSE" | "NFO" | "MCX";
  source_type: QuoteSource;
  observed_at_ms: number;
  received_at_ms: number;
  sequence: number | null;
  field_completeness: string[];
  freshness: QuoteFreshness;
  quality: QuoteQuality;
  fallback: boolean;
  fallback_reason: string | null;
  market_state: MarketDataState;
};

const CLIENT_STALE_MS = 5_000;

export function currentMarketState(
  quote: Tick | null | undefined,
  nowMs = Date.now(),
): MarketDataState {
  if (!quote) {
    return "DISCONNECTED";
  }
  if (quote.observed_at_ms <= 0 || nowMs - quote.observed_at_ms > CLIENT_STALE_MS) {
    return "STALE";
  }
  return quote.market_state;
}

export function quoteDisplayStatus(
  quote: Tick | null | undefined,
  nowMs = Date.now(),
): QuoteDisplayStatus {
  const state = currentMarketState(quote, nowMs);
  if (state === "LIVE") return "live";
  if (state === "DEGRADED") return "degraded";
  if (state === "STALE") return "stale";
  return "none";
}

export function displayQuotePrice(
  fallback: number | null | undefined,
  quote: Tick | null | undefined,
  nowMs = Date.now(),
) {
  const state = currentMarketState(quote, nowMs);
  return state === "LIVE" || state === "DEGRADED" ? quote?.ltp ?? fallback : fallback;
}

export function executableQuotePrice(quote: Tick | null | undefined, nowMs = Date.now()) {
  return currentMarketState(quote, nowMs) === "LIVE" &&
    quote?.quality === "EXECUTABLE_STREAM" &&
    quote.source_type === "ANGEL_STREAM" &&
    !quote.fallback
    ? quote.ltp
    : 0;
}

export function marketStateColor(state: MarketDataState) {
  if (state === "LIVE") return "#00d4aa";
  if (state === "DEGRADED") return "#a78bfa";
  if (state === "STALE" || state === "STARTING") return "#f59e0b";
  return "#ff4444";
}
