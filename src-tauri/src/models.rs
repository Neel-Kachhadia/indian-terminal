use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub schema_version: u16,
    pub symbol: String,
    pub token: String,
    pub ltp: f64,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<u64>,
    pub oi: Option<u64>,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub exchange: Exchange,
    pub source_type: QuoteSource,
    pub observed_at_ms: i64,
    pub received_at_ms: i64,
    pub sequence: Option<u64>,
    pub field_completeness: Vec<String>,
    pub freshness: QuoteFreshness,
    pub quality: QuoteQuality,
    pub fallback: bool,
    pub fallback_reason: Option<String>,
    pub market_state: MarketDataState,
    #[serde(default)]
    pub requested_symbol: String,
    #[serde(default)]
    pub canonical_symbol: String,
    #[serde(default)]
    pub age_ms: i64,
    #[serde(default)]
    pub connection_state: MarketDataState,
    #[serde(default)]
    pub decision_eligible: bool,
}

impl Tick {
    pub const SCHEMA_VERSION: u16 = 1;
    // Quote mode publishes on market changes; active but less-liquid F&O
    // equities can legitimately remain unchanged for several seconds.
    pub const STREAM_FRESH_MS: i64 = 30_000;

    pub fn evaluate_market_state(
        &mut self,
        now_ms: i64,
        expected_feed_live: bool,
        expected_feed_starting: bool,
    ) {
        let age_ms = now_ms.saturating_sub(self.observed_at_ms);
        self.freshness = if self.observed_at_ms <= 0 {
            QuoteFreshness::Unknown
        } else if age_ms <= Self::STREAM_FRESH_MS {
            QuoteFreshness::Fresh
        } else {
            QuoteFreshness::Stale
        };
        self.market_state = if self.freshness == QuoteFreshness::Stale {
            MarketDataState::Stale
        } else if expected_feed_starting {
            MarketDataState::Starting
        } else if self.fallback
            || self.source_type != QuoteSource::AngelStream
            || self.quality != QuoteQuality::ExecutableStream
        {
            MarketDataState::Degraded
        } else if expected_feed_live && self.freshness == QuoteFreshness::Fresh {
            MarketDataState::Live
        } else {
            MarketDataState::Disconnected
        };
        self.age_ms = age_ms.max(0);
        self.connection_state = self.market_state;
        self.decision_eligible = self.market_state == MarketDataState::Live
            && self.freshness == QuoteFreshness::Fresh
            && self.source_type == QuoteSource::AngelStream
            && self.quality == QuoteQuality::ExecutableStream
            && !self.fallback;
    }

    pub fn is_newer_than(&self, existing: &Tick) -> bool {
        match (self.sequence, existing.sequence) {
            (Some(incoming), Some(current)) if incoming != current => incoming > current,
            _ if self.observed_at_ms != existing.observed_at_ms => {
                self.observed_at_ms > existing.observed_at_ms
            }
            _ => self.received_at_ms > existing.received_at_ms,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QuoteSource {
    AngelStream,
    NseChain,
    Cache,
    Historical,
    Derived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QuoteFreshness {
    Fresh,
    Stale,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QuoteQuality {
    ExecutableStream,
    ReferenceOnly,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MarketDataState {
    Live,
    Degraded,
    Stale,
    #[default]
    Disconnected,
    Starting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Exchange {
    NSE,
    BSE,
    NFO,
    MCX,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionLeg {
    pub schema_version: u16,
    pub symbol: String,
    pub expiry: String,
    pub strike: f64,
    pub option_type: OptionType,
    pub ltp: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub volume: Option<u64>,
    pub oi: Option<u64>,
    pub oi_change: Option<i64>,
    pub iv: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub pricing_model: Option<String>,
    pub calculation_status: Option<String>,
    pub iv_observation: Option<serde_json::Value>,
    pub source_type: QuoteSource,
    pub observed_at_ms: i64,
    pub received_at_ms: i64,
    pub sequence: Option<u64>,
    pub field_completeness: Vec<String>,
    pub freshness: QuoteFreshness,
    pub quality: QuoteQuality,
    pub fallback: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionType {
    CE,
    PE,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub symbol: String,
    pub alert_type: AlertType,
    pub threshold: f64,
    pub condition: AlertCondition,
    pub active: bool,
    pub triggered_at: Option<i64>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertType {
    Price,
    OI,
    PCR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertCondition {
    Above,
    Below,
    CrossesUp,
    CrossesDown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub symbol: String,
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewAlert {
    pub symbol: String,
    pub alert_type: AlertType,
    pub threshold: f64,
    pub condition: AlertCondition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVSurfaceData {
    pub surface: IVSurfaceGrid,
    pub skew: IVSkew,
    pub term_structure: IVTermStructure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVSurfaceGrid {
    pub strikes: Vec<f64>,
    pub expiries: Vec<String>,
    pub dtes: Vec<i32>,
    pub iv_grid: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVSkew {
    pub expiry: String,
    pub strikes: Vec<f64>,
    pub call_ivs: Vec<f64>,
    pub put_ivs: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVTermStructure {
    pub expiries: Vec<String>,
    pub dtes: Vec<i32>,
    pub atm_call_iv: Vec<f64>,
    pub atm_put_iv: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyLeg {
    pub symbol: String,
    pub strike: f64,
    pub expiry: String,
    pub option_type: StrategyLegType,
    pub action: TradeAction,
    pub quantity: i32,
    pub entry_price: f64,
    pub lot_size: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyLegType {
    CE,
    PE,
    FUT,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    BUY,
    SELL,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffResult {
    pub spot_range: Vec<f64>,
    pub pnl_at_expiry: Vec<f64>,
    // Today's P&L / Greeks require both a rate and a dividend-yield input
    // (a genuinely observed current rate, or an explicitly disclosed
    // configured assumption) for every option leg; absent that, the backend
    // reports them as None rather than fabricating a value, so these two
    // must stay optional or deserializing a real "unavailable" response
    // fails outright.
    pub pnl_today: Option<Vec<f64>>,
    pub analytical_status: String,
    pub risk_free_rate_source: Option<String>,
    pub dividend_yield_source: Option<String>,
    pub breakevens: Vec<f64>,
    pub max_profit: Option<f64>,
    pub max_loss: Option<f64>,
    pub net_greeks: Option<NetGreeks>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenerCondition {
    pub field: String,
    pub operator: String,
    pub value: f64,
    pub timeframe: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenerResult {
    pub symbol: String,
    pub ltp: f64,
    pub matched_conditions: Vec<String>,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorResult {
    pub timestamps: Vec<i64>,
    pub indicators: HashMap<String, Vec<Option<f64>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStatus {
    pub status: String,
    pub next_change: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instrument {
    pub symbol: String,
    pub token: String,
    pub exchange: String,
    pub segment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    pub symbol: String,
    pub status: String,
    pub started_at: i64,
}

#[cfg(test)]
mod quote_tests {
    use super::{Exchange, MarketDataState, QuoteFreshness, QuoteQuality, QuoteSource, Tick};

    fn tick(observed_at_ms: i64, sequence: Option<u64>, fallback: bool) -> Tick {
        Tick {
            schema_version: Tick::SCHEMA_VERSION,
            symbol: "NIFTY".into(),
            token: "99123456".into(),
            ltp: 24_000.0,
            open: Some(23_900.0),
            high: Some(24_100.0),
            low: Some(23_850.0),
            close: Some(23_950.0),
            volume: Some(10),
            oi: None,
            bid: None,
            ask: None,
            exchange: Exchange::NSE,
            source_type: if fallback {
                QuoteSource::NseChain
            } else {
                QuoteSource::AngelStream
            },
            observed_at_ms,
            received_at_ms: observed_at_ms + 1,
            sequence,
            field_completeness: vec!["ltp".into(), "ohlc".into(), "volume".into()],
            freshness: QuoteFreshness::Fresh,
            quality: if fallback {
                QuoteQuality::ReferenceOnly
            } else {
                QuoteQuality::ExecutableStream
            },
            fallback,
            fallback_reason: fallback.then(|| "NSE chain underlying reference".into()),
            market_state: MarketDataState::Starting,
            requested_symbol: String::from("NIFTY"),
            canonical_symbol: String::from("NIFTY"),
            age_ms: 0,
            connection_state: MarketDataState::Starting,
            decision_eligible: false,
        }
    }

    #[test]
    fn stale_stream_cannot_be_live() {
        let mut quote = tick(1_000, None, false);
        quote.evaluate_market_state(1_000 + Tick::STREAM_FRESH_MS + 1, true, false);
        assert_eq!(quote.freshness, QuoteFreshness::Stale);
        assert_eq!(quote.market_state, MarketDataState::Stale);
    }

    #[test]
    fn fresh_fallback_is_degraded_not_live() {
        let mut quote = tick(10_000, None, true);
        quote.evaluate_market_state(10_100, true, false);
        assert_eq!(quote.market_state, MarketDataState::Degraded);
    }

    #[test]
    fn out_of_order_sequence_is_rejected() {
        let current = tick(10_000, Some(8), false);
        let incoming = tick(11_000, Some(7), false);
        assert!(!incoming.is_newer_than(&current));
    }
}
