use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub symbol: String,
    pub token: String,
    pub ltp: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub oi: u64,
    pub bid: f64,
    pub ask: f64,
    pub timestamp: i64,
    pub exchange: Exchange,
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
    pub symbol: String,
    pub expiry: String,
    pub strike: f64,
    pub option_type: OptionType,
    pub ltp: f64,
    pub bid: f64,
    pub ask: f64,
    pub volume: u64,
    pub oi: u64,
    pub oi_change: i64,
    pub iv: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
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
    pub pnl_today: Vec<f64>,
    pub breakevens: Vec<f64>,
    pub max_profit: Option<f64>,
    pub max_loss: Option<f64>,
    pub net_greeks: NetGreeks,
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
