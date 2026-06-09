use chrono::Utc;
use redis::Commands;
use serde::Deserialize;
use uuid::Uuid;

use crate::models::{
    Alert, DownloadStatus, Instrument, MarketStatus, OptionLeg, OptionType, PayoffResult,
    StrategyLeg, Tick, OHLCV,
};
use crate::{alert_engine, state::AppState};

#[tauri::command]
pub async fn get_quote(symbol: String, state: tauri::State<'_, AppState>) -> Result<Tick, String> {
    let key = format!("tick:{symbol}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;

    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;

    let payload = payload.ok_or_else(|| String::from("no data yet"))?;
    serde_json::from_str(&payload).map_err(|error| format!("tick JSON parse failed: {error}"))
}

#[tauri::command]
pub async fn get_chain(
    symbol: String,
    expiry: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<OptionLeg>, String> {
    let key = format!("chain:{symbol}:{expiry}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;

    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;

    let Some(payload) = payload else {
        return Ok(Vec::new());
    };

    serde_json::from_str(&payload).map_err(|error| format!("chain JSON parse failed: {error}"))
}

#[tauri::command]
pub async fn get_ohlcv(
    symbol: String,
    from_date: Option<String>,
    to_date: Option<String>,
) -> Result<Vec<OHLCV>, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    let url = format!("http://127.0.0.1:8765/ohlcv/{encoded_symbol}");
    let mut query_params: Vec<(&str, String)> = Vec::new();

    if let Some(value) = from_date {
        query_params.push(("from_date", value));
    }

    if let Some(value) = to_date {
        query_params.push(("to_date", value));
    }

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .query(&query_params)
        .send()
        .await
        .map_err(|error| format!("ohlcv request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("ohlcv request failed ({status}): {body}"));
    }

    let rows = response
        .json::<Vec<ApiOhlcv>>()
        .await
        .map_err(|error| format!("ohlcv JSON parse failed: {error}"))?;

    Ok(rows
        .into_iter()
        .map(|row| OHLCV {
            symbol: row.symbol.unwrap_or_else(|| symbol.clone()),
            date: row.date,
            open: row.open,
            high: row.high,
            low: row.low,
            close: row.close,
            volume: row.volume,
        })
        .collect())
}

#[tauri::command]
pub async fn get_pcr(
    symbol: String,
    expiry: String,
    state: tauri::State<'_, AppState>,
) -> Result<f64, String> {
    let key = format!("pcr:{symbol}:{expiry}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;

    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;

    let Some(payload) = payload else {
        return Ok(0.0);
    };

    payload
        .parse::<f64>()
        .map_err(|error| format!("PCR parse failed: {error}"))
}

#[tauri::command]
pub async fn get_expiries(symbol: String) -> Result<Vec<String>, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    let url = format!("http://127.0.0.1:8765/expiries/{encoded_symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|error| format!("expiries request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("expiries request failed ({status}): {body}"));
    }

    response
        .json::<Vec<String>>()
        .await
        .map_err(|error| format!("expiries JSON parse failed: {error}"))
}

#[tauri::command]
pub async fn get_chain_greeks(_symbol: String, _expiry: String) -> Result<Vec<OptionLeg>, String> {
    Ok(Vec::new())
}

#[tauri::command]
pub async fn get_iv_surface(symbol: String) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/iv_surface/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("iv_surface request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("iv_surface request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("iv_surface parse failed: {error}"))
}

#[tauri::command]
pub async fn get_payoff(
    legs: Vec<StrategyLeg>,
    spot: f64,
    spot_range_pct: Option<f64>,
) -> Result<PayoffResult, String> {
    let url = "http://127.0.0.1:8765/payoff";
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "legs": legs,
        "spot": spot,
        "spot_range_pct": spot_range_pct.unwrap_or(0.20),
    });

    let response = client
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("payoff request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("payoff request failed ({status}): {body}"));
    }

    let value = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("payoff parse failed: {error}"))?;

    if let Some(error) = value.get("error").and_then(|item| item.as_str()) {
        return Err(error.to_string());
    }

    serde_json::from_value(value).map_err(|error| format!("payoff result parse failed: {error}"))
}

#[tauri::command]
pub async fn get_futures_basis(symbol: String) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/futures/basis/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("futures basis request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("futures basis request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("futures basis parse failed: {error}"))
}

#[tauri::command]
pub async fn get_screener_results(
    conditions: Vec<serde_json::Value>,
    logic: Option<String>,
    universe: Option<String>,
    limit: Option<u64>,
) -> Result<serde_json::Value, String> {
    let url = "http://127.0.0.1:8765/screener";
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "conditions": conditions,
        "logic": logic.unwrap_or_else(|| String::from("AND")),
        "universe": universe.unwrap_or_else(|| String::from("NIFTY50")),
        "limit": limit.unwrap_or(50),
    });

    let response = client
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("screener request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("screener request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("screener parse failed: {error}"))
}

#[tauri::command]
pub async fn get_indicators(
    symbol: String,
    timeframe: String,
    indicators: Vec<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let url = "http://127.0.0.1:8765/indicators";
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "symbol": symbol,
        "timeframe": timeframe,
        "indicators": indicators,
    });

    let response = client
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("indicators request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("indicators request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("indicators parse failed: {error}"))
}

#[tauri::command]
pub async fn set_alert(
    symbol: String,
    alert_type: String,
    threshold: f64,
    condition: String,
    _state: tauri::State<'_, AppState>,
) -> Result<Alert, String> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(String::from(
            "threshold must be a non-negative finite number",
        ));
    }

    let symbol = symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        return Err(String::from("symbol is required"));
    }

    let alert = Alert {
        id: Uuid::new_v4().to_string(),
        symbol,
        alert_type: alert_engine::parse_alert_type(&alert_type)?,
        threshold,
        condition: alert_engine::parse_alert_condition(&condition)?,
        active: true,
        triggered_at: None,
        created_at: Utc::now().timestamp_millis(),
    };

    alert_engine::save_alert(&alert)?;
    Ok(alert)
}

#[tauri::command]
pub async fn delete_alert(id: String, _state: tauri::State<'_, AppState>) -> Result<(), String> {
    alert_engine::delete_alert(&id)
}

#[tauri::command]
pub async fn get_alerts(_state: tauri::State<'_, AppState>) -> Result<Vec<Alert>, String> {
    alert_engine::load_active_alerts()
}

#[tauri::command]
pub async fn get_market_status() -> Result<MarketStatus, String> {
    Ok(MarketStatus {
        status: String::from("closed"),
        next_change: 0,
    })
}

#[tauri::command]
pub async fn get_instrument_list(
    _exchange: String,
    _segment: String,
) -> Result<Vec<Instrument>, String> {
    Ok(Vec::new())
}

#[tauri::command]
pub async fn download_historical(
    symbol: String,
    _from_date: String,
) -> Result<DownloadStatus, String> {
    Ok(DownloadStatus {
        symbol,
        status: String::from("queued"),
        started_at: Utc::now().timestamp_millis(),
    })
}

#[allow(dead_code)]
fn _placeholder_option_leg() -> OptionLeg {
    OptionLeg {
        symbol: String::new(),
        expiry: String::new(),
        strike: 0.0,
        option_type: OptionType::CE,
        ltp: 0.0,
        bid: 0.0,
        ask: 0.0,
        volume: 0,
        oi: 0,
        oi_change: 0,
        iv: None,
        delta: None,
        gamma: None,
        theta: None,
        vega: None,
    }
}

#[derive(Debug, Deserialize)]
struct ApiOhlcv {
    symbol: Option<String>,
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}
