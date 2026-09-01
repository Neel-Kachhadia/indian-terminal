use base64::Engine;
use chrono::Utc;
use redis::Commands;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use uuid::Uuid;

const MAX_EQUITY_PDF_BYTES: usize = 32 * 1024 * 1024;

use crate::models::{
    Alert, DownloadStatus, Instrument, MarketStatus, OptionLeg, PayoffResult, StrategyLeg, Tick,
    OHLCV,
};
use crate::{
    alert_engine,
    exchange_calendar::exchange_calendar,
    state::{AppState, FeedPhase},
};

#[tauri::command]
pub async fn get_quote(symbol: String, state: tauri::State<'_, AppState>) -> Result<Tick, String> {
    let symbol = normalized_redis_symbol(&symbol)?;
    let key = format!("tick:{symbol}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;

    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;

    let payload = payload.ok_or_else(|| String::from("no data yet"))?;
    let mut tick: Tick = serde_json::from_str(&payload)
        .map_err(|error| format!("tick JSON parse failed: {error}"))?;
    drop(redis_guard);
    let phase = state.feed_status()?.phase;
    tick.evaluate_market_state(
        Utc::now().timestamp_millis(),
        phase == FeedPhase::Live,
        matches!(
            phase,
            FeedPhase::StartingBackend
                | FeedPhase::WaitingForHealth
                | FeedPhase::Authenticated
                | FeedPhase::ConnectingFeed
                | FeedPhase::Subscribing
                | FeedPhase::WaitingForData
        ),
    );
    Ok(tick)
}

#[tauri::command]
pub async fn get_quotes(
    symbols: Vec<String>,
    state: tauri::State<'_, AppState>,
) -> Result<HashMap<String, Tick>, String> {
    const MAX_BATCH_SYMBOLS: usize = 500;
    if symbols.len() > MAX_BATCH_SYMBOLS {
        return Err(format!("quote batch exceeds {MAX_BATCH_SYMBOLS} symbols"));
    }

    let mut seen = HashSet::with_capacity(symbols.len());
    let mut normalized = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        let symbol = normalized_redis_symbol(&symbol)?;
        if seen.insert(symbol.clone()) {
            normalized.push(symbol);
        }
    }
    if normalized.is_empty() {
        return Ok(HashMap::new());
    }

    let keys = normalized
        .iter()
        .map(|symbol| format!("tick:{symbol}"))
        .collect::<Vec<_>>();
    let payloads: Vec<Option<String>> = {
        let mut redis_guard = state.redis.lock().await;
        let connection = redis_guard
            .as_mut()
            .ok_or_else(|| String::from("redis unavailable"))?;
        redis::cmd("MGET")
            .arg(&keys)
            .query(connection)
            .map_err(|error| format!("redis quote batch failed: {error}"))?
    };

    let phase = state.feed_status()?.phase;
    let now_ms = Utc::now().timestamp_millis();
    let feed_live = phase == FeedPhase::Live;
    let feed_connecting = matches!(
        phase,
        FeedPhase::StartingBackend
            | FeedPhase::WaitingForHealth
            | FeedPhase::Authenticated
            | FeedPhase::ConnectingFeed
            | FeedPhase::Subscribing
            | FeedPhase::WaitingForData
    );
    let mut ticks = HashMap::with_capacity(normalized.len());
    for (symbol, payload) in normalized.into_iter().zip(payloads) {
        let Some(payload) = payload else {
            continue;
        };
        let mut tick: Tick = serde_json::from_str(&payload)
            .map_err(|error| format!("tick JSON parse failed for {symbol}: {error}"))?;
        tick.evaluate_market_state(now_ms, feed_live, feed_connecting);
        ticks.insert(symbol, tick);
    }
    Ok(ticks)
}

fn normalized_redis_symbol(symbol: &str) -> Result<String, String> {
    let normalized = symbol.trim().to_ascii_uppercase();
    if normalized.is_empty()
        || normalized.len() > 32
        || !normalized.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b' ' | b'-' | b'_' | b'&' | b'.')
        })
    {
        return Err(String::from("invalid quote symbol"));
    }
    Ok(normalized)
}

fn validated_equity_pdf_url(value: &str) -> Result<url::Url, String> {
    let parsed = url::Url::parse(value).map_err(|_| String::from("invalid equity PDF URL"))?;
    let host = parsed
        .host_str()
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| String::from("equity PDF URL missing host"))?;
    let allowed_host = host == "nseindia.com" || host.ends_with(".nseindia.com");
    if parsed.scheme() != "https"
        || !allowed_host
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port().is_some_and(|port| port != 443)
        || parsed.fragment().is_some()
    {
        return Err(String::from(
            "equity PDF URL is not an allowed NSE HTTPS resource",
        ));
    }
    Ok(parsed)
}

fn validate_equity_pdf_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() > MAX_EQUITY_PDF_BYTES {
        return Err(String::from("equity PDF payload size is invalid"));
    }
    if !bytes.starts_with(b"%PDF-") {
        return Err(String::from("equity PDF payload signature is invalid"));
    }
    Ok(())
}

#[tauri::command]
pub async fn fetch_equity_pdf(url: String) -> Result<String, String> {
    let url = validated_equity_pdf_url(&url)?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|error| format!("equity PDF client failed: {error}"))?;
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("equity PDF request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "equity PDF request returned HTTP {}",
            response.status()
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_EQUITY_PDF_BYTES as u64)
    {
        return Err(String::from("equity PDF payload exceeds size limit"));
    }

    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("equity PDF stream failed: {error}"))?
    {
        if bytes.len().saturating_add(chunk.len()) > MAX_EQUITY_PDF_BYTES {
            return Err(String::from("equity PDF payload exceeds size limit"));
        }
        bytes.extend_from_slice(&chunk);
    }
    validate_equity_pdf_bytes(&bytes)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

#[cfg(test)]
mod quote_batch_tests {
    use super::{normalized_redis_symbol, validate_equity_pdf_bytes, validated_equity_pdf_url};

    #[test]
    fn quote_symbols_are_normalized_and_bounded() {
        assert_eq!(normalized_redis_symbol(" nifty 50 ").unwrap(), "NIFTY 50");
        assert!(normalized_redis_symbol("").is_err());
        assert!(normalized_redis_symbol("NIFTY:INJECT").is_err());
        assert!(normalized_redis_symbol(&"A".repeat(33)).is_err());
        assert!(validated_equity_pdf_url(
            "https://nsearchives.nseindia.com/corporate/example.pdf?x=1"
        )
        .is_ok());
        for rejected in [
            "http://www.nseindia.com/example.pdf",
            "https://nseindia.com.evil.invalid/example.pdf",
            "https://127.0.0.1/example.pdf",
            "https://user@nseindia.com/example.pdf",
            "https://www.nseindia.com:444/example.pdf",
            "https://www.nseindia.com/example.pdf#fragment",
        ] {
            assert!(validated_equity_pdf_url(rejected).is_err(), "{rejected}");
        }
        assert!(validate_equity_pdf_bytes(b"%PDF-1.7\nfixture").is_ok());
        assert!(validate_equity_pdf_bytes(b"<html>not a PDF</html>").is_err());
    }
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
    state: tauri::State<'_, AppState>,
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
        .header("X-Process-Token", process_token(&state).await?)
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
) -> Result<Option<f64>, String> {
    let key = format!("pcr:{symbol}:{expiry}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;

    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;

    let Some(payload) = payload else {
        return Ok(None);
    };

    payload
        .parse::<f64>()
        .map(Some)
        .map_err(|error| format!("PCR parse failed: {error}"))
}

#[tauri::command]
pub async fn get_expiries(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    let url = format!("http://127.0.0.1:8765/expiries/{encoded_symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
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
pub async fn get_lot_size(
    symbol: String,
    expiry: String,
    state: tauri::State<'_, AppState>,
) -> Result<i32, String> {
    let url = format!("http://127.0.0.1:8765/instruments/lot/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
        .query(&[("expiry", expiry)])
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("lot-size request failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("lot-size request failed ({status}): {body}"));
    }
    let payload = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("lot-size JSON parse failed: {error}"))?;
    let lot_size = payload
        .get("lot_size")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| String::from("lot-size response missing lot_size"))?;
    i32::try_from(lot_size)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| String::from("lot-size response is invalid"))
}

#[tauri::command]
pub async fn get_chain_greeks(_symbol: String, _expiry: String) -> Result<Vec<OptionLeg>, String> {
    Ok(Vec::new())
}

#[tauri::command]
pub async fn get_iv_surface(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/iv_surface/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
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
    state: tauri::State<'_, AppState>,
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
        .header("X-Process-Token", process_token(&state).await?)
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
pub async fn get_futures_basis(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/futures/basis/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
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
pub async fn get_signals(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/signals/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("signals request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("signals request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("signals parse failed: {error}"))
}

#[tauri::command]
pub async fn get_all_signals(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/signals/all/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|error| format!("all signals request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("all signals request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("all signals parse failed: {error}"))
}

#[tauri::command]
pub async fn get_sentiment(state: tauri::State<'_, AppState>) -> Result<serde_json::Value, String> {
    let url = "http://127.0.0.1:8765/sentiment/current";
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|error| format!("sentiment request failed: {error}"))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("sentiment parse failed: {error}"))?;

    Ok(data)
}

#[tauri::command]
pub async fn get_cross_asset(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = "http://127.0.0.1:8765/cross_asset/snapshot";
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .map_err(|error| format!("cross_asset request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("cross_asset request failed ({status}): {body}"));
    }

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("cross_asset parse failed: {error}"))?;

    Ok(data)
}

#[tauri::command]
pub async fn get_ml_prediction(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/ml/predict/{symbol}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|error| format!("ml prediction request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("ml prediction request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("ml prediction parse failed: {error}"))
}

#[tauri::command]
pub async fn get_ml_accuracy(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let response = client
        .get("http://127.0.0.1:8765/ml/accuracy")
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("ml accuracy request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("ml accuracy request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("ml accuracy parse failed: {error}"))
}

#[tauri::command]
pub async fn get_bif_all(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    get_python_json(&format!("bif/all/{encoded_symbol}"), 30, &state).await
}

#[tauri::command]
pub async fn get_bif_cds(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    get_python_json(&format!("bif/cds/{encoded_symbol}"), 15, &state).await
}

#[tauri::command]
pub async fn get_bif_history(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    get_python_json(&format!("bif/history/{encoded_symbol}"), 15, &state).await
}

#[tauri::command]
pub async fn get_ohlc_forecast(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    get_python_json(&format!("ml/ohlc_forecast/{encoded_symbol}"), 15, &state).await
}

#[tauri::command]
pub async fn train_all_stocks(
    force: Option<bool>,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    post_python_json(
        &format!("ml/train_all_stocks?force={}", force.unwrap_or(false)),
        &serde_json::json!({}),
        10,
        &state,
    )
    .await
}

#[tauri::command]
pub async fn get_training_status(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("ml/training_status", 10, &state).await
}

#[tauri::command]
pub async fn get_trained_symbols(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("ml/trained_symbols", 10, &state).await
}

#[tauri::command]
pub async fn train_single_symbol(
    symbol: String,
    force: Option<bool>,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    post_python_json(
        &format!(
            "ml/train_symbol/{encoded_symbol}?force={}",
            force.unwrap_or(false)
        ),
        &serde_json::json!({}),
        10,
        &state,
    )
    .await
}

#[tauri::command]
pub async fn get_decision_intelligence(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("decision/intelligence/NIFTY", 30, &state).await
}

#[tauri::command]
pub async fn get_market_brief(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("narrative/market_brief", 20, &state).await
}

#[tauri::command]
pub async fn get_intelligence_patterns(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("intelligence/patterns", 15, &state).await
}

#[tauri::command]
pub async fn get_intelligence_regime(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("intelligence/regime", 15, &state).await
}

#[tauri::command]
pub async fn get_risk_var(state: tauri::State<'_, AppState>) -> Result<serde_json::Value, String> {
    get_python_json("risk/var", 15, &state).await
}

#[tauri::command]
pub async fn get_options_flow(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("intelligence/options_flow", 15, &state).await
}

#[tauri::command]
pub async fn get_data_summary(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let response = client
        .get("http://127.0.0.1:8765/data/summary")
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("data summary request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("data summary request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("data summary parse failed: {error}"))
}

#[tauri::command]
pub async fn get_data_errors(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let response = client
        .get("http://127.0.0.1:8765/data/errors")
        .header("X-Process-Token", process_token(&state).await?)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| format!("data errors request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("data errors request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("data errors parse failed: {error}"))
}

#[tauri::command]
pub async fn get_stock_snapshot(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    get_python_json(&format!("stocks/snapshot/{encoded_symbol}"), 10, &state).await
}

#[tauri::command]
pub async fn get_stock_universe(
    universe: String,
    limit: Option<u64>,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let clean_universe = universe.trim().to_ascii_uppercase();
    let limit = limit.unwrap_or(250);
    get_python_json(
        &format!("stocks/universe?universe={clean_universe}&limit={limit}"),
        15,
        &state,
    )
    .await
}

#[tauri::command]
pub async fn get_equity_summary(
    symbol: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let encoded_symbol = symbol.replace(' ', "%20");
    get_python_json(&format!("equity/summary/{encoded_symbol}"), 20, &state).await
}

#[tauri::command]
pub async fn run_screener(
    body: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let payload = serde_json::from_str::<serde_json::Value>(&body)
        .map_err(|error| format!("screener body parse failed: {error}"))?;
    post_python_json("screener/run", &payload, 15, &state).await
}

#[tauri::command]
pub async fn run_backtest(
    params: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let payload = serde_json::from_str::<serde_json::Value>(&params)
        .map_err(|error| format!("backtest params parse failed: {error}"))?;
    post_python_json("backtest/run", &payload, 15, &state).await
}

#[tauri::command]
pub async fn get_backtest_status(
    job_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json(&format!("backtest/status/{job_id}"), 15, &state).await
}

#[tauri::command]
pub async fn get_backtest_result(
    job_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json(&format!("backtest/result/{job_id}"), 20, &state).await
}

#[tauri::command]
pub async fn list_strategies(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("backtest/strategies", 10, &state).await
}

#[tauri::command]
pub async fn get_backtest_history(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("backtest/history", 10, &state).await
}

#[tauri::command]
pub async fn get_screener_results(
    conditions: Vec<serde_json::Value>,
    logic: Option<String>,
    universe: Option<String>,
    limit: Option<u64>,
    state: tauri::State<'_, AppState>,
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
        .header("X-Process-Token", process_token(&state).await?)
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
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = "http://127.0.0.1:8765/indicators";
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "symbol": symbol,
        "timeframe": timeframe,
        "indicators": indicators,
    });

    // Backend's own bounded read budget for this query is 8s (see
    // technical/indicators.py::_load_ohlcv); this must stay comfortably
    // above that so this client never aborts the connection before the
    // backend's own timeout has a chance to return an honest error.
    let response = client
        .post(url)
        .header("X-Process-Token", process_token(&state).await?)
        .json(&body)
        .timeout(std::time::Duration::from_secs(12))
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
    let now = Utc::now();
    let calendar = exchange_calendar()?;
    Ok(MarketStatus {
        status: calendar.session_state(now)?.to_ascii_lowercase(),
        next_change: calendar.next_session_change_ms(now)?,
    })
}

#[tauri::command]
pub async fn get_market_session(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("market/session", 5, &state).await
}

fn backend_payload_ready(payload: &serde_json::Value) -> bool {
    payload
        .pointer("/readiness/ready")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
        || payload
            .pointer("/readiness/status")
            .and_then(serde_json::Value::as_str)
            == Some("READY")
}

#[tauri::command]
pub async fn wait_for_backend_ready(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    const STARTUP_BUDGET: Duration = Duration::from_secs(90);
    let deadline = Instant::now() + STARTUP_BUDGET;
    let mut attempt = 0_u32;

    loop {
        let last_error = match get_python_json("health", 3, &state).await {
            Ok(payload) if backend_payload_ready(&payload) => return Ok(payload),
            Ok(payload) => format!(
                "backend not ready: {}",
                payload
                    .pointer("/readiness/status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("UNKNOWN")
            ),
            Err(error) => error,
        };
        if Instant::now() >= deadline {
            return Err(format!("backend readiness budget exhausted: {last_error}"));
        }
        let exponent = attempt.min(5);
        let delay_ms = (250_u64.saturating_mul(1_u64 << exponent)).min(4_000);
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        attempt = attempt.saturating_add(1);
    }
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

#[tauri::command]
pub async fn get_broker_auth_status(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("broker/auth/status", 5, &state).await
}

#[tauri::command]
pub async fn get_integrity_status(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    get_python_json("app/integrity", 5, &state).await
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

async fn process_token(state: &tauri::State<'_, AppState>) -> Result<String, String> {
    for _ in 0..100 {
        if let Ok(token) = state.process_token() {
            return Ok(token);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    state.process_token()
}

async fn get_python_json(
    path: &str,
    timeout_secs: u64,
    state: &tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/{path}");
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("X-Process-Token", process_token(state).await?)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .send()
        .await
        .map_err(|error| format!("{path} request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("{path} request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("{path} parse failed: {error}"))
}

async fn post_python_json(
    path: &str,
    payload: &serde_json::Value,
    timeout_secs: u64,
    state: &tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:8765/{path}");
    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .header("X-Process-Token", process_token(state).await?)
        .json(payload)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .send()
        .await
        .map_err(|error| format!("{path} request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("{path} request failed ({status}): {body}"));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("{path} parse failed: {error}"))
}

#[cfg(test)]
mod backend_gate_tests {
    use super::backend_payload_ready;

    #[test]
    fn startup_gate_opens_only_for_explicit_ready_payload() {
        assert!(backend_payload_ready(&serde_json::json!({
            "readiness": {"ready": true, "status": "READY"}
        })));
        assert!(!backend_payload_ready(&serde_json::json!({
            "status": "ALIVE",
            "readiness": {"ready": false, "status": "NOT_READY"}
        })));
        assert!(!backend_payload_ready(&serde_json::json!({})));
    }
}
