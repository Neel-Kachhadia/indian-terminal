use std::{cmp::Ordering, time::Duration};

use chrono::{Datelike, Duration as ChronoDuration, Timelike, Utc, Weekday};
use redis::Commands;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::json;
use tauri::Emitter;
use tokio::time;

use crate::{
    models::{OptionLeg, OptionType, Tick},
    state::AppState,
};

const NSE_CHAIN_URL: &str = "http://127.0.0.1:8765/chain/NIFTY";
const GREEKS_URL: &str = "http://127.0.0.1:8765/greeks";
const NIFTY_SYMBOL: &str = "NIFTY";
const REDIS_TTL_SECONDS: u64 = 15;

#[derive(Debug)]
enum PollError {
    SessionExpired,
    RateLimited,
    Other(String),
}

#[derive(Debug, Deserialize)]
struct NseChainResponse {
    records: Option<NseRecords>,
}

#[derive(Debug, Deserialize)]
struct NseRecords {
    data: Option<Vec<NseRow>>,
    #[serde(rename = "underlyingValue")]
    underlying_value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct NseRow {
    #[serde(rename = "strikePrice")]
    strike_price: f64,
    #[serde(rename = "expiryDate")]
    expiry_date: String,
    #[serde(rename = "CE")]
    ce: Option<NseOptionLeg>,
    #[serde(rename = "PE")]
    pe: Option<NseOptionLeg>,
}

#[derive(Debug, Deserialize)]
struct NseOptionLeg {
    #[serde(rename = "lastPrice", default)]
    last_price: f64,
    #[serde(rename = "bidprice", default)]
    bid_price: f64,
    #[serde(rename = "askPrice", default)]
    ask_price: f64,
    #[serde(rename = "openInterest", default)]
    open_interest: f64,
    #[serde(rename = "changeinOpenInterest", default)]
    oi_change: f64,
    #[serde(rename = "totalTradedVolume", default)]
    volume: f64,
    #[serde(rename = "impliedVolatility", default)]
    iv: f64,
}

#[derive(Debug, Deserialize)]
struct GreeksResponse {
    legs: Vec<GreekLeg>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GreekLeg {
    strike: f64,
    option_type: OptionType,
    iv: Option<f64>,
    delta: Option<f64>,
    gamma: Option<f64>,
    theta: Option<f64>,
    vega: Option<f64>,
}

pub async fn run(state: AppState) -> Result<(), String> {
    let client = build_client();
    let mut cookies;

    loop {
        println!("poller: fetching NSE session cookie...");
        match fetch_nse_cookies(&client).await {
            Ok(c) => {
                cookies = c;
                println!("poller: NSE session established");
            }
            Err(error) => {
                println!("poller: NSE session failed: {error}");
                time::sleep(Duration::from_secs(10)).await;
                continue;
            }
        }

        let mut interval = time::interval(Duration::from_secs(3));
        let mut consecutive_errors = 0u32;

        loop {
            interval.tick().await;

            let expiry = nearest_expiry_thursday();

            match fetch_chain(&client, &cookies, &expiry).await {
                Ok((legs, spot)) => {
                    consecutive_errors = 0;
                    let pcr = compute_pcr(&legs);

                    if spot > 0.0 {
                        update_spot_in_redis(&state, spot).await;
                    }

                    match write_chain_to_redis(&state, &expiry, &legs, pcr).await {
                        Ok(()) => {
                            if spot > 0.0 {
                                match enrich_chain_with_greeks(
                                    &client, &state, &expiry, &legs, spot, pcr,
                                )
                                .await
                                {
                                    Ok(count) => {
                                        println!("poller: greeks computed for {count} strikes");
                                    }
                                    Err(error) => {
                                        println!("poller: greeks skipped: {error}");
                                    }
                                }
                            } else {
                                println!("poller: greeks skipped: missing spot");
                            }

                            emit_chain_update(&state, &expiry);
                            println!(
                                "poller: chain fetched for NIFTY {expiry} spot={spot:.2} pcr={pcr:.2}"
                            );
                        }
                        Err(error) => {
                            println!("poller: redis write failed: {error}");
                        }
                    }
                }
                Err(PollError::SessionExpired) => {
                    println!("poller: NSE session expired, refreshing...");
                    break;
                }
                Err(PollError::RateLimited) => {
                    println!("poller: NSE rate limited, backing off 30s");
                    time::sleep(Duration::from_secs(30)).await;
                    consecutive_errors += 1;
                    if consecutive_errors > 5 {
                        println!("poller: too many errors, refreshing session");
                        break;
                    }
                }
                Err(PollError::Other(error)) => {
                    println!("poller: {error}");
                    consecutive_errors += 1;
                    time::sleep(Duration::from_secs(5)).await;
                    if consecutive_errors > 10 {
                        println!("poller: too many errors, refreshing session");
                        break;
                    }
                }
            }
        }
    }
}

async fn enrich_chain_with_greeks(
    client: &Client,
    state: &AppState,
    expiry: &str,
    legs: &[OptionLeg],
    spot: f64,
    pcr: f64,
) -> Result<usize, String> {
    let response = client
        .post(GREEKS_URL)
        .timeout(Duration::from_secs(5))
        .json(&json!({
            "symbol": NIFTY_SYMBOL,
            "expiry": expiry,
            "spot": spot,
            "legs": legs,
        }))
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(format!("request failed ({status}): {body}"));
    }

    let greeks = response
        .json::<GreeksResponse>()
        .await
        .map_err(|error| format!("JSON parse failed: {error}"))?;

    let mut enriched = legs.to_vec();
    merge_greeks(&mut enriched, &greeks.legs);
    let computed_count = count_greek_strikes(&enriched);

    if computed_count == 0 {
        if let Some(error) = greeks.error {
            return Err(format!("endpoint returned no Greeks: {error}"));
        }
    } else if let Some(error) = greeks.error {
        println!("poller: greeks partial error: {error}");
    }

    write_chain_to_redis(state, expiry, &enriched, pcr).await?;
    Ok(computed_count)
}

fn build_client() -> Client {
    Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

async fn fetch_nse_cookies(client: &Client) -> Result<String, String> {
    let response = client
        .get("https://www.nseindia.com")
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "en-US,en;q=0.5")
        .header("Connection", "keep-alive")
        .send()
        .await
        .map_err(|e| format!("NSE homepage request failed: {e}"))?;

    let mut cookie_str = String::new();
    for (name, value) in response.headers() {
        if name.as_str().to_lowercase() == "set-cookie" {
            if let Ok(v) = value.to_str() {
                let cookie_part = v.split(';').next().unwrap_or("").trim().to_string();
                if !cookie_part.is_empty() {
                    if !cookie_str.is_empty() {
                        cookie_str.push_str("; ");
                    }
                    cookie_str.push_str(&cookie_part);
                }
            }
        }
    }

    if cookie_str.is_empty() {
        return Err("no cookies received from NSE".to_string());
    }

    time::sleep(Duration::from_millis(500)).await;

    Ok(cookie_str)
}

async fn fetch_chain(
    client: &Client,
    _cookies: &str,
    expiry: &str,
) -> Result<(Vec<OptionLeg>, f64), PollError> {
    let response = client
        .get(NSE_CHAIN_URL)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| PollError::Other(format!("chain request failed: {e}")))?;

    let status = response.status();

    if status == StatusCode::TOO_MANY_REQUESTS {
        return Err(PollError::RateLimited);
    }

    if !status.is_success() {
        return Err(PollError::Other(format!("chain request failed ({status})")));
    }

    let body = response
        .text()
        .await
        .map_err(|e| PollError::Other(format!("chain body read failed: {e}")))?;

    // Check for Python-level error response
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
        if val.get("error").is_some() {
            return Err(PollError::SessionExpired);
        }
    }

    let chain: NseChainResponse = serde_json::from_str(&body)
        .map_err(|e| PollError::Other(format!("chain JSON parse failed: {e}")))?;

    let records = chain
        .records
        .ok_or_else(|| PollError::Other("chain missing records".to_string()))?;

    let spot = records.underlying_value.unwrap_or(0.0);
    let target_expiry_lower = expiry.to_lowercase();

    let mut legs = Vec::new();
    for row in records.data.unwrap_or_default() {
        if row.expiry_date.to_lowercase() != target_expiry_lower {
            continue;
        }

        if let Some(ce) = row.ce {
            legs.push(map_nse_leg(
                &row.expiry_date,
                row.strike_price,
                OptionType::CE,
                &ce,
            ));
        }
        if let Some(pe) = row.pe {
            legs.push(map_nse_leg(
                &row.expiry_date,
                row.strike_price,
                OptionType::PE,
                &pe,
            ));
        }
    }

    legs.sort_by(|a, b| {
        a.strike
            .partial_cmp(&b.strike)
            .unwrap_or(Ordering::Equal)
            .then_with(|| option_type_rank(&a.option_type).cmp(&option_type_rank(&b.option_type)))
    });

    Ok((legs, spot))
}

fn merge_greeks(legs: &mut [OptionLeg], greeks: &[GreekLeg]) {
    for greek in greeks {
        for leg in legs.iter_mut() {
            if !strikes_match(leg.strike, greek.strike) {
                continue;
            }
            if option_type_rank(&leg.option_type) != option_type_rank(&greek.option_type) {
                continue;
            }

            leg.iv = greek.iv;
            leg.delta = greek.delta;
            leg.gamma = greek.gamma;
            leg.theta = greek.theta;
            leg.vega = greek.vega;
            break;
        }
    }
}

fn count_greek_strikes(legs: &[OptionLeg]) -> usize {
    let mut strikes = Vec::new();
    for leg in legs {
        if leg.delta.is_none() || leg.gamma.is_none() || leg.theta.is_none() || leg.vega.is_none() {
            continue;
        }
        if strikes
            .iter()
            .any(|strike: &f64| strikes_match(*strike, leg.strike))
        {
            continue;
        }
        strikes.push(leg.strike);
    }
    strikes.len()
}

fn strikes_match(left: f64, right: f64) -> bool {
    (left - right).abs() < 0.01
}

fn map_nse_leg(
    expiry: &str,
    strike: f64,
    option_type: OptionType,
    leg: &NseOptionLeg,
) -> OptionLeg {
    OptionLeg {
        symbol: NIFTY_SYMBOL.to_string(),
        expiry: expiry.to_string(),
        strike,
        option_type,
        ltp: leg.last_price,
        bid: leg.bid_price,
        ask: leg.ask_price,
        volume: leg.volume as u64,
        oi: leg.open_interest as u64,
        oi_change: leg.oi_change as i64,
        iv: if leg.iv > 0.0 { Some(leg.iv) } else { None },
        delta: None,
        gamma: None,
        theta: None,
        vega: None,
    }
}

async fn update_spot_in_redis(state: &AppState, spot: f64) {
    let mut redis_guard = state.redis.lock().await;
    let connection = match redis_guard.as_mut() {
        Some(c) => c,
        None => return,
    };

    let existing: redis::RedisResult<Option<String>> = connection.get("tick:NIFTY");
    if matches!(existing, Ok(Some(_))) {
        return;
    }

    let tick = Tick {
        symbol: NIFTY_SYMBOL.to_string(),
        token: String::new(),
        ltp: spot,
        open: 0.0,
        high: 0.0,
        low: 0.0,
        close: 0.0,
        volume: 0,
        oi: 0,
        bid: 0.0,
        ask: 0.0,
        timestamp: Utc::now().timestamp_millis(),
        exchange: crate::models::Exchange::NSE,
    };

    if let Ok(payload) = serde_json::to_string(&tick) {
        let _: redis::RedisResult<()> = connection.set_ex("tick:NIFTY", payload, 15);
    }
}

async fn write_chain_to_redis(
    state: &AppState,
    expiry: &str,
    legs: &[OptionLeg],
    pcr: f64,
) -> Result<(), String> {
    let chain_payload =
        serde_json::to_string(legs).map_err(|e| format!("chain serialize failed: {e}"))?;

    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;

    let chain_key = format!("chain:{NIFTY_SYMBOL}:{expiry}");
    let chain_result: redis::RedisResult<()> =
        connection.set_ex(&chain_key, chain_payload, REDIS_TTL_SECONDS);
    chain_result.map_err(|e| format!("SET {chain_key} failed: {e}"))?;

    let pcr_key = format!("pcr:{NIFTY_SYMBOL}:{expiry}");
    let pcr_result: redis::RedisResult<()> =
        connection.set_ex(&pcr_key, pcr.to_string(), REDIS_TTL_SECONDS);
    pcr_result.map_err(|e| format!("SET {pcr_key} failed: {e}"))?;

    Ok(())
}

fn emit_chain_update(state: &AppState, expiry: &str) {
    let Some(app_handle) = state.app_handle() else {
        return;
    };

    let payload = json!({
        "symbol": NIFTY_SYMBOL,
        "expiry": expiry,
    });

    if let Err(error) = app_handle.emit("chain_update", payload) {
        println!("poller: chain_update emit failed: {error}");
    }
}

fn compute_pcr(legs: &[OptionLeg]) -> f64 {
    let mut call_oi = 0_u64;
    let mut put_oi = 0_u64;

    for leg in legs {
        match leg.option_type {
            OptionType::CE => call_oi = call_oi.saturating_add(leg.oi),
            OptionType::PE => put_oi = put_oi.saturating_add(leg.oi),
        }
    }

    if call_oi == 0 {
        return 0.0;
    }

    put_oi as f64 / call_oi as f64
}

fn nearest_expiry_thursday() -> String {
    let now_ist = Utc::now() + ChronoDuration::minutes(330);
    let today = now_ist.date_naive();
    let current_weekday = today.weekday().num_days_from_monday() as i64;
    // NSE changed NIFTY expiry from Thursday to Tuesday in 2024
    let tuesday = Weekday::Tue.num_days_from_monday() as i64;
    let mut days_until = (tuesday - current_weekday + 7) % 7;
    let before_close = now_ist.hour() < 15 || (now_ist.hour() == 15 && now_ist.minute() < 30);
    if days_until == 0 && !before_close {
        days_until = 7;
    }
    let expiry = today + ChronoDuration::days(days_until);
    expiry.format("%d-%b-%Y").to_string()
}
fn option_type_rank(option_type: &OptionType) -> u8 {
    match option_type {
        OptionType::CE => 0,
        OptionType::PE => 1,
    }
}
