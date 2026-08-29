use std::{cmp::Ordering, time::Duration};

use chrono::{DateTime, Utc};
use redis::Commands;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::json;
use tauri::Emitter;
use tokio::time;

use crate::{
    exchange_calendar::nearest_weekly_expiry_label,
    models::{
        MarketDataState, OptionLeg, OptionType, QuoteFreshness, QuoteQuality, QuoteSource, Tick,
    },
    state::AppState,
};

const NSE_CHAIN_URL: &str = "http://127.0.0.1:8765/chain/NIFTY";
const GREEKS_URL: &str = "http://127.0.0.1:8765/greeks";
const NIFTY_SYMBOL: &str = "NIFTY";
const REDIS_TTL_SECONDS: u64 = 15;

#[derive(Debug, PartialEq, Eq)]
enum PollError {
    SessionExpired,
    RateLimited,
    Other(String),
}

#[derive(Debug, Deserialize)]
struct NseChainResponse {
    records: Option<NseRecords>,
    provenance: Option<NseProvenance>,
}

#[derive(Debug, Deserialize)]
struct NseRecords {
    data: Option<Vec<NseRow>>,
    #[serde(rename = "underlyingValue")]
    underlying_value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct NseProvenance {
    symbol: String,
    observed_at: String,
    freshness: QuoteFreshness,
    decision_eligible: bool,
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
    last_price: Option<f64>,
    #[serde(rename = "bidprice", alias = "bidPrice", default)]
    bid_price: Option<f64>,
    #[serde(rename = "askPrice", alias = "askprice", default)]
    ask_price: Option<f64>,
    #[serde(rename = "openInterest", default)]
    open_interest: Option<f64>,
    #[serde(rename = "changeinOpenInterest", default)]
    oi_change: Option<f64>,
    #[serde(rename = "totalTradedVolume", default)]
    volume: Option<f64>,
    #[serde(rename = "impliedVolatility", default)]
    iv: Option<f64>,
}

#[derive(Debug)]
struct ChainSnapshot {
    legs: Vec<OptionLeg>,
    spot: f64,
    observed_at_ms: i64,
    freshness: QuoteFreshness,
    decision_eligible: bool,
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
    wait_for_process_token(&state).await;
    let client = build_client();
    let mut cookies;

    loop {
        crate::bt_info!("poller", "fetching NSE session cookie");
        match fetch_nse_cookies(&client).await {
            Ok(c) => {
                cookies = c;
                crate::bt_info!("poller", "NSE session established");
            }
            Err(error) => {
                crate::bt_error!("poller", "NSE session failed: {error}");
                time::sleep(Duration::from_secs(10)).await;
                continue;
            }
        }

        let mut interval = time::interval(Duration::from_secs(3));
        let mut consecutive_errors = 0u32;
        let mut success_logged = false;

        loop {
            interval.tick().await;

            let expiry = match nearest_weekly_expiry_label(Utc::now()) {
                Ok(expiry) => expiry,
                Err(error) => {
                    crate::bt_error!("poller", "expiry unavailable: {error}");
                    time::sleep(Duration::from_secs(60)).await;
                    continue;
                }
            };

            match fetch_chain(&client, &state, &cookies, &expiry).await {
                Ok(snapshot) => {
                    consecutive_errors = 0;
                    let pcr = compute_pcr(&snapshot.legs);

                    update_spot_in_redis(
                        &state,
                        snapshot.spot,
                        snapshot.observed_at_ms,
                        snapshot.freshness,
                    )
                    .await;

                    match write_chain_to_redis(&state, &expiry, &snapshot.legs, pcr).await {
                        Ok(()) => {
                            let mut computed_greeks = 0;
                            if snapshot.decision_eligible {
                                match enrich_chain_with_greeks(
                                    &client,
                                    &state,
                                    &expiry,
                                    &snapshot.legs,
                                    snapshot.spot,
                                    pcr,
                                )
                                .await
                                {
                                    Ok(count) => {
                                        computed_greeks = count;
                                    }
                                    Err(error) => {
                                        crate::bt_error!("poller", "greeks skipped: {error}");
                                    }
                                }
                            } else {
                                crate::bt_info!(
                                    "poller",
                                    "greeks skipped: chain is not decision-eligible"
                                );
                            }

                            emit_chain_update(&state, &expiry);
                            if !success_logged {
                                let pcr_label = pcr
                                    .map(|value| format!("{value:.2}"))
                                    .unwrap_or_else(|| String::from("unknown"));
                                crate::bt_info!(
                                    "poller",
                                    "chain ready for NIFTY {expiry} spot={:.2} pcr={} freshness={:?} greeks={} ",
                                    snapshot.spot,
                                    pcr_label,
                                    snapshot.freshness,
                                    computed_greeks,
                                );
                                success_logged = true;
                            }
                        }
                        Err(error) => {
                            crate::bt_error!("poller", "redis write failed: {error}");
                        }
                    }
                }
                Err(PollError::SessionExpired) => {
                    crate::bt_info!("poller", "NSE session expired, refreshing");
                    break;
                }
                Err(PollError::RateLimited) => {
                    crate::bt_error!("poller", "NSE rate limited, backing off 30s");
                    time::sleep(Duration::from_secs(30)).await;
                    consecutive_errors += 1;
                    if consecutive_errors > 5 {
                        crate::bt_error!("poller", "too many errors, refreshing session");
                        break;
                    }
                }
                Err(PollError::Other(error)) => {
                    crate::bt_error!("poller", "{error}");
                    consecutive_errors += 1;
                    time::sleep(Duration::from_secs(5)).await;
                    if consecutive_errors > 10 {
                        crate::bt_error!("poller", "too many errors, refreshing session");
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
    pcr: Option<f64>,
) -> Result<usize, String> {
    let response = client
        .post(GREEKS_URL)
        .header("X-Process-Token", state.process_token()?)
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
        crate::bt_error!("poller", "greeks partial error: {error}");
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

async fn wait_for_process_token(state: &AppState) {
    for _ in 0..100 {
        if state.process_token().is_ok() {
            return;
        }
        time::sleep(Duration::from_millis(100)).await;
    }
    crate::bt_error!("poller", "process token unavailable after startup wait");
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
    state: &AppState,
    _cookies: &str,
    expiry: &str,
) -> Result<ChainSnapshot, PollError> {
    let token = state
        .process_token()
        .map_err(|error| PollError::Other(format!("process token unavailable: {error}")))?;
    let response = client
        .get(NSE_CHAIN_URL)
        .header("Accept", "application/json")
        .header("X-Process-Token", token)
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

    parse_chain_body(&body, expiry)
}

fn parse_chain_body(body: &str, expiry: &str) -> Result<ChainSnapshot, PollError> {
    // Check for Python-level error response.
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(body) {
        if val.get("error").is_some() {
            return Err(PollError::SessionExpired);
        }
    }

    let chain: NseChainResponse = serde_json::from_str(body)
        .map_err(|e| PollError::Other(format!("chain JSON parse failed: {e}")))?;

    let records = chain
        .records
        .ok_or_else(|| PollError::Other("chain missing records".to_string()))?;
    let provenance = chain
        .provenance
        .ok_or_else(|| PollError::Other("chain missing provenance".to_string()))?;
    if provenance.symbol != NIFTY_SYMBOL {
        return Err(PollError::Other(format!(
            "chain symbol mismatch: expected {NIFTY_SYMBOL}, received {}",
            provenance.symbol
        )));
    }
    if provenance.decision_eligible && provenance.freshness != QuoteFreshness::Fresh {
        return Err(PollError::Other(String::from(
            "chain provenance marks non-fresh data decision-eligible",
        )));
    }
    let observed_at_ms = DateTime::parse_from_rfc3339(&provenance.observed_at)
        .map_err(|error| PollError::Other(format!("chain observed_at invalid: {error}")))?
        .timestamp_millis();

    let spot = records
        .underlying_value
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| PollError::Other("chain missing valid underlying value".to_string()))?;
    let target_expiry_lower = expiry.to_lowercase();

    let mut legs = Vec::new();
    let rows = records
        .data
        .ok_or_else(|| PollError::Other("chain missing rows".to_string()))?;
    for row in rows {
        if !row.strike_price.is_finite() || row.strike_price <= 0.0 {
            return Err(PollError::Other(String::from(
                "chain contains invalid strike",
            )));
        }
        if row.expiry_date.to_lowercase() != target_expiry_lower {
            continue;
        }

        if let Some(ce) = row.ce {
            legs.push(map_nse_leg(
                &row.expiry_date,
                row.strike_price,
                OptionType::CE,
                &ce,
                observed_at_ms,
                provenance.freshness,
            )?);
        }
        if let Some(pe) = row.pe {
            legs.push(map_nse_leg(
                &row.expiry_date,
                row.strike_price,
                OptionType::PE,
                &pe,
                observed_at_ms,
                provenance.freshness,
            )?);
        }
    }

    if legs.is_empty() {
        return Err(PollError::Other(format!(
            "chain has no legs for expiry {expiry}"
        )));
    }

    legs.sort_by(|a, b| {
        a.strike
            .partial_cmp(&b.strike)
            .unwrap_or(Ordering::Equal)
            .then_with(|| option_type_rank(&a.option_type).cmp(&option_type_rank(&b.option_type)))
    });

    Ok(ChainSnapshot {
        legs,
        spot,
        observed_at_ms,
        freshness: provenance.freshness,
        decision_eligible: provenance.decision_eligible,
    })
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
    observed_at_ms: i64,
    freshness: QuoteFreshness,
) -> Result<OptionLeg, PollError> {
    let received_at_ms = Utc::now().timestamp_millis();
    let ltp = finite_nonnegative(leg.last_price)
        .ok_or_else(|| PollError::Other(String::from("chain leg missing valid lastPrice")))?;
    let bid = finite_nonnegative(leg.bid_price);
    let ask = finite_nonnegative(leg.ask_price);
    let volume = finite_nonnegative_u64(leg.volume);
    let oi = finite_nonnegative_u64(leg.open_interest);
    let oi_change = finite_i64(leg.oi_change);
    let iv = finite_positive(leg.iv);
    let mut field_completeness = vec![String::from("ltp")];
    for (field, present) in [
        ("bid", bid.is_some()),
        ("ask", ask.is_some()),
        ("volume", volume.is_some()),
        ("oi", oi.is_some()),
        ("oi_change", oi_change.is_some()),
        ("iv", iv.is_some()),
    ] {
        if present {
            field_completeness.push(field.into());
        }
    }

    Ok(OptionLeg {
        schema_version: Tick::SCHEMA_VERSION,
        symbol: NIFTY_SYMBOL.to_string(),
        expiry: expiry.to_string(),
        strike,
        option_type,
        ltp,
        bid,
        ask,
        volume,
        oi,
        oi_change,
        iv,
        delta: None,
        gamma: None,
        theta: None,
        vega: None,
        source_type: QuoteSource::NseChain,
        observed_at_ms,
        received_at_ms,
        sequence: None,
        field_completeness,
        freshness,
        quality: QuoteQuality::ReferenceOnly,
        fallback: false,
        fallback_reason: None,
    })
}

fn finite_nonnegative(value: Option<f64>) -> Option<f64> {
    value.filter(|number| number.is_finite() && *number >= 0.0)
}

fn finite_positive(value: Option<f64>) -> Option<f64> {
    value.filter(|number| number.is_finite() && *number > 0.0)
}

fn finite_nonnegative_u64(value: Option<f64>) -> Option<u64> {
    value
        .filter(|number| {
            number.is_finite()
                && *number >= 0.0
                && number.fract() == 0.0
                && *number <= u64::MAX as f64
        })
        .map(|number| number as u64)
}

fn finite_i64(value: Option<f64>) -> Option<i64> {
    value
        .filter(|number| {
            number.is_finite()
                && number.fract() == 0.0
                && *number >= i64::MIN as f64
                && *number <= i64::MAX as f64
        })
        .map(|number| number as i64)
}

async fn update_spot_in_redis(
    state: &AppState,
    spot: f64,
    observed_at_ms: i64,
    freshness: QuoteFreshness,
) {
    let mut redis_guard = state.redis.lock().await;
    let connection = match redis_guard.as_mut() {
        Some(c) => c,
        None => return,
    };

    let existing: redis::RedisResult<Option<String>> = connection.get("tick:NIFTY");
    if matches!(existing, Ok(Some(_))) {
        return;
    }

    let received_at_ms = Utc::now().timestamp_millis();
    let tick = Tick {
        schema_version: Tick::SCHEMA_VERSION,
        symbol: NIFTY_SYMBOL.to_string(),
        token: String::from("NSE_CHAIN:NIFTY"),
        ltp: spot,
        open: None,
        high: None,
        low: None,
        close: None,
        volume: None,
        oi: None,
        bid: None,
        ask: None,
        exchange: crate::models::Exchange::NSE,
        source_type: QuoteSource::NseChain,
        observed_at_ms,
        received_at_ms,
        sequence: None,
        field_completeness: vec!["ltp".into()],
        freshness,
        quality: QuoteQuality::ReferenceOnly,
        fallback: true,
        fallback_reason: Some(String::from(
            "Angel stream unavailable; NSE chain underlying reference",
        )),
        market_state: if freshness == QuoteFreshness::Fresh {
            MarketDataState::Degraded
        } else {
            MarketDataState::Stale
        },
    };

    if let Ok(payload) = serde_json::to_string(&tick) {
        let _: redis::RedisResult<()> = connection.set_ex("tick:NIFTY", payload, 15);
    }
}

async fn write_chain_to_redis(
    state: &AppState,
    expiry: &str,
    legs: &[OptionLeg],
    pcr: Option<f64>,
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
    if let Some(pcr) = pcr {
        let pcr_result: redis::RedisResult<()> =
            connection.set_ex(&pcr_key, pcr.to_string(), REDIS_TTL_SECONDS);
        pcr_result.map_err(|e| format!("SET {pcr_key} failed: {e}"))?;
    }

    let latest_leg = legs.iter().max_by_key(|leg| leg.observed_at_ms);
    let observed_at_ms = latest_leg.map(|leg| leg.observed_at_ms);
    let freshness = latest_leg
        .map(|leg| leg.freshness)
        .unwrap_or(QuoteFreshness::Unknown);

    let health = json!({
        "schema_version": 1,
        "state": if freshness == QuoteFreshness::Fresh { "LIVE" } else { "STALE" },
        "source": "RUST_NSE_CHAIN_POLLER",
        "last_success_ms": Utc::now().timestamp_millis(),
        "observed_at_ms": observed_at_ms,
        "freshness": freshness,
        "expiry": expiry,
        "leg_count": legs.len(),
        "pcr_available": pcr.is_some(),
    });
    let health_result: redis::RedisResult<()> =
        connection.set_ex("health:nse_chain_poller", health.to_string(), 30);
    health_result.map_err(|e| format!("SET poller health failed: {e}"))?;

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
        crate::bt_error!("poller", "chain_update emit failed: {error}");
    }
}

fn compute_pcr(legs: &[OptionLeg]) -> Option<f64> {
    let mut call_oi = 0_u64;
    let mut put_oi = 0_u64;

    for leg in legs {
        let oi = leg.oi?;
        match leg.option_type {
            OptionType::CE => call_oi = call_oi.saturating_add(oi),
            OptionType::PE => put_oi = put_oi.saturating_add(oi),
        }
    }

    if call_oi == 0 {
        return None;
    }

    Some(put_oi as f64 / call_oi as f64)
}

fn option_type_rank(option_type: &OptionType) -> u8 {
    match option_type {
        OptionType::CE => 0,
        OptionType::PE => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_chain_body, PollError};

    #[test]
    fn malformed_chain_payload_fails_closed() {
        let error = parse_chain_body("not-json", "27-Aug-2026").unwrap_err();
        assert!(
            matches!(error, PollError::Other(message) if message.contains("JSON parse failed"))
        );
    }

    #[test]
    fn upstream_error_forces_session_refresh() {
        assert_eq!(
            parse_chain_body(r#"{"error":"session expired"}"#, "27-Aug-2026").unwrap_err(),
            PollError::SessionExpired
        );
    }

    #[test]
    fn valid_chain_is_filtered_and_sorted_deterministically() {
        let body = r#"{
            "records": {
                "underlyingValue": 25123.45,
                "data": [
                    {
                        "strikePrice": 25200,
                        "expiryDate": "27-Aug-2026",
                        "CE": {"lastPrice": 10, "openInterest": 20},
                        "PE": {"lastPrice": 30, "openInterest": 40}
                    },
                    {
                        "strikePrice": 25100,
                        "expiryDate": "27-Aug-2026",
                        "CE": {"lastPrice": 50, "openInterest": 60}
                    },
                    {
                        "strikePrice": 25000,
                        "expiryDate": "03-Sep-2026",
                        "CE": {"lastPrice": 70, "openInterest": 80}
                    }
                ]
            },
            "provenance": {
                "symbol": "NIFTY",
                "observed_at": "2026-08-27T10:05:00+05:30",
                "freshness": "FRESH",
                "decision_eligible": true
            }
        }"#;
        let snapshot = parse_chain_body(body, "27-Aug-2026").unwrap();
        assert_eq!(snapshot.spot, 25123.45);
        assert!(snapshot.decision_eligible);
        assert_eq!(snapshot.freshness, crate::models::QuoteFreshness::Fresh);
        assert_eq!(snapshot.legs.len(), 3);
        assert_eq!(snapshot.legs[0].strike, 25100.0);
        assert_eq!(snapshot.legs[1].strike, 25200.0);
        assert_eq!(snapshot.legs[2].strike, 25200.0);
        assert_eq!(snapshot.legs[0].bid, None);
        assert!(!snapshot.legs[0]
            .field_completeness
            .contains(&String::from("bid")));
    }

    #[test]
    fn missing_provenance_and_fake_decision_eligibility_fail_closed() {
        let no_provenance = r#"{
            "records": {"underlyingValue": 25123.45, "data": []}
        }"#;
        assert!(matches!(
            parse_chain_body(no_provenance, "27-Aug-2026").unwrap_err(),
            PollError::Other(message) if message.contains("missing provenance")
        ));

        let inconsistent = r#"{
            "records": {"underlyingValue": 25123.45, "data": []},
            "provenance": {
                "symbol": "NIFTY",
                "observed_at": "2026-08-27T10:05:00+05:30",
                "freshness": "STALE",
                "decision_eligible": true
            }
        }"#;
        assert!(matches!(
            parse_chain_body(inconsistent, "27-Aug-2026").unwrap_err(),
            PollError::Other(message) if message.contains("non-fresh")
        ));
    }
}
