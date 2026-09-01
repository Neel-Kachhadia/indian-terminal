use std::ops::Range;
use std::time::Duration;

use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use redis::Commands;
use serde::Deserialize;
use serde_json::json;
use tokio::time::{timeout_at, Instant};
use tokio_tungstenite::tungstenite::handshake::client::Request;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::{
    models::{Exchange, MarketDataState, QuoteFreshness, QuoteQuality, QuoteSource, Tick},
    state::{AppState, FeedPhase},
};

const LOCAL_FEED_BRIDGE_URL: &str = "ws://127.0.0.1:8765/internal/angel/feed";
const FEED_INSTRUMENT_URL: &str = "http://127.0.0.1:8765/instruments/feed/NIFTY";
const INSTRUMENT_SCHEMA_VERSION: u32 = 1;
const REDIS_TTL_SECONDS: u64 = 60;
const INITIAL_DATA_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_EXCHANGE_TIMESTAMP_AGE_MS: i64 = 30_000;
const MAX_EXCHANGE_TIMESTAMP_FUTURE_SKEW_MS: i64 = 5_000;

// Angel SmartAPI WebSocket V2 byte contract. Offsets match official SDK.
const MODE_OFFSET: usize = 0;
const EXCHANGE_OFFSET: usize = 1;
const TOKEN_RANGE: Range<usize> = 2..27;
const SEQUENCE_RANGE: Range<usize> = 27..35;
const EXCHANGE_TIMESTAMP_RANGE: Range<usize> = 35..43;
const LTP_RANGE: Range<usize> = 43..51;
const LAST_TRADED_QUANTITY_RANGE: Range<usize> = 51..59;
const AVERAGE_TRADED_PRICE_RANGE: Range<usize> = 59..67;
const VOLUME_RANGE: Range<usize> = 67..75;
const TOTAL_BUY_QUANTITY_RANGE: Range<usize> = 75..83;
const TOTAL_SELL_QUANTITY_RANGE: Range<usize> = 83..91;
const OPEN_RANGE: Range<usize> = 91..99;
const HIGH_RANGE: Range<usize> = 99..107;
const LOW_RANGE: Range<usize> = 107..115;
const CLOSE_RANGE: Range<usize> = 115..123;
const LAST_TRADED_TIMESTAMP_RANGE: Range<usize> = 123..131;
const OPEN_INTEREST_RANGE: Range<usize> = 131..139;
const OPEN_INTEREST_CHANGE_RANGE: Range<usize> = 139..147;
const SNAP_DEPTH_RANGE: Range<usize> = 147..347;
const UPPER_CIRCUIT_RANGE: Range<usize> = 347..355;
const LOWER_CIRCUIT_RANGE: Range<usize> = 355..363;
const YEAR_HIGH_RANGE: Range<usize> = 363..371;
const YEAR_LOW_RANGE: Range<usize> = 371..379;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SubscriptionMode {
    Ltp = 1,
    Quote = 2,
    SnapQuote = 3,
}

impl SubscriptionMode {
    fn expected_len(self) -> usize {
        match self {
            Self::Ltp => 51,
            Self::Quote => 123,
            Self::SnapQuote => 379,
        }
    }
}

impl TryFrom<u8> for SubscriptionMode {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ltp),
            2 => Ok(Self::Quote),
            3 => Ok(Self::SnapQuote),
            _ => Err(format!("unsupported Angel subscription mode: {value}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DecodedAngelPacket {
    mode: SubscriptionMode,
    exchange_type: u8,
    token: String,
    sequence_number: i64,
    exchange_timestamp: i64,
    ltp: i64,
    volume: Option<u64>,
    open: Option<i64>,
    high: Option<i64>,
    low: Option<i64>,
    close: Option<i64>,
    open_interest: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct ResolvedFeedInstrument {
    schema_version: u32,
    token: String,
    symbol: String,
    canonical_symbol: String,
    name: String,
    exchange: String,
    instrument_type: String,
}

impl ResolvedFeedInstrument {
    fn validate(self) -> Result<Self, String> {
        let identity = |value: &str| {
            value
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>()
                .to_ascii_uppercase()
        };
        if self.schema_version != INSTRUMENT_SCHEMA_VERSION
            || self.exchange != "NSE"
            || self.instrument_type != "AMXIDX"
            || identity(&self.symbol) != "NIFTY"
            || !matches!(
                identity(&self.canonical_symbol).as_str(),
                "NIFTY" | "NIFTY50"
            )
            || !matches!(identity(&self.name).as_str(), "NIFTY" | "NIFTY50")
            || self.token.is_empty()
            || self.token.len() > 25
            || !self.token.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(String::from(
                "canonical NIFTY index instrument validation failed",
            ));
        }
        Ok(self)
    }

    fn exchange_type(&self) -> Result<u8, String> {
        match self.exchange.as_str() {
            "NSE" => Ok(1),
            _ => Err(String::from("unsupported Angel feed exchange")),
        }
    }
}

pub async fn run(state: AppState) -> Result<(), String> {
    let instrument = resolve_feed_instrument(&state).await.map_err(|error| {
        let message = format!("CONFIGURATION / INSTRUMENT RESOLUTION ERROR: {error}");
        let _ = state.set_feed_phase(FeedPhase::Degraded, Some(message.clone()));
        message
    })?;
    crate::bt_info!("ingester", "authenticating");

    run_websocket_at(
        &state,
        &instrument,
        &state.process_token()?,
        LOCAL_FEED_BRIDGE_URL,
        INITIAL_DATA_TIMEOUT,
    )
    .await
}

async fn run_websocket_at(
    state: &AppState,
    instrument: &ResolvedFeedInstrument,
    process_token: &str,
    websocket_url: &str,
    initial_data_timeout: Duration,
) -> Result<(), String> {
    let exchange_type = instrument.exchange_type()?;
    let request = local_bridge_request_at(websocket_url, process_token)?;
    let (mut websocket, _response) = connect_async(request)
        .await
        .map_err(|error| format!("secure Angel feed bridge failed: {error}"))?;
    state.set_feed_phase(FeedPhase::Authenticated, None)?;
    state.set_feed_phase(FeedPhase::ConnectingFeed, None)?;
    crate::bt_info!("ingester", "connected to secure feed bridge");

    state.set_feed_phase(FeedPhase::Subscribing, None)?;
    let subscription = subscription_request(instrument)?;
    websocket
        .send(Message::Text(subscription.to_string()))
        .await
        .map_err(|error| format!("Angel subscription failed: {error}"))?;
    state.set_feed_phase(FeedPhase::WaitingForData, None)?;
    crate::bt_info!(
        "ingester",
        "subscription sent; waiting for verified market data"
    );

    let initial_deadline = Instant::now() + initial_data_timeout;
    let mut live_proven = false;
    loop {
        let message = if live_proven {
            websocket.next().await
        } else {
            match timeout_at(initial_deadline, websocket.next()).await {
                Ok(message) => message,
                Err(_) => {
                    let reason = "Angel subscription produced no valid market data before timeout";
                    let _ = state.set_feed_phase(FeedPhase::Degraded, Some(reason.to_string()));
                    return Err(reason.to_string());
                }
            }
        };

        match message {
            Some(Ok(Message::Text(text))) => handle_text_control(&text)?,
            Some(Ok(Message::Binary(bytes))) => {
                let packet =
                    match decode_packet(&bytes, Some(SubscriptionMode::Quote), Some(exchange_type))
                    {
                        Ok(packet) => packet,
                        Err(error) => {
                            crate::bt_error!(
                                "ingester",
                                "rejected malformed market packet: {error}"
                            );
                            continue;
                        }
                    };
                if packet.token != instrument.token {
                    continue;
                }
                let tick = packet_to_tick(packet, instrument)?;
                if !live_proven {
                    state.set_feed_phase(FeedPhase::Live, None)?;
                    live_proven = true;
                }
                publish_tick(state, &tick).await;
                write_tick_to_redis(state, &tick).await;
            }
            Some(Ok(Message::Close(_))) => return Err(String::from("Angel websocket closed")),
            Some(Ok(Message::Ping(_)))
            | Some(Ok(Message::Pong(_)))
            | Some(Ok(Message::Frame(_))) => {}
            Some(Err(error)) => return Err(format!("Angel websocket error: {error}")),
            None => return Err(String::from("Angel websocket stream ended")),
        }
    }
}

async fn resolve_feed_instrument(state: &AppState) -> Result<ResolvedFeedInstrument, String> {
    resolve_feed_instrument_at(FEED_INSTRUMENT_URL, &state.process_token()?).await
}

async fn resolve_feed_instrument_at(
    url: &str,
    process_token: &str,
) -> Result<ResolvedFeedInstrument, String> {
    let response = reqwest::Client::new()
        .get(url)
        .header("X-Process-Token", process_token)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|error| format!("instrument service unavailable: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "instrument service rejected resolution with HTTP {}",
            response.status()
        ));
    }
    response
        .json::<ResolvedFeedInstrument>()
        .await
        .map_err(|error| format!("instrument response invalid: {error}"))?
        .validate()
}

fn subscription_request(instrument: &ResolvedFeedInstrument) -> Result<serde_json::Value, String> {
    Ok(json!({
        "correlationID": "niftyquote",
        "action": 1,
        "params": {
            "mode": SubscriptionMode::Quote as u8,
            "tokenList": [
                {
                    "exchangeType": instrument.exchange_type()?,
                    "tokens": [instrument.token.clone()]
                }
            ]
        }
    }))
}

#[cfg(test)]
fn local_bridge_request(process_token: &str) -> Result<Request, String> {
    local_bridge_request_at(LOCAL_FEED_BRIDGE_URL, process_token)
}

fn local_bridge_request_at(websocket_url: &str, process_token: &str) -> Result<Request, String> {
    use tokio_tungstenite::tungstenite::handshake::client::generate_key;

    if process_token.trim().is_empty() {
        return Err(String::from("compute process token unavailable"));
    }
    let parsed = reqwest::Url::parse(websocket_url)
        .map_err(|error| format!("invalid Angel websocket URL: {error}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| String::from("Angel websocket URL missing host"))?;
    let host_header = parsed
        .port()
        .map_or_else(|| host.to_string(), |port| format!("{host}:{port}"));
    Request::builder()
        .uri(websocket_url)
        .header("Host", host_header)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", generate_key())
        .header("X-Process-Token", process_token)
        .body(())
        .map_err(|error| error.to_string())
}

fn handle_text_control(text: &str) -> Result<(), String> {
    if text.eq_ignore_ascii_case("pong") {
        return Ok(());
    }
    let payload: serde_json::Value = serde_json::from_str(text)
        .map_err(|_| String::from("malformed Angel text control message"))?;
    if payload.get("status").and_then(serde_json::Value::as_bool) == Some(false) {
        return Err(String::from("Angel rejected websocket subscription"));
    }
    Ok(())
}

fn decode_packet(
    bytes: &[u8],
    expected_mode: Option<SubscriptionMode>,
    expected_exchange: Option<u8>,
) -> Result<DecodedAngelPacket, String> {
    let mode_byte = *bytes
        .get(MODE_OFFSET)
        .ok_or_else(|| String::from("truncated Angel packet: missing mode"))?;
    let mode = SubscriptionMode::try_from(mode_byte)?;
    if let Some(expected) = expected_mode {
        if mode != expected {
            return Err(format!(
                "wrong Angel subscription mode: expected {:?}, received {:?}",
                expected, mode
            ));
        }
    }
    if bytes.len() != mode.expected_len() {
        return Err(format!(
            "wrong Angel {:?} packet length: expected {}, received {}",
            mode,
            mode.expected_len(),
            bytes.len()
        ));
    }

    let exchange_type = *bytes
        .get(EXCHANGE_OFFSET)
        .ok_or_else(|| String::from("truncated Angel packet: missing exchange type"))?;
    if !matches!(exchange_type, 1 | 2 | 3 | 4 | 5 | 7 | 13) {
        return Err(format!("unsupported Angel exchange type: {exchange_type}"));
    }
    if let Some(expected) = expected_exchange {
        if exchange_type != expected {
            return Err(format!(
                "wrong Angel exchange type: expected {expected}, received {exchange_type}"
            ));
        }
    }

    let token = parse_token(bytes)?;
    let sequence_number = read_i64(bytes, SEQUENCE_RANGE, "sequence_number")?;
    let exchange_timestamp = read_i64(bytes, EXCHANGE_TIMESTAMP_RANGE, "exchange_timestamp")?;
    let ltp = read_i64(bytes, LTP_RANGE, "last_traded_price")?;
    if sequence_number < 0 || exchange_timestamp <= 0 || ltp <= 0 {
        return Err(String::from("malformed Angel common packet fields"));
    }

    let mut packet = DecodedAngelPacket {
        mode,
        exchange_type,
        token,
        sequence_number,
        exchange_timestamp,
        ltp,
        volume: None,
        open: None,
        high: None,
        low: None,
        close: None,
        open_interest: None,
    };
    if matches!(mode, SubscriptionMode::Quote | SubscriptionMode::SnapQuote) {
        let last_quantity = read_i64(bytes, LAST_TRADED_QUANTITY_RANGE, "last_traded_quantity")?;
        let average_price = read_i64(bytes, AVERAGE_TRADED_PRICE_RANGE, "average_traded_price")?;
        let volume = read_nonnegative_u64(bytes, VOLUME_RANGE, "volume_trade_for_the_day")?;
        let total_buy = read_f64(bytes, TOTAL_BUY_QUANTITY_RANGE, "total_buy_quantity")?;
        let total_sell = read_f64(bytes, TOTAL_SELL_QUANTITY_RANGE, "total_sell_quantity")?;
        let open = read_i64(bytes, OPEN_RANGE, "open_price_of_the_day")?;
        let high = read_i64(bytes, HIGH_RANGE, "high_price_of_the_day")?;
        let low = read_i64(bytes, LOW_RANGE, "low_price_of_the_day")?;
        let close = read_i64(bytes, CLOSE_RANGE, "closed_price")?;
        if last_quantity < 0
            || average_price < 0
            || !total_buy.is_finite()
            || total_buy < 0.0
            || !total_sell.is_finite()
            || total_sell < 0.0
            || open <= 0
            || high < open.max(close).max(ltp)
            || low <= 0
            || low > open.min(close).min(ltp)
            || close <= 0
        {
            return Err(String::from("malformed Angel Quote OHLC/quantity fields"));
        }
        packet.volume = Some(volume);
        packet.open = Some(open);
        packet.high = Some(high);
        packet.low = Some(low);
        packet.close = Some(close);
    }
    if mode == SubscriptionMode::SnapQuote {
        let last_trade_timestamp =
            read_i64(bytes, LAST_TRADED_TIMESTAMP_RANGE, "last_traded_timestamp")?;
        let open_interest = read_nonnegative_u64(bytes, OPEN_INTEREST_RANGE, "open_interest")?;
        let _oi_change = read_i64(
            bytes,
            OPEN_INTEREST_CHANGE_RANGE,
            "open_interest_change_percentage",
        )?;
        let depth = bytes
            .get(SNAP_DEPTH_RANGE)
            .ok_or_else(|| String::from("truncated Angel SnapQuote depth"))?;
        let upper = read_i64(bytes, UPPER_CIRCUIT_RANGE, "upper_circuit_limit")?;
        let lower = read_i64(bytes, LOWER_CIRCUIT_RANGE, "lower_circuit_limit")?;
        let year_high = read_i64(bytes, YEAR_HIGH_RANGE, "52_week_high_price")?;
        let year_low = read_i64(bytes, YEAR_LOW_RANGE, "52_week_low_price")?;
        if last_trade_timestamp <= 0
            || depth.len() != 200
            || upper < 0
            || lower < 0
            || year_high < 0
            || year_low < 0
        {
            return Err(String::from("malformed Angel SnapQuote fields"));
        }
        packet.open_interest = Some(open_interest);
    }
    Ok(packet)
}

fn parse_token(bytes: &[u8]) -> Result<String, String> {
    let raw = bytes
        .get(TOKEN_RANGE)
        .ok_or_else(|| String::from("truncated Angel packet token"))?;
    let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
    let token = std::str::from_utf8(&raw[..end])
        .map_err(|_| String::from("malformed Angel token encoding"))?
        .trim();
    if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(String::from("malformed Angel token"));
    }
    Ok(token.to_string())
}

fn read_i64(bytes: &[u8], range: Range<usize>, field: &str) -> Result<i64, String> {
    let raw = bytes
        .get(range)
        .ok_or_else(|| format!("truncated Angel field: {field}"))?;
    let array: [u8; 8] = raw
        .try_into()
        .map_err(|_| format!("wrong Angel field width: {field}"))?;
    Ok(i64::from_le_bytes(array))
}

fn read_nonnegative_u64(bytes: &[u8], range: Range<usize>, field: &str) -> Result<u64, String> {
    let value = read_i64(bytes, range, field)?;
    u64::try_from(value).map_err(|_| format!("negative Angel field: {field}"))
}

fn read_f64(bytes: &[u8], range: Range<usize>, field: &str) -> Result<f64, String> {
    let raw = bytes
        .get(range)
        .ok_or_else(|| format!("truncated Angel field: {field}"))?;
    let array: [u8; 8] = raw
        .try_into()
        .map_err(|_| format!("wrong Angel field width: {field}"))?;
    Ok(f64::from_le_bytes(array))
}

fn packet_to_tick(
    packet: DecodedAngelPacket,
    instrument: &ResolvedFeedInstrument,
) -> Result<Tick, String> {
    packet_to_tick_at(packet, instrument, Utc::now().timestamp_millis())
}

fn packet_to_tick_at(
    packet: DecodedAngelPacket,
    instrument: &ResolvedFeedInstrument,
    received_at_ms: i64,
) -> Result<Tick, String> {
    if packet.exchange_type != instrument.exchange_type()? || packet.token != instrument.token {
        return Err(String::from(
            "unsupported exchange for canonical NIFTY tick",
        ));
    }
    let observed_at_ms = if packet.exchange_timestamp < 1_000_000_000_000 {
        packet.exchange_timestamp.saturating_mul(1_000)
    } else {
        packet.exchange_timestamp
    };
    if received_at_ms.saturating_sub(observed_at_ms) > MAX_EXCHANGE_TIMESTAMP_AGE_MS
        || observed_at_ms.saturating_sub(received_at_ms) > MAX_EXCHANGE_TIMESTAMP_FUTURE_SKEW_MS
    {
        return Err(String::from("stale Angel exchange timestamp"));
    }
    let mut completeness = vec!["ltp".into()];
    if packet.volume.is_some() {
        completeness.extend(["open", "high", "low", "close", "volume"].map(String::from));
    }
    if packet.open_interest.is_some() {
        completeness.push("oi".into());
    }
    Ok(Tick {
        schema_version: Tick::SCHEMA_VERSION,
        symbol: instrument.symbol.clone(),
        token: packet.token,
        ltp: price_to_rupees(packet.ltp),
        open: packet.open.map(price_to_rupees),
        high: packet.high.map(price_to_rupees),
        low: packet.low.map(price_to_rupees),
        close: packet.close.map(price_to_rupees),
        volume: packet.volume,
        oi: packet.open_interest,
        bid: None,
        ask: None,
        exchange: Exchange::NSE,
        source_type: QuoteSource::AngelStream,
        observed_at_ms,
        received_at_ms,
        sequence: u64::try_from(packet.sequence_number).ok(),
        field_completeness: completeness,
        freshness: QuoteFreshness::Fresh,
        quality: QuoteQuality::ExecutableStream,
        fallback: false,
        fallback_reason: None,
        market_state: MarketDataState::Live,
    })
}

fn price_to_rupees(price: i64) -> f64 {
    price as f64 / 100.0
}

async fn publish_tick(state: &AppState, tick: &Tick) {
    let tick_tx = state.tick_tx.lock().await;
    let _ = tick_tx.send(tick.clone());
}

async fn write_tick_to_redis(state: &AppState, tick: &Tick) {
    let payload = match serde_json::to_string(tick) {
        Ok(payload) => payload,
        Err(error) => {
            crate::bt_error!("ingester", "tick serialize failed: {error}");
            return;
        }
    };
    let mut redis_guard = state.redis.lock().await;
    let connection = match redis_guard.as_mut() {
        Some(connection) => connection,
        None => {
            crate::bt_error!("ingester", "redis unavailable; tick not cached");
            return;
        }
    };
    let key = format!("tick:{}", tick.symbol);
    match store_tick_monotonic(connection, &key, &payload, tick) {
        Ok(accepted) if accepted > 0 => {
            let health = json!({
                "schema_version": 1,
                "state": "LIVE",
                "source": "RUST_TAURI_INGESTER",
                "last_success_ms": Utc::now().timestamp_millis(),
                "symbol": tick.symbol,
            });
            let _: redis::RedisResult<()> =
                connection.set_ex("health:angel_stream", health.to_string(), 120);
        }
        Ok(_) => {}
        Err(error) => crate::bt_error!("ingester", "redis write failed: {error}"),
    }
}

fn store_tick_monotonic(
    connection: &mut redis::Connection,
    key: &str,
    payload: &str,
    tick: &Tick,
) -> redis::RedisResult<i32> {
    redis::Script::new(
        r#"
        local current_raw = redis.call('GET', KEYS[1])
        if current_raw then
            local ok, current = pcall(cjson.decode, current_raw)
            if ok and current.schema_version == 1 then
                local current_seq = tonumber(current.sequence)
                local incoming_seq = tonumber(ARGV[4])
                if current_seq and incoming_seq and incoming_seq ~= current_seq then
                    if incoming_seq < current_seq then return 0 end
                else
                    local current_observed = tonumber(current.observed_at_ms) or 0
                    local current_received = tonumber(current.received_at_ms) or 0
                    local incoming_observed = tonumber(ARGV[2]) or 0
                    local incoming_received = tonumber(ARGV[3]) or 0
                    if incoming_observed < current_observed then return 0 end
                    if incoming_observed == current_observed and incoming_received <= current_received then return 0 end
                end
            end
        end
        redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[5])
        return 1
        "#,
    )
    .key(key)
    .arg(payload)
    .arg(tick.observed_at_ms)
    .arg(tick.received_at_ms)
    .arg(tick.sequence.map(|value| value.to_string()).unwrap_or_default())
    .arg(REDIS_TTL_SECONDS)
    .invoke(connection)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_packet, local_bridge_request, packet_to_tick_at, resolve_feed_instrument_at,
        run_websocket_at, subscription_request, ResolvedFeedInstrument, SubscriptionMode,
    };
    use crate::{
        models::Tick,
        state::{AppState, FeedPhase},
    };
    use chrono::Utc;
    use futures_util::{SinkExt, StreamExt};
    use std::time::Duration;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        sync::broadcast,
        time::timeout,
    };
    use tokio_tungstenite::{accept_async, tungstenite::protocol::Message as WsMessage};

    const FIXTURE_TOKEN: &str = "99123456";
    const CAPTURED_NIFTY_QUOTE_HEX: &str = "020139393932363030300000000000000000000000000000000000000000000000000098a5e75ba0010000ddbc240000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004bbd240000000000ebd6240000000000b6a224000000000068be240000000000";

    fn instrument(token: &str) -> ResolvedFeedInstrument {
        ResolvedFeedInstrument {
            schema_version: 1,
            token: token.to_string(),
            symbol: String::from("NIFTY"),
            canonical_symbol: String::from("NIFTY"),
            name: String::from("NIFTY 50"),
            exchange: String::from("NSE"),
            instrument_type: String::from("AMXIDX"),
        }
    }

    fn write_i64(packet: &mut [u8], start: usize, value: i64) {
        packet[start..start + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn common_fixture(mode: u8, length: usize) -> Vec<u8> {
        // Literals below come from official SmartAPI Python SDK, independent of parser constants.
        let mut packet = vec![0_u8; length];
        packet[0] = mode;
        packet[1] = 1;
        packet[2..10].copy_from_slice(FIXTURE_TOKEN.as_bytes());
        write_i64(&mut packet, 27, 42);
        write_i64(&mut packet, 35, 1_756_000_000_123);
        write_i64(&mut packet, 43, 2_450_025);
        packet
    }

    fn quote_fixture(mode: u8, length: usize) -> Vec<u8> {
        let mut packet = common_fixture(mode, length);
        write_i64(&mut packet, 51, 75);
        write_i64(&mut packet, 59, 2_448_000);
        write_i64(&mut packet, 67, 987_654);
        packet[75..83].copy_from_slice(&1_250.0_f64.to_le_bytes());
        packet[83..91].copy_from_slice(&1_100.0_f64.to_le_bytes());
        write_i64(&mut packet, 91, 2_440_000);
        write_i64(&mut packet, 99, 2_460_000);
        write_i64(&mut packet, 107, 2_430_000);
        write_i64(&mut packet, 115, 2_445_000);
        packet
    }

    fn feed_state() -> (AppState, broadcast::Receiver<Tick>) {
        let (tick_tx, tick_rx) = broadcast::channel(8);
        let state = AppState::new(None, tick_tx);
        state
            .set_feed_phase(FeedPhase::WaitingForHealth, None)
            .unwrap();
        (state, tick_rx)
    }

    async fn websocket_fixture(payload: Option<Vec<u8>>) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut websocket = accept_async(stream).await.unwrap();
            let subscription = websocket.next().await.unwrap().unwrap();
            let request: serde_json::Value =
                serde_json::from_str(subscription.to_text().unwrap()).unwrap();
            assert_eq!(
                request["params"]["tokenList"][0]["tokens"][0],
                FIXTURE_TOKEN
            );
            if let Some(bytes) = payload {
                websocket.send(WsMessage::Binary(bytes)).await.unwrap();
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        (format!("ws://{address}"), server)
    }

    async fn instrument_http_fixture(body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_string();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let size = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..size]);
            assert!(request.contains("x-process-token: fixture-process-token"));
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{address}/instruments/feed/NIFTY")
    }

    #[test]
    fn official_ltp_packet_decodes_only_fields_present_in_mode() {
        let packet =
            decode_packet(&common_fixture(1, 51), Some(SubscriptionMode::Ltp), Some(1)).unwrap();
        assert_eq!(packet.token, FIXTURE_TOKEN);
        assert_eq!(packet.ltp, 2_450_025);
        assert_eq!(packet.exchange_timestamp, 1_756_000_000_123);
        assert_eq!(packet.volume, None);
        assert_eq!(packet.open, None);
        assert_eq!(packet.high, None);
        assert_eq!(packet.low, None);
        assert_eq!(packet.close, None);
    }

    #[test]
    fn official_quote_packet_decodes_documented_offsets() {
        let packet = decode_packet(
            &quote_fixture(2, 123),
            Some(SubscriptionMode::Quote),
            Some(1),
        )
        .unwrap();
        assert_eq!(packet.token, FIXTURE_TOKEN);
        assert_eq!(packet.ltp, 2_450_025);
        assert_eq!(packet.volume, Some(987_654));
        assert_eq!(packet.open, Some(2_440_000));
        assert_eq!(packet.high, Some(2_460_000));
        assert_eq!(packet.low, Some(2_430_000));
        assert_eq!(packet.close, Some(2_445_000));
        assert_eq!(packet.exchange_timestamp, 1_756_000_000_123);
    }

    #[test]
    fn captured_nifty_quote_packet_decodes_and_proves_fresh_live_data() {
        // Read-only Smart Stream capture, 2026-09-01 13:08:23 IST.  Packet
        // contains no credentials; SHA-256 d582787599f8bc66a9107b6d55ffcddf9acfa607312cac2c560d42a0189ff5c1.
        let bytes = hex::decode(CAPTURED_NIFTY_QUOTE_HEX).unwrap();
        let packet = decode_packet(&bytes, Some(SubscriptionMode::Quote), Some(1)).unwrap();
        assert_eq!(packet.token, "99926000");
        assert_eq!(packet.sequence_number, 0);
        assert_eq!(packet.exchange_timestamp, 1_788_248_303_000);
        assert_eq!(packet.ltp, 2_407_645);
        assert_eq!(packet.volume, Some(0));
        assert_eq!(packet.open, Some(2_407_755));
        assert_eq!(packet.high, Some(2_414_315));
        assert_eq!(packet.low, Some(2_400_950));
        assert_eq!(packet.close, Some(2_408_040));

        let tick = packet_to_tick_at(packet, &instrument("99926000"), 1_788_248_304_500).unwrap();
        assert_eq!(tick.ltp, 24_076.45);
        assert_eq!(tick.observed_at_ms, 1_788_248_303_000);
    }

    #[test]
    fn quote_packet_rejects_impossible_ohlc_and_quantity() {
        let mut impossible_ohlc = quote_fixture(2, 123);
        write_i64(&mut impossible_ohlc, 99, 2_400_000);
        assert!(decode_packet(&impossible_ohlc, Some(SubscriptionMode::Quote), Some(1)).is_err());

        let mut impossible_quantity = quote_fixture(2, 123);
        impossible_quantity[75..83].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(
            decode_packet(&impossible_quantity, Some(SubscriptionMode::Quote), Some(1)).is_err()
        );
    }

    #[test]
    fn stale_or_future_quote_timestamp_never_becomes_a_live_tick() {
        let bytes = hex::decode(CAPTURED_NIFTY_QUOTE_HEX).unwrap();
        let packet = decode_packet(&bytes, Some(SubscriptionMode::Quote), Some(1)).unwrap();
        assert!(
            packet_to_tick_at(packet.clone(), &instrument("99926000"), 1_788_248_333_001)
                .unwrap_err()
                .contains("stale")
        );
        assert!(
            packet_to_tick_at(packet, &instrument("99926000"), 1_788_248_297_999)
                .unwrap_err()
                .contains("stale")
        );
    }

    #[test]
    fn official_snap_quote_packet_decodes_quote_and_open_interest() {
        let mut fixture = quote_fixture(3, 379);
        write_i64(&mut fixture, 123, 1_756_000_000_000);
        write_i64(&mut fixture, 131, 555_000);
        write_i64(&mut fixture, 139, 250);
        write_i64(&mut fixture, 347, 2_700_000);
        write_i64(&mut fixture, 355, 2_200_000);
        write_i64(&mut fixture, 363, 2_800_000);
        write_i64(&mut fixture, 371, 1_900_000);
        let packet = decode_packet(&fixture, Some(SubscriptionMode::SnapQuote), Some(1)).unwrap();
        assert_eq!(packet.volume, Some(987_654));
        assert_eq!(packet.open_interest, Some(555_000));
        assert_eq!(packet.open, Some(2_440_000));
        assert_eq!(packet.high, Some(2_460_000));
        assert_eq!(packet.low, Some(2_430_000));
        assert_eq!(packet.close, Some(2_445_000));
    }

    #[test]
    fn packet_contract_rejects_truncated_wrong_length_and_mode() {
        assert!(decode_packet(&quote_fixture(2, 123)[..122], None, Some(1)).is_err());
        assert!(decode_packet(&common_fixture(1, 52), None, Some(1)).is_err());
        assert!(decode_packet(&common_fixture(4, 51), None, Some(1)).is_err());
        assert!(decode_packet(
            &common_fixture(1, 51),
            Some(SubscriptionMode::Quote),
            Some(1)
        )
        .is_err());
    }

    #[test]
    fn packet_contract_rejects_malformed_token_fields_and_exchange() {
        let mut malformed = common_fixture(1, 51);
        malformed[2..10].copy_from_slice(b"99A23456");
        assert!(decode_packet(&malformed, None, Some(1)).is_err());
        let mut unsupported = common_fixture(1, 51);
        unsupported[1] = 99;
        assert!(decode_packet(&unsupported, None, None).is_err());
    }

    #[test]
    fn local_bridge_request_carries_only_process_authorization() {
        let request = local_bridge_request("fixture-process-token").unwrap();
        let headers = request.headers();
        assert_eq!(
            headers.get("X-Process-Token").unwrap(),
            "fixture-process-token"
        );
        for forbidden in [
            "Authorization",
            "x-api-key",
            "x-client-code",
            "x-feed-token",
        ] {
            assert!(!headers.contains_key(forbidden));
        }
    }

    #[test]
    fn subscription_uses_only_resolved_validated_instrument() {
        let resolved = instrument(FIXTURE_TOKEN).validate().unwrap();
        let request = subscription_request(&resolved).unwrap();
        assert_eq!(
            request["params"]["tokenList"][0]["tokens"][0],
            FIXTURE_TOKEN
        );
        assert_eq!(request["params"]["tokenList"][0]["exchangeType"], 1);
    }

    #[test]
    fn wrong_or_missing_instrument_identity_fails_before_subscription() {
        assert!(instrument("").validate().is_err());
        let mut wrong = instrument(FIXTURE_TOKEN);
        wrong.instrument_type = String::from("EQ");
        assert!(wrong.validate().is_err());
        let mut wrong_exchange = instrument(FIXTURE_TOKEN);
        wrong_exchange.exchange = String::from("NFO");
        assert!(wrong_exchange.validate().is_err());
    }

    #[tokio::test]
    async fn local_instrument_contract_accepts_only_canonical_index_identity() {
        let good = format!(
            r#"{{"schema_version":1,"token":"{FIXTURE_TOKEN}","symbol":"NIFTY","canonical_symbol":"NIFTY","name":"NIFTY 50","exchange":"NSE","instrument_type":"AMXIDX"}}"#
        );
        let url = instrument_http_fixture(&good).await;
        assert_eq!(
            resolve_feed_instrument_at(&url, "fixture-process-token")
                .await
                .unwrap()
                .token,
            FIXTURE_TOKEN
        );

        let wrong = format!(
            r#"{{"schema_version":1,"token":"{FIXTURE_TOKEN}","symbol":"NIFTY","canonical_symbol":"NIFTY","name":"NIFTY 50","exchange":"NSE","instrument_type":"EQ"}}"#
        );
        let url = instrument_http_fixture(&wrong).await;
        assert!(resolve_feed_instrument_at(&url, "fixture-process-token")
            .await
            .is_err());

        let missing = r#"{"schema_version":1,"token":"","symbol":"NIFTY","canonical_symbol":"NIFTY","name":"NIFTY 50","exchange":"NSE","instrument_type":"AMXIDX"}"#;
        let url = instrument_http_fixture(missing).await;
        assert!(resolve_feed_instrument_at(&url, "fixture-process-token")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn websocket_lifecycle_reaches_live_only_after_matching_valid_tick() {
        let mut payload = quote_fixture(2, 123);
        write_i64(&mut payload, 35, Utc::now().timestamp_millis());
        let (url, server) = websocket_fixture(Some(payload)).await;
        let (state, mut ticks) = feed_state();
        let task_state = state.clone();
        let task = tokio::spawn(async move {
            run_websocket_at(
                &task_state,
                &instrument(FIXTURE_TOKEN),
                "fixture-process-token",
                &url,
                Duration::from_secs(1),
            )
            .await
        });
        let tick = timeout(Duration::from_secs(1), ticks.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tick.token, FIXTURE_TOKEN);
        assert_eq!(state.feed_status().unwrap().phase, FeedPhase::Live);
        task.abort();
        server.abort();
    }

    #[tokio::test]
    async fn malformed_and_wrong_token_frames_never_prove_live() {
        let malformed = quote_fixture(2, 123)[..122].to_vec();
        for payload in [malformed, {
            let mut wrong = quote_fixture(2, 123);
            wrong[2..10].copy_from_slice(b"88123456");
            wrong
        }] {
            let (url, server) = websocket_fixture(Some(payload)).await;
            let (state, _) = feed_state();
            let result = run_websocket_at(
                &state,
                &instrument(FIXTURE_TOKEN),
                "fixture-process-token",
                &url,
                Duration::from_millis(80),
            )
            .await;
            assert!(result.unwrap_err().contains("no valid market data"));
            assert_eq!(state.feed_status().unwrap().phase, FeedPhase::Degraded);
            state
                .set_feed_phase(FeedPhase::Reconnecting, Some(String::from("retry")))
                .unwrap();
            assert_eq!(state.feed_status().unwrap().phase, FeedPhase::Reconnecting);
            server.abort();
        }
    }

    #[tokio::test]
    async fn silent_subscription_times_out_degraded_without_false_ready() {
        let (url, server) = websocket_fixture(None).await;
        let (state, _) = feed_state();
        let result = run_websocket_at(
            &state,
            &instrument(FIXTURE_TOKEN),
            "fixture-process-token",
            &url,
            Duration::from_millis(80),
        )
        .await;
        assert!(result.unwrap_err().contains("no valid market data"));
        assert_eq!(state.feed_status().unwrap().phase, FeedPhase::Degraded);
        server.abort();
    }
}
