use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use redis::Commands;
use serde::Deserialize;
use serde_json::json;
use tokio_tungstenite::tungstenite::handshake::client::Request;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::{
    auth,
    models::{Exchange, Tick},
    state::AppState,
};

const WEBSOCKET_URL: &str = "wss://smartapisocket.angelone.in/smart-stream";
const NIFTY_TOKEN: &str = "26000";
const NIFTY_SYMBOL: &str = "NIFTY";
const REDIS_TTL_SECONDS: u64 = 60;

#[derive(Debug, Deserialize)]
struct AngelTick {
    #[serde(rename = "type")]
    message_type: String,
    token: String,
    exchange_type: u8,
    last_traded_price: i64,
    open_price_of_the_day: i64,
    high_price_of_the_day: i64,
    low_price_of_the_day: i64,
    closed_price: i64,
    total_traded_volume: u64,
    open_interest: u64,
}

pub async fn run(state: AppState) -> Result<(), String> {
    let client_id = match auth::get_client_id() {
        Ok(client_id) => client_id,
        Err(error) => {
            println!("ingester: {error}");
            return Ok(());
        }
    };

    println!("ingester: authenticating...");
    let (jwt_token, feed_token) = match auth::get_jwt_token().await {
        Ok(tokens) => tokens,
        Err(error) => {
            println!("ingester: auth failed: {error}");
            return Ok(());
        }
    };

    let request = websocket_request(&client_id, &jwt_token, &feed_token)?;
    let (mut websocket, _response) = match connect_async(request).await {
        Ok(connection) => connection,
        Err(error) => {
            println!("ingester: websocket connect failed: {error}");
            return Ok(());
        }
    };
    println!("ingester: connected to websocket");

    let subscription = json!({
        "action": 1,
        "params": {
            "mode": 2,
            "tokenList": [
                { "exchangeType": 1, "tokens": [NIFTY_TOKEN] }
            ]
        }
    });

    if let Err(error) = websocket
        .send(Message::Text(subscription.to_string()))
        .await
    {
        println!("ingester: subscribe failed: {error}");
        return Ok(());
    }
    println!("ingester: subscribed to NIFTY");

    while let Some(message) = websocket.next().await {
        match message {
            Ok(Message::Text(text)) => {
                handle_text_message(&state, &text).await;
            }
            Ok(Message::Binary(bytes)) => {
                handle_binary_message(&state, &bytes).await;
            }
            Ok(Message::Close(_)) => {
                println!("websocket: disconnected");
                return Ok(());
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) | Ok(Message::Frame(_)) => {}
            Err(error) => {
                println!("websocket: error: {error}");
                return Ok(());
            }
        }
    }

    println!("websocket: disconnected");
    Ok(())
}

fn websocket_request(
    client_id: &str,
    jwt_token: &str,
    feed_token: &str,
) -> Result<Request, String> {
    use tokio_tungstenite::tungstenite::handshake::client::generate_key;

    Request::builder()
        .uri(WEBSOCKET_URL)
        .header("Host", "smartapisocket.angelone.in")
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", generate_key())
        .header("Authorization", format!("Bearer {jwt_token}"))
        .header("x-client-code", client_id)
        .header("x-feed-token", feed_token)
        .header("x-client-type", "WEB")
        .body(())
        .map_err(|error| error.to_string())
}

async fn handle_text_message(state: &AppState, text: &str) {
    match parse_tick(text) {
        Ok(Some(tick)) => {
            publish_tick(state, &tick).await;
            write_tick_to_redis(state, &tick).await;
        }
        Ok(None) => {}
        Err(error) => println!("ingester: tick parse failed: {error}; raw: {text}"),
    }
}

fn parse_tick(text: &str) -> Result<Option<Tick>, String> {
    let message: AngelTick = serde_json::from_str(text).map_err(|error| error.to_string())?;

    if message.message_type != "ticker_data" {
        return Ok(None);
    }

    if message.token != NIFTY_TOKEN {
        return Ok(None);
    }

    Ok(Some(Tick {
        symbol: NIFTY_SYMBOL.to_string(),
        token: message.token,
        ltp: price_to_rupees(message.last_traded_price),
        open: price_to_rupees(message.open_price_of_the_day),
        high: price_to_rupees(message.high_price_of_the_day),
        low: price_to_rupees(message.low_price_of_the_day),
        close: price_to_rupees(message.closed_price),
        volume: message.total_traded_volume,
        oi: message.open_interest,
        bid: 0.0,
        ask: 0.0,
        timestamp: Utc::now().timestamp_millis(),
        exchange: match message.exchange_type {
            1 => Exchange::NSE,
            _ => Exchange::NSE,
        },
    }))
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
            println!("ingester: tick serialize failed: {error}");
            return;
        }
    };

    let mut redis_guard = state.redis.lock().await;
    let connection = match redis_guard.as_mut() {
        Some(connection) => connection,
        None => {
            println!("ingester: redis unavailable; tick not cached");
            return;
        }
    };

    let key = format!("tick:{}", tick.symbol);
    let result: redis::RedisResult<()> = connection.set_ex(&key, payload, REDIS_TTL_SECONDS);
    if let Err(error) = result {
        println!("ingester: redis write failed: {error}");
    }
}

async fn handle_binary_message(state: &AppState, bytes: &[u8]) {
    if bytes.len() < 50 {
        return;
    }

    let token_bytes = &bytes[2..26];
    let token = String::from_utf8_lossy(token_bytes)
        .trim_matches('\0')
        .trim()
        .to_string();

    if token != NIFTY_TOKEN {
        return;
    }

    if bytes.len() < 100 {
        return;
    }

    let ltp = i64::from_le_bytes(bytes[43..51].try_into().unwrap_or([0; 8]));
    let volume = i64::from_le_bytes(bytes[51..59].try_into().unwrap_or([0; 8])) as u64;
    let open = i64::from_le_bytes(bytes[59..67].try_into().unwrap_or([0; 8]));
    let high = i64::from_le_bytes(bytes[67..75].try_into().unwrap_or([0; 8]));
    let low = i64::from_le_bytes(bytes[75..83].try_into().unwrap_or([0; 8]));
    let close = i64::from_le_bytes(bytes[83..91].try_into().unwrap_or([0; 8]));

    let tick = Tick {
        symbol: NIFTY_SYMBOL.to_string(),
        token,
        ltp: price_to_rupees(ltp),
        open: price_to_rupees(open),
        high: price_to_rupees(high),
        low: price_to_rupees(low),
        close: price_to_rupees(close),
        volume,
        oi: 0,
        bid: 0.0,
        ask: 0.0,
        timestamp: Utc::now().timestamp_millis(),
        exchange: Exchange::NSE,
    };

    publish_tick(state, &tick).await;
    write_tick_to_redis(state, &tick).await;
}
