use std::{
    collections::{BTreeSet, HashMap},
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};

use chrono::Utc;
use redis::Commands;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tauri::Emitter;
use tauri_plugin_notification::NotificationExt;
use tokio::time;

use crate::{
    exchange_calendar::nearest_weekly_expiry_label,
    models::{Alert, AlertCondition, AlertType, OptionLeg, OptionType, Tick},
    state::AppState,
};

const ALERTS_SCHEMA: &str = r#"
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    threshold REAL NOT NULL,
    condition TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    triggered_at INTEGER,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS alert_observations (
    alert_id TEXT PRIMARY KEY,
    raw_value REAL NOT NULL,
    condition_value REAL,
    observed_at INTEGER NOT NULL,
    FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS alert_outbox (
    event_id TEXT PRIMARY KEY,
    alert_id TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    delivery_state TEXT NOT NULL DEFAULT 'PENDING',
    event_delivered_at INTEGER,
    notification_delivered_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_at INTEGER,
    provider_reference TEXT,
    last_error TEXT,
    FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
);
"#;

#[derive(Debug, Default, Clone)]
struct MarketSnapshot {
    prices: HashMap<String, f64>,
    open_interest: HashMap<String, f64>,
    pcr: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct PendingDelivery {
    event_id: String,
    payload: Value,
    event_delivered: bool,
    notification_delivered: bool,
}

trait DeliverySink {
    fn emit_event(&mut self, delivery: &PendingDelivery) -> Result<(), String>;
    fn show_notification(&mut self, delivery: &PendingDelivery) -> Result<(), String>;
}

struct TauriDeliverySink<'a> {
    state: &'a AppState,
}

impl DeliverySink for TauriDeliverySink<'_> {
    fn emit_event(&mut self, delivery: &PendingDelivery) -> Result<(), String> {
        let app_handle = self
            .state
            .app_handle()
            .ok_or_else(|| String::from("app handle unavailable"))?;
        app_handle
            .emit("alert_triggered", delivery.payload.clone())
            .map_err(|error| format!("event emit failed: {error}"))
    }

    fn show_notification(&mut self, delivery: &PendingDelivery) -> Result<(), String> {
        let app_handle = self
            .state
            .app_handle()
            .ok_or_else(|| String::from("app handle unavailable"))?;
        let symbol = delivery
            .payload
            .get("symbol")
            .and_then(Value::as_str)
            .unwrap_or("Market");
        let alert_type = delivery
            .payload
            .get("alert_type")
            .and_then(Value::as_str)
            .unwrap_or("value");
        let value = delivery
            .payload
            .get("value")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        let threshold = delivery
            .payload
            .get("threshold")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        app_handle
            .notification()
            .builder()
            .title("Indian Market Terminal")
            .body(format!(
                "{symbol} {alert_type} alert: {value:.2} crossed {threshold:.2}"
            ))
            .show()
            .map_err(|error| format!("desktop notification failed: {error}"))
    }
}

pub async fn run(state: AppState) -> Result<(), String> {
    let db_path = config_db_path()?;
    init_db_at(&db_path)?;
    let mut interval = time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;
        if state.is_shutdown_requested() {
            return Ok(());
        }
        if let Err(error) = evaluate_alerts(&state, &db_path).await {
            crate::bt_error!("alert", "evaluation failed: {error}");
        }
        let mut sink = TauriDeliverySink { state: &state };
        if let Err(error) = deliver_pending_at(&db_path, &mut sink) {
            crate::bt_error!("alert", "delivery retry failed: {error}");
        }
    }
}

pub fn config_db_path() -> Result<PathBuf, String> {
    if let Some(base) = env::var_os("BHARATTERMINAL_APP_DATA") {
        return Ok(PathBuf::from(base).join("data").join("config.db"));
    }

    #[cfg(debug_assertions)]
    {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| String::from("development workspace root unavailable"))?;
        Ok(workspace.join("data").join("config.db"))
    }

    #[cfg(not(debug_assertions))]
    {
        let appdata = env::var_os("APPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| String::from("APPDATA unavailable"))?;
        Ok(appdata
            .join("BharatTerminal")
            .join("data")
            .join("config.db"))
    }
}

pub fn init_db() -> Result<(), String> {
    init_db_at(&config_db_path()?)
}

fn init_db_at(path: &Path) -> Result<(), String> {
    drop(connect_and_init(path)?);
    Ok(())
}

pub fn load_active_alerts() -> Result<Vec<Alert>, String> {
    load_active_alerts_at(&config_db_path()?)
}

fn load_active_alerts_at(path: &Path) -> Result<Vec<Alert>, String> {
    let connection = connect_and_init(path)?;
    load_alerts(&connection)
}

fn load_alerts(connection: &Connection) -> Result<Vec<Alert>, String> {
    let mut statement = connection
        .prepare(
            "SELECT id, symbol, alert_type, threshold, condition, active, triggered_at, created_at
             FROM alerts ORDER BY id ASC",
        )
        .map_err(db_error)?;
    let mut rows = statement.query([]).map_err(db_error)?;
    let mut alerts = Vec::new();
    while let Some(row) = rows.next().map_err(db_error)? {
        alerts.push(Alert {
            id: row.get(0).map_err(db_error)?,
            symbol: row.get(1).map_err(db_error)?,
            alert_type: parse_alert_type(&row.get::<_, String>(2).map_err(db_error)?)?,
            threshold: row.get(3).map_err(db_error)?,
            condition: parse_alert_condition(&row.get::<_, String>(4).map_err(db_error)?)?,
            active: row.get::<_, i64>(5).map_err(db_error)? != 0,
            triggered_at: row.get(6).map_err(db_error)?,
            created_at: row.get(7).map_err(db_error)?,
        });
    }
    Ok(alerts)
}

pub fn save_alert(alert: &Alert) -> Result<(), String> {
    save_alert_at(&config_db_path()?, alert)
}

fn save_alert_at(path: &Path, alert: &Alert) -> Result<(), String> {
    let connection = connect_and_init(path)?;
    connection
        .execute(
            "INSERT INTO alerts
             (id, symbol, alert_type, threshold, condition, active, triggered_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                alert.id,
                alert.symbol,
                alert_type_to_db(&alert.alert_type),
                alert.threshold,
                alert_condition_to_db(&alert.condition),
                if alert.active { 1_i64 } else { 0_i64 },
                alert.triggered_at,
                alert.created_at,
            ],
        )
        .map_err(db_error)?;
    Ok(())
}

pub fn delete_alert(id: &str) -> Result<(), String> {
    let connection = connect_and_init(&config_db_path()?)?;
    connection
        .execute("DELETE FROM alerts WHERE id = ?1", params![id])
        .map_err(db_error)?;
    Ok(())
}

pub fn parse_alert_type(value: &str) -> Result<AlertType, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "price" => Ok(AlertType::Price),
        "oi" => Ok(AlertType::OI),
        "pcr" => Ok(AlertType::PCR),
        other => Err(format!("unknown alert_type: {other}")),
    }
}

pub fn parse_alert_condition(value: &str) -> Result<AlertCondition, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "above" => Ok(AlertCondition::Above),
        "below" => Ok(AlertCondition::Below),
        "crosses_up" => Ok(AlertCondition::CrossesUp),
        "crosses_down" => Ok(AlertCondition::CrossesDown),
        other => Err(format!("unknown alert condition: {other}")),
    }
}

async fn evaluate_alerts(state: &AppState, path: &Path) -> Result<(), String> {
    let alerts = load_active_alerts_at(path)?
        .into_iter()
        .filter(|alert| alert.active)
        .collect::<Vec<_>>();
    if alerts.is_empty() {
        return Ok(());
    }
    let snapshot = capture_snapshot(state, &alerts).await;
    evaluate_snapshot_at(path, &alerts, &snapshot, Utc::now().timestamp_millis())?;
    Ok(())
}

async fn capture_snapshot(state: &AppState, alerts: &[Alert]) -> MarketSnapshot {
    let expiry = match nearest_weekly_expiry_label(Utc::now()) {
        Ok(expiry) => expiry,
        Err(error) => {
            crate::bt_error!("alert", "expiry unavailable: {error}");
            return MarketSnapshot::default();
        }
    };
    let mut price_symbols = BTreeSet::new();
    let mut oi_symbols = BTreeSet::new();
    let mut pcr_symbols = BTreeSet::new();
    for alert in alerts {
        match alert.alert_type {
            AlertType::Price => {
                price_symbols.insert(alert.symbol.clone());
            }
            AlertType::OI => {
                oi_symbols.insert(alert.symbol.clone());
            }
            AlertType::PCR => {
                pcr_symbols.insert(alert.symbol.clone());
            }
        }
    }

    let mut snapshot = MarketSnapshot::default();
    for symbol in price_symbols {
        match read_tick(state, &symbol).await {
            Ok(tick) => {
                snapshot.prices.insert(symbol, tick.ltp);
            }
            Err(error) => crate::bt_error!("alert", "price snapshot skipped: {error}"),
        }
    }
    for symbol in oi_symbols {
        match read_chain(state, &symbol, &expiry).await {
            Ok(legs) => {
                if let Some(total) = total_open_interest(&legs) {
                    snapshot.open_interest.insert(symbol, total as f64);
                } else {
                    crate::bt_error!("alert", "OI snapshot skipped: chain OI incomplete");
                }
            }
            Err(error) => crate::bt_error!("alert", "OI snapshot skipped: {error}"),
        }
    }
    for symbol in pcr_symbols {
        match read_pcr(state, &symbol, &expiry).await {
            Ok(pcr) => {
                snapshot.pcr.insert(symbol, pcr);
            }
            Err(error) => crate::bt_error!("alert", "PCR snapshot skipped: {error}"),
        }
    }
    snapshot
}

fn evaluate_snapshot_at(
    path: &Path,
    alerts: &[Alert],
    snapshot: &MarketSnapshot,
    timestamp: i64,
) -> Result<Vec<String>, String> {
    let mut connection = connect_and_init(path)?;
    let transaction = connection.transaction().map_err(db_error)?;
    let mut ordered = alerts
        .iter()
        .filter(|alert| alert.active)
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.id.cmp(&right.id));
    let mut triggered = Vec::new();

    for alert in ordered {
        let raw_value = match alert.alert_type {
            AlertType::Price => snapshot.prices.get(&alert.symbol).copied(),
            AlertType::OI => snapshot.open_interest.get(&alert.symbol).copied(),
            AlertType::PCR => snapshot.pcr.get(&alert.symbol).copied(),
        };
        let Some(raw_value) = raw_value.filter(|value| value.is_finite()) else {
            continue;
        };
        let previous = load_observation(&transaction, &alert.id)?;
        let condition_value = match alert.alert_type {
            AlertType::OI => previous.map(|(raw, _)| (raw_value - raw).abs()),
            AlertType::Price | AlertType::PCR => Some(raw_value),
        };
        let previous_condition = previous.and_then(|(_, value)| value);

        upsert_observation(
            &transaction,
            &alert.id,
            raw_value,
            condition_value,
            timestamp,
        )?;

        let Some(value) = condition_value else {
            continue;
        };
        if !threshold_matches(&alert.condition, previous_condition, value, alert.threshold) {
            continue;
        }

        let event_id = format!("{}:{timestamp}", alert.id);
        let mut payload = json!({
            "event_id": event_id,
            "idempotency_key": event_id,
            "delivery_semantics": "AT_LEAST_ONCE_WITH_STABLE_EVENT_ID",
            "delivery_state": "PENDING",
            "attempt_count": 0,
            "last_attempt_at": null,
            "provider_reference": null,
            "alert_id": alert.id,
            "symbol": alert.symbol,
            "alert_type": alert_type_to_db(&alert.alert_type),
            "condition": alert_condition_to_db(&alert.condition),
            "value": value,
            "threshold": alert.threshold,
            "timestamp": timestamp,
        });
        let payload_hash = payload_hash_for_value(&payload)?;
        payload
            .as_object_mut()
            .ok_or_else(|| String::from("alert event payload is not an object"))?
            .insert("payload_hash".into(), Value::String(payload_hash.clone()));
        transaction
            .execute(
                "INSERT OR IGNORE INTO alert_outbox
                 (event_id, alert_id, payload_json, payload_hash, created_at, delivery_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'PENDING')",
                params![
                    event_id,
                    alert.id,
                    payload.to_string(),
                    payload_hash,
                    timestamp
                ],
            )
            .map_err(db_error)?;
        transaction
            .execute(
                "UPDATE alerts SET triggered_at = ?1, active = 0
                 WHERE id = ?2 AND active = 1",
                params![timestamp, alert.id],
            )
            .map_err(db_error)?;
        triggered.push(alert.id.clone());
    }

    transaction.commit().map_err(db_error)?;
    Ok(triggered)
}

fn load_observation(
    transaction: &Transaction<'_>,
    alert_id: &str,
) -> Result<Option<(f64, Option<f64>)>, String> {
    transaction
        .query_row(
            "SELECT raw_value, condition_value FROM alert_observations WHERE alert_id = ?1",
            params![alert_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(db_error)
}

fn upsert_observation(
    transaction: &Transaction<'_>,
    alert_id: &str,
    raw_value: f64,
    condition_value: Option<f64>,
    timestamp: i64,
) -> Result<(), String> {
    transaction
        .execute(
            "INSERT INTO alert_observations
             (alert_id, raw_value, condition_value, observed_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(alert_id) DO UPDATE SET
                 raw_value = excluded.raw_value,
                 condition_value = excluded.condition_value,
                 observed_at = excluded.observed_at",
            params![alert_id, raw_value, condition_value, timestamp],
        )
        .map_err(db_error)?;
    Ok(())
}

fn deliver_pending_at(path: &Path, sink: &mut impl DeliverySink) -> Result<(), String> {
    let pending = load_pending_deliveries(path)?;
    for delivery in pending {
        if !delivery.event_delivered {
            match sink.emit_event(&delivery) {
                Ok(()) => mark_delivery(path, &delivery.event_id, "event_delivered_at")?,
                Err(error) => {
                    record_delivery_failure(path, &delivery.event_id, &error)?;
                    continue;
                }
            }
        }
        if !delivery.notification_delivered {
            match sink.show_notification(&delivery) {
                Ok(()) => mark_delivery(path, &delivery.event_id, "notification_delivered_at")?,
                Err(error) => record_delivery_failure(path, &delivery.event_id, &error)?,
            }
        }
    }
    Ok(())
}

fn load_pending_deliveries(path: &Path) -> Result<Vec<PendingDelivery>, String> {
    let connection = connect_and_init(path)?;
    let mut statement = connection
        .prepare(
            "SELECT event_id, payload_json, payload_hash,
                    event_delivered_at, notification_delivered_at
             FROM alert_outbox
             WHERE event_delivered_at IS NULL OR notification_delivered_at IS NULL
             ORDER BY created_at ASC, event_id ASC",
        )
        .map_err(db_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<i64>>(4)?,
            ))
        })
        .map_err(db_error)?;
    let mut deliveries = Vec::new();
    for row in rows {
        let (event_id, payload_json, payload_hash, event_at, notification_at) =
            row.map_err(db_error)?;
        deliveries.push(PendingDelivery {
            event_id,
            payload: verified_payload(&payload_json, &payload_hash)?,
            event_delivered: event_at.is_some(),
            notification_delivered: notification_at.is_some(),
        });
    }
    Ok(deliveries)
}

fn mark_delivery(path: &Path, event_id: &str, column: &str) -> Result<(), String> {
    let (column, provider) = match column {
        "event_delivered_at" => ("event_delivered_at", "tauri-event"),
        "notification_delivered_at" => ("notification_delivered_at", "desktop-notification"),
        _ => return Err(String::from("invalid delivery column")),
    };
    let connection = connect_and_init(path)?;
    let now = Utc::now().timestamp_millis();
    let provider_reference = format!("{provider}:{event_id}");
    connection
        .execute(
            &format!(
                "UPDATE alert_outbox SET {column} = ?1,
                    attempts = attempts + 1,
                    attempt_count = attempt_count + 1,
                    last_attempt_at = ?1,
                    provider_reference = ?2,
                    last_error = NULL
                 WHERE event_id = ?3 AND {column} IS NULL"
            ),
            params![now, provider_reference, event_id],
        )
        .map_err(db_error)?;
    connection
        .execute(
            "UPDATE alert_outbox SET delivery_state = CASE
                WHEN event_delivered_at IS NOT NULL AND notification_delivered_at IS NOT NULL
                    THEN 'DELIVERED'
                ELSE 'PARTIAL'
             END WHERE event_id = ?1",
            params![event_id],
        )
        .map_err(db_error)?;
    Ok(())
}

fn record_delivery_failure(path: &Path, event_id: &str, error: &str) -> Result<(), String> {
    let connection = connect_and_init(path)?;
    let now = Utc::now().timestamp_millis();
    connection
        .execute(
            "UPDATE alert_outbox
             SET attempts = attempts + 1,
                 attempt_count = attempt_count + 1,
                 last_attempt_at = ?1,
                 delivery_state = 'RETRY',
                 last_error = ?2
             WHERE event_id = ?3",
            params![now, error, event_id],
        )
        .map_err(db_error)?;
    Ok(())
}

async fn read_tick(state: &AppState, symbol: &str) -> Result<Tick, String> {
    let key = format!("tick:{symbol}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;
    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;
    let tick: Tick = serde_json::from_str(&payload.ok_or_else(|| format!("{key} not cached"))?)
        .map_err(|error| format!("{key} JSON parse failed: {error}"))?;
    validate_alert_tick(&tick, symbol, Utc::now().timestamp_millis())?;
    Ok(tick)
}

fn validate_alert_tick(tick: &Tick, expected_symbol: &str, now_ms: i64) -> Result<(), String> {
    if tick.schema_version != Tick::SCHEMA_VERSION
        || tick.symbol != expected_symbol
        || tick.token.trim().is_empty()
        || !matches!(tick.exchange, crate::models::Exchange::NSE)
        || tick.source_type != crate::models::QuoteSource::AngelStream
        || tick.freshness != crate::models::QuoteFreshness::Fresh
        || tick.quality != crate::models::QuoteQuality::ExecutableStream
        || tick.fallback
        || tick.market_state != crate::models::MarketDataState::Live
        || !tick.ltp.is_finite()
        || tick.ltp <= 0.0
        || tick.observed_at_ms <= 0
        || tick.received_at_ms < tick.observed_at_ms
        || tick.observed_at_ms > now_ms.saturating_add(1_000)
        || now_ms.saturating_sub(tick.observed_at_ms) > Tick::STREAM_FRESH_MS
    {
        return Err(String::from("quote rejected by alert provenance policy"));
    }
    Ok(())
}

async fn read_chain(
    state: &AppState,
    symbol: &str,
    expiry: &str,
) -> Result<Vec<OptionLeg>, String> {
    let key = format!("chain:{symbol}:{expiry}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;
    validate_chain_health(connection, expiry, Utc::now().timestamp_millis())?;
    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;
    let legs: Vec<OptionLeg> =
        serde_json::from_str(&payload.ok_or_else(|| format!("{key} not cached"))?)
            .map_err(|error| format!("{key} JSON parse failed: {error}"))?;
    let now_ms = Utc::now().timestamp_millis();
    if legs.is_empty()
        || legs.iter().any(|leg| {
            leg.schema_version != Tick::SCHEMA_VERSION
                || leg.symbol != symbol
                || leg.expiry != expiry
                || leg.source_type != crate::models::QuoteSource::NseChain
                || leg.freshness != crate::models::QuoteFreshness::Fresh
                || leg.quality != crate::models::QuoteQuality::ReferenceOnly
                || leg.fallback
                || leg.observed_at_ms <= 0
                || now_ms.saturating_sub(leg.observed_at_ms) > 15_000
        })
    {
        return Err(String::from(
            "option chain rejected by alert provenance policy",
        ));
    }
    Ok(legs)
}

async fn read_pcr(state: &AppState, symbol: &str, expiry: &str) -> Result<f64, String> {
    let key = format!("pcr:{symbol}:{expiry}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;
    validate_chain_health(connection, expiry, Utc::now().timestamp_millis())?;
    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;
    let value = payload
        .ok_or_else(|| format!("{key} not cached"))?
        .parse::<f64>()
        .map_err(|error| format!("{key} parse failed: {error}"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(String::from("PCR rejected by alert quality policy"));
    }
    Ok(value)
}

fn validate_chain_health(
    connection: &mut redis::Connection,
    expected_expiry: &str,
    now_ms: i64,
) -> Result<(), String> {
    let raw: Option<String> = connection
        .get("health:nse_chain_poller")
        .map_err(|error| format!("chain health unavailable: {error}"))?;
    let value: Value =
        serde_json::from_str(&raw.ok_or_else(|| String::from("chain health missing"))?)
            .map_err(|_| String::from("chain health invalid"))?;
    let observed = value
        .get("last_success_ms")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if value.get("schema_version").and_then(Value::as_u64) != Some(1)
        || value.get("state").and_then(Value::as_str) != Some("LIVE")
        || value.get("source").and_then(Value::as_str) != Some("RUST_NSE_CHAIN_POLLER")
        || value.get("expiry").and_then(Value::as_str) != Some(expected_expiry)
        || observed <= 0
        || observed > now_ms.saturating_add(1_000)
        || now_ms.saturating_sub(observed) > 15_000
    {
        return Err(String::from(
            "chain health rejected by alert provenance policy",
        ));
    }
    Ok(())
}

fn threshold_matches(
    condition: &AlertCondition,
    previous: Option<f64>,
    current: f64,
    threshold: f64,
) -> bool {
    match condition {
        AlertCondition::Above => current >= threshold,
        AlertCondition::Below => current <= threshold,
        AlertCondition::CrossesUp => previous
            .map(|previous| previous < threshold && current >= threshold)
            .unwrap_or(false),
        AlertCondition::CrossesDown => previous
            .map(|previous| previous > threshold && current <= threshold)
            .unwrap_or(false),
    }
}

fn total_open_interest(legs: &[OptionLeg]) -> Option<u64> {
    legs.iter().try_fold(0_u64, |total, leg| {
        let oi = leg.oi?;
        match leg.option_type {
            OptionType::CE | OptionType::PE => Some(total.saturating_add(oi)),
        }
    })
}

fn payload_hash_for_value(value: &Value) -> Result<String, String> {
    let mut canonical = value.clone();
    canonical
        .as_object_mut()
        .ok_or_else(|| String::from("alert event payload is not an object"))?
        .remove("payload_hash");
    let encoded = serde_json::to_vec(&canonical)
        .map_err(|error| format!("alert event payload serialization failed: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn verified_payload(payload_json: &str, stored_hash: &str) -> Result<Value, String> {
    let payload: Value = serde_json::from_str(payload_json)
        .map_err(|error| format!("alert outbox payload invalid: {error}"))?;
    let embedded_hash = payload
        .get("payload_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| String::from("alert outbox payload hash missing"))?;
    let actual_hash = payload_hash_for_value(&payload)?;
    if embedded_hash != stored_hash || embedded_hash != actual_hash {
        return Err(String::from("alert outbox payload hash mismatch"));
    }
    Ok(payload)
}

fn connect_and_init(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("alerts DB dir create failed: {error}"))?;
    }
    let connection = Connection::open(path)
        .map_err(|error| format!("alerts DB open {} failed: {error}", path.display()))?;
    connection.execute_batch(ALERTS_SCHEMA).map_err(db_error)?;
    migrate_alert_outbox(&connection)?;
    Ok(connection)
}

fn migrate_alert_outbox(connection: &Connection) -> Result<(), String> {
    let columns = connection
        .prepare("PRAGMA table_info(alert_outbox)")
        .map_err(db_error)?
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(db_error)?
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(db_error)?;
    for (name, definition) in [
        ("payload_hash", "TEXT NOT NULL DEFAULT ''"),
        ("delivery_state", "TEXT NOT NULL DEFAULT 'PENDING'"),
        ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
        ("last_attempt_at", "INTEGER"),
        ("provider_reference", "TEXT"),
    ] {
        if !columns.contains(name) {
            connection
                .execute(
                    &format!("ALTER TABLE alert_outbox ADD COLUMN {name} {definition}"),
                    [],
                )
                .map_err(db_error)?;
        }
    }
    connection
        .execute(
            "UPDATE alert_outbox SET attempt_count = attempts
             WHERE attempt_count = 0 AND attempts > 0",
            [],
        )
        .map_err(db_error)?;

    let legacy_rows = {
        let mut statement = connection
            .prepare(
                "SELECT event_id, payload_json, delivery_state, attempt_count,
                        last_attempt_at, provider_reference
                 FROM alert_outbox WHERE payload_hash = ''",
            )
            .map_err(db_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            })
            .map_err(db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(db_error)?;
        rows
    };
    for (event_id, payload_json, state, attempts, last_attempt, provider) in legacy_rows {
        let mut payload: Value = serde_json::from_str(&payload_json)
            .map_err(|error| format!("legacy alert outbox payload invalid: {error}"))?;
        let object = payload
            .as_object_mut()
            .ok_or_else(|| String::from("legacy alert outbox payload is not an object"))?;
        object.insert(
            "delivery_semantics".into(),
            Value::String(String::from("AT_LEAST_ONCE_WITH_STABLE_EVENT_ID")),
        );
        object.insert("delivery_state".into(), Value::String(state));
        object.insert("attempt_count".into(), Value::from(attempts));
        object.insert("last_attempt_at".into(), Value::from(last_attempt));
        object.insert("provider_reference".into(), Value::from(provider));
        let digest = payload_hash_for_value(&payload)?;
        payload
            .as_object_mut()
            .expect("validated object")
            .insert("payload_hash".into(), Value::String(digest.clone()));
        connection
            .execute(
                "UPDATE alert_outbox SET payload_json = ?1, payload_hash = ?2 WHERE event_id = ?3",
                params![payload.to_string(), digest, event_id],
            )
            .map_err(db_error)?;
    }
    Ok(())
}

fn db_error(error: rusqlite::Error) -> String {
    format!("alerts DB error: {error}")
}

fn alert_type_to_db(alert_type: &AlertType) -> &'static str {
    match alert_type {
        AlertType::Price => "price",
        AlertType::OI => "oi",
        AlertType::PCR => "pcr",
    }
}

fn alert_condition_to_db(condition: &AlertCondition) -> &'static str {
    match condition {
        AlertCondition::Above => "above",
        AlertCondition::Below => "below",
        AlertCondition::CrossesUp => "crosses_up",
        AlertCondition::CrossesDown => "crosses_down",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(1);

    fn test_db(name: &str) -> (PathBuf, PathBuf) {
        let root = env::temp_dir().join(format!(
            "indian-terminal-alert-{name}-{}-{}",
            std::process::id(),
            TEST_ID.fetch_add(1, Ordering::Relaxed)
        ));
        (root.join("config.db"), root)
    }

    fn alert(id: &str, alert_type: AlertType, condition: AlertCondition, threshold: f64) -> Alert {
        Alert {
            id: id.into(),
            symbol: "NIFTY".into(),
            alert_type,
            threshold,
            condition,
            active: true,
            triggered_at: None,
            created_at: 1,
        }
    }

    fn price_snapshot(value: f64) -> MarketSnapshot {
        MarketSnapshot {
            prices: HashMap::from([("NIFTY".into(), value)]),
            ..MarketSnapshot::default()
        }
    }

    fn trusted_tick(now_ms: i64) -> Tick {
        Tick {
            schema_version: Tick::SCHEMA_VERSION,
            symbol: "NIFTY".into(),
            token: "99123456".into(),
            ltp: 24_000.0,
            open: None,
            high: None,
            low: None,
            close: None,
            volume: None,
            oi: None,
            bid: None,
            ask: None,
            exchange: crate::models::Exchange::NSE,
            source_type: crate::models::QuoteSource::AngelStream,
            observed_at_ms: now_ms,
            received_at_ms: now_ms,
            sequence: Some(1),
            field_completeness: vec!["ltp".into()],
            freshness: crate::models::QuoteFreshness::Fresh,
            quality: crate::models::QuoteQuality::ExecutableStream,
            fallback: false,
            fallback_reason: None,
            market_state: crate::models::MarketDataState::Live,
        }
    }

    #[test]
    fn alert_quote_policy_rejects_stale_untrusted_and_wrong_instrument() {
        let now = 1_800_000_000_000;
        assert!(validate_alert_tick(&trusted_tick(now), "NIFTY", now).is_ok());
        let mut stale = trusted_tick(now - Tick::STREAM_FRESH_MS - 1);
        stale.freshness = crate::models::QuoteFreshness::Stale;
        assert!(validate_alert_tick(&stale, "NIFTY", now).is_err());
        let mut fallback = trusted_tick(now);
        fallback.fallback = true;
        assert!(validate_alert_tick(&fallback, "NIFTY", now).is_err());
        let mut disconnected = trusted_tick(now);
        disconnected.market_state = crate::models::MarketDataState::Disconnected;
        assert!(validate_alert_tick(&disconnected, "NIFTY", now).is_err());
        let mut wrong_source = trusted_tick(now);
        wrong_source.source_type = crate::models::QuoteSource::NseChain;
        assert!(validate_alert_tick(&wrong_source, "NIFTY", now).is_err());
        assert!(validate_alert_tick(&trusted_tick(now), "BANKNIFTY", now).is_err());
    }

    #[test]
    fn condition_semantics_are_explicit() {
        assert!(threshold_matches(&AlertCondition::Above, None, 10.0, 10.0));
        assert!(threshold_matches(&AlertCondition::Below, None, 10.0, 10.0));
        assert!(threshold_matches(
            &AlertCondition::CrossesUp,
            Some(9.0),
            10.0,
            10.0
        ));
        assert!(!threshold_matches(
            &AlertCondition::CrossesUp,
            Some(10.0),
            11.0,
            10.0
        ));
        assert!(threshold_matches(
            &AlertCondition::CrossesDown,
            Some(11.0),
            10.0,
            10.0
        ));
        assert!(!threshold_matches(
            &AlertCondition::CrossesDown,
            None,
            9.0,
            10.0
        ));
    }

    #[test]
    fn evaluation_is_order_independent() {
        let alerts = vec![
            alert("b", AlertType::Price, AlertCondition::Above, 100.0),
            alert("a", AlertType::Price, AlertCondition::Below, 200.0),
        ];
        let reversed = alerts.iter().rev().cloned().collect::<Vec<_>>();
        let (left, left_root) = test_db("order-left");
        let (right, right_root) = test_db("order-right");
        for item in &alerts {
            save_alert_at(&left, item).unwrap();
            save_alert_at(&right, item).unwrap();
        }
        let left_result = evaluate_snapshot_at(&left, &alerts, &price_snapshot(150.0), 10).unwrap();
        let right_result =
            evaluate_snapshot_at(&right, &reversed, &price_snapshot(150.0), 10).unwrap();
        assert_eq!(left_result, vec!["a", "b"]);
        assert_eq!(left_result, right_result);
        fs::remove_dir_all(left_root).unwrap();
        fs::remove_dir_all(right_root).unwrap();
    }

    #[test]
    fn oi_baseline_is_per_alert_and_survives_restart() {
        let (path, root) = test_db("oi");
        let alerts = vec![
            alert("oi-a", AlertType::OI, AlertCondition::Above, 40.0),
            alert("oi-b", AlertType::OI, AlertCondition::Below, 60.0),
        ];
        for item in &alerts {
            save_alert_at(&path, item).unwrap();
        }
        let first = MarketSnapshot {
            open_interest: HashMap::from([("NIFTY".into(), 1_000.0)]),
            ..MarketSnapshot::default()
        };
        assert!(evaluate_snapshot_at(&path, &alerts, &first, 10)
            .unwrap()
            .is_empty());
        let restarted_alerts = load_active_alerts_at(&path).unwrap();
        let second = MarketSnapshot {
            open_interest: HashMap::from([("NIFTY".into(), 1_050.0)]),
            ..MarketSnapshot::default()
        };
        assert_eq!(
            evaluate_snapshot_at(&path, &restarted_alerts, &second, 20).unwrap(),
            vec!["oi-a", "oi-b"]
        );
        let connection = connect_and_init(&path).unwrap();
        let observations: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM alert_observations WHERE raw_value = 1050 AND condition_value = 50",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(observations, 2);
        drop(connection);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn crossing_state_survives_restart_without_duplicate_trigger() {
        let (path, root) = test_db("restart");
        let item = alert("cross", AlertType::Price, AlertCondition::CrossesUp, 100.0);
        save_alert_at(&path, &item).unwrap();
        assert!(evaluate_snapshot_at(
            &path,
            std::slice::from_ref(&item),
            &price_snapshot(90.0),
            10
        )
        .unwrap()
        .is_empty());
        let restarted = load_active_alerts_at(&path).unwrap();
        assert_eq!(
            evaluate_snapshot_at(&path, &restarted, &price_snapshot(110.0), 20).unwrap(),
            vec!["cross"]
        );
        assert!(
            evaluate_snapshot_at(&path, &restarted, &price_snapshot(120.0), 30)
                .unwrap()
                .is_empty()
        );
        let connection = connect_and_init(&path).unwrap();
        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM alert_outbox", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
        drop(connection);
        fs::remove_dir_all(root).unwrap();
    }

    #[derive(Default)]
    struct TestSink {
        events: usize,
        notifications: usize,
        fail_notification_once: bool,
        fail_event_after_effect_once: bool,
        event_ids: Vec<String>,
    }

    impl DeliverySink for TestSink {
        fn emit_event(&mut self, delivery: &PendingDelivery) -> Result<(), String> {
            self.events += 1;
            self.event_ids.push(delivery.event_id.clone());
            if self.fail_event_after_effect_once {
                self.fail_event_after_effect_once = false;
                return Err("simulated crash after event effect".into());
            }
            Ok(())
        }

        fn show_notification(&mut self, _delivery: &PendingDelivery) -> Result<(), String> {
            self.notifications += 1;
            if self.fail_notification_once {
                self.fail_notification_once = false;
                return Err("simulated notification failure".into());
            }
            Ok(())
        }
    }

    #[test]
    fn persistence_precedes_delivery_and_delivery_is_idempotent() {
        let (path, root) = test_db("delivery");
        let item = alert("delivery", AlertType::Price, AlertCondition::Above, 100.0);
        save_alert_at(&path, &item).unwrap();
        evaluate_snapshot_at(&path, &[item], &price_snapshot(110.0), 10).unwrap();
        let connection = connect_and_init(&path).unwrap();
        let (payload_json, payload_hash, state, attempts): (String, String, String, i64) =
            connection
                .query_row(
                    "SELECT payload_json, payload_hash, delivery_state, attempt_count
                     FROM alert_outbox",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .unwrap();
        let payload = verified_payload(&payload_json, &payload_hash).unwrap();
        assert_eq!(
            payload["delivery_semantics"],
            "AT_LEAST_ONCE_WITH_STABLE_EVENT_ID"
        );
        assert_eq!(payload["delivery_state"], "PENDING");
        assert_eq!(payload["attempt_count"], 0);
        assert!(payload.get("last_attempt_at").is_some());
        assert!(payload.get("provider_reference").is_some());
        assert_eq!(state, "PENDING");
        assert_eq!(attempts, 0);
        drop(connection);
        let loaded = load_active_alerts_at(&path).unwrap();
        assert!(!loaded[0].active);
        let mut sink = TestSink::default();
        deliver_pending_at(&path, &mut sink).unwrap();
        deliver_pending_at(&path, &mut sink).unwrap();
        assert_eq!((sink.events, sink.notifications), (1, 1));
        let connection = connect_and_init(&path).unwrap();
        let (state, attempts, last_attempt, provider): (String, i64, Option<i64>, Option<String>) =
            connection
                .query_row(
                    "SELECT delivery_state, attempt_count, last_attempt_at, provider_reference
                     FROM alert_outbox",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .unwrap();
        assert_eq!(state, "DELIVERED");
        assert_eq!(attempts, 2);
        assert!(last_attempt.is_some());
        assert!(provider.unwrap().starts_with("desktop-notification:"));
        drop(connection);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_notification_retries_without_reemitting_event() {
        let (path, root) = test_db("failed-notification");
        let item = alert("failed", AlertType::Price, AlertCondition::Above, 100.0);
        save_alert_at(&path, &item).unwrap();
        evaluate_snapshot_at(&path, &[item], &price_snapshot(110.0), 10).unwrap();
        let mut sink = TestSink {
            fail_notification_once: true,
            ..TestSink::default()
        };
        deliver_pending_at(&path, &mut sink).unwrap();
        deliver_pending_at(&path, &mut sink).unwrap();
        assert_eq!((sink.events, sink.notifications), (1, 2));
        assert!(load_pending_deliveries(&path).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn crash_window_retries_with_same_consumer_idempotency_key() {
        let (path, root) = test_db("event-crash-window");
        let item = alert(
            "crash-window",
            AlertType::Price,
            AlertCondition::Above,
            100.0,
        );
        save_alert_at(&path, &item).unwrap();
        evaluate_snapshot_at(&path, &[item], &price_snapshot(110.0), 10).unwrap();
        let mut sink = TestSink {
            fail_event_after_effect_once: true,
            ..TestSink::default()
        };
        deliver_pending_at(&path, &mut sink).unwrap();
        deliver_pending_at(&path, &mut sink).unwrap();
        assert_eq!(sink.events, 2);
        assert_eq!(sink.event_ids.len(), 2);
        assert_eq!(sink.event_ids[0], sink.event_ids[1]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sqlite_alert_roundtrip() {
        let (path, root) = test_db("roundtrip");
        let item = alert("alert-1", AlertType::Price, AlertCondition::Below, 24_000.0);
        save_alert_at(&path, &item).unwrap();
        assert_eq!(load_active_alerts_at(&path).unwrap().len(), 1);
        let connection = connect_and_init(&path).unwrap();
        connection
            .execute("DELETE FROM alerts WHERE id = ?1", params!["alert-1"])
            .unwrap();
        drop(connection);
        assert!(load_active_alerts_at(&path).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}
