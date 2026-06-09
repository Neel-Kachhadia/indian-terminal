use std::{collections::HashMap, env, fs, path::PathBuf, time::Duration};

use chrono::{Datelike, Duration as ChronoDuration, Timelike, Utc, Weekday};
use redis::Commands;
use rusqlite::{params, Connection};
use serde_json::json;
use tauri::Emitter;
use tauri_plugin_notification::NotificationExt;
use tokio::time;

use crate::{
    models::{Alert, AlertCondition, AlertType, OptionLeg, OptionType, Tick},
    state::AppState,
};

const ALERTS_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS alerts (
    id            TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    alert_type    TEXT NOT NULL,
    threshold     REAL NOT NULL,
    condition     TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    triggered_at  INTEGER,
    created_at    INTEGER NOT NULL
);
"#;

pub async fn run(state: AppState) -> Result<(), String> {
    init_db()?;

    let mut previous_prices = HashMap::new();
    let mut previous_oi = HashMap::new();
    let mut interval = time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;

        if let Err(error) = evaluate_alerts(&state, &mut previous_prices, &mut previous_oi).await {
            println!("alert: evaluation failed: {error}");
        }
    }
}

pub fn init_db() -> Result<(), String> {
    let connection = connect_and_init()?;
    drop(connection);
    Ok(())
}

pub fn load_active_alerts() -> Result<Vec<Alert>, String> {
    let connection = connect_and_init()?;
    let mut statement = connection
        .prepare(
            "SELECT id, symbol, alert_type, threshold, condition, active, triggered_at, created_at
             FROM alerts
             ORDER BY active DESC, created_at DESC",
        )
        .map_err(db_error)?;

    let mut rows = statement.query([]).map_err(db_error)?;
    let mut alerts = Vec::new();

    while let Some(row) = rows.next().map_err(db_error)? {
        let alert_type: String = row.get(2).map_err(db_error)?;
        let condition: String = row.get(4).map_err(db_error)?;
        let active: i64 = row.get(5).map_err(db_error)?;

        alerts.push(Alert {
            id: row.get(0).map_err(db_error)?,
            symbol: row.get(1).map_err(db_error)?,
            alert_type: parse_alert_type(&alert_type)?,
            threshold: row.get(3).map_err(db_error)?,
            condition: parse_alert_condition(&condition)?,
            active: active != 0,
            triggered_at: row.get(6).map_err(db_error)?,
            created_at: row.get(7).map_err(db_error)?,
        });
    }

    Ok(alerts)
}

pub fn save_alert(alert: &Alert) -> Result<(), String> {
    let connection = connect_and_init()?;
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

pub fn mark_triggered(id: &str, timestamp: i64) -> Result<(), String> {
    let connection = connect_and_init()?;
    connection
        .execute(
            "UPDATE alerts SET triggered_at = ?1, active = 0 WHERE id = ?2",
            params![timestamp, id],
        )
        .map_err(db_error)?;

    Ok(())
}

pub fn delete_alert(id: &str) -> Result<(), String> {
    let connection = connect_and_init()?;
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

pub fn config_db_path() -> Result<PathBuf, String> {
    let mut path = home_dir()?;
    path.push(".indian-terminal");
    path.push("config.db");
    Ok(path)
}

async fn evaluate_alerts(
    state: &AppState,
    previous_prices: &mut HashMap<String, f64>,
    previous_oi: &mut HashMap<String, u64>,
) -> Result<(), String> {
    let alerts = load_active_alerts()?;

    for alert in alerts.into_iter().filter(|alert| alert.active) {
        let trigger_value = match alert.alert_type {
            AlertType::Price => evaluate_price_alert(state, &alert, previous_prices).await,
            AlertType::OI => evaluate_oi_alert(state, &alert, previous_oi).await,
            AlertType::PCR => evaluate_pcr_alert(state, &alert, previous_prices).await,
        };

        match trigger_value {
            Ok(Some(value)) => trigger_alert(state, &alert, value)?,
            Ok(None) => {}
            Err(error) => println!("alert: {} skipped: {error}", alert.id),
        }
    }

    Ok(())
}

async fn evaluate_price_alert(
    state: &AppState,
    alert: &Alert,
    previous_prices: &mut HashMap<String, f64>,
) -> Result<Option<f64>, String> {
    let tick = read_tick(state, &alert.symbol).await?;
    let previous = previous_prices.insert(alert.id.clone(), tick.ltp);

    if threshold_matches(&alert.condition, previous, tick.ltp, alert.threshold) {
        return Ok(Some(tick.ltp));
    }

    Ok(None)
}

async fn evaluate_oi_alert(
    state: &AppState,
    alert: &Alert,
    previous_oi: &mut HashMap<String, u64>,
) -> Result<Option<f64>, String> {
    let expiry = nearest_expiry_tuesday();
    let legs = read_chain(state, &alert.symbol, &expiry).await?;
    let total_oi = total_open_interest(&legs);
    let key = format!("{}:{expiry}", alert.symbol);
    let previous = previous_oi.insert(key, total_oi);

    let Some(previous) = previous else {
        return Ok(None);
    };

    let change = if total_oi >= previous {
        total_oi - previous
    } else {
        previous - total_oi
    };
    let value = change as f64;

    if value > alert.threshold {
        return Ok(Some(value));
    }

    Ok(None)
}

async fn evaluate_pcr_alert(
    state: &AppState,
    alert: &Alert,
    previous_prices: &mut HashMap<String, f64>,
) -> Result<Option<f64>, String> {
    let expiry = nearest_expiry_tuesday();
    let pcr = read_pcr(state, &alert.symbol, &expiry).await?;
    let key = format!("pcr:{}", alert.id);
    let previous = previous_prices.insert(key, pcr);

    if threshold_matches(&alert.condition, previous, pcr, alert.threshold) {
        return Ok(Some(pcr));
    }

    Ok(None)
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
    let payload = payload.ok_or_else(|| format!("{key} not cached"))?;

    serde_json::from_str(&payload).map_err(|error| format!("{key} JSON parse failed: {error}"))
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
    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;
    let payload = payload.ok_or_else(|| format!("{key} not cached"))?;

    serde_json::from_str(&payload).map_err(|error| format!("{key} JSON parse failed: {error}"))
}

async fn read_pcr(state: &AppState, symbol: &str, expiry: &str) -> Result<f64, String> {
    let key = format!("pcr:{symbol}:{expiry}");
    let mut redis_guard = state.redis.lock().await;
    let connection = redis_guard
        .as_mut()
        .ok_or_else(|| String::from("redis unavailable"))?;
    let payload: Option<String> = connection
        .get(&key)
        .map_err(|error| format!("redis GET {key} failed: {error}"))?;
    let payload = payload.ok_or_else(|| format!("{key} not cached"))?;

    payload
        .parse::<f64>()
        .map_err(|error| format!("{key} parse failed: {error}"))
}

fn trigger_alert(state: &AppState, alert: &Alert, value: f64) -> Result<(), String> {
    let timestamp = Utc::now().timestamp_millis();
    let alert_type = alert_type_to_db(&alert.alert_type);
    let payload = json!({
        "alert_id": alert.id,
        "symbol": alert.symbol,
        "alert_type": alert_type,
        "value": value,
        "threshold": alert.threshold,
        "timestamp": timestamp,
    });

    if let Some(app_handle) = state.app_handle() {
        if let Err(error) = app_handle.emit("alert_triggered", payload) {
            println!("alert: event emit failed: {error}");
        }

        let body = format!(
            "{} {alert_type} alert: {:.2} crossed {:.2}",
            alert.symbol, value, alert.threshold
        );
        if let Err(error) = app_handle
            .notification()
            .builder()
            .title("Indian Market Terminal")
            .body(body)
            .show()
        {
            println!("alert: desktop notification failed: {error}");
        }
    }

    mark_triggered(&alert.id, timestamp)?;
    println!(
        "alert: triggered {alert_type} for {} at {value}",
        alert.symbol
    );

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

fn total_open_interest(legs: &[OptionLeg]) -> u64 {
    let mut call_oi = 0_u64;
    let mut put_oi = 0_u64;

    for leg in legs {
        match leg.option_type {
            OptionType::CE => call_oi = call_oi.saturating_add(leg.oi),
            OptionType::PE => put_oi = put_oi.saturating_add(leg.oi),
        }
    }

    call_oi.saturating_add(put_oi)
}

fn connect_and_init() -> Result<Connection, String> {
    let path = config_db_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("alerts DB dir create failed: {error}"))?;
    }

    let connection = Connection::open(&path)
        .map_err(|error| format!("alerts DB open {} failed: {error}", path.display()))?;
    connection.execute_batch(ALERTS_SCHEMA).map_err(db_error)?;
    Ok(connection)
}

fn home_dir() -> Result<PathBuf, String> {
    let key = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    env::var_os(key)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| format!("{key} is not set"))
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

fn nearest_expiry_tuesday() -> String {
    let now_ist = Utc::now() + ChronoDuration::minutes(330);
    let today = now_ist.date_naive();
    let current_weekday = today.weekday().num_days_from_monday() as i64;
    let tuesday = Weekday::Tue.num_days_from_monday() as i64;
    let mut days_until = (tuesday - current_weekday + 7) % 7;
    let before_close = now_ist.hour() < 15 || (now_ist.hour() == 15 && now_ist.minute() < 30);

    if days_until == 0 && !before_close {
        days_until = 7;
    }

    let expiry = today + ChronoDuration::days(days_until);
    expiry.format("%d-%b-%Y").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_alert_roundtrip() {
        let root = env::temp_dir().join(format!(
            "indian-terminal-alert-test-{}",
            Utc::now().timestamp_millis()
        ));
        let previous_home = env::var_os("USERPROFILE");
        env::set_var("USERPROFILE", &root);

        init_db().expect("init_db failed");

        let alert = Alert {
            id: String::from("alert-1"),
            symbol: String::from("NIFTY"),
            alert_type: AlertType::Price,
            threshold: 24_000.0,
            condition: AlertCondition::Below,
            active: true,
            triggered_at: None,
            created_at: 1_717_000_000_000,
        };

        save_alert(&alert).expect("save_alert failed");
        let loaded = load_active_alerts().expect("load_active_alerts failed");
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].active);
        assert_eq!(loaded[0].symbol, "NIFTY");

        mark_triggered("alert-1", 1_717_000_001_000).expect("mark_triggered failed");
        let loaded = load_active_alerts().expect("load after trigger failed");
        assert_eq!(loaded.len(), 1);
        assert!(!loaded[0].active);
        assert_eq!(loaded[0].triggered_at, Some(1_717_000_001_000));

        delete_alert("alert-1").expect("delete_alert failed");
        let loaded = load_active_alerts().expect("load after delete failed");
        assert!(loaded.is_empty());

        if let Some(previous_home) = previous_home {
            env::set_var("USERPROFILE", previous_home);
        } else {
            env::remove_var("USERPROFILE");
        }
        let _ = fs::remove_dir_all(root);
    }
}
