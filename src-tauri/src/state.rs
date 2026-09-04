use crate::models::Tick;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::process::Child;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::AppHandle;
use tokio::sync::{broadcast, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FeedPhase {
    StartingBackend,
    WaitingForHealth,
    Authenticated,
    ConnectingFeed,
    Subscribing,
    WaitingForData,
    Live,
    Degraded,
    Reconnecting,
    Failed,
    ShuttingDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeChildObservation {
    Missing,
    Running,
    Exited(Option<i32>),
}

#[derive(Debug, Clone, Serialize)]
pub struct FeedRuntimeState {
    pub phase: FeedPhase,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
    pub updated_at_epoch_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeedCoverageSnapshot {
    pub schema_version: u16,
    pub provider: &'static str,
    pub connection_owner: &'static str,
    pub writer_owner: &'static str,
    pub connection_state: FeedPhase,
    pub provider_token_limit: usize,
    pub expected: usize,
    pub expected_symbols: Vec<String>,
    pub subscribed: usize,
    pub subscribed_symbols: Vec<String>,
    pub receiving: usize,
    pub fresh: usize,
    pub fresh_symbols: Vec<String>,
    pub stale: usize,
    pub stale_symbols: Vec<String>,
    pub unavailable: usize,
    pub unavailable_symbols: Vec<String>,
    pub unknown_tokens: u64,
    pub parser_errors: u64,
    pub duplicate_subscriptions: u64,
    pub reconnect_count: u64,
    pub packet_count: u64,
    pub quote_updates_per_second: f64,
    pub initialized_at_ms: Option<i64>,
    pub last_packet_time_ms: Option<i64>,
    pub coverage_stabilization_ms: Option<i64>,
    pub freshness_threshold_ms: i64,
}

#[derive(Debug, Default)]
struct FeedCoverageRuntime {
    expected_symbols: HashSet<String>,
    subscribed_symbols: HashSet<String>,
    last_received_by_symbol: HashMap<String, i64>,
    provider_token_limit: usize,
    unknown_tokens: u64,
    parser_errors: u64,
    duplicate_subscriptions: u64,
    reconnect_count: u64,
    packet_count: u64,
    initialized_at_ms: Option<i64>,
    first_packet_time_ms: Option<i64>,
    last_packet_time_ms: Option<i64>,
    coverage_stabilization_ms: Option<i64>,
}

impl FeedRuntimeState {
    fn new() -> Self {
        Self {
            phase: FeedPhase::StartingBackend,
            consecutive_failures: 0,
            last_error: None,
            updated_at_epoch_ms: now_epoch_ms(),
        }
    }

    fn transition(&mut self, next: FeedPhase, reason: Option<String>) -> Result<(), String> {
        if self.phase != next && !valid_feed_transition(self.phase, next) {
            return Err(format!(
                "invalid feed-state transition: {:?} -> {:?}",
                self.phase, next
            ));
        }
        self.phase = next;
        self.updated_at_epoch_ms = now_epoch_ms();
        if let Some(reason) = reason {
            self.last_error = Some(reason);
        }
        if matches!(next, FeedPhase::Live | FeedPhase::Authenticated) {
            self.consecutive_failures = 0;
            self.last_error = None;
        }
        Ok(())
    }
}

fn valid_feed_transition(current: FeedPhase, next: FeedPhase) -> bool {
    use FeedPhase::{
        Authenticated, ConnectingFeed, Degraded, Failed, Live, Reconnecting, ShuttingDown,
        StartingBackend, Subscribing, WaitingForData, WaitingForHealth,
    };
    matches!(
        (current, next),
        (StartingBackend, WaitingForHealth | Failed | ShuttingDown)
            | (
                WaitingForHealth,
                Authenticated | Degraded | Failed | ShuttingDown
            )
            | (
                Authenticated,
                ConnectingFeed | Degraded | Failed | ShuttingDown
            )
            | (
                ConnectingFeed,
                Subscribing | Reconnecting | Degraded | Failed | ShuttingDown
            )
            | (
                Subscribing,
                WaitingForData | Reconnecting | Degraded | Failed | ShuttingDown
            )
            | (
                WaitingForData,
                Live | Reconnecting | Degraded | Failed | ShuttingDown
            )
            | (Live, Reconnecting | Degraded | Failed | ShuttingDown)
            | (
                Degraded,
                WaitingForHealth | Reconnecting | Failed | ShuttingDown
            )
            | (
                Reconnecting,
                WaitingForHealth | ConnectingFeed | Failed | ShuttingDown
            )
            | (Failed, WaitingForHealth | ShuttingDown)
    )
}

#[derive(Clone)]
pub struct AppState {
    pub app_handle: Arc<StdMutex<Option<AppHandle>>>,
    pub compute_pid: Arc<StdMutex<Option<u32>>>,
    pub compute_child: Arc<StdMutex<Option<Child>>>,
    pub process_token: Arc<StdMutex<Option<String>>>,
    pub ingester_started: Arc<AtomicBool>,
    pub shutdown_requested: Arc<AtomicBool>,
    pub feed_runtime: Arc<StdMutex<FeedRuntimeState>>,
    pub redis: Arc<Mutex<Option<redis::Connection>>>,
    pub tick_tx: Arc<Mutex<broadcast::Sender<Tick>>>,
    feed_coverage: Arc<StdMutex<FeedCoverageRuntime>>,
}

impl AppState {
    pub fn new(redis: Option<redis::Connection>, tick_tx: broadcast::Sender<Tick>) -> Self {
        Self {
            app_handle: Arc::new(StdMutex::new(None)),
            compute_pid: Arc::new(StdMutex::new(None)),
            compute_child: Arc::new(StdMutex::new(None)),
            process_token: Arc::new(StdMutex::new(None)),
            ingester_started: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            feed_runtime: Arc::new(StdMutex::new(FeedRuntimeState::new())),
            redis: Arc::new(Mutex::new(redis)),
            tick_tx: Arc::new(Mutex::new(tick_tx)),
            feed_coverage: Arc::new(StdMutex::new(FeedCoverageRuntime::default())),
        }
    }

    pub fn set_app_handle(&self, app_handle: AppHandle) -> Result<(), String> {
        let mut guard = self
            .app_handle
            .lock()
            .map_err(|error| format!("app handle lock failed: {error}"))?;
        *guard = Some(app_handle);
        Ok(())
    }

    pub fn app_handle(&self) -> Option<AppHandle> {
        match self.app_handle.lock() {
            Ok(guard) => guard.clone(),
            Err(error) => {
                crate::bt_error!("state", "app handle lock failed: {error}");
                None
            }
        }
    }

    pub fn set_process_token(&self, token: String) -> Result<(), String> {
        let mut guard = self
            .process_token
            .lock()
            .map_err(|error| format!("process token lock failed: {error}"))?;
        *guard = Some(token);
        Ok(())
    }

    pub fn process_token(&self) -> Result<String, String> {
        self.process_token
            .lock()
            .map_err(|error| format!("process token lock failed: {error}"))?
            .clone()
            .ok_or_else(|| String::from("compute process token unavailable"))
    }

    pub fn set_compute_pid(&self, pid: u32) -> Result<(), String> {
        let mut guard = self
            .compute_pid
            .lock()
            .map_err(|error| format!("compute pid lock failed: {error}"))?;
        *guard = Some(pid);
        Ok(())
    }

    pub fn set_compute_child(&self, child: Child) -> Result<(), String> {
        let pid = child.id();
        let mut child_guard = self
            .compute_child
            .lock()
            .map_err(|error| format!("compute child lock failed: {error}"))?;
        *child_guard = Some(child);
        drop(child_guard);
        self.set_compute_pid(pid)
    }

    pub fn observe_compute_child(&self) -> Result<ComputeChildObservation, String> {
        let observation = {
            let mut child_guard = self
                .compute_child
                .lock()
                .map_err(|error| format!("compute child lock failed: {error}"))?;
            match child_guard.as_mut() {
                None => ComputeChildObservation::Missing,
                Some(child) => match child.try_wait() {
                    Ok(None) => ComputeChildObservation::Running,
                    Ok(Some(status)) => {
                        let _ = child_guard.take();
                        ComputeChildObservation::Exited(status.code())
                    }
                    Err(error) => {
                        return Err(format!("compute process status failed: {error}"));
                    }
                },
            }
        };

        if matches!(observation, ComputeChildObservation::Exited(_)) {
            if let Ok(mut pid) = self.compute_pid.lock() {
                *pid = None;
            }
            if let Ok(mut token) = self.process_token.lock() {
                *token = None;
            }
        }
        Ok(observation)
    }

    pub fn terminate_compute(&self) -> Result<(), String> {
        self.request_shutdown();
        let mut child_guard = self
            .compute_child
            .lock()
            .map_err(|error| format!("compute child lock failed: {error}"))?;
        if let Some(mut child) = child_guard.take() {
            match child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => {
                    child
                        .kill()
                        .map_err(|error| format!("compute process kill failed: {error}"))?;
                    child
                        .wait()
                        .map_err(|error| format!("compute process wait failed: {error}"))?;
                }
                Err(error) => return Err(format!("compute process status failed: {error}")),
            }
        }
        if let Ok(mut pid) = self.compute_pid.lock() {
            *pid = None;
        }
        if let Ok(mut token) = self.process_token.lock() {
            *token = None;
        }
        Ok(())
    }

    pub fn compute_pid(&self) -> Option<u32> {
        match self.compute_pid.lock() {
            Ok(guard) => *guard,
            Err(error) => {
                crate::bt_error!("state", "compute pid lock failed: {error}");
                None
            }
        }
    }

    pub fn mark_ingester_started(&self) -> bool {
        self.ingester_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub fn reset_ingester_started(&self) {
        self.ingester_started.store(false, Ordering::SeqCst);
    }

    pub fn set_feed_phase(&self, phase: FeedPhase, reason: Option<String>) -> Result<(), String> {
        self.feed_runtime
            .lock()
            .map_err(|error| format!("feed state lock failed: {error}"))?
            .transition(phase, reason)
    }

    pub fn record_feed_failure(&self, error: String) -> Result<u32, String> {
        let mut state = self
            .feed_runtime
            .lock()
            .map_err(|lock_error| format!("feed state lock failed: {lock_error}"))?;
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.last_error = Some(error);
        state.updated_at_epoch_ms = now_epoch_ms();
        Ok(state.consecutive_failures)
    }

    pub fn feed_status(&self) -> Result<FeedRuntimeState, String> {
        self.feed_runtime
            .lock()
            .map_err(|error| format!("feed state lock failed: {error}"))
            .map(|state| state.clone())
    }

    pub fn configure_feed_coverage(
        &self,
        symbols: impl IntoIterator<Item = String>,
        provider_token_limit: usize,
        duplicate_subscriptions: u64,
    ) -> Result<(), String> {
        let expected_symbols = symbols.into_iter().collect::<HashSet<_>>();
        let mut coverage = self
            .feed_coverage
            .lock()
            .map_err(|error| format!("feed coverage lock failed: {error}"))?;
        coverage.expected_symbols = expected_symbols;
        coverage.subscribed_symbols.clear();
        coverage.last_received_by_symbol.clear();
        coverage.provider_token_limit = provider_token_limit;
        coverage.duplicate_subscriptions = duplicate_subscriptions;
        coverage.packet_count = 0;
        coverage.initialized_at_ms = Some(now_epoch_ms() as i64);
        coverage.first_packet_time_ms = None;
        coverage.last_packet_time_ms = None;
        coverage.coverage_stabilization_ms = None;
        Ok(())
    }

    pub fn record_subscribed_symbols(
        &self,
        symbols: impl IntoIterator<Item = String>,
    ) -> Result<(), String> {
        let mut coverage = self
            .feed_coverage
            .lock()
            .map_err(|error| format!("feed coverage lock failed: {error}"))?;
        coverage.subscribed_symbols.extend(symbols);
        Ok(())
    }

    pub fn record_feed_packet(&self, symbol: &str, received_at_ms: i64) -> Result<(), String> {
        let mut coverage = self
            .feed_coverage
            .lock()
            .map_err(|error| format!("feed coverage lock failed: {error}"))?;
        coverage.packet_count = coverage.packet_count.saturating_add(1);
        coverage.first_packet_time_ms.get_or_insert(received_at_ms);
        coverage.last_packet_time_ms = Some(received_at_ms);
        coverage
            .last_received_by_symbol
            .insert(symbol.to_string(), received_at_ms);
        if coverage.coverage_stabilization_ms.is_none()
            && coverage.last_received_by_symbol.len() == coverage.expected_symbols.len()
        {
            coverage.coverage_stabilization_ms = coverage
                .initialized_at_ms
                .map(|started| received_at_ms.saturating_sub(started));
        }
        Ok(())
    }

    pub fn record_unknown_feed_token(&self) {
        if let Ok(mut coverage) = self.feed_coverage.lock() {
            coverage.unknown_tokens = coverage.unknown_tokens.saturating_add(1);
        }
    }

    pub fn record_feed_parser_error(&self) {
        if let Ok(mut coverage) = self.feed_coverage.lock() {
            coverage.parser_errors = coverage.parser_errors.saturating_add(1);
        }
    }

    pub fn record_feed_reconnect(&self) {
        if let Ok(mut coverage) = self.feed_coverage.lock() {
            coverage.reconnect_count = coverage.reconnect_count.saturating_add(1);
        }
    }

    pub fn feed_coverage(&self) -> Result<FeedCoverageSnapshot, String> {
        let phase = self.feed_status()?.phase;
        let coverage = self
            .feed_coverage
            .lock()
            .map_err(|error| format!("feed coverage lock failed: {error}"))?;
        let now_ms = now_epoch_ms() as i64;
        let mut expected_symbols = coverage
            .expected_symbols
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut subscribed_symbols = coverage
            .subscribed_symbols
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut fresh_symbols = Vec::new();
        let mut stale_symbols = Vec::new();
        let mut unavailable_symbols = Vec::new();
        for symbol in &expected_symbols {
            match coverage.last_received_by_symbol.get(symbol) {
                Some(received) if now_ms.saturating_sub(*received) <= Tick::STREAM_FRESH_MS => {
                    fresh_symbols.push(symbol.clone());
                }
                Some(_) => stale_symbols.push(symbol.clone()),
                None => unavailable_symbols.push(symbol.clone()),
            }
        }
        expected_symbols.sort();
        subscribed_symbols.sort();
        fresh_symbols.sort();
        stale_symbols.sort();
        unavailable_symbols.sort();
        let elapsed_seconds = coverage
            .first_packet_time_ms
            .map(|started| now_ms.saturating_sub(started).max(1) as f64 / 1_000.0);
        Ok(FeedCoverageSnapshot {
            schema_version: 1,
            provider: "ANGEL_ONE_SMART_STREAM_V2",
            connection_owner: "PYTHON_SECURE_FEED_BRIDGE",
            writer_owner: "RUST_TAURI_INGESTER",
            connection_state: phase,
            provider_token_limit: coverage.provider_token_limit,
            expected: expected_symbols.len(),
            expected_symbols,
            subscribed: subscribed_symbols.len(),
            subscribed_symbols,
            receiving: coverage.last_received_by_symbol.len(),
            fresh: fresh_symbols.len(),
            fresh_symbols,
            stale: stale_symbols.len(),
            stale_symbols,
            unavailable: unavailable_symbols.len(),
            unavailable_symbols,
            unknown_tokens: coverage.unknown_tokens,
            parser_errors: coverage.parser_errors,
            duplicate_subscriptions: coverage.duplicate_subscriptions,
            reconnect_count: coverage.reconnect_count,
            packet_count: coverage.packet_count,
            quote_updates_per_second: elapsed_seconds
                .map(|elapsed| coverage.packet_count as f64 / elapsed)
                .unwrap_or(0.0),
            initialized_at_ms: coverage.initialized_at_ms,
            last_packet_time_ms: coverage.last_packet_time_ms,
            coverage_stabilization_ms: coverage.coverage_stabilization_ms,
            freshness_threshold_ms: Tick::STREAM_FRESH_MS,
        })
    }

    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
        let _ = self.set_feed_phase(FeedPhase::ShuttingDown, None);
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{FeedPhase, FeedRuntimeState};

    #[test]
    fn feed_state_machine_accepts_reconnect_cycle() {
        let mut state = FeedRuntimeState::new();
        state.transition(FeedPhase::WaitingForHealth, None).unwrap();
        state.transition(FeedPhase::Authenticated, None).unwrap();
        state.transition(FeedPhase::ConnectingFeed, None).unwrap();
        state.transition(FeedPhase::Subscribing, None).unwrap();
        state.transition(FeedPhase::WaitingForData, None).unwrap();
        state.transition(FeedPhase::Live, None).unwrap();
        state
            .transition(FeedPhase::Reconnecting, Some("disconnect".into()))
            .unwrap();
        state.transition(FeedPhase::WaitingForHealth, None).unwrap();
        assert_eq!(state.phase, FeedPhase::WaitingForHealth);
    }

    #[test]
    fn feed_state_machine_rejects_false_ready_transition() {
        let mut state = FeedRuntimeState::new();
        let error = state.transition(FeedPhase::Live, None).unwrap_err();
        assert!(error.contains("invalid feed-state transition"));
        assert_eq!(state.phase, FeedPhase::StartingBackend);
    }
}
