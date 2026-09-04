use std::{
    env,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use indian_terminal::{
    commands,
    models::Tick,
    state::{AppState, ComputeChildObservation, FeedPhase},
};
use tauri::{AppHandle, Manager};
use tokio::sync::broadcast;

fn main() {
    let (tick_tx, _tick_rx) = broadcast::channel::<Tick>(1024);
    let redis = create_redis_connection();
    let state = AppState::new(redis, tick_tx);

    let app = tauri::Builder::default()
        // Must be the first plugin registered (Tauri requirement): it takes
        // an OS-level single-instance lock before anything else in setup()
        // runs. A second launch -- installer auto-launch racing a manual
        // launch, a shortcut double-click, Windows startup -- never reaches
        // this process's own setup() at all; Tauri hands its argv/cwd to the
        // FIRST instance's callback and the second process exits immediately,
        // so there is structurally only ever one backend, one collector
        // supervisor, and one DB writer/remote-client per machine session.
        // Reproduced live before this fix: a silent NSIS install's own
        // auto-launch plus a manual launch produced two full process trees
        // (two indian-terminal.exe, two bharatterminal-compute.exe) with two
        // real "BharatTerminal" windows simultaneously.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .setup(|app| {
            let app_state = app.state::<AppState>().inner().clone();
            if let Err(error) = app_state.set_app_handle(app.handle().clone()) {
                indian_terminal::bt_error!("state", "app handle set failed: {error}");
            }

            ensure_release_resources(app.handle()).map_err(std::io::Error::other)?;
            ensure_collector_task_registered();

            let state1 = app.state::<AppState>().inner().clone();
            let state2 = app.state::<AppState>().inner().clone();
            let state3 = app.state::<AppState>().inner().clone();
            let state4 = app.state::<AppState>().inner().clone();
            let state5 = app.state::<AppState>().inner().clone();
            tauri::async_runtime::spawn(async move {
                supervise_python_subprocess(state4).await;
            });
            tauri::async_runtime::spawn(async move {
                supervise_redis(state5).await;
            });
            tauri::async_runtime::spawn(async move {
                supervise_ingester(state1).await;
            });
            tauri::async_runtime::spawn(async move {
                if let Err(error) = indian_terminal::poller::run(state2).await {
                    indian_terminal::bt_error!("poller", "{error}");
                }
            });
            tauri::async_runtime::spawn(async move {
                if let Err(error) = indian_terminal::alert_engine::run(state3).await {
                    indian_terminal::bt_error!("alert_engine", "{error}");
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_quote,
            commands::get_quotes,
            commands::get_live_quote_coverage,
            commands::get_chain,
            commands::get_pcr,
            commands::get_ohlcv,
            commands::get_expiries,
            commands::get_lot_size,
            commands::get_chain_greeks,
            commands::get_iv_surface,
            commands::get_payoff,
            commands::get_futures_basis,
            commands::get_signals,
            commands::get_all_signals,
            commands::get_sentiment,
            commands::get_cross_asset,
            commands::get_ml_prediction,
            commands::get_ml_accuracy,
            commands::get_bif_all,
            commands::get_bif_cds,
            commands::get_bif_history,
            commands::get_ohlc_forecast,
            commands::train_all_stocks,
            commands::get_training_status,
            commands::get_trained_symbols,
            commands::train_single_symbol,
            commands::get_decision_intelligence,
            commands::get_market_brief,
            commands::get_intelligence_patterns,
            commands::get_intelligence_regime,
            commands::get_risk_var,
            commands::get_options_flow,
            commands::get_data_summary,
            commands::get_data_errors,
            commands::get_stock_snapshot,
            commands::get_stock_universe,
            commands::get_equity_summary,
            commands::get_equity_valuation,
            commands::get_valuation_universe,
            commands::get_valuation_status,
            commands::run_screener,
            commands::run_backtest,
            commands::get_backtest_status,
            commands::get_backtest_result,
            commands::list_strategies,
            commands::get_backtest_history,
            commands::get_screener_results,
            commands::get_indicators,
            commands::set_alert,
            commands::delete_alert,
            commands::get_alerts,
            commands::get_market_status,
            commands::get_market_session,
            commands::wait_for_backend_ready,
            commands::get_instrument_list,
            commands::download_historical,
            commands::get_broker_auth_status,
            commands::get_integrity_status,
            commands::fetch_equity_pdf
        ])
        .build(tauri::generate_context!());

    match app {
        Ok(app) => app.run(|app_handle, event| {
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                let state = app_handle.state::<AppState>();
                if let Err(error) = state.terminate_compute() {
                    indian_terminal::bt_error!("shutdown", "{error}");
                }
            }
        }),
        Err(error) => indian_terminal::bt_error!("tauri", "{error}"),
    }
}

fn create_redis_connection() -> Option<redis::Connection> {
    match connect_redis() {
        Ok(connection) => {
            indian_terminal::bt_info!("redis", "reachable");
            Some(connection)
        }
        Err(error) => {
            indian_terminal::bt_error!("redis", "not reachable: {error}");
            None
        }
    }
}

fn connect_redis() -> Result<redis::Connection, String> {
    let url = env::var("BHARATTERMINAL_REDIS_URL")
        .map_err(|_| String::from("authenticated local Redis URL unavailable"))?;
    validate_local_redis_url(&url)?;
    connect_redis_at(&url)
}

fn validate_local_redis_url(value: &str) -> Result<(), String> {
    let parsed = url::Url::parse(value).map_err(|_| String::from("invalid Redis URL"))?;
    if parsed.scheme() != "redis" {
        return Err(String::from("Redis URL must use redis scheme on loopback"));
    }
    if !matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "::1")) {
        return Err(String::from("Redis must be bound to loopback"));
    }
    if parsed.password().is_none_or(str::is_empty) {
        return Err(String::from("Redis authentication is required"));
    }
    Ok(())
}

fn connect_redis_at(url: &str) -> Result<redis::Connection, String> {
    let client = match redis::Client::open(url) {
        Ok(client) => client,
        Err(error) => return Err(format!("client init failed: {error}")),
    };

    let connection = client
        .get_connection_with_timeout(Duration::from_secs(2))
        .map_err(|error| error.to_string())?;
    connection
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| error.to_string())?;
    connection
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| error.to_string())?;
    Ok(connection)
}

async fn supervise_redis(state: AppState) {
    let mut retry_attempt = 0_u32;
    while !state.is_shutdown_requested() {
        let failure = {
            let mut guard = state.redis.lock().await;
            match guard.as_mut() {
                Some(connection) => match redis::cmd("PING").query::<String>(connection) {
                    Ok(response) if response == "PONG" => None,
                    Ok(response) => {
                        *guard = None;
                        Some(format!("unexpected PING response: {response}"))
                    }
                    Err(error) => {
                        *guard = None;
                        Some(format!("connection lost: {error}"))
                    }
                },
                None => Some(String::from("connection unavailable")),
            }
        };

        if failure.is_none() {
            retry_attempt = 0;
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        if retry_attempt == 0 {
            indian_terminal::bt_error!("redis", "{}", failure.unwrap_or_default());
        }
        match connect_redis() {
            Ok(connection) => {
                let mut guard = state.redis.lock().await;
                *guard = Some(connection);
                retry_attempt = 0;
                indian_terminal::bt_info!("redis", "connection restored");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Err(error) => {
                if retry_attempt.is_multiple_of(10) {
                    indian_terminal::bt_error!("redis", "reconnect failed: {error}");
                }
                tokio::time::sleep(retry_delay(retry_attempt, std::process::id())).await;
                retry_attempt = retry_attempt.saturating_add(1);
            }
        }
    }
}

fn launch_python_subprocess(state: AppState) {
    let executable = python_binary_path();
    let compute_dir = compute_dir();

    if !cfg!(debug_assertions) && !executable.exists() {
        indian_terminal::bt_error!(
            "python",
            "compute binary missing at {}",
            executable.display()
        );
        return;
    }

    let mut command = Command::new(&executable);
    let process_token = match new_process_token() {
        Ok(token) => token,
        Err(error) => {
            indian_terminal::bt_error!("python", "process-token generation failed: {error}");
            return;
        }
    };
    if let Err(error) = state.set_process_token(process_token.clone()) {
        indian_terminal::bt_error!("python", "process-token store failed: {error}");
        return;
    }
    command.env("BHARATTERMINAL_PROCESS_TOKEN", process_token);
    if let Ok(current_executable) = env::current_exe() {
        command.env("BHARATTERMINAL_TAURI_EXECUTABLE", current_executable);
    }
    if let Ok(config_path) = indian_terminal::alert_engine::config_db_path() {
        if let Some(app_data_root) = config_path.parent().and_then(|data| data.parent()) {
            command.env("BHARATTERMINAL_APP_DATA", app_data_root);
        }
    }
    if let Some(app_handle) = state.app_handle() {
        if let Ok(resource_dir) = app_handle.path().resource_dir() {
            let resource_root = resolve_resource_root(&resource_dir);
            command.env("BHARATTERMINAL_RESOURCE_ROOT", &resource_root);
            command.env("BHARATTERMINAL_PROTECTED_ASSET_ROOT", &resource_root);
        }
    }
    if cfg!(debug_assertions) {
        command
            .env("BHARATTERMINAL_ALLOW_UNTRUSTED_DEV", "1")
            .arg("run")
            .arg("--project")
            .arg(".")
            .arg("--frozen")
            .arg("python")
            .arg("-m")
            .arg("uvicorn")
            .arg("main:app")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg("8765")
            .current_dir(compute_dir);
    }

    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    match child {
        Ok(mut child) => {
            #[cfg(windows)]
            assign_child_to_kill_on_close_job(&child);
            if let Some(stdout) = child.stdout.take() {
                thread::spawn(move || {
                    let reader = BufReader::new(stdout);
                    for line in reader.lines().map_while(Result::ok) {
                        indian_terminal::bt_info!("python.stdout", "{line}");
                    }
                });
            }
            if let Some(stderr) = child.stderr.take() {
                thread::spawn(move || {
                    let reader = BufReader::new(stderr);
                    for line in reader.lines().map_while(Result::ok) {
                        log_python_stderr_line(&line);
                    }
                });
            }
            if let Err(error) = state.set_compute_child(child) {
                indian_terminal::bt_error!("python", "child supervision failed: {error}");
                return;
            }
            indian_terminal::bt_info!("python", "subprocess launched");
        }
        Err(error) => indian_terminal::bt_error!("python", "subprocess launch failed: {error}"),
    }
}

/// Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` assigned to the
/// Python compute child. The OS closes this process's last handle to the job
/// whenever this process exits for any reason (normal exit, crash, or a
/// forceful external termination such as `taskkill /F`), which makes Windows
/// kill every process still assigned to the job. This is what bounds the
/// child's lifetime to the parent's even when parent cleanup code never runs.
#[cfg(windows)]
fn kill_on_close_job_handle() -> windows_sys::Win32::Foundation::HANDLE {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    static JOB: OnceLock<isize> = OnceLock::new();

    let raw = *JOB.get_or_init(|| unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            indian_terminal::bt_error!("python", "job-object create failed");
            return 0;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            indian_terminal::bt_error!("python", "job-object kill-on-close config failed");
            return 0;
        }
        job as isize
    });
    raw as HANDLE
}

#[cfg(windows)]
fn assign_child_to_kill_on_close_job(child: &std::process::Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;

    let job = kill_on_close_job_handle();
    if job.is_null() {
        return;
    }
    let process_handle = child.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
    let ok = unsafe { AssignProcessToJobObject(job, process_handle) };
    if ok == 0 {
        indian_terminal::bt_error!(
            "python",
            "job-object assignment failed: child will not be bounded to parent lifetime"
        );
    }
}

fn python_stderr_is_error(line: &str) -> bool {
    if line.starts_with("INFO:") || line.starts_with("WARNING:") {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|value| value.get("level")?.as_str().map(str::to_owned))
        .is_none_or(|level| matches!(level.as_str(), "ERROR" | "CRITICAL" | "FATAL"))
}

fn log_python_stderr_line(line: &str) {
    if python_stderr_is_error(line) {
        indian_terminal::bt_error!("python.stderr", "{line}");
    } else {
        indian_terminal::bt_info!("python.stderr", "{line}");
    }
}

async fn supervise_python_subprocess(state: AppState) {
    let mut retry_attempt = 0_u32;
    while !state.is_shutdown_requested() {
        match state
            .observe_compute_child()
            .map(|observation| compute_recovery_action(observation, state.is_shutdown_requested()))
        {
            Ok(ComputeRecoveryAction::Monitor) => {
                retry_attempt = 0;
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
            Ok(ComputeRecoveryAction::Restart(code)) => {
                if code.is_some() {
                    indian_terminal::bt_error!(
                        "python",
                        "subprocess exited with code {code:?}; restarting"
                    );
                }
                launch_python_subprocess(state.clone());
                tokio::time::sleep(retry_delay(retry_attempt, std::process::id())).await;
                retry_attempt = retry_attempt.saturating_add(1);
            }
            Ok(ComputeRecoveryAction::Stop) => break,
            Err(error) => {
                indian_terminal::bt_error!("python", "subprocess supervision failed: {error}");
                tokio::time::sleep(retry_delay(retry_attempt, std::process::id())).await;
                retry_attempt = retry_attempt.saturating_add(1);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComputeRecoveryAction {
    Monitor,
    Restart(Option<i32>),
    Stop,
}

fn compute_recovery_action(
    observation: ComputeChildObservation,
    shutdown_requested: bool,
) -> ComputeRecoveryAction {
    if shutdown_requested {
        return ComputeRecoveryAction::Stop;
    }
    match observation {
        ComputeChildObservation::Running => ComputeRecoveryAction::Monitor,
        ComputeChildObservation::Missing => ComputeRecoveryAction::Restart(None),
        ComputeChildObservation::Exited(code) => ComputeRecoveryAction::Restart(code),
    }
}

fn new_process_token() -> Result<String, String> {
    let mut bytes = [0_u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|error| error.to_string())?;
    Ok(hex::encode(bytes))
}

fn python_binary_path() -> PathBuf {
    #[cfg(not(debug_assertions))]
    {
        for candidate in release_sidecar_candidates() {
            if candidate.exists() {
                return candidate;
            }
        }
    }

    PathBuf::from("uv")
}

#[cfg(not(debug_assertions))]
fn release_sidecar_candidates() -> Vec<PathBuf> {
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    vec![
        exe_dir.join("bharatterminal-compute.exe"),
        exe_dir.join("bharatterminal-compute-x86_64-pc-windows-msvc.exe"),
        exe_dir.join("resources").join("bharatterminal-compute.exe"),
        exe_dir
            .join("resources")
            .join("bharatterminal-compute-x86_64-pc-windows-msvc.exe"),
    ]
}

/// Registers/refreshes the `BharatTerminal_SmartCollector` Task Scheduler
/// task by running the installed compute sidecar with `collector-task
/// install` (see `auto_collector.setup_autostart.setup`, which is
/// idempotent -- `schtasks /Create ... /F` -- so calling this on every
/// launch just keeps the registered Action's Command/Arguments/WorkingDirectory
/// pinned to THIS install's sidecar path, self-healing an upgrade or a task
/// deleted out-of-band). Debug builds skip this entirely: there is no
/// installed sidecar to point Task Scheduler at during `cargo tauri dev`,
/// and the collector is expected to run manually from the repo in dev.
/// Failure is logged, never fatal -- the rest of the app must still start
/// even if Task Scheduler registration is unavailable (e.g. non-interactive
/// service accounts, restricted test environments).
#[cfg(not(debug_assertions))]
fn ensure_collector_task_registered() {
    let executable = python_binary_path();
    if !executable.exists() {
        indian_terminal::bt_error!(
            "collector-task",
            "compute sidecar missing at {}; cannot register collector task",
            executable.display()
        );
        return;
    }
    match Command::new(&executable)
        .arg("collector-task")
        .arg("install")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) if output.status.success() => {
            indian_terminal::bt_info!("collector-task", "registered");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            indian_terminal::bt_error!(
                "collector-task",
                "registration failed: {} {}",
                stderr.trim(),
                stdout.trim()
            );
        }
        Err(error) => {
            indian_terminal::bt_error!("collector-task", "registration failed: {error}");
        }
    }
}

#[cfg(debug_assertions)]
fn ensure_collector_task_registered() {}

fn compute_dir() -> PathBuf {
    let mut path = project_root();
    path.push("compute");
    path
}

fn project_root() -> PathBuf {
    let current_dir = match env::current_dir() {
        Ok(path) => path,
        Err(_) => PathBuf::from("."),
    };

    if current_dir.file_name().and_then(|name| name.to_str()) == Some("src-tauri") {
        match current_dir.parent() {
            Some(parent) => parent.to_path_buf(),
            None => current_dir,
        }
    } else {
        current_dir
    }
}

async fn supervise_ingester(state: AppState) {
    const RETRY_BUDGET: u32 = 8;
    let client = reqwest::Client::new();
    let mut retry_attempt = 0_u32;

    let mut last_logged_blocker: Option<String> = None;
    while !state.is_shutdown_requested() {
        let _ = state.set_feed_phase(FeedPhase::WaitingForHealth, None);
        if let Err(error) = wait_for_backend_ready(&client, &state).await {
            let blocker = extract_blocking_components(&error);
            if last_logged_blocker.as_deref() != Some(blocker.as_str()) {
                indian_terminal::bt_error!(
                    "ingester",
                    "backend not ready, feed cannot start yet; blocking: {blocker}"
                );
                last_logged_blocker = Some(blocker);
            }
            let failures = state
                .record_feed_failure(error.clone())
                .unwrap_or(RETRY_BUDGET);
            let phase = if failures >= RETRY_BUDGET {
                FeedPhase::Failed
            } else {
                FeedPhase::Degraded
            };
            let _ = state.set_feed_phase(phase, Some(error));
            tokio::time::sleep(retry_delay(retry_attempt, std::process::id())).await;
            retry_attempt = retry_attempt.saturating_add(1);
            continue;
        }

        if !state.mark_ingester_started() {
            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }

        let result = indian_terminal::ingester::run(state.clone()).await;
        state.reset_ingester_started();
        if state.is_shutdown_requested() {
            break;
        }
        let reached_live = state
            .feed_status()
            .map(|status| status.phase == FeedPhase::Live)
            .unwrap_or(false);
        let error = result
            .err()
            .unwrap_or_else(|| "Angel stream ended".to_string());
        indian_terminal::bt_error!("ingester", "{error}");
        state.record_feed_reconnect();
        let _ = state.record_feed_failure(error.clone());
        let _ = state.set_feed_phase(FeedPhase::Reconnecting, Some(error));
        tokio::time::sleep(retry_delay(retry_attempt, std::process::id())).await;
        retry_attempt = if reached_live {
            0
        } else {
            retry_attempt.saturating_add(1)
        };
    }

    state.reset_ingester_started();
    let _ = state.set_feed_phase(FeedPhase::ShuttingDown, None);
}

async fn wait_for_backend_ready(client: &reqwest::Client, state: &AppState) -> Result<(), String> {
    wait_for_backend_ready_at(
        client,
        state,
        "http://127.0.0.1:8765/health",
        20,
        Duration::from_secs(3),
    )
    .await
}

async fn wait_for_backend_ready_at(
    client: &reqwest::Client,
    state: &AppState,
    health_url: &str,
    max_attempts: u32,
    request_timeout: Duration,
) -> Result<(), String> {
    let mut last_error = String::from("compute process token unavailable");
    for attempt in 0..max_attempts {
        if state.is_shutdown_requested() {
            return Err(String::from("shutdown requested"));
        }
        let token = match state.process_token() {
            Ok(token) => token,
            Err(error) => {
                last_error = error;
                tokio::time::sleep(retry_delay(attempt, std::process::id())).await;
                continue;
            }
        };
        match client
            .get(health_url)
            .header("X-Process-Token", token)
            .timeout(request_timeout)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(payload)
                        if payload
                            .pointer("/readiness/status")
                            .and_then(|value| value.as_str())
                            == Some("READY")
                            || payload.get("status").and_then(|value| value.as_str())
                                == Some("ok") =>
                    {
                        return Ok(());
                    }
                    Ok(payload) => last_error = format!("backend not ready: {payload}"),
                    Err(error) => last_error = format!("health parse failed: {error}"),
                }
            }
            Ok(response) => last_error = format!("health returned HTTP {}", response.status()),
            Err(error) => last_error = format!("health request failed: {error}"),
        }
        tokio::time::sleep(retry_delay(attempt, std::process::id())).await;
    }
    Err(format!(
        "backend readiness retry budget exhausted: {last_error}"
    ))
}

/// Pulls `readiness.blocking_components` out of a `wait_for_backend_ready`
/// error string for a short, greppable log line instead of the multi-KB
/// health snapshot the error otherwise carries verbatim.
fn extract_blocking_components(error: &str) -> String {
    let Some(start) = error.find('{') else {
        return error.chars().take(200).collect();
    };
    match serde_json::from_str::<serde_json::Value>(&error[start..]) {
        Ok(payload) => payload
            .pointer("/readiness/blocking_components")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown (no readiness.blocking_components in payload)".to_string()),
        Err(_) => error.chars().take(200).collect(),
    }
}

fn retry_delay(attempt: u32, salt: u32) -> Duration {
    let exponent = attempt.min(5);
    let base_ms = 250_u64.saturating_mul(1_u64 << exponent);
    let jitter_ms = ((salt as u64)
        .wrapping_mul(31)
        .wrapping_add(attempt as u64 * 17))
        % 251;
    Duration::from_millis((base_ms + jitter_ms).min(8_000))
}

fn ensure_release_resources(app: &AppHandle) -> Result<(), String> {
    #[cfg(debug_assertions)]
    {
        let _ = app;
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    {
        use std::fs;

        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|error| format!("resource dir unavailable: {error}"))?;
        let resource_root = resolve_resource_root(&resource_dir);
        indian_terminal::release_integrity::verify_release_resources(&resource_root)?;
        let appdata = resolve_bootstrap_data_root(
            env::var_os("BHARATTERMINAL_APP_DATA").map(PathBuf::from),
            env::var_os("APPDATA").map(PathBuf::from),
        )?;
        let data_dir = appdata.join("data");
        let models_dir = appdata.join("models");
        fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
        fs::create_dir_all(&models_dir).map_err(|error| error.to_string())?;

        let db_source = [
            resource_root.join("data").join("terminal.db"),
            resource_root.join("terminal.db"),
        ]
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or_else(|| resource_root.join("data").join("terminal.db"));
        let db_target = data_dir.join("terminal.db");
        if !db_target.exists() {
            if let Some(gzip_path) = find_db_gzip(&resource_root) {
                decompress_db_gzip(&gzip_path, &db_target)?;
            } else if let Some(parts_dir) = find_db_parts_dir(&resource_root) {
                reassemble_db_parts(&parts_dir, &db_target)?;
            } else if db_source.exists() {
                fs::copy(&db_source, &db_target).map_err(|error| error.to_string())?;
            }
        }

        let model_source = resource_root.join("compute").join("models");
        if model_source.exists() {
            copy_dir_missing(&model_source, &models_dir)?;
        }
        Ok(())
    }
}

/// Tauri's NSIS bundler mirrors each declared resource path relative to
/// `tauri.conf.json`'s own directory (`src-tauri/`); every resource this
/// product ships (`../compute/dist/...`, `../release/...`,
/// `../bundle-data/...`, `../dist/**/*`) climbs ONE level out of `src-tauri/`
/// to reach its real location at the repo root, since compute/release/
/// bundle-data/dist all live as siblings of src-tauri/, not inside it.
/// Tauri's bundler cannot place files outside the resource root it controls,
/// so it materializes that one `../` as a literal `_up_` directory instead
/// -- `resource_dir()` at runtime still points at the resource root itself,
/// NOT into `_up_`. Every resource_dir()-relative lookup in this file (the
/// integrity manifest, the DB bootstrap, models, and the
/// BHARATTERMINAL_RESOURCE_ROOT/PROTECTED_ASSET_ROOT env vars the Python
/// sidecar uses to run its OWN integrity_check()) has to resolve through
/// this same `_up_` indirection or it silently looks in the wrong place.
/// Reproduced live: a real NSIS install's first launch panicked with
/// "release integrity metadata missing" before this fix, even though the
/// signed manifest was sitting right there on disk one level down.
#[cfg(not(debug_assertions))]
fn resolve_resource_root(resource_dir: &Path) -> PathBuf {
    let escaped = resource_dir.join("_up_");
    if escaped.is_dir() {
        escaped
    } else {
        resource_dir.to_path_buf()
    }
}

#[cfg(debug_assertions)]
fn resolve_resource_root(resource_dir: &Path) -> PathBuf {
    resource_dir.to_path_buf()
}

/// Resolves the release-build bootstrap data root the same way every other
/// subsystem does (alert_engine::config_db_path, structured_log::log_path):
/// an explicit BHARATTERMINAL_APP_DATA override is used as-is (it already
/// names the root, not a `BharatTerminal` subfolder to append it to);
/// otherwise fall back to `%APPDATA%\BharatTerminal`. Pulled out as a pure
/// function -- no direct env::var calls -- so the override-wins behavior is
/// unit-testable without mutating process-global env vars, which would race
/// across parallel `cargo test` threads.
#[cfg(any(not(debug_assertions), test))]
fn resolve_bootstrap_data_root(
    app_data_override: Option<PathBuf>,
    appdata_env: Option<PathBuf>,
) -> Result<PathBuf, String> {
    match app_data_override {
        Some(base) => Ok(base),
        None => appdata_env
            .map(|base| base.join("BharatTerminal"))
            .ok_or_else(|| String::from("APPDATA unavailable")),
    }
}

#[cfg(test)]
mod bootstrap_data_root_tests {
    use super::resolve_bootstrap_data_root;
    use std::path::PathBuf;

    #[test]
    fn explicit_override_wins_over_appdata() {
        let root = resolve_bootstrap_data_root(
            Some(PathBuf::from(r"D:\BharatTerminal\data")),
            Some(PathBuf::from(r"C:\Users\test\AppData\Roaming")),
        )
        .expect("override branch never fails");
        assert_eq!(root, PathBuf::from(r"D:\BharatTerminal\data"));
    }

    #[test]
    fn override_wins_even_when_appdata_absent() {
        let root =
            resolve_bootstrap_data_root(Some(PathBuf::from(r"D:\BharatTerminal\data")), None)
                .expect("override branch never fails");
        assert_eq!(root, PathBuf::from(r"D:\BharatTerminal\data"));
    }

    #[test]
    fn falls_back_to_appdata_bharatterminal_when_no_override() {
        let root = resolve_bootstrap_data_root(
            None,
            Some(PathBuf::from(r"C:\Users\test\AppData\Roaming")),
        )
        .expect("fallback branch has a value");
        assert_eq!(
            root,
            PathBuf::from(r"C:\Users\test\AppData\Roaming\BharatTerminal")
        );
    }

    #[test]
    fn errors_explicitly_when_neither_is_set() {
        let result = resolve_bootstrap_data_root(None, None);
        assert!(
            result.is_err(),
            "must fail explicitly, never silently pick a path"
        );
    }

    #[test]
    fn override_value_used_verbatim_even_if_path_does_not_exist_yet() {
        // Directory creation (fs::create_dir_all) happens after this
        // resolves -- a not-yet-existing configured root is the normal
        // first-run case, not an error condition for the resolver itself.
        let root =
            resolve_bootstrap_data_root(Some(PathBuf::from(r"D:\does-not-exist-yet\data")), None)
                .expect("resolver does not stat the path");
        assert_eq!(root, PathBuf::from(r"D:\does-not-exist-yet\data"));
    }

    /// End-to-end reproduction of the actual bug, on real temp directories
    /// standing in for "D:\...\data" (configured override) and
    /// "C:\...\AppData\Roaming" (the old default) -- this exercises the same
    /// resolve-then-create_dir_all sequence ensure_release_resources runs,
    /// not just the resolver in isolation. Before the fix this test's C-side
    /// fake path would gain a BharatTerminal\data + \models pair on every
    /// run; after the fix it must stay completely empty.
    #[test]
    fn cold_start_never_recreates_state_under_fake_c_when_override_points_at_fake_d() {
        use std::fs;

        let unique = format!(
            "bharatterminal_recreation_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let sandbox = std::env::temp_dir().join(unique);
        let fake_d_configured_root = sandbox.join("fake-d").join("BharatTerminal").join("data");
        let fake_c_appdata_roaming = sandbox.join("fake-c").join("AppData").join("Roaming");
        fs::create_dir_all(&fake_d_configured_root).expect("set up fake D sandbox");
        fs::create_dir_all(&fake_c_appdata_roaming).expect("set up fake C sandbox");

        // Exactly the call ensure_release_resources makes: an explicit
        // override is configured (fake D), and APPDATA (fake C) is also
        // present in the environment, as it always is on Windows.
        let resolved = resolve_bootstrap_data_root(
            Some(fake_d_configured_root.clone()),
            Some(fake_c_appdata_roaming.clone()),
        )
        .expect("override branch never fails");

        // Same side effects ensure_release_resources performs on the
        // resolved root.
        fs::create_dir_all(resolved.join("data")).expect("create data dir");
        fs::create_dir_all(resolved.join("models")).expect("create models dir");

        let fake_c_bharatterminal = fake_c_appdata_roaming.join("BharatTerminal");
        assert!(
            !fake_c_bharatterminal.exists(),
            "recreation-trap regression: a BharatTerminal dir was created under the fake C:\\...\\AppData\\Roaming even though an explicit D-style override was configured"
        );
        assert!(
            resolved.join("data").is_dir(),
            "data dir must exist under the configured D-style root"
        );
        assert!(
            resolved.join("models").is_dir(),
            "models dir must exist under the configured D-style root"
        );
        assert_eq!(
            resolved, fake_d_configured_root,
            "mutable state must resolve to the configured root, not a derived/second path"
        );

        let _ = fs::remove_dir_all(&sandbox);
    }
}

#[cfg(not(debug_assertions))]
fn copy_dir_missing(source: &PathBuf, target: &PathBuf) -> Result<(), String> {
    use std::fs;

    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            fs::create_dir_all(&target_path).map_err(|error| error.to_string())?;
            copy_dir_missing(&source_path, &target_path)?;
        } else if !target_path.exists() {
            fs::copy(&source_path, &target_path).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
fn find_db_gzip(resource_dir: &PathBuf) -> Option<PathBuf> {
    [
        resource_dir.join("bundle-data").join("terminal.db.gz"),
        resource_dir.join("terminal.db.gz"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

#[cfg(not(debug_assertions))]
fn decompress_db_gzip(gzip_path: &PathBuf, target: &PathBuf) -> Result<(), String> {
    use std::{
        fs::{self, File},
        io::{Read, Write},
    };

    let tmp = target.with_extension("db.tmp");
    let input = File::open(gzip_path).map_err(|error| error.to_string())?;
    let mut decoder = flate2::read::GzDecoder::new(input);
    let mut output = File::create(&tmp).map_err(|error| error.to_string())?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = decoder
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| error.to_string())?;
    }
    output.flush().map_err(|error| error.to_string())?;
    fs::rename(tmp, target).map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(not(debug_assertions))]
fn find_db_parts_dir(resource_dir: &PathBuf) -> Option<PathBuf> {
    [
        resource_dir.join("bundle-data").join("terminal.db.parts"),
        resource_dir.join("terminal.db.parts"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

#[cfg(not(debug_assertions))]
fn reassemble_db_parts(parts_dir: &PathBuf, target: &PathBuf) -> Result<(), String> {
    use std::{
        fs::{self, File},
        io::{Read, Write},
    };

    let mut parts = fs::read_dir(parts_dir)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("terminal.db.part"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    parts.sort();
    if parts.is_empty() {
        return Err(String::from("terminal.db parts not found"));
    }

    let tmp = target.with_extension("db.tmp");
    let mut output = File::create(&tmp).map_err(|error| error.to_string())?;
    let mut buffer = vec![0u8; 1024 * 1024];
    for part in parts {
        let mut input = File::open(&part).map_err(|error| error.to_string())?;
        loop {
            let read = input.read(&mut buffer).map_err(|error| error.to_string())?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .map_err(|error| error.to_string())?;
        }
    }
    output.flush().map_err(|error| error.to_string())?;
    fs::rename(tmp, target).map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod process_token_tests {
    use super::{
        compute_recovery_action, connect_redis_at, new_process_token, python_stderr_is_error,
        retry_delay, validate_local_redis_url, wait_for_backend_ready_at, ComputeRecoveryAction,
    };
    use indian_terminal::{
        models::Tick,
        state::{AppState, ComputeChildObservation},
    };
    use std::{
        env, fs,
        net::TcpListener as StdTcpListener,
        process::{Command, Stdio},
        thread,
        time::Duration,
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        sync::broadcast,
    };

    #[test]
    fn process_token_has_256_bits_and_rotates() {
        let first = new_process_token().expect("token");
        let second = new_process_token().expect("token");
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
        assert!(!python_stderr_is_error("INFO:     server ready"));
        assert!(!python_stderr_is_error(
            r#"{"level":"WARNING","event":"development mode"}"#
        ));
        assert!(python_stderr_is_error(
            r#"{"level":"ERROR","event":"request failed"}"#
        ));
        assert!(python_stderr_is_error("Traceback (most recent call last):"));
    }

    #[test]
    fn redis_requires_loopback_and_authentication() {
        assert!(validate_local_redis_url("redis://:opaque@127.0.0.1:6379/").is_ok());
        assert!(validate_local_redis_url("redis://127.0.0.1:6379/").is_err());
        assert!(validate_local_redis_url("redis://:opaque@192.0.2.1:6379/").is_err());
    }

    #[test]
    fn compute_crash_and_missing_child_require_restart_but_shutdown_does_not() {
        assert_eq!(
            compute_recovery_action(ComputeChildObservation::Exited(Some(17)), false),
            ComputeRecoveryAction::Restart(Some(17))
        );
        assert_eq!(
            compute_recovery_action(ComputeChildObservation::Missing, false),
            ComputeRecoveryAction::Restart(None)
        );
        assert_eq!(
            compute_recovery_action(ComputeChildObservation::Running, false),
            ComputeRecoveryAction::Monitor
        );
        assert_eq!(
            compute_recovery_action(ComputeChildObservation::Exited(Some(17)), true),
            ComputeRecoveryAction::Stop
        );
    }

    #[test]
    fn reconnect_backoff_is_bounded() {
        for attempt in 0..100 {
            let delay = retry_delay(attempt, 17);
            assert!(delay >= Duration::from_millis(250));
            assert!(delay <= Duration::from_secs(8));
        }
    }

    #[tokio::test]
    async fn backend_readiness_retries_slow_and_malformed_responses() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let responses = [
                ("503 Service Unavailable", r#"{"status":"starting"}"#),
                ("200 OK", "not-json"),
                ("200 OK", r#"{"readiness":{"status":"READY"}}"#),
            ];
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 2048];
                let _ = stream.read(&mut request).await.unwrap();
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let (tick_tx, _tick_rx) = broadcast::channel::<Tick>(1);
        let state = AppState::new(None, tick_tx);
        state.set_process_token(String::from("test-token")).unwrap();
        let client = reqwest::Client::new();
        wait_for_backend_ready_at(
            &client,
            &state,
            &format!("http://{address}/health"),
            3,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        server.await.unwrap();
    }

    #[test]
    #[ignore = "external boundary: set BHARATTERMINAL_REDIS_SERVER_TEST_BINARY"]
    fn redis_connection_recovers_after_isolated_server_restart() {
        let binary = env::var("BHARATTERMINAL_REDIS_SERVER_TEST_BINARY")
            .expect("set BHARATTERMINAL_REDIS_SERVER_TEST_BINARY");
        let port = StdTcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .unwrap()
            .port();
        let test_root = env::temp_dir().join(format!(
            "bharatterminal-redis-restart-{}-{port}",
            std::process::id()
        ));
        fs::create_dir_all(&test_root).unwrap();
        let url = format!("redis://:test-only-credential@127.0.0.1:{port}/");
        let launch = || {
            Command::new(&binary)
                .args([
                    "--port",
                    &port.to_string(),
                    "--bind",
                    "127.0.0.1",
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                    "--requirepass",
                    "test-only-credential",
                    "--dir",
                    test_root.to_str().unwrap(),
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap()
        };

        let mut first_server = launch();
        thread::sleep(Duration::from_millis(300));
        let mut stale_connection = connect_redis_at(&url).unwrap();
        assert_eq!(
            redis::cmd("PING")
                .query::<String>(&mut stale_connection)
                .unwrap(),
            "PONG"
        );
        first_server.kill().unwrap();
        first_server.wait().unwrap();
        assert!(redis::cmd("PING")
            .query::<String>(&mut stale_connection)
            .is_err());

        let mut second_server = launch();
        thread::sleep(Duration::from_millis(300));
        let mut restored_connection = connect_redis_at(&url).unwrap();
        assert_eq!(
            redis::cmd("PING")
                .query::<String>(&mut restored_connection)
                .unwrap(),
            "PONG"
        );
        second_server.kill().unwrap();
        second_server.wait().unwrap();
        fs::remove_dir_all(&test_root).unwrap();
    }
}
