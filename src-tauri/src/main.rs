use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
};

use indian_terminal::{commands, models::Tick, state::AppState};
use tauri::Manager;
use tokio::sync::broadcast;

fn main() {
    let (tick_tx, _tick_rx) = broadcast::channel::<Tick>(1024);
    let redis = create_redis_connection();
    let state = AppState::new(redis, tick_tx);

    let run_result = tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .manage(state)
        .setup(|app| {
            let app_state = app.state::<AppState>().inner().clone();
            if let Err(error) = app_state.set_app_handle(app.handle().clone()) {
                println!("state: app handle set failed: {error}");
            }

            let state1 = app.state::<AppState>().inner().clone();
            let state2 = app.state::<AppState>().inner().clone();
            let state3 = app.state::<AppState>().inner().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(error) = indian_terminal::ingester::run(state1).await {
                    eprintln!("ingester: {error}");
                }
            });
            tauri::async_runtime::spawn(async move {
                if let Err(error) = indian_terminal::poller::run(state2).await {
                    eprintln!("poller: {error}");
                }
            });
            tauri::async_runtime::spawn(async move {
                if let Err(error) = indian_terminal::alert_engine::run(state3).await {
                    eprintln!("alert_engine: {error}");
                }
            });
            launch_python_subprocess();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_quote,
            commands::get_chain,
            commands::get_pcr,
            commands::get_ohlcv,
            commands::get_expiries,
            commands::get_chain_greeks,
            commands::get_iv_surface,
            commands::get_payoff,
            commands::get_futures_basis,
            commands::get_screener_results,
            commands::get_indicators,
            commands::set_alert,
            commands::delete_alert,
            commands::get_alerts,
            commands::get_market_status,
            commands::get_instrument_list,
            commands::download_historical
        ])
        .run(tauri::generate_context!());

    if let Err(error) = run_result {
        eprintln!("tauri: {error}");
    }
}

fn create_redis_connection() -> Option<redis::Connection> {
    let client = match redis::Client::open("redis://127.0.0.1/") {
        Ok(client) => client,
        Err(error) => {
            println!("redis: client init failed: {error}");
            return None;
        }
    };

    match client.get_connection() {
        Ok(connection) => {
            println!("redis: reachable");
            Some(connection)
        }
        Err(error) => {
            println!("redis: not reachable: {error}");
            None
        }
    }
}

fn launch_python_subprocess() {
    let python = python_binary_path();
    let compute_dir = compute_dir();

    if !python.exists() {
        println!("python: virtualenv binary missing at {}", python.display());
        return;
    }

    let child = Command::new(&python)
        .arg("-m")
        .arg("uvicorn")
        .arg("main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("8765")
        .current_dir(compute_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(_child) => println!("python: subprocess launched"),
        Err(error) => println!("python: subprocess launch failed: {error}"),
    }
}

fn python_binary_path() -> PathBuf {
    let mut path = compute_dir();
    path.push(".venv");

    if env::consts::OS == "windows" {
        path.push("Scripts");
        path.push("python.exe");
    } else {
        path.push("bin");
        path.push("python");
    }

    path
}

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
