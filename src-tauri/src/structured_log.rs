use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::Utc;
use serde_json::json;

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;
const LOG_BACKUPS: usize = 5;
static LOG_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn write(level: &str, component: &str, message: &str) {
    let payload = json!({
        "timestamp": Utc::now().to_rfc3339(),
        "level": level,
        "logger": component,
        "event": message,
    });
    let line = payload.to_string();
    eprintln!("{line}");
    let lock = LOG_LOCK.get_or_init(|| Mutex::new(()));
    let Ok(_guard) = lock.lock() else {
        return;
    };
    let Some(path) = log_path() else {
        return;
    };
    if rotate_if_needed(&path, line.len() as u64 + 1).is_err() {
        return;
    }
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

fn log_path() -> Option<PathBuf> {
    let root = std::env::var_os("BHARATTERMINAL_APP_DATA")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("APPDATA").map(|value| PathBuf::from(value).join("BharatTerminal"))
        })?;
    Some(root.join("data").join("logs").join("desktop.jsonl"))
}

fn rotate_if_needed(path: &Path, incoming: u64) -> std::io::Result<()> {
    let size = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    if size.saturating_add(incoming) <= MAX_LOG_BYTES {
        return Ok(());
    }
    for index in (1..=LOG_BACKUPS).rev() {
        let source = if index == 1 {
            path.to_path_buf()
        } else {
            PathBuf::from(format!("{}.{}", path.display(), index - 1))
        };
        let target = PathBuf::from(format!("{}.{}", path.display(), index));
        if source.exists() {
            if target.exists() {
                fs::remove_file(&target)?;
            }
            fs::rename(source, target)?;
        }
    }
    Ok(())
}

#[macro_export]
macro_rules! bt_info {
    ($component:expr, $($arg:tt)*) => {
        $crate::structured_log::write("INFO", $component, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! bt_error {
    ($component:expr, $($arg:tt)*) => {
        $crate::structured_log::write("ERROR", $component, &format!($($arg)*))
    };
}
