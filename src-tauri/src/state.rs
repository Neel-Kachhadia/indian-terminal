use crate::models::Tick;
use std::sync::{Arc, Mutex as StdMutex};
use tauri::AppHandle;
use tokio::sync::{broadcast, Mutex};

#[derive(Clone)]
pub struct AppState {
    pub app_handle: Arc<StdMutex<Option<AppHandle>>>,
    pub redis: Arc<Mutex<Option<redis::Connection>>>,
    pub tick_tx: Arc<Mutex<broadcast::Sender<Tick>>>,
}

impl AppState {
    pub fn new(redis: Option<redis::Connection>, tick_tx: broadcast::Sender<Tick>) -> Self {
        Self {
            app_handle: Arc::new(StdMutex::new(None)),
            redis: Arc::new(Mutex::new(redis)),
            tick_tx: Arc::new(Mutex::new(tick_tx)),
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
                println!("state: app handle lock failed: {error}");
                None
            }
        }
    }
}
