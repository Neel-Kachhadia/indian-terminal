use crate::state::AppState;

pub async fn run(_state: AppState) -> Result<(), String> {
    crate::bt_info!("historical", "not yet implemented");
    Ok(())
}
