// Trusted source, compiled into both release build gate and runtime verifier.
// Generated release files cannot add keys. Rotation requires source review.
pub const ACTIVE_KEY_ID: &str = "e205031af11b9da1";

pub fn public_key_hex(key_id: &str) -> Option<&'static str> {
    match key_id {
        "e205031af11b9da1" => {
            Some("1aa6ed25f36c421b0d308c3996f192818826a42145ffe633a7de41085eec5878")
        }
        _ => None,
    }
}
