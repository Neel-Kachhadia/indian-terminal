use std::{collections::BTreeSet, fs, path::Path};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::Value;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::trusted_release_keys::{public_key_hex, ACTIVE_KEY_ID};

pub fn verify_release_resources(root: &Path) -> Result<(), String> {
    let manifest_path = unique_candidate(
        root,
        &["release/resource-manifest.json", "resource-manifest.json"],
    )?;
    let signature_path = unique_candidate(
        root,
        &["release/resource-manifest.sig", "resource-manifest.sig"],
    )?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path).map_err(|_| String::from("release manifest unreadable"))?,
    )
    .map_err(|_| String::from("release manifest invalid"))?;
    let signature: Value = serde_json::from_slice(
        &fs::read(signature_path).map_err(|_| String::from("release signature unreadable"))?,
    )
    .map_err(|_| String::from("release signature invalid"))?;
    if manifest["status"] != "COHERENT"
        || manifest["phase"] != "FINAL"
        || manifest["signature_required"] != true
        || signature["status"] != "SIGNED"
        || signature["algorithm"] != "Ed25519"
        || signature["key_id"] != ACTIVE_KEY_ID
        || manifest["signing_key_id"] != ACTIVE_KEY_ID
    {
        return Err(String::from("release trust metadata rejected"));
    }
    let payload =
        serde_json::to_vec(&manifest).map_err(|_| String::from("manifest encode failed"))?;
    let digest = format!("{:x}", Sha256::digest(&payload));
    if signature["manifest_sha256"].as_str() != Some(digest.as_str()) {
        return Err(String::from("release manifest digest mismatch"));
    }
    let public: [u8; 32] = hex::decode(
        public_key_hex(ACTIVE_KEY_ID)
            .ok_or_else(|| String::from("active release key missing from trusted source"))?,
    )
    .map_err(|_| String::from("pinned release key invalid"))?
    .try_into()
    .map_err(|_| String::from("pinned release key length invalid"))?;
    let raw_signature = BASE64
        .decode(signature["signature_base64"].as_str().unwrap_or_default())
        .map_err(|_| String::from("release signature encoding invalid"))?;
    VerifyingKey::from_bytes(&public)
        .map_err(|_| String::from("pinned release key invalid"))?
        .verify(
            &payload,
            &Signature::from_slice(&raw_signature)
                .map_err(|_| String::from("release signature length invalid"))?,
        )
        .map_err(|_| String::from("release signature verification failed"))?;

    verify_tauri_executable(&manifest)?;

    let entries = manifest["protected_assets"]
        .as_array()
        .ok_or_else(|| String::from("protected asset contract missing"))?;
    let roots = manifest["protected_roots"]
        .as_array()
        .ok_or_else(|| String::from("protected root contract missing"))?;
    let mut expected = BTreeSet::new();
    for entry in entries {
        let relative = safe_relative(entry["runtime_path"].as_str().unwrap_or_default())?;
        if !expected.insert(relative.to_string()) {
            return Err(String::from("duplicate protected asset"));
        }
        let path = root.join(relative);
        let bytes = fs::read(&path).map_err(|_| format!("protected asset missing: {relative}"))?;
        if entry["size"].as_u64() != Some(bytes.len() as u64)
            || entry["sha256"].as_str() != Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
        {
            return Err(format!("protected asset modified: {relative}"));
        }
    }
    require_scope(&expected, roots)?;
    for item in roots {
        let relative = safe_relative(item.as_str().unwrap_or_default())?;
        let directory = root.join(relative);
        if !directory.is_dir() {
            return Err(format!("protected root missing: {relative}"));
        }
        for entry in WalkDir::new(directory).into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() {
                let actual = entry
                    .path()
                    .strip_prefix(root)
                    .map_err(|_| String::from("protected asset outside root"))?
                    .to_string_lossy()
                    .replace('\\', "/");
                if !expected.contains(&actual) {
                    return Err(format!("extra protected asset: {actual}"));
                }
            }
        }
    }
    Ok(())
}

fn verify_tauri_executable(manifest: &Value) -> Result<(), String> {
    let record = manifest["tauri_executable"]
        .as_object()
        .ok_or_else(|| String::from("Tauri executable integrity record missing"))?;
    let executable =
        std::env::current_exe().map_err(|_| String::from("Tauri executable path unavailable"))?;
    if record.get("name").and_then(Value::as_str)
        != executable.file_name().and_then(|name| name.to_str())
    {
        return Err(String::from("Tauri executable identity mismatch"));
    }
    let bytes = fs::read(executable).map_err(|_| String::from("Tauri executable unreadable"))?;
    if record.get("size").and_then(Value::as_u64) != Some(bytes.len() as u64)
        || record.get("sha256").and_then(Value::as_str)
            != Some(
                format!(
                    "{:x}",
                    Sha256::digest(normalize_tauri_bundle_type_marker(&bytes))
                )
                .as_str(),
            )
    {
        return Err(String::from("Tauri executable modified"));
    }
    Ok(())
}

/// `cargo tauri bundle` embeds a placeholder `__TAURI_BUNDLE_TYPE_VAR_UNK` at
/// link time, then patches those trailing 3 bytes in place to record which
/// bundle format actually produced this executable ("NSS" for NSIS --
/// confirmed by diffing a freshly-linked exe against the same exe after
/// `cargo tauri bundle --bundles nsis`; matches that command's own
/// "Patching ... with bundle type information: nsis" log line). Whichever
/// build step last touched the exe determines which variant is on disk when
/// `scripts/release_gate.py` hashes it for the manifest, so this same
/// normalization (canonicalizing to "UNK") has to run on both sides of the
/// hash comparison -- see the matching Python
/// `_normalize_tauri_bundle_type_marker` this must stay byte-for-byte
/// identical to. This targets ONLY this one known, Tauri-owned,
/// fixed-length marker; any other byte change, including this exact field
/// holding some third, unrecognized value, still changes the digest.
fn normalize_tauri_bundle_type_marker(data: &[u8]) -> Vec<u8> {
    const PREFIX: &[u8] = b"__TAURI_BUNDLE_TYPE_VAR_";
    const CANONICAL_SUFFIX: &[u8] = b"UNK";
    // Only these two are legitimate Tauri-owned values for THIS product
    // (we only ever ship NSIS): "UNK" is the pre-patch linker placeholder,
    // "NSS" is what `cargo tauri bundle --bundles nsis` patches it to. Any
    // OTHER 3-byte value at this position -- including a suffix an attacker
    // deliberately set to something outside this set specifically to slip
    // through as "just another bundle-type variant" -- must NOT be
    // normalized away, or this stops being a real integrity check.
    const KNOWN_SUFFIXES: [&[u8]; 2] = [b"UNK", b"NSS"];
    let mut normalized = data.to_vec();
    let mut start = 0usize;
    while let Some(offset) = find_subslice(&normalized[start..], PREFIX) {
        let index = start + offset;
        let suffix_start = index + PREFIX.len();
        let suffix_end = suffix_start + CANONICAL_SUFFIX.len();
        if suffix_end > normalized.len() {
            break;
        }
        if KNOWN_SUFFIXES.contains(&&normalized[suffix_start..suffix_end]) {
            normalized[suffix_start..suffix_end].copy_from_slice(CANONICAL_SUFFIX);
        }
        start = index + PREFIX.len() + CANONICAL_SUFFIX.len();
    }
    normalized
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn unique_candidate(root: &Path, candidates: &[&str]) -> Result<std::path::PathBuf, String> {
    let matches = candidates
        .iter()
        .map(|relative| root.join(relative))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(String::from("release integrity metadata missing")),
        _ => Err(String::from("ambiguous release integrity metadata")),
    }
}

fn safe_relative(value: &str) -> Result<&str, String> {
    if value.is_empty()
        || value.contains('\\')
        || value.starts_with('/')
        || value.split('/').any(|part| part.is_empty() || part == "..")
    {
        return Err(String::from("protected path invalid"));
    }
    Ok(value)
}

fn require_scope(expected: &BTreeSet<String>, roots: &[Value]) -> Result<(), String> {
    let mut scope = expected.clone();
    scope.extend(
        roots
            .iter()
            .filter_map(Value::as_str)
            .map(|value| format!("{}/", value.trim_end_matches('/'))),
    );
    for (name, prefix, executable) in [
        ("sidecar", "compute/dist/", true),
        ("frontend", "dist/", false),
        ("models", "compute/models/", false),
        ("database bootstrap", "bundle-data/", false),
        ("release configuration", "release/", false),
    ] {
        let present = scope.iter().any(|value| {
            value.starts_with(prefix)
                && (!executable || value.to_ascii_lowercase().ends_with(".exe"))
        });
        if !present {
            return Err(format!("protected scope omits {name}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod bundle_type_marker_tests {
    use super::normalize_tauri_bundle_type_marker;
    use sha2::{Digest, Sha256};

    #[test]
    fn unpatched_and_patched_markers_normalize_identically() {
        let unpatched = b"prefix __TAURI_BUNDLE_TYPE_VAR_UNK suffix".to_vec();
        let patched = b"prefix __TAURI_BUNDLE_TYPE_VAR_NSS suffix".to_vec();
        assert_eq!(
            normalize_tauri_bundle_type_marker(&unpatched),
            normalize_tauri_bundle_type_marker(&patched)
        );
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(normalize_tauri_bundle_type_marker(&unpatched))
            ),
            format!(
                "{:x}",
                Sha256::digest(normalize_tauri_bundle_type_marker(&patched))
            )
        );
    }

    #[test]
    fn other_byte_changes_still_change_the_digest() {
        let baseline = b"prefix __TAURI_BUNDLE_TYPE_VAR_UNK suffix".to_vec();
        let tampered_marker = b"prefix __TAURI_BUNDLE_TYPE_VAR_XXX suffix".to_vec();
        let tampered_elsewhere = b"PREFIX __TAURI_BUNDLE_TYPE_VAR_UNK suffix".to_vec();
        assert_ne!(
            normalize_tauri_bundle_type_marker(&baseline),
            normalize_tauri_bundle_type_marker(&tampered_marker)
        );
        assert_ne!(
            normalize_tauri_bundle_type_marker(&baseline),
            normalize_tauri_bundle_type_marker(&tampered_elsewhere)
        );
    }
}
