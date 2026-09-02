use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

#[path = "src/trusted_release_keys.rs"]
mod trusted_release_keys;

fn main() {
    #[cfg(target_os = "windows")]
    println!(
        "cargo:rustc-link-arg=/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'"
    );
    println!("cargo:rerun-if-changed=../release/resource-manifest.json");
    println!("cargo:rerun-if-changed=../release/resource-manifest.sig");
    println!("cargo:rerun-if-changed=src/trusted_release_keys.rs");
    println!("cargo:rerun-if-changed=../compute");
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=src");
    if env::var("PROFILE").as_deref() == Ok("release") {
        verify_coherent_release();
    }
    tauri_build::build()
}

fn verify_coherent_release() {
    let root = PathBuf::from("..");
    let manifest_path = root.join("release/resource-manifest.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("release manifest missing: {error}")),
    )
    .unwrap_or_else(|error| panic!("release manifest invalid: {error}"));
    assert_eq!(
        manifest["status"], "COHERENT",
        "release manifest is not coherent"
    );
    assert_eq!(
        env::var("BHARATTERMINAL_RELEASE_PHASE").as_deref(),
        Ok("PREBUNDLE"),
        "release compilation requires the explicit PREBUNDLE phase"
    );
    assert_eq!(
        manifest["phase"], "PREBUNDLE",
        "release compilation requires a signed prebundle manifest"
    );
    verify_manifest_signature(&root, &manifest);
    let expected = manifest["source_sha256"]
        .as_str()
        .expect("release manifest source_sha256 missing");
    let actual = source_fingerprint(&root);
    assert_eq!(
        expected, actual,
        "release source changed after artifacts were built"
    );
}

fn verify_manifest_signature(root: &Path, manifest: &serde_json::Value) {
    let signature: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("release/resource-manifest.sig"))
            .unwrap_or_else(|error| panic!("release manifest signature missing: {error}")),
    )
    .expect("release manifest signature invalid");
    assert_eq!(
        signature["status"], "SIGNED",
        "release manifest is unsigned"
    );
    assert_eq!(
        signature["algorithm"], "Ed25519",
        "unsupported release signature"
    );
    assert_eq!(
        signature["key_id"],
        trusted_release_keys::ACTIVE_KEY_ID,
        "release manifest signing key is untrusted"
    );
    assert_eq!(
        manifest["signing_key_id"],
        trusted_release_keys::ACTIVE_KEY_ID,
        "release manifest key id is untrusted"
    );
    let payload = serde_json::to_vec(manifest).expect("release manifest canonicalization failed");
    let expected_digest = format!("{:x}", Sha256::digest(&payload));
    assert_eq!(
        signature["manifest_sha256"].as_str(),
        Some(expected_digest.as_str()),
        "release manifest signed digest mismatch"
    );
    let public_bytes = hex::decode(
        trusted_release_keys::public_key_hex(trusted_release_keys::ACTIVE_KEY_ID)
            .expect("active release key missing from trusted source"),
    )
    .expect("release public key is not hexadecimal");
    let public_array: [u8; 32] = public_bytes
        .try_into()
        .expect("release public key must be 32 bytes");
    let verifying_key =
        VerifyingKey::from_bytes(&public_array).expect("release public key invalid");
    let signature_bytes = BASE64
        .decode(
            signature["signature_base64"]
                .as_str()
                .expect("release signature missing"),
        )
        .expect("release signature base64 invalid");
    let signature =
        Signature::from_slice(&signature_bytes).expect("release signature length invalid");
    verifying_key
        .verify(&payload, &signature)
        .expect("release manifest signature verification failed");
}

// Must stay byte-for-byte equivalent to canonicalize_source_bytes() in
// scripts/release_gate.py: the signed source fingerprint must depend only on
// committed source CONTENT, never on which line-ending bytes a given
// checkout happens to carry on disk. This repo's own .gitattributes already
// declares `* text=auto eol=lf`, but that attribute only governs what git
// WRITES on checkout/checkin -- it does not retroactively rewrite bytes
// already sitting in a working tree, so raw fs::read() bytes can still carry
// CRLF on a Windows checkout even for files git considers LF-normalized.
fn canonicalize_source_bytes(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'\r' && data.get(i + 1) == Some(&b'\n') {
            i += 1;
            continue;
        }
        out.push(data[i]);
        i += 1;
    }
    out
}

fn source_fingerprint(root: &Path) -> String {
    let excluded = [
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        ".venv-clean",
        ".venv-release",
        "__pycache__",
        "build",
        "dist",
        "logs",
        "target",
        "node_modules",
        "saved_models",
    ];
    let mut files = BTreeSet::new();
    for base in [
        "compute",
        "contracts",
        "public",
        "scripts",
        "src",
        "src-tauri/src",
    ] {
        for entry in WalkDir::new(root.join(base))
            .into_iter()
            .filter_entry(|entry| {
                if !entry.file_type().is_dir() {
                    return true;
                }
                let name = entry.file_name().to_string_lossy();
                // Must match scripts/release_gate.py's source_files() exactly:
                // this Rust reimplementation is independently re-verified at
                // compile time (verify_coherent_release, above) against the
                // source_sha256 Python computed and signed into the PREBUNDLE
                // manifest -- any divergence in which directories get walked
                // makes the two hashes permanently, deterministically unequal
                // regardless of whether the source actually changed, since
                // both sides are otherwise byte-for-byte identical (same
                // base dirs, same explicit exclusion names, same digest
                // framing). Python excludes ANY directory name starting with
                // "." or "_" as a blanket rule; this used to only exclude
                // three specific dot-prefixes, so any other dot/underscore
                // directory under a scanned base (e.g. a `.route-audit`
                // scratch dir) silently diverged the two fingerprints and
                // failed `cargo tauri build`'s coherence assertion on every
                // single build, misreported as "release source changed
                // after artifacts were built" even with zero real drift.
                !excluded.contains(&name.as_ref())
                    && !name.starts_with('.')
                    && !name.starts_with('_')
            })
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let relative = path.strip_prefix(root).expect("source path outside root");
            if relative
                .components()
                .any(|part| excluded.contains(&part.as_os_str().to_string_lossy().as_ref()))
            {
                continue;
            }
            files.insert(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    for relative in [
        "package.json",
        "pnpm-lock.yaml",
        ".gitignore",
        "index.html",
        "tailwind.config.ts",
        "tsconfig.json",
        "vite.config.ts",
        "src-tauri/Cargo.toml",
        "src-tauri/Cargo.lock",
        "src-tauri/tauri.conf.json",
        "src-tauri/build.rs",
        "src-tauri/installer-hooks.nsh",
        "build_release.ps1",
        "release/release-spec.json",
        "release/required-api.json",
    ] {
        files.insert(relative.to_string());
    }

    let mut digest = Sha256::new();
    for relative in files {
        let path_bytes = relative.as_bytes();
        digest.update((path_bytes.len() as u32).to_be_bytes());
        digest.update(path_bytes);
        let raw = fs::read(root.join(&relative)).expect("source file unreadable");
        digest.update(Sha256::digest(canonicalize_source_bytes(&raw)));
    }
    format!("{:x}", digest.finalize())
}
