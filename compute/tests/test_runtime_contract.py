from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_all_development_launchers_use_canonical_frozen_uv_lock() -> None:
    rust = (ROOT / "src-tauri" / "src" / "main.rs").read_text("utf-8")
    collector = (ROOT / "compute" / "auto_collector" / "launcher.bat").read_text("utf-8")
    assert 'PathBuf::from("uv")' in rust
    assert '.arg("--frozen")' in rust
    assert ".venv" not in rust
    assert "uv run" in collector and "--frozen" in collector
    assert ".venv" not in collector


def test_canonical_runtime_pins_one_duckdb_version() -> None:
    project = (ROOT / "compute" / "pyproject.toml").read_text("utf-8")
    assert project.count('duckdb==1.5.5') == 1


def test_successful_option_polling_does_not_emit_per_poll_log_spam() -> None:
    python_api = (ROOT / "compute" / "main.py").read_text("utf-8")
    rust_poller = (ROOT / "src-tauri" / "src" / "poller.rs").read_text("utf-8")
    assert 'print(f"greeks: computed' not in python_api
    assert "chain fetched for NIFTY" not in rust_poller
    assert "if !success_logged" in rust_poller
    assert "chain ready for NIFTY" in rust_poller
