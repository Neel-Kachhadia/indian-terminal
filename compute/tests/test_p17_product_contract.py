from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient

from data import error_logger
from security import credential_manager
from security.log_scrubber import scrub_text


ROOT = Path(__file__).resolve().parents[2]
SENTINELS = {
    "ANGEL_CLIENT_ID": "P17_SENTINEL_CLIENT_7913",
    "ANGEL_PASSWORD": "P17_SENTINEL_PASSWORD_7913",
    "ANGEL_TOTP_SECRET": "P17SENTINELTOTP7913",
    "ANGEL_API_KEY": "P17_SENTINEL_API_KEY_7913",
    "GROQ_API_KEY": "gsk_P17_SENTINEL_GROQ_7913_VALUE",
}


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, key: str, value: str) -> None:
        self.values[(service, key)] = value

    def get_password(self, service: str, key: str) -> str | None:
        return self.values.get((service, key))

    def delete_password(self, service: str, key: str) -> None:
        self.values.pop((service, key), None)


def test_shell_is_initial_ui_and_has_no_login_demo_or_secret_bridge() -> None:
    app = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    header = (ROOT / "src" / "components" / "layout" / "ApexHeader.tsx").read_text(
        encoding="utf-8"
    )
    frontend = app + header
    for forbidden in (
        "WelcomeScreen",
        "CredentialsForm",
        "TRY DEMO",
        "ENTER CREDENTIALS",
        "Switch to Demo",
        "Update Credentials",
        "reset_app",
        "set_demo_mode",
        "setup_full_mode",
        "localStorage",
        "sessionStorage",
        "angelPassword",
        "angelTotpSecret",
        "angelApiKey",
        "groqApiKey",
    ):
        assert forbidden not in frontend
    assert "<ApexHeader" in app
    assert "brokerAuth" in app
    assert "Angel authentication not connected." in app


def test_login_demo_ipc_and_http_surfaces_are_removed() -> None:
    commands = (ROOT / "src-tauri" / "src" / "commands.rs").read_text(encoding="utf-8")
    tauri_main = (ROOT / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
    registered = commands + tauri_main
    for forbidden in (
        "set_demo_mode",
        "setup_full_mode",
        "reset_app",
        '"auth/demo"',
        '"auth/setup"',
        '"auth/reset"',
        '"auth/angel/session"',
        "angel_password: String",
        "angel_totp_secret: String",
        "angel_api_key: String",
    ):
        assert forbidden not in registered
    assert not (ROOT / "src-tauri" / "src" / "auth.rs").exists()


def test_sentinel_secrets_never_leave_bounded_api(monkeypatch) -> None:
    fake = FakeKeyring()
    monkeypatch.setattr(credential_manager, "keyring", fake)
    monkeypatch.setattr(credential_manager, "_require_keyring", lambda: None)
    assert credential_manager.provision_credentials(SENTINELS, replace=True)

    main = importlib.import_module("main")
    main._set_broker_auth_status(
        "DISCONNECTED",
        "Angel credentials unavailable. Run secure credential provisioning.",
    )
    client = TestClient(main.app)
    try:
        headers = {"X-Process-Token": main.PROCESS_TOKEN}
        response = client.get("/broker/auth/status", headers=headers)
        assert response.status_code == 200
        assert set(response.json()) == {"state", "message"}
        serialized = response.text + json.dumps(main.app.openapi(), sort_keys=True)
        assert all(value not in serialized for value in SENTINELS.values())

        paths = set(main.app.openapi()["paths"])
        assert not any("credential" in path.lower() or "secret" in path.lower() for path in paths)
        assert "/auth/angel/session" not in paths
    finally:
        client.close()


def test_bounded_broker_status_never_reads_secret_store(monkeypatch) -> None:
    main = importlib.import_module("main")
    main._set_broker_auth_status("DISCONNECTED", "Angel authentication not connected.")
    monkeypatch.setattr(
        main.credential_manager,
        "status",
        lambda: (_ for _ in ()).throw(AssertionError("GET must not read secrets")),
    )
    client = TestClient(main.app)
    try:
        response = client.get(
            "/broker/auth/status",
            headers={"X-Process-Token": main.PROCESS_TOKEN},
        )
        assert response.status_code == 200
        assert response.json() == {
            "state": "DISCONNECTED",
            "message": "Angel authentication not connected.",
        }
    finally:
        client.close()


def test_internal_feed_bridge_requires_process_token(monkeypatch) -> None:
    main = importlib.import_module("main")

    async def bounded_bridge(websocket) -> None:
        await websocket.accept()
        await websocket.send_json({"state": "CONNECTED"})
        await websocket.close()

    monkeypatch.setattr(main, "_bridge_angel_feed", bounded_bridge)
    client = TestClient(main.app)
    try:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/internal/angel/feed"):
                pass
        with client.websocket_connect(
            "/internal/angel/feed",
            headers={"X-Process-Token": main.PROCESS_TOKEN},
        ) as websocket:
            assert websocket.receive_json() == {"state": "CONNECTED"}
    finally:
        client.close()


def test_angel_bridge_keeps_transport_pings_without_forcing_pong_timeout(monkeypatch) -> None:
    """Angel produces valid data but does not reliably return RFC pong
    frames.  The default websockets ping timeout closed the live feed at
    almost exactly 40 seconds, causing an endless reconnect/auth loop."""
    main = importlib.import_module("main")
    captured: dict = {}

    class FakeRemote:
        pass

    class FakeContext:
        async def __aenter__(self):
            return FakeRemote()

        async def __aexit__(self, *_args):
            return False

    class FakeLocal:
        async def accept(self):
            return None

        async def close(self, **_kwargs):
            return None

    def fake_connect(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeContext()

    async def completed_forward(*_args):
        return None

    monkeypatch.setattr(
        main.credential_manager,
        "get_angel_credentials",
        lambda: {
            "ANGEL_CLIENT_ID": "fixture-client",
            "ANGEL_PASSWORD": "fixture-password",
            "ANGEL_TOTP_SECRET": "fixture-totp",
            "ANGEL_API_KEY": "fixture-api-key",
        },
    )
    monkeypatch.setattr(
        main,
        "_create_angel_session",
        lambda _credentials: {
            "client_id": "fixture-client",
            "api_key": "fixture-api-key",
            "jwt_token": "fixture-jwt",
            "feed_token": "fixture-feed",
        },
    )
    monkeypatch.setattr(main.websockets, "connect", fake_connect)
    monkeypatch.setattr(main, "_forward_local_feed", completed_forward)
    monkeypatch.setattr(main, "_forward_remote_feed", completed_forward)

    asyncio.run(main._bridge_angel_feed(FakeLocal()))

    assert captured["url"] == "wss://smartapisocket.angelone.in/smart-stream"
    assert captured["ping_interval"] == 20
    assert captured["ping_timeout"] is None


def test_sentinel_is_scrubbed_from_persistent_error_log(monkeypatch, tmp_path: Path) -> None:
    database = tmp_path / "errors.db"
    monkeypatch.setattr(error_logger, "ERROR_LOG_PATH", database)
    logger = error_logger.ErrorLogger()
    sentinel = SENTINELS["GROQ_API_KEY"]
    logger.log("p17", "sentinel", f"api_key={sentinel}")
    logger.mark("p17", "2026-08-27", "failed", f"token={sentinel}")
    raw = database.read_bytes()
    assert sentinel.encode() not in raw
    assert sentinel not in scrub_text(f"api_key={sentinel}")


def test_no_broker_mutation_sdk_calls_or_secret_child_arguments() -> None:
    production_files = []
    for path in (ROOT / "compute").rglob("*.py"):
        relative_parts = path.relative_to(ROOT / "compute").parts
        if any(part == "tests" or part.startswith(".") for part in relative_parts):
            continue
        if any(part in {"build", "dist", "__pycache__"} for part in relative_parts):
            continue
        production_files.append(path)
    production = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in production_files
    )
    for forbidden in (
        ".placeOrder(",
        ".modifyOrder(",
        ".cancelOrder(",
        ".gttCreateRule(",
        ".gttModifyRule(",
        ".gttCancelRule(",
    ):
        assert forbidden not in production

    launcher = (ROOT / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
    for forbidden in (
        "ANGEL_CLIENT_ID",
        "ANGEL_PASSWORD",
        "ANGEL_TOTP_SECRET",
        "ANGEL_API_KEY",
        "GROQ_API_KEY",
        "jwt_token",
        "feed_token",
    ):
        assert forbidden not in launcher
    assert 'command.env("BHARATTERMINAL_PROCESS_TOKEN", process_token)' in launcher


def test_produced_textual_artifacts_do_not_contain_sentinels() -> None:
    candidates: list[Path] = []
    for root in (ROOT / "dist", ROOT / "release"):
        if root.is_dir():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    for path in candidates:
        if path.suffix.lower() not in {".css", ".html", ".js", ".json", ".map", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert all(value not in text for value in SENTINELS.values()), path
