from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from security import credential_manager
from security.log_scrubber import scrub_payload, scrub_text


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.fail_key: str | None = None

    def set_password(self, service: str, key: str, value: str) -> None:
        if key == self.fail_key:
            self.fail_key = None
            raise RuntimeError("simulated OS store failure")
        self.values[(service, key)] = value

    def get_password(self, service: str, key: str) -> str | None:
        return self.values.get((service, key))

    def delete_password(self, service: str, key: str) -> None:
        self.values.pop((service, key), None)


@pytest.mark.contract
def test_credential_write_rolls_back_on_os_store_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeKeyring()
    monkeypatch.setattr(credential_manager, "keyring", fake)
    monkeypatch.setattr(credential_manager, "_require_keyring", lambda: None)
    fake.values[(credential_manager.SERVICE_PREFIX, "ANGEL_CLIENT_ID")] = "old-client"
    fake.fail_key = "ANGEL_TOTP_SECRET"
    with pytest.raises(RuntimeError, match="simulated"):
        credential_manager.provision_credentials(
            {
                "ANGEL_CLIENT_ID": "new-client",
                "ANGEL_PASSWORD": "password",
                "ANGEL_TOTP_SECRET": "totp-secret",
                "ANGEL_API_KEY": "api-key",
                "GROQ_API_KEY": "gsk_example_not_real",
            },
            replace=True,
        )
    assert fake.values == {(credential_manager.SERVICE_PREFIX, "ANGEL_CLIENT_ID"): "old-client"}


@pytest.mark.contract
def test_secret_scrubber_redacts_nested_payload_and_text() -> None:
    secret = "gsk_abcdefghijklmnop"
    assert secret not in scrub_text(f"api_key={secret}")
    payload = scrub_payload({"angel_password": "plain", "nested": {"jwt": "token"}})
    assert payload == {"angel_password": "[REDACTED]", "nested": {"jwt": "[REDACTED]"}}


@pytest.mark.contract
def test_local_api_auth_origin_size_and_safe_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    main = importlib.import_module("main")
    monkeypatch.setattr(
        main.credential_manager,
        "status",
        lambda: credential_manager.CredentialStatus(
            has_angel_credentials=False,
            has_groq_key=False,
        ),
    )
    client = TestClient(main.app)
    try:
        public = client.get("/health/live")
        assert public.status_code == 200
        assert set(public.json()) == {"status"}
        assert client.get("/health").status_code == 403
        assert client.get("/broker/auth/status").status_code == 403
        headers = {"X-Process-Token": main.PROCESS_TOKEN}
        diagnostic = client.get("/health", headers=headers)
        assert diagnostic.status_code == 200
        assert "components" in diagnostic.json()
        assert client.get(
            "/broker/auth/status",
            headers={**headers, "Host": "evil.example"},
        ).status_code == 400
        assert client.get(
            "/broker/auth/status",
            headers={**headers, "Origin": "https://evil.example"},
        ).status_code == 403
        assert client.post(
            "/greeks",
            headers=headers,
            content=b"x" * (main.MAX_REQUEST_BODY_BYTES + 1),
        ).status_code == 413
        invalid_secret = "should-never-be-echoed"
        response = client.post(
            "/auth/setup",
            headers=headers,
            json={"removed_field": invalid_secret},
        )
        assert response.status_code == 404
        assert invalid_secret not in response.text

        schema_paths = set(main.app.openapi()["paths"])
        assert not schema_paths.intersection(
            {
                "/app/mode",
                "/auth/demo",
                "/auth/setup",
                "/auth/validate",
                "/auth/reset",
                "/auth/angel/session",
                "/credentials",
                "/secrets",
            }
        )

        calls = 0
        original = main.secrets.compare_digest

        def observed_compare(left: str, right: str) -> bool:
            nonlocal calls
            calls += 1
            return original(left, right)

        monkeypatch.setattr(main.secrets, "compare_digest", observed_compare)
        client.get("/broker/auth/status", headers=headers)
        assert calls == 1
    finally:
        client.close()


@pytest.mark.release
def test_no_custom_crypto_or_bundled_secret_source() -> None:
    security_dir = Path(__file__).resolve().parents[1] / "security"
    assert not (security_dir / "obfuscate.py").exists()
    assert not (security_dir / "bundled_groq_key.py").exists()
    groq_source = (Path(__file__).resolve().parents[1] / "intelligence" / "groq_client.py").read_text(encoding="utf-8")
    assert "XOR" not in groq_source
    assert "bundled_groq" not in groq_source
