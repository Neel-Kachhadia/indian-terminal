from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests

from data.nse_session import NSESession


@dataclass
class FakeResponse:
    status_code: int


class ScriptedSession:
    def __init__(self, outcomes: list[FakeResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, timeout: int, **_kwargs):
        self.calls.append((url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.recovery
def test_provider_timeout_is_bounded_and_explicitly_unavailable() -> None:
    provider = NSESession()
    scripted = ScriptedSession([FakeResponse(200), requests.Timeout("simulated")])
    provider.session = scripted  # type: ignore[assignment]

    assert provider.get("https://www.nseindia.com/api/option-chain-indices") is None
    assert scripted.calls == [
        ("https://www.nseindia.com", 10),
        ("https://www.nseindia.com/api/option-chain-indices", 10),
    ]


@pytest.mark.recovery
def test_provider_403_refreshes_session_once_then_recovers() -> None:
    provider = NSESession()
    scripted = ScriptedSession(
        [FakeResponse(200), FakeResponse(403), FakeResponse(200), FakeResponse(200)]
    )
    provider.session = scripted  # type: ignore[assignment]

    response = provider.get("https://www.nseindia.com/api/option-chain-indices")
    assert response is not None
    assert response.status_code == 200
    assert len(scripted.calls) == 4
    assert all(timeout == 10 for _, timeout in scripted.calls)
