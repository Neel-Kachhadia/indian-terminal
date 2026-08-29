from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException

import main


class _Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "{}"

    def json(self) -> dict[str, object]:
        return self._payload


class _Session:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.headers: dict[str, str] = {}
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> _Response:
        self.urls.append(url)
        if "GetQuoteApi" in url:
            return _Response(self.payload)
        return _Response({})


class _OpenCalendar:
    timezone = ZoneInfo("Asia/Kolkata")

    def session_snapshot(self, _instant: dt.datetime) -> dict[str, str]:
        return {"current_state": "OPEN"}


def _payload(*, underlying: str = "BANKNIFTY") -> dict[str, object]:
    timestamp = dt.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%b-%Y %H:%M:%S")
    return {
        "timestamp": timestamp,
        "data": [
            {
                "underlying": underlying,
                "underlyingValue": 55_432.1,
                "strikePrice": 55_500,
                "expiryDate": "03-Sep-2026",
                "optionType": "CE",
                "lastPrice": 123.45,
                "openInterest": 100,
                "changeinOpenInterest": -5,
                "totalTradedVolume": 25,
            },
            {
                "underlying": underlying,
                "underlyingValue": 55_432.1,
                "strikePrice": 55_500,
                "expiryDate": "03-Sep-2026",
                "optionType": "PE",
                "lastPrice": 141.2,
                "openInterest": 120,
                "changeinOpenInterest": 6,
                "totalTradedVolume": 35,
            },
        ],
    }


def test_chain_uses_requested_underlying_and_never_invents_absent_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(_payload())
    monkeypatch.setattr(main._requests, "Session", lambda: session)
    monkeypatch.setattr(main._time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main, "exchange_calendar", lambda: _OpenCalendar())

    result = main.get_chain_from_nse("BANKNIFTY")

    assert result["records"]["underlyingValue"] == 55_432.1
    assert result["provenance"]["symbol"] == "BANKNIFTY"
    assert result["provenance"]["freshness"] == "FRESH"
    assert result["provenance"]["decision_eligible"] is True
    leg = result["records"]["data"][0]["CE"]
    assert leg["bidprice"] is None
    assert leg["askPrice"] is None
    assert leg["impliedVolatility"] is None
    assert not any("allIndices" in url for url in session.urls)


def test_chain_rejects_cross_symbol_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_payload(underlying="NIFTY"))
    monkeypatch.setattr(main._requests, "Session", lambda: session)
    monkeypatch.setattr(main._time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main, "exchange_calendar", lambda: _OpenCalendar())

    with pytest.raises(HTTPException) as raised:
        main.get_chain_from_nse("BANKNIFTY")

    assert raised.value.status_code == 503
    assert "underlying mismatch" in str(raised.value.detail)
