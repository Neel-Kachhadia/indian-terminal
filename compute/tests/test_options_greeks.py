from __future__ import annotations

import datetime as dt

import pytest
from py_vollib.black import black

import options.greeks as greeks
from data.risk_free_rate import (
    FBIL_QUOTE_CONVENTION,
    FBIL_SOURCE_ID,
    FBIL_TBILL_LATEST_URL,
    OPTION_RATE_CONVENTION,
    RATE_CONVERSION_METHOD,
    RateSelection,
)
from options.iv_contract import IST, validated_live_observation


UTC = dt.timezone.utc


def _fixture(
    now: dt.datetime,
) -> tuple[list[dict[str, object]], str, RateSelection, dict[str, object]]:
    observed = now - dt.timedelta(seconds=10)
    expiry = observed.astimezone(IST).date() + dt.timedelta(days=30)
    expiry_at = dt.datetime.combine(expiry, dt.time(15, 30), IST).astimezone(UTC)
    tenor = (expiry_at - observed).total_seconds() / (365 * 86400)
    rate = RateSelection(
        observation_date=observed.astimezone(IST).date() - dt.timedelta(days=1),
        requested_days=30,
        selected_tenor_days=30,
        selected_tenor_label="30 Days",
        rate_decimal=0.052,
        rate_pct=5.237,
        gap_days=0,
        method="NEAREST_NEIGHBOR",
        source=FBIL_SOURCE_ID,
        source_url=FBIL_TBILL_LATEST_URL,
        published_at=observed - dt.timedelta(days=1),
        retrieved_at=observed - dt.timedelta(seconds=1),
        quote_convention=FBIL_QUOTE_CONVENTION,
        rate_convention=OPTION_RATE_CONVENTION,
        conversion_method=RATE_CONVERSION_METHOD,
    )
    forward, strike, sigma = 101.0, 100.0, 0.23
    call = black("c", forward, strike, tenor, rate.rate_decimal, sigma)
    put = black("p", forward, strike, tenor, rate.rate_decimal, sigma)
    legs = [
        {
            "symbol": "NIFTY", "expiry": expiry.isoformat(), "option_type": "CE",
            "strike": strike, "bid": call - 0.01, "ask": call + 0.01,
            "volume": 100, "oi": 200,
        },
        {
            "symbol": "NIFTY", "expiry": expiry.isoformat(), "option_type": "PE",
            "strike": strike, "bid": put - 0.01, "ask": put + 0.01,
            "volume": 100, "oi": 200,
        },
    ]
    provenance = {
        "source": "NSE_GET_SYMBOL_DERIVATIVES_DATA",
        "observed_at": observed.isoformat(),
        "source_timestamp": observed.isoformat(),
        "spot_timestamp": observed.isoformat(),
        "spot_source": "NSE_GET_SYMBOL_DERIVATIVES_DATA",
        "spot_quality": "OBSERVED_VERIFIED",
        "decision_eligible": True,
    }
    return legs, expiry.strftime("%d-%b-%Y"), rate, provenance


def test_live_greeks_require_pair_and_recover_black76_iv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = dt.datetime.now(UTC).replace(microsecond=0)
    legs, expiry, rate, provenance = _fixture(now)
    monkeypatch.setattr(greeks, "latest_available_rate", lambda *args, **kwargs: rate)
    result = greeks.compute_chain_greeks(
        legs, 100.0, expiry, provenance=provenance, now=now
    )
    assert [leg["calculation_status"] for leg in result] == ["VALID", "VALID"]
    assert result[0]["iv"] == pytest.approx(0.23, abs=1e-8)
    assert result[1]["iv"] == pytest.approx(0.23, abs=1e-8)
    assert result[0]["delta"] > 0 and result[1]["delta"] < 0
    assert validated_live_observation(result[0], as_of=now) is not None


def test_live_greeks_missing_rate_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = dt.datetime.now(UTC).replace(microsecond=0)
    legs, expiry, _, provenance = _fixture(now)
    monkeypatch.setattr(greeks, "latest_available_rate", lambda *args, **kwargs: None)
    result = greeks.compute_chain_greeks(
        legs, 100.0, expiry, provenance=provenance, now=now
    )
    assert all(
        leg["calculation_status"] == "MISSING_RATE" and leg["iv"] is None
        for leg in result
    )


def test_live_greeks_reject_stale_source_before_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = dt.datetime.now(UTC).replace(microsecond=0)
    legs, expiry, rate, provenance = _fixture(now)
    stale = now - dt.timedelta(seconds=181)
    provenance.update(
        {
            "observed_at": stale.isoformat(),
            "source_timestamp": stale.isoformat(),
            "spot_timestamp": stale.isoformat(),
        }
    )
    monkeypatch.setattr(greeks, "latest_available_rate", lambda *args, **kwargs: rate)
    result = greeks.compute_chain_greeks(
        legs, 100.0, expiry, provenance=provenance, now=now
    )
    assert all(
        leg["calculation_status"] == "STALE" and leg["iv"] is None
        for leg in result
    )


def test_live_greeks_bad_spot_and_non_executable_price_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = dt.datetime.now(UTC).replace(microsecond=0)
    legs, expiry, rate, provenance = _fixture(now)
    bad_spot = greeks.compute_chain_greeks(
        legs, 0.0, expiry, provenance=provenance, now=now
    )
    assert all(leg["calculation_status"] == "BAD_SPOT" for leg in bad_spot)
    monkeypatch.setattr(greeks, "latest_available_rate", lambda *args, **kwargs: rate)
    crossed = [dict(legs[0], bid=10.0, ask=9.0), legs[1]]
    unavailable = greeks.compute_chain_greeks(
        crossed, 100.0, expiry, provenance=provenance, now=now
    )
    assert all(
        leg["calculation_status"] == "NON_EXECUTABLE_PRICE"
        for leg in unavailable
    )


def test_live_greeks_expired_contract_is_explicit() -> None:
    now = dt.datetime.now(UTC).replace(microsecond=0)
    observed = now - dt.timedelta(seconds=10)
    provenance = {
        "source": "NSE",
        "observed_at": observed.isoformat(),
        "source_timestamp": observed.isoformat(),
        "spot_timestamp": observed.isoformat(),
        "spot_source": "NSE",
        "spot_quality": "OBSERVED_VERIFIED",
        "decision_eligible": True,
    }
    expiry = (observed.astimezone(IST).date() - dt.timedelta(days=1)).isoformat()
    result = greeks.compute_chain_greeks(
        [
            {
                "symbol": "NIFTY", "expiry": expiry, "option_type": "CE",
                "strike": 100.0, "bid": 1.0, "ask": 1.1,
            }
        ],
        100.0,
        expiry,
        provenance=provenance,
        now=now,
    )
    assert result[0]["calculation_status"] == "EXPIRED"
