from __future__ import annotations

import options.greeks as greeks
from py_vollib.black_scholes import black_scholes


def test_scalar_greeks_recover_known_volatility_without_optional_jit_runtime(
    monkeypatch,
) -> None:
    time_to_expiry = 0.1
    volatility = 0.2
    price = black_scholes(
        "c",
        100.0,
        100.0,
        time_to_expiry,
        greeks.RISK_FREE_RATE,
        volatility,
    )
    monkeypatch.setattr(greeks, "_time_to_expiry", lambda _expiry: time_to_expiry)

    result = greeks.compute_chain_greeks(
        [{"option_type": "CE", "strike": 100.0, "ltp": price}],
        100.0,
        "01-Jan-2030",
    )

    assert abs(result[0]["iv"] - volatility) < 1e-8
    assert 0.4 <= result[0]["delta"] <= 0.6
    assert result[0]["gamma"] > 0.0
    assert result[0]["theta"] < 0.0
    assert result[0]["vega"] > 0.0
