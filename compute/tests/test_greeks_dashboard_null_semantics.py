"""A missing per-leg Greek/LTP must not be coalesced into a fabricated 0.0
contribution to the NET risk figure -- that would silently understate real
exposure on a risk dashboard. See compute/risk/greeks_dashboard.py.
"""

from __future__ import annotations

from risk.greeks_dashboard import compute_greeks_dashboard


def _leg(strike: float, option_type: str, **fields) -> dict:
    base = {"strike": strike, "delta": 0.5, "gamma": 0.01, "theta": -1.0, "vega": 2.0, "ltp": 10.0}
    base["option_type"] = option_type
    base.update(fields)
    return base


def test_complete_legs_produce_a_real_net_greek_and_premium() -> None:
    legs = [
        _leg(100, "CE"),
        _leg(100, "PE"),
    ]
    result = compute_greeks_dashboard("TEST", legs, spot=100.0)
    straddle = result["templates"]["atm_straddle"]
    assert straddle["net_greeks"]["delta"] == 1.0
    assert straddle["data_complete"] is True
    assert straddle["theta_decay_today"] is not None
    assert straddle["pnl_if_spot_moves"]


def test_missing_leg_delta_makes_net_delta_none_not_zero() -> None:
    legs = [
        _leg(100, "CE", delta=None),
        _leg(100, "PE"),
    ]
    result = compute_greeks_dashboard("TEST", legs, spot=100.0)
    straddle = result["templates"]["atm_straddle"]
    # Old (buggy) behavior would have silently produced delta=0.5 (only the
    # PE leg's real value, CE's None coalesced to 0.0) -- a fabricated,
    # misleadingly-precise net figure. Correct behavior: unavailable.
    assert straddle["net_greeks"]["delta"] is None
    # Other Greeks were fully present and must still be reported normally
    # (both legs side=+1, gamma=0.01 each -> net 0.02).
    assert straddle["net_greeks"]["gamma"] == 0.02
    assert straddle["data_complete"] is False


def test_missing_leg_ltp_makes_premium_and_pnl_unavailable() -> None:
    legs = [
        _leg(100, "CE", ltp=None),
        _leg(100, "PE"),
    ]
    result = compute_greeks_dashboard("TEST", legs, spot=100.0)
    straddle = result["templates"]["atm_straddle"]
    assert straddle["pnl_if_spot_moves"] == {}
    assert straddle["break_even_range"] == []
    assert straddle["data_complete"] is False
    # Greeks unaffected by a missing LTP.
    assert straddle["net_greeks"]["delta"] == 1.0
