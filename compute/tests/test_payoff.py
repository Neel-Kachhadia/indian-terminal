from __future__ import annotations

import datetime as dt

import pytest

import options.payoff as payoff_module
from data.risk_free_rate import RateSelection
from options.payoff import compute_payoff


FUTURE_EXPIRY = (dt.date.today() + dt.timedelta(days=30)).isoformat()


def _fake_rate(rate_decimal: float = 0.065) -> RateSelection:
    today = dt.date.today()
    return RateSelection(
        observation_date=today,
        requested_days=30,
        selected_tenor_days=30,
        selected_tenor_label="30 Days",
        rate_decimal=rate_decimal,
        rate_pct=rate_decimal * 100.0,
        gap_days=0,
        method="NEAREST_NEIGHBOR",
        source="FBIL_TBILL_CURVE",
        source_url="https://www.fbil.org.in/wasdm/tbill/fetch?authenticated=false",
    )


def _leg(**overrides) -> dict:
    leg = {
        "symbol": "NIFTY",
        "strike": 100,
        "expiry": FUTURE_EXPIRY,
        "option_type": "CE",
        "action": "BUY",
        "quantity": 1,
        "entry_price": 5,
        "lot_size": 1,
    }
    leg.update(overrides)
    return leg


def test_long_call_reference_payoff() -> None:
    result = compute_payoff(
        [
            {
                "symbol": "NIFTY",
                "strike": 23_400,
                "expiry": "2026-08-27",
                "option_type": "CE",
                "action": "BUY",
                "quantity": 1,
                "entry_price": 221,
                "lot_size": 75,
            }
        ],
        23_400,
    )

    assert result["breakevens"] == [23_621.0]
    assert result["max_loss"] == -16_575.0
    assert result["max_profit"] is None


@pytest.mark.parametrize("quantity", [0, -1])
def test_payoff_rejects_non_positive_quantity(quantity: int) -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        compute_payoff(
            [
                {
                    "strike": 23_400,
                    "expiry": "2026-08-27",
                    "option_type": "CE",
                    "action": "BUY",
                    "quantity": quantity,
                    "entry_price": 221,
                    "lot_size": 75,
                }
            ],
            23_400,
        )


def test_no_rate_no_dividend_is_unavailable_not_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    """No observed rate, no configured assumption, no dividend: fail closed."""
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: None)
    result = compute_payoff([_leg()], 100)

    assert result["pnl_today"] is None
    assert result["net_greeks"] is None
    assert result["analytical_status"] == "MISSING_RATE"
    assert result["risk_free_rate_source"] is None
    assert result["dividend_yield_source"] is None


def test_observed_rate_is_preferred_over_configured_assumption(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine current FBIL rate wins even if a configured rate is also supplied."""
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate(0.07))
    result = compute_payoff(
        [_leg()],
        100,
        risk_free_rate=0.065,
        risk_free_rate_source="CONFIGURED_ASSUMPTION",
        dividend_yield=0.0,
        dividend_yield_source="CONFIGURED_ASSUMPTION",
    )

    assert result["analytical_status"] == "VALID"
    assert result["risk_free_rate_source"] == "OBSERVED_MARKET_RATE"
    assert result["dividend_yield_source"] == "CONFIGURED_ASSUMPTION"
    assert result["pnl_today"] is not None


def test_configured_assumption_used_only_when_no_observed_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: None)
    result = compute_payoff(
        [_leg()],
        100,
        risk_free_rate=0.065,
        risk_free_rate_source="CONFIGURED_ASSUMPTION",
        dividend_yield=0.01,
        dividend_yield_source="CONFIGURED_ASSUMPTION",
    )

    assert result["risk_free_rate_source"] == "CONFIGURED_ASSUMPTION"
    assert result["analytical_status"] == "VALID"
    assert result["pnl_today"] is not None


def test_unlabelled_rate_is_never_silently_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied rate without the CONFIGURED_ASSUMPTION label must
    never be treated as valid -- there is no third, unlabelled truth tier."""
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: None)
    result = compute_payoff(
        [_leg()],
        100,
        risk_free_rate=0.065,
        risk_free_rate_source=None,
        dividend_yield=0.01,
        dividend_yield_source="CONFIGURED_ASSUMPTION",
    )
    assert result["analytical_status"] == "MISSING_RATE"
    assert result["pnl_today"] is None


def test_missing_dividend_assumption_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dividend has no observed source in this system: must always be an
    explicit configured assumption, or the leg fails closed."""
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate())
    result = compute_payoff([_leg()], 100)

    assert result["analytical_status"] == "MISSING_DIVIDEND_ASSUMPTION"
    assert result["pnl_today"] is None
    assert result["net_greeks"] is None


def test_zero_dividend_assumption_accepted_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_DIVIDEND_ASSUMPTION is an honest label for 'nobody configured a
    dividend, model assumes zero' -- distinct from CONFIGURED_ASSUMPTION,
    which claims somebody actually supplied a value."""
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate())
    result = compute_payoff(
        [_leg()], 100,
        dividend_yield=0.0, dividend_yield_source="ZERO_DIVIDEND_ASSUMPTION",
    )
    assert result["analytical_status"] == "VALID"
    assert result["dividend_yield_source"] == "ZERO_DIVIDEND_ASSUMPTION"
    assert result["pnl_today"] is not None


def test_zero_dividend_assumption_rejected_when_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_DIVIDEND_ASSUMPTION must genuinely mean zero -- a nonzero value
    under that label is not a permitted assumption and fails closed."""
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate())
    result = compute_payoff(
        [_leg()], 100,
        dividend_yield=0.01, dividend_yield_source="ZERO_DIVIDEND_ASSUMPTION",
    )
    assert result["analytical_status"] == "MISSING_DIVIDEND_ASSUMPTION"
    assert result["pnl_today"] is None


def test_dividend_cannot_be_labelled_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate())
    result = compute_payoff(
        [_leg()], 100,
        dividend_yield=0.01, dividend_yield_source="OBSERVED_MARKET_RATE",
    )
    assert result["analytical_status"] == "MISSING_DIVIDEND_ASSUMPTION"


def test_expired_leg_is_unavailable_regardless_of_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate())
    result = compute_payoff(
        [_leg(expiry="2020-01-01")],
        100,
        dividend_yield=0.0, dividend_yield_source="CONFIGURED_ASSUMPTION",
    )
    assert result["analytical_status"] == "EXPIRED_OR_INVALID_EXPIRY"
    assert result["pnl_today"] is None


def test_unsolvable_iv_does_not_use_a_fallback_volatility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: _fake_rate())
    result = compute_payoff(
        [_leg(strike=23_400, expiry=FUTURE_EXPIRY, entry_price=0.01, lot_size=75)],
        23_400,
        dividend_yield=0.0, dividend_yield_source="CONFIGURED_ASSUMPTION",
    )

    assert result["pnl_today"] is None
    assert result["net_greeks"] is None
    assert result["analytical_status"] == "NO_CONVERGENCE"
    # pure intrinsic-value math (no IV involved) must remain unaffected by
    # the failed today-P&L solve
    assert result["max_loss"] == pytest.approx(-0.75, abs=0.01)


def test_missing_lot_size_is_never_silently_defaulted() -> None:
    with pytest.raises(ValueError, match="lot_size is required"):
        compute_payoff(
            [
                {
                    "symbol": "NIFTY",
                    "strike": 23_400,
                    "expiry": "2026-08-27",
                    "option_type": "CE",
                    "action": "BUY",
                    "quantity": 1,
                    "entry_price": 221,
                }
            ],
            23_400,
        )


def test_future_leg_needs_no_rate_or_dividend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(payoff_module, "latest_available_rate", lambda t: None)
    result = compute_payoff(
        [
            {
                "symbol": "NIFTY", "strike": 0, "expiry": FUTURE_EXPIRY,
                "option_type": "FUT", "action": "BUY", "quantity": 1,
                "entry_price": 100, "lot_size": 1,
            }
        ],
        100,
    )
    assert result["analytical_status"] == "VALID"
    assert result["pnl_today"] is not None
    assert result["net_greeks"]["delta"] == 1.0
