"""Live option-chain IV/Greeks with source-preserving fail-closed inputs."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from py_vollib.black.greeks.analytical import delta, gamma, theta, vega

from data.risk_free_rate import latest_available_rate
from options.iv_contract import (
    CONTRACT_VERSION,
    IST,
    CalculationStatus,
    canonical_observation_id,
    timestamp,
    validate_observation,
)
from options.iv_pricing import PricingModel, SolveStatus, option_flag, solve_iv_with_provenance


MAX_LIVE_AGE_SECONDS = 180
MIN_QUOTE_PRICE = 0.0
MIN_IV = 0.01
MAX_IV = 5.0
PRICE_SEMANTIC = "LIVE_TOP_OF_BOOK_BID_ASK_MIDPOINT"
FORWARD_SEMANTIC = "PUT_CALL_PARITY_SAME_STRIKE_EXECUTABLE_MIDS"


def compute_chain_greeks(
    legs: list[dict[str, Any]],
    spot: float,
    expiry_str: str,
    *,
    provenance: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Compute Black-76 IV only from synchronized executable CE/PE quotes.

    Forward comes from same-strike put-call parity. Missing/crossed quotes,
    stale source, missing rate, identity mismatch, or solver failure leaves IV
    and every Greek `None` with explicit status. NSE-reported IV is ignored.
    """
    output = [_with_empty_greeks(leg) for leg in legs]
    current = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    spot_value = _positive(spot)
    if spot_value is None:
        return _unavailable(output, CalculationStatus.BAD_SPOT)
    if not isinstance(provenance, dict) or provenance.get("decision_eligible") is not True:
        return _unavailable(output, CalculationStatus.SOURCE_UNVERIFIED)
    observed = timestamp(provenance.get("observed_at"))
    source_timestamp = timestamp(provenance.get("source_timestamp") or provenance.get("observed_at"))
    spot_timestamp = timestamp(provenance.get("spot_timestamp") or provenance.get("observed_at"))
    source = str(provenance.get("source") or "").strip().upper()
    spot_source = str(provenance.get("spot_source") or source).strip().upper()
    spot_quality = str(provenance.get("spot_quality") or "").strip().upper()
    if (
        observed is None or source_timestamp != observed or spot_timestamp != observed
        or source in {"", "UNKNOWN", "NONE"} or spot_source in {"", "UNKNOWN", "NONE"}
        or spot_quality != "OBSERVED_VERIFIED"
    ):
        return _unavailable(output, CalculationStatus.SOURCE_UNVERIFIED)
    age = (current - observed).total_seconds()
    if not 0 <= age <= MAX_LIVE_AGE_SECONDS:
        return _unavailable(output, CalculationStatus.STALE)
    try:
        expiry_day = _expiry_date(expiry_str)
    except ValueError:
        return _unavailable(output, CalculationStatus.INSUFFICIENT_INPUT)
    expiry_at = dt.datetime.combine(expiry_day, dt.time(15, 30), IST).astimezone(dt.timezone.utc)
    time_to_expiry = (expiry_at - observed).total_seconds() / (365 * 86400)
    if time_to_expiry <= 0:
        return _unavailable(output, CalculationStatus.EXPIRED)
    try:
        rate = latest_available_rate(time_to_expiry, as_of=observed)
    except Exception:
        rate = None
    if rate is None or rate.rate_convention != "CONTINUOUS_ACT_365" or rate.retrieved_at is None:
        return _unavailable(output, CalculationStatus.MISSING_RATE)

    pairs = _paired_mid_quotes(output)
    for index, leg in enumerate(output):
        strike = _positive(leg.get("strike"))
        option_type = str(leg.get("option_type") or "").strip().upper()
        if strike is None or option_type not in {"CE", "PE"}:
            leg["calculation_status"] = CalculationStatus.INSUFFICIENT_INPUT.value
            continue
        pair = pairs.get(strike)
        if pair is None or option_type not in pair:
            leg["calculation_status"] = CalculationStatus.NON_EXECUTABLE_PRICE.value
            leg["calculation_reason"] = "same-strike executable CE/PE pair required"
            continue
        option_price = pair[option_type]
        forward = strike + math.exp(rate.rate_decimal * time_to_expiry) * (pair["CE"] - pair["PE"])
        if not math.isfinite(forward) or forward <= 0:
            leg["calculation_status"] = CalculationStatus.SOURCE_UNVERIFIED.value
            leg["calculation_reason"] = "put-call parity forward invalid"
            continue
        solved = solve_iv_with_provenance(
            option_flag(option_type), option_price, spot_value, strike, time_to_expiry,
            rate.rate_decimal, pricing_model=PricingModel.BLACK_76,
            dividend_yield=None, forward_price=forward,
        )
        if solved.status is not SolveStatus.OK:
            leg["calculation_status"] = _solver_status(solved.status).value
            leg["calculation_reason"] = solved.reason
            continue
        sigma = float(solved.iv)
        try:
            greek_values = {
                "iv": sigma,
                "delta": float(delta(option_flag(option_type), forward, strike, time_to_expiry, rate.rate_decimal, sigma)),
                "gamma": float(gamma(option_flag(option_type), forward, strike, time_to_expiry, rate.rate_decimal, sigma)),
                "theta": float(theta(option_flag(option_type), forward, strike, time_to_expiry, rate.rate_decimal, sigma)),
                "vega": float(vega(option_flag(option_type), forward, strike, time_to_expiry, rate.rate_decimal, sigma)),
            }
        except Exception as error:
            leg["calculation_status"] = CalculationStatus.NO_CONVERGENCE.value
            leg["calculation_reason"] = f"greeks:{type(error).__name__}"
            continue
        if not _valid_values(greek_values, option_type):
            leg["calculation_status"] = CalculationStatus.NO_CONVERGENCE.value
            leg["calculation_reason"] = "non-finite or out-of-domain Greek"
            continue
        row = _observation_row(
            leg, spot_value, expiry_day, observed, current, source, spot_source,
            option_price, forward, rate, time_to_expiry, solved,
        )
        try:
            validate_observation(row)
            if (
                str(leg.get("symbol") or "").strip().upper() != row["symbol"]
                or _expiry_date(str(leg.get("expiry") or "")) != expiry_day
            ):
                raise ValueError("contract identity mismatch")
        except (TypeError, ValueError, OverflowError) as error:
            leg["calculation_status"] = CalculationStatus.SOURCE_UNVERIFIED.value
            leg["calculation_reason"] = str(error)
            continue
        leg.update(greek_values)
        leg["iv_observation"] = row
        leg["pricing_model"] = PricingModel.BLACK_76.value
        leg["calculation_status"] = CalculationStatus.VALID.value
    return output


def _observation_row(
    leg: dict[str, Any],
    spot: float,
    expiry: dt.date,
    observed: dt.datetime,
    created: dt.datetime,
    source: str,
    spot_source: str,
    option_price: float,
    forward: float,
    rate: Any,
    time_to_expiry: float,
    solved: Any,
) -> dict[str, Any]:
    symbol = str(leg.get("symbol") or "").strip().upper()
    row = {
        "observation_id": "",
        "symbol": symbol,
        "underlying": symbol,
        "expiry": expiry.isoformat(),
        "strike": float(leg["strike"]),
        "option_type": str(leg["option_type"]).strip().upper(),
        "date": observed.astimezone(IST).date().isoformat(),
        "observed_at": observed.isoformat(),
        "source_timestamp": observed.isoformat(),
        "available_at": created.isoformat(),
        "iv": solved.iv,
        "contract_token": leg.get("contract_token"),
        "option_price": option_price,
        "option_price_semantic": PRICE_SEMANTIC,
        "option_price_source": source,
        "option_price_quality": "EXECUTABLE_BID_ASK_MIDPOINT",
        "spot_price": spot,
        "spot_source": spot_source,
        "spot_timestamp": observed.isoformat(),
        "spot_quality": "OBSERVED_VERIFIED",
        "forward_price": forward,
        "forward_source": f"{FORWARD_SEMANTIC}:{source}",
        "forward_timestamp": observed.isoformat(),
        "forward_quality": "DERIVED_FROM_EXECUTABLE_QUOTES",
        "risk_free_rate": rate.rate_decimal,
        "risk_free_tenor_days": rate.selected_tenor_days,
        "risk_free_source": rate.source,
        "risk_free_source_url": rate.source_url,
        "risk_free_observation_date": rate.observation_date.isoformat(),
        "risk_free_published_at": rate.published_at.isoformat() if rate.published_at else None,
        "risk_free_retrieved_at": rate.retrieved_at.isoformat(),
        "risk_free_convention": rate.rate_convention,
        "time_to_expiry_years": time_to_expiry,
        "dividend_yield": None,
        "dividend_source": None,
        "pricing_model": PricingModel.BLACK_76.value,
        "solver": solved.solver,
        "solver_version": solved.solver_version,
        "repriced_price": solved.repriced_price,
        "repricing_abs_error": solved.abs_error,
        "repricing_rel_error": solved.rel_error,
        "volume": _integer(leg.get("volume")),
        "oi": _integer(leg.get("oi")),
        "source": f"{source}+{rate.source}",
        "quality": "IV_VALID",
        "calculation_status": CalculationStatus.VALID.value,
        "reason": None,
        "observation_kind": "LIVE",
        "created_at": created.isoformat(),
        "calculation_version": solved.calculation_version,
        "contract_version": CONTRACT_VERSION,
    }
    row["observation_id"] = canonical_observation_id(row)
    return row


def _paired_mid_quotes(legs: list[dict[str, Any]]) -> dict[float, dict[str, float]]:
    pairs: dict[float, dict[str, float]] = {}
    for leg in legs:
        strike = _positive(leg.get("strike"))
        option_type = str(leg.get("option_type") or "").strip().upper()
        bid = _positive(leg.get("bid"))
        ask = _positive(leg.get("ask"))
        if strike is None or option_type not in {"CE", "PE"} or bid is None or ask is None or ask < bid:
            continue
        pairs.setdefault(strike, {})[option_type] = (bid + ask) / 2.0
    return {strike: pair for strike, pair in pairs.items() if set(pair) == {"CE", "PE"}}


def _with_empty_greeks(leg: dict[str, Any]) -> dict[str, Any]:
    result = dict(leg)
    result.pop("iv_observation", None)
    result.update({
        "iv": None, "delta": None, "gamma": None, "theta": None, "vega": None,
        "pricing_model": None,
        "calculation_status": CalculationStatus.NON_EXECUTABLE_PRICE.value,
    })
    return result


def _unavailable(legs: list[dict[str, Any]], status: CalculationStatus) -> list[dict[str, Any]]:
    for leg in legs:
        leg["calculation_status"] = status.value
    return legs


def _solver_status(status: SolveStatus) -> CalculationStatus:
    if status is SolveStatus.ARBITRAGE_REJECTED:
        return CalculationStatus.ARBITRAGE_VIOLATION
    if status is SolveStatus.OUT_OF_IV_BOUNDS:
        return CalculationStatus.BAD_OPTION_PRICE
    return CalculationStatus.NO_CONVERGENCE


def _expiry_date(value: str) -> dt.date:
    text = str(value or "").strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError("invalid expiry")


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > MIN_QUOTE_PRICE else None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _valid_values(values: dict[str, float], option_type: str) -> bool:
    if any(not math.isfinite(value) for value in values.values()):
        return False
    if not MIN_IV <= values["iv"] <= MAX_IV or values["gamma"] < 0 or values["vega"] <= 0:
        return False
    if option_type == "CE" and not 0 <= values["delta"] <= 1:
        return False
    if option_type == "PE" and not -1 <= values["delta"] <= 0:
        return False
    return True
