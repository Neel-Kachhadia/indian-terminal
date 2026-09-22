"""Shared, provenance-producing Black-Scholes IV solver.

Single implementation used by both the live option-chain path
(options/greeks.py) and the historical IV backfill
(options/iv_backfill.py) -- spec requirement: no second, inconsistent
live-IV implementation. Wraps py_vollib (already an existing project
dependency) rather than reimplementing numerical finance code.

Every candidate is checked against explicit no-arbitrage bounds BEFORE the
solver runs, and every solved IV is validated by repricing it and comparing
against the input price, so a persisted row is always reproducible from its
own stored inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from py_vollib.black import black as _black76_price
from py_vollib.black_scholes import black_scholes as _bs_price
from py_vollib.black_scholes_merton import black_scholes_merton as _bsm_price
from scipy.optimize import brentq


SOLVER_NAME = "scipy.brentq+py_vollib.pricing_oracle"
try:
    from importlib.metadata import version as _pkg_version

    SOLVER_VERSION = f"py_vollib=={_pkg_version('py_vollib')}"
except Exception:  # pragma: no cover - defensive only
    SOLVER_VERSION = "py_vollib==unknown"

CALCULATION_VERSION = "iv-pricing-v3"
MAX_ITERATIONS = 128
VOLATILITY_TOLERANCE = 1e-12

MIN_IV = 1e-6
MAX_IV = 5.0
# Repricing tolerance: a solved IV must reprice back to within this relative
# (or, for very cheap options, absolute) error of the original input price.
# Chosen as a conservative solver-health check, not a market-fit target --
# py_vollib's rational solver is accurate to numerical precision for any
# genuinely arbitrage-free input, so a real miss here means the input was
# pathological, not that the tolerance is loose.
MAX_REPRICING_RELATIVE_ERROR = 1e-9
MAX_REPRICING_ABSOLUTE_ERROR = 1e-8


class ArbitrageReason(StrEnum):
    INVALID_OPTION_TYPE = "INVALID_OPTION_TYPE"
    INVALID_SPOT = "INVALID_SPOT"
    INVALID_STRIKE = "INVALID_STRIKE"
    INVALID_RATE = "INVALID_RATE"
    INVALID_T = "INVALID_T"
    PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE"
    PRICE_BELOW_INTRINSIC = "PRICE_BELOW_INTRINSIC"
    PRICE_ABOVE_BOUND = "PRICE_ABOVE_BOUND"


class SolveStatus(StrEnum):
    OK = "OK"
    ARBITRAGE_REJECTED = "ARBITRAGE_REJECTED"
    NO_IV_SOLUTION = "NO_IV_SOLUTION"
    REPRICING_FAILED = "REPRICING_FAILED"
    OUT_OF_IV_BOUNDS = "OUT_OF_IV_BOUNDS"


class PricingModel(StrEnum):
    BLACK_SCHOLES = "BLACK_SCHOLES"
    BLACK_SCHOLES_MERTON = "BLACK_SCHOLES_MERTON"
    BLACK_76 = "BLACK_76"


@dataclass(frozen=True)
class ArbitrageCheck:
    ok: bool
    reason: ArbitrageReason | None
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class IVSolveResult:
    status: SolveStatus
    reason: str | None
    iv: float | None
    repriced_price: float | None
    abs_error: float | None
    rel_error: float | None
    solver: str
    solver_version: str
    calculation_version: str


def check_arbitrage_bounds(
    flag: str,
    price: float | None,
    spot: float | None,
    strike: float | None,
    time_to_expiry_years: float | None,
    rate: float | None,
    *,
    pricing_model: PricingModel | str = PricingModel.BLACK_SCHOLES,
    dividend_yield: float | None = 0.0,
    forward_price: float | None = None,
) -> ArbitrageCheck:
    """European-equivalent Black-Scholes no-arbitrage bound check, run
    before any numerical solve is attempted."""
    if flag not in {"c", "p"}:
        return ArbitrageCheck(False, ArbitrageReason.INVALID_OPTION_TYPE, None, None)
    try:
        model = PricingModel(pricing_model)
    except ValueError:
        return ArbitrageCheck(False, ArbitrageReason.INVALID_OPTION_TYPE, None, None)
    if spot is None or not math.isfinite(spot) or spot <= 0:
        return ArbitrageCheck(False, ArbitrageReason.INVALID_SPOT, None, None)
    if strike is None or not math.isfinite(strike) or strike <= 0:
        return ArbitrageCheck(False, ArbitrageReason.INVALID_STRIKE, None, None)
    if rate is None or not math.isfinite(rate) or not (-0.5 < rate < 1.0):
        return ArbitrageCheck(False, ArbitrageReason.INVALID_RATE, None, None)
    if time_to_expiry_years is None or not math.isfinite(time_to_expiry_years) or not 0 < time_to_expiry_years <= 30:
        return ArbitrageCheck(False, ArbitrageReason.INVALID_T, None, None)
    if price is None or not math.isfinite(price) or price <= 0:
        return ArbitrageCheck(False, ArbitrageReason.PRICE_UNAVAILABLE, None, None)

    discount = math.exp(-rate * time_to_expiry_years)
    if model is PricingModel.BLACK_76:
        if forward_price is None or not math.isfinite(forward_price) or forward_price <= 0:
            return ArbitrageCheck(False, ArbitrageReason.INVALID_SPOT, None, None)
        if flag == "c":
            lower = discount * max(forward_price - strike, 0.0)
            upper = discount * forward_price
        else:
            lower = discount * max(strike - forward_price, 0.0)
            upper = discount * strike
    else:
        q = 0.0 if model is PricingModel.BLACK_SCHOLES else dividend_yield
        if q is None or not math.isfinite(q) or not (-0.5 < q < 1.0):
            return ArbitrageCheck(False, ArbitrageReason.INVALID_RATE, None, None)
        discounted_spot = spot * math.exp(-q * time_to_expiry_years)
        discounted_strike = strike * discount
        if flag == "c":
            lower = max(0.0, discounted_spot - discounted_strike)
            upper = discounted_spot
        else:
            lower = max(0.0, discounted_strike - discounted_spot)
            upper = discounted_strike

    if price <= lower:
        return ArbitrageCheck(False, ArbitrageReason.PRICE_BELOW_INTRINSIC, lower, upper)
    if price >= upper:
        return ArbitrageCheck(False, ArbitrageReason.PRICE_ABOVE_BOUND, lower, upper)
    return ArbitrageCheck(True, None, lower, upper)


def solve_iv_with_provenance(
    flag: str,
    price: float,
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    rate: float,
    *,
    pricing_model: PricingModel | str = PricingModel.BLACK_SCHOLES,
    dividend_yield: float | None = 0.0,
    forward_price: float | None = None,
) -> IVSolveResult:
    """Full defensible pipeline: arbitrage check -> solve -> IV-bounds check
    -> repricing validation. Never forces a value; every rejection carries
    an exact status/reason."""
    bounds = check_arbitrage_bounds(
        flag,
        price,
        spot,
        strike,
        time_to_expiry_years,
        rate,
        pricing_model=pricing_model,
        dividend_yield=dividend_yield,
        forward_price=forward_price,
    )
    if not bounds.ok:
        return IVSolveResult(
            SolveStatus.ARBITRAGE_REJECTED,
            bounds.reason.value if bounds.reason else None,
            None, None, None, None,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )

    try:
        model = PricingModel(pricing_model)

        def residual(sigma: float) -> float:
            return _model_price(
                model,
                flag,
                spot,
                strike,
                time_to_expiry_years,
                rate,
                sigma,
                dividend_yield=dividend_yield,
                forward_price=forward_price,
            ) - price

        if residual(MIN_IV) > 0 or residual(MAX_IV) < 0:
            return IVSolveResult(
                SolveStatus.OUT_OF_IV_BOUNDS, "price_outside_volatility_bracket",
                None, None, None, None, SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
            )
        sigma, convergence = brentq(
            residual, MIN_IV, MAX_IV, xtol=VOLATILITY_TOLERANCE,
            rtol=1e-14, maxiter=MAX_ITERATIONS, full_output=True, disp=False,
        )
        if not convergence.converged:
            raise RuntimeError("iteration_cap")
    except Exception as error:  # solver-internal numerical failure
        return IVSolveResult(
            SolveStatus.NO_IV_SOLUTION,
            f"solver_exception:{type(error).__name__}",
            None, None, None, None,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )

    if sigma is None or not math.isfinite(sigma):
        return IVSolveResult(
            SolveStatus.NO_IV_SOLUTION,
            "non_finite_sigma",
            None, None, None, None,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )
    if not (MIN_IV <= sigma <= MAX_IV):
        return IVSolveResult(
            SolveStatus.OUT_OF_IV_BOUNDS,
            f"sigma_outside_[{MIN_IV},{MAX_IV}]",
            float(sigma), None, None, None,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )

    try:
        repriced = _model_price(
            PricingModel(pricing_model),
            flag,
            spot,
            strike,
            time_to_expiry_years,
            rate,
            sigma,
            dividend_yield=dividend_yield,
            forward_price=forward_price,
        )
    except Exception as error:
        return IVSolveResult(
            SolveStatus.REPRICING_FAILED,
            f"repricing_exception:{type(error).__name__}",
            float(sigma), None, None, None,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )
    if not math.isfinite(repriced):
        return IVSolveResult(
            SolveStatus.REPRICING_FAILED,
            "non_finite_repriced_price",
            float(sigma), None, None, None,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )

    abs_error = abs(repriced - price)
    rel_error = abs_error / price if price > 0 else float("inf")
    if abs_error > MAX_REPRICING_ABSOLUTE_ERROR and rel_error > MAX_REPRICING_RELATIVE_ERROR:
        return IVSolveResult(
            SolveStatus.REPRICING_FAILED,
            f"repricing_error_exceeds_tolerance(abs={abs_error:.4f},rel={rel_error:.4f})",
            float(sigma), repriced, abs_error, rel_error,
            SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
        )

    return IVSolveResult(
        SolveStatus.OK,
        None,
        float(sigma), repriced, abs_error, rel_error,
        SOLVER_NAME, SOLVER_VERSION, CALCULATION_VERSION,
    )


def _model_price(
    model: PricingModel,
    flag: str,
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    rate: float,
    sigma: float,
    *,
    dividend_yield: float | None,
    forward_price: float | None,
) -> float:
    if model is PricingModel.BLACK_76:
        if forward_price is None:
            raise ValueError("forward price required")
        return float(_black76_price(flag, forward_price, strike, time_to_expiry_years, rate, sigma))
    if model is PricingModel.BLACK_SCHOLES_MERTON:
        if dividend_yield is None:
            raise ValueError("dividend yield required")
        return float(
            _bsm_price(
                flag,
                spot,
                strike,
                time_to_expiry_years,
                rate,
                sigma,
                dividend_yield,
            )
        )
    return float(_bs_price(flag, spot, strike, time_to_expiry_years, rate, sigma))


def option_flag(option_type: str) -> str | None:
    normalized = str(option_type or "").strip().upper()
    if normalized in {"CE", "C", "CALL"}:
        return "c"
    if normalized in {"PE", "P", "PUT"}:
        return "p"
    return None
