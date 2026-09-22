"""Canonical, fail-closed IV observation contract.

`iv_history` stores immutable observations. One row is one contract at one
source timestamp, never a cross-strike/cross-expiry average. Historical
reconstructions and live observations remain distinct. Downstream history
reads use historical reconstructions unless explicitly requested otherwise.
No automatic retention or destructive rewrite is allowed.

Rates and volatilities are annualized decimals. Time is ACT/365 to 15:30 IST.
BLACK_76 rows use a proven same-expiry forward and no dividend yield.
BLACK_SCHOLES_MERTON rows require explicit, proven dividend yield. Legacy rows
may remain stored but cannot validate or enter decision paths.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from enum import StrEnum
from typing import Any

from options.iv_pricing import PricingModel, SolveStatus, option_flag, solve_iv_with_provenance


CONTRACT_VERSION = "iv-observation-v3"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MAX_RATE_AGE_DAYS = 4  # admits a Friday curve on Monday; rejects week-old observations


class CalculationStatus(StrEnum):
    VALID = "VALID"
    INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
    NON_EXECUTABLE_PRICE = "NON_EXECUTABLE_PRICE"
    EXPIRED = "EXPIRED"
    NO_CONVERGENCE = "NO_CONVERGENCE"
    BAD_SPOT = "BAD_SPOT"
    BAD_OPTION_PRICE = "BAD_OPTION_PRICE"
    MISSING_RATE = "MISSING_RATE"
    STALE = "STALE"
    SOURCE_UNVERIFIED = "SOURCE_UNVERIFIED"
    ARBITRAGE_VIOLATION = "ARBITRAGE_VIOLATION"


# One ordered shape shared by migration, reads and WriterService.
COLUMN_TYPES: dict[str, str] = {
    "observation_id": "TEXT",
    "symbol": "TEXT",
    "underlying": "TEXT",
    "expiry": "DATE",
    "strike": "DOUBLE",
    "option_type": "TEXT",
    "date": "DATE",
    "observed_at": "TEXT",
    "source_timestamp": "TEXT",
    "available_at": "TEXT",
    "iv": "DOUBLE",
    "contract_token": "TEXT",
    "option_price": "DOUBLE",
    "option_price_semantic": "TEXT",
    "option_price_source": "TEXT",
    "option_price_quality": "TEXT",
    "spot_price": "DOUBLE",
    "spot_source": "TEXT",
    "spot_timestamp": "TEXT",
    "spot_quality": "TEXT",
    "forward_price": "DOUBLE",
    "forward_source": "TEXT",
    "forward_timestamp": "TEXT",
    "forward_quality": "TEXT",
    "risk_free_rate": "DOUBLE",
    "risk_free_tenor_days": "INTEGER",
    "risk_free_source": "TEXT",
    "risk_free_source_url": "TEXT",
    "risk_free_observation_date": "DATE",
    "risk_free_published_at": "TEXT",
    "risk_free_retrieved_at": "TEXT",
    "risk_free_convention": "TEXT",
    "time_to_expiry_years": "DOUBLE",
    "dividend_yield": "DOUBLE",
    "dividend_source": "TEXT",
    "pricing_model": "TEXT",
    "solver": "TEXT",
    "solver_version": "TEXT",
    "repriced_price": "DOUBLE",
    "repricing_abs_error": "DOUBLE",
    "repricing_rel_error": "DOUBLE",
    "volume": "BIGINT",
    "oi": "BIGINT",
    "source": "TEXT",
    "quality": "TEXT",
    "calculation_status": "TEXT",
    "reason": "TEXT",
    "observation_kind": "TEXT",
    "created_at": "TEXT",
    "calculation_version": "TEXT",
    "contract_version": "TEXT",
}
OBSERVATION_COLUMNS = tuple(COLUMN_TYPES)


def timestamp(value: Any) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(dt.timezone.utc) if parsed.utcoffset() is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_observation_id(row: dict[str, Any]) -> str:
    identity = {
        name: row.get(name)
        for name in (
            "symbol", "expiry", "strike", "option_type", "observed_at",
            "observation_kind", "option_price_source", "pricing_model",
        )
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return "iv:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_observation(row: dict[str, Any]) -> None:
    """Raise before persistence/read promotion when provenance is incomplete."""
    if set(row) != set(OBSERVATION_COLUMNS):
        raise ValueError("SOURCE_UNVERIFIED: invalid observation schema")
    required_text = (
        "observation_id", "symbol", "underlying", "option_price_semantic",
        "option_price_source", "option_price_quality", "spot_source",
        "spot_quality", "risk_free_source", "risk_free_source_url",
        "risk_free_convention", "pricing_model", "solver", "solver_version",
        "source", "quality", "calculation_status", "observation_kind",
        "calculation_version", "contract_version",
    )
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required_text):
        raise ValueError("SOURCE_UNVERIFIED: missing identity or provenance")
    if row["contract_version"] != CONTRACT_VERSION:
        raise ValueError("SOURCE_UNVERIFIED: legacy observation")
    if row["calculation_status"] != CalculationStatus.VALID:
        raise ValueError("SOURCE_UNVERIFIED: failed observation is not persistable")
    if row["observation_id"] != canonical_observation_id(row):
        raise ValueError("SOURCE_UNVERIFIED: observation id mismatch")
    if row["symbol"] != row["underlying"] or row["symbol"] != row["symbol"].strip().upper():
        raise ValueError("SOURCE_UNVERIFIED: invalid underlying identity")
    if row["option_type"] not in {"CE", "PE"}:
        raise ValueError("SOURCE_UNVERIFIED: invalid contract type")
    if row["observation_kind"] not in {"LIVE", "HISTORICAL_RECONSTRUCTION"}:
        raise ValueError("SOURCE_UNVERIFIED: invalid observation kind")
    expected_price_quality = (
        "EXECUTABLE_BID_ASK_MIDPOINT"
        if row["observation_kind"] == "LIVE"
        else "VALID_EXECUTABLE_OBSERVATION"
    )
    if row["option_price_quality"] != expected_price_quality:
        raise ValueError(CalculationStatus.NON_EXECUTABLE_PRICE)
    if row["spot_quality"] != "OBSERVED_VERIFIED":
        raise ValueError(CalculationStatus.SOURCE_UNVERIFIED)
    if row["quality"] not in {"IV_VALID", "IV_VALID_LOW_LIQUIDITY"}:
        raise ValueError(CalculationStatus.SOURCE_UNVERIFIED)
    if row["risk_free_convention"] != "CONTINUOUS_ACT_365":
        raise ValueError(CalculationStatus.MISSING_RATE)

    instants = {
        key: timestamp(row.get(key))
        for key in (
            "observed_at", "source_timestamp", "spot_timestamp", "available_at",
            "risk_free_retrieved_at", "created_at",
        )
    }
    if any(value is None for value in instants.values()):
        raise ValueError("SOURCE_UNVERIFIED: timestamps required")
    observed = instants["observed_at"]
    try:
        expiry = dt.date.fromisoformat(str(row["expiry"]))
        trade_date = dt.date.fromisoformat(str(row["date"]))
        rate_date = dt.date.fromisoformat(str(row["risk_free_observation_date"]))
    except (TypeError, ValueError) as error:
        raise ValueError("SOURCE_UNVERIFIED: invalid dates") from error
    if observed.astimezone(IST).date() != trade_date:
        raise ValueError("SOURCE_UNVERIFIED: trade date mismatch")
    expiry_at = dt.datetime.combine(expiry, dt.time(15, 30), IST)
    expected_t = (expiry_at - observed).total_seconds() / (365 * 86400)
    if expected_t <= 0:
        raise ValueError(CalculationStatus.EXPIRED)
    if instants["source_timestamp"] != observed:
        raise ValueError("STALE: option timestamp mismatch")
    if abs((instants["spot_timestamp"] - observed).total_seconds()) > 1:
        raise ValueError("STALE: unsynchronized spot and option observation")
    if not 0 <= (trade_date - rate_date).days <= MAX_RATE_AGE_DAYS:
        raise ValueError("MISSING_RATE: future or stale rate date")
    published = timestamp(row.get("risk_free_published_at"))
    if rate_date == trade_date and (published is None or published > observed):
        raise ValueError("MISSING_RATE: same-day rate not yet published")
    if published is not None and published > observed:
        raise ValueError("MISSING_RATE: future rate publication")
    if instants["available_at"] < observed or instants["created_at"] < instants["available_at"]:
        raise ValueError("SOURCE_UNVERIFIED: invalid availability order")
    if instants["risk_free_retrieved_at"] > instants["available_at"]:
        raise ValueError("SOURCE_UNVERIFIED: rate retrieved after observation availability")
    if row["observation_kind"] == "LIVE" and instants["risk_free_retrieved_at"] > observed:
        raise ValueError("MISSING_RATE: live rate was not retrieved by observation time")

    numeric = (
        "iv", "strike", "spot_price", "option_price", "time_to_expiry_years",
        "risk_free_rate", "repriced_price", "repricing_abs_error",
        "repricing_rel_error",
    )
    if any(not finite(row.get(key)) for key in numeric):
        raise ValueError(CalculationStatus.INSUFFICIENT_INPUT)
    if row["spot_price"] <= 0:
        raise ValueError(CalculationStatus.BAD_SPOT)
    if row["option_price"] <= 0:
        raise ValueError(CalculationStatus.BAD_OPTION_PRICE)
    if not math.isclose(row["time_to_expiry_years"], expected_t, abs_tol=1e-12):
        raise ValueError("SOURCE_UNVERIFIED: time to expiry mismatch")

    try:
        model = PricingModel(row["pricing_model"])
    except ValueError as error:
        raise ValueError("SOURCE_UNVERIFIED: unsupported pricing model") from error
    forward_price = row.get("forward_price")
    dividend_yield = row.get("dividend_yield")
    if model is PricingModel.BLACK_76:
        if not finite(forward_price) or forward_price <= 0:
            raise ValueError("INSUFFICIENT_INPUT: forward price")
        for key in ("forward_source", "forward_quality"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("SOURCE_UNVERIFIED: forward provenance")
        forward_timestamp = timestamp(row.get("forward_timestamp"))
        if forward_timestamp is None or abs((forward_timestamp - observed).total_seconds()) > 1:
            raise ValueError("STALE: forward timestamp mismatch")
        if dividend_yield is not None or row.get("dividend_source") is not None:
            raise ValueError("SOURCE_UNVERIFIED: dividend yield not used by BLACK_76")
    elif model is PricingModel.BLACK_SCHOLES_MERTON:
        if not finite(dividend_yield) or not isinstance(row.get("dividend_source"), str):
            raise ValueError("INSUFFICIENT_INPUT: dividend yield provenance")
        forward_price = None
    else:
        if dividend_yield != 0.0 or not isinstance(row.get("dividend_source"), str):
            raise ValueError("INSUFFICIENT_INPUT: proven zero dividend required")
        forward_price = None

    solved = solve_iv_with_provenance(
        option_flag(row["option_type"]), row["option_price"], row["spot_price"],
        row["strike"], expected_t, row["risk_free_rate"], pricing_model=model,
        dividend_yield=dividend_yield, forward_price=forward_price,
    )
    if solved.status is not SolveStatus.OK or not math.isclose(row["iv"], solved.iv, abs_tol=1e-9):
        raise ValueError("SOURCE_UNVERIFIED: IV does not reproduce")
    comparisons = (
        ("repriced_price", solved.repriced_price),
        ("repricing_abs_error", solved.abs_error),
        ("repricing_rel_error", solved.rel_error),
    )
    if any(not math.isclose(row[key], value, abs_tol=1e-8, rel_tol=1e-9) for key, value in comparisons):
        raise ValueError("SOURCE_UNVERIFIED: repricing provenance mismatch")


def validated_live_observation(
    leg: dict[str, Any],
    *,
    as_of: dt.datetime | None = None,
    max_age_seconds: int = 180,
) -> dict[str, Any] | None:
    """Return a current canonical live IV row tied to this exact chain leg."""
    row = leg.get("iv_observation")
    if not isinstance(row, dict):
        return None
    try:
        validate_observation(row)
        if row["observation_kind"] != "LIVE" or leg.get("calculation_status") != "VALID":
            return None
        observed = timestamp(row["observed_at"])
        current = (as_of or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        if observed is None or not 0 <= (current - observed).total_seconds() <= max_age_seconds:
            return None
        if str(leg.get("symbol") or "").strip().upper() != row["symbol"]:
            return None
        if str(leg.get("option_type") or "").strip().upper() != row["option_type"]:
            return None
        if not finite(leg.get("strike")) or not math.isclose(float(leg["strike"]), row["strike"], abs_tol=1e-9):
            return None
        if _contract_date(leg.get("expiry")) != dt.date.fromisoformat(str(row["expiry"])):
            return None
        if not finite(leg.get("iv")) or not math.isclose(float(leg["iv"]), row["iv"], abs_tol=1e-9):
            return None
        return row
    except (TypeError, ValueError, OverflowError):
        return None


def _contract_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def daily_atm_rows(
    rows: list[dict[str, Any]],
    *,
    as_of: dt.datetime,
    observation_kind: str = "HISTORICAL_RECONSTRUCTION",
) -> list[dict[str, Any]]:
    """Select one actual front-expiry ATM call per completed trade date."""
    days: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            validate_observation(row)
            if row["observation_kind"] != observation_kind or row["option_type"] != "CE":
                continue
            if dt.date.fromisoformat(str(row["date"])) >= as_of.astimezone(IST).date():
                continue
            if timestamp(row["available_at"]) > as_of:
                continue
            rank = (
                str(row["expiry"]),
                abs(float(row["strike"]) - float(row["spot_price"])),
                float(row["strike"]),
                str(row["observed_at"]),
            )
            previous = days.get(str(row["date"]))
            if previous is None or rank < previous["rank"]:
                days[str(row["date"])] = {"rank": rank, "row": row}
        except (TypeError, ValueError, OverflowError):
            continue
    return [days[day]["row"] for day in sorted(days)]
