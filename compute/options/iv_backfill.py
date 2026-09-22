"""Bounded, dry-run-first historical IV reconstruction.

Only authoritative NSE F&O daily rows already marked executable are admitted.
Each option uses same-date, exact-expiry NSE futures close as Black-76 forward,
same-date verified cash/index close for required spot provenance, and latest
non-future FBIL T-Bill curve point within seven calendar days. No UNKNOWN row,
cross-expiry forward, same-day post-close rate, interpolation, or fallback can
produce an IV observation.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from backtest.price_quality import HistoricalPriceQuality, classify_historical_option_row
from data.index_aliases import index_symbol_candidates
from data.risk_free_rate import (
    FBIL_QUOTE_CONVENTION,
    OPTION_RATE_CONVENTION,
    MAX_RATE_AGE_DAYS,
    TenorPoint,
    RateSelection,
    select_rate_for_tenor,
    time_to_expiry_years_to_days,
)
from db.write_service import ReadOperation, WriteOperation, resolve_read, submit_write
from options.iv_contract import (
    CONTRACT_VERSION,
    IST,
    CalculationStatus,
    canonical_observation_id,
    validate_observation,
)
from options.iv_pricing import (
    CALCULATION_VERSION as SOLVER_CALCULATION_VERSION,
    PricingModel,
    SolveStatus,
    option_flag,
    solve_iv_with_provenance,
)


LOW_LIQUIDITY_VOLUME_THRESHOLD = 100
CALCULATION_VERSION = f"iv-backfill-v2+{SOLVER_CALCULATION_VERSION}"
OPTION_PRICE_SEMANTIC = "NSE_FO_BHAVCOPY_DAILY_CLOSE_LAST_TRADED_PRICE"
FORWARD_PRICE_SEMANTIC = "NSE_FO_BHAVCOPY_EXACT_EXPIRY_FUTURES_DAILY_CLOSE"
SPOT_TIMESTAMP_SEMANTIC = "NSE_SESSION_CLOSE_EFFECTIVE_15_30_IST"
BACKFILL_LOOKBACK_DAYS = 370
WRITE_BATCH_SIZE = 500


@dataclass
class BackfillCounters:
    fo_rows_seen: int = 0
    fo_rows_futures_skipped: int = 0
    option_candidates: int = 0
    price_quality_counts: dict[str, int] = field(default_factory=dict)
    spot_unavailable: int = 0
    forward_unavailable: int = 0
    rate_unavailable: int = 0
    expired_or_zero_t: int = 0
    arbitrage_rejected: int = 0
    solver_no_solution: int = 0
    solver_out_of_bounds: int = 0
    repricing_failed: int = 0
    provenance_rejected: int = 0
    duplicate_source_rows: int = 0
    iv_valid: int = 0
    iv_valid_low_liquidity: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    def bump_reason(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def total_iv_eligible(self) -> int:
        return self.iv_valid + self.iv_valid_low_liquidity

    def rejected(self) -> int:
        return self.option_candidates - self.total_iv_eligible()

    def merge(self, other: "BackfillCounters") -> None:
        for name in (
            "fo_rows_seen", "fo_rows_futures_skipped", "option_candidates",
            "spot_unavailable", "forward_unavailable", "rate_unavailable",
            "expired_or_zero_t", "arbitrage_rejected", "solver_no_solution",
            "solver_out_of_bounds", "repricing_failed", "provenance_rejected",
            "iv_valid", "iv_valid_low_liquidity",
            "duplicate_source_rows",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for target, source in (
            (self.price_quality_counts, other.price_quality_counts),
            (self.reason_counts, other.reason_counts),
        ):
            for key, value in source.items():
                target[key] = target.get(key, 0) + value

    def as_dict(self) -> dict[str, Any]:
        return {
            "fo_rows_seen": self.fo_rows_seen,
            "fo_rows_futures_skipped": self.fo_rows_futures_skipped,
            "option_candidates": self.option_candidates,
            "price_quality_counts": dict(sorted(self.price_quality_counts.items())),
            "spot_unavailable": self.spot_unavailable,
            "forward_unavailable": self.forward_unavailable,
            "rate_unavailable": self.rate_unavailable,
            "expired_or_zero_t": self.expired_or_zero_t,
            "arbitrage_rejected": self.arbitrage_rejected,
            "solver_no_solution": self.solver_no_solution,
            "solver_out_of_bounds": self.solver_out_of_bounds,
            "repricing_failed": self.repricing_failed,
            "provenance_rejected": self.provenance_rejected,
            "duplicate_source_rows": self.duplicate_source_rows,
            "iv_valid": self.iv_valid,
            "iv_valid_low_liquidity": self.iv_valid_low_liquidity,
            "total_iv_eligible": self.total_iv_eligible(),
            "rejected": self.rejected(),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


@dataclass(frozen=True)
class CandidateFailure:
    symbol: str
    expiry: str
    strike: float
    option_type: str
    date: str
    stage: str
    reason: str


def _default_date_range() -> tuple[dt.date, dt.date]:
    today = dt.datetime.now(IST).date()
    return today - dt.timedelta(days=BACKFILL_LOOKBACK_DAYS), today


def load_rate_curve(from_date: dt.date, to_date: dt.date) -> dict[dt.date, list[TenorPoint]]:
    result = resolve_read(
        ReadOperation.LOAD_RISK_FREE_RATE_CURVE,
        {"from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
    )
    by_date: dict[dt.date, list[TenorPoint]] = {}
    for raw in result.get("rows", []):
        if len(raw) < 12:
            continue
        try:
            point = TenorPoint(
                observation_date=dt.date.fromisoformat(str(raw[0])),
                tenor_days=int(raw[1]),
                tenor_label=str(raw[2]),
                rate_pct=float(raw[3]),
                published_at=_timestamp(raw[7]),
                source=str(raw[5]),
                source_url=str(raw[6]),
                retrieved_at=_timestamp(raw[8]),
                quote_convention=str(raw[9]),
            )
            if (
                point.quote_convention != FBIL_QUOTE_CONVENTION
                or str(raw[10]) != OPTION_RATE_CONVENTION
                or not math.isclose(point.rate_decimal, float(raw[4]), rel_tol=1e-12, abs_tol=1e-12)
            ):
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        by_date.setdefault(point.observation_date, []).append(point)
    return by_date


def load_spot_series(symbol: str, from_date: dt.date, to_date: dt.date) -> dict[dt.date, dict[str, Any]]:
    candidates = index_symbol_candidates(symbol) or [symbol]
    result = resolve_read(
        ReadOperation.LOAD_TECHNICAL_OHLCV_RANGE,
        {"symbols": candidates, "from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
    )
    columns = result.get("columns") or (
        "symbol", "date", "open", "high", "low", "close", "volume",
        "observed_at", "source", "quality", "derivation_type",
    )
    preference = {name: index for index, name in enumerate(candidates)}
    decoded = [dict(zip(columns, raw)) for raw in result.get("rows", [])]
    decoded.sort(key=lambda row: preference.get(str(row.get("symbol")), len(candidates)))
    by_date: dict[dt.date, dict[str, Any]] = {}
    for row in decoded:
        try:
            day = dt.date.fromisoformat(str(row["date"]))
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            day in by_date or not math.isfinite(close) or close <= 0
            or row.get("quality") != "OBSERVED_VERIFIED"
            or str(row.get("source") or "").upper() in {"", "UNKNOWN"}
        ):
            continue
        by_date[day] = {"price": close, "source": str(row["source"]), "quality": str(row["quality"])}
    return by_date


def _select_rate_as_of(
    curve: dict[dt.date, list[TenorPoint]],
    observed_at: dt.datetime,
    requested_days: int,
) -> RateSelection | None:
    trade_date = observed_at.astimezone(IST).date()
    eligible_dates = []
    for rate_date, points in curve.items():
        age = (trade_date - rate_date).days
        if not 0 <= age <= MAX_RATE_AGE_DAYS:
            continue
        if rate_date == trade_date and not any(
            point.published_at is not None and point.published_at <= observed_at for point in points
        ):
            continue
        eligible_dates.append(rate_date)
    if not eligible_dates:
        return None
    selected_date = max(eligible_dates)
    eligible_points = [
        point for point in curve[selected_date]
        if selected_date < trade_date or (point.published_at is not None and point.published_at <= observed_at)
    ]
    return select_rate_for_tenor(eligible_points, requested_days)


def _record_failure(
    failures: list[CandidateFailure] | None,
    limit: int,
    row: dict[str, Any],
    stage: str,
    reason: str,
) -> None:
    if failures is None or len(failures) >= limit:
        return
    failures.append(CandidateFailure(
        str(row.get("symbol") or ""), str(row.get("expiry") or ""),
        float(row.get("strike") or 0.0), str(row.get("option_type") or ""),
        str(row.get("date") or ""), stage, reason,
    ))


def _session_close(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(15, 30), IST).astimezone(dt.timezone.utc)


def _build_iv_row(
    candidate: dict[str, Any],
    option_price: float,
    spot: dict[str, Any],
    forward: dict[str, Any],
    rate: RateSelection,
    solve: Any,
    observed_at: dt.datetime,
    now: dt.datetime,
) -> dict[str, Any]:
    volume = _integer(candidate.get("volume"))
    quality = "IV_VALID_LOW_LIQUIDITY" if (volume or 0) < LOW_LIQUIDITY_VOLUME_THRESHOLD else "IV_VALID"
    row = {
        "observation_id": "",
        "symbol": str(candidate["symbol"]).strip().upper(),
        "underlying": str(candidate["symbol"]).strip().upper(),
        "expiry": str(candidate["expiry"]),
        "strike": float(candidate["strike"]),
        "option_type": str(candidate["option_type"]),
        "date": str(candidate["date"]),
        "observed_at": observed_at.isoformat(),
        "source_timestamp": observed_at.isoformat(),
        "available_at": now.isoformat(),
        "iv": solve.iv,
        "contract_token": None,
        "option_price": option_price,
        "option_price_semantic": OPTION_PRICE_SEMANTIC,
        "option_price_source": str(candidate["source"]),
        "option_price_quality": HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION.value,
        "spot_price": float(spot["price"]),
        "spot_source": str(spot["source"]),
        "spot_timestamp": observed_at.isoformat(),
        "spot_quality": str(spot["quality"]),
        "forward_price": float(forward["price"]),
        "forward_source": str(forward["source"]),
        "forward_timestamp": observed_at.isoformat(),
        "forward_quality": HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION.value,
        "risk_free_rate": rate.rate_decimal,
        "risk_free_tenor_days": rate.selected_tenor_days,
        "risk_free_source": rate.source,
        "risk_free_source_url": rate.source_url,
        "risk_free_observation_date": rate.observation_date.isoformat(),
        "risk_free_published_at": rate.published_at.isoformat() if rate.published_at else None,
        "risk_free_retrieved_at": rate.retrieved_at.isoformat() if rate.retrieved_at else None,
        "risk_free_convention": rate.rate_convention,
        "time_to_expiry_years": (
            dt.datetime.combine(candidate["expiry"], dt.time(15, 30), IST).astimezone(dt.timezone.utc)
            - observed_at
        ).total_seconds() / (365 * 86400),
        "dividend_yield": None,
        "dividend_source": None,
        "pricing_model": PricingModel.BLACK_76.value,
        "solver": solve.solver,
        "solver_version": solve.solver_version,
        "repriced_price": solve.repriced_price,
        "repricing_abs_error": solve.abs_error,
        "repricing_rel_error": solve.rel_error,
        "volume": volume,
        "oi": _integer(candidate.get("oi")),
        "source": f"{candidate['source']}+{spot['source']}+{forward['source']}+{rate.source}",
        "quality": quality,
        "calculation_status": CalculationStatus.VALID.value,
        "reason": None,
        "observation_kind": "HISTORICAL_RECONSTRUCTION",
        "created_at": now.isoformat(),
        "calculation_version": CALCULATION_VERSION,
        "contract_version": CONTRACT_VERSION,
    }
    row["observation_id"] = canonical_observation_id(row)
    return row


def derive_symbol_candidates(
    symbol: str,
    *,
    from_date: dt.date,
    to_date: dt.date,
    counters: BackfillCounters,
    failures: list[CandidateFailure] | None = None,
    max_failures_recorded: int = 200,
    rate_curve: dict[dt.date, list[TenorPoint]] | None = None,
) -> list[dict[str, Any]]:
    fo_result = resolve_read(
        ReadOperation.LOAD_FO_EOD_RANGE,
        {"symbol": symbol, "from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
    )
    columns = fo_result.get("columns") or (
        "symbol", "expiry", "strike", "option_type", "date", "open", "high",
        "low", "close", "volume", "oi", "oi_change", "source", "source_format",
        "price_quality",
    )
    rows = [dict(zip(columns, raw)) for raw in fo_result.get("rows", [])]
    counters.fo_rows_seen += len(rows)
    curve = rate_curve if rate_curve is not None else load_rate_curve(from_date - dt.timedelta(days=MAX_RATE_AGE_DAYS), to_date)
    spots = load_spot_series(symbol, from_date, to_date)

    forward_candidates: dict[tuple[dt.date, dt.date], dict[tuple[Any, ...], dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("option_type") or "").strip().upper() != "FUT":
            continue
        counters.fo_rows_futures_skipped += 1
        observation = classify_historical_option_row(row, price_type="close")
        if observation.quality is not HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION:
            continue
        try:
            day = _date(row["date"])
            expiry = _date(row["expiry"])
        except (KeyError, ValueError):
            continue
        value = {
            "price": observation.price,
            "source": f"{observation.source}:{FORWARD_PRICE_SEMANTIC}",
        }
        fingerprint = (value["price"], value["source"])
        forward_candidates.setdefault((day, expiry), {})[fingerprint] = value
    forwards = {
        key: next(iter(candidates.values()))
        for key, candidates in forward_candidates.items()
        if len(candidates) == 1
    }
    ambiguous_forwards = {
        key for key, candidates in forward_candidates.items() if len(candidates) > 1
    }

    option_fingerprints: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    for row in rows:
        option_type = str(row.get("option_type") or "").strip().upper()
        if option_type not in {"CE", "PE"}:
            continue
        identity = (
            str(row.get("symbol") or "").strip().upper(), str(row.get("expiry")),
            row.get("strike"), option_type, str(row.get("date")),
        )
        fingerprint = tuple(
            row.get(name)
            for name in ("open", "high", "low", "close", "volume", "oi", "source", "price_quality")
        )
        option_fingerprints.setdefault(identity, set()).add(fingerprint)
    processed_identities: set[tuple[Any, ...]] = set()

    now = dt.datetime.now(dt.timezone.utc)
    ready: list[dict[str, Any]] = []
    for row in rows:
        option_type = str(row.get("option_type") or "").strip().upper()
        if option_type not in {"CE", "PE"}:
            continue
        counters.option_candidates += 1
        row["option_type"] = option_type
        identity = (
            str(row.get("symbol") or "").strip().upper(), str(row.get("expiry")),
            row.get("strike"), option_type, str(row.get("date")),
        )
        if len(option_fingerprints.get(identity, set())) > 1 or identity in processed_identities:
            counters.duplicate_source_rows += 1
            counters.bump_reason("SOURCE_UNVERIFIED:DUPLICATE_CONTRACT_OBSERVATION")
            _record_failure(
                failures, max_failures_recorded, row, "IDENTITY",
                "SOURCE_UNVERIFIED:DUPLICATE_CONTRACT_OBSERVATION",
            )
            continue
        processed_identities.add(identity)
        observation = classify_historical_option_row(row, price_type="close")
        counters.price_quality_counts[observation.quality.value] = counters.price_quality_counts.get(observation.quality.value, 0) + 1
        if observation.quality is not HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION:
            counters.bump_reason(f"PRICE_QUALITY:{observation.quality.value}:{observation.reason}")
            _record_failure(failures, max_failures_recorded, row, "PRICE", observation.reason)
            continue
        try:
            trade_date = _date(row["date"])
            expiry = _date(row["expiry"])
            strike = float(row["strike"])
        except (KeyError, TypeError, ValueError):
            counters.bump_reason("SOURCE_UNVERIFIED:CONTRACT_IDENTITY")
            continue
        row.update({"date": trade_date, "expiry": expiry, "strike": strike, "symbol": symbol})
        observed_at = _session_close(trade_date)
        expiry_at = dt.datetime.combine(expiry, dt.time(15, 30), IST).astimezone(dt.timezone.utc)
        t = (expiry_at - observed_at).total_seconds() / (365 * 86400)
        if t <= 0:
            counters.expired_or_zero_t += 1
            counters.bump_reason(CalculationStatus.EXPIRED.value)
            _record_failure(failures, max_failures_recorded, row, "EXPIRY", CalculationStatus.EXPIRED.value)
            continue
        spot = spots.get(trade_date)
        if spot is None:
            counters.spot_unavailable += 1
            counters.bump_reason(CalculationStatus.BAD_SPOT.value)
            _record_failure(failures, max_failures_recorded, row, "SPOT", CalculationStatus.BAD_SPOT.value)
            continue
        forward = forwards.get((trade_date, expiry))
        if forward is None:
            counters.forward_unavailable += 1
            reason = (
                "FORWARD_AMBIGUOUS_EXACT_EXPIRY"
                if (trade_date, expiry) in ambiguous_forwards
                else "FORWARD_UNAVAILABLE_EXACT_EXPIRY"
            )
            counters.bump_reason(reason)
            _record_failure(failures, max_failures_recorded, row, "FORWARD", reason)
            continue
        rate = _select_rate_as_of(curve, observed_at, time_to_expiry_years_to_days(t))
        if rate is None:
            counters.rate_unavailable += 1
            counters.bump_reason(CalculationStatus.MISSING_RATE.value)
            _record_failure(failures, max_failures_recorded, row, "RATE", CalculationStatus.MISSING_RATE.value)
            continue
        solved = solve_iv_with_provenance(
            option_flag(option_type), observation.price, float(spot["price"]), strike,
            t, rate.rate_decimal, pricing_model=PricingModel.BLACK_76,
            dividend_yield=None, forward_price=float(forward["price"]),
        )
        if solved.status is not SolveStatus.OK:
            counters.bump_reason(f"{solved.status.value}:{solved.reason}")
            if solved.status is SolveStatus.ARBITRAGE_REJECTED:
                counters.arbitrage_rejected += 1
            elif solved.status is SolveStatus.NO_IV_SOLUTION:
                counters.solver_no_solution += 1
            elif solved.status is SolveStatus.OUT_OF_IV_BOUNDS:
                counters.solver_out_of_bounds += 1
            else:
                counters.repricing_failed += 1
            _record_failure(failures, max_failures_recorded, row, solved.status.value, str(solved.reason))
            continue
        built = _build_iv_row(row, observation.price, spot, forward, rate, solved, observed_at, now)
        try:
            validate_observation(built)
        except (TypeError, ValueError, OverflowError) as error:
            counters.provenance_rejected += 1
            counters.bump_reason(f"PROVENANCE:{error}")
            _record_failure(failures, max_failures_recorded, row, "PROVENANCE", str(error))
            continue
        if built["quality"] == "IV_VALID_LOW_LIQUIDITY":
            counters.iv_valid_low_liquidity += 1
        else:
            counters.iv_valid += 1
        ready.append(built)
    return ready


def persist_rows(
    rows: list[dict[str, Any]],
    *,
    database: Any = None,
    queue_path: Any = None,
) -> int:
    """Bounded, chunked, idempotent persistence via the canonical writer.

    Each chunk's request_id is deterministic (derived from its first/last
    observation_id), so resubmitting the same `rows` after a crash -- whether
    some chunks already landed or not -- reaches the exact same final state:
    already-applied chunks replay as no-ops (both at the writer-ledger level
    via request_id, and at the row level via `_write_iv_rows`'s own
    content-equality check), never-applied chunks insert normally.
    `database`/`queue_path` default to production when omitted (unchanged
    behavior); tests pass isolated paths explicitly.
    """
    total = 0
    kwargs: dict[str, Any] = {}
    if database is not None:
        kwargs["database"] = database
    if queue_path is not None:
        kwargs["queue_path"] = queue_path
    for start in range(0, len(rows), WRITE_BATCH_SIZE):
        batch = rows[start : start + WRITE_BATCH_SIZE]
        request_id = f"iv-backfill:{batch[0]['observation_id']}:{batch[-1]['observation_id']}"
        result = submit_write(
            WriteOperation.UPSERT_IV_OBSERVATIONS,
            {"rows": batch},
            request_id=request_id,
            **kwargs,
        )
        total += int(result["upserted"])
    return total


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ivs = sorted(float(row["iv"]) for row in rows if isinstance(row.get("iv"), (int, float)) and math.isfinite(row["iv"]))
    duplicate_count = len(rows) - len({row["observation_id"] for row in rows})
    contract_keys = {(row["symbol"], row["expiry"], row["strike"], row["option_type"]) for row in rows}
    return {
        "symbols_covered": sorted({row["symbol"] for row in rows}),
        "dates_covered": sorted({str(row["date"]) for row in rows}),
        "contracts_covered": len(contract_keys),
        "expiries_covered": sorted({str(row["expiry"]) for row in rows}),
        "option_type_distribution": dict(sorted(Counter(row["option_type"] for row in rows).items())),
        "iv_percentiles": {str(p): _percentile(ivs, p) for p in (1, 5, 25, 50, 75, 95, 99)} if ivs else {},
        "iv_min": min(ivs) if ivs else None,
        "iv_max": max(ivs) if ivs else None,
        "non_finite_count": len(rows) - len(ivs),
        "duplicate_key_count": duplicate_count,
    }


def run(
    symbols: list[str],
    *,
    from_date: dt.date | None = None,
    to_date: dt.date | None = None,
    dry_run: bool = True,
    failures_per_symbol: int = 50,
    rate_curve: dict[dt.date, list[TenorPoint]] | None = None,
) -> dict[str, Any]:
    default_from, default_to = _default_date_range()
    from_date = from_date or default_from
    to_date = to_date or default_to
    if from_date > to_date or not symbols or len(symbols) > 500:
        raise ValueError("bounded symbol/date range required")
    overall = BackfillCounters()
    per_symbol: dict[str, Any] = {}
    failures: list[CandidateFailure] = []
    rows: list[dict[str, Any]] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        counters = BackfillCounters()
        symbol_failures: list[CandidateFailure] = []
        symbol_rows = derive_symbol_candidates(
            symbol, from_date=from_date, to_date=to_date, counters=counters,
            failures=symbol_failures, max_failures_recorded=failures_per_symbol,
            rate_curve=rate_curve,
        )
        overall.merge(counters)
        per_symbol[symbol] = counters.as_dict()
        failures.extend(symbol_failures)
        rows.extend(symbol_rows)
    persisted = 0 if dry_run else persist_rows(rows)
    return {
        "dry_run": dry_run,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "symbols": [str(symbol).strip().upper() for symbol in symbols],
        "overall": overall.as_dict(),
        "candidate_rows": overall.option_candidates,
        "valid_rows": len(rows),
        "rejected_rows": overall.rejected(),
        "unknown_price_quality_exclusions": overall.price_quality_counts.get("UNKNOWN", 0),
        "missing_rate_exclusions": overall.rate_unavailable,
        "no_convergence_count": overall.solver_no_solution,
        "persisted_rows": persisted,
        "distribution": _distribution(rows),
        "per_symbol": per_symbol,
        "failures_sample": [vars(item) for item in failures[:500]],
        "rows": rows,
    }


def _timestamp(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(dt.timezone.utc) if parsed.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("empty values")
    position = (len(values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
