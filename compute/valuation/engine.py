from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Any, Iterable


MINIMUM_PEERS = 5
MAX_FUNDAMENTAL_AGE_DAYS = 200
FINANCIAL_SECTOR_MARKERS = (
    "BANK",
    "FINANC",
    "NBFC",
    "INSURANCE",
    "ASSET MANAGEMENT",
    "CAPITAL MARKET",
)


def build_valuation(
    *,
    symbol: str,
    sector: str | None,
    market_price: float | None,
    market_price_as_of: str | None,
    quote_quality: str,
    financials: dict[str, Any] | None,
    peers: Iterable[dict[str, Any]],
    evaluated_on: dt.date | None = None,
) -> dict[str, Any]:
    """Estimate fair value from observed TTM EPS and robust sector peer P/E.

    DCF, dividend, and residual-income methods are intentionally unavailable
    until their required observed inputs exist. The returned range is the
    retained peer P/E interquartile range, expanded by target EPS instability.
    """
    normalized = symbol.strip().upper()
    as_of = evaluated_on or dt.date.today()
    # Computed once, independent of which gate below fires first, so a real
    # fact (e.g. negative TTM EPS) is never lost just because an earlier gate
    # (e.g. missing sector) already made the primary reason something else.
    secondary_reasons = _secondary_reasons(financials, as_of)
    price = _finite_positive(market_price)
    if price is None:
        return insufficient_valuation(
            normalized,
            market_price=None,
            market_price_as_of=market_price_as_of,
            quote_quality=quote_quality,
            reason="MARKET_PRICE_UNAVAILABLE",
            secondary_reasons=secondary_reasons,
        )
    normalized_sector = str(sector or "").strip().upper()
    if not normalized_sector:
        return insufficient_valuation(
            normalized,
            market_price=price,
            market_price_as_of=market_price_as_of,
            quote_quality=quote_quality,
            reason="SECTOR_CLASSIFICATION_UNAVAILABLE",
            secondary_reasons=secondary_reasons,
        )
    if _is_financial_sector(normalized_sector):
        return insufficient_valuation(
            normalized,
            market_price=price,
            market_price_as_of=market_price_as_of,
            quote_quality=quote_quality,
            reason="BOOK_VALUE_AND_ROE_REQUIRED_FOR_FINANCIAL_SECTOR",
            sector=normalized_sector,
            secondary_reasons=secondary_reasons,
            warnings=[
                "P/E-only valuation rejected for bank/NBFC/financial-sector business.",
                "Residual-income or P/B-ROE inputs are unavailable.",
            ],
        )

    target_eps = _ttm_eps(financials, as_of)
    if target_eps["status"] != "VALID":
        return insufficient_valuation(
            normalized,
            market_price=price,
            market_price_as_of=market_price_as_of,
            quote_quality=quote_quality,
            reason=str(target_eps["reason"]),
            sector=normalized_sector,
            data_as_of=target_eps.get("data_as_of"),
            secondary_reasons=secondary_reasons,
        )

    peer_rows: list[dict[str, Any]] = []
    for peer in peers:
        peer_symbol = str(peer.get("symbol") or "").strip().upper()
        if not peer_symbol or peer_symbol == normalized:
            continue
        peer_sector = str(peer.get("sector") or "").strip().upper()
        if peer_sector != normalized_sector:
            continue
        peer_price = _finite_positive(peer.get("market_price"))
        peer_eps = _ttm_eps(peer.get("financials"), as_of)
        if peer_price is None or peer_eps["status"] != "VALID":
            continue
        eps_value = float(peer_eps["eps_ttm"])
        if eps_value <= 0:
            continue
        multiple = peer_price / eps_value
        if math.isfinite(multiple) and multiple > 0:
            peer_rows.append(
                {"symbol": peer_symbol, "pe_ttm": multiple, "data_as_of": peer_eps["data_as_of"]}
            )

    retained, excluded = _robust_peer_sample(peer_rows)
    if len(retained) < MINIMUM_PEERS:
        return insufficient_valuation(
            normalized,
            market_price=price,
            market_price_as_of=market_price_as_of,
            quote_quality=quote_quality,
            reason="INADEQUATE_VALID_SECTOR_PEERS",
            sector=normalized_sector,
            data_as_of=target_eps.get("data_as_of"),
            peer_count=len(retained),
            secondary_reasons=secondary_reasons,
            warnings=[f"Need at least {MINIMUM_PEERS} valid same-sector peers after robust filtering."],
        )

    multiples = sorted(float(row["pe_ttm"]) for row in retained)
    peer_low = _quantile(multiples, 0.25)
    peer_central = statistics.median(multiples)
    peer_high = _quantile(multiples, 0.75)
    eps_ttm = float(target_eps["eps_ttm"])
    eps_uncertainty = float(target_eps["eps_uncertainty_fraction"])
    fair_value = eps_ttm * peer_central
    fair_value_low = max(0.0, eps_ttm * peer_low * (1.0 - eps_uncertainty))
    fair_value_high = eps_ttm * peer_high * (1.0 + eps_uncertainty)
    if not all(math.isfinite(value) and value > 0 for value in (fair_value_low, fair_value, fair_value_high)):
        return insufficient_valuation(
            normalized,
            market_price=price,
            market_price_as_of=market_price_as_of,
            quote_quality=quote_quality,
            reason="NON_FINITE_VALUATION_RESULT",
            sector=normalized_sector,
            data_as_of=target_eps.get("data_as_of"),
            secondary_reasons=secondary_reasons,
        )
    fair_value_low, fair_value_high = sorted((fair_value_low, fair_value_high))
    classification = (
        "UNDERVALUED"
        if price < fair_value_low
        else "OVERVALUED"
        if price > fair_value_high
        else "FAIRLY_VALUED"
    )
    dispersion = math.log(peer_high / peer_low) if peer_low > 0 else math.inf
    evidence_quality, evidence_score = _evidence_quality(
        peer_count=len(retained),
        fundamental_age_days=int(target_eps["age_days"]),
        peer_log_iqr=dispersion,
        eps_uncertainty=eps_uncertainty,
    )
    discount_premium_pct = ((fair_value - price) / fair_value) * 100.0
    data_dates = [str(target_eps["data_as_of"])] + [
        str(row["data_as_of"]) for row in retained if row.get("data_as_of")
    ]
    return {
        "symbol": normalized,
        "secondary_reasons": [],
        "sector": normalized_sector,
        "market_price": _round(price),
        "market_price_as_of": market_price_as_of,
        "quote_quality": quote_quality,
        "fair_value": _round(fair_value),
        "fair_value_low": _round(fair_value_low),
        "fair_value_high": _round(fair_value_high),
        "price_to_fair_value": _round(price / fair_value, 4),
        "upside_downside_pct": _round(((fair_value / price) - 1.0) * 100.0),
        "discount_premium_pct": _round(discount_premium_pct),
        "margin_of_safety_pct": _round(max(0.0, discount_premium_pct)),
        "classification": classification,
        "evidence_quality": evidence_quality,
        "evidence_score": evidence_score,
        "evidence_score_semantics": "ordinal evidence score; not calibrated probability",
        "methods_used": ["RELATIVE_PE_TTM"],
        "methods": [
            {
                "method": "RELATIVE_PE_TTM",
                "status": "VALID",
                "estimate": _round(fair_value),
                "low": _round(fair_value_low),
                "high": _round(fair_value_high),
                "weight": 1.0,
                "target_eps_ttm": _round(eps_ttm, 4),
                "peer_pe_median": _round(peer_central, 4),
                "peer_pe_p25": _round(peer_low, 4),
                "peer_pe_p75": _round(peer_high, 4),
                "peer_count": len(retained),
                "excluded_outlier_count": excluded,
            }
        ],
        "peer_count": len(retained),
        "data_as_of": min(data_dates) if data_dates else target_eps.get("data_as_of"),
        "status": "VALID",
        "reason": None,
        "uncertainty": {
            "basis": "same-sector peer P/E interquartile range expanded by target quarterly EPS instability",
            "target_eps_uncertainty_fraction": _round(eps_uncertainty, 4),
            "peer_log_iqr": _round(dispersion, 4),
        },
        "confidence_formula": {
            "type": "ordinal_evidence",
            "components": [
                "fundamental completeness and freshness",
                "valid peer sample size",
                "peer multiple dispersion",
                "target EPS stability",
            ],
        },
        "warnings": [
            "DCF unavailable: observed free cash flow, capital structure, beta, and sourced discount-rate inputs absent.",
            "Relative valuation is not a next-day price forecast or target price.",
        ],
    }


def insufficient_valuation(
    symbol: str,
    *,
    market_price: float | None,
    market_price_as_of: str | None,
    quote_quality: str,
    reason: str,
    sector: str | None = None,
    data_as_of: str | None = None,
    peer_count: int = 0,
    warnings: list[str] | None = None,
    secondary_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol.strip().upper(),
        "secondary_reasons": [r for r in (secondary_reasons or []) if r != reason],
        "sector": sector,
        "market_price": _round(market_price) if market_price is not None else None,
        "market_price_as_of": market_price_as_of,
        "quote_quality": quote_quality,
        "fair_value": None,
        "fair_value_low": None,
        "fair_value_high": None,
        "price_to_fair_value": None,
        "upside_downside_pct": None,
        "discount_premium_pct": None,
        "margin_of_safety_pct": None,
        "classification": "INDETERMINATE",
        "evidence_quality": "INSUFFICIENT_EVIDENCE",
        "evidence_score": 0,
        "evidence_score_semantics": "ordinal evidence score; not calibrated probability",
        "methods_used": [],
        "methods": [
            {"method": "RELATIVE_PE_TTM", "status": "INSUFFICIENT_DATA", "reason": reason},
            {"method": "DCF", "status": "INSUFFICIENT_DATA", "reason": "REQUIRED_OBSERVED_INPUTS_UNAVAILABLE"},
        ],
        "peer_count": peer_count,
        "data_as_of": data_as_of,
        "status": "INSUFFICIENT_DATA",
        "reason": reason,
        "uncertainty": None,
        "confidence_formula": {"type": "ordinal_evidence", "components": []},
        "warnings": warnings
        or ["No fair value emitted because required evidence is insufficient."],
    }


def _secondary_reasons(financials: dict[str, Any] | None, as_of: dt.date) -> list[str]:
    """Real facts about the target that an earlier-firing gate (missing
    price, sector, or financial-sector exclusion) must not cause to be
    lost entirely -- computed independently of gate order, before any gate
    runs, so e.g. a genuinely negative TTM EPS is still visible even when
    the primary reason reported is SECTOR_CLASSIFICATION_UNAVAILABLE."""
    eps_result = _ttm_eps(financials, as_of)
    if eps_result["status"] == "INVALID" and eps_result.get("reason") == "NON_POSITIVE_TTM_EARNINGS":
        return ["NEGATIVE_OR_ZERO_TTM_EPS"]
    return []


def _ttm_eps(financials: dict[str, Any] | None, as_of: dt.date) -> dict[str, Any]:
    rows = financials.get("quarters") if isinstance(financials, dict) else None
    if not isinstance(rows, list):
        return {"status": "INVALID", "reason": "QUARTERLY_FINANCIALS_UNAVAILABLE"}
    by_date: dict[dt.date, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            report_date = dt.date.fromisoformat(str(row.get("report_date")))
        except ValueError:
            continue
        eps = _finite(row.get("eps"))
        if report_date > as_of or eps is None:
            continue
        by_date.setdefault(report_date, eps)
    ordered = sorted(by_date.items(), reverse=True)[:4]
    if len(ordered) < 4:
        return {
            "status": "INVALID",
            "reason": "FOUR_DISTINCT_QUARTERS_REQUIRED",
            "data_as_of": ordered[0][0].isoformat() if ordered else None,
        }
    newest, oldest = ordered[0][0], ordered[-1][0]
    span_days = (newest - oldest).days
    age_days = (as_of - newest).days
    if age_days > MAX_FUNDAMENTAL_AGE_DAYS:
        return {
            "status": "INVALID",
            "reason": "STALE_FUNDAMENTALS",
            "data_as_of": newest.isoformat(),
        }
    if span_days < 240 or span_days > 460:
        return {
            "status": "INVALID",
            "reason": "NON_CONTIGUOUS_QUARTERLY_HISTORY",
            "data_as_of": newest.isoformat(),
        }
    values = [value for _, value in ordered]
    eps_ttm = sum(values)
    if not math.isfinite(eps_ttm) or eps_ttm <= 0:
        return {
            "status": "INVALID",
            "reason": "NON_POSITIVE_TTM_EARNINGS",
            "data_as_of": newest.isoformat(),
        }
    mean_abs = statistics.fmean(abs(value) for value in values)
    instability = statistics.pstdev(values) / mean_abs if mean_abs > 0 else 1.0
    return {
        "status": "VALID",
        "reason": None,
        "eps_ttm": eps_ttm,
        "eps_uncertainty_fraction": min(0.75, max(0.0, instability * 0.25)),
        "data_as_of": newest.isoformat(),
        "age_days": age_days,
        "quarter_dates": [day.isoformat() for day, _ in ordered],
    }


def _robust_peer_sample(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if len(rows) < MINIMUM_PEERS:
        return rows, 0
    logs = sorted(math.log(float(row["pe_ttm"])) for row in rows)
    q1 = _quantile(logs, 0.25)
    q3 = _quantile(logs, 0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    retained = [row for row in rows if low <= math.log(float(row["pe_ttm"])) <= high]
    return retained, len(rows) - len(retained)


def _evidence_quality(
    *, peer_count: int, fundamental_age_days: int, peer_log_iqr: float, eps_uncertainty: float
) -> tuple[str, int]:
    score = sum(
        (
            peer_count >= 10,
            fundamental_age_days <= 120,
            peer_log_iqr <= 0.5,
            eps_uncertainty <= 0.25,
        )
    )
    return ("HIGH_EVIDENCE" if score == 4 else "MODERATE_EVIDENCE" if score >= 2 else "LOW_EVIDENCE", score)


def _is_financial_sector(sector: str) -> bool:
    return any(marker in sector for marker in FINANCIAL_SECTOR_MARKERS)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(float(value)) else None
