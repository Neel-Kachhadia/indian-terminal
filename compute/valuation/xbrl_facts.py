"""Extracts real financial facts (revenue, PAT, EPS, ...) from NSE-hosted
XBRL quarterly-result filings.

Why this module exists: NSE's `corporates-financial-results` API returns
filing METADATA only (audited flag, dates, seqNumber, a link to the XBRL
document) -- it has no revenue/PAT/EPS fields at all. Verified live against
RELIANCE/TCS/INFY on 2026-09-03. The actual figures live only in the linked
XBRL XML under the `in-bse-fin` (BSE Ind-AS 2020-03-31) taxonomy. This module
parses that XML and resolves the single standalone-quarter figure for each
canonical metric, with full provenance, or refuses (returns None / raises
XbrlRejected) rather than guess.

Context-shape note (verified live, RELIANCE all 4 quarter-types + TCS/INFY
Q3, 2026-09-03): a filing's outer JSON metadata (`fromDate`, `toDate`) gives
the true single-quarter window reliably. Matching that window against the
XBRL's own plain (non-scenario) duration contexts yields:
  - Q1 and Q4 filings: exactly one matching context -- unambiguous.
  - Q2 and Q3 filings: exactly two matching contexts sharing the identical
    (start, end) pair -- NSE's template duplicates the quarter's own dates
    onto a second context that actually holds year-to-date cumulative
    figures with corrupted period metadata. The two are disambiguated by
    magnitude (the standalone quarter's revenue is the smaller of the two,
    since year-to-date-so-far revenue cannot be smaller than any single
    quarter within it for a revenue-positive business).
Any other context count for a filing is a shape this module has not been
verified against -- it is refused (UNRESOLVED), never guessed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

import defusedxml.ElementTree as _defused_ET  # hardened parse entrypoint; ET retained for types/ParseError

MAX_XBRL_BYTES = 8 * 1024 * 1024  # real quarterly filings run tens-hundreds of KB

# Concept names verified live against real NSE filings (RELIANCE, TCS, INFY,
# 2026-09-03) under the in-bse-fin (BSE Ind-AS 2020-03-31) taxonomy. A
# concept/tag not in this table is simply not extracted -- never guessed at
# from an unverified taxonomy variant.
CANONICAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue_from_operations": ("RevenueFromOperations",),
    "other_income": ("OtherIncome",),
    "total_income": ("Income",),
    "profit_before_tax": ("ProfitBeforeTax",),
    "tax_expense": ("TaxExpense",),
    "profit_after_tax": ("ProfitLossForPeriod",),
    "eps_basic": (
        "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "BasicEarningsLossPerShareFromContinuingOperations",
    ),
    "eps_diluted": (
        "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "DilutedEarningsLossPerShareFromContinuingOperations",
    ),
    "face_value_per_share": ("FaceValueOfEquityShareCapital",),
    "paid_up_equity_capital": ("PaidUpValueOfEquityShareCapital",),
}

_CONCEPT_TO_CANONICAL = {
    concept: canonical for canonical, concepts in CANONICAL_CONCEPTS.items() for concept in concepts
}


class XbrlRejected(Exception):
    """Filing cannot be safely parsed/trusted. Callers must treat this as
    UNRESOLVED evidence, never fall back to a fabricated value."""


@dataclass(frozen=True)
class XbrlContext:
    context_id: str
    entity: str | None
    start: dt.date | None
    end: dt.date | None
    instant: dt.date | None
    has_scenario: bool


@dataclass(frozen=True)
class FinancialFact:
    symbol: str
    canonical_metric: str
    original_xbrl_concept: str
    context_id: str
    period_start: str | None
    period_end: str | None
    raw_value: float
    unit: str | None
    decimals: str | None
    normalized_value: float


def safe_parse_xbrl(content: bytes, *, source_url: str) -> ET.Element:
    """Bounded, DOCTYPE-refusing XML parse. Raises XbrlRejected instead of
    ever attempting to resolve an external entity or parse an oversized
    document."""
    if not source_url.lower().startswith("https://"):
        raise XbrlRejected(f"refusing non-HTTPS XBRL source: {source_url}")
    if len(content) > MAX_XBRL_BYTES:
        raise XbrlRejected(f"XBRL payload exceeds {MAX_XBRL_BYTES} bytes: {source_url}")
    head = content[:4096]
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise XbrlRejected(f"refusing XBRL with inline DOCTYPE/ENTITY: {source_url}")
    try:
        return _defused_ET.fromstring(content)
    except ET.ParseError as error:
        raise XbrlRejected(f"XBRL parse failed for {source_url}: {error}") from error


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _namespace(tag: str) -> str:
    return tag[1:].split("}")[0] if tag.startswith("{") else ""


def _parse_date(text: str | None) -> dt.date | None:
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text.strip()[:10])
    except ValueError:
        return None


def parse_contexts(root: ET.Element) -> dict[str, XbrlContext]:
    contexts: dict[str, XbrlContext] = {}
    for el in root:
        if _local(el.tag) != "context":
            continue
        context_id = el.attrib.get("id")
        if not context_id:
            continue
        entity: str | None = None
        start = end = instant = None
        has_scenario = False
        for child in el.iter():
            local = _local(child.tag)
            if local == "identifier":
                entity = (child.text or "").strip() or None
            elif local == "startDate":
                start = _parse_date(child.text)
            elif local == "endDate":
                end = _parse_date(child.text)
            elif local == "instant":
                instant = _parse_date(child.text)
            elif local == "scenario":
                has_scenario = True
        contexts[context_id] = XbrlContext(context_id, entity, start, end, instant, has_scenario)
    return contexts


def parse_facts(root: ET.Element, contexts: dict[str, XbrlContext], *, symbol: str) -> list[FinancialFact]:
    """Every plain (non-scenario, entity-matched) fact whose concept is in
    CANONICAL_CONCEPTS. Dimensional (segment-level) duplicates of the same
    concept are excluded so nothing gets double-counted against the
    consolidated/standalone top-line figure."""
    facts: list[FinancialFact] = []
    for el in root:
        local = _local(el.tag)
        if local in ("context", "unit", "schemaRef"):
            continue
        canonical = _CONCEPT_TO_CANONICAL.get(local)
        if canonical is None:
            continue
        context_id = el.attrib.get("contextRef")
        ctx = contexts.get(context_id) if context_id else None
        if ctx is None or ctx.has_scenario:
            continue
        if ctx.entity and ctx.entity.strip().upper() != symbol.strip().upper():
            continue
        raw_text = (el.text or "").strip()
        if not raw_text:
            continue
        try:
            raw_value = float(raw_text)
        except ValueError:
            continue
        namespace = _namespace(el.tag)
        facts.append(
            FinancialFact(
                symbol=symbol,
                canonical_metric=canonical,
                original_xbrl_concept=f"{{{namespace}}}{local}" if namespace else local,
                context_id=context_id,
                period_start=ctx.start.isoformat() if ctx.start else None,
                period_end=ctx.end.isoformat() if ctx.end else (ctx.instant.isoformat() if ctx.instant else None),
                raw_value=raw_value,
                unit=el.attrib.get("unitRef"),
                decimals=el.attrib.get("decimals"),
                # Every filing observed reports the full-precision rupee (or
                # per-share) amount as fact text; `decimals` is provenance
                # metadata about reporting precision, not a scale multiplier.
                normalized_value=raw_value,
            )
        )
    return facts


def select_quarter_context(
    contexts: dict[str, XbrlContext],
    *,
    symbol: str,
    quarter_start: dt.date,
    quarter_end: dt.date,
    facts: list[FinancialFact],
) -> str | None:
    """Return the context id holding the standalone-quarter figures, or None
    if this filing's context shape hasn't been verified (see module
    docstring). Never guesses."""
    candidates = [
        ctx.context_id
        for ctx in contexts.values()
        if ctx.start == quarter_start
        and ctx.end == quarter_end
        and not ctx.has_scenario
        and (ctx.entity is None or ctx.entity.strip().upper() == symbol.strip().upper())
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        by_context_revenue = {
            fact.context_id: abs(fact.raw_value)
            for fact in facts
            if fact.canonical_metric == "revenue_from_operations" and fact.context_id in candidates
        }
        if len(by_context_revenue) != 2 or by_context_revenue[candidates[0]] == by_context_revenue[candidates[1]]:
            # Can't disambiguate without a magnitude signal (or the two are
            # tied, which happens legitimately for a real Q1 filing that
            # doesn't need two contexts at all) -- refuse rather than guess.
            return None
        return min(candidates, key=lambda cid: by_context_revenue[cid])
    return None


def resolve_quarter(
    content: bytes,
    *,
    symbol: str,
    source_url: str,
    quarter_start: dt.date,
    quarter_end: dt.date,
) -> dict[str, Any] | None:
    """Full pipeline: safe-parse -> contexts -> facts -> select the
    standalone-quarter context -> canonical metric dict with provenance.
    Returns None (never a partially-fabricated dict) when the filing can't
    be safely resolved."""
    try:
        root = safe_parse_xbrl(content, source_url=source_url)
    except XbrlRejected:
        return None
    contexts = parse_contexts(root)
    facts = parse_facts(root, contexts, symbol=symbol)
    quarter_context_id = select_quarter_context(
        contexts,
        symbol=symbol,
        quarter_start=quarter_start,
        quarter_end=quarter_end,
        facts=facts,
    )
    if quarter_context_id is None:
        return None
    quarter_facts = {fact.canonical_metric: fact for fact in facts if fact.context_id == quarter_context_id}
    if not quarter_facts:
        return None
    result: dict[str, Any] = {
        "source_url": source_url,
        "context_id": quarter_context_id,
        "period_start": quarter_start.isoformat(),
        "period_end": quarter_end.isoformat(),
        "facts": {
            metric: {
                "value": fact.normalized_value,
                "original_xbrl_concept": fact.original_xbrl_concept,
                "unit": fact.unit,
                "decimals": fact.decimals,
            }
            for metric, fact in quarter_facts.items()
        },
    }
    eps_fact = quarter_facts.get("eps_basic")
    pat_fact = quarter_facts.get("profit_after_tax")
    revenue_fact = quarter_facts.get("revenue_from_operations")
    result["eps"] = eps_fact.normalized_value if eps_fact else None
    result["pat"] = pat_fact.normalized_value if pat_fact else None
    result["revenue"] = revenue_fact.normalized_value if revenue_fact else None
    return result
