"""Extracts real financial facts from BSE's Integrated Filing (Finance)
inline-XBRL documents.

Why this module exists: NSE's structured `corporates-financial-results` /
`results-comparision` endpoints stopped updating after the 31-Dec-2024
quarter (verified live, 2026-09-03, systemic across 9 large-caps) while raw
disclosure continued normally -- root cause: SEBI mandated a new
"Integrated Filing (Finance)" regime starting the Mar-2025 quarter, and
NSE's legacy structured endpoints were never migrated to it. BSE's
`Integratedfinancedata` API *was* migrated and is genuinely current
(verified live through the Jun-2026 quarter across RELIANCE, TCS, INFY,
HDFCBANK, SBIN, ITC, BAJFINANCE, LT, SWIGGY).

This new `in-capmkt` (SEBI, 2026-01-31) taxonomy is unambiguous by
construction, unlike the old NSE `in-bse-fin` one:
  - Consolidated vs Standalone are separate filing rows/documents, each
    self-declaring its own scope via the `NatureOfReportStandaloneConsolidated`
    fact -- no magnitude-based disambiguation needed.
  - Quarter vs Year-to-date/annual are separate filing rows/documents
    (`Quarter_Name` literally contains "-Qtr-" or "-Yearly-"), each with
    their own real (non-duplicated) context period.
  - Banks/NBFCs file under a distinct document type (`IFBanking`, vs
    `IFIndas` for non-financial issuers) with bank-specific concepts
    (InterestEarned/InterestExpended/...) rather than RevenueFromOperations
    -- confirmed live against HDFCBANK. The existing valuation engine
    already refuses P/E-only valuation for financial-sector issuers
    upstream of this module, so only the generic (Ind-AS) concept table is
    wired into the quarterly-EPS/PAT/revenue pipeline for now; the banking
    table is captured here for provenance completeness only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

import defusedxml.ElementTree as _defused_ET  # hardened parse entrypoint; ET retained for types/ParseError

MAX_XBRL_BYTES = 8 * 1024 * 1024

_XBRLI = "{http://www.xbrl.org/2003/instance}"
_IX = "{http://www.xbrl.org/2013/inlineXBRL}"

# Concept names verified live against real BSE Integrated Filing documents
# (RELIANCE Consolidated + Standalone, Jun-2026 quarter, 2026-09-03) under
# the in-capmkt (SEBI 2026-01-31) taxonomy, Ind-AS (non-financial) template.
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

# Banking-template (IFBanking document type) concepts, verified live against
# HDFCBANK Jun-2026 quarter. Captured for provenance/future use; not wired
# into the P/E TTM pipeline -- the valuation engine already routes
# financial-sector issuers to BANK_VALUATION_INPUTS_UNAVAILABLE upstream.
CANONICAL_CONCEPTS_BANKING: dict[str, tuple[str, ...]] = {
    "interest_earned": ("InterestEarned",),
    "interest_expended": ("InterestExpended",),
    "operating_profit_before_provisions": ("OperatingProfitBeforeProvisionAndContingencies",),
    "provisions_other_than_tax": ("ProvisionsOtherThanTaxAndContingencies",),
    "profit_after_tax": ("ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",),
}

_CONCEPT_TO_CANONICAL = {
    concept: canonical for canonical, concepts in CANONICAL_CONCEPTS.items() for concept in concepts
}

_IDENTITY_CONCEPTS = ("ISIN", "ScripCode", "Symbol", "NameOfTheCompany")
_SCOPE_CONCEPT = "NatureOfReportStandaloneConsolidated"


class BseXbrlRejected(Exception):
    """Filing cannot be safely parsed/trusted. Callers must treat this as
    UNRESOLVED evidence, never fall back to a fabricated value."""


@dataclass(frozen=True)
class BseContext:
    context_id: str
    entity: str | None
    start: dt.date | None
    end: dt.date | None
    instant: dt.date | None
    has_scenario: bool


@dataclass(frozen=True)
class BseFinancialFact:
    canonical_metric: str
    original_xbrl_concept: str
    context_id: str
    period_start: str | None
    period_end: str | None
    raw_value: float
    unit: str | None
    decimals: str | None
    scale: str | None
    normalized_value: float


def safe_parse_ixbrl(content: bytes, *, source_url: str) -> ET.Element:
    if not source_url.lower().startswith("https://"):
        raise BseXbrlRejected(f"refusing non-HTTPS source: {source_url}")
    if len(content) > MAX_XBRL_BYTES:
        raise BseXbrlRejected(f"payload exceeds {MAX_XBRL_BYTES} bytes: {source_url}")
    head = content[:4096]
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise BseXbrlRejected(f"refusing document with inline DOCTYPE/ENTITY: {source_url}")
    try:
        return _defused_ET.fromstring(content)
    except ET.ParseError as error:
        raise BseXbrlRejected(f"parse failed for {source_url}: {error}") from error


def _parse_date(text: str | None) -> dt.date | None:
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text.strip()[:10])
    except ValueError:
        return None


def parse_contexts(root: ET.Element) -> dict[str, BseContext]:
    contexts: dict[str, BseContext] = {}
    for el in root.iter(f"{_XBRLI}context"):
        context_id = el.attrib.get("id")
        if not context_id:
            continue
        entity: str | None = None
        start = end = instant = None
        has_scenario = False
        for child in el.iter():
            local = child.tag.split("}")[-1]
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
        contexts[context_id] = BseContext(context_id, entity, start, end, instant, has_scenario)
    return contexts


def parse_identity_and_scope(root: ET.Element) -> dict[str, str]:
    """Non-numeric self-declared facts: ISIN/ScripCode/Symbol/company name
    and the Consolidated-vs-Standalone scope declaration. Used to positively
    confirm issuer identity and accounting scope rather than inferring
    either from filenames or list position."""
    result: dict[str, str] = {}
    for el in root.iter(f"{_IX}nonNumeric"):
        name = el.attrib.get("name", "")
        local = name.split(":")[-1]
        if local in _IDENTITY_CONCEPTS or local == _SCOPE_CONCEPT:
            text = (el.text or "").strip()
            if text:
                result.setdefault(local, text)
    return result


def parse_facts(root: ET.Element, contexts: dict[str, BseContext]) -> list[BseFinancialFact]:
    facts: list[BseFinancialFact] = []
    for el in root.iter(f"{_IX}nonFraction"):
        name = el.attrib.get("name", "")
        local = name.split(":")[-1]
        canonical = _CONCEPT_TO_CANONICAL.get(local)
        if canonical is None:
            continue
        context_id = el.attrib.get("contextRef")
        ctx = contexts.get(context_id) if context_id else None
        if ctx is None or ctx.has_scenario:
            continue
        raw_text = (el.text or "").strip().replace(",", "")
        if not raw_text:
            continue
        try:
            raw_value = float(raw_text)
        except ValueError:
            continue
        if el.attrib.get("sign") == "-":
            raw_value = -raw_value
        scale = el.attrib.get("scale")
        multiplier = 10.0 ** int(scale) if scale not in (None, "") else 1.0
        facts.append(
            BseFinancialFact(
                canonical_metric=canonical,
                original_xbrl_concept=name,
                context_id=context_id,
                period_start=ctx.start.isoformat() if ctx.start else None,
                period_end=ctx.end.isoformat() if ctx.end else (ctx.instant.isoformat() if ctx.instant else None),
                raw_value=raw_value,
                unit=el.attrib.get("unitRef"),
                decimals=el.attrib.get("decimals"),
                scale=scale,
                normalized_value=raw_value * multiplier,
            )
        )
    return facts


def resolve_quarter(
    content: bytes,
    *,
    source_url: str,
    expected_isin: str | None = None,
    expected_period_start: dt.date | None = None,
    expected_period_end: dt.date | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Full pipeline for one BSE Integrated Filing document. Each document
    is scope-specific by construction (Consolidated vs Standalone are
    separate documents), but a Q4 document carries BOTH the standalone
    quarter AND the cumulative full-year figures as two distinct,
    correctly-dated duration contexts (verified live against RELIANCE
    Q4 FY26, 2026-09-03 -- unlike NSE's old taxonomy, BSE's own context
    dates are NOT corrupted here, so exact-date matching is sufficient and
    no magnitude disambiguation is needed). When the caller supplies the
    expected quarter window, the context matching those exact dates is
    selected; otherwise this falls back to requiring exactly one duration
    context in the whole document (true for Q1-Q3 filings). Returns None
    (never a partially-fabricated dict) when identity can't be confirmed,
    the expected period isn't found, or no canonical facts resolve."""
    if diagnostics is None:
        diagnostics = {}
    try:
        root = safe_parse_ixbrl(content, source_url=source_url)
    except BseXbrlRejected:
        diagnostics["reason"] = "PARSE_REJECTED"
        return None
    identity = parse_identity_and_scope(root)
    if expected_isin and identity.get("ISIN") and identity["ISIN"].strip().upper() != expected_isin.strip().upper():
        # Identity mismatch is a hard fail -- never attribute one issuer's
        # filing to another.
        diagnostics["reason"] = "IDENTITY_MISMATCH"
        return None
    scope = identity.get(_SCOPE_CONCEPT)
    if scope not in ("Consolidated", "Standalone"):
        diagnostics["reason"] = "NO_SCOPE_DECLARATION"
        return None
    contexts = parse_contexts(root)
    facts = parse_facts(root, contexts)
    if not facts:
        diagnostics["reason"] = "NO_CANONICAL_CONCEPTS"
        return None
    duration_ids = {f.context_id for f in facts if contexts[f.context_id].start and contexts[f.context_id].end}
    if expected_period_start and expected_period_end:
        duration_ids = {
            cid
            for cid in duration_ids
            if contexts[cid].start == expected_period_start and contexts[cid].end == expected_period_end
        }
    if len(duration_ids) != 1:
        diagnostics["reason"] = "PERIOD_AMBIGUOUS"
        return None
    context_id = next(iter(duration_ids))
    quarter_facts = {f.canonical_metric: f for f in facts if f.context_id == context_id}
    ctx = contexts[context_id]
    result: dict[str, Any] = {
        "source_url": source_url,
        "context_id": context_id,
        "period_start": ctx.start.isoformat() if ctx.start else None,
        "period_end": ctx.end.isoformat() if ctx.end else None,
        "scope": scope,
        "isin": identity.get("ISIN"),
        "scripcode": identity.get("ScripCode"),
        "symbol": identity.get("Symbol"),
        "facts": {
            metric: {
                "value": fact.normalized_value,
                "original_xbrl_concept": fact.original_xbrl_concept,
                "unit": fact.unit,
                "decimals": fact.decimals,
                "scale": fact.scale,
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
