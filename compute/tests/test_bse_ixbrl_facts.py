"""Regression tests for valuation.bse_ixbrl_facts against real BSE Integrated
Filing (Finance) inline-XBRL documents.

Fixtures under tests/fixtures/bse_xbrl/ are unmodified filings fetched live
from bseindia.com (2026-09-03) for the Jun-2026 quarter -- see manifest.json
for exact source URLs. Pinned against BSE's own TabResults_PAR dashboard
widget values as an independent cross-check.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from valuation.bse_ixbrl_facts import (
    BseXbrlRejected,
    parse_contexts,
    parse_facts,
    parse_identity_and_scope,
    resolve_quarter,
    safe_parse_ixbrl,
)

FIXTURES = Path(__file__).parent / "fixtures" / "bse_xbrl"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_consolidated_filing_extracts_real_revenue_pat_eps() -> None:
    result = resolve_quarter(
        _load("reliance_consolidated_jun2026.html"),
        source_url="https://www.bseindia.com/x.html",
        expected_isin="INE002A01018",
    )
    assert result is not None
    assert result["scope"] == "Consolidated"
    assert result["period_start"] == "2026-04-01"
    assert result["period_end"] == "2026-06-30"
    # Pinned against BSE's own dashboard widget: Revenue 3,11,850 Cr, EPS 15.48
    assert result["revenue"] == pytest.approx(3_118_500_000_000.0)
    assert result["eps"] == pytest.approx(15.48)


def test_standalone_filing_extracts_different_real_figures() -> None:
    result = resolve_quarter(
        _load("reliance_standalone_jun2026.html"),
        source_url="https://www.bseindia.com/x.html",
        expected_isin="INE002A01018",
    )
    assert result is not None
    assert result["scope"] == "Standalone"
    # Pinned against BSE's own dashboard widget: Revenue 1,66,013 Cr, EPS 9.81
    assert result["revenue"] == pytest.approx(1_660_130_000_000.0)
    assert result["eps"] == pytest.approx(9.81)
    # Standalone and consolidated must never be silently interchangeable.
    assert result["revenue"] != pytest.approx(3_118_500_000_000.0)


def test_scope_is_self_declared_not_inferred() -> None:
    content = _load("reliance_consolidated_jun2026.html")
    root = safe_parse_ixbrl(content, source_url="https://www.bseindia.com/x.html")
    identity = parse_identity_and_scope(root)
    assert identity["NatureOfReportStandaloneConsolidated"] == "Consolidated"
    assert identity["ISIN"] == "INE002A01018"
    assert identity["ScripCode"] == "500325"


def test_identity_mismatch_refuses_rather_than_attributes_wrong_issuer() -> None:
    result = resolve_quarter(
        _load("reliance_consolidated_jun2026.html"),
        source_url="https://www.bseindia.com/x.html",
        expected_isin="INE467B01029",  # TCS ISIN, deliberately wrong for this filing
    )
    assert result is None


def test_banking_document_has_no_revenue_from_operations_concept() -> None:
    # Banks report InterestEarned/InterestExpended, not RevenueFromOperations
    # -- confirmed live, 2026-09-03. Some generic concepts (OtherIncome,
    # TaxExpense, ProfitLossForPeriod, EPS) are shared across both document
    # types, but revenue_from_operations must never resolve for a bank --
    # that's the concrete signal the valuation engine's financial-sector
    # gate has nothing spurious to accidentally trust.
    content = _load("hdfcbank_consolidated_jun2026.html")
    root = safe_parse_ixbrl(content, source_url="https://www.bseindia.com/x.html")
    contexts = parse_contexts(root)
    facts = parse_facts(root, contexts)
    assert not any(fact.canonical_metric == "revenue_from_operations" for fact in facts)


def test_refuses_oversized_payload() -> None:
    huge = b"<html>" + b"x" * (9 * 1024 * 1024) + b"</html>"
    with pytest.raises(BseXbrlRejected):
        safe_parse_ixbrl(huge, source_url="https://www.bseindia.com/x.html")


def test_refuses_non_https_source() -> None:
    with pytest.raises(BseXbrlRejected):
        safe_parse_ixbrl(b"<html></html>", source_url="http://www.bseindia.com/x.html")


def test_refuses_inline_doctype() -> None:
    malicious = b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><html>&xxe;</html>'
    with pytest.raises(BseXbrlRejected):
        safe_parse_ixbrl(malicious, source_url="https://www.bseindia.com/x.html")


def test_q4_filing_selects_quarter_context_not_annual_cumulative() -> None:
    # This single real filing carries BOTH the Jan-Mar standalone quarter
    # (context OneD) and the Apr-Mar full-year cumulative (context FourD)
    # -- verified live (RELIANCE Q4 FY26, 2026-09-03). Unlike NSE's
    # equivalent bug, BSE's own context dates here are NOT corrupted, so
    # exact-date matching against the caller-supplied expected window must
    # select the real quarter, never the ~4x-larger annual figure.
    content = _load("reliance_consolidated_q4_and_annual_mar2026.html")
    result = resolve_quarter(
        content,
        source_url="https://www.bseindia.com/x.html",
        expected_isin="INE002A01018",
        expected_period_start=dt.date(2026, 1, 1),
        expected_period_end=dt.date(2026, 3, 31),
    )
    assert result is not None
    assert result["period_start"] == "2026-01-01"
    assert result["period_end"] == "2026-03-31"
    assert result["revenue"] == pytest.approx(2_986_210_000_000.0)
    assert result["revenue"] != pytest.approx(10_756_750_000_000.0)  # the annual figure


def test_q4_filing_without_expected_window_refuses_ambiguous_two_context_doc() -> None:
    # Without a caller-supplied expected window, a document with two
    # candidate duration contexts must refuse rather than guess which one
    # is the real quarter.
    content = _load("reliance_consolidated_q4_and_annual_mar2026.html")
    result = resolve_quarter(
        content, source_url="https://www.bseindia.com/x.html", expected_isin="INE002A01018"
    )
    assert result is None


def test_missing_scope_declaration_resolves_to_none() -> None:
    minimal = (
        b"<html xmlns='http://www.w3.org/1999/xhtml' "
        b"xmlns:ix='http://www.xbrl.org/2013/inlineXBRL' "
        b"xmlns:xbrli='http://www.xbrl.org/2003/instance'>"
        b"<xbrli:context id='OneD'><xbrli:entity><xbrli:identifier>500325</xbrli:identifier></xbrli:entity>"
        b"<xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate>"
        b"</xbrli:period></xbrli:context>"
        b"<ix:nonFraction name='in-capmkt:RevenueFromOperations' contextRef='OneD' unitRef='INR' scale='7'>100</ix:nonFraction>"
        b"</html>"
    )
    result = resolve_quarter(minimal, source_url="https://www.bseindia.com/x.html")
    assert result is None
