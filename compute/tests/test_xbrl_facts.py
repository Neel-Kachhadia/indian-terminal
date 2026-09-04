"""Regression tests for valuation.xbrl_facts against real NSE XBRL filings.

Fixtures under tests/fixtures/xbrl/ are unmodified filings fetched live from
nsearchives.nseindia.com for RELIANCE (2026-09-03) -- see manifest.json for
exact source URLs and filing dates. These pin real known-good numbers so a
future taxonomy/regex change is caught without a live network call.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from valuation.xbrl_facts import (
    XbrlRejected,
    parse_contexts,
    parse_facts,
    resolve_quarter,
    safe_parse_xbrl,
    select_quarter_context,
)

FIXTURES = Path(__file__).parent / "fixtures" / "xbrl"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text("utf-8"))


def _load(relating_to: str) -> bytes:
    return (FIXTURES / MANIFEST[relating_to]["file"]).read_bytes()


def _meta(relating_to: str) -> dict:
    return MANIFEST[relating_to]


def test_q1_filing_has_single_unambiguous_quarter_context() -> None:
    meta = _meta("First Quarter")
    content = _load("First Quarter")
    root = safe_parse_xbrl(content, source_url=meta["xbrl_source"])
    contexts = parse_contexts(root)
    facts = parse_facts(root, contexts, symbol="RELIANCE")
    ctx_id = select_quarter_context(
        contexts,
        symbol="RELIANCE",
        quarter_start=dt.date(2024, 4, 1),
        quarter_end=dt.date(2024, 6, 30),
        facts=facts,
    )
    assert ctx_id == "OneD"


def test_q4_filing_has_single_unambiguous_quarter_context() -> None:
    meta = _meta("Fourth Quarter")
    content = _load("Fourth Quarter")
    root = safe_parse_xbrl(content, source_url=meta["xbrl_source"])
    contexts = parse_contexts(root)
    facts = parse_facts(root, contexts, symbol="RELIANCE")
    ctx_id = select_quarter_context(
        contexts,
        symbol="RELIANCE",
        quarter_start=dt.date(2024, 1, 1),
        quarter_end=dt.date(2024, 3, 31),
        facts=facts,
    )
    assert ctx_id == "OneD"


def test_q3_filing_disambiguates_duplicate_dated_contexts_by_magnitude() -> None:
    meta = _meta("Third Quarter")
    content = _load("Third Quarter")
    root = safe_parse_xbrl(content, source_url=meta["xbrl_source"])
    contexts = parse_contexts(root)
    # The known real-world defect this test pins: OneD and FourD share the
    # exact same declared (start, end) even though FourD's fact values are
    # actually 9-month cumulative, not quarterly.
    assert contexts["OneD"].start == contexts["FourD"].start == dt.date(2024, 10, 1)
    assert contexts["OneD"].end == contexts["FourD"].end == dt.date(2024, 12, 31)
    facts = parse_facts(root, contexts, symbol="RELIANCE")
    ctx_id = select_quarter_context(
        contexts,
        symbol="RELIANCE",
        quarter_start=dt.date(2024, 10, 1),
        quarter_end=dt.date(2024, 12, 31),
        facts=facts,
    )
    assert ctx_id == "OneD"


def test_q3_resolve_quarter_extracts_real_revenue_pat_eps() -> None:
    meta = _meta("Third Quarter")
    content = _load("Third Quarter")
    result = resolve_quarter(
        content,
        symbol="RELIANCE",
        source_url=meta["xbrl_source"],
        quarter_start=dt.date(2024, 10, 1),
        quarter_end=dt.date(2024, 12, 31),
    )
    assert result is not None
    # Pinned against the real filing: RevenueFromOperations OneD = 1,282,600,000,000.00
    assert result["revenue"] == pytest.approx(1_282_600_000_000.00)
    assert result["pat"] == pytest.approx(87_210_000_000.00)
    assert result["context_id"] == "OneD"
    assert result["facts"]["revenue_from_operations"]["original_xbrl_concept"].endswith(
        "RevenueFromOperations"
    )


def test_q3_never_selects_the_cumulative_duplicate_context() -> None:
    meta = _meta("Third Quarter")
    content = _load("Third Quarter")
    result = resolve_quarter(
        content,
        symbol="RELIANCE",
        source_url=meta["xbrl_source"],
        quarter_start=dt.date(2024, 10, 1),
        quarter_end=dt.date(2024, 12, 31),
    )
    assert result is not None
    # FourD's revenue (~9-month cumulative) must never be the resolved figure.
    assert result["revenue"] != pytest.approx(3_966_450_000_000.00)


def test_refuses_oversized_payload() -> None:
    huge = b"<xbrli:xbrl>" + b"x" * (9 * 1024 * 1024) + b"</xbrli:xbrl>"
    with pytest.raises(XbrlRejected):
        safe_parse_xbrl(huge, source_url="https://nsearchives.nseindia.com/x.xml")


def test_refuses_non_https_source() -> None:
    with pytest.raises(XbrlRejected):
        safe_parse_xbrl(b"<xbrli:xbrl></xbrli:xbrl>", source_url="http://nsearchives.nseindia.com/x.xml")


def test_refuses_inline_doctype() -> None:
    malicious = b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><xbrli:xbrl>&xxe;</xbrli:xbrl>'
    with pytest.raises(XbrlRejected):
        safe_parse_xbrl(malicious, source_url="https://nsearchives.nseindia.com/x.xml")


def test_ambiguous_context_shape_resolves_to_none_not_a_guess() -> None:
    # Zero matching-window contexts (a filing shape never verified) must
    # refuse rather than pick something.
    ctx_id = select_quarter_context(
        {},
        symbol="RELIANCE",
        quarter_start=dt.date(2024, 10, 1),
        quarter_end=dt.date(2024, 12, 31),
        facts=[],
    )
    assert ctx_id is None
