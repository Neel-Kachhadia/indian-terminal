"""Regression tests for the canonical NSE<->BSE identity resolver.

ISINs used here are real-world values verified live (2026-09-03): RELIANCE's
agrees across NSE/BSE; ASIANPAINT/BAJFINANCE/360ONE's disagree. All four are
checksum-valid ISO 6166 identifiers -- confirmed here as much as it confirms
anything upstream.
"""

from __future__ import annotations

from valuation.identity_resolver import isin_check_digit_valid, resolve_identity


def test_check_digit_accepts_known_real_isins() -> None:
    assert isin_check_digit_valid("US0378331005")  # Apple
    assert isin_check_digit_valid("US5949181045")  # Microsoft
    assert isin_check_digit_valid("INE002A01018")  # RELIANCE, agrees NSE/BSE


def test_check_digit_rejects_mistyped_isin() -> None:
    assert not isin_check_digit_valid("INE002A01019")  # last digit flipped
    assert not isin_check_digit_valid("not-an-isin")
    assert not isin_check_digit_valid("INE002A0101")  # too short


def test_matching_isins_resolve_to_match_and_are_eligible() -> None:
    result = resolve_identity(
        requested_symbol="RELIANCE", issuer_name="Reliance Industries Ltd",
        nse_isin="INE002A01018", bse_isin="INE002A01018",
        bse_registry_recognizes_nse_isin=None,
    )
    assert result.resolution_status == "MATCH"
    assert result.valuation_eligible
    assert result.current_authoritative_isin == "INE002A01018"


def test_both_valid_but_disagreeing_and_bse_does_not_recognize_nse_isin() -> None:
    # Real observed case: ASIANPAINT. Both checksums valid; BSE's live
    # search index has zero record of NSE's reported ISIN.
    result = resolve_identity(
        requested_symbol="ASIANPAINT", issuer_name="Asian Paints Ltd",
        nse_isin="INE021A01018", bse_isin="INE021A01026",
        bse_registry_recognizes_nse_isin=False,
    )
    assert result.resolution_status == "NSE_STALE_IDENTIFIER"
    assert not result.valuation_eligible  # directional evidence, not proof
    assert result.current_authoritative_isin == "INE021A01026"


def test_both_valid_disagreeing_no_registry_signal_stays_unresolved() -> None:
    result = resolve_identity(
        requested_symbol="EXAMPLE", issuer_name="Example Ltd",
        nse_isin="INE021A01018", bse_isin="INE021A01026",
        bse_registry_recognizes_nse_isin=None,
    )
    assert result.resolution_status == "UNRESOLVED"
    assert not result.valuation_eligible


def test_nse_isin_fails_checksum_bse_wins() -> None:
    result = resolve_identity(
        requested_symbol="EXAMPLE", issuer_name="Example Ltd",
        nse_isin="INE021A01019", bse_isin="INE021A01026",  # NSE last digit wrong
        bse_registry_recognizes_nse_isin=None,
    )
    assert result.resolution_status == "NSE_STALE_IDENTIFIER"
    assert result.reason_code == "NSE_ISIN_FAILS_CHECKSUM"
    assert not result.valuation_eligible


def test_bse_isin_fails_checksum_nse_wins() -> None:
    result = resolve_identity(
        requested_symbol="EXAMPLE", issuer_name="Example Ltd",
        nse_isin="INE021A01018", bse_isin="INE021A01027",  # BSE last digit wrong
        bse_registry_recognizes_nse_isin=None,
    )
    assert result.resolution_status == "BSE_STALE_IDENTIFIER"
    assert result.reason_code == "BSE_ISIN_FAILS_CHECKSUM"
    assert not result.valuation_eligible


def test_both_isins_malformed_is_unresolved() -> None:
    result = resolve_identity(
        requested_symbol="EXAMPLE", issuer_name="Example Ltd",
        nse_isin="INE021A01019", bse_isin="INE021A01027",
        bse_registry_recognizes_nse_isin=None,
    )
    assert result.resolution_status == "UNRESOLVED"
    assert result.reason_code == "BOTH_ISINS_FAIL_CHECKSUM"


def test_missing_isin_on_one_side_is_unresolved() -> None:
    result = resolve_identity(
        requested_symbol="EXAMPLE", issuer_name="Example Ltd",
        nse_isin=None, bse_isin="INE021A01026",
        bse_registry_recognizes_nse_isin=None,
    )
    assert result.resolution_status == "UNRESOLVED"
    assert result.reason_code == "MISSING_ISIN_ON_ONE_SIDE"


def test_never_marks_valuation_eligible_without_match_or_proven_transition() -> None:
    # No resolution path other than MATCH/HISTORICAL_ISIN_TRANSITION may
    # ever set valuation_eligible True -- this is the hard safety
    # invariant the whole module exists to enforce.
    for status in (
        "NSE_STALE_IDENTIFIER", "BSE_STALE_IDENTIFIER", "SECURITY_CLASS_MISMATCH",
        "SOURCE_CONFLICT", "UNRESOLVED",
    ):
        from valuation.identity_resolver import IdentityResolution
        r = IdentityResolution(
            "X", None, "A", "B", None, None, None, "src", "2026-01-01", status, "test",
        )
        assert not r.valuation_eligible
