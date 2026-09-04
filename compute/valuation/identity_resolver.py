"""Canonical NSE<->BSE issuer identity resolver.

Context: NSE's frozen (since ~Jan-2025) `corporates-financial-results`
dataset reports an ISIN for many symbols that disagrees with BSE's live
security-master ISIN for the same symbol -- verified live for 69/210
canonical equities (2026-09-03). A real dual-listed security has exactly
ONE depository-issued (NSDL/CDSL) ISIN shared by both exchanges, so this
disagreement is evidence of a data-quality defect on one side, not a
legitimate dual-identity -- but WHICH side is wrong cannot be determined
without an authoritative depository lookup this project has no free public
access to (a genuine EXTERNAL_BOUNDARY).

This module applies the two objective, self-contained checks that ARE
available without any such lookup:

1. ISO 6166 check-digit validation on both candidate ISINs (detects
   malformed/mistyped data; does NOT arbitrate between two well-formed
   ISINs).
2. Whether BSE's own live security-master search recognizes NSE's
   candidate ISIN at all (a real signal, not a guess: if BSE's active
   registry has zero record of it, NSE's copy is not BSE's current
   identifier for that security).

Per policy: only MATCH or a PROVEN historical transition unlocks valuation
eligibility. Directional evidence (e.g. NSE_STALE_IDENTIFIER) is recorded
and surfaced, but does NOT by itself make a symbol valuation-eligible --
this module never resolves an ambiguity into eligibility, only classifies
it more precisely than a bare "unresolved".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

RESOLUTION_STATUSES = (
    "MATCH",
    "HISTORICAL_ISIN_TRANSITION",
    "NSE_STALE_IDENTIFIER",
    "BSE_STALE_IDENTIFIER",
    "SECURITY_CLASS_MISMATCH",
    "SOURCE_CONFLICT",
    "UNRESOLVED",
)

VALUATION_ELIGIBLE_STATUSES = frozenset({"MATCH", "HISTORICAL_ISIN_TRANSITION"})


def isin_check_digit_valid(isin: str) -> bool:
    """ISO 6166 Luhn check-digit validation. Detects malformed/mistyped
    ISINs; does not (and cannot) prove which of two well-formed ISINs is
    the currently-correct one for a given security."""
    isin = (isin or "").strip().upper()
    if len(isin) != 12 or not isin[:2].isalpha():
        return False
    body, check = isin[:11], isin[11]
    if not check.isdigit():
        return False
    digits = ""
    for ch in body:
        if ch.isdigit():
            digits += ch
        elif ch.isalpha():
            digits += str(ord(ch) - ord("A") + 10)
        else:
            return False
    total = 0
    for i, d in enumerate(digits[::-1]):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return str((10 - (total % 10)) % 10) == check


@dataclass(frozen=True)
class IdentityResolution:
    requested_symbol: str
    issuer_name: str | None
    nse_isin: str | None
    bse_isin: str | None
    current_authoritative_isin: str | None
    previous_isin: str | None
    transition_date: str | None
    evidence_source: str
    evidence_date: str
    resolution_status: str
    reason_code: str

    @property
    def valuation_eligible(self) -> bool:
        return self.resolution_status in VALUATION_ELIGIBLE_STATUSES


def resolve_identity(
    *,
    requested_symbol: str,
    issuer_name: str | None,
    nse_isin: str | None,
    bse_isin: str | None,
    bse_registry_recognizes_nse_isin: bool | None,
) -> IdentityResolution:
    """Classify one symbol's NSE-vs-BSE ISIN evidence. `bse_registry_recognizes_nse_isin`
    is the result of searching BSE's live security master for the NSE-reported
    ISIN (None if that check wasn't run -- e.g. no NSE ISIN to check)."""
    now = datetime.now(timezone.utc).isoformat()
    if not nse_isin or not bse_isin:
        return IdentityResolution(
            requested_symbol, issuer_name, nse_isin, bse_isin, None, None, None,
            "NSE_STALE_STRUCTURED_DATASET / BSE_LIVE_SECURITY_MASTER", now,
            "UNRESOLVED", "MISSING_ISIN_ON_ONE_SIDE",
        )
    if nse_isin.strip().upper() == bse_isin.strip().upper():
        return IdentityResolution(
            requested_symbol, issuer_name, nse_isin, bse_isin, nse_isin, None, None,
            "NSE_STALE_STRUCTURED_DATASET / BSE_LIVE_SECURITY_MASTER", now,
            "MATCH", "ISIN_AGREES_ACROSS_EXCHANGES",
        )
    nse_valid = isin_check_digit_valid(nse_isin)
    bse_valid = isin_check_digit_valid(bse_isin)
    if not nse_valid and bse_valid:
        return IdentityResolution(
            requested_symbol, issuer_name, nse_isin, bse_isin, bse_isin, None, None,
            "ISO_6166_CHECK_DIGIT", now, "NSE_STALE_IDENTIFIER", "NSE_ISIN_FAILS_CHECKSUM",
        )
    if nse_valid and not bse_valid:
        return IdentityResolution(
            requested_symbol, issuer_name, nse_isin, bse_isin, nse_isin, None, None,
            "ISO_6166_CHECK_DIGIT", now, "BSE_STALE_IDENTIFIER", "BSE_ISIN_FAILS_CHECKSUM",
        )
    if not nse_valid and not bse_valid:
        return IdentityResolution(
            requested_symbol, issuer_name, nse_isin, bse_isin, None, None, None,
            "ISO_6166_CHECK_DIGIT", now, "UNRESOLVED", "BOTH_ISINS_FAIL_CHECKSUM",
        )
    # Both well-formed (the observed case for all 69 live ambiguities,
    # 2026-09-03) -- checksum validity alone cannot arbitrate. Use the
    # BSE-registry-recognition signal as directional (not proof) evidence.
    if bse_registry_recognizes_nse_isin is False:
        return IdentityResolution(
            requested_symbol, issuer_name, nse_isin, bse_isin, bse_isin, None, None,
            "BSE_LIVE_SECURITY_MASTER_SEARCH", now, "NSE_STALE_IDENTIFIER",
            "BSE_REGISTRY_DOES_NOT_RECOGNIZE_NSE_ISIN_NOT_PROVEN",
        )
    return IdentityResolution(
        requested_symbol, issuer_name, nse_isin, bse_isin, None, None, None,
        "NONE_AVAILABLE_WITHOUT_NSDL_CDSL_ACCESS", now, "UNRESOLVED",
        "NO_AUTHORITATIVE_DEPOSITORY_ACCESS_AVAILABLE",
    )
