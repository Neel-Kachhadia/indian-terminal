"""Regression contract for the NSE corporate-data endpoint rename.

NSE retired `corp-info` (corpType=announcements|filings), `financials-results`,
and `annual-report?symbol=` -- all four returned HTTP 404 for every symbol,
including RELIANCE, as of 2026-08-25 (verified live against nseindia.com's
own bundled JS and a direct request). The replacements were confirmed live
to return 200 with symbol-scoped, field-compatible payloads:

  corp-info?corpType=announcements  -> corporate-announcements?index=equities
  corp-info?corpType=filings        -> corporate-announcements?index=equities
                                        (NSE has no separate filings feed left)
  financials-results                -> corporates-financial-results?index=equities
  annual-report?symbol=              -> annual-reports?index=equities&symbol=

This test pins the source strings so a future accidental revert (or another
stale-endpoint drift) is caught without a live network call.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_nse_corporate_endpoints_use_current_contract_not_retired_ones() -> None:
    source = (ROOT / "compute" / "main.py").read_text("utf-8")

    assert "api/corporate-announcements" in source
    assert "api/corporates-financial-results" in source
    assert "api/annual-reports" in source

    assert "api/corp-info" not in source
    assert '"https://www.nseindia.com/api/financials-results"' not in source
    assert "api/annual-report?symbol=" not in source
