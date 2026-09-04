from __future__ import annotations

import datetime as dt

from valuation.engine import build_valuation


AS_OF = dt.date(2026, 9, 3)
DATES = ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30")


def financials(eps_ttm: float = 10.0) -> dict:
    return {
        "source": "NSE_CORPORATE_FINANCIAL_RESULTS",
        "quarters": [
            {"report_date": day, "eps": eps_ttm / 4.0, "revenue": 100.0}
            for day in DATES
        ],
    }


def peers() -> list[dict]:
    multiples = (8.0, 9.0, 10.0, 11.0, 12.0, 10.5, 9.5, 2000.0)
    return [
        {
            "symbol": f"PEER{index}",
            "sector": "INDUSTRIALS",
            "market_price": multiple * 10.0,
            "market_price_as_of": "2026-09-02",
            "financials": financials(10.0),
        }
        for index, multiple in enumerate(multiples)
    ]


def value(price: float, **overrides):
    args = {
        "symbol": "TARGET",
        "sector": "INDUSTRIALS",
        "market_price": price,
        "market_price_as_of": "2026-09-02",
        "quote_quality": "PREVIOUS_CLOSE",
        "financials": financials(),
        "peers": peers(),
        "evaluated_on": AS_OF,
    }
    args.update(overrides)
    return build_valuation(**args)


def test_classification_uses_fair_value_range_not_point_estimate_cutoff() -> None:
    undervalued = value(50.0)
    fairly_valued = value(100.0)
    overvalued = value(160.0)

    assert undervalued["classification"] == "UNDERVALUED"
    assert fairly_valued["classification"] == "FAIRLY_VALUED"
    assert overvalued["classification"] == "OVERVALUED"
    assert undervalued["market_price"] < undervalued["fair_value_low"]
    assert overvalued["market_price"] > overvalued["fair_value_high"]


def test_relative_valuation_excludes_target_and_robust_outlier() -> None:
    result = value(100.0)

    assert result["status"] == "VALID"
    assert result["methods_used"] == ["RELATIVE_PE_TTM"]
    assert result["methods"][0]["excluded_outlier_count"] == 1
    assert result["peer_count"] == 7
    assert 90.0 <= result["fair_value"] <= 110.0
    assert result["evidence_score_semantics"].endswith("not calibrated probability")


def test_negative_earnings_fail_closed() -> None:
    result = value(100.0, financials=financials(-10.0))

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["classification"] == "INDETERMINATE"
    assert result["fair_value"] is None
    assert result["reason"] == "NON_POSITIVE_TTM_EARNINGS"


def test_financial_sector_rejects_pe_only_method() -> None:
    result = value(100.0, sector="BANKING")

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["reason"] == "BOOK_VALUE_AND_ROE_REQUIRED_FOR_FINANCIAL_SECTOR"
    assert result["fair_value"] is None


def test_missing_quarter_and_stale_fundamentals_fail_closed() -> None:
    missing = financials()
    missing["quarters"] = missing["quarters"][:3]
    stale = financials()
    for row in stale["quarters"]:
        year, month, day = map(int, row["report_date"].split("-"))
        row["report_date"] = f"{year - 2:04d}-{month:02d}-{day:02d}"

    missing_result = value(100.0, financials=missing)
    stale_result = value(100.0, financials=stale)

    assert missing_result["reason"] == "FOUR_DISTINCT_QUARTERS_REQUIRED"
    assert stale_result["reason"] == "STALE_FUNDAMENTALS"


def test_no_peer_group_means_no_fair_value() -> None:
    result = value(100.0, peers=[])

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["reason"] == "INADEQUATE_VALID_SECTOR_PEERS"
    assert result["peer_count"] == 0


def test_negative_eps_survives_as_secondary_reason_when_sector_missing() -> None:
    # A real fact about the target (negative TTM EPS) must not be lost just
    # because an earlier gate (missing sector) fires first and becomes the
    # primary reason -- this is the SWIGGY-shaped case: sector unavailable
    # AND negative EPS are both true, but only one can be primary.
    negative = financials(-40.0)
    result = value(100.0, sector=None, financials=negative)

    assert result["reason"] == "SECTOR_CLASSIFICATION_UNAVAILABLE"
    assert "NEGATIVE_OR_ZERO_TTM_EPS" in result["secondary_reasons"]


def test_negative_eps_is_still_primary_when_sector_is_available() -> None:
    negative = financials(-40.0)
    result = value(100.0, financials=negative)

    assert result["reason"] == "NON_POSITIVE_TTM_EARNINGS"
    # Already the primary reason -- must not be duplicated into secondary.
    assert "NON_POSITIVE_TTM_EARNINGS" not in result["secondary_reasons"]


def test_valid_result_carries_empty_secondary_reasons() -> None:
    result = value(80.0)
    assert result["status"] == "VALID"
    assert result["secondary_reasons"] == []
