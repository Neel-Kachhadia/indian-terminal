from __future__ import annotations

import pytest

from signals.options_signals import _compute_gex, compute_options_signals


def test_gex_scales_from_validated_lot_size_not_hardcoded_contract_size() -> None:
    legs = [
        {
            "symbol": "NIFTY",
            "strike": 25_000,
            "option_type": "CE",
            "oi": 100,
            "gamma": 0.001,
            "iv": None,
            "delta": 0.5,
        }
    ]

    result_65, _ = _compute_gex(legs, 25_000, 65, True)
    result_75, _ = _compute_gex(legs, 25_000, 75, True)

    assert result_65 == pytest.approx(result_75 * 65 / 75)


def test_options_signals_reject_missing_or_invalid_lot_size() -> None:
    with pytest.raises(ValueError, match="positive lot_size"):
        compute_options_signals([], 25_000, "03-Sep-2026", 0)


def test_missing_chain_fields_remain_unavailable_not_neutral_or_zero() -> None:
    result = compute_options_signals(
        [
            {
                "symbol": "NIFTY",
                "strike": 25_000,
                "option_type": "CE",
                "oi": None,
                "iv": None,
                "gamma": None,
                "delta": None,
            }
        ],
        25_000,
        "03-Sep-2026",
        65,
    )

    assert result["oi_complete"] is False
    assert result["pcr"] is None
    assert result["total_call_oi"] is None
    assert result["gex"] is None
    assert result["iv_rank"] is None
    assert result["iv_percentile"] is None
    assert result["expected_move"] is None
