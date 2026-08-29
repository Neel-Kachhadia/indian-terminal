from __future__ import annotations

from signals.cat03_options import _iv_percentile, _iv_rank


def test_iv_rank_unavailable_when_value_missing() -> None:
    assert _iv_rank(None, [10.0, 12.0, 14.0, 16.0, 18.0]) is None


def test_iv_rank_unavailable_when_history_too_short() -> None:
    assert _iv_rank(15.0, [10.0, 12.0]) is None


def test_iv_rank_unavailable_when_history_degenerate() -> None:
    assert _iv_rank(15.0, [15.0, 15.0, 15.0, 15.0, 15.0]) is None


def test_iv_rank_computed_when_data_sufficient() -> None:
    result = _iv_rank(15.0, [10.0, 12.0, 14.0, 16.0, 18.0])
    assert result == 62.5


def test_iv_percentile_unavailable_when_value_missing() -> None:
    assert _iv_percentile(None, [10.0, 12.0, 14.0, 16.0, 18.0]) is None


def test_iv_percentile_unavailable_when_history_too_short() -> None:
    assert _iv_percentile(15.0, [10.0, 12.0]) is None


def test_iv_percentile_computed_when_data_sufficient() -> None:
    result = _iv_percentile(15.0, [10.0, 12.0, 14.0, 16.0, 18.0])
    assert result == 60.0
