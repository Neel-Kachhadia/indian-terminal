"""Regression oracle for the m01_vol_regime premature-maturity/lookahead
defect: vol_regime_target() used pandas .std(skipna=True) over its 5
future-session window, so a row with only 1 of 5 future sessions available
produced a finite statistic (std of a single sample, ddof=0, is 0.0) instead
of NaN. Chained through resolve_pending_prediction_outcomes(), this let an
m01 prediction resolve after 1 elapsed trading session instead of the
intended 5 -- a real lookahead/premature-maturity defect, not merely an
unresolved edge case.

Cases A-P below are the required regression matrix. Cases that only need the
target function's own maturity gating (no DB/WriterService involved) call
vol_regime_target() directly through the exact object
(_OhlcvOnlyContext-shaped) the production resolver uses -- this is the real
target computation, not a mock. Cases that need to prove the full production
pipeline (WriterService -> resolve_pending_prediction_outcomes -> target
function) never resolves prematurely go through submit_write for real.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from db import duckdb_client
from ml.feature_engineering import MLContext, vol_regime_target


class _Ctx:
    """Same minimal shape ml.monitor._OhlcvOnlyContext uses: vol_regime_target
    only ever reads .ohlcv."""

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        self.ohlcv = ohlcv


def _frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "close": closes})


def _label_on(dates: list[str], closes: list[float], anchor: str) -> object:
    labels = vol_regime_target(_Ctx(_frame(dates, closes)))
    anchor_date = pd.to_datetime(anchor).date()
    if anchor_date not in labels.index:
        return None
    value = labels.get(anchor_date)
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


# Base calendar: 20 consecutive business days from 2026-01-05 (Monday).
_BASE_DATES = [d.date().isoformat() for d in pd.bdate_range("2026-01-05", periods=20)]
# Flat closes so any resolved label is deterministically "LOW" (realised vol
# 0% < 10%), isolating the maturity-gating behaviour under test from the
# volatility-bucketing thresholds.
_FLAT_CLOSES = [100.0] * 20
_ANCHOR = _BASE_DATES[0]


@pytest.mark.contract
@pytest.mark.parametrize("n_future", [1, 2, 3, 4])
def test_partial_future_window_stays_unresolved(n_future: int) -> None:
    """A. B. C. D. -- 1, 2, 3, 4 future sessions must never mature."""
    dates = _BASE_DATES[: 1 + n_future]
    closes = _FLAT_CLOSES[: 1 + n_future]
    assert _label_on(dates, closes, _ANCHOR) is None, (
        f"{n_future} future session(s) present must NOT be resolvable (pre-fix defect: "
        "std(skipna=True) treated a partial window as complete)"
    )


@pytest.mark.contract
def test_exactly_five_future_sessions_resolves() -> None:
    """E. exactly 5 future trading sessions -> resolves."""
    dates = _BASE_DATES[:6]
    closes = _FLAT_CLOSES[:6]
    assert _label_on(dates, closes, _ANCHOR) == "LOW"


@pytest.mark.contract
def test_five_sessions_spanning_weekend_counts_trading_sessions() -> None:
    """F. Business-day date range already skips Sat/Sun; vol_regime_target
    operates on row position within the OHLCV frame (which by construction
    only ever contains real trading sessions), not on calendar days, so a
    5-row window that spans a weekend still requires exactly 5 real
    sessions."""
    # _BASE_DATES[0..5] spans Mon 2026-01-05 .. Mon 2026-01-12, crossing one
    # weekend, and contains exactly anchor + 5 future trading sessions.
    dates = _BASE_DATES[:6]
    assert dates[5] == "2026-01-12"
    closes = _FLAT_CLOSES[:6]
    assert _label_on(dates, closes, _ANCHOR) == "LOW"
    # Only 4 of those (still spanning the same weekend) must not resolve.
    assert _label_on(dates[:5], closes[:5], _ANCHOR) is None


@pytest.mark.contract
def test_five_sessions_spanning_holiday_gap_counts_eligible_sessions() -> None:
    """G. A holiday means no OHLCV row is ever collected for that calendar
    date (bhavcopy simply has no row) -- simulate that missing-row gap
    directly and confirm the target still requires 5 present rows, not 5
    calendar days."""
    dates = list(_BASE_DATES[:7])
    del dates[3]  # remove one calendar date entirely, as a holiday would
    assert len(dates) == 6  # anchor + 5 real sessions despite the gap
    closes = _FLAT_CLOSES[: len(dates)]
    assert _label_on(dates, closes, _ANCHOR) == "LOW"
    assert _label_on(dates[:5], closes[:5], _ANCHOR) is None


@pytest.mark.contract
def test_one_required_session_missing_stays_unresolved() -> None:
    """H. Anchor + 4 of the 5 required sessions (one dropped from the
    middle) must not resolve."""
    dates = list(_BASE_DATES[:6])
    closes = list(_FLAT_CLOSES[:6])
    del dates[3]
    del closes[3]
    assert len(dates) == 5  # anchor + only 4 future rows
    assert _label_on(dates, closes, _ANCHOR) is None


@pytest.mark.contract
def test_duplicate_future_date_cannot_satisfy_horizon() -> None:
    """I. Duplicating one future date must not let 4 distinct sessions count
    as 5."""
    dates = list(_BASE_DATES[:5]) + [_BASE_DATES[4]]  # last date duplicated
    closes = list(_FLAT_CLOSES[:5]) + [_FLAT_CLOSES[4]]
    assert _label_on(dates, closes, _ANCHOR) is None, (
        "a duplicated date must collapse to one session, not manufacture a 5th"
    )


@pytest.mark.contract
def test_non_finite_future_observation_is_unresolved() -> None:
    """J. A non-finite close (NaN or inf-producing) anywhere in the 5-session
    window must yield unresolved, never a finite label derived from partial
    finite data."""
    dates = _BASE_DATES[:6]
    closes = [100.0, 100.0, float("nan"), 100.0, 100.0, 100.0]
    assert _label_on(dates, closes, _ANCHOR) is None

    # A zero close mid-window makes pct_change() produce +/-inf, which must
    # also gate as unresolved rather than propagate into a spurious label.
    closes_zero = [100.0, 100.0, 0.0, 100.0, 100.0, 100.0]
    assert _label_on(dates, closes_zero, _ANCHOR) is None


@pytest.mark.contract
def test_out_of_order_input_rows_interpreted_chronologically() -> None:
    """K. Rows out of chronological order must be sorted before target
    computation, deterministically, regardless of input order."""
    dates = _BASE_DATES[:6]
    closes = _FLAT_CLOSES[:6]
    order = [3, 0, 5, 1, 4, 2]
    shuffled_dates = [dates[i] for i in order]
    shuffled_closes = [closes[i] for i in order]
    assert _label_on(shuffled_dates, shuffled_closes, _ANCHOR) == "LOW"
    assert _label_on(dates, closes, _ANCHOR) == _label_on(shuffled_dates, shuffled_closes, _ANCHOR)


@pytest.mark.contract
def test_six_or_more_future_sessions_uses_only_intended_horizon() -> None:
    """L. Extra sessions beyond the 5-session horizon must not be consumed
    (target must not silently widen its window)."""
    dates = _BASE_DATES[:10]
    closes = _FLAT_CLOSES[:10]
    label_6 = _label_on(dates[:7], closes[:7], _ANCHOR)
    label_10 = _label_on(dates, closes, _ANCHOR)
    assert label_6 == "LOW"
    assert label_10 == label_6


@pytest.mark.contract
def test_days_outside_horizon_never_alter_resolved_label() -> None:
    """M. A violent move on day 6 (outside the 5-session horizon) must not
    change the label already determined by days 1-5."""
    dates = _BASE_DATES[:7]
    calm_closes = [100.0] * 6 + [100.0]
    label_calm = _label_on(dates, calm_closes, _ANCHOR)

    volatile_day_six = [100.0] * 6 + [180.0]  # +80% on day 6, outside horizon
    label_with_shock = _label_on(dates, volatile_day_six, _ANCHOR)
    assert label_calm == "LOW"
    assert label_with_shock == "LOW", "day 6 is outside the 5-session target horizon"


@pytest.mark.contract
def test_repeated_direct_computation_is_idempotent() -> None:
    """N. (target-function level) Recomputing the target from the same
    immutable OHLCV must always yield the same label -- pure function, no
    hidden state."""
    dates = _BASE_DATES[:6]
    closes = _FLAT_CLOSES[:6]
    first = _label_on(dates, closes, _ANCHOR)
    second = _label_on(dates, closes, _ANCHOR)
    assert first == second == "LOW"


@pytest.mark.contract
def test_training_and_direct_target_computation_agree() -> None:
    """P. training_frame_with_target's target column (the training-time
    path) must exactly agree with calling vol_regime_target directly (the
    resolver's path) for the same OHLCV -- one source of truth."""
    from ml.feature_engineering import training_frame_with_target

    dates = _BASE_DATES[:10]
    closes = _FLAT_CLOSES[:10]
    ohlcv = _frame(dates, closes)
    ctx = MLContext(
        symbol="TESTSTK",
        signal_payload={},
        current_features={},
        feature_frame=pd.DataFrame({"date": dates}),
        ohlcv=ohlcv,
        iv_history=pd.DataFrame(),
        bif_history=pd.DataFrame(),
        fo_eod_rows=0,
        current_spot=None,
        spot_provenance={},
        market_data_eligibility=None,  # type: ignore[arg-type]
        decision_requested_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    trained = training_frame_with_target(ctx, vol_regime_target)
    direct = vol_regime_target(_Ctx(ohlcv))
    for _, row in trained.iterrows():
        assert row["target"] == direct.get(row["date"]), (
            "training-time target and resolver-time target must never disagree"
        )
    # Rows the resolver would call unresolved must never appear in the
    # training frame either (dropna(subset=["target"]) in
    # training_frame_with_target already enforces this -- assert it holds).
    unresolved_dates = {d for d in direct.index if direct.get(d) is None or (isinstance(direct.get(d), float) and direct.get(d) != direct.get(d))}
    assert not (set(trained["date"]) & unresolved_dates)


@pytest.mark.contract
@pytest.mark.integration
def test_writer_service_pipeline_never_resolves_m01_prematurely(
    isolated_app_data: Path,
) -> None:
    """Full-stack regression for the original defect report: drive real
    predictions through WriterService -> resolve_pending_prediction_outcomes
    -> vol_regime_target for m01_vol_regime, at day 1, day 4, and day 5 of
    elapsed OHLCV history, and confirm maturity happens only at day 5 (O:
    'restart between day 4 and day 5' is exercised by resolving twice, with
    the day-5 row only inserted on the second pass, standing in for a
    process restart/time advance between resolver runs)."""
    from db.write_service import WriteOperation, WriterService, submit_write

    duckdb_client.DB_PATH = isolated_app_data / "data" / "terminal.db"
    duckdb_client.init_schema()
    database = duckdb_client.DB_PATH
    queue = isolated_app_data / "data" / "writer.sqlite"

    dates = [d.date().isoformat() for d in pd.bdate_range("2026-01-05", periods=6)]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    anchor_day = dates[0]

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        import duckdb as duckdb_module

        def insert_ohlcv_through(n_rows: int) -> None:
            with duckdb_module.connect(str(database)) as connection:
                with duckdb_client._writer_transaction_scope(connection):
                    connection.execute("DELETE FROM ohlcv WHERE symbol = 'M01TEST'")
                    for day, close in list(zip(dates, closes))[:n_rows]:
                        connection.execute(
                            "INSERT INTO ohlcv VALUES ('M01TEST', CAST(? AS DATE), ?, ?, ?, ?, 1000, 'NSE_EQUITY_BHAVCOPY')",
                            [day, close, close + 1, close - 1, close],
                        )

        prediction_id = submit_write(
            WriteOperation.LOG_PREDICTION,
            {
                "model_id": "m01_vol_regime",
                "prediction": "LOW",
                "probabilities": {"LOW": 0.6, "MEDIUM": 0.3, "HIGH": 0.1},
                "confidence": 0.6,
                "features": {"spot": 1.0},
                "attribution": {
                    "requested_symbol": "M01TEST",
                    "data_as_of": anchor_day,
                    "model_version": "v1",
                    "artifact_hash": "a" * 64,
                    "feature_schema_version": "features-v1:test",
                    "release_version": "0.1.0",
                    "validation_status": "VALIDATED",
                    "fallback_status": "PRIMARY",
                },
            },
            database=database,
            queue_path=queue,
        )

        # Only anchor + 1 future session present -- must NOT resolve.
        insert_ohlcv_through(2)
        summary = submit_write(
            WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m01_vol_regime"]},
            database=database,
            queue_path=queue,
        )
        assert summary["resolved"] == 0
        assert summary["not_yet_resolvable"] == 1

        # Only anchor + 4 future sessions present -- must still NOT resolve
        # (this is exactly the pre-fix defect: std(skipna=True) would have
        # made this resolve after just 1 session).
        insert_ohlcv_through(5)
        summary = submit_write(
            WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m01_vol_regime"]},
            database=database,
            queue_path=queue,
        )
        assert summary["resolved"] == 0, "m01 must not mature before all 5 future sessions exist"
        assert summary["not_yet_resolvable"] == 1

        with duckdb_module.connect(str(database)) as connection:
            row = connection.execute(
                "SELECT outcome FROM ml_predictions WHERE id = ?", [prediction_id]
            ).fetchone()
        assert row == (None,), "still unresolved with only 4 of 5 future sessions"

        # All 5 future sessions now present -- must resolve.
        insert_ohlcv_through(6)
        summary = submit_write(
            WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m01_vol_regime"]},
            database=database,
            queue_path=queue,
        )
        assert summary["resolved"] == 1
        assert summary["not_yet_resolvable"] == 0

        with duckdb_module.connect(str(database)) as connection:
            row = connection.execute(
                "SELECT outcome, correct FROM ml_predictions WHERE id = ?", [prediction_id]
            ).fetchone()
        assert row == ("LOW", 1)

        # Idempotent re-run after resolution.
        summary = submit_write(
            WriteOperation.RESOLVE_PENDING_PREDICTION_OUTCOMES,
            {"model_ids": ["m01_vol_regime"]},
            database=database,
            queue_path=queue,
        )
        assert summary["candidates"] == 0
        assert summary["resolved"] == 0
    finally:
        service.stop()
