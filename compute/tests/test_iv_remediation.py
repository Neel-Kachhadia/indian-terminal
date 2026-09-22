from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import duckdb
import pytest
from py_vollib.black import black
from py_vollib.black_scholes import black_scholes

from backtest.price_quality import HistoricalPriceQuality, classify_historical_option_row
from bif.vol_clock import MINIMUM_IV_OBSERVATIONS, compute_vrc
from data.risk_free_rate import (
    FBIL_QUOTE_CONVENTION,
    FBIL_SOURCE_ID,
    FBIL_TBILL_LATEST_URL,
    OPTION_RATE_CONVENTION,
    RATE_CONVERSION_METHOD,
    RateProvenanceError,
    TenorPoint,
    latest_available_rate,
    parse_tenor_label,
    select_rate_for_tenor,
)
from db.duckdb_client import _create_schema_objects_at, connect_observational
from db.migrations import migrate_database
from db.writer_lease import release_writer_lease
from db.write_service import (
    ReadOperation,
    WriteOperation,
    WriterRequestFailed,
    WriterService,
    _write_iv_rows,
    submit_read,
    submit_write,
)
from options import iv_backfill, iv_pricing
from options.iv_contract import (
    CONTRACT_VERSION,
    IST,
    canonical_observation_id,
    validate_observation,
)
from options.iv_pricing import (
    ArbitrageReason,
    PricingModel,
    SolveStatus,
    check_arbitrage_bounds,
    solve_iv_with_provenance,
)


UTC = dt.timezone.utc
FO_COLUMNS = [
    "symbol", "expiry", "strike", "option_type", "date", "open", "high",
    "low", "close", "volume", "oi", "oi_change", "source", "source_format",
    "price_quality",
]
SPOT_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close", "volume",
    "observed_at", "source", "quality", "derivation_type",
]


def _point(
    days: int,
    *,
    day: dt.date = dt.date(2026, 8, 31),
    rate_pct: float = 5.5,
    published_at: dt.datetime | None = dt.datetime(2026, 8, 31, 12, tzinfo=UTC),
    retrieved_at: dt.datetime | None = None,
) -> TenorPoint:
    return TenorPoint(
        observation_date=day,
        tenor_days=days,
        tenor_label=f"{days} Days",
        rate_pct=rate_pct,
        published_at=published_at,
        source=FBIL_SOURCE_ID,
        source_url=FBIL_TBILL_LATEST_URL,
        retrieved_at=retrieved_at or dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, UTC),
    )


def test_fbil_tenor_and_discount_factor_conversion_contract() -> None:
    assert parse_tenor_label("7 Days") == 7
    assert parse_tenor_label("3 Months") == 90
    assert parse_tenor_label("12 Months") == 365
    with pytest.raises(RateProvenanceError):
        parse_tenor_label("banana")

    point = _point(90, rate_pct=5.5)
    tenor = 90 / 365
    assert math.exp(-point.rate_decimal * tenor) == pytest.approx(
        1 / (1 + 0.055 * tenor), abs=1e-15
    )


def test_rate_selection_is_nearest_bounded_and_provenanced() -> None:
    selection = select_rate_for_tenor([_point(7), _point(30), _point(90)], 24)
    assert selection is not None
    assert selection.selected_tenor_days == 30
    assert selection.rate_decimal == pytest.approx(_point(30).rate_decimal)
    assert selection.quote_convention == FBIL_QUOTE_CONVENTION
    assert selection.rate_convention == OPTION_RATE_CONVENTION
    assert selection.conversion_method == RATE_CONVERSION_METHOD
    assert select_rate_for_tenor([_point(90)], 7) is None


def test_rate_selection_rejects_missing_retrieval_or_unverified_source() -> None:
    missing_retrieval = TenorPoint(
        observation_date=dt.date(2026, 8, 31), tenor_days=30, tenor_label="30 Days",
        rate_pct=5.5, published_at=None, source=FBIL_SOURCE_ID,
        source_url=FBIL_TBILL_LATEST_URL, retrieved_at=None,
    )
    unverified = TenorPoint(
        observation_date=dt.date(2026, 8, 31), tenor_days=30, tenor_label="30 Days",
        rate_pct=5.5, published_at=None, source="TEST", source_url="https://example.invalid",
        retrieved_at=dt.datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert select_rate_for_tenor([missing_retrieval], 30) is None
    assert select_rate_for_tenor([unverified], 30) is None


def test_latest_rate_rejects_stale_and_post_close_same_day_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import db.write_service as write_service

    def encoded(point: TenorPoint) -> list[Any]:
        return [
            point.observation_date.isoformat(), point.tenor_days, point.tenor_label,
            point.rate_pct, point.rate_decimal, point.source, point.source_url,
            point.published_at.isoformat() if point.published_at else None,
            point.retrieved_at.isoformat() if point.retrieved_at else None,
            FBIL_QUOTE_CONVENTION, OPTION_RATE_CONVENTION, RATE_CONVERSION_METHOD,
        ]

    stale = _point(30, day=dt.date(2026, 9, 4))
    monkeypatch.setattr(write_service, "resolve_read", lambda *args, **kwargs: {"rows": [encoded(stale)]})
    assert latest_available_rate(30 / 365, as_of=dt.datetime(2026, 9, 11, 10, tzinfo=UTC)) is None

    after_close = _point(
        30, day=dt.date(2026, 9, 11),
        published_at=dt.datetime(2026, 9, 11, 12, tzinfo=UTC),
    )
    monkeypatch.setattr(write_service, "resolve_read", lambda *args, **kwargs: {"rows": [encoded(after_close)]})
    assert latest_available_rate(30 / 365, as_of=dt.datetime(2026, 9, 11, 10, tzinfo=UTC)) is None

    prior = _point(
        30, day=dt.date(2026, 9, 10),
        retrieved_at=dt.datetime(2026, 9, 11, 9, tzinfo=UTC),
    )
    monkeypatch.setattr(write_service, "resolve_read", lambda *args, **kwargs: {"rows": [encoded(prior)]})
    assert latest_available_rate(30 / 365, as_of=dt.datetime(2026, 9, 11, 10, tzinfo=UTC)) is not None


@pytest.mark.parametrize(
    ("flag", "spot", "strike", "tenor", "sigma"),
    [
        ("c", 100.0, 100.0, 0.5, 0.20),
        ("p", 100.0, 100.0, 0.5, 0.20),
        ("c", 140.0, 100.0, 0.5, 0.35),
        ("p", 100.0, 140.0, 0.5, 0.35),
        ("c", 100.0, 110.0, 2 / 365, 0.50),
        ("p", 100.0, 90.0, 5 / 365, 0.45),
    ],
)
def test_black_scholes_oracle_and_iv_inversion(
    flag: str, spot: float, strike: float, tenor: float, sigma: float
) -> None:
    rate = 0.052
    price = black_scholes(flag, spot, strike, tenor, rate, sigma)
    solved = solve_iv_with_provenance(flag, price, spot, strike, tenor, rate)
    assert solved.status is SolveStatus.OK
    assert solved.iv == pytest.approx(sigma, abs=1e-8)
    assert solved.abs_error <= 1e-8


@pytest.mark.parametrize("flag", ["c", "p"])
def test_black76_oracle_and_iv_inversion(flag: str) -> None:
    forward, strike, tenor, rate, sigma = 101.5, 100.0, 0.08, 0.052, 0.24
    price = black(flag, forward, strike, tenor, rate, sigma)
    solved = solve_iv_with_provenance(
        flag, price, 100.0, strike, tenor, rate,
        pricing_model=PricingModel.BLACK_76, dividend_yield=None, forward_price=forward,
    )
    assert solved.status is SolveStatus.OK
    assert solved.iv == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"spot": 0.0}, ArbitrageReason.INVALID_SPOT),
        ({"strike": 0.0}, ArbitrageReason.INVALID_STRIKE),
        ({"rate": float("nan")}, ArbitrageReason.INVALID_RATE),
        ({"time_to_expiry_years": 0.0}, ArbitrageReason.INVALID_T),
        ({"price": 0.0}, ArbitrageReason.PRICE_UNAVAILABLE),
    ],
)
def test_solver_fails_closed_on_pathological_inputs(
    kwargs: dict[str, float], reason: ArbitrageReason
) -> None:
    values = {
        "flag": "c", "price": 5.0, "spot": 100.0, "strike": 100.0,
        "time_to_expiry_years": 0.1, "rate": 0.05,
    }
    values.update(kwargs)
    result = check_arbitrage_bounds(**values)
    assert result.ok is False
    assert result.reason is reason


def test_solver_rejects_prices_outside_arbitrage_bounds() -> None:
    below = solve_iv_with_provenance("c", 1.0, 200.0, 100.0, 0.5, 0.05)
    above = solve_iv_with_provenance("p", 200.0, 100.0, 100.0, 0.5, 0.05)
    assert below.status is SolveStatus.ARBITRAGE_REJECTED
    assert above.status is SolveStatus.ARBITRAGE_REJECTED
    assert below.iv is above.iv is None


def test_solver_has_explicit_convergence_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    price = black_scholes("c", 100.0, 100.0, 0.2, 0.05, 0.2)
    monkeypatch.setattr(
        iv_pricing,
        "brentq",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cap")),
    )
    result = solve_iv_with_provenance("c", price, 100.0, 100.0, 0.2, 0.05)
    assert result.status is SolveStatus.NO_IV_SOLUTION
    assert result.reason == "solver_exception:RuntimeError"
    assert result.iv is None


def test_unknown_price_quality_remains_excluded_even_when_price_looks_plausible() -> None:
    row = {
        "symbol": "RELIANCE", "expiry": "2026-09-25", "strike": 3000.0,
        "option_type": "CE", "date": "2026-09-01", "open": 40.0,
        "high": 46.0, "low": 39.0, "close": 45.0, "volume": 120,
        "source": "SOME_UNVERIFIED_SOURCE", "price_quality": None,
    }
    assert classify_historical_option_row(row).quality is HistoricalPriceQuality.UNKNOWN


def _fo_row(
    option_type: str,
    expiry: dt.date,
    strike: float,
    close: float,
    *,
    source: str = "NSE_FO_BHAVCOPY",
    quality: str | None = "VALID_EXECUTABLE_OBSERVATION",
) -> list[Any]:
    return [
        "RELIANCE", expiry, strike, option_type, dt.date(2026, 9, 1), close,
        close + 1, max(0.01, close - 1), close, 500, 1000, 10,
        source, "NSE_FO_BHAVCOPY_V1", quality,
    ]


def _spot_row() -> list[Any]:
    return [
        "RELIANCE", dt.date(2026, 9, 1), 99.0, 101.0, 98.0, 100.0, 1_000_000,
        "2026-09-01T15:30:00+05:30", "NSE_EQUITY_BHAVCOPY", "OBSERVED_VERIFIED", "OBSERVED",
    ]


def _patch_backfill_reads(
    monkeypatch: pytest.MonkeyPatch,
    fo_rows: list[list[Any]],
    spot_rows: list[list[Any]],
) -> None:
    def fake_read(operation: ReadOperation, payload: dict[str, Any]) -> dict[str, Any]:
        if operation is ReadOperation.LOAD_FO_EOD_RANGE:
            return {"columns": FO_COLUMNS, "rows": fo_rows}
        if operation is ReadOperation.LOAD_TECHNICAL_OHLCV_RANGE:
            return {"columns": SPOT_COLUMNS, "rows": spot_rows}
        raise AssertionError(operation)

    monkeypatch.setattr(iv_backfill, "resolve_read", fake_read)


def _valid_backfill_inputs(
    sigma: float = 0.22,
) -> tuple[list[list[Any]], dict[dt.date, list[TenorPoint]]]:
    expiry = dt.date(2026, 9, 25)
    observed = dt.datetime(2026, 9, 1, 10, tzinfo=UTC)
    tenor = (
        dt.datetime.combine(expiry, dt.time(15, 30), IST).astimezone(UTC) - observed
    ).total_seconds() / (365 * 86400)
    point = _point(30)
    call_price = black("c", 101.0, 100.0, tenor, point.rate_decimal, sigma)
    rows = [
        _fo_row("FUT", expiry, 0.0, 101.0),
        _fo_row("CE", expiry, 100.0, call_price),
    ]
    return rows, {point.observation_date: [point]}


def test_backfill_builds_reproducible_black76_row() -> None:
    fo_rows, curve = _valid_backfill_inputs()
    expiry = dt.date(2026, 9, 25)
    observed = dt.datetime(2026, 9, 1, 10, tzinfo=UTC)
    tenor = (
        dt.datetime.combine(expiry, dt.time(15, 30), IST).astimezone(UTC) - observed
    ).total_seconds() / (365 * 86400)
    rate = select_rate_for_tenor(curve[dt.date(2026, 8, 31)], round(tenor * 365))
    assert rate is not None
    price = float(fo_rows[1][8])
    solved = solve_iv_with_provenance(
        "c", price, 100.0, 100.0, tenor, rate.rate_decimal,
        pricing_model=PricingModel.BLACK_76, dividend_yield=None, forward_price=101.0,
    )
    candidate = {
        "symbol": "RELIANCE", "expiry": expiry, "strike": 100.0,
        "option_type": "CE", "date": dt.date(2026, 9, 1),
        "source": "NSE_FO_BHAVCOPY", "volume": 500, "oi": 1000,
    }
    row = iv_backfill._build_iv_row(
        candidate,
        price,
        {"price": 100.0, "source": "NSE_EQUITY_BHAVCOPY", "quality": "OBSERVED_VERIFIED"},
        {"price": 101.0, "source": "NSE_FO_BHAVCOPY:EXACT_EXPIRY_FUTURE"},
        rate,
        solved,
        observed,
        dt.datetime(2026, 9, 2, tzinfo=UTC),
    )
    validate_observation(row)
    assert row["pricing_model"] == "BLACK_76"
    assert row["dividend_yield"] is None
    assert row["observation_kind"] == "HISTORICAL_RECONSTRUCTION"


def test_backfill_excludes_unknown_rows_and_accepts_exact_expiry_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fo_rows, curve = _valid_backfill_inputs()
    fo_rows.append(
        _fo_row("PE", dt.date(2026, 9, 25), 95.0, 8.0, source="UNKNOWN_VENDOR", quality=None)
    )
    _patch_backfill_reads(monkeypatch, fo_rows, [_spot_row()])
    counters = iv_backfill.BackfillCounters()
    rows = iv_backfill.derive_symbol_candidates(
        "RELIANCE", from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1),
        counters=counters, rate_curve=curve,
    )
    assert len(rows) == 1
    assert counters.price_quality_counts["UNKNOWN"] == 1
    assert rows[0]["forward_price"] == 101.0


def test_backfill_deduplicates_exact_source_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    fo_rows, curve = _valid_backfill_inputs()
    fo_rows.append(list(fo_rows[1]))
    _patch_backfill_reads(monkeypatch, fo_rows, [_spot_row()])
    counters = iv_backfill.BackfillCounters()
    rows = iv_backfill.derive_symbol_candidates(
        "RELIANCE", from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1),
        counters=counters, rate_curve=curve,
    )
    assert len(rows) == 1
    assert counters.duplicate_source_rows == 1
    assert len({row["observation_id"] for row in rows}) == 1


def test_backfill_forbids_cross_expiry_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    point = _point(30)
    rows = [
        _fo_row("FUT", dt.date(2026, 10, 29), 0.0, 103.0),
        _fo_row("CE", dt.date(2026, 9, 25), 100.0, 4.0),
    ]
    _patch_backfill_reads(monkeypatch, rows, [_spot_row()])
    counters = iv_backfill.BackfillCounters()
    result = iv_backfill.derive_symbol_candidates(
        "RELIANCE", from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1),
        counters=counters, rate_curve={point.observation_date: [point]},
    )
    assert result == []
    assert counters.forward_unavailable == 1


def test_backfill_forbids_same_day_rate_published_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fo_rows, _ = _valid_backfill_inputs()
    _patch_backfill_reads(monkeypatch, fo_rows, [_spot_row()])
    same_day_late = _point(
        30,
        day=dt.date(2026, 9, 1),
        published_at=dt.datetime(2026, 9, 1, 12, tzinfo=UTC),
    )
    counters = iv_backfill.BackfillCounters()
    result = iv_backfill.derive_symbol_candidates(
        "RELIANCE", from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1),
        counters=counters,
        rate_curve={same_day_late.observation_date: [same_day_late]},
    )
    assert result == []
    assert counters.rate_unavailable == 1


def test_backfill_missing_spot_and_missing_rate_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fo_rows, curve = _valid_backfill_inputs()
    _patch_backfill_reads(monkeypatch, fo_rows, [])
    counters = iv_backfill.BackfillCounters()
    assert iv_backfill.derive_symbol_candidates(
        "RELIANCE", from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1),
        counters=counters, rate_curve=curve,
    ) == []
    assert counters.spot_unavailable == 1

    _patch_backfill_reads(monkeypatch, fo_rows, [_spot_row()])
    counters = iv_backfill.BackfillCounters()
    assert iv_backfill.derive_symbol_candidates(
        "RELIANCE", from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1),
        counters=counters, rate_curve={},
    ) == []
    assert counters.rate_unavailable == 1


def _canonical_row(
    *,
    kind: str = "HISTORICAL_RECONSTRUCTION",
    observed: dt.datetime | None = None,
) -> dict[str, Any]:
    observed = observed or dt.datetime(2026, 9, 1, 10, tzinfo=UTC)
    trade_day = observed.astimezone(IST).date()
    expiry = trade_day + dt.timedelta(days=24)
    expiry_at = dt.datetime.combine(expiry, dt.time(15, 30), IST).astimezone(UTC)
    tenor = (expiry_at - observed).total_seconds() / (365 * 86400)
    rate_point = _point(
        30,
        day=trade_day - dt.timedelta(days=1),
        published_at=observed - dt.timedelta(days=1),
        retrieved_at=observed - dt.timedelta(hours=1),
    )
    rate = select_rate_for_tenor([rate_point], round(tenor * 365))
    assert rate is not None
    forward, spot, strike, sigma = 101.0, 100.0, 100.0, 0.22
    price = black("c", forward, strike, tenor, rate.rate_decimal, sigma)
    solved = solve_iv_with_provenance(
        "c", price, spot, strike, tenor, rate.rate_decimal,
        pricing_model=PricingModel.BLACK_76, dividend_yield=None, forward_price=forward,
    )
    created = max(dt.datetime(2026, 9, 2, tzinfo=UTC), observed + dt.timedelta(seconds=1))
    live = kind == "LIVE"
    row: dict[str, Any] = {
        "observation_id": "",
        "symbol": "RELIANCE",
        "underlying": "RELIANCE",
        "expiry": expiry.isoformat(),
        "strike": strike,
        "option_type": "CE",
        "date": trade_day.isoformat(),
        "observed_at": observed.isoformat(),
        "source_timestamp": observed.isoformat(),
        "available_at": created.isoformat(),
        "iv": solved.iv,
        "contract_token": "123" if live else None,
        "option_price": price,
        "option_price_semantic": (
            "LIVE_TOP_OF_BOOK_BID_ASK_MIDPOINT" if live else iv_backfill.OPTION_PRICE_SEMANTIC
        ),
        "option_price_source": (
            "NSE_GET_SYMBOL_DERIVATIVES_DATA" if live else "NSE_FO_BHAVCOPY"
        ),
        "option_price_quality": (
            "EXECUTABLE_BID_ASK_MIDPOINT" if live else "VALID_EXECUTABLE_OBSERVATION"
        ),
        "spot_price": spot,
        "spot_source": (
            "NSE_GET_SYMBOL_DERIVATIVES_DATA" if live else "NSE_EQUITY_BHAVCOPY"
        ),
        "spot_timestamp": observed.isoformat(),
        "spot_quality": "OBSERVED_VERIFIED",
        "forward_price": forward,
        "forward_source": (
            "PUT_CALL_PARITY" if live else "NSE_FO_BHAVCOPY:EXACT_EXPIRY_FUTURE"
        ),
        "forward_timestamp": observed.isoformat(),
        "forward_quality": (
            "DERIVED_FROM_EXECUTABLE_QUOTES" if live else "VALID_EXECUTABLE_OBSERVATION"
        ),
        "risk_free_rate": rate.rate_decimal,
        "risk_free_tenor_days": rate.selected_tenor_days,
        "risk_free_source": rate.source,
        "risk_free_source_url": rate.source_url,
        "risk_free_observation_date": rate.observation_date.isoformat(),
        "risk_free_published_at": (
            rate.published_at.isoformat() if rate.published_at else None
        ),
        "risk_free_retrieved_at": (
            rate.retrieved_at.isoformat() if rate.retrieved_at else None
        ),
        "risk_free_convention": OPTION_RATE_CONVENTION,
        "time_to_expiry_years": tenor,
        "dividend_yield": None,
        "dividend_source": None,
        "pricing_model": "BLACK_76",
        "solver": solved.solver,
        "solver_version": solved.solver_version,
        "repriced_price": solved.repriced_price,
        "repricing_abs_error": solved.abs_error,
        "repricing_rel_error": solved.rel_error,
        "volume": 500,
        "oi": 1000,
        "source": f"NSE+{FBIL_SOURCE_ID}",
        "quality": "IV_VALID",
        "calculation_status": "VALID",
        "reason": None,
        "observation_kind": kind,
        "created_at": created.isoformat(),
        "calculation_version": solved.calculation_version,
        "contract_version": CONTRACT_VERSION,
    }
    row["observation_id"] = canonical_observation_id(row)
    validate_observation(row)
    return row


def _migrated_database(path: Path) -> None:
    migrate_database(path, schema_initializer=_create_schema_objects_at)


@pytest.mark.integration
def test_writer_iv_is_idempotent_immutable_and_restart_safe(tmp_path: Path) -> None:
    database, queue = tmp_path / "terminal.db", tmp_path / "writer.sqlite"
    _migrated_database(database)
    row = _canonical_row()
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        first = submit_write(
            WriteOperation.UPSERT_IV_OBSERVATIONS,
            {"rows": [row]},
            request_id="iv-restart-test",
            database=database,
            queue_path=queue,
        )
        assert first == {"upserted": 1}
    finally:
        service.stop()

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        replay = submit_write(
            WriteOperation.UPSERT_IV_OBSERVATIONS,
            {"rows": [row]},
            request_id="iv-restart-test",
            database=database,
            queue_path=queue,
        )
        duplicate = submit_write(
            WriteOperation.UPSERT_IV_OBSERVATIONS,
            {"rows": [row]},
            database=database,
            queue_path=queue,
        )
        assert replay == {"upserted": 0}
        assert duplicate == {"upserted": 0}
        changed = dict(row)
        changed["iv"] = float(row["iv"]) + 0.01
        with pytest.raises(WriterRequestFailed):
            submit_write(
                WriteOperation.UPSERT_IV_OBSERVATIONS,
                {"rows": [changed]},
                database=database,
                queue_path=queue,
            )
    finally:
        service.stop()

    connection = connect_observational(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM iv_history").fetchone()[0] == 1
    finally:
        connection.close()


def test_direct_iv_write_requires_writer_transaction(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    _migrated_database(database)
    connection = duckdb.connect(str(database))
    try:
        with pytest.raises(RuntimeError, match="WriterService"):
            _write_iv_rows(connection, [_canonical_row()])
    finally:
        connection.close()


@pytest.mark.integration
def test_live_and_reconstructed_observations_do_not_collide(tmp_path: Path) -> None:
    database, queue = tmp_path / "terminal.db", tmp_path / "writer.sqlite"
    _migrated_database(database)
    historical = _canonical_row()
    live = _canonical_row(
        kind="LIVE",
        observed=dt.datetime.now(UTC).replace(microsecond=0) - dt.timedelta(seconds=5),
    )
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        result = submit_write(
            WriteOperation.UPSERT_IV_OBSERVATIONS,
            {"rows": [historical, live]},
            database=database,
            queue_path=queue,
        )
        assert result == {"upserted": 2}
        raw = submit_read(
            ReadOperation.LOAD_IV_HISTORY_ROWS_FOR_SYMBOL,
            {"symbol": "RELIANCE"},
            database=database,
            queue_path=queue,
        )
        assert {row[0] for row in raw["rows"]} == {
            historical["observation_id"],
            live["observation_id"],
        }
        daily = submit_read(
            ReadOperation.LOAD_FEATURE_IV_HISTORY,
            {
                "candidates": ["RELIANCE"],
                "as_of": dt.datetime.now(UTC).isoformat(),
            },
            database=database,
            queue_path=queue,
        )
        assert len(daily["rows"]) == 1
        assert daily["rows"][0][0] == historical["date"]
        assert daily["rows"][0][1] == pytest.approx(historical["iv"])
    finally:
        service.stop()


@pytest.mark.integration
def test_rate_writer_is_idempotent_and_rejects_conflicting_revision(tmp_path: Path) -> None:
    database, queue = tmp_path / "terminal.db", tmp_path / "writer.sqlite"
    _migrated_database(database)
    point = _point(90)
    row = {
        "observation_date": point.observation_date.isoformat(),
        "tenor_days": 90,
        "tenor_label": "90 Days",
        "rate_pct": point.rate_pct,
        "rate_decimal": point.rate_decimal,
        "source": point.source,
        "source_url": point.source_url,
        "published_at": point.published_at.isoformat() if point.published_at else None,
        "retrieved_at": point.retrieved_at.isoformat() if point.retrieved_at else None,
        "quote_convention": FBIL_QUOTE_CONVENTION,
        "rate_convention": OPTION_RATE_CONVENTION,
        "conversion_method": RATE_CONVERSION_METHOD,
    }
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        assert submit_write(
            WriteOperation.UPSERT_RISK_FREE_RATE_CURVE,
            {"rows": [row]}, database=database, queue_path=queue,
        ) == {"upserted": 1}
        assert submit_write(
            WriteOperation.UPSERT_RISK_FREE_RATE_CURVE,
            {"rows": [row]}, database=database, queue_path=queue,
        ) == {"upserted": 0}
        conflict = dict(row)
        conflict["rate_pct"] = 6.0
        conflict["rate_decimal"] = _point(90, rate_pct=6.0).rate_decimal
        with pytest.raises(WriterRequestFailed):
            submit_write(
                WriteOperation.UPSERT_RISK_FREE_RATE_CURVE,
                {"rows": [conflict]}, database=database, queue_path=queue,
            )
    finally:
        service.stop()


def test_vrc_keeps_minimum_history_and_never_turns_missing_iv_into_zero() -> None:
    short = [
        {"date": str(index), "atm_iv": 0.15}
        for index in range(MINIMUM_IV_OBSERVATIONS - 1)
    ]
    assert compute_vrc(short, {"atm_iv": 0.16}, [], {})["reason_code"] == "INSUFFICIENT_HISTORY"
    assert compute_vrc(short, {"atm_iv": None}, [], {})["reason_code"] == "NO_CURRENT_IV"
    enough = [
        {"date": str(index), "atm_iv": 0.12 + (index % 5) * 0.01}
        for index in range(40)
    ]
    result = compute_vrc(
        enough,
        {"atm_iv": 0.18},
        [{"close": 100 + index % 3} for index in range(25)],
        {},
    )
    assert result["is_valid"] is True
    assert result["iv_observations"] == 40


def _dump_iv_history(database: Path) -> list[tuple]:
    connection = connect_observational(database)
    try:
        return connection.execute(
            "SELECT observation_id, iv, calculation_status, source "
            "FROM iv_history ORDER BY observation_id"
        ).fetchall()
    finally:
        connection.close()


@pytest.mark.integration
def test_backfill_persist_rows_idempotent_and_resumable_after_interrupt(tmp_path: Path) -> None:
    """iv_backfill.persist_rows (the real batched production entry point,
    not just the lower-level writer) must converge to the exact same final
    dataset whether it runs once uninterrupted, or is interrupted after a
    partial write and then resumed by re-submitting its full candidate set."""
    rows = [
        _canonical_row(observed=dt.datetime(2026, 9, 1 + index, 10, tzinfo=UTC))
        for index in range(5)
    ]
    assert len({row["observation_id"] for row in rows}) == 5

    reference_db, reference_queue = tmp_path / "reference.db", tmp_path / "reference-writer.sqlite"
    _migrated_database(reference_db)
    service = WriterService(reference_db, queue_path=reference_queue)
    service.start()
    try:
        upserted = iv_backfill.persist_rows(rows, database=reference_db, queue_path=reference_queue)
        assert upserted == 5
    finally:
        service.stop()
    reference_state = _dump_iv_history(reference_db)
    assert len(reference_state) == 5
    release_writer_lease()  # this process's lease is per-database; free it before the second DB

    resumed_db, resumed_queue = tmp_path / "resumed.db", tmp_path / "resumed-writer.sqlite"
    _migrated_database(resumed_db)

    # "Interrupt": only the first 2 of 5 candidate rows land before crashing.
    service = WriterService(resumed_db, queue_path=resumed_queue)
    service.start()
    try:
        partial = iv_backfill.persist_rows(rows[:2], database=resumed_db, queue_path=resumed_queue)
        assert partial == 2
    finally:
        service.stop()

    # "Resume": a fresh service instance (new process, in spirit), and the
    # backfill re-submits its FULL candidate set -- exactly how this
    # architecture resumes, per the writer's own idempotency guarantee.
    service = WriterService(resumed_db, queue_path=resumed_queue)
    service.start()
    try:
        resumed = iv_backfill.persist_rows(rows, database=resumed_db, queue_path=resumed_queue)
        assert resumed == 3  # only the 3 rows that never landed
        # Same batch, same content -> same deterministic request_id. Durable
        # ledger replay performs no mutation and reports zero actual-new rows.
        replay = iv_backfill.persist_rows(rows, database=resumed_db, queue_path=resumed_queue)
        assert replay == 0
    finally:
        service.stop()

    resumed_state = _dump_iv_history(resumed_db)
    assert resumed_state == reference_state
    assert len(resumed_state) == 5
    assert len({row[0] for row in resumed_state}) == 5


@pytest.mark.integration
def test_iv_actual_new_count_for_reorder_and_in_request_duplicates(tmp_path: Path) -> None:
    database, queue = tmp_path / "terminal.db", tmp_path / "writer.sqlite"
    _migrated_database(database)
    first = _canonical_row(observed=dt.datetime(2026, 9, 1, 10, tzinfo=UTC))
    second = _canonical_row(observed=dt.datetime(2026, 9, 2, 10, tzinfo=UTC))
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        assert iv_backfill.persist_rows(
            [first, first, second], database=database, queue_path=queue
        ) == 2
        assert iv_backfill.persist_rows(
            [second, first], database=database, queue_path=queue
        ) == 0

        conflicting = dict(first)
        conflicting["iv"] = float(first["iv"]) + 0.01
        with pytest.raises(WriterRequestFailed):
            submit_write(
                WriteOperation.UPSERT_IV_OBSERVATIONS,
                {"rows": [first, conflicting]},
                database=database,
                queue_path=queue,
            )
    finally:
        service.stop()

    assert len(_dump_iv_history(database)) == 2
