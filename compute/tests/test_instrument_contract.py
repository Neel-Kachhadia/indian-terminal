from __future__ import annotations

import pandas as pd
import pytest
import duckdb

from data.instrument_master import InstrumentMaster
from data.instrument_schema import (
    INSTRUMENT_SCHEMA_VERSION,
    InstrumentSchemaError,
    read_instrument_frame,
)
from live.multi_feed import build_token_list
from db.write_service import WriteOperation, WriterService, submit_write


def _upstream_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "token": "101",
                "symbol": "RELIANCE-EQ",
                "name": "RELIANCE",
                "exch_seg": "NSE",
                "instrumenttype": "EQ",
                "expiry": "",
                "strike": "0",
                "optiontype": "",
                "lotsize": "1",
                "tick_size": "5",
                "isin": "INE002A01018",
            },
            {
                "token": "201",
                "symbol": "RELIANCE26AUGFUT",
                "name": "RELIANCE",
                "exch_seg": "NFO",
                "instrumenttype": "FUTSTK",
                "expiry": "2026-08-27",
                "strike": "0",
                "optiontype": "",
                "lotsize": "250",
                "tick_size": "5",
                "isin": "",
            },
            {
                "token": "99926000",
                # Angel's real raw scripmaster puts the space-containing long
                # form in `symbol` and the short form in `name` for indices
                # -- the reverse of the short/long convention equities use.
                # Confirmed live against the real production instrument
                # master (D:\BharatTerminal\data\terminal.db /
                # %APPDATA%\BharatTerminal\data\terminal.db, checked
                # 2026-08-28 via the typed RESOLVE_INDEX_INSTRUMENT read).
                # canonicalize_instrument_frame() passes symbol/name through
                # unrenamed, so this raw-upstream-shaped fixture must match
                # that real field order for tests to be genuinely
                # representative (resolve_index_feed's query matches both
                # columns against the same alias set, so this ordering
                # doesn't change whether resolution WORKS -- only whether
                # the fixture is honest about which field holds which text).
                "symbol": "NIFTY 50",
                "name": "NIFTY",
                "exch_seg": "NSE",
                "instrumenttype": "AMXIDX",
                "expiry": "",
                "strike": "0",
                "optiontype": "",
                "lotsize": "1",
                "tick_size": "5",
                "isin": "",
            },
            {
                "token": "99926009",
                "symbol": "NIFTY BANK",
                "name": "BANKNIFTY",
                "exch_seg": "NSE",
                "instrumenttype": "AMXIDX",
                "expiry": "",
                "strike": "0",
                "optiontype": "",
                "lotsize": "1",
                "tick_size": "5",
                "isin": "",
            },
            {
                "token": "99926037",
                "symbol": "NIFTY FIN SERVICE",
                "name": "FINNIFTY",
                "exch_seg": "NSE",
                "instrumenttype": "AMXIDX",
                "expiry": "",
                "strike": "0",
                "optiontype": "",
                "lotsize": "1",
                "tick_size": "5",
                "isin": "",
            },
            {
                "token": "99926074",
                "symbol": "NIFTY MID SELECT",
                "name": "MIDCPNIFTY",
                "exch_seg": "NSE",
                "instrumenttype": "AMXIDX",
                "expiry": "",
                "strike": "0",
                "optiontype": "",
                "lotsize": "1",
                "tick_size": "5",
                "isin": "",
            },
        ]
    )


@pytest.mark.contract
def test_producer_fixture_is_readable_by_live_consumer(tmp_path) -> None:
    frame = InstrumentMaster()._normalise(_upstream_fixture())
    path = tmp_path / "instruments.parquet"
    frame.to_parquet(path, index=False)

    stored = read_instrument_frame(path)
    tokens, mapping = build_token_list(path)

    assert set(stored["schema_version"]) == {INSTRUMENT_SCHEMA_VERSION}
    assert "instrument_type" in stored.columns
    assert "exchange" in stored.columns
    # 101 (RELIANCE spot) + the 4 canonical index tokens the fixture now
    # carries real rows for (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY) -- all
    # 4 must be discovered by build_token_list's live-feed subscription list.
    assert set(tokens) == {"101", "99926000", "99926009", "99926037", "99926074"}
    assert mapping["101"] == "RELIANCE"
    assert mapping["99926000"] == "NIFTY"
    assert mapping["99926009"] == "BANKNIFTY"
    assert mapping["99926037"] == "FINNIFTY"
    assert mapping["99926074"] == "MIDCPNIFTY"


@pytest.mark.contract
def test_legacy_consumer_fields_fail_loudly(tmp_path) -> None:
    path = tmp_path / "legacy.parquet"
    _upstream_fixture().to_parquet(path, index=False)

    with pytest.raises(InstrumentSchemaError, match="canonical"):
        read_instrument_frame(path)


def test_duplicate_token_exchange_identity_is_rejected() -> None:
    raw = pd.concat([_upstream_fixture(), _upstream_fixture().iloc[[0]]], ignore_index=True)

    with pytest.raises(InstrumentSchemaError, match="duplicate"):
        InstrumentMaster()._normalise(raw)


def test_normalizing_canonical_frame_is_idempotent() -> None:
    master = InstrumentMaster()
    first = master._normalise(_upstream_fixture())
    second = master._normalise(first)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.recovery
def test_invalid_cache_publication_preserves_previous_good_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix behavior overwrote the canonical parquet directly, swallowed
    the validation/write error, then allowed the refresh to report success."""
    cache = tmp_path / "instruments.parquet"
    previous = InstrumentMaster()._normalise(_upstream_fixture().iloc[[0]])
    previous.to_parquet(cache, index=False)
    previous_bytes = cache.read_bytes()
    candidate = InstrumentMaster()._normalise(_upstream_fixture())

    monkeypatch.setattr("data.instrument_master.INSTRUMENTS_CACHE", cache)

    def _write_invalid(self, path, **_kwargs) -> None:
        del self
        path.write_bytes(b"not parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _write_invalid)

    with pytest.raises(Exception):
        InstrumentMaster()._write_cache(candidate)

    assert cache.read_bytes() == previous_bytes
    assert list(tmp_path.glob(f".{cache.name}.*.tmp")) == []


def test_write_cache_fsync_uses_a_write_capable_handle(tmp_path, monkeypatch) -> None:
    """Reproduces a real deployed defect, deterministic (not a timing race):
    `_write_cache` re-opened the just-written temp file in read-only mode
    ("rb") purely to call os.fsync() on it. On Windows, os.fsync() /
    FlushFileBuffers requires a handle opened with write access to succeed
    even when nothing is written through it -- the read-only handle raised
    `OSError: [Errno 9] Bad file descriptor` on every single call, not
    occasionally. This surfaced on every real installed startup as
    "Instrument master refresh failed: OSError: [Errno 9] Bad file
    descriptor" (main.py's background refresh thread), with the underlying
    traceback previously discarded (see main.py's _refresh_instrument_master_safely,
    also fixed to log the full exception now). Must succeed end-to-end."""
    cache = tmp_path / "instruments.parquet"
    monkeypatch.setattr("data.instrument_master.INSTRUMENTS_CACHE", cache)
    df = InstrumentMaster()._normalise(_upstream_fixture())

    cache_sha256 = InstrumentMaster()._write_cache(df)

    assert cache.is_file()
    assert len(cache_sha256) == 64  # sha256 hex digest
    # The published file must actually be durable/fsynced content, not left
    # mid-write or as a dangling .tmp file.
    assert list(tmp_path.glob(f".{cache.name}.*.tmp")) == []
    assert len(read_instrument_frame(cache)) == len(df)


@pytest.mark.recovery
def test_fresh_cache_repairs_db_after_precommit_crash_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh cache alone is not proof the same snapshot committed to DuckDB.
    Missing DB/cache identity must rehydrate the DB, including stale removal."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    cache = tmp_path / "instruments.parquet"
    frame = InstrumentMaster()._normalise(_upstream_fixture())
    frame.to_parquet(cache, index=False)
    stale = frame.iloc[[0]].copy()
    stale.loc[:, "token"] = "stale-token"
    stale.loc[:, "symbol"] = "STALE-EQ"
    with duckdb.connect(database) as connection:
        connection.register("stale_fixture", pd.concat([frame, stale], ignore_index=True))
        connection.execute("CREATE TABLE instruments AS SELECT * FROM stale_fixture")
    monkeypatch.setattr("data.instrument_master.INSTRUMENTS_CACHE", cache)

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        first = submit_write(
            WriteOperation.REFRESH_INSTRUMENT_MASTER,
            {"force": False},
            database=database,
            queue_path=queue,
        )
        second = submit_write(
            WriteOperation.REFRESH_INSTRUMENT_MASTER,
            {"force": False},
            database=database,
            queue_path=queue,
        )
    finally:
        service.stop()
    with duckdb.connect(database, read_only=True) as connection:
        rows = connection.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
        state = connection.execute(
            "SELECT row_count, schema_version FROM instrument_master_state"
        ).fetchone()
        stale_rows = connection.execute(
            "SELECT COUNT(*) FROM instruments WHERE token = 'stale-token'"
        ).fetchone()[0]

    assert first == {"rows": len(frame)}
    assert second == {"rows": 0}
    assert rows == len(frame)
    assert state == (len(frame), INSTRUMENT_SCHEMA_VERSION)
    assert stale_rows == 0


@pytest.mark.contract
def test_nifty_feed_token_resolves_from_canonical_master(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_index_feed() now always goes through resolve_read(), which
    dispatches locally when this process itself owns a live writer
    connection for the database (simulated here via
    db.write_service._lookup_writer_connection) instead of ever opening a
    second raw connection to the file."""
    database = tmp_path / "instruments.db"
    connection = duckdb.connect(database)
    try:
        frame = InstrumentMaster()._normalise(_upstream_fixture())
        connection.register("fixture", frame)
        connection.execute("CREATE TABLE instruments AS SELECT * FROM fixture")
        monkeypatch.setattr("db.write_service._lookup_writer_connection", lambda database: connection)

        resolved = InstrumentMaster().resolve_index_feed("NIFTY")
        assert resolved == {
            "schema_version": INSTRUMENT_SCHEMA_VERSION,
            "token": "99926000",
            "symbol": "NIFTY",
            "canonical_symbol": "NIFTY 50",
            "name": "NIFTY",
            "exchange": "NSE",
            "instrument_type": "AMXIDX",
        }
    finally:
        connection.close()


@pytest.mark.contract
def test_index_feed_uses_typed_remote_read_when_no_local_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces a real live defect: this process has no local writer
    connection for terminal.db (proven live: exactly the GUI/backend state
    when a DIFFERENT process, e.g. the standalone collector, holds it as
    writer authority -- DuckDB has no cross-process concurrent reader).
    resolve_read() must proactively route to the typed
    RESOLVE_INDEX_INSTRUMENT read over the cross-process queue -- it must
    never attempt a raw read against the file in this case, not even as a
    first attempt that then falls back on failure."""

    captured: dict[str, object] = {}

    def _fake_submit_read(operation, payload, **_kwargs):
        captured["operation"] = operation
        captured["payload"] = payload
        from db.write_service import ReadOperation

        assert operation is ReadOperation.RESOLVE_INDEX_INSTRUMENT
        return {
            "normalized": payload["symbol"],
            "rows": [
                {
                    "schema_version": INSTRUMENT_SCHEMA_VERSION,
                    "token": "99926000",
                    "symbol": "NIFTY",
                    "name": "NIFTY 50",
                    "exchange": "NSE",
                    "instrument_type": "AMXIDX",
                }
            ],
        }

    monkeypatch.setattr(
        "db.write_service.writer_service_status", lambda queue_path: {"live": True}
    )
    monkeypatch.setattr("db.write_service.submit_read", _fake_submit_read)

    resolved = InstrumentMaster().resolve_index_feed("NIFTY")
    assert resolved["token"] == "99926000"
    assert resolved["symbol"] == "NIFTY"
    assert captured["payload"] == {"symbol": "NIFTY"}


@pytest.mark.contract
@pytest.mark.parametrize("symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
def test_index_feed_remote_fallback_covers_each_supported_index(
    symbol: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "MIDCPNIFTY": "MIDCPNIFTY",
    }[symbol]

    def _fake_submit_read(operation, payload, **_kwargs):
        return {
            "normalized": payload["symbol"],
            "rows": [
                {
                    "schema_version": INSTRUMENT_SCHEMA_VERSION,
                    "token": "1",
                    "symbol": canonical,
                    "name": canonical,
                    "exchange": "NSE",
                    "instrument_type": "AMXIDX",
                }
            ],
        }

    monkeypatch.setattr(
        "db.write_service.writer_service_status", lambda queue_path: {"live": True}
    )
    monkeypatch.setattr("db.write_service.submit_read", _fake_submit_read)

    resolved = InstrumentMaster().resolve_index_feed(symbol)
    assert resolved["symbol"] == canonical


@pytest.mark.contract
@pytest.mark.parametrize(
    "symbol,expected_token,expected_canonical,expected_raw_symbol_field,expected_raw_name_field",
    [
        ("NIFTY", "99926000", "NIFTY", "NIFTY 50", "NIFTY"),
        ("BANKNIFTY", "99926009", "BANKNIFTY", "NIFTY BANK", "BANKNIFTY"),
        ("NIFTYBANK", "99926009", "BANKNIFTY", "NIFTY BANK", "BANKNIFTY"),
        ("FINNIFTY", "99926037", "FINNIFTY", "NIFTY FIN SERVICE", "FINNIFTY"),
        ("MIDCPNIFTY", "99926074", "MIDCPNIFTY", "NIFTY MID SELECT", "MIDCPNIFTY"),
    ],
)
def test_all_canonical_indexes_resolve_from_real_shaped_master(
    symbol: str,
    expected_token: str,
    expected_canonical: str,
    expected_raw_symbol_field: str,
    expected_raw_name_field: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces a real found gap: MIDCPNIFTY was entirely absent from the
    (previously triplicated, now single-source) index alias registry, so
    resolve_index_feed("MIDCPNIFTY") raised "unsupported index feed symbol"
    even though the real production instrument master has always carried a
    valid NSE/AMXIDX row for it (token 99926074, raw symbol field
    "NIFTY MID SELECT"). This end-to-end test resolves all 4 required
    indexes -- plus the alternate "NIFTYBANK" alias form -- against
    instrument-master rows shaped exactly like the real production data,
    including the real (and easy to get backwards) field convention: for
    these index rows Angel's raw scripmaster puts the space-containing long
    form in `symbol` and the short form in `name` -- confirmed live against
    the real production instrument master via a labeled
    RESOLVE_INDEX_INSTRUMENT read, 2026-08-28."""
    database = tmp_path / "instruments.db"
    connection = duckdb.connect(database)
    try:
        frame = InstrumentMaster()._normalise(_upstream_fixture())
        connection.register("fixture", frame)
        connection.execute("CREATE TABLE instruments AS SELECT * FROM fixture")
        monkeypatch.setattr("db.write_service._lookup_writer_connection", lambda database: connection)

        resolved = InstrumentMaster().resolve_index_feed(symbol)
        assert resolved["token"] == expected_token
        assert resolved["symbol"] == expected_canonical
        assert resolved["canonical_symbol"] == expected_raw_symbol_field
        assert resolved["name"] == expected_raw_name_field
        assert resolved["instrument_type"] == "AMXIDX"
    finally:
        connection.close()


@pytest.mark.contract
def test_index_feed_resolution_fails_closed_for_unknown_symbol() -> None:
    with pytest.raises(InstrumentSchemaError, match="unsupported index feed symbol"):
        InstrumentMaster().resolve_index_feed("NOT A REAL INDEX")


@pytest.mark.contract
def test_nifty_feed_resolution_rejects_missing_or_wrong_type(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "instruments.db"
    connection = duckdb.connect(database)
    try:
        frame = InstrumentMaster()._normalise(_upstream_fixture())
        frame.loc[frame["symbol"] == "NIFTY 50", "instrument_type"] = "EQ"
        connection.register("fixture", frame)
        connection.execute("CREATE TABLE instruments AS SELECT * FROM fixture")
        monkeypatch.setattr("db.write_service._lookup_writer_connection", lambda database: connection)

        with pytest.raises(InstrumentSchemaError, match="missing or ambiguous"):
            InstrumentMaster().resolve_index_feed("NIFTY")
    finally:
        connection.close()
