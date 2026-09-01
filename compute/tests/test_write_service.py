from __future__ import annotations

import ast
import os
import sqlite3
import threading
import time
import re
from pathlib import Path

import duckdb
import pytest

from db import duckdb_client, write_service
from db.duckdb_client import connect_observational, get_connection, writer_connection
from db.write_service import (
    HEARTBEAT_STALE_SECONDS,
    ReadOperation,
    WriteOperation,
    WriterBackpressureError,
    WriterRequestFailed,
    WriterService,
    WriterServiceError,
    WriterUnavailableError,
    _dispatch_read,
    _ensure_queue_schema,
    _own_process_start_time,
    _owner_liveness,
    _utc_text,
    resolve_read,
    submit_read,
    submit_write,
    writer_service_status,
)


def test_lot_size_read_is_expiry_specific_and_never_fabricates_default() -> None:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            """
            CREATE TABLE instruments(
                token TEXT, symbol TEXT, name TEXT, exchange TEXT,
                instrument_type TEXT, expiry DATE, lot_size INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO instruments VALUES
                ('1', 'NIFTY01SEP26CE', 'NIFTY', 'NFO', 'OPTIDX', DATE '2026-09-01', 65),
                ('2', 'NIFTY29SEP26CE', 'NIFTY', 'NFO', 'OPTIDX', DATE '2026-09-29', 65),
                ('3', 'BANKNIFTY29SEP26CE', 'BANKNIFTY', 'NFO', 'OPTIDX', DATE '2026-09-29', 30)
            """
        )
        assert _dispatch_read(
            ReadOperation.GET_LOT_SIZE.value,
            {"symbol": "NIFTY", "expiry": "2026-09-01"},
            connection,
        ) == {"lot_size": 65}
        assert _dispatch_read(
            ReadOperation.GET_LOT_SIZE.value,
            {"symbol": "BANKNIFTY", "expiry": "2026-09-29"},
            connection,
        ) == {"lot_size": 30}
        assert _dispatch_read(
            ReadOperation.GET_LOT_SIZE.value,
            {"symbol": "MISSING", "expiry": "2026-09-01"},
            connection,
        ) == {"lot_size": None}
    finally:
        connection.close()


def _fii_payload(day: str = "2026-08-21") -> dict[str, object]:
    return {
        "date": day,
        "fii_buy": 100.0,
        "fii_sell": 80.0,
        "fii_net": 20.0,
        "dii_buy": 70.0,
        "dii_sell": 65.0,
        "dii_net": 5.0,
        "fii_net_futures": -2.0,
        "raw_json": "{}",
    }


@pytest.mark.integration
def test_submit_read_serves_typed_reads_through_the_live_writer(tmp_path: Path) -> None:
    """Proves the collector-facing read queue: a caller with no writer
    authority of its own gets a typed, bounded read serviced by whichever
    process owns the live connection -- the mechanism that lets the
    standalone collector stop opening its own cross-process DuckDB handle."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    connection = duckdb.connect(database)
    connection.execute(
        "CREATE TABLE ohlcv(symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
        "low DOUBLE, close DOUBLE, volume BIGINT, source VARCHAR)"
    )
    connection.execute(
        "INSERT INTO ohlcv VALUES ('NIFTY 50', '2026-08-20', 1,2,1,2,0,'NSE_INDEX_BHAVCOPY'), "
        "('NIFTY 50', '2026-08-21', 1,2,1,2,0,'NSE_INDEX_BHAVCOPY')"
    )
    connection.close()
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        result = submit_read(
            ReadOperation.LATEST_DATE_IN,
            {"dataset": "ohlcv", "symbol": "NIFTY 50"},
            database=database,
            queue_path=queue,
        )
        assert result == {"date": "2026-08-21"}

        result = submit_read(
            ReadOperation.ROW_COUNT_FOR_DATE,
            {"dataset": "ohlcv", "date": "2026-08-21"},
            database=database,
            queue_path=queue,
        )
        assert result == {"count": 1}
    finally:
        service.stop()


def test_submit_read_rejects_non_allowlisted_dataset(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        with pytest.raises(WriterRequestFailed, match="not allowlisted"):
            submit_read(
                ReadOperation.LATEST_DATE_IN,
                {"dataset": "writer_requests; DROP TABLE ohlcv"},
                database=database,
                queue_path=queue,
            )
    finally:
        service.stop()


def test_submit_read_rejects_wrong_operation_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ReadOperation"):
        submit_read("LATEST_DATE_IN", {})  # type: ignore[arg-type]


@pytest.mark.integration
def test_collector_read_queue_and_typed_writer_operate_concurrently_with_gui_reads(
    tmp_path: Path,
) -> None:
    """Simulates real production concurrency: GUI/backend observational reads
    (ObservationalConnection, in-process) running alongside collector-style
    typed reads (submit_read, out-of-process shaped) and a typed write, all
    against the one live writer connection."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    connection = duckdb.connect(database)
    connection.execute("CREATE TABLE sentinel(id INTEGER)")
    connection.execute("INSERT INTO sentinel VALUES (1), (2), (3)")
    connection.close()
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        errors: list[Exception] = []

        def gui_reader() -> None:
            try:
                for _ in range(20):
                    reader = connect_observational(database)
                    assert reader.execute("SELECT COUNT(*) FROM sentinel").fetchone()[0] == 3
                    reader.close()
            except Exception as error:  # pragma: no cover - surfaced via errors list
                errors.append(error)

        threads = [threading.Thread(target=gui_reader) for _ in range(4)]
        for thread in threads:
            thread.start()

        result = submit_write(
            WriteOperation.UPSERT_FII_DII,
            _fii_payload(),
            database=database,
            queue_path=queue,
        )
        assert result == {"date": "2026-08-21", "persisted": True}

        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert errors == []
    finally:
        service.stop()


@pytest.mark.integration
def test_backend_reader_and_collector_typed_writer_operate_concurrently(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    connection = duckdb.connect(database)
    connection.execute("CREATE TABLE sentinel(id INTEGER)")
    connection.close()
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        reader = connect_observational(database)
        result = submit_write(
            WriteOperation.UPSERT_FII_DII,
            _fii_payload(),
            database=database,
            queue_path=queue,
        )
        assert reader.execute("SELECT COUNT(*) FROM sentinel").fetchone()[0] == 0
        with pytest.raises(PermissionError):
            reader.execute("INSERT INTO sentinel VALUES (1)")
        reader.close()
        assert result == {"date": "2026-08-21", "persisted": True}
        verify = duckdb.connect(database)
        try:
            assert verify.execute("SELECT fii_net, dii_net FROM fii_dii_daily").fetchone() == (
                20.0,
                5.0,
            )
        finally:
            verify.close()
    finally:
        service.stop()


@pytest.mark.recovery
def test_writer_failure_is_acknowledged_and_transaction_does_not_partially_apply(
    tmp_path: Path,
) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        bad = _fii_payload()
        bad.pop("raw_json")
        with pytest.raises(WriterRequestFailed, match="payload schema"):
            submit_write(
                WriteOperation.UPSERT_FII_DII,
                bad,
                database=database,
                queue_path=queue,
            )
        verify = connect_observational(database)
        try:
            assert "fii_dii_daily" not in {row[0] for row in verify.execute("SHOW TABLES").fetchall()}
        finally:
            verify.close()
    finally:
        service.stop()


@pytest.mark.recovery
def test_writer_crash_detection_rejects_new_requests(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    service = WriterService(database, queue_path=queue)
    service.start()
    service.stop()
    assert writer_service_status(queue)["live"] is False
    with pytest.raises(WriterUnavailableError):
        submit_write(
            WriteOperation.UPSERT_FII_DII,
            _fii_payload(),
            database=database,
            queue_path=queue,
        )


@pytest.mark.recovery
def test_restart_recovers_running_request_to_pending_then_acknowledges(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    _ensure_queue_schema(queue)
    now = "2026-08-23T00:00:00+00:00"
    with sqlite3.connect(queue) as connection:
        connection.execute(
            "INSERT INTO writer_requests "
            "(request_id, operation, payload_json, state, owner_id, created_at, updated_at) "
            "VALUES ('recover-me', 'UPSERT_FII_DII', ?, 'RUNNING', 'dead-owner', ?, ?)",
            ['{"date":"2026-08-21","fii_buy":100.0,"fii_sell":80.0,"fii_net":20.0,"dii_buy":70.0,"dii_sell":65.0,"dii_net":5.0,"fii_net_futures":-2.0,"raw_json":"{}"}', now, now],
        )
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        deadline = time.monotonic() + 5
        state = "RUNNING"
        while time.monotonic() < deadline:
            with sqlite3.connect(queue) as connection:
                state = connection.execute(
                    "SELECT state FROM writer_requests WHERE request_id = 'recover-me'"
                ).fetchone()[0]
            if state in {"SUCCEEDED", "FAILED"}:
                break
            time.sleep(0.05)
        assert state == "SUCCEEDED"
    finally:
        service.stop()


def test_writer_boundary_does_not_accept_sql_operation() -> None:
    with pytest.raises(TypeError, match="WriteOperation"):
        submit_write("SELECT * FROM ohlcv", {})  # type: ignore[arg-type]


def test_writer_queue_applies_bounded_backpressure(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    entered = threading.Event()
    release = threading.Event()

    def blocking_dispatch(*_: object) -> dict[str, bool]:
        entered.set()
        assert release.wait(timeout=5)
        return {"ok": True}

    service = WriterService(
        database,
        queue_path=queue,
        capacity=1,
        dispatcher=blocking_dispatch,
    )
    service.start()
    first_error: list[Exception] = []

    def first_request() -> None:
        try:
            submit_write(
                WriteOperation.UPSERT_FII_DII,
                _fii_payload(),
                database=database,
                queue_path=queue,
                capacity=1,
            )
        except Exception as error:
            first_error.append(error)

    thread = threading.Thread(target=first_request)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(WriterBackpressureError):
            submit_write(
                WriteOperation.UPSERT_FII_DII,
                _fii_payload("2026-08-22"),
                database=database,
                queue_path=queue,
                capacity=1,
            )
    finally:
        release.set()
        thread.join(timeout=5)
        service.stop()
    assert first_error == []


def test_slow_dispatch_keeps_heartbeat_fresh_and_submitter_connected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(write_service, "HEARTBEAT_INTERVAL_SECONDS", 0.03)
    monkeypatch.setattr(write_service, "HEARTBEAT_STALE_SECONDS", 0.15)

    def slow_dispatch(*_: object) -> dict[str, bool]:
        entered.set()
        assert release.wait(timeout=2)
        return {"completed": True}

    service = WriterService(database, queue_path=queue, dispatcher=slow_dispatch)
    service.start()
    first_heartbeat = writer_service_status(queue)["heartbeat_at"]
    results: list[object] = []

    def submit() -> None:
        try:
            results.append(
                submit_write(
                    WriteOperation.UPSERT_FII_DII,
                    _fii_payload(),
                    database=database,
                    queue_path=queue,
                    timeout=2,
                )
            )
        except Exception as error:
            results.append(error)

    request = threading.Thread(target=submit)
    request.start()
    assert entered.wait(timeout=2)
    try:
        deadline = time.monotonic() + 1
        heartbeat_advanced = False
        while time.monotonic() < deadline:
            status = writer_service_status(queue)
            if status["heartbeat_at"] != first_heartbeat:
                heartbeat_advanced = True
                break
            time.sleep(0.02)
        assert heartbeat_advanced
        time.sleep(0.2)
        assert writer_service_status(queue)["responsive"] is True
    finally:
        release.set()
        request.join(timeout=2)
        service.stop()

    assert results == [{"completed": True}]


def test_default_connection_cannot_bypass_live_writer_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    connection = duckdb.connect(database)
    connection.execute("CREATE TABLE sentinel(id INTEGER)")
    connection.close()
    monkeypatch.setattr(duckdb_client, "DB_PATH", database)
    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        normal = get_connection()
        with pytest.raises(PermissionError, match="non-read query"):
            normal.execute("INSERT INTO sentinel VALUES (1)")
        normal.close()
    finally:
        service.stop()


def test_writer_capability_is_denied_outside_service_transaction() -> None:
    with pytest.raises(PermissionError, match="WriterService"):
        writer_connection()


def test_application_has_no_ambiguous_or_unclassified_duckdb_connection_path() -> None:
    root = Path(__file__).parents[1]
    allowed_raw_connect = {
        Path("db/duckdb_client.py"),
        Path("db/migrations.py"),
        Path("db/write_service.py"),
    }
    ambiguous: list[str] = []
    unclassified: list[str] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "tests" or any(part.startswith(".") for part in relative.parts):
            continue
        source = path.read_text(encoding="utf-8")
        if relative != Path("db/duckdb_client.py") and re.search(r"\bget_connection\s*\(", source):
            ambiguous.append(relative.as_posix())
        if "duckdb.connect(" in source and relative not in allowed_raw_connect:
            unclassified.append(relative.as_posix())
    assert ambiguous == []
    assert unclassified == []


def test_raw_duckdb_connect_sites_are_exactly_owned_and_classified() -> None:
    root = Path(__file__).parents[1]
    expected = {
        ("db/duckdb_client.py", "ObservationalConnection.execute", True),
        ("db/duckdb_client.py", "_create_schema_objects_at", False),
        ("db/migrations.py", "_record_migration", False),
        ("db/migrations.py", "_apply_sql_migration", False),
        ("db/migrations.py", "_apply_decision_eligible_view_migration", False),
        ("db/migrations.py", "_current_version", True),
        ("db/migrations.py", "_validate_database", True),
        ("db/write_service.py", "WriterService.start", False),
    }
    actual: set[tuple[str, str, bool]] = set()

    class ConnectVisitor(ast.NodeVisitor):
        def __init__(self, relative: str) -> None:
            self.relative = relative
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

        def visit_Call(self, node: ast.Call) -> None:
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "duckdb"
                and function.attr == "connect"
            ):
                read_only = any(
                    keyword.arg == "read_only"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
                actual.add((self.relative, ".".join(self.scope), read_only))
            self.generic_visit(node)

    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "tests" or any(part.startswith(".") for part in relative.parts):
            continue
        ConnectVisitor(relative.as_posix()).visit(ast.parse(path.read_text(encoding="utf-8")))

    assert actual == expected


def test_observational_facade_never_retains_raw_duckdb_handle(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    duckdb.connect(database).execute("CREATE TABLE sentinel(id INTEGER)").close()
    reader = connect_observational(database)
    assert reader.execute("SELECT COUNT(*) FROM sentinel").fetchone() == (0,)
    assert not hasattr(reader, "_connection")
    with pytest.raises(AttributeError):
        getattr(reader, "_connection").execute("INSERT INTO sentinel VALUES (1)")
    assert not any(isinstance(value, duckdb.DuckDBPyConnection) for value in vars(reader).values())


def test_collector_has_no_direct_duckdb_connection_capability() -> None:
    root = Path(__file__).parents[1]
    for relative in ("auto_collector/monitor.py", "auto_collector/smart_collector.py"):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                any(alias.name == "duckdb" for alias in node.names)
                if isinstance(node, ast.Import)
                else node.module == "duckdb"
            )
            for node in ast.walk(tree)
        ), relative
        assert "duckdb.connect" not in source


def _seed_writer_runtime(
    queue: Path,
    *,
    owner_id: str,
    pid: int,
    process_start_time: int | None,
    heartbeat_at: str,
) -> None:
    _ensure_queue_schema(queue)
    with sqlite3.connect(queue) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO writer_runtime "
            "(singleton, owner_id, pid, process_start_time, state, heartbeat_at) "
            "VALUES (1, ?, ?, ?, 'RUNNING', ?)",
            [owner_id, pid, process_start_time, heartbeat_at],
        )


@pytest.mark.recovery
def test_dead_owner_process_reclaimed_immediately_despite_fresh_heartbeat(
    tmp_path: Path,
) -> None:
    """Exact HIGH-1 reproduction: writer A dies uncleanly, heartbeat is still
    fresh (< HEARTBEAT_STALE_SECONDS old), and a new legitimate writer B must
    identify the dead owner and reclaim immediately rather than being denied.
    """
    import subprocess
    import sys

    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = dead.pid
    assert dead.wait(timeout=10) == 0

    _seed_writer_runtime(
        queue,
        owner_id="dead-writer",
        pid=dead_pid,
        process_start_time=1,
        heartbeat_at=_utc_text(),
    )

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        assert writer_service_status(queue)["live"] is True
    finally:
        service.stop()


@pytest.mark.recovery
def test_live_owner_is_never_bypassed_even_with_stale_heartbeat(tmp_path: Path) -> None:
    """A provably-alive owner must be rejected outright: liveness proof takes
    priority over heartbeat freshness in either direction."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    own_start = _own_process_start_time()
    _seed_writer_runtime(
        queue,
        owner_id="still-alive",
        pid=os.getpid(),
        process_start_time=own_start,
        heartbeat_at="2020-01-01T00:00:00+00:00",
    )

    service = WriterService(database, queue_path=queue)
    with pytest.raises(WriterServiceError, match="already active"):
        service.start()


@pytest.mark.recovery
def test_pid_reuse_does_not_impersonate_original_owner(tmp_path: Path) -> None:
    """A lease recorded against PID X / identity A must not be mistaken for
    an unrelated live process that now happens to hold PID X."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    # Our own PID is genuinely alive, but the recorded identity (start time)
    # does not match it -- simulating a different, now-dead process that
    # previously held this PID.
    _seed_writer_runtime(
        queue,
        owner_id="stale-identity-same-pid",
        pid=os.getpid(),
        process_start_time=1,
        heartbeat_at=_utc_text(),
    )

    assert _owner_liveness(os.getpid(), 1) is False

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        assert writer_service_status(queue)["live"] is True
    finally:
        service.stop()


@pytest.mark.recovery
def test_stale_heartbeat_reclaim_still_works_when_liveness_unprovable(tmp_path: Path) -> None:
    """Preserve the pre-existing conservative fallback for the case liveness
    cannot be established (e.g. no recorded process identity)."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    _seed_writer_runtime(
        queue,
        owner_id="legacy-no-identity",
        pid=999999999,
        process_start_time=None,
        heartbeat_at="2020-01-01T00:00:00+00:00",
    )
    assert _owner_liveness(999999999, None) is None

    service = WriterService(database, queue_path=queue)
    service.start()
    try:
        assert writer_service_status(queue)["live"] is True
    finally:
        service.stop()


@pytest.mark.recovery
def test_unprovable_liveness_with_fresh_heartbeat_still_denies(tmp_path: Path) -> None:
    """Fail closed: when liveness cannot be proven and the heartbeat is
    fresh, still deny the competing writer (two-live-writer invariant)."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    _seed_writer_runtime(
        queue,
        owner_id="legacy-no-identity-fresh",
        pid=999999999,
        process_start_time=None,
        heartbeat_at=_utc_text(),
    )

    service = WriterService(database, queue_path=queue)
    with pytest.raises(WriterServiceError, match="already active"):
        service.start()


@pytest.mark.recovery
def test_repeated_crash_restart_never_wedges_for_full_heartbeat_ttl(tmp_path: Path) -> None:
    """Crash-loop: each restart after an unclean death must reclaim
    immediately, not accumulate wait time across iterations."""
    import subprocess
    import sys

    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()

    for iteration in range(3):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_pid = dead.pid
        assert dead.wait(timeout=10) == 0
        _seed_writer_runtime(
            queue,
            owner_id=f"crash-{iteration}",
            pid=dead_pid,
            process_start_time=1,
            heartbeat_at=_utc_text(),
        )
        started = time.monotonic()
        service = WriterService(database, queue_path=queue)
        service.start()
        elapsed = time.monotonic() - started
        try:
            assert writer_service_status(queue)["live"] is True
        finally:
            service.stop()
        assert elapsed < HEARTBEAT_STALE_SECONDS


def test_second_same_process_writer_is_rejected_even_with_different_queue(
    tmp_path: Path,
) -> None:
    database = tmp_path / "terminal.db"
    duckdb.connect(database).close()
    first = WriterService(database, queue_path=tmp_path / "one.sqlite")
    second = WriterService(database, queue_path=tmp_path / "two.sqlite")
    first.start()
    try:
        with pytest.raises(WriterServiceError, match="already active"):
            second.start()
    finally:
        first.stop()


@pytest.mark.recovery
def test_commit_before_ack_replays_by_marker_without_duplicate_mutation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    dispatch_calls = 0

    def dispatch(*_: object) -> dict[str, int]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        connection = writer_connection()
        connection.execute("CREATE TABLE IF NOT EXISTS effects(id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO effects VALUES (1)")
        connection.close()
        return {"effects": 1}

    class CrashAfterCommitService(WriterService):
        crashed = False

        def _ack(self, request_id: str, state: str, **kwargs: object) -> None:
            if state == "SUCCEEDED" and not self.crashed:
                self.crashed = True
                raise RuntimeError("simulated crash after DuckDB commit")
            super()._ack(request_id, state, **kwargs)

    crashed = CrashAfterCommitService(database, queue_path=queue, dispatcher=dispatch)
    crashed.start()
    try:
        with pytest.raises(WriterUnavailableError):
            submit_write(
                WriteOperation.UPSERT_FII_DII,
                _fii_payload(),
                database=database,
                queue_path=queue,
                request_id="stable-effect-1",
            )
    finally:
        crashed.stop()

    recovered = WriterService(database, queue_path=queue, dispatcher=dispatch)
    recovered.start()
    try:
        result = submit_write(
            WriteOperation.UPSERT_FII_DII,
            _fii_payload(),
            database=database,
            queue_path=queue,
            request_id="stable-effect-1",
        )
        assert result == {"effects": 1}
    finally:
        recovered.stop()

    verify = duckdb.connect(database, read_only=True)
    try:
        assert verify.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM writer_applied_requests").fetchone()[0] == 1
    finally:
        verify.close()
    assert dispatch_calls == 1


def test_shutdown_drains_request_already_accepted(tmp_path: Path) -> None:
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    entered = threading.Event()
    release = threading.Event()

    def blocking_dispatch(*_: object) -> dict[str, bool]:
        entered.set()
        assert release.wait(timeout=5)
        return {"drained": True}

    service = WriterService(database, queue_path=queue, dispatcher=blocking_dispatch)
    service.start()
    results: list[object] = []

    def submit() -> None:
        try:
            results.append(
                submit_write(
                    WriteOperation.UPSERT_FII_DII,
                    _fii_payload(),
                    database=database,
                    queue_path=queue,
                )
            )
        except Exception as error:
            results.append(error)

    request = threading.Thread(target=submit)
    request.start()
    assert entered.wait(timeout=5)
    stopped: list[Exception] = []

    def stop() -> None:
        try:
            service.stop(timeout=5)
        except Exception as error:
            stopped.append(error)

    shutdown = threading.Thread(target=stop)
    shutdown.start()
    time.sleep(0.1)
    release.set()
    request.join(timeout=5)
    shutdown.join(timeout=5)
    assert stopped == []
    assert results == [{"drained": True}]


def test_resolve_read_forwards_caller_timeout_to_cross_process_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the Chart-indicators defect: resolve_read used to hand
    submit_read a hardcoded 30s ceiling no matter what the caller actually
    needed, so an interactive read had no way to fail faster than a full
    background-read budget. Proves the plumbing without needing a second
    process: the routing decision (no local writer, a live remote owner)
    is forced directly."""
    calls: list[dict[str, object]] = []

    def fake_status(_queue_path: object) -> dict[str, object]:
        return {"live": True}

    def fake_submit_read(operation: object, payload: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"rows": []}

    monkeypatch.setattr(write_service, "writer_service_status", fake_status)
    monkeypatch.setattr(write_service, "submit_read", fake_submit_read)
    monkeypatch.setattr(write_service, "_ACTIVE_WRITER_CONNECTION", type("_C", (), {"get": staticmethod(lambda: None)})())
    monkeypatch.setattr(write_service, "_lookup_writer_connection", lambda _database: None)

    resolve_read(ReadOperation.LATEST_DATE_IN, {"dataset": "ohlcv"}, timeout=8.0)
    assert calls[-1]["timeout"] == 8.0

    resolve_read(ReadOperation.LATEST_DATE_IN, {"dataset": "ohlcv"})
    assert calls[-1]["timeout"] == 30.0, "unspecified callers must keep today's existing 30s ceiling"


def test_indicator_scale_read_fails_fast_under_contention_instead_of_starving(
    tmp_path: Path,
) -> None:
    """A read using the tight 8s-class budget must not starve behind a
    single slow write the way the generic 30s ceiling would tolerate --
    it must come back (as a clean, honest timeout) well under the 10s+
    a UI client is willing to wait, proving the queue's read-priority
    fairness plus a caller-scoped timeout together bound worst-case
    latency for interactive reads."""
    database = tmp_path / "terminal.db"
    queue = tmp_path / "writer.sqlite"
    duckdb.connect(database).close()
    entered = threading.Event()
    release = threading.Event()

    def blocking_dispatch(*_: object) -> dict[str, bool]:
        entered.set()
        assert release.wait(timeout=5)
        return {"ok": True}

    service = WriterService(database, queue_path=queue, dispatcher=blocking_dispatch)
    service.start()
    writer_result: list[object] = []

    def submit_blocking_write() -> None:
        try:
            writer_result.append(
                submit_write(
                    WriteOperation.UPSERT_FII_DII,
                    _fii_payload(),
                    database=database,
                    queue_path=queue,
                    timeout=5,
                )
            )
        except Exception as error:
            writer_result.append(error)

    thread = threading.Thread(target=submit_blocking_write)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        started = time.monotonic()
        with pytest.raises(WriterUnavailableError, match="reader request timed out"):
            submit_read(
                ReadOperation.LATEST_DATE_IN,
                {"dataset": "ohlcv"},
                database=database,
                queue_path=queue,
                timeout=1.0,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, "an 8s-class interactive read must not wait out a 30s-class ceiling"
    finally:
        release.set()
        thread.join(timeout=5)
        service.stop()
    assert writer_result == [{"ok": True}]
