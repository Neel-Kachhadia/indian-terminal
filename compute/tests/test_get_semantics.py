from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import duckdb
import pytest
import requests
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from db import duckdb_client
from security import credential_manager


FRAMEWORK_GET_ROUTES = {"/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}
PATH_VALUES = {
    "{symbol}": "NIFTY",
    "{job_id}": "missing-job",
    "{strategy}": "all",
}


class ReadOnlyRedisFixture:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self.values.get(str(key))

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.values.get(str(key)) for key in keys]

    def scan_iter(self, **_: Any):
        return iter(())

    def ttl(self, key: str) -> int:
        return 30 if str(key) in self.values else -2

    def hgetall(self, _: str) -> dict[str, str]:
        return {}

    def state_digest(self) -> str:
        return _json_hash(self.values)

    def __getattr__(self, name: str) -> Any:
        if name.lower() in {
            "set",
            "setex",
            "delete",
            "hset",
            "incr",
            "decr",
            "expire",
            "publish",
            "pipeline",
        }:
            raise AssertionError(f"GET attempted Redis mutation: {name}")
        raise AttributeError(name)


class UnavailableResponse:
    status_code = 503
    content = b""
    text = "external boundary disabled by GET-purity harness"
    headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return {}

    def raise_for_status(self) -> None:
        raise requests.HTTPError(self.text)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_manifest(root: Path) -> dict[str, tuple[str, int | None, str | None]]:
    if not root.exists():
        return {}
    manifest: dict[str, tuple[str, int | None, str | None]] = {".": ("DIRECTORY", None, None)}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            manifest[relative] = ("DIRECTORY", None, None)
            continue
        if not path.is_file():
            manifest[relative] = ("OTHER", None, None)
            continue
        size = path.stat().st_size
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except PermissionError:
            digest = "LOCKED-BY-CURRENT-TEST-PROCESS"
        manifest[relative] = ("FILE", size, digest)
    return manifest


def _duckdb_digest(path: Path) -> str:
    connection = duckdb.connect(path, read_only=True)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM duckdb_tables() ORDER BY table_name"
            ).fetchall()
        ]
        payload: dict[str, Any] = {}
        for table in tables:
            quoted = table.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            rows = connection.execute(f'SELECT * FROM "{quoted}" ORDER BY ALL').fetchall()
            payload[table] = {"columns": columns, "rows": rows}
        return _json_hash(payload)
    finally:
        connection.close()


def _job_state_digest(main: Any) -> str:
    return _json_hash(
        {
            "download": copy.deepcopy(main._download_status),
            "backfill": copy.deepcopy(main._backfill_state),
            "parser_report": copy.deepcopy(main._parser_test_report),
            "ml_training": copy.deepcopy(main._ml_training_state),
            "stock_training": copy.deepcopy(main._stock_training_state),
            "backtest_jobs": copy.deepcopy(main._backtest_jobs),
            "broker_auth_status": copy.deepcopy(main._broker_auth_status),
            "sentiment_score_history": list(main._sentiment_score_history),
            "sentiment_article_history": list(main._sentiment_article_history),
        }
    )


PROCESS_LOCAL_MEMOIZATION = {
    "main": {
        "_market_signals_cache",
        "_sentiment_cache",
        "_sentiment_cache_time",
        "_cross_asset_cache",
        "_cross_asset_cache_time",
        "_all_signals_cache",
        "_all_signals_cache_time",
        "_apex_cache",
        "_ohlc_forecast_cache",
        "_ohlc_forecast_cache_time",
        "_bif_engine",
    },
    "data.stock_snapshot": {"_UNIVERSE_CACHE"},
    "ml.models.common": {"_ARTIFACT_CACHE"},
}


def _application_get_routes(app: Any) -> list[APIRoute]:
    return sorted(
        (
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and "GET" in route.methods
            and route.path not in FRAMEWORK_GET_ROUTES
        ),
        key=lambda route: route.path,
    )


def _concrete_path(route: APIRoute) -> str:
    path = route.path
    for parameter, value in PATH_VALUES.items():
        path = path.replace(parameter, value)
    assert "{" not in path, f"GET route needs a purity fixture: {route.path}"
    if route.path == "/ohlcv/{symbol}":
        path += "?from_date=2026-06-01&to_date=2026-08-21"
    return path


def _seed_read_fixtures(database: Path) -> None:
    connection = duckdb.connect(database)
    try:
        rows = []
        for offset in range(82):
            day = connection.execute(
                "SELECT CAST('2026-06-01' AS DATE) + ?",
                [offset],
            ).fetchone()[0]
            base = 24_000.0 + offset
            rows.append(
                (
                    "NIFTY 50",
                    day,
                    base,
                    base + 25,
                    base - 25,
                    base + 5,
                    1_000_000 + offset,
                    "NSE_INDEX_BHAVCOPY",
                )
            )
        connection.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        connection.execute(
            """
            INSERT INTO instruments
                (schema_version, token, symbol, name, exchange, instrument_type,
                 expiry, strike, option_type, lot_size, tick_size, isin, updated_at)
            VALUES
                (1, '99123456', 'NIFTY', 'NIFTY 50', 'NSE', 'AMXIDX',
                 NULL, NULL, NULL, 1, 0.05, NULL, CURRENT_TIMESTAMP),
                (1, '99123457', 'BANKNIFTY', 'NIFTY BANK', 'NSE', 'AMXIDX',
                 NULL, NULL, NULL, 1, 0.05, NULL, CURRENT_TIMESTAMP),
                (1, '99123458', 'FINNIFTY', 'NIFTY FIN SERVICE', 'NSE', 'AMXIDX',
                 NULL, NULL, NULL, 1, 0.05, NULL, CURRENT_TIMESTAMP),
                (1, '99123459', 'MIDCPNIFTY', 'NIFTY MID SELECT', 'NSE', 'AMXIDX',
                 NULL, NULL, NULL, 1, 0.05, NULL, CURRENT_TIMESTAMP)
            """
        )
    finally:
        connection.close()


def test_get_route_inventory_is_dynamic_and_complete() -> None:
    main = importlib.import_module("main")
    routes = _application_get_routes(main.app)
    assert routes
    assert len({route.path for route in routes}) == len(routes)
    assert all("GET" in route.methods for route in routes)


def test_instrument_schema_error_gets_stable_bounded_contract_from_any_route() -> None:
    """Reproduces a real found gap: only routes with their OWN local
    `try/except InstrumentSchemaError` converted it to a typed 503 -- any
    OTHER route that could raise the same error (a real, expected "no
    instrument master loaded yet" precondition, not an unexpected fault)
    fell through to a raw, undifferentiated 500. A GLOBAL exception handler
    closes this for every route uniformly, present or future, rather than
    depending on each call site remembering to add its own try/except.
    This test proves the global handler via a route that deliberately has
    NO local try/except of its own."""
    from data.instrument_schema import InstrumentSchemaError

    main = importlib.import_module("main")

    @main.app.get("/__test_only_raises_instrument_schema_error")
    def _raise_instrument_schema_error() -> None:
        raise InstrumentSchemaError("simulated: instrument master not loaded")

    client = TestClient(main.app, raise_server_exceptions=False)
    try:
        response = client.get(
            "/__test_only_raises_instrument_schema_error",
            headers={"X-Process-Token": main.PROCESS_TOKEN},
        )
    finally:
        client.close()
        main.app.router.routes = [
            route
            for route in main.app.router.routes
            if getattr(route, "path", None) != "/__test_only_raises_instrument_schema_error"
        ]

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "instrument_master_unavailable"
    assert "simulated: instrument master not loaded" in body["detail"]


def test_unexpected_exception_never_leaks_traceback_or_path() -> None:
    """Reproduces a real found gap: an unhandled exception on a route
    without its own try/except fell through to a raw 500 with no stable,
    caller-parseable contract. The fix must NOT swallow the failure (still
    a real 500, not a fake 200) but must never leak a stack trace or
    filesystem path to the client."""
    main = importlib.import_module("main")

    @main.app.get("/__test_only_raises_unexpected_error")
    def _raise_unexpected_error() -> None:
        raise RuntimeError(r"boom at D:\BharatTerminal\indian-terminal\secret\path.txt")

    client = TestClient(main.app, raise_server_exceptions=False)
    try:
        response = client.get(
            "/__test_only_raises_unexpected_error",
            headers={"X-Process-Token": main.PROCESS_TOKEN},
        )
    finally:
        client.close()
        main.app.router.routes = [
            route
            for route in main.app.router.routes
            if getattr(route, "path", None) != "/__test_only_raises_unexpected_error"
        ]

    assert response.status_code == 500
    body = response.json()
    assert body == {"error": "internal_error", "detail": "an unexpected error occurred"}
    assert "BharatTerminal" not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text


def test_get_cache_inventory_is_explicit_and_process_local() -> None:
    for module_name, names in PROCESS_LOCAL_MEMOIZATION.items():
        module = importlib.import_module(module_name)
        missing = sorted(name for name in names if not hasattr(module, name))
        assert missing == [], (module_name, missing)
    main = importlib.import_module("main")
    discovered = {
        name
        for name, value in vars(main).items()
        if ("cache" in name.lower() or name == "_bif_engine")
        and not name.startswith("__")
        and not callable(value)
        and not name.endswith("_lock")
        and not name.isupper()
    }
    assert discovered == PROCESS_LOCAL_MEMOIZATION["main"]


@pytest.mark.contract
def test_every_get_route_preserves_all_persistent_state(
    isolated_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = isolated_app_data / "data" / "terminal.db"
    monkeypatch.setattr(duckdb_client, "DB_PATH", database)
    duckdb_client.init_schema()
    _seed_read_fixtures(database)

    main = importlib.import_module("main")
    monkeypatch.setattr(
        main.credential_manager,
        "status",
        lambda: credential_manager.CredentialStatus(
            has_angel_credentials=False,
            has_groq_key=False,
        ),
    )
    redis_fixture = ReadOnlyRedisFixture()
    monkeypatch.setattr(main, "_redis_client", redis_fixture)

    def unavailable_request(*_: Any, **__: Any) -> UnavailableResponse:
        return UnavailableResponse()

    monkeypatch.setattr(requests.sessions.Session, "request", unavailable_request)
    monkeypatch.setattr(requests.api, "request", unavailable_request)
    from cross_asset import fetcher as cross_asset_fetcher
    from sentiment import rss_fetcher

    monkeypatch.setattr(
        cross_asset_fetcher,
        "fetch_cross_asset_snapshot",
        lambda nifty_spot=0.0: {
            "source_state": "UNAVAILABLE_TEST_BOUNDARY",
            "nifty_spot": nifty_spot,
        },
    )
    monkeypatch.setattr(rss_fetcher, "fetch_all_feeds", lambda **_: [])
    monkeypatch.setattr(rss_fetcher, "fetch_nse_announcements", lambda: [])

    main._all_signals_cache.clear()
    main._all_signals_cache_time.clear()
    main._apex_cache.clear()
    main._sentiment_cache.clear()
    main._cross_asset_cache.clear()
    # Reproduced live: conftest.py sets BHARATTERMINAL_APP_DATA once,
    # process-wide, to the shared session root -- and nests pytest's own
    # tmp_path basetemp *inside* that same root (config.option.basetemp =
    # TEST_APP_DATA_ROOT / "pytest-tmp"). Re-reading that env var here
    # scanned every OTHER test's own tmp_path subdirectory as if it were
    # this test's runtime state: any sibling test's ordinary (correct)
    # teardown activity landing inside this test's before/after window
    # (e.g. a SQLite WAL checkpoint on connection close) surfaced as a
    # false GET-purity violation, non-deterministically, depending only on
    # collection order and teardown timing -- never on what this test's
    # own GET requests actually did. isolated_app_data (used two lines
    # above for `database`) is this test's own already-isolated directory;
    # scope the filesystem snapshot to exactly that, not the shared root.
    runtime_root = isolated_app_data.resolve()

    before = {
        "duckdb": _duckdb_digest(database),
        "filesystem_tree": _tree_manifest(runtime_root),
        "redis": redis_fixture.state_digest(),
        "jobs": _job_state_digest(main),
    }
    routes = _application_get_routes(main.app)
    client = TestClient(main.app)
    responses: dict[str, list[int]] = {route.path: [] for route in routes}
    try:
        headers = {"X-Process-Token": main.PROCESS_TOKEN}
        for _ in range(2):
            for route in routes:
                path = _concrete_path(route)
                response = client.get(path, headers=headers)
                responses[route.path].append(response.status_code)
                assert response.status_code not in {500, 501, 502, 504}, (
                    route.path,
                    response.status_code,
                    response.text,
                )
    finally:
        client.close()

    deadline = time.monotonic() + 2
    while main._sentiment_fetch_in_progress and time.monotonic() < deadline:
        time.sleep(0.01)

    after = {
        "duckdb": _duckdb_digest(database),
        "filesystem_tree": _tree_manifest(runtime_root),
        "redis": redis_fixture.state_digest(),
        "jobs": _job_state_digest(main),
    }
    assert set(responses) == {route.path for route in routes}
    assert all(len(statuses) == 2 for statuses in responses.values())
    assert after == before


def test_get_purity_snapshot_is_scoped_to_this_tests_own_directory(
    isolated_app_data: Path,
) -> None:
    """Regression for the exact live flake: BHARATTERMINAL_APP_DATA is one
    fixed, session-wide directory (conftest.py), and pytest's own tmp_path
    basetemp is nested *inside* it -- so a filesystem snapshot rooted at
    os.environ["BHARATTERMINAL_APP_DATA"] (the old, buggy computation)
    silently includes every sibling test's own tmp_path subdirectory.
    Reproduced live: some sibling test's ordinary teardown (e.g. a SQLite
    WAL checkpoint on connection close) landing inside that shared root
    during this test's before/after window surfaced as a false GET-purity
    violation, non-deterministically, only in fresh-clone full-suite runs
    where collection/teardown timing happened to line up. The snapshot
    root must be this test's own isolated_app_data, never the wider
    session root a sibling test also happens to write under."""
    shared_root = Path(os.environ["BHARATTERMINAL_APP_DATA"]).resolve()
    assert isolated_app_data.resolve().is_relative_to(shared_root), (
        "test fixture assumption changed: isolated_app_data is no longer "
        "nested under the shared session root -- re-examine whether the "
        "contamination this regression guards against is still possible"
    )

    sibling_dir = shared_root / "pytest-tmp" / "some_other_test0"
    sibling_dir.mkdir(parents=True, exist_ok=True)
    before = _tree_manifest(isolated_app_data.resolve())

    (sibling_dir / "unrelated_sibling_artifact.txt").write_bytes(b"not this test's state")

    after = _tree_manifest(isolated_app_data.resolve())

    assert after == before, "a sibling test's own tmp directory must never appear in this scan"
