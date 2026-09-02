from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any

from data.paths import DATA_DIR


REGISTRY_PATH = DATA_DIR / "ml" / "model_registry.db"
VALID_STATUSES = {"VALIDATED", "REJECTED", "CORRUPT", "RETIRED"}
FALLBACK_STATUSES = {"PRIMARY", "PROXY", "FALLBACK", "NONE"}
SAFE_SERIALIZATION_FORMAT = "SKOPS"
SAFE_ARTIFACT_FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema_version(columns: list[str]) -> str:
    encoded = json.dumps(columns, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"features-v1:{hashlib.sha256(encoded).hexdigest()[:16]}"


def ensure_registry() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_version TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                universe TEXT NOT NULL,
                training_data_horizon TEXT NOT NULL,
                feature_schema_version TEXT NOT NULL,
                training_timestamp TEXT NOT NULL,
                artifact_path TEXT NOT NULL UNIQUE,
                artifact_hash TEXT NOT NULL,
                artifact_size INTEGER NOT NULL,
                serialization_format TEXT NOT NULL DEFAULT 'UNTRUSTED_LEGACY',
                artifact_format_version INTEGER NOT NULL DEFAULT 0,
                code_version TEXT NOT NULL,
                release_version TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                fallback_status TEXT NOT NULL,
                train_start TEXT NOT NULL,
                train_end TEXT NOT NULL,
                validation_start TEXT NOT NULL,
                validation_end TEXT NOT NULL,
                test_start TEXT NOT NULL,
                test_end TEXT NOT NULL,
                train_rows INTEGER NOT NULL,
                validation_sample_count INTEGER NOT NULL,
                test_sample_count INTEGER NOT NULL,
                validation_metric_name TEXT,
                validation_metric_value REAL,
                calibration_metric_name TEXT,
                calibration_metric_value REAL,
                calibration_sample_count INTEGER NOT NULL,
                dataset_fingerprint TEXT,
                runtime_versions TEXT,
                expected_feature_count INTEGER,
                registered_at TEXT NOT NULL,
                retired_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_model_registry_lookup
            ON model_registry(model_id, symbol, validation_status, training_timestamp DESC);
            """
        )
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(model_registry)").fetchall()
        }
        if "serialization_format" not in columns:
            connection.execute(
                "ALTER TABLE model_registry ADD COLUMN serialization_format TEXT NOT NULL DEFAULT 'UNTRUSTED_LEGACY'"
            )
        if "artifact_format_version" not in columns:
            connection.execute(
                "ALTER TABLE model_registry ADD COLUMN artifact_format_version INTEGER NOT NULL DEFAULT 0"
            )
        for name, data_type in (
            ("dataset_fingerprint", "TEXT"),
            ("runtime_versions", "TEXT"),
            ("expected_feature_count", "INTEGER"),
        ):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE model_registry ADD COLUMN {name} {data_type}"
                )


def register_artifact(path: Path, artifact: dict[str, Any], saved_root: Path) -> dict[str, Any]:
    ensure_registry()
    resolved = path.resolve()
    root = saved_root.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("model artifact is outside the runtime model directory") from error
    if artifact.get("validation_status") != "VALIDATED":
        raise RuntimeError("only validated model artifacts may be registered")
    _validate_metadata(artifact)
    record = {
        "model_version": str(artifact["model_version"]),
        "model_id": str(artifact["model_id"]),
        "symbol": str(artifact["symbol"]),
        "universe": str(artifact["universe"]),
        "training_data_horizon": str(artifact["training_data_horizon"]),
        "feature_schema_version": str(artifact["feature_schema_version"]),
        "training_timestamp": str(artifact["training_timestamp"]),
        "artifact_path": relative,
        "artifact_hash": sha256_file(resolved),
        "artifact_size": resolved.stat().st_size,
        "serialization_format": str(artifact["serialization_format"]),
        "artifact_format_version": int(artifact["artifact_format_version"]),
        "code_version": str(artifact["code_version"]),
        "release_version": str(artifact["release_version"]),
        "validation_status": "VALIDATED",
        "fallback_status": str(artifact.get("fallback_status", "PRIMARY")),
        "train_start": str(artifact["train_start"]),
        "train_end": str(artifact["train_end"]),
        "validation_start": str(artifact["validation_start"]),
        "validation_end": str(artifact["validation_end"]),
        "test_start": str(artifact["test_start"]),
        "test_end": str(artifact["test_end"]),
        "train_rows": int(artifact["train_rows"]),
        "validation_sample_count": int(artifact["validation_sample_count"]),
        "test_sample_count": int(artifact["test_sample_count"]),
        "validation_metric_name": artifact.get("validation_metric_name"),
        "validation_metric_value": artifact.get("validation_metric_value"),
        "calibration_metric_name": artifact.get("calibration_metric_name"),
        "calibration_metric_value": artifact.get("calibration_metric_value"),
        "calibration_sample_count": int(artifact["calibration_sample_count"]),
        "dataset_fingerprint": str(artifact["dataset_fingerprint"]),
        "runtime_versions": json.dumps(artifact["runtime_versions"], sort_keys=True),
        "expected_feature_count": int(artifact["expected_feature_count"]),
        "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "retired_at": None,
    }
    if record["fallback_status"] not in FALLBACK_STATUSES:
        raise RuntimeError("invalid model fallback status")
    if (
        record["serialization_format"] != SAFE_SERIALIZATION_FORMAT
        or record["artifact_format_version"] != SAFE_ARTIFACT_FORMAT_VERSION
    ):
        raise RuntimeError("unsafe model serialization format")
    columns = list(record)
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            f"INSERT INTO model_registry ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [record[column] for column in columns],
        )
        connection.commit()
    return record


def latest_validated_record(model_id: str, symbol: str) -> dict[str, Any] | None:
    if not REGISTRY_PATH.is_file():
        return None
    with _connect(read_only=True) as connection:
        if not _supports_safe_artifacts(connection):
            return None
        row = connection.execute(
            """
            SELECT * FROM model_registry
            WHERE model_id = ? AND symbol = ? AND validation_status = 'VALIDATED'
              AND retired_at IS NULL
              AND serialization_format = 'SKOPS' AND artifact_format_version = 1
            ORDER BY training_timestamp DESC, registered_at DESC
            LIMIT 1
            """,
            [model_id, symbol],
        ).fetchone()
    return dict(row) if row else None


def validated_records(model_ids: list[str] | None = None) -> list[dict[str, Any]]:
    if not REGISTRY_PATH.is_file():
        return []
    parameters: list[Any] = []
    where = "validation_status = 'VALIDATED' AND retired_at IS NULL"
    if model_ids:
        where += f" AND model_id IN ({', '.join('?' for _ in model_ids)})"
        parameters.extend(model_ids)
    with _connect(read_only=True) as connection:
        if not _supports_safe_artifacts(connection):
            return []
        where += " AND serialization_format = 'SKOPS' AND artifact_format_version = 1"
        rows = connection.execute(
            f"SELECT * FROM model_registry WHERE {where} ORDER BY training_timestamp DESC",
            parameters,
        ).fetchall()
    return [dict(row) for row in rows]


def registry_health() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file():
        return {
            "state": "EMPTY",
            "path": str(REGISTRY_PATH),
            "validated_models": 0,
            "corrupt_models": 0,
            "latest_training_timestamp": None,
            "reason": "model registry not created",
        }
    try:
        with _connect(read_only=True) as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN validation_status = 'VALIDATED' AND retired_at IS NULL
                                  AND serialization_format = 'SKOPS' AND artifact_format_version = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN validation_status = 'CORRUPT'
                                  OR serialization_format != 'SKOPS' OR artifact_format_version != 1 THEN 1 ELSE 0 END),
                    MAX(CASE WHEN validation_status = 'VALIDATED'
                                  AND serialization_format = 'SKOPS' AND artifact_format_version = 1
                             THEN training_timestamp END)
                FROM model_registry
                """
            ).fetchone()
    except sqlite3.Error as error:
        return {
            "state": "UNAVAILABLE",
            "path": str(REGISTRY_PATH),
            "validated_models": 0,
            "corrupt_models": 0,
            "latest_training_timestamp": None,
            "reason": f"registry read failed: {type(error).__name__}",
        }
    validated = int(row[0] or 0)
    corrupt = int(row[1] or 0)
    return {
        "state": "HEALTHY" if validated and not corrupt else "DEGRADED" if corrupt else "EMPTY",
        "path": str(REGISTRY_PATH),
        "validated_models": validated,
        "corrupt_models": corrupt,
        "latest_training_timestamp": row[2],
        "reason": None if validated and not corrupt else "no validated models" if not corrupt else "corrupt models recorded",
    }


def resolve_verified_path(record: dict[str, Any], saved_root: Path) -> Path | None:
    root = saved_root.resolve()
    path = (root / str(record["artifact_path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if (
        not path.is_file()
        or path.stat().st_size != int(record["artifact_size"])
        or sha256_file(path) != str(record["artifact_hash"])
    ):
        return None
    return path


def read_verified_artifact_bytes(
    record: dict[str, Any],
    saved_root: Path,
    *,
    maximum_bytes: int,
) -> tuple[Path, bytes] | None:
    """Hash and return one bounded open-file snapshot; never verify then reopen."""
    root = saved_root.resolve()
    path = (root / str(record["artifact_path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        expected_size = int(record["artifact_size"])
        if (
            not stat.S_ISREG(metadata.st_mode)
            or expected_size <= 0
            or expected_size > maximum_bytes
            or metadata.st_size != expected_size
        ):
            return None
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            return None
        data = b"".join(chunks)
        if hashlib.sha256(data).hexdigest() != str(record["artifact_hash"]):
            return None
        return path, data
    finally:
        os.close(descriptor)


def retire_missing_versions(model_id: str, symbol: str, retained_paths: set[str]) -> None:
    ensure_registry()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT model_version, artifact_path FROM model_registry WHERE model_id = ? AND symbol = ? AND retired_at IS NULL",
            [model_id, symbol],
        ).fetchall()
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        for row in rows:
            if str(row["artifact_path"]) not in retained_paths:
                connection.execute(
                    "UPDATE model_registry SET validation_status = 'RETIRED', retired_at = ? WHERE model_version = ?",
                    [now, row["model_version"]],
                )


def _validate_metadata(artifact: dict[str, Any]) -> None:
    required = {
        "model_id", "model_version", "symbol", "universe", "training_data_horizon",
        "feature_schema_version", "training_timestamp", "code_version", "release_version",
        "validation_status", "fallback_status", "train_start", "train_end",
        "serialization_format", "artifact_format_version",
        "validation_start", "validation_end", "test_start", "test_end", "train_rows",
        "validation_sample_count", "test_sample_count", "calibration_sample_count",
        "dataset_fingerprint", "runtime_versions", "expected_feature_count",
    }
    missing = sorted(field for field in required if artifact.get(field) in (None, ""))
    if missing:
        raise RuntimeError(f"model metadata missing fields: {', '.join(missing)}")
    if not (
        str(artifact["train_end"]) < str(artifact["validation_start"])
        and str(artifact["validation_end"]) < str(artifact["test_start"])
    ):
        raise RuntimeError("model temporal split boundaries overlap")


def _set_status(model_version: str, status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError("invalid model registry status")
    with _connect() as connection:
        connection.execute(
            "UPDATE model_registry SET validation_status = ? WHERE model_version = ?",
            [status, model_version],
        )


def _connect(*, read_only: bool = False) -> sqlite3.Connection:
    if read_only and not REGISTRY_PATH.exists():
        raise RuntimeError("model registry does not exist")
    connection = sqlite3.connect(
        f"file:{REGISTRY_PATH.as_posix()}?mode=ro" if read_only else REGISTRY_PATH,
        uri=read_only,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    if not read_only:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
    return connection


def _supports_safe_artifacts(connection: sqlite3.Connection) -> bool:
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(model_registry)").fetchall()}
    return {"serialization_format", "artifact_format_version"}.issubset(columns)
