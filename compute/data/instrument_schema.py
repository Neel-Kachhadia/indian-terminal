from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


INSTRUMENT_SCHEMA_VERSION = 1
CANONICAL_INSTRUMENT_COLUMNS = (
    "schema_version",
    "token",
    "symbol",
    "name",
    "exchange",
    "instrument_type",
    "expiry",
    "strike",
    "option_type",
    "lot_size",
    "tick_size",
    "isin",
    "updated_at",
)

UPSTREAM_RENAMES = {
    "exch_seg": "exchange",
    "instrumenttype": "instrument_type",
    "optiontype": "option_type",
    "lotsize": "lot_size",
}


class InstrumentSchemaError(ValueError):
    pass


def canonicalize_instrument_frame(
    raw: pd.DataFrame,
    *,
    updated_at: datetime | None = None,
) -> pd.DataFrame:
    frame = raw.copy()
    if "schema_version" in frame.columns:
        frame = frame.loc[:, CANONICAL_INSTRUMENT_COLUMNS].copy()
        validate_instrument_frame(frame)
        return frame
    strike_is_already_rupees = "exchange" in frame.columns and "instrument_type" in frame.columns
    collisions = [
        (legacy, canonical)
        for legacy, canonical in UPSTREAM_RENAMES.items()
        if legacy in frame.columns and canonical in frame.columns
    ]
    if collisions:
        names = ", ".join(f"{legacy}/{canonical}" for legacy, canonical in collisions)
        raise InstrumentSchemaError(f"ambiguous instrument columns: {names}")
    frame = frame.rename(
        columns={legacy: canonical for legacy, canonical in UPSTREAM_RENAMES.items() if legacy in frame.columns}
    )

    for column in CANONICAL_INSTRUMENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    text_columns = (
        "token",
        "symbol",
        "name",
        "exchange",
        "instrument_type",
        "option_type",
        "isin",
    )
    for column in text_columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.upper()

    timestamp = updated_at or datetime.now(UTC)
    frame["schema_version"] = INSTRUMENT_SCHEMA_VERSION
    frame["updated_at"] = timestamp.isoformat()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce").fillna(0)
    if not strike_is_already_rupees:
        frame["strike"] = frame["strike"] / 100
    frame["lot_size"] = pd.to_numeric(frame["lot_size"], errors="coerce").fillna(0).astype(int)
    frame["tick_size"] = pd.to_numeric(frame["tick_size"], errors="coerce")
    frame["expiry"] = pd.to_datetime(
        frame["expiry"], format="mixed", errors="coerce"
    ).dt.date
    frame = frame.loc[:, CANONICAL_INSTRUMENT_COLUMNS].copy()
    validate_instrument_frame(frame)
    return frame


def validate_instrument_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_INSTRUMENT_COLUMNS if column not in frame.columns]
    if missing:
        raise InstrumentSchemaError(f"missing canonical instrument columns: {', '.join(missing)}")
    versions = {int(value) for value in frame["schema_version"].dropna().unique().tolist()}
    if versions != {INSTRUMENT_SCHEMA_VERSION}:
        raise InstrumentSchemaError(
            f"instrument schema version mismatch: expected {INSTRUMENT_SCHEMA_VERSION}, got {sorted(versions)}"
        )
    for column in ("token", "symbol", "exchange"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise InstrumentSchemaError(f"instrument {column} must be non-empty")
    duplicated = frame.duplicated(subset=["token", "exchange"], keep=False)
    if duplicated.any():
        raise InstrumentSchemaError("duplicate instrument token/exchange identity")


def read_instrument_frame(path: Any) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_instrument_frame(frame)
    return frame.loc[:, CANONICAL_INSTRUMENT_COLUMNS].copy()
