from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import pandas as pd

from db.write_service import WriteOperation, submit_write


def ensure_training_log_table() -> None:
    submit_write(WriteOperation.ENSURE_ML_TRAINING_LOG, {})


def log_training_result(
    symbol: str,
    model_id: str,
    result: dict[str, Any] | None,
    *,
    context: Any | None = None,
    status: str | None = None,
    error_msg: str | None = None,
) -> None:
    payload = result or {}
    resolved_status = status or ("success" if payload.get("is_trained") else "failed")
    resolved_error = error_msg
    if resolved_error is None and resolved_status != "success":
        resolved_error = str(payload.get("error") or payload.get("reason") or "training failed")

    n_rows = _int_or_none(payload.get("train_rows"))
    if n_rows is None:
        n_rows = _int_or_none(payload.get("available_rows"))
    if n_rows is None and context is not None:
        n_rows = _int_or_none(getattr(context, "available_rows", None))

    n_features = _int_or_none(payload.get("n_features"))
    if n_features is None:
        columns = payload.get("feature_columns")
        if isinstance(columns, list):
            n_features = len(columns)
    if n_features is None and context is not None:
        frame = getattr(context, "feature_frame", pd.DataFrame())
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            n_features = int(len([col for col in frame.columns if col != "date"]))

    bif_rows_used = _int_or_none(payload.get("bif_rows_used"))
    if bif_rows_used is None and context is not None:
        frame = getattr(context, "feature_frame", pd.DataFrame())
        if isinstance(frame, pd.DataFrame) and "bif_available" in frame.columns:
            bif_rows_used = int(pd.to_numeric(frame["bif_available"], errors="coerce").fillna(0).sum())

    ohlcv_rows_used = _int_or_none(payload.get("ohlcv_rows_used"))
    if ohlcv_rows_used is None and context is not None:
        ohlcv = getattr(context, "ohlcv", pd.DataFrame())
        if isinstance(ohlcv, pd.DataFrame):
            ohlcv_rows_used = int(len(ohlcv))

    submit_write(
        WriteOperation.RECORD_ML_TRAINING_LOG,
        {
            "symbol": symbol.strip().upper(),
            "model_id": model_id,
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_rows": n_rows,
            "n_features": n_features,
            "train_score": _float_or_none(payload.get("train_score")),
            "val_score": _float_or_none(payload.get("val_score")),
            "feature_importance_top5": json.dumps(_top_feature_importance(payload), sort_keys=True),
            "bif_rows_used": bif_rows_used,
            "ohlcv_rows_used": ohlcv_rows_used,
            "status": resolved_status,
            "error_msg": resolved_error,
        },
    )


def _top_feature_importance(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("feature_importance_top5") or payload.get("feature_importance") or {}
    if isinstance(raw, list):
        pairs: list[tuple[str, float]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("feature") or "").strip()
            value = _float_or_none(item.get("importance") or item.get("value"))
            if name and value is not None:
                pairs.append((name, value))
    elif isinstance(raw, dict):
        pairs = [
            (str(name), float(value))
            for name, value in raw.items()
            if _float_or_none(value) is not None
        ]
    else:
        pairs = []
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:5]
    return {name: round(value, 6) for name, value in pairs}


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
