from __future__ import annotations

from typing import Any

import numpy as np


MAX_CONFIDENCE = 0.85


def normalise_probabilities(
    labels: list[str],
    raw: Any,
    *,
    fallback_label: str | None = None,
) -> dict[str, float]:
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size != len(labels) or not np.isfinite(values).all() or values.sum() <= 0:
        values = np.zeros(len(labels), dtype=float)
        if fallback_label in labels:
            values[labels.index(fallback_label)] = 1.0
        else:
            values[:] = 1.0 / max(1, len(labels))
    else:
        values = np.clip(values, 0.0, None)
        values = values / values.sum()
    return {label: round(float(value), 6) for label, value in zip(labels, values)}


def clipped_confidence(probabilities: dict[str, float], multiplier: float = 1.0) -> float:
    if not probabilities:
        return 0.0
    confidence = max(float(value) for value in probabilities.values()) * float(multiplier)
    return round(max(0.0, min(MAX_CONFIDENCE, confidence)), 6)


def fit_calibrated_classifier(
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    method: str,
    n_splits: int,
    gap: int,
) -> tuple[Any, bool, str | None]:
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit

        split_count = max(2, min(n_splits, len(y_train) // 30))
        if split_count < 2:
            return estimator.fit(x_train, y_train), False, "not enough rows for calibration split"

        cv = TimeSeriesSplit(n_splits=split_count, gap=gap)
        calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
        calibrated.fit(x_train, y_train)
        return calibrated, True, None
    except Exception as error:
        try:
            estimator.fit(x_train, y_train)
            return estimator, False, f"calibration unavailable: {error}"
        except Exception as fit_error:
            raise RuntimeError(f"classifier training failed: {fit_error}") from fit_error


def expected_calibration_error(
    probabilities: list[float],
    outcomes: list[int],
    bins: int = 10,
) -> float | None:
    if not probabilities or len(probabilities) != len(outcomes):
        return None
    probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(outcomes, dtype=float)
    if probs.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high if high < 1.0 else probs <= high)
        if not mask.any():
            continue
        weight = float(mask.mean())
        ece += weight * abs(float(probs[mask].mean()) - float(actual[mask].mean()))
    return round(ece, 6)
