from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


PRICE_POLICY_VERSION = "fo-executable-v1"
VERIFIED_EXECUTION_SOURCES = frozenset({"NSE_FO_BHAVCOPY"})


class HistoricalPriceQuality(StrEnum):
    VALID_EXECUTABLE_OBSERVATION = "VALID_EXECUTABLE_OBSERVATION"
    VALID_NON_EXECUTABLE_REFERENCE = "VALID_NON_EXECUTABLE_REFERENCE"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HistoricalPriceObservation:
    price: float | None
    quality: HistoricalPriceQuality
    source: str
    observed_on: dt.date | None
    reason: str
    policy_version: str = PRICE_POLICY_VERSION

    @property
    def executable(self) -> bool:
        return self.quality is HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION


def classify_historical_option_row(
    row: Mapping[str, Any],
    *,
    price_type: str = "close",
) -> HistoricalPriceObservation:
    source = str(row.get("source") or "UNKNOWN").strip().upper()
    trade_day = _date_or_none(row.get("date"))
    expiry = _date_or_none(row.get("expiry"))
    price = _finite_or_none(row.get(price_type))
    recorded = _recorded_quality(row.get("price_quality"))

    if recorded in {HistoricalPriceQuality.INVALID, HistoricalPriceQuality.UNKNOWN}:
        return HistoricalPriceObservation(price, recorded, source, trade_day, "recorded_quality")
    if not str(row.get("symbol") or "").strip():
        return HistoricalPriceObservation(price, HistoricalPriceQuality.INVALID, source, trade_day, "missing_symbol")
    if str(row.get("option_type") or "").strip().upper() not in {"CE", "PE", "FUT"}:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.INVALID, source, trade_day, "invalid_contract_type")
    if trade_day is None or expiry is None:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.INVALID, source, trade_day, "missing_contract_date")
    if trade_day > expiry:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.INVALID, source, trade_day, "expired_contract")

    values = {name: _finite_or_none(row.get(name)) for name in ("open", "high", "low", "close")}
    if any(value is None for value in values.values()) or price is None:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.INVALID, source, trade_day, "non_finite_ohlc")
    open_price = values["open"] or 0.0
    high = values["high"] or 0.0
    low = values["low"] or 0.0
    close = values["close"] or 0.0
    if close > 0 and open_price == high == low == 0:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.UNKNOWN, source, trade_day, "zero_ohl_unproven_close")
    if min(open_price, high, low, close) < 0 or low > min(open_price, close) or high < max(open_price, close) or high < low:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.INVALID, source, trade_day, "invalid_candle")
    if price <= 0:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.VALID_NON_EXECUTABLE_REFERENCE, source, trade_day, "non_positive_fill_price")

    volume = _integer_or_none(row.get("volume"))
    if volume is None or volume <= 0:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.VALID_NON_EXECUTABLE_REFERENCE, source, trade_day, "no_traded_volume")
    if source not in VERIFIED_EXECUTION_SOURCES:
        return HistoricalPriceObservation(price, HistoricalPriceQuality.UNKNOWN, source, trade_day, "unverified_source")
    if recorded is HistoricalPriceQuality.VALID_NON_EXECUTABLE_REFERENCE:
        return HistoricalPriceObservation(price, recorded, source, trade_day, "recorded_non_executable")
    return HistoricalPriceObservation(
        price,
        HistoricalPriceQuality.VALID_EXECUTABLE_OBSERVATION,
        source,
        trade_day,
        "verified_ohlc_volume_source",
    )


def quality_for_ingest(row: Mapping[str, Any]) -> str:
    return classify_historical_option_row(row).quality.value


def _recorded_quality(value: Any) -> HistoricalPriceQuality | None:
    if value in (None, ""):
        return None
    try:
        return HistoricalPriceQuality(str(value).strip().upper())
    except ValueError:
        return HistoricalPriceQuality.UNKNOWN


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _date_or_none(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
