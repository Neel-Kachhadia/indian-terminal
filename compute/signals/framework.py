from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence


class SignalType(Enum):
    CONTINUOUS = "continuous"
    DIRECTIONAL = "directional"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    RATIO = "ratio"


class SignalRegime(Enum):
    ALL = "all"
    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"
    TRENDING = "trending"
    RANGING = "ranging"


class DataSource(Enum):
    OHLCV_DAILY = "ohlcv_daily"
    OPTIONS_CHAIN = "options_chain"
    REDIS_TICK = "redis_tick"
    FII_DII = "fii_dii"
    CROSS_ASSET = "cross_asset"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    FO_EOD = "fo_eod"
    TICK_DATA = "tick_data"
    POSITION_DATA = "position"


@dataclass
class Signal:
    name: str
    category: str
    value: Any
    signal_type: SignalType
    regime: SignalRegime
    data_source: DataSource
    is_available: bool
    description: str
    interpretation: str
    z_score: float | None
    percentile_52w: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "value": _json_value(self.value),
            "signal_type": self.signal_type.value,
            "regime": self.regime.value,
            "data_source": self.data_source.value,
            "is_available": self.is_available,
            "description": self.description,
            "interpretation": self.interpretation,
            "z_score": _json_value(self.z_score),
            "percentile_52w": _json_value(self.percentile_52w),
        }


@dataclass(frozen=True)
class SignalSpec:
    name: str
    signal_type: SignalType = SignalType.CONTINUOUS
    regime: SignalRegime = SignalRegime.ALL
    data_source: DataSource = DataSource.OHLCV_DAILY
    description: str = ""


@dataclass
class SignalContext:
    symbol: str
    ohlcv_df: Any
    chain_legs: list[dict[str, Any]]
    spot: float
    expiry_str: str
    cross_asset_snapshot: dict[str, Any]
    sentiment_data: dict[str, Any]
    fii_dii_data: dict[str, Any]
    india_vix: float | None
    lot_size: int | None = None
    computed_at: _dt.datetime = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )


InterpretationFn = Callable[[Any], str]


def build_signals(
    category: str,
    specs: Sequence[SignalSpec],
    values: dict[str, Any],
    available_sources: dict[DataSource, bool],
    histories: dict[str, Iterable[Any]] | None = None,
    interpretations: dict[str, str | InterpretationFn] | None = None,
) -> dict[str, Signal]:
    histories = histories or {}
    interpretations = interpretations or {}
    result: dict[str, Signal] = {}

    for spec in specs:
        value_present = spec.name in values
        value = clean_value(values.get(spec.name))
        source_available = available_sources.get(spec.data_source, False)
        is_available = bool(source_available and value_present and value is not None)
        history = histories.get(spec.name)
        z_score, percentile = distribution_stats(value, history)
        interpretation = _interpretation(spec, value, interpretations.get(spec.name))
        description = spec.description or f"{category} signal: {spec.name}"

        result[spec.name] = Signal(
            name=spec.name,
            category=category,
            value=value,
            signal_type=spec.signal_type,
            regime=spec.regime,
            data_source=spec.data_source,
            is_available=is_available,
            description=description,
            interpretation=interpretation,
            z_score=z_score,
            percentile_52w=percentile,
        )

    return result


def specs_from_names(
    names: Sequence[str],
    *,
    signal_type: SignalType = SignalType.CONTINUOUS,
    regime: SignalRegime = SignalRegime.ALL,
    data_source: DataSource = DataSource.OHLCV_DAILY,
    descriptions: dict[str, str] | None = None,
    type_overrides: dict[str, SignalType] | None = None,
    regime_overrides: dict[str, SignalRegime] | None = None,
    source_overrides: dict[str, DataSource] | None = None,
) -> list[SignalSpec]:
    descriptions = descriptions or {}
    type_overrides = type_overrides or {}
    regime_overrides = regime_overrides or {}
    source_overrides = source_overrides or {}
    return [
        SignalSpec(
            name=name,
            signal_type=type_overrides.get(name, signal_type),
            regime=regime_overrides.get(name, regime),
            data_source=source_overrides.get(name, data_source),
            description=descriptions.get(name, ""),
        )
        for name in names
    ]


def source_availability(ctx: SignalContext) -> dict[DataSource, bool]:
    ohlcv_available = getattr(ctx.ohlcv_df, "empty", True) is False
    chain_available = bool(ctx.chain_legs)
    cross_asset = bool(ctx.cross_asset_snapshot)
    sentiment = bool(ctx.sentiment_data)
    fii_dii = bool(ctx.fii_dii_data) and not bool(ctx.fii_dii_data.get("error"))
    india_vix = ctx.india_vix is not None and _finite(ctx.india_vix)
    spot = ctx.spot is not None and _finite(ctx.spot) and float(ctx.spot) > 0

    return {
        DataSource.OHLCV_DAILY: ohlcv_available,
        DataSource.OPTIONS_CHAIN: chain_available,
        DataSource.REDIS_TICK: spot,
        DataSource.FII_DII: fii_dii,
        DataSource.CROSS_ASSET: cross_asset,
        DataSource.SENTIMENT: sentiment,
        DataSource.MACRO: True,
        DataSource.FO_EOD: False,
        DataSource.TICK_DATA: False,
        DataSource.POSITION_DATA: False,
    }


def numeric_history_value(signal: Signal) -> float | None:
    value = signal.value
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if signal.signal_type in {SignalType.BINARY, SignalType.DIRECTIONAL}:
        return _finite_float(value)
    return _finite_float(value)


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return value
    number = _finite_float(value)
    if number is not None:
        return number
    return value


def distribution_stats(
    value: Any,
    history: Iterable[Any] | None,
) -> tuple[float | None, float | None]:
    current = _finite_float(value)
    if current is None or history is None:
        return None, None

    numbers = [_finite_float(item) for item in history]
    numbers = [item for item in numbers if item is not None]
    if len(numbers) < 20:
        return None, None

    numbers = numbers[-252:]
    mean = sum(numbers) / len(numbers)
    variance = sum((item - mean) ** 2 for item in numbers) / len(numbers)
    std = math.sqrt(variance)
    z_score = (current - mean) / std if std > 1e-12 else None
    percentile = sum(1 for item in numbers if item <= current) / len(numbers) * 100.0
    return _round_optional(z_score), _round_optional(percentile)


def _interpretation(
    spec: SignalSpec,
    value: Any,
    override: str | InterpretationFn | None,
) -> str:
    if callable(override):
        return override(value)
    if isinstance(override, str):
        return override
    if value is None:
        return "Data unavailable or insufficient history."
    if spec.signal_type == SignalType.CATEGORICAL:
        return f"Current regime: {value}."
    if spec.signal_type == SignalType.BINARY:
        return "Condition active." if float(value) == 1.0 else "Condition inactive."
    if spec.signal_type == SignalType.DIRECTIONAL:
        number = _finite_float(value)
        if number is None or abs(number) < 1e-12:
            return "Neutral directional reading."
        return "Bullish directional reading." if number > 0 else "Bearish directional reading."
    return f"Current value: {value}."


def _json_value(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite(value: Any) -> bool:
    return _finite_float(value) is not None


def _round_optional(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 6)
