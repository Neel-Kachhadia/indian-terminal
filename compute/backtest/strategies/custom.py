from __future__ import annotations

from typing import Any

from .base import BaseStrategy


def strategy_from_definition(definition: dict[str, Any], key: str = "custom") -> BaseStrategy:
    if not isinstance(definition, dict):
        raise ValueError("strategy definition must be an object")
    if not definition.get("legs"):
        raise ValueError("strategy definition requires at least one leg")
    return BaseStrategy(key, definition)
