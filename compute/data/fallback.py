from dataclasses import dataclass
from typing import Any


@dataclass
class DataResult:
    data: Any
    source: str
    is_stale: bool = False
    is_estimated: bool = False
    confidence: str = "high"
    error: str | None = None
    note: str | None = None
    staleness_days: int = 0

    @property
    def available(self) -> bool:
        return self.data is not None and self.confidence != "none"

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "is_stale": self.is_stale,
            "is_estimated": self.is_estimated,
            "confidence": self.confidence,
            "error": self.error,
            "note": self.note,
        }
