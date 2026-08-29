from __future__ import annotations

from typing import Any

from backtest.spot_quality import INVALID_DERIVED_SPOT_SOURCE, VERIFIED_INDEX_SPOT_SOURCES
from db.write_service import ReadOperation, resolve_read


DERIVED_NIFTY_SOURCE = INVALID_DERIVED_SPOT_SOURCE.lower()


def verified_nifty_spot_availability() -> dict[str, Any]:
    """Report canonical index observations; never derive spot from futures."""
    return _spot_availability("VERIFIED_SPOT_LOOKUP")


def _spot_availability(operation: str) -> dict[str, Any]:
    verified_sources = sorted(VERIFIED_INDEX_SPOT_SOURCES)
    result = resolve_read(
        ReadOperation.LOAD_NIFTY_SPOT_AVAILABILITY_STATS,
        {"verified_sources": verified_sources, "derived_source": DERIVED_NIFTY_SOURCE},
    )
    verified = result["verified"]
    quarantined = result["quarantined"]
    rows = int(verified[0] or 0)
    return {
        "symbol": "NIFTY",
        "operation": operation,
        "available": rows > 0,
        "status": "AVAILABLE" if rows > 0 else "UNAVAILABLE_VERIFIED_SPOT_DATA",
        "verified_sources": verified_sources,
        "verified_rows": rows,
        "min_date": str(verified[1]) if verified[1] is not None else None,
        "max_date": str(verified[2]) if verified[2] is not None else None,
        "quarantined_derived_rows": int(quarantined[0] or 0),
        "rows_added": 0,
    }
