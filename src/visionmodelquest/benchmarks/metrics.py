from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be greater than zero and at most one")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def summarise(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p95": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": nearest_rank_percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }

