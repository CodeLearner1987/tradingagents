"""Similarity matching for historical market states."""

from math import sqrt
from collections.abc import Iterable, Sequence

from tradingagents.pattern_memory.models import PatternMatch, PatternObservation, ReliabilityRecord


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Pattern feature vectors must have the same length")
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Convert Euclidean distance into a bounded similarity in (0, 1]."""
    return 1.0 / (1.0 + _distance(left, right))


def find_matches(
    current_features: Sequence[float],
    observations: Iterable[PatternObservation],
    *,
    timeframe: str | None = None,
    ticker: str | None = None,
    reliabilities: dict[str, ReliabilityRecord] | None = None,
    limit: int = 10,
) -> list[PatternMatch]:
    """Return the closest historical observations with optional reliability weighting."""
    if limit <= 0:
        return []

    records: list[PatternMatch] = []
    reliability_map = reliabilities or {}
    for observation in observations:
        if observation.next_return is None:
            continue
        if timeframe is not None and observation.timeframe != timeframe:
            continue
        if ticker is not None and observation.ticker != ticker:
            continue
        similarity = _similarity(current_features, observation.features)
        reliability = reliability_map.get(observation.pattern_id, ReliabilityRecord(observation.pattern_id)).score
        records.append(PatternMatch(observation, similarity, reliability))

    records.sort(key=lambda match: (-match.weight, match.observation.timestamp, match.observation.pattern_id))
    return records[:limit]
