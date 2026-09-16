"""Weighted prediction from matched historical patterns."""

from collections.abc import Sequence

from tradingagents.pattern_memory.models import PatternMatch, Prediction


def predict_from_matches(matches: Sequence[PatternMatch]) -> Prediction:
    """Estimate the next return/range from weighted historical outcomes."""
    usable = [m for m in matches if m.observation.next_return is not None and m.weight > 0]
    if not usable:
        raise ValueError("At least one usable pattern match is required")

    total_weight = sum(match.weight for match in usable)
    expected_return = sum(
        match.weight * match.observation.next_return  # type: ignore[operator]
        for match in usable
    ) / total_weight

    ranges = [m for m in usable if m.observation.next_range is not None]
    expected_range = None
    if ranges:
        expected_range = sum(
            match.weight * match.observation.next_range  # type: ignore[operator]
            for match in ranges
        ) / sum(match.weight for match in ranges)

    timeframes = {match.observation.timeframe for match in usable}
    timeframe = next(iter(timeframes)) if len(timeframes) == 1 else "mixed"
    return Prediction(
        expected_return=expected_return,
        expected_range=expected_range,
        matches_used=len(usable),
        total_weight=total_weight,
        evidence_ids=tuple(match.observation.pattern_id for match in usable),
        timeframe=timeframe,
    )
