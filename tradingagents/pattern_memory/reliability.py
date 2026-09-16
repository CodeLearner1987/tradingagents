"""Online reliability scoring for historical pattern memory."""

from tradingagents.pattern_memory.models import ReliabilityRecord


def update_reliability(
    record: ReliabilityRecord,
    *,
    predicted_return: float,
    actual_return: float,
    learning_rate: float = 0.25,
) -> ReliabilityRecord:
    """Update reliability from prediction error without unbounded scores.

    A perfect prediction pushes the score toward 1.0; larger errors push it
    toward 0.0. The update is incremental so one observation cannot dominate
    the historical record.
    """
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")

    error = abs(predicted_return - actual_return)
    accuracy = 1.0 / (1.0 + error)
    score = (1.0 - learning_rate) * record.score + learning_rate * accuracy
    mean_error = (
        record.mean_error * record.observations + error
    ) / (record.observations + 1)
    return ReliabilityRecord(
        pattern_id=record.pattern_id,
        score=min(1.0, max(0.0, score)),
        observations=record.observations + 1,
        mean_error=mean_error,
    )
