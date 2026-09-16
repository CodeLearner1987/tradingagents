"""Evaluate a prediction against the realized outcome."""

from tradingagents.pattern_memory.models import PatternEvaluation


def evaluate_prediction(
    pattern_id: str,
    *,
    predicted_return: float,
    actual_return: float,
) -> PatternEvaluation:
    """Return absolute and scale-normalized prediction error."""
    absolute_error = abs(predicted_return - actual_return)
    denominator = max(abs(actual_return), 1e-12)
    return PatternEvaluation(
        pattern_id=pattern_id,
        predicted_return=predicted_return,
        actual_return=actual_return,
        absolute_error=absolute_error,
        relative_error=absolute_error / denominator,
    )
