"""Data contracts for historical pattern memory."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PatternObservation:
    """A normalized market state and its observed next-period outcome."""

    pattern_id: str
    ticker: str
    timeframe: str
    timestamp: datetime
    features: tuple[float, ...]
    next_return: float | None = None
    next_range: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReliabilityRecord:
    """Bounded reliability state for one historical pattern."""

    pattern_id: str
    score: float = 1.0
    observations: int = 0
    mean_error: float = 0.0


@dataclass(frozen=True)
class PatternMatch:
    """A historical observation matched against the current state."""

    observation: PatternObservation
    similarity: float
    reliability: float = 1.0

    @property
    def weight(self) -> float:
        """Combined similarity/reliability weight used by the predictor."""
        return max(0.0, self.similarity) * max(0.0, self.reliability)


@dataclass(frozen=True)
class Prediction:
    """Weighted historical estimate with enough evidence for auditing."""

    expected_return: float
    expected_range: float | None
    matches_used: int
    total_weight: float
    evidence_ids: tuple[str, ...]
    timeframe: str


@dataclass(frozen=True)
class PatternEvaluation:
    """Comparison between a prediction and the realized next period."""

    pattern_id: str
    predicted_return: float
    actual_return: float
    absolute_error: float
    relative_error: float
