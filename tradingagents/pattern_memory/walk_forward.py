"""Leakage-safe walk-forward validation for pattern memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from tradingagents.pattern_memory.evaluator import evaluate_prediction
from tradingagents.pattern_memory.features import build_features, feature_vector
from tradingagents.pattern_memory.matcher import find_matches
from tradingagents.pattern_memory.models import PatternObservation, ReliabilityRecord
from tradingagents.pattern_memory.predictor import predict_from_matches
from tradingagents.pattern_memory.reliability import update_reliability


@dataclass(frozen=True)
class ValidationResult:
    ticker: str
    timeframe: str
    predictions: int
    directional_hits: int
    absolute_error_sum: float
    insufficient_data: int
    evaluated_rows: int

    @property
    def directional_accuracy(self) -> float | None:
        return self.directional_hits / self.predictions if self.predictions else None

    @property
    def mae(self) -> float | None:
        return self.absolute_error_sum / self.predictions if self.predictions else None

    @property
    def insufficient_rate(self) -> float:
        return self.insufficient_data / self.evaluated_rows if self.evaluated_rows else 0.0


def _timestamp(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def walk_forward_validate(
    frame: pd.DataFrame,
    *,
    ticker: str,
    timeframe: str = "1d",
    k: int = 10,
    min_history: int = 100,
    reliability_learning_rate: float = 0.25,
) -> ValidationResult:
    """Validate chronologically, exposing only outcomes known before each test row."""
    if k <= 0:
        raise ValueError("k must be positive")
    if min_history < 1:
        raise ValueError("min_history must be positive")
    if len(frame) < 2:
        raise ValueError("At least two OHLCV rows are required")

    data = frame.sort_index().copy()
    features = build_features(data)
    observations: list[PatternObservation] = []
    reliabilities: dict[str, ReliabilityRecord] = {}
    predictions = directional_hits = insufficient_data = 0
    absolute_error_sum = 0.0
    evaluated_rows = 0

    for position in range(len(data) - 1):
        vector = feature_vector(features, position)
        if vector is None:
            continue
        evaluated_rows += 1

        # Only observations whose next candle has already closed are eligible.
        # At row t, the newest eligible historical pattern is t-1.
        historical = observations.copy()
        if position < min_history or not historical:
            pattern_id = f"{ticker}:{timeframe}:{position}"
            actual_return = float(data["Close"].iloc[position + 1] / data["Close"].iloc[position] - 1.0)
            observations.append(
                PatternObservation(
                    pattern_id=pattern_id,
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=_timestamp(data.index[position]),
                    features=vector,
                    next_return=actual_return,
                    metadata={"source": "walk_forward"},
                )
            )
            reliabilities[pattern_id] = ReliabilityRecord(pattern_id)
            continue

        matches = find_matches(
            vector,
            historical,
            ticker=ticker,
            timeframe=timeframe,
            reliabilities=reliabilities,
            limit=k,
        )

        actual_return = float(data["Close"].iloc[position + 1] / data["Close"].iloc[position] - 1.0)
        if not matches:
            insufficient_data += 1
        else:
            prediction = predict_from_matches(matches)
            evaluation = evaluate_prediction(
                f"{ticker}:{timeframe}:{position}",
                predicted_return=prediction.expected_return,
                actual_return=actual_return,
            )
            predictions += 1
            directional_hits += int(
                (prediction.expected_return >= 0 and actual_return >= 0)
                or (prediction.expected_return < 0 and actual_return < 0)
            )
            absolute_error_sum += evaluation.absolute_error

            # Reliability is updated only after the outcome for this test row
            # is known. This keeps the next prediction causal.
            for match in matches:
                current = reliabilities.get(
                    match.observation.pattern_id,
                    ReliabilityRecord(match.observation.pattern_id),
                )
                reliabilities[match.observation.pattern_id] = update_reliability(
                    current,
                    predicted_return=prediction.expected_return,
                    actual_return=actual_return,
                    learning_rate=reliability_learning_rate,
                )

        pattern_id = f"{ticker}:{timeframe}:{position}"
        observations.append(
            PatternObservation(
                pattern_id=pattern_id,
                ticker=ticker,
                timeframe=timeframe,
                timestamp=_timestamp(data.index[position]),
                features=vector,
                next_return=actual_return,
                metadata={"source": "walk_forward"},
            )
        )
        reliabilities.setdefault(pattern_id, ReliabilityRecord(pattern_id))

    return ValidationResult(
        ticker=ticker,
        timeframe=timeframe,
        predictions=predictions,
        directional_hits=directional_hits,
        absolute_error_sum=absolute_error_sum,
        insufficient_data=insufficient_data,
        evaluated_rows=evaluated_rows,
    )
