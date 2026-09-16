"""Deterministic historical-pattern matching and prediction primitives.

This package is prediction-only: it does not place trades or make execution
recommendations. It is designed to sit beside the existing TradingAgents
LLM workflow and provide traceable historical evidence.
"""

from tradingagents.pattern_memory.evaluator import evaluate_prediction
from tradingagents.pattern_memory.matcher import find_matches
from tradingagents.pattern_memory.models import (
    PatternEvaluation,
    PatternMatch,
    PatternObservation,
    Prediction,
    ReliabilityRecord,
)
from tradingagents.pattern_memory.predictor import predict_from_matches
from tradingagents.pattern_memory.reliability import update_reliability
from tradingagents.pattern_memory.walk_forward import ValidationResult, walk_forward_validate

__all__ = [
    "PatternEvaluation",
    "PatternMatch",
    "PatternObservation",
    "Prediction",
    "ReliabilityRecord",
    "ValidationResult",
    "evaluate_prediction",
    "find_matches",
    "predict_from_matches",
    "update_reliability",
    "walk_forward_validate",
]
