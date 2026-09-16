from datetime import datetime, timezone

import pytest

from tradingagents.pattern_memory import (
    PatternObservation,
    ReliabilityRecord,
    evaluate_prediction,
    find_matches,
    predict_from_matches,
    update_reliability,
)


@pytest.fixture
def observations() -> list[PatternObservation]:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        PatternObservation("p1", "BTC", "1h", timestamp, (1.0, 1.0), 0.10, 0.20),
        PatternObservation("p2", "BTC", "1h", timestamp, (1.2, 1.0), 0.20, 0.30),
        PatternObservation("p3", "BTC", "1h", timestamp, (5.0, 5.0), -0.50, 0.80),
    ]


def test_matcher_returns_closest_patterns(observations):
    matches = find_matches((1.0, 1.0), observations, ticker="BTC", timeframe="1h")
    assert [match.observation.pattern_id for match in matches] == ["p1", "p2", "p3"]
    assert matches[0].similarity == 1.0


def test_predictor_weights_historical_outcomes(observations):
    matches = find_matches((1.0, 1.0), observations, limit=2)
    prediction = predict_from_matches(matches)
    assert prediction.matches_used == 2
    assert prediction.expected_return > 0.10
    assert prediction.expected_range is not None
    assert prediction.evidence_ids == ("p1", "p2")


def test_reliability_update_is_bounded():
    record = ReliabilityRecord("p1")
    updated = update_reliability(record, predicted_return=0.1, actual_return=0.1)
    assert updated.score == 1.0
    assert updated.observations == 1

    updated = update_reliability(
        updated, predicted_return=0.1, actual_return=-0.9, learning_rate=1.0
    )
    assert 0.0 <= updated.score <= 1.0
    assert updated.mean_error == pytest.approx(0.5)


def test_evaluator_is_traceable():
    evaluation = evaluate_prediction("p1", predicted_return=0.1, actual_return=0.2)
    assert evaluation.pattern_id == "p1"
    assert evaluation.absolute_error == pytest.approx(0.1)
    assert evaluation.relative_error == pytest.approx(0.5)
