"""Tests for chronological Pattern Memory validation."""

import numpy as np
import pandas as pd

from tradingagents.pattern_memory.walk_forward import walk_forward_validate


def _frame(rows: int = 140) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = 100 + np.arange(rows, dtype=float) * 0.25
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(rows, 1000.0),
        },
        index=index,
    )


def test_walk_forward_is_deterministic() -> None:
    frame = _frame()
    first = walk_forward_validate(frame, ticker="TEST", min_history=100)
    second = walk_forward_validate(frame, ticker="TEST", min_history=100)

    assert first == second
    assert first.predictions > 0
    assert first.directional_accuracy == 1.0


def test_short_history_reports_no_predictions() -> None:
    result = walk_forward_validate(_frame(30), ticker="TEST", min_history=100)
    assert result.predictions == 0
    assert result.insufficient_rate == 0.0
