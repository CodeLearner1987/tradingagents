"""Target construction for the offline ML experiments (tradingagents.ml.targets)."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.ml.targets import (
    build_targets,
    direction_label,
    forward_log_return,
    near_zero_mask,
)


def _close(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


@pytest.mark.unit
def test_forward_log_return_matches_formula_and_leaves_tail_missing():
    close = _close([100, 110, 99, 99, 120])
    fwd = forward_log_return(close, 2)
    assert fwd.name == "fwd_ret_2"
    np.testing.assert_allclose(fwd.iloc[:3], np.log([99 / 100, 99 / 110, 120 / 99]))
    assert fwd.iloc[-2:].isna().all()


@pytest.mark.unit
def test_direction_label_zero_is_not_up_and_missing_stays_missing():
    close = _close([100, 101, 101, 100, 100])
    label = direction_label(forward_log_return(close, 1))
    assert label.name == "y_1"
    assert str(label.dtype) == "Int8"
    # up, flat (exactly zero -> 0), down, flat, unknown
    assert label.iloc[:4].tolist() == [1, 0, 0, 0]
    assert label.isna().iloc[-1]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "5"])
def test_horizon_must_be_positive_int(bad):
    with pytest.raises(ValueError):
        forward_log_return(_close([1, 2, 3]), bad)


@pytest.mark.unit
def test_near_zero_mask_uses_trailing_volatility():
    rng = np.random.default_rng(1)
    close = _close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))))
    fwd = forward_log_return(close, 1)
    mask = near_zero_mask(fwd, close, 1, k=0.5, vol_window=20)

    sigma = np.log(close).diff().rolling(20, min_periods=20).std()
    expected = (fwd.abs() <= 0.5 * sigma).fillna(False)
    pd.testing.assert_series_equal(mask, expected.rename("near_zero_1"))
    # No volatility estimate yet (warm-up) or no future return -> never flagged.
    assert not mask.iloc[:20].any()
    assert not mask.iloc[-1]


@pytest.mark.unit
def test_build_targets_sorts_by_date_and_names_columns():
    dates = pd.bdate_range("2024-01-01", periods=4)
    ohlcv = pd.DataFrame({
        "Date": dates[::-1],  # deliberately unsorted
        "Open": 1.0, "High": 1.0, "Low": 1.0,
        "Close": [103.0, 102.0, 101.0, 100.0],
        "Volume": 1.0,
    })
    t = build_targets(ohlcv, horizons=(1,))
    assert list(t.columns) == ["fwd_ret_1", "y_1", "near_zero_1"]
    assert t.index.is_monotonic_increasing
    assert t["y_1"].iloc[:3].tolist() == [1, 1, 1]
