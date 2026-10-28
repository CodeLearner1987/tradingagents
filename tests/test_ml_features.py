"""Feature construction for the offline ML experiments (tradingagents.ml.features).

The causality tests are the core guard against look-ahead bias: features at
row t must not change when any row after t changes.
"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.ml.features import (
    FEATURE_COLUMNS,
    WARMUP_ROWS,
    build_features,
)


def _synthetic_ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
    })


@pytest.mark.unit
def test_columns_index_and_warmup():
    feats = build_features(_synthetic_ohlcv())
    assert tuple(feats.columns) == FEATURE_COLUMNS
    assert isinstance(feats.index, pd.DatetimeIndex)
    assert feats.iloc[: WARMUP_ROWS - 1].isna().all().all()
    after = feats.iloc[WARMUP_ROWS - 1 :]
    assert after.notna().all().all()
    assert np.isfinite(after.to_numpy()).all()


@pytest.mark.unit
def test_no_raw_price_levels_in_features():
    # Scale-free: multiplying all prices by a constant leaves features unchanged.
    base = _synthetic_ohlcv()
    scaled = base.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 37.0
    pd.testing.assert_frame_equal(build_features(base), build_features(scaled), atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("cut", [WARMUP_ROWS, 250, 320, 398])
def test_changing_future_rows_never_changes_past_features(cut):
    base = _synthetic_ohlcv()
    perturbed = base.copy()
    rng = np.random.default_rng(99)
    future = perturbed.index > cut
    factor = np.exp(rng.normal(0, 0.2, future.sum()))
    for col in ("Open", "High", "Low", "Close"):
        perturbed.loc[future, col] *= factor
    perturbed.loc[future, "Volume"] *= 10

    a = build_features(base).iloc[: cut + 1]
    b = build_features(perturbed).iloc[: cut + 1]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_truncated_history_gives_identical_features():
    base = _synthetic_ohlcv()
    full = build_features(base)
    truncated = build_features(base.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], truncated)


@pytest.mark.unit
def test_unsorted_input_is_sorted_and_duplicates_rejected():
    base = _synthetic_ohlcv()
    shuffled = base.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_features(base), build_features(shuffled))

    dup = pd.concat([base, base.iloc[[10]]])
    with pytest.raises(ValueError, match="duplicate"):
        build_features(dup)


@pytest.mark.unit
def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Volume"):
        build_features(_synthetic_ohlcv().drop(columns=["Volume"]))


@pytest.mark.unit
def test_degenerate_volume_yields_nan_not_inf():
    ohlcv = _synthetic_ohlcv()
    ohlcv["Volume"] = 0.0  # e.g. an index with no reported volume
    feats = build_features(ohlcv)
    assert feats["volume_z20"].isna().all()
    assert not np.isinf(feats.to_numpy(dtype=float)).any()


@pytest.mark.unit
def test_accepts_load_ohlcv_shaped_frame():
    # load_ohlcv returns a RangeIndex frame with a datetime Date column and may
    # carry extra columns (e.g. Dividends); extras are ignored.
    ohlcv = _synthetic_ohlcv()
    ohlcv["Dividends"] = 0.0
    assert tuple(build_features(ohlcv).columns) == FEATURE_COLUMNS
