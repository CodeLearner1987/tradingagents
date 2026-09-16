"""Deterministic feature construction for historical pattern memory."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "return_1",
    "return_3",
    "return_5",
    "range_pct",
    "body_pct",
    "close_location",
    "volume_change",
    "volatility_5",
)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time OHLCV features without using future rows.

    The returned rows are aligned to the input index. Early rows can be NaN
    because their rolling history is not yet available.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=frame.index)

    close = pd.to_numeric(frame["Close"], errors="coerce")
    open_ = pd.to_numeric(frame["Open"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")

    range_pct = (high - low) / close.replace(0, np.nan)
    body_pct = (close - open_).abs() / close.replace(0, np.nan)
    denominator = (high - low).replace(0, np.nan)
    close_location = (close - low) / denominator

    returns = close.pct_change()
    features = pd.DataFrame(
        {
            "return_1": returns,
            "return_3": close.pct_change(3),
            "return_5": close.pct_change(5),
            "range_pct": range_pct,
            "body_pct": body_pct,
            "close_location": close_location,
            "volume_change": volume.pct_change().replace([np.inf, -np.inf], np.nan),
            "volatility_5": returns.rolling(5, min_periods=5).std(),
        },
        index=frame.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


def feature_vector(features: pd.DataFrame, position: int) -> tuple[float, ...] | None:
    """Return a finite feature vector for one row, or None if incomplete."""
    values = features.iloc[position][list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return tuple(float(value) for value in values)
