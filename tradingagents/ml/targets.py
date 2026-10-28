"""Prediction targets: forward h-day log return and its up/down label.

Conventions (shared by every horizon):

- ``r(t, h) = ln(Close[t+h] / Close[t])`` where ``t+h`` counts *trading rows*,
  not calendar days. ``Close`` is the split/dividend-adjusted close that
  ``load_ohlcv`` returns (yfinance ``auto_adjust=True``).
- ``y(t, h) = 1`` if ``r(t, h) > 0`` else ``0``. An exactly-zero return is
  "not up" (class 0); its frequency is reported by the evaluation stage.
- The last ``h`` rows have no future close, so their return and label are
  missing (NaN / <NA>) and must be dropped, never filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.ml.features import prepare_ohlcv


def _check_horizon(horizon: int) -> None:
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return ``ln(close[t+horizon] / close[t])``; NaN for the last ``horizon`` rows."""
    _check_horizon(horizon)
    log_close = np.log(close.astype(float))
    return (log_close.shift(-horizon) - log_close).rename(f"fwd_ret_{horizon}")


def direction_label(forward_return: pd.Series) -> pd.Series:
    """Map forward returns to 1 (strictly positive) / 0 (zero or negative).

    Missing returns stay missing (nullable ``Int8``) so callers cannot
    accidentally train on rows whose outcome is unknown.
    """
    label = pd.Series(pd.NA, index=forward_return.index, dtype="Int8")
    known = forward_return.notna()
    label[known] = (forward_return[known] > 0).astype("int8")
    return label.rename(str(forward_return.name).replace("fwd_ret", "y"))


def near_zero_mask(
    forward_return: pd.Series,
    close: pd.Series,
    horizon: int,
    k: float = 0.1,
    vol_window: int = 20,
) -> pd.Series:
    """Flag rows whose move is too small to call: ``|r| <= k * sigma_t * sqrt(h)``.

    ``sigma_t`` is the trailing ``vol_window``-day std of daily log returns
    known at ``t``. Intended for the dead-zone *training* variant only;
    evaluation still scores every labelled row. Rows with unknown return or
    volatility are not flagged.
    """
    _check_horizon(horizon)
    daily = np.log(close.astype(float)).diff()
    sigma = daily.rolling(vol_window, min_periods=vol_window).std()
    threshold = k * sigma * np.sqrt(horizon)
    mask = forward_return.abs() <= threshold
    return mask.fillna(False).astype(bool).rename(f"near_zero_{horizon}")


def build_targets(ohlcv: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    """Return a ``Date``-indexed frame with ``fwd_ret_h``, ``y_h`` and ``near_zero_h``.

    Uses the same validation and date ordering as
    :func:`tradingagents.ml.features.build_features`, so the two frames align
    row-for-row on their index.
    """
    df = prepare_ohlcv(ohlcv)
    close = pd.Series(df["Close"].to_numpy(), index=pd.DatetimeIndex(df["Date"], name="Date"))
    columns = {}
    for h in horizons:
        fwd = forward_log_return(close, h)
        columns[fwd.name] = fwd
        label = direction_label(fwd)
        columns[label.name] = label
        mask = near_zero_mask(fwd, close, h)
        columns[mask.name] = mask
    return pd.DataFrame(columns, index=close.index)
