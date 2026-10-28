"""Past-only, scale-free features built from daily OHLCV.

Every feature at row ``t`` uses rows ``<= t`` only. Raw price levels are
excluded on purpose: adjusted prices are rescaled after later splits and
dividends, so levels leak future corporate actions while ratios and returns
do not. Indicators reuse the stockstats definitions the market analyst sees
(RSI-14, MACD 12/26/9 histogram, Bollinger 20/2, ATR-14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from stockstats import wrap

REQUIRED_COLUMNS: tuple[str, ...] = ("Date", "Open", "High", "Low", "Close", "Volume")

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1", "ret_5", "ret_20",
    "vol_20",
    "dist_ema10", "dist_sma50", "dist_sma200",
    "rsi_14",
    "macdh_norm",
    "boll_pctb",
    "atr_norm",
    "volume_z20",
)

# Rows needed before every feature is fully warmed up (the 200-day SMA is the
# longest window). stockstats emits values from the first row, before its
# indicators have enough history, so earlier rows are blanked explicitly.
WARMUP_ROWS = 200


def prepare_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, parse dates, sort ascending, reject duplicate dates."""
    missing = [c for c in REQUIRED_COLUMNS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")
    df = ohlcv.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError("OHLCV frame has duplicate dates")
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[price_cols] = df[price_cols].astype(float)
    return df


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a ``Date``-indexed frame of :data:`FEATURE_COLUMNS`.

    The first ``WARMUP_ROWS - 1`` rows are all-NaN; later rows may still be
    NaN where an input is degenerate (e.g. constant volume). Callers drop
    incomplete rows; nothing here forward- or back-fills.
    """
    df = prepare_ohlcv(ohlcv)
    close = df["Close"]
    log_close = np.log(close)
    daily_ret = log_close.diff()

    feats = pd.DataFrame(index=df.index)
    for k in (1, 5, 20):
        feats[f"ret_{k}"] = log_close - log_close.shift(k)
    feats["vol_20"] = daily_ret.rolling(20, min_periods=20).std()

    ema10 = close.ewm(span=10, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    feats["dist_ema10"] = close / ema10 - 1
    feats["dist_sma50"] = close / sma50 - 1
    feats["dist_sma200"] = close / sma200 - 1

    stock = wrap(df.copy())
    feats["rsi_14"] = stock["rsi"].to_numpy() / 100.0
    feats["macdh_norm"] = stock["macdh"].to_numpy() / close
    upper, lower = stock["boll_ub"].to_numpy(), stock["boll_lb"].to_numpy()
    feats["boll_pctb"] = (close - lower) / (upper - lower)
    feats["atr_norm"] = stock["atr"].to_numpy() / close

    log_vol = np.log1p(df["Volume"])
    vol_mean = log_vol.rolling(20, min_periods=20).mean()
    vol_std = log_vol.rolling(20, min_periods=20).std()
    feats["volume_z20"] = (log_vol - vol_mean) / vol_std

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats.iloc[: WARMUP_ROWS - 1] = np.nan
    feats.index = pd.DatetimeIndex(df["Date"], name="Date")
    return feats.loc[:, list(FEATURE_COLUMNS)]
