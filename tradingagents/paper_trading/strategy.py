"""SMA-crossover signals and position sizing, shared by the backtest and the live loop.

Signals are evaluated on the close of a completed daily bar ``t`` and acted on
at the next session's open, so a decision never uses the bar it trades on.

- Entry: ``SMA_fast[t] > SMA_slow[t]`` and ``SMA_fast[t-1] <= SMA_slow[t-1]``.
- Exit:  ``SMA_fast[t] < SMA_slow[t]`` while holding. The first such bar after
  an entry is exactly the cross below, and a missed session still exits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from tradingagents.paper_trading.config import StrategyConfig


def sma_signals(close: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    """Return ``sma_fast``, ``sma_slow``, ``entry`` and ``exit`` columns aligned to ``close``."""
    fast = close.rolling(cfg.fast_window, min_periods=cfg.fast_window).mean()
    slow = close.rolling(cfg.slow_window, min_periods=cfg.slow_window).mean()
    above = fast > slow
    was_at_or_below = (fast.shift(1) <= slow.shift(1)) & slow.shift(1).notna()
    return pd.DataFrame({
        "sma_fast": fast,
        "sma_slow": slow,
        "entry": (above & was_at_or_below).fillna(False).astype(bool),
        "exit": (fast < slow).fillna(False).astype(bool),
    }, index=close.index)


@dataclass(frozen=True)
class Signal:
    """Signal from the most recent completed bar."""

    bar_date: str
    close: float
    sma_fast: float
    sma_slow: float
    entry: bool
    exit: bool


def latest_signal(bars: pd.DataFrame, cfg: StrategyConfig) -> Signal:
    """Signal on the last row of ``bars`` (``Date``/``Close``, completed sessions only)."""
    if len(bars) < cfg.slow_window + 1:
        raise ValueError(
            f"need at least {cfg.slow_window + 1} completed bars, got {len(bars)}"
        )
    ordered = bars.sort_values("Date")
    sig = sma_signals(ordered["Close"].astype(float).reset_index(drop=True), cfg).iloc[-1]
    last = ordered.iloc[-1]
    return Signal(
        bar_date=pd.Timestamp(last["Date"]).strftime("%Y-%m-%d"),
        close=float(last["Close"]),
        sma_fast=float(sig["sma_fast"]),
        sma_slow=float(sig["sma_slow"]),
        entry=bool(sig["entry"]),
        exit=bool(sig["exit"]),
    )


def entry_limit_price(reference_price: float, cfg: StrategyConfig) -> float:
    return round(reference_price * (1 + cfg.entry_limit_pct), 2)


def stop_price_for(avg_entry_price: float, cfg: StrategyConfig) -> float:
    """Protective stop ``stop_loss_pct`` below the actual average entry fill."""
    return round(avg_entry_price * (1 - cfg.stop_loss_pct), 2)


def plan_quantity(equity: float, cash: float, reference_price: float, cfg: StrategyConfig) -> int:
    """Whole shares so the planned stop-out loses at most ``risk_per_trade_pct`` of equity.

    Also capped so ``qty * entry limit price <= cash``: the order can never
    borrow, whatever the account's margin buying power is.
    """
    if equity <= 0 or cash <= 0 or reference_price <= 0:
        return 0
    per_share_risk = reference_price * cfg.stop_loss_pct
    by_risk = math.floor(equity * cfg.risk_per_trade_pct / per_share_risk)
    by_cash = math.floor(cash / entry_limit_price(reference_price, cfg))
    return max(0, min(by_risk, by_cash))
