"""Daily-bar backtest of the SMA-crossover rule versus buy-and-hold.

Timing mirrors the live loop exactly:

1. Signals are computed on the close of bar ``t`` from closes ``<= t``.
2. Entries and signal exits execute at bar ``t+1``'s open with adverse
   slippage. An entry is a marketable limit capped ``entry_limit_pct`` above
   the prior close; if the slipped open is above the cap, no trade happens.
3. The protective stop sits ``stop_loss_pct`` below the actual entry fill and
   is checked against each bar's low, including the entry bar. A gap below
   the stop fills at the (worse) open. Stop fills also pay slippage.
4. A position still open at the end of a period is sold at the final close,
   with slippage and fees, so every result is net of exit costs.

Each period starts flat with the same capital. Indicator warm-up may use bars
before the period start (past data only); nothing after a bar is ever used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.paper_trading.config import StrategyConfig
from tradingagents.paper_trading.strategy import (
    entry_limit_price,
    plan_quantity,
    sma_signals,
    stop_price_for,
)

DEFAULT_SLIPPAGES: tuple[float, ...] = (0.0005, 0.0010, 0.0025)
SPLIT_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("development", 0.6), ("validation", 0.2), ("test", 0.2),
)
MIN_TRADES_FOR_CONFIDENCE = 30


@dataclass(frozen=True)
class CostModel:
    """Per-side adverse slippage plus US equity sell-side regulatory fees.

    Alpaca charges no commission on US equities; it passes through the SEC
    Section 31 fee (rate x sell notional) and the FINRA TAF (per share sold,
    capped per trade). Defaults are the published rates at the time of writing
    and are deliberately on the high side; check the current rates before
    relying on exact cost figures.
    """

    slippage_pct: float
    commission_per_share: float = 0.0
    sec_fee_rate: float = 27.80e-6
    taf_per_share: float = 0.000166
    taf_max: float = 8.30

    def buy_fees(self, qty: int, price: float) -> float:
        return qty * self.commission_per_share

    def sell_fees(self, qty: int, price: float) -> float:
        taf = min(qty * self.taf_per_share, self.taf_max)
        return qty * self.commission_per_share + qty * price * self.sec_fee_rate + taf


@dataclass(frozen=True)
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    qty: int
    exit_reason: str
    pnl: float

    @property
    def return_pct(self) -> float:
        return self.pnl / (self.entry_price * self.qty)


@dataclass
class BacktestResult:
    label: str
    start: str
    end: str
    initial_capital: float
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    exposure: float = 0.0
    skipped_entries: int = 0

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])

    @property
    def net_return(self) -> float:
        return self.final_equity / self.initial_capital - 1

    @property
    def max_drawdown(self) -> float:
        curve = self.equity
        return float((curve / curve.cummax() - 1).min())

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        return sum(t.pnl > 0 for t in self.trades) / len(self.trades)


def prepare_bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Validate daily OHLC bars and return them sorted with a RangeIndex."""
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"bars are missing columns: {missing}")
    df = ohlcv.loc[:, required].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError("bars have duplicate dates")
    df[required[1:]] = df[required[1:]].astype(float)
    if df[required[1:]].isna().any().any():
        raise ValueError("bars contain missing prices")
    return df


def _period_bounds(bars: pd.DataFrame, start, end) -> tuple[int, int]:
    dates = bars["Date"]
    si = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    ei = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if si > ei:
        raise ValueError(f"no bars between {start} and {end}")
    return si, ei


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def run_strategy(
    ohlcv: pd.DataFrame,
    cfg: StrategyConfig,
    costs: CostModel,
    start,
    end,
    initial_capital: float = 100_000.0,
    label: str = "",
) -> BacktestResult:
    bars = prepare_bars(ohlcv)
    si, ei = _period_bounds(bars, start, end)
    sig = sma_signals(bars["Close"], cfg)
    o, h, lo, c = (bars[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    s = costs.slippage_pct

    cash, qty = float(initial_capital), 0
    entry_px = stop = 0.0
    entry_cost = 0.0
    entry_day = ""
    pending_entry = pending_exit = False
    trades: list[Trade] = []
    equity = np.empty(ei - si + 1)
    held_bars = skipped = 0

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal cash, qty
        proceeds = qty * px - costs.sell_fees(qty, px)
        cash += proceeds
        trades.append(Trade(entry_day, entry_px, _day(bars["Date"][i]), px, qty,
                            reason, proceeds - entry_cost))
        qty = 0

    for k, i in enumerate(range(si, ei + 1)):
        # Open: act on decisions made at the previous close.
        if pending_exit and qty:
            close_trade(i, o[i] * (1 - s), "signal_exit")
        if pending_entry and qty == 0:
            ref = c[i - 1]
            fill = o[i] * (1 + s)
            if fill > entry_limit_price(ref, cfg):
                skipped += 1
            else:
                n = plan_quantity(cash, cash, ref, cfg)
                while n > 0 and n * fill + costs.buy_fees(n, fill) > cash:
                    n -= 1
                if n > 0:
                    qty, entry_px = n, fill
                    entry_cost = n * fill + costs.buy_fees(n, fill)
                    cash -= entry_cost
                    stop = stop_price_for(fill, cfg)
                    entry_day = _day(bars["Date"][i])
        pending_entry = pending_exit = False

        # Intraday: the protective stop is live from the entry bar onwards.
        if qty and lo[i] <= stop:
            trigger = o[i] if o[i] <= stop else stop
            close_trade(i, trigger * (1 - s), "stop_loss")

        # Close: decide for the next open.
        if qty and sig["exit"][i]:
            pending_exit = True
        elif qty == 0 and sig["entry"][i]:
            pending_entry = True

        held_bars += qty > 0
        equity[k] = cash + qty * c[i]

    if qty:
        close_trade(ei, c[ei] * (1 - s), "end_of_period")
        equity[-1] = cash

    return BacktestResult(
        label=label,
        start=_day(bars["Date"][si]),
        end=_day(bars["Date"][ei]),
        initial_capital=initial_capital,
        equity=pd.Series(equity, index=bars["Date"][si: ei + 1].to_numpy()),
        trades=trades,
        exposure=held_bars / len(equity),
        skipped_entries=skipped,
    )


def run_buy_and_hold(
    ohlcv: pd.DataFrame,
    costs: CostModel,
    start,
    end,
    initial_capital: float = 100_000.0,
    label: str = "",
) -> BacktestResult:
    """Buy whole shares at the first open of the period, sell at the final close."""
    bars = prepare_bars(ohlcv)
    si, ei = _period_bounds(bars, start, end)
    s = costs.slippage_pct
    fill = bars["Open"][si] * (1 + s)
    qty = math.floor(initial_capital / fill)
    while qty > 0 and qty * fill + costs.buy_fees(qty, fill) > initial_capital:
        qty -= 1
    cost = qty * fill + costs.buy_fees(qty, fill)
    cash = initial_capital - cost
    closes = bars["Close"][si: ei + 1].to_numpy()
    equity = cash + qty * closes
    exit_px = closes[-1] * (1 - s)
    proceeds = qty * exit_px - costs.sell_fees(qty, exit_px)
    equity[-1] = cash + proceeds
    trade = Trade(_day(bars["Date"][si]), fill, _day(bars["Date"][ei]), exit_px, qty,
                  "end_of_period", proceeds - cost)
    return BacktestResult(
        label=label,
        start=_day(bars["Date"][si]),
        end=_day(bars["Date"][ei]),
        initial_capital=initial_capital,
        equity=pd.Series(equity, index=bars["Date"][si: ei + 1].to_numpy()),
        trades=[trade],
        exposure=1.0,
    )


def chronological_splits(ohlcv: pd.DataFrame, cfg: StrategyConfig) -> list[tuple[str, str, str]]:
    """Consecutive, non-overlapping (label, start, end) periods after indicator warm-up.

    The rule has no fitted parameters, so the periods are for checking
    whether results hold up over time, not for tuning. ``full`` spans all.
    """
    bars = prepare_bars(ohlcv)
    usable = bars["Date"].iloc[cfg.slow_window:].reset_index(drop=True)
    if len(usable) < 30:
        raise ValueError("not enough bars after warm-up for chronological splits")
    periods, cursor = [], 0
    for n, (label, frac) in enumerate(SPLIT_FRACTIONS):
        stop = len(usable) if n == len(SPLIT_FRACTIONS) - 1 else cursor + int(len(usable) * frac)
        periods.append((label, _day(usable[cursor]), _day(usable[stop - 1])))
        cursor = stop
    periods.append(("full", _day(usable.iloc[0]), _day(usable.iloc[-1])))
    return periods


def run_report(
    ohlcv: pd.DataFrame,
    cfg: StrategyConfig,
    slippages: tuple[float, ...] = DEFAULT_SLIPPAGES,
    initial_capital: float = 100_000.0,
    base_costs: CostModel | None = None,
) -> pd.DataFrame:
    """One row per (period, slippage): strategy and buy-and-hold metrics side by side."""
    base = base_costs or CostModel(slippage_pct=0.0)
    rows = []
    for label, start, end in chronological_splits(ohlcv, cfg):
        for slip in slippages:
            costs = CostModel(slip, base.commission_per_share, base.sec_fee_rate,
                              base.taf_per_share, base.taf_max)
            strat = run_strategy(ohlcv, cfg, costs, start, end, initial_capital, label)
            bh = run_buy_and_hold(ohlcv, costs, start, end, initial_capital, label)
            reasons = pd.Series([t.exit_reason for t in strat.trades], dtype=object)
            rows.append({
                "period": label, "start": start, "end": end, "slippage_pct": slip * 100,
                "strategy_net_return": strat.net_return,
                "strategy_max_drawdown": strat.max_drawdown,
                "trades": strat.trade_count,
                "win_rate": strat.win_rate,
                "exposure": strat.exposure,
                "stop_exits": int((reasons == "stop_loss").sum()),
                "skipped_entries": strat.skipped_entries,
                "buy_hold_net_return": bh.net_return,
                "buy_hold_max_drawdown": bh.max_drawdown,
            })
    return pd.DataFrame(rows)


def format_report(report: pd.DataFrame, cfg: StrategyConfig, source: str) -> str:
    """Markdown report. States what the numbers do and do not show."""
    def pct(v) -> str:
        return "n/a" if pd.isna(v) else f"{v:+.2%}"

    lines = [
        f"# SMA {cfg.fast_window}/{cfg.slow_window} crossover on {cfg.symbol} vs buy-and-hold",
        "",
        f"Data: {source}. Stop {cfg.stop_loss_pct:.1%} below entry fill, risk "
        f"{cfg.risk_per_trade_pct:.2%} of equity per trade, whole shares, long only, "
        "no leverage. Slippage is adverse on every entry and exit; sell-side SEC "
        "and FINRA fees included; $0 commission.",
        "",
        "| Period | Dates | Slippage/side | Strategy net | Strategy max DD | Trades | "
        "Win rate | Exposure | Buy&hold net | Buy&hold max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report.itertuples():
        win = "n/a" if pd.isna(r.win_rate) else f"{r.win_rate:.0%}"
        lines.append(
            f"| {r.period} | {r.start} → {r.end} | {r.slippage_pct:.2f}% | "
            f"{pct(r.strategy_net_return)} | {pct(r.strategy_max_drawdown)} | {r.trades} | "
            f"{win} | {r.exposure:.0%} | {pct(r.buy_hold_net_return)} | "
            f"{pct(r.buy_hold_max_drawdown)} |"
        )

    split_rows = report[report["period"] != "full"]
    beat = int((split_rows["strategy_net_return"] > split_rows["buy_hold_net_return"]).sum())
    full = report[report["period"] == "full"]
    total_trades = int(full["trades"].max()) if not full.empty else 0
    lines += [
        "",
        "## What this shows",
        "",
        f"- The strategy beat buy-and-hold in {beat} of {len(split_rows)} "
        "period × slippage combinations (development/validation/test).",
        f"- {total_trades} round trips over the full history"
        + (f" — fewer than {MIN_TRADES_FOR_CONFIDENCE}, too few to tell skill from luck."
           if total_trades < MIN_TRADES_FOR_CONFIDENCE else "."),
        f"- The strategy commits at most about "
        f"{cfg.risk_per_trade_pct / cfg.stop_loss_pct:.0%} of equity per trade "
        f"({cfg.risk_per_trade_pct:.1%} risk / {cfg.stop_loss_pct:.0%} stop), "
        "so its returns and drawdowns are not risk-equivalent to 100%-invested "
        "buy-and-hold; compare drawdowns as well as returns.",
        "- Backtests on adjusted daily bars ignore intraday stop-fill quality, "
        "partial fills, halts and data errors. Past results do not establish "
        "future profitability.",
    ]
    return "\n".join(lines)
