"""SMA strategy, sizing and backtest (tradingagents.paper_trading.strategy/backtest)."""

import math

import numpy as np
import pandas as pd
import pytest

from tradingagents.paper_trading.backtest import (
    CostModel,
    chronological_splits,
    format_report,
    run_buy_and_hold,
    run_report,
    run_strategy,
)
from tradingagents.paper_trading.config import StrategyConfig
from tradingagents.paper_trading.strategy import (
    latest_signal,
    plan_quantity,
    sma_signals,
    stop_price_for,
)

CFG = StrategyConfig()


def synthetic_bars(n=900, seed=3):
    """Trending regimes so the SMA rule actually trades, plus gaps and wide ranges."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.002, 0.0015], size=n // 60 + 1), 60)[:n]
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.011, n)))
    open_ = close * np.exp(rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, n))
    return pd.DataFrame({"Date": pd.bdate_range("2015-01-02", periods=n),
                         "Open": open_, "High": high, "Low": low, "Close": close})


# --- signals and sizing -------------------------------------------------------


@pytest.mark.unit
def test_entry_is_cross_above_and_exit_is_fast_below_slow():
    close = pd.Series(np.r_[np.linspace(100, 80, 60), np.linspace(80, 110, 60)])
    sig = sma_signals(close, CFG)
    fast = close.rolling(20).mean()
    slow = close.rolling(50).mean()
    entries = sig.index[sig["entry"]].tolist()
    assert len(entries) == 1
    e = entries[0]
    assert fast[e] > slow[e] and fast[e - 1] <= slow[e - 1]
    assert (sig["exit"] == (fast < slow)).all()
    assert not sig["entry"].iloc[:50].any()  # no signal before SMA50 exists


@pytest.mark.unit
def test_latest_signal_needs_enough_bars_and_uses_last_row():
    bars = synthetic_bars(120)
    with pytest.raises(ValueError):
        latest_signal(bars.iloc[:50], CFG)
    sig = latest_signal(bars.sample(frac=1, random_state=1), CFG)  # order-insensitive
    assert sig.bar_date == bars["Date"].iloc[-1].strftime("%Y-%m-%d")
    assert sig.close == pytest.approx(bars["Close"].iloc[-1])


@pytest.mark.unit
def test_plan_quantity_risk_and_cash_caps():
    # Risk cap: 0.5% of 100k = 500; 2% of 500 = 10/share -> 50 shares.
    assert plan_quantity(100_000, 100_000, 500.0, CFG) == 50
    # Cash cap: 10k cash / (500 * 1.03) -> 19 shares.
    assert plan_quantity(100_000, 10_000, 500.0, CFG) == 19
    assert plan_quantity(1_000, 1_000, 500.0, CFG) == 0
    assert plan_quantity(100_000, 0, 500.0, CFG) == 0
    assert stop_price_for(512.37, CFG) == round(512.37 * 0.98, 2)


@pytest.mark.unit
def test_config_rejects_unapproved_settings():
    for bad in ({"symbol": "QQQ"}, {"max_open_positions": 2}, {"fast_window": 50},
                {"stop_loss_pct": 0}, {"risk_per_trade_pct": 0.05}):
        with pytest.raises(ValueError):
            StrategyConfig(**bad)
    assert StrategyConfig().fingerprint() != StrategyConfig(stop_loss_pct=0.03).fingerprint()


# --- backtest mechanics -------------------------------------------------------


def _strategy(bars, slip=0.001, **kw):
    dates = bars["Date"]
    return run_strategy(bars, CFG, CostModel(slip, **kw), dates.iloc[50], dates.iloc[-1])


@pytest.mark.unit
def test_backtest_trades_and_fills_follow_the_rules():
    bars = synthetic_bars()
    res = _strategy(bars, slip=0.001)
    assert res.trade_count >= 5
    by_date = bars.set_index(bars["Date"].dt.strftime("%Y-%m-%d"))
    sig = sma_signals(bars["Close"], CFG)
    sig.index = by_date.index
    for t in res.trades:
        o = by_date.loc[t.entry_date, "Open"]
        assert t.entry_price == pytest.approx(o * 1.001)
        prev = by_date.index[by_date.index.get_loc(t.entry_date) - 1]
        assert sig.loc[prev, "entry"]  # entered the open after a cross-above close
        stop = stop_price_for(t.entry_price, CFG)
        bar = by_date.loc[t.exit_date]
        if t.exit_reason == "stop_loss":
            assert bar["Low"] <= stop
            assert t.exit_price == pytest.approx(min(bar["Open"], stop) * 0.999)
        elif t.exit_reason == "signal_exit":
            assert t.exit_price == pytest.approx(bar["Open"] * 0.999)
            prev_exit = by_date.index[by_date.index.get_loc(t.exit_date) - 1]
            assert sig.loc[prev_exit, "exit"]
        # Planned risk never exceeds 0.5% of the capital available at entry.
        assert t.qty * t.entry_price * CFG.stop_loss_pct <= 0.005 * 100_000 * 1.5


@pytest.mark.unit
def test_no_future_leakage_perturbing_later_bars_keeps_earlier_results():
    bars = synthetic_bars()
    cut = 600
    perturbed = bars.copy()
    rng = np.random.default_rng(7)
    factor = np.exp(rng.normal(0, 0.15, len(bars) - cut - 1))
    for col in ("Open", "High", "Low", "Close"):
        perturbed.loc[cut + 1:, col] *= factor
    a, b = _strategy(bars), _strategy(perturbed)
    cut_date = bars["Date"].iloc[cut]
    pd.testing.assert_series_equal(a.equity[a.equity.index <= cut_date],
                                   b.equity[b.equity.index <= cut_date])
    def closed_by_cut(r):
        return [t for t in r.trades if pd.Timestamp(t.exit_date) <= cut_date]

    assert closed_by_cut(a) == closed_by_cut(b)


@pytest.mark.unit
def test_gap_below_stop_fills_at_open_not_stop():
    n = 80
    close = np.r_[np.linspace(100, 90, 55), np.linspace(90, 100, n - 55)]
    bars = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=n),
                         "Open": close, "High": close * 1.001, "Low": close * 0.999,
                         "Close": close})
    sig = sma_signals(bars["Close"], CFG)
    e = int(sig.index[sig["entry"]][0])
    # Next bar: enter at its open; the bar after gaps 10% below.
    bars.loc[e + 2, ["Open", "High", "Low", "Close"]] = bars.loc[e + 1, "Open"] * np.array(
        [0.90, 0.91, 0.89, 0.905])
    res = run_strategy(bars, CFG, CostModel(0.0), bars["Date"][50], bars["Date"].iloc[-1])
    stop_trade = next(t for t in res.trades if t.exit_reason == "stop_loss")
    assert stop_trade.exit_price == pytest.approx(bars.loc[e + 2, "Open"])
    assert stop_trade.exit_price < stop_price_for(stop_trade.entry_price, CFG)


@pytest.mark.unit
def test_entry_skipped_when_open_gaps_above_limit():
    n = 80
    close = np.r_[np.linspace(100, 90, 55), np.linspace(90, 100, n - 55)]
    bars = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=n),
                         "Open": close, "High": close, "Low": close, "Close": close})
    sig = sma_signals(bars["Close"], CFG)
    e = int(sig.index[sig["entry"]][0])
    bars.loc[e + 1, "Open"] = bars.loc[e, "Close"] * 1.05  # above the 3% cap
    res = run_strategy(bars, CFG, CostModel(0.0), bars["Date"][50], bars["Date"][e + 1])
    assert res.skipped_entries == 1 and res.trade_count == 0


@pytest.mark.unit
def test_more_slippage_never_helps_and_fees_apply_on_sells():
    bars = synthetic_bars()
    returns = [_strategy(bars, s).net_return for s in (0.0005, 0.001, 0.0025)]
    assert returns[0] > returns[1] > returns[2]
    costs = CostModel(0.0)
    assert costs.buy_fees(100, 500.0) == 0.0
    assert costs.sell_fees(100, 500.0) == pytest.approx(100 * 500 * 27.80e-6 + 100 * 0.000166)
    assert costs.sell_fees(10**6, 1.0) == pytest.approx(10**6 * 27.80e-6 + 8.30)  # TAF cap


@pytest.mark.unit
def test_buy_and_hold_by_hand():
    bars = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=3),
                         "Open": [100.0, 101, 102], "High": [101.0, 102, 111],
                         "Low": [99.0, 100, 101], "Close": [100.0, 102, 110]})
    costs = CostModel(0.001, sec_fee_rate=0.0, taf_per_share=0.0)
    res = run_buy_and_hold(bars, costs, "2020-01-01", "2020-01-03", 10_000)
    qty = math.floor(10_000 / 100.1)
    cash = 10_000 - qty * 100.1
    assert res.final_equity == pytest.approx(cash + qty * 110 * 0.999)
    assert res.max_drawdown <= 0


@pytest.mark.unit
def test_splits_are_chronological_contiguous_and_after_warmup():
    bars = synthetic_bars()
    periods = chronological_splits(bars, CFG)
    assert [p[0] for p in periods] == ["development", "validation", "test", "full"]
    dates = list(bars["Date"].dt.strftime("%Y-%m-%d"))
    assert periods[0][1] == dates[CFG.slow_window]
    for (_, _, end), (_, start, _) in zip(periods[:2], periods[1:3], strict=True):
        assert dates.index(start) == dates.index(end) + 1
    assert periods[2][2] == dates[-1] == periods[3][2]


@pytest.mark.unit
def test_report_has_every_period_and_slippage_and_no_profit_claim():
    bars = synthetic_bars()
    report = run_report(bars, CFG)
    assert len(report) == 4 * 3
    assert sorted(report["slippage_pct"].unique()) == pytest.approx([0.05, 0.10, 0.25])
    for col in ("strategy_net_return", "strategy_max_drawdown", "trades", "win_rate",
                "buy_hold_net_return"):
        assert col in report.columns
    text = format_report(report, CFG, "synthetic")
    assert "too few to tell skill from luck" in text or "round trips" in text
    assert "do not establish future profitability" in text
    assert "profitable strategy" not in text.lower()
