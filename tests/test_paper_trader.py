"""Paper-trading loop behaviour against a simulated Alpaca broker. No network.

Each test drives PaperTrader.run_once() through one order scenario and checks
what was submitted, what was logged, and whether entries are blocked.
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.paper_trading.broker import BrokerError, OrderRejected, UncertainOrderState
from tradingagents.paper_trading.config import StrategyConfig
from tradingagents.paper_trading.strategy import sma_signals
from tradingagents.paper_trading.trader import (
    PaperTrader,
    TraderState,
    approval_status,
    record_approval,
    request_stop,
    revoke_approval,
    run_loop,
)

CFG = StrategyConfig()
SECRETS = ("PKFAKEKEY0000001", "fake-secret-do-not-log")
ACCOUNT = "PA0001TEST"
TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}


def _bars(closes, start="2024-01-02"):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Date": pd.bdate_range(start, periods=len(closes)),
                         "Open": closes, "High": closes * 1.005, "Low": closes * 0.995,
                         "Close": closes, "Volume": 1e6})


def _bars_ending_on(kind):
    """Bars whose last row carries an entry signal, an exit signal, or neither."""
    if kind == "exit":
        closes = np.r_[np.linspace(400, 500, 80), np.linspace(500, 430, 50)]
    else:
        closes = np.r_[np.linspace(500, 420, 80), np.linspace(420, 520, 80)]
    sig = sma_signals(pd.Series(closes), CFG)
    if kind == "entry":
        idx = int(sig.index[sig["entry"]][0])
    elif kind == "exit":
        idx = int(sig.index[sig["exit"] & sig["exit"].shift(1, fill_value=False)][3])
    else:  # trending up, already above: no cross
        idx = int(sig.index[sig["entry"]][0]) + 5
        assert not sig["entry"][idx] and not sig["exit"][idx]
    return _bars(closes[: idx + 1])


def _new_signal_bar(broker, kind):
    """Shift the series one business day later so the next session sees a new bar."""
    last = broker.bars["Date"].iloc[-1]
    bars = _bars_ending_on(kind)
    bars["Date"] = pd.bdate_range(last - pd.offsets.BDay(len(bars) - 2), periods=len(bars))
    broker.bars = bars
    broker.session = (bars["Date"].iloc[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


class FakeBroker:
    secrets = SECRETS

    def __init__(self, bars, *, is_open=True, cash=100_000.0):
        self.bars = bars
        self.session = (bars["Date"].iloc[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        self.is_open = is_open
        self.cash = cash
        self.account_number = ACCOUNT
        self.orders: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        self.submitted: list[dict] = []
        self.fill_price = 500.0
        self.entry = "fill"   # fill | partial | open | reject | uncertain_found | uncertain_lost
        self.stop = "accept"  # accept | reject | unconfirmed
        self.exit = "fill"    # fill | reject
        self._ids = itertools.count(1)

    # account / clock / data
    def get_account(self):
        return {"account_number": self.account_number, "status": "ACTIVE", "cash": str(self.cash),
                "equity": str(self.cash), "trading_blocked": False, "account_blocked": False}

    def get_clock(self):
        return {"timestamp": f"{self.session}T09:31:00-04:00", "is_open": self.is_open}

    def get_daily_bars(self, symbol, start, end):
        return self.bars[self.bars["Date"] <= pd.Timestamp(end)]

    # positions
    def list_positions(self):
        return list(self.positions.values())

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def open_position(self, qty, avg=500.0, symbol="SPY"):
        self.positions[symbol] = {"symbol": symbol, "qty": str(qty), "avg_entry_price": str(avg)}

    # orders
    def _new(self, status, filled=0, **o):
        oid = f"o{next(self._ids)}"
        order = {"id": oid, "status": status, "filled_qty": str(filled),
                 "filled_avg_price": str(self.fill_price) if filled else None, **o}
        self.orders[oid] = order
        return order

    def list_open_orders(self, symbol):
        return [o for o in self.orders.values() if o["status"] not in TERMINAL]

    def get_order(self, oid):
        return self.orders.get(oid)

    def get_order_by_client_id(self, cid):
        return next((o for o in self.orders.values() if o["client_order_id"] == cid), None)

    def cancel_order(self, oid):
        if self.orders[oid]["status"] not in TERMINAL:
            self.orders[oid]["status"] = "canceled"

    def submit_order(self, **o):
        self.submitted.append(o)
        qty = int(o["qty"])
        if o["side"] == "buy":
            mode = self.entry
            if mode == "reject":
                raise OrderRejected("HTTP 403: insufficient buying power")
            if mode == "uncertain_lost":
                raise UncertainOrderState("POST /v2/orders: Timeout")
            if mode == "open":
                return self._new("new", **o)
            filled = qty // 2 if mode == "partial" else qty
            order = self._new("partially_filled" if mode == "partial" else "filled", filled, **o)
            self.open_position(filled, self.fill_price)
            if mode == "uncertain_found":
                raise UncertainOrderState("POST /v2/orders: Timeout")
            return order
        if o["type"] == "stop":
            if self.stop == "reject":
                raise OrderRejected("HTTP 422: stop price invalid")
            return self._new("suspended" if self.stop == "unconfirmed" else "new", **o)
        # market sell
        if self.exit == "reject":
            raise OrderRejected("HTTP 422: market closed")
        held = int(self.positions["SPY"]["qty"])
        reserved = sum(int(x["qty"]) for x in self.list_open_orders("SPY") if x["side"] == "sell")
        if reserved:
            raise OrderRejected("HTTP 403: insufficient qty available (held for orders)")
        order = self._new("filled", held, **o)
        self.positions.pop("SPY")
        return order

    def trigger_stop(self):
        stop = next(o for o in self.list_open_orders("SPY") if o["type"] == "stop")
        stop.update(status="filled", filled_qty=stop["qty"], filled_avg_price=stop["stop_price"])
        self.positions.pop("SPY")


@pytest.fixture
def run(tmp_path):
    def make(broker, approved=True, submit=True):
        if approved:
            record_approval(tmp_path, CFG, broker.account_number)
        return PaperTrader(broker, CFG, tmp_path, submit_orders=submit,
                           poll_interval=1.0, fill_timeout=3.0, stop_confirm_timeout=3.0,
                           sleep=lambda s: None)
    make.dir = tmp_path
    return make


def events(path):
    return [json.loads(x) for x in (path / "events.jsonl").read_text().splitlines()]


def names(path):
    return [e["event"] for e in events(path)]


# --- approval gate ------------------------------------------------------------


@pytest.mark.unit
def test_no_orders_without_recorded_approval(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    report = run(broker, approved=False).run_once()
    assert broker.submitted == []
    assert report.mode == "dry_run"
    assert "would_enter" in names(run.dir)
    assert any("no approval recorded" in w for w in report.warnings)


@pytest.mark.unit
def test_no_orders_when_not_started_with_submit_flag_even_if_approved(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    run(broker, approved=True, submit=False).run_once()
    assert broker.submitted == []


@pytest.mark.unit
def test_approval_is_bound_to_settings_and_account(tmp_path):
    record_approval(tmp_path, CFG, ACCOUNT)
    assert approval_status(tmp_path, CFG, ACCOUNT)[0]
    assert not approval_status(tmp_path, StrategyConfig(stop_loss_pct=0.03), ACCOUNT)[0]
    assert not approval_status(tmp_path, CFG, "PA9999OTHER")[0]
    assert revoke_approval(tmp_path) and not approval_status(tmp_path, CFG, ACCOUNT)[0]


# --- entries and protective stops ---------------------------------------------


@pytest.mark.unit
def test_entry_fill_then_stop_placed_and_verified(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    report = run(broker).run_once()
    buy, stop = broker.submitted
    close = broker.bars["Close"].iloc[-1]
    expected_qty = min(int(500 // (close * 0.02)), int(100_000 // round(close * 1.03, 2)))
    assert buy["side"] == "buy" and buy["type"] == "limit" and int(buy["qty"]) == expected_qty
    assert buy["limit_price"] == f"{round(close * 1.03, 2):.2f}"
    # Stop is 2% below the ACTUAL average fill (500.00), not the signal close.
    assert stop["type"] == "stop" and stop["stop_price"] == "490.00"
    assert stop["time_in_force"] == "gtc" and stop["qty"] == buy["qty"]
    ev = names(run.dir)
    assert ev.index("fill") < ev.index("stop_placed") < ev.index("stop_verified")
    assert not report.halted
    state = TraderState.load(run.dir / "state.json")
    assert state.position["qty"] == expected_qty and state.stop_order_id


@pytest.mark.unit
def test_rejected_stop_halts_and_blocks_further_entries(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.stop = "reject"
    trader = run(broker)
    report = trader.run_once()
    assert report.halted and "rejected" in report.halt_reason

    # Position is closed by hand; a new cross-above bar arrives next session.
    broker.positions.clear()
    _new_signal_bar(broker, "entry")
    before = len(broker.submitted)
    report = trader.run_once()
    buys = [o for o in broker.submitted[before:] if o["side"] == "buy"]
    assert buys == []
    assert "entry_blocked" in names(run.dir)


@pytest.mark.unit
def test_unconfirmed_stop_halts(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.stop = "unconfirmed"
    report = run(broker).run_once()
    assert report.halted and "not confirmed" in report.halt_reason


@pytest.mark.unit
def test_partial_fill_cancels_remainder_and_protects_filled_qty(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.entry = "partial"
    report = run(broker).run_once()
    buy, stop = broker.submitted
    entry_order = broker.get_order_by_client_id(buy["client_order_id"])
    assert entry_order["status"] == "canceled"
    assert int(stop["qty"]) == int(buy["qty"]) // 2
    assert "partial_fill" in names(run.dir) and not report.halted


@pytest.mark.unit
def test_unfilled_entry_is_cancelled_without_stop(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.entry = "open"
    report = run(broker).run_once()
    assert len(broker.submitted) == 1
    assert broker.orders["o1"]["status"] == "canceled"
    assert any("not filled" in a for a in report.actions) and not report.halted


@pytest.mark.unit
def test_rejected_entry_is_logged_and_not_retried(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.entry = "reject"
    trader = run(broker)
    report = trader.run_once()
    assert "order_rejected" in names(run.dir) and not report.halted
    trader.run_once()  # same session: decisions already made
    assert len(broker.submitted) == 1


@pytest.mark.unit
def test_uncertain_submit_found_by_client_id_continues_safely(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.entry = "uncertain_found"
    report = run(broker).run_once()
    assert [o["side"] for o in broker.submitted] == ["buy", "sell"]  # one buy, then the stop
    assert "order_uncertain" in names(run.dir) and "stop_verified" in names(run.dir)
    assert not report.halted


@pytest.mark.unit
def test_uncertain_submit_not_found_halts_without_resubmitting(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.entry = "uncertain_lost"
    trader = run(broker)
    report = trader.run_once()
    assert len(broker.submitted) == 1
    assert report.halted and "state unknown" in report.halt_reason
    trader.run_once()
    assert len(broker.submitted) == 1
    assert "pending_entry_not_found" in names(run.dir)


# --- existing positions, exits and stop-outs ----------------------------------


@pytest.mark.unit
def test_unprotected_position_in_dry_run_halts_and_reports(run):
    broker = FakeBroker(_bars_ending_on("none"))
    broker.open_position(40, 500.0)
    report = run(broker, approved=False).run_once()
    assert broker.submitted == []
    assert report.halted and "no confirmed protective stop" in report.halt_reason


@pytest.mark.unit
def test_existing_correct_stop_is_verified_not_replaced(run):
    broker = FakeBroker(_bars_ending_on("none"))
    broker.open_position(40, 500.0)
    broker._new("new", symbol="SPY", qty="40", side="sell", type="stop",
                stop_price="490.00", time_in_force="gtc", client_order_id="manual")
    report = run(broker).run_once()
    assert broker.submitted == [] and "stop_verified" in names(run.dir) and not report.halted


@pytest.mark.unit
def test_wrong_stop_is_replaced(run):
    broker = FakeBroker(_bars_ending_on("none"))
    broker.open_position(40, 500.0)
    broker._new("new", symbol="SPY", qty="40", side="sell", type="stop",
                stop_price="450.00", time_in_force="gtc", client_order_id="old")
    run(broker).run_once()
    assert broker.orders["o1"]["status"] == "canceled"
    assert broker.submitted[0]["stop_price"] == "490.00"


@pytest.mark.unit
def test_exit_signal_releases_stop_then_sells(run):
    broker = FakeBroker(_bars_ending_on("exit"))
    broker.open_position(40, 500.0)
    broker._new("new", symbol="SPY", qty="40", side="sell", type="stop",
                stop_price="490.00", time_in_force="gtc", client_order_id="s")
    report = run(broker).run_once()
    assert broker.orders["o1"]["status"] == "canceled"
    assert broker.submitted[-1]["type"] == "market" and broker.submitted[-1]["side"] == "sell"
    assert "SPY" not in broker.positions
    exits = [e for e in events(run.dir) if e["event"] == "exit"]
    assert exits[-1]["reason"] == "sma_cross_down" and not report.halted


@pytest.mark.unit
def test_rejected_exit_restores_stop_and_halts(run):
    broker = FakeBroker(_bars_ending_on("exit"))
    broker.open_position(40, 500.0)
    broker._new("new", symbol="SPY", qty="40", side="sell", type="stop",
                stop_price="490.00", time_in_force="gtc", client_order_id="s")
    broker.exit = "reject"
    report = run(broker).run_once()
    active_stops = [o for o in broker.list_open_orders("SPY") if o["type"] == "stop"]
    assert len(active_stops) == 1 and active_stops[0]["stop_price"] == "490.00"
    assert report.halted and "exit order rejected" in report.halt_reason


@pytest.mark.unit
def test_stop_fill_between_ticks_is_logged_as_stop_loss(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    trader = run(broker)
    trader.run_once()
    broker.trigger_stop()
    broker.session = (pd.Timestamp(broker.session) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    trader.run_once()
    exits = [e for e in events(run.dir) if e["event"] == "exit"]
    assert exits[-1]["reason"] == "stop_loss" and exits[-1]["exit_price"] == "490.00"
    assert TraderState.load(run.dir / "state.json").position is None


@pytest.mark.unit
def test_orphan_stop_without_position_is_cancelled(run):
    broker = FakeBroker(_bars_ending_on("none"))
    broker._new("new", symbol="SPY", qty="40", side="sell", type="stop",
                stop_price="490.00", time_in_force="gtc", client_order_id="s")
    run(broker).run_once()
    assert broker.orders["o1"]["status"] == "canceled"
    assert "orphan_stop_cancelled" in names(run.dir)


@pytest.mark.unit
def test_other_symbol_position_blocks_entries(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.open_position(10, 100.0, symbol="QQQ")
    report = run(broker).run_once()
    assert not [o for o in broker.submitted if o["side"] == "buy"]
    assert report.halted and "QQQ" in report.halt_reason


# --- scheduling, data freshness, logging --------------------------------------


@pytest.mark.unit
def test_market_closed_makes_no_decisions(run):
    broker = FakeBroker(_bars_ending_on("entry"), is_open=False)
    report = run(broker).run_once()
    assert broker.submitted == [] and report.signal["entry"] is True
    assert any("market closed" in a for a in report.actions)


@pytest.mark.unit
def test_stale_bars_make_no_decision(run):
    broker = FakeBroker(_bars_ending_on("entry"))
    broker.session = (pd.Timestamp(broker.session) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    report = run(broker).run_once()
    assert broker.submitted == [] and any("days old" in w for w in report.warnings)


@pytest.mark.unit
def test_secrets_never_reach_the_event_log(run):
    broker = FakeBroker(_bars_ending_on("entry"))

    def leaky():
        raise BrokerError(f"boom with {SECRETS[0]} and {SECRETS[1]}")

    broker.get_clock = leaky
    run(broker).run_once()
    text = (run.dir / "events.jsonl").read_text()
    assert SECRETS[0] not in text and SECRETS[1] not in text and "[REDACTED]" in text


@pytest.mark.unit
def test_loop_runs_until_stop_requested(run):
    broker = FakeBroker(_bars_ending_on("none"))
    trader = run(broker, approved=False)
    ticks = []

    def fake_sleep(_):
        ticks.append(1)
        if len(ticks) == 3:
            request_stop(run.dir)

    run_loop(trader, interval=2.0, sleep=fake_sleep)
    ev = names(run.dir)
    assert ev[0] == "loop_start" and ev[-1] == "loop_stop"
    assert ev.count("tick_start") == 2


@pytest.mark.unit
def test_same_signal_bar_is_never_acted_on_twice(run):
    # A lagging feed serves yesterday's cross-above bar again after a stop-out.
    broker = FakeBroker(_bars_ending_on("entry"))
    trader = run(broker)
    trader.run_once()
    broker.trigger_stop()
    broker.session = (pd.Timestamp(broker.session) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    report = trader.run_once()
    assert [o["side"] for o in broker.submitted].count("buy") == 1
    assert any("already processed" in a for a in report.actions)
