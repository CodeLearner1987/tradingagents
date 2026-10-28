"""The paper-trading loop: reconcile, protect, then act on the daily signal.

Each tick (``PaperTrader.run_once``) does, in order:

1. Check the account is an active Alpaca *paper* account.
2. Resolve any entry order left pending by an earlier tick.
3. Reconcile positions: block entries on anything other than one long SPY
   position, and record why a tracked position disappeared (stop or other).
4. **Protect**: an open position must have exactly one active GTC sell stop
   for its full quantity at ``stop_loss_pct`` below the average entry fill.
   If that cannot be confirmed, the trader halts: no new entries until a
   human runs ``resume``.
5. Once per session, while the market is open, evaluate the signal on the
   last completed daily bar and enter or exit.

Order submission happens only when the trader was started with
``submit_orders=True`` **and** a human-recorded approval for this exact
strategy configuration and paper account exists. Otherwise every decision is
logged as ``would_enter`` / ``would_exit`` and nothing is sent.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from tradingagents.paper_trading.broker import (
    AlpacaPaperClient,
    BrokerError,
    OrderRejected,
    UncertainOrderState,
    mask,
)
from tradingagents.paper_trading.config import PAPER_TRADING_URL, StrategyConfig
from tradingagents.paper_trading.strategy import (
    entry_limit_price,
    latest_signal,
    plan_quantity,
    stop_price_for,
)

logger = logging.getLogger(__name__)

APPROVAL_PHRASE = "APPROVE PAPER ORDERS"
RESUME_PHRASE = "RESUME"

EVENTS_FILE = "events.jsonl"
STATE_FILE = "state.json"
APPROVAL_FILE = "approval.json"
STOP_FILE = "STOP"

TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
ACTIVE_STOP_STATUSES = {
    "new", "accepted", "pending_new", "partially_filled", "accepted_for_bidding", "held",
}
# Bars older than this (calendar days before the session) mean the data feed
# is stale; no decision is taken on them.
MAX_SIGNAL_AGE_DAYS = 5
BARS_LOOKBACK_DAYS = 200


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _account_digest(account_number: str) -> str:
    return hashlib.sha256(account_number.encode()).hexdigest()[:16]


# --- event log ------------------------------------------------------------------


class EventLog:
    """Append-only JSON-lines log. Secret values are scrubbed from every line."""

    def __init__(self, path: Path, secrets: tuple[str, ...] = ()):
        self.path = path
        self._secrets = tuple(s for s in secrets if s)
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields) -> dict:
        record = {"ts": _utcnow(), "event": event, **fields}
        line = json.dumps(record, default=str, sort_keys=False)
        for secret in self._secrets:
            line = line.replace(secret, "[REDACTED]")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(line)
        return json.loads(line)

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]


# --- persistent state -----------------------------------------------------------


@dataclass
class TraderState:
    halted: bool = False
    halt_reason: str = ""
    last_decision_session: str | None = None
    # Date of the daily bar whose signal was last acted on. A lagging data
    # feed can serve the same bar on two sessions; it is never acted on twice.
    last_signal_bar: str | None = None
    pending_entry: dict | None = None
    position: dict | None = None
    stop_order_id: str | None = None
    stop_seq: int = 0

    @classmethod
    def load(cls, path: Path) -> TraderState:
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)


# --- human approval -------------------------------------------------------------


def record_approval(directory: Path, cfg: StrategyConfig, account_number: str) -> dict:
    """Write the approval file. Callers must have obtained a typed human confirmation."""
    record = {
        "approved_at": _utcnow(),
        "approved_by": getpass.getuser(),
        "endpoint": PAPER_TRADING_URL,
        "config": asdict(cfg),
        "config_fingerprint": cfg.fingerprint(),
        "account_digest": _account_digest(account_number),
        "account_hint": mask(account_number),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / APPROVAL_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def revoke_approval(directory: Path) -> bool:
    path = directory / APPROVAL_FILE
    if path.exists():
        path.unlink()
        return True
    return False


def approval_status(
    directory: Path, cfg: StrategyConfig, account_number: str | None = None
) -> tuple[bool, str]:
    path = directory / APPROVAL_FILE
    if not path.exists():
        return False, "no approval recorded (run the 'approve' command)"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "approval file unreadable"
    if record.get("endpoint") != PAPER_TRADING_URL:
        return False, "approval is not for the paper endpoint"
    if record.get("config_fingerprint") != cfg.fingerprint():
        return False, "strategy settings changed since approval; approve again"
    if account_number is not None and record.get("account_digest") != _account_digest(account_number):
        return False, "approval was recorded for a different paper account"
    return True, f"approved by {record.get('approved_by')} at {record.get('approved_at')}"


def request_stop(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STOP_FILE).write_text(_utcnow(), encoding="utf-8")


def stop_requested(directory: Path) -> bool:
    return (directory / STOP_FILE).exists()


# --- the trader -----------------------------------------------------------------


@dataclass
class TickReport:
    session: str = ""
    mode: str = "dry_run"
    signal: dict | None = None
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""


def _order_type(order: dict) -> str:
    return order.get("type") or order.get("order_type") or ""


def _remaining_qty(order: dict) -> int:
    return int(float(order.get("qty") or 0) - float(order.get("filled_qty") or 0))


class PaperTrader:
    def __init__(
        self,
        broker: AlpacaPaperClient,
        cfg: StrategyConfig,
        directory: Path,
        *,
        submit_orders: bool = False,
        poll_interval: float = 2.0,
        fill_timeout: float = 90.0,
        stop_confirm_timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.broker = broker
        self.cfg = cfg
        self.dir = Path(directory)
        self.submit_orders = submit_orders
        self.poll_interval = poll_interval
        self.fill_timeout = fill_timeout
        self.stop_confirm_timeout = stop_confirm_timeout
        self._sleep = sleep
        self.log = EventLog(self.dir / EVENTS_FILE, secrets=getattr(broker, "secrets", ()))
        self.state = TraderState.load(self.dir / STATE_FILE)
        self._report = TickReport()
        self._can_submit = False
        self._order_activity = False

    # --- public ------------------------------------------------------------

    def run_once(self) -> TickReport:
        self._report = report = TickReport()
        self._order_activity = False
        try:
            account = self.broker.get_account()
            clock = self.broker.get_clock()
            session = pd.Timestamp(clock["timestamp"]).tz_convert("America/New_York")
            report.session = session.strftime("%Y-%m-%d")
            self._can_submit = self._submission_allowed(account)
            report.mode = "submit" if self._can_submit else "dry_run"
            self.log.write("tick_start", session=report.session, mode=report.mode,
                           market_open=bool(clock.get("is_open")))

            self._resolve_pending_entry()
            position = self._reconcile_positions()
            if position is not None:
                self._ensure_stop(position)
            else:
                self._cancel_orphan_stops()

            if not clock.get("is_open"):
                self._preview_signal(session)
                report.actions.append("market closed: no entry/exit decisions this tick")
            elif self.state.last_decision_session == report.session:
                report.actions.append("decisions already made for this session")
            else:
                self._decide(session, account, position)
        except BrokerError as exc:
            self.log.write("error", kind=type(exc).__name__, message=str(exc))
            report.warnings.append(f"broker error: {exc}")
            if self._order_activity:
                self._halt(f"broker error during order handling: {exc}")
        except Exception as exc:  # noqa: BLE001 — any surprise must halt, not crash silently
            self.log.write("error", kind=type(exc).__name__, message=str(exc))
            report.warnings.append(f"unexpected error: {type(exc).__name__}: {exc}")
            self._halt(f"unexpected error: {type(exc).__name__}")
        finally:
            self.state.save(self.dir / STATE_FILE)
            report.halted, report.halt_reason = self.state.halted, self.state.halt_reason
            self.log.write("tick_end", session=report.session, halted=report.halted,
                           actions=report.actions, warnings=report.warnings)
        return report

    def resume(self) -> None:
        """Clear a halt. Only call after a human has reviewed the reason."""
        self.log.write("resume", previous_reason=self.state.halt_reason)
        self.state.halted, self.state.halt_reason = False, ""
        self.state.save(self.dir / STATE_FILE)

    # --- gates -------------------------------------------------------------

    def _submission_allowed(self, account: dict) -> bool:
        problems = []
        if str(account.get("status", "")).upper() != "ACTIVE":
            problems.append(f"account status {account.get('status')}")
        if account.get("trading_blocked") or account.get("account_blocked"):
            problems.append("account is blocked from trading")
        if float(account.get("cash") or 0) < 0:
            problems.append("negative cash (margin in use)")
        for p in problems:
            self._report.warnings.append(p)
            self.log.write("account_problem", detail=p)
        if not self.submit_orders:
            return False
        ok, why = approval_status(self.dir, self.cfg, str(account.get("account_number", "")))
        if not ok:
            self._report.warnings.append(f"order submission disabled: {why}")
            self.log.write("submission_disabled", reason=why)
            return False
        if stop_requested(self.dir):
            self._report.warnings.append("order submission disabled: stop requested")
            return False
        return not problems

    def _halt(self, reason: str) -> None:
        if not self.state.halted:
            self.state.halted, self.state.halt_reason = True, reason
        self._report.warnings.append(f"HALTED: {reason}")
        self.log.write("halt", reason=reason)
        self.state.save(self.dir / STATE_FILE)

    # --- order helpers -----------------------------------------------------

    def _cid(self, session: str, kind: str) -> str:
        return f"sma-{self.cfg.symbol}-{session.replace('-', '')}-{kind}"

    def _wait_terminal(self, order_id: str, timeout: float) -> dict | None:
        polls = max(1, int(timeout / self.poll_interval))
        order = None
        for n in range(polls):
            order = self.broker.get_order(order_id)
            if order and order.get("status") in TERMINAL_STATUSES:
                return order
            if n < polls - 1:
                self._sleep(self.poll_interval)
        return order

    def _cancel_and_wait(self, order: dict) -> dict | None:
        self._order_activity = True
        self.broker.cancel_order(order["id"])
        self.log.write("order_cancel_requested", order_id=order["id"],
                       client_order_id=order.get("client_order_id"))
        return self._wait_terminal(order["id"], self.stop_confirm_timeout)

    def _submit(self, kind: str, **order) -> dict | None:
        """Submit and return the order, ``None`` if rejected.

        On an uncertain outcome the order is looked up by its client id; if it
        is still unknown, :class:`UncertainOrderState` propagates.
        """
        self._order_activity = True
        cid = order["client_order_id"]
        self.log.write("order_submit", kind=kind, **order)
        try:
            placed = self.broker.submit_order(**order)
        except OrderRejected as exc:
            self.log.write("order_rejected", kind=kind, client_order_id=cid, reason=str(exc))
            self._report.actions.append(f"{kind} order rejected: {exc}")
            return None
        except UncertainOrderState as exc:
            self.log.write("order_uncertain", kind=kind, client_order_id=cid, reason=str(exc))
            placed = self.broker.get_order_by_client_id(cid)
            if placed is None:
                raise UncertainOrderState(
                    f"{kind} order {cid} state unknown after {exc}; not resubmitting"
                ) from None
        self.log.write("order_accepted", kind=kind, order_id=placed.get("id"),
                       client_order_id=cid, status=placed.get("status"))
        return placed

    # --- reconcile ---------------------------------------------------------

    def _resolve_pending_entry(self) -> None:
        pending = self.state.pending_entry
        if not pending:
            return
        order = self.broker.get_order_by_client_id(pending["client_order_id"])
        if order is None:
            self.log.write("pending_entry_not_found", **pending)
            self.state.pending_entry = None
            return
        if order.get("status") not in TERMINAL_STATUSES:
            if not self._can_submit:
                self._report.warnings.append("an entry order is still open; entries blocked")
                return
            order = self._cancel_and_wait(order) or order
        self.log.write("pending_entry_resolved", client_order_id=pending["client_order_id"],
                       status=order.get("status"), filled_qty=order.get("filled_qty"),
                       filled_avg_price=order.get("filled_avg_price"))
        if order.get("status") in TERMINAL_STATUSES:
            self.state.pending_entry = None

    def _reconcile_positions(self) -> dict | None:
        positions = self.broker.list_positions()
        others = sorted(p["symbol"] for p in positions if p["symbol"] != self.cfg.symbol)
        if others:
            self._halt(f"unexpected open positions {others}; only {self.cfg.symbol} is approved")
        pos = next((p for p in positions if p["symbol"] == self.cfg.symbol), None)
        if pos is not None and float(pos["qty"]) <= 0:
            self._halt(f"{self.cfg.symbol} position is short ({pos['qty']}); not managed")
            return None

        tracked = self.state.position
        if tracked and pos is None:
            self._record_closed_position(tracked)
        elif pos is not None:
            current = {"qty": int(float(pos["qty"])), "avg_entry_price": float(pos["avg_entry_price"])}
            if not tracked:
                self.log.write("position_adopted", symbol=self.cfg.symbol, **current)
            elif (tracked["qty"], tracked["avg_entry_price"]) != (current["qty"], current["avg_entry_price"]):
                self.log.write("position_changed", previous=tracked, current=current)
            self.state.position = {**(tracked or {}), **current}
        return pos

    def _record_closed_position(self, tracked: dict) -> None:
        reason, price = "closed_outside_workflow", None
        if self.state.stop_order_id:
            stop = self.broker.get_order(self.state.stop_order_id)
            if stop and stop.get("status") == "filled":
                reason, price = "stop_loss", stop.get("filled_avg_price")
        self.log.write("exit", reason=reason, symbol=self.cfg.symbol, qty=tracked.get("qty"),
                       avg_entry_price=tracked.get("avg_entry_price"), exit_price=price)
        self._report.actions.append(f"position closed ({reason})")
        self.state.position, self.state.stop_order_id = None, None

    # --- protection --------------------------------------------------------

    def _stop_orders(self) -> list[dict]:
        return [
            o for o in self.broker.list_open_orders(self.cfg.symbol)
            if o.get("side") == "sell" and _order_type(o) in ("stop", "stop_limit")
        ]

    def _ensure_stop(self, position: dict) -> bool:
        qty = int(float(position["qty"]))
        target = stop_price_for(float(position["avg_entry_price"]), self.cfg)
        stops = self._stop_orders()
        good = [
            o for o in stops
            if o.get("status") in ACTIVE_STOP_STATUSES
            and _remaining_qty(o) == qty
            and abs(float(o.get("stop_price") or 0) - target) < 0.006
        ]
        if len(good) == 1 and len(stops) == 1:
            self.state.stop_order_id = good[0]["id"]
            self.log.write("stop_verified", order_id=good[0]["id"], stop_price=target, qty=qty,
                           status=good[0].get("status"))
            return True

        if not self._can_submit:
            self._halt(
                f"{self.cfg.symbol} position ({qty} sh) has no confirmed protective stop at "
                f"{target:.2f} and order submission is not enabled"
            )
            return False

        for o in stops:
            final = self._cancel_and_wait(o)
            if final and final.get("status") == "filled":
                self.log.write("stop_filled_during_replacement", order_id=o["id"])
                self._report.actions.append("stop filled while being replaced; re-check next tick")
                return True
            if not final or final.get("status") not in TERMINAL_STATUSES:
                self._halt(f"could not cancel mismatched stop {o['id']}")
                return False

        self.state.stop_seq += 1
        session = self._report.session or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        placed = self._submit(
            "stop",
            symbol=self.cfg.symbol, qty=str(qty), side="sell", type="stop",
            stop_price=f"{target:.2f}", time_in_force="gtc",
            client_order_id=self._cid(session, f"stop{self.state.stop_seq}"),
        )
        if placed is None:
            self._halt(f"protective stop at {target:.2f} was rejected")
            return False

        status, order = placed.get("status"), placed
        for _ in range(max(1, int(self.stop_confirm_timeout / self.poll_interval))):
            if status in ACTIVE_STOP_STATUSES or status in TERMINAL_STATUSES:
                break
            self._sleep(self.poll_interval)
            order = self.broker.get_order(placed["id"]) or order
            status = order.get("status")

        if status in ACTIVE_STOP_STATUSES:
            self.state.stop_order_id = placed["id"]
            self.log.write("stop_placed", order_id=placed["id"], stop_price=target, qty=qty,
                           status=status)
            self.log.write("stop_verified", order_id=placed["id"], stop_price=target, qty=qty,
                           status=status)
            self._report.actions.append(f"protective stop active: {qty} sh @ {target:.2f}")
            return True
        if status == "filled":
            self.state.stop_order_id = placed["id"]
            self.log.write("stop_filled_immediately", order_id=placed["id"])
            return True
        self._halt(f"protective stop not confirmed (status {status!r})")
        return False

    def _cancel_orphan_stops(self) -> None:
        stops = self._stop_orders()
        if not stops:
            return
        if not self._can_submit:
            self._halt(f"{len(stops)} sell stop(s) open with no position; could open a short")
            return
        for o in stops:
            self._cancel_and_wait(o)
            self.log.write("orphan_stop_cancelled", order_id=o["id"])

    # --- decisions ---------------------------------------------------------

    def _load_signal(self, session: pd.Timestamp):
        day = session.normalize().tz_localize(None)
        start = (day - timedelta(days=BARS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = (day - timedelta(days=1)).strftime("%Y-%m-%d")
        bars = self.broker.get_daily_bars(self.cfg.symbol, start, end)
        bars = bars[pd.to_datetime(bars["Date"]) < day]
        sig = latest_signal(bars, self.cfg)
        age = (day - pd.Timestamp(sig.bar_date)).days
        return sig, age

    def _preview_signal(self, session: pd.Timestamp) -> None:
        try:
            sig, _ = self._load_signal(session)
        except (BrokerError, ValueError) as exc:
            self._report.warnings.append(f"signal preview unavailable: {exc}")
            return
        self._report.signal = asdict(sig)

    def _decide(self, session: pd.Timestamp, account: dict, position: dict | None) -> None:
        sig, age = self._load_signal(session)
        self._report.signal = asdict(sig)
        self.log.write("signal", **asdict(sig), bar_age_days=age)
        if age > MAX_SIGNAL_AGE_DAYS:
            self._report.warnings.append(f"latest bar is {age} days old; no decision taken")
            return

        session_str = self._report.session
        if sig.bar_date == self.state.last_signal_bar:
            self._report.actions.append(f"signal bar {sig.bar_date} already processed")
            if self._can_submit:
                self.state.last_decision_session = session_str
            return
        if position is not None and sig.exit:
            self._exit(position, session_str, "sma_cross_down")
        elif position is None and sig.entry:
            if self.state.halted:
                self.log.write("entry_blocked", reason=self.state.halt_reason)
                self._report.actions.append("entry signal ignored: trader is halted")
            elif self.state.pending_entry:
                self.log.write("entry_blocked", reason="an earlier entry is unresolved")
                self._report.actions.append("entry signal ignored: earlier entry unresolved")
            else:
                self._enter(sig, session_str, account)
        else:
            self._report.actions.append("no signal")

        if self._can_submit:
            self.state.last_decision_session = session_str
            self.state.last_signal_bar = sig.bar_date

    def _enter(self, sig, session: str, account: dict) -> None:
        equity, cash = float(account["equity"]), float(account["cash"])
        qty = plan_quantity(equity, cash, sig.close, self.cfg)
        limit = entry_limit_price(sig.close, self.cfg)
        plan = {"qty": qty, "limit_price": limit, "reference_close": sig.close,
                "planned_stop": stop_price_for(sig.close, self.cfg),
                "planned_risk": round(qty * sig.close * self.cfg.stop_loss_pct, 2)}
        if qty < 1:
            self.log.write("entry_skipped", reason="position size rounds to zero", **plan)
            self._report.actions.append("entry signal but size rounds to 0 shares")
            return
        if not self._can_submit:
            self.log.write("would_enter", **plan)
            self._report.actions.append(f"DRY RUN: would buy {qty} {self.cfg.symbol} limit {limit:.2f}")
            return

        cid = self._cid(session, "entry")
        self.state.pending_entry = {"client_order_id": cid, "session": session}
        self.state.save(self.dir / STATE_FILE)
        placed = self._submit(
            "entry", symbol=self.cfg.symbol, qty=str(qty), side="buy", type="limit",
            limit_price=f"{limit:.2f}", time_in_force="day", client_order_id=cid,
        )
        if placed is None:
            self.state.pending_entry = None
            return

        final = self._wait_terminal(placed["id"], self.fill_timeout)
        if not final or final.get("status") not in TERMINAL_STATUSES:
            final = self._cancel_and_wait(final or placed)
        if not final or final.get("status") not in TERMINAL_STATUSES:
            self._halt(f"entry order {cid} did not reach a final state")
            return

        filled = int(float(final.get("filled_qty") or 0))
        self.state.pending_entry = None
        self.log.write("order_final", kind="entry", order_id=final["id"], status=final["status"],
                       qty=qty, filled_qty=filled, filled_avg_price=final.get("filled_avg_price"))
        if filled == 0:
            self._report.actions.append(f"entry not filled ({final['status']})")
            return
        if filled < qty:
            self.log.write("partial_fill", kind="entry", requested=qty, filled=filled)

        position = self.broker.get_position(self.cfg.symbol)
        if position is None:
            self._halt("entry filled but no position is visible")
            return
        self.state.position = {"qty": int(float(position["qty"])),
                               "avg_entry_price": float(position["avg_entry_price"]),
                               "entry_session": session}
        self.log.write("fill", kind="entry", qty=self.state.position["qty"],
                       avg_entry_price=self.state.position["avg_entry_price"])
        self._report.actions.append(
            f"bought {self.state.position['qty']} {self.cfg.symbol} @ "
            f"{self.state.position['avg_entry_price']:.2f}"
        )
        self._ensure_stop(position)

    def _exit(self, position: dict, session: str, reason: str) -> None:
        qty = int(float(position["qty"]))
        if not self._can_submit:
            self.log.write("would_exit", reason=reason, qty=qty)
            self._report.actions.append(f"DRY RUN: would sell {qty} {self.cfg.symbol} ({reason})")
            return

        # The stop reserves the shares; release it before selling.
        for o in self._stop_orders():
            final = self._cancel_and_wait(o)
            if final and final.get("status") == "filled":
                self.log.write("exit_preempted_by_stop", order_id=o["id"])
                return
            if not final or final.get("status") not in TERMINAL_STATUSES:
                self._halt(f"could not release stop {o['id']} before exit; position stays protected")
                return

        try:
            placed = self._submit(
                "exit", symbol=self.cfg.symbol, qty=str(qty), side="sell", type="market",
                time_in_force="day", client_order_id=self._cid(session, "exit"),
            )
        except UncertainOrderState:
            self._restore_protection("exit order state unknown")
            raise
        if placed is None:
            self._restore_protection("exit order rejected")
            return

        final = self._wait_terminal(placed["id"], self.fill_timeout)
        if not final or final.get("status") not in TERMINAL_STATUSES:
            final = self._cancel_and_wait(final or placed) or final or placed
        avg_entry = float(position["avg_entry_price"])
        price = final.get("filled_avg_price")
        self.log.write("exit", reason=reason, order_id=placed["id"], status=final.get("status"),
                       qty=qty, filled_qty=final.get("filled_qty"), exit_price=price,
                       avg_entry_price=avg_entry,
                       approx_pnl=(round((float(price) - avg_entry) * float(final.get("filled_qty") or 0), 2)
                                   if price else None))

        remaining = self.broker.get_position(self.cfg.symbol)
        if remaining is not None:
            self._restore_protection(f"exit left {remaining['qty']} sh open")
            return
        self.state.position, self.state.stop_order_id = None, None
        self._report.actions.append(f"sold {qty} {self.cfg.symbol} ({reason}) @ {price}")

    def _restore_protection(self, why: str) -> None:
        position = self.broker.get_position(self.cfg.symbol)
        if position is not None:
            self._ensure_stop(position)
        self._halt(why)


def run_loop(
    trader: PaperTrader,
    interval: float = 60.0,
    max_ticks: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Tick every ``interval`` seconds until a STOP file appears (``stop`` command)."""
    stop_file = trader.dir / STOP_FILE
    if stop_file.exists():
        stop_file.unlink()
    trader.log.write("loop_start", interval=interval, submit_orders=trader.submit_orders)
    ticks = 0
    try:
        while not stop_requested(trader.dir):
            trader.run_once()
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            waited = 0.0
            while waited < interval and not stop_requested(trader.dir):
                sleep(1.0)
                waited += 1.0
    finally:
        trader.log.write("loop_stop", ticks=ticks)
