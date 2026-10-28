"""Command line for the SMA paper-trading workflow.

    python -m tradingagents.paper_trading backtest [--source yfinance|alpaca|csv]
    python -m tradingagents.paper_trading verify        # read-only paper checks
    python -m tradingagents.paper_trading status
    python -m tradingagents.paper_trading run-once      # dry run unless --submit-orders
    python -m tradingagents.paper_trading approve       # typed human approval
    python -m tradingagents.paper_trading start [--submit-orders]
    python -m tradingagents.paper_trading stop | resume | revoke | logs
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.markdown import Markdown

from tradingagents.paper_trading.backtest import format_report, run_report
from tradingagents.paper_trading.broker import AlpacaPaperClient, BrokerError, mask
from tradingagents.paper_trading.config import (
    KEY_ID_ENV,
    PAPER_TRADING_URL,
    SECRET_KEY_ENV,
    StrategyConfig,
    state_dir,
)
from tradingagents.paper_trading.trader import (
    APPROVAL_PHRASE,
    RESUME_PHRASE,
    EventLog,
    PaperTrader,
    TraderState,
    approval_status,
    record_approval,
    request_stop,
    revoke_approval,
    run_loop,
)

app = typer.Typer(add_completion=False, help="SMA 20/50 crossover on SPY — Alpaca PAPER only.")
console = Console()

StopPct = typer.Option(0.02, help="Protective stop below the average entry fill (0.02 = 2%).")


def _cfg(stop_pct: float = 0.02) -> StrategyConfig:
    return StrategyConfig(stop_loss_pct=stop_pct)


def _client() -> AlpacaPaperClient:
    try:
        return AlpacaPaperClient.from_env()
    except (BrokerError, ValueError) as exc:
        console.print(f"[red]Cannot create the Alpaca paper client:[/red] {exc}")
        console.print(f"Set {KEY_ID_ENV} and {SECRET_KEY_ENV} (paper keys) in your .env.")
        raise typer.Exit(2) from None


def _print_report(report) -> None:
    color = "red" if report.halted else "green"
    console.print(f"[bold]Session {report.session}[/bold]  mode: [bold]{report.mode}[/bold]  "
                  f"halted: [{color}]{report.halted}[/{color}] {report.halt_reason}")
    if report.signal:
        s = report.signal
        console.print(f"Signal on {s['bar_date']}: close {s['close']:.2f}, "
                      f"SMA20 {s['sma_fast']:.2f}, SMA50 {s['sma_slow']:.2f}, "
                      f"entry={s['entry']}, exit={s['exit']}")
    for a in report.actions:
        console.print(f"  • {a}")
    for w in report.warnings:
        console.print(f"  [yellow]! {w}[/yellow]")


def _load_bars(source: str, csv: Path | None, symbol: str) -> tuple[pd.DataFrame, str]:
    if source == "csv":
        if csv is None:
            raise typer.BadParameter("--csv is required with --source csv")
        raw = pd.read_csv(csv)
        rename = {c: c.capitalize() for c in raw.columns}
        rename.update({c: "Date" for c in raw.columns if c.lower() in ("date", "timestamp")})
        return raw.rename(columns=rename), f"CSV file {csv.name}"
    if source == "alpaca":
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        bars = _client().get_daily_bars(symbol, "2016-01-01", end)
        return bars, "Alpaca market data (split/dividend-adjusted daily bars)"
    if source == "yfinance":
        from tradingagents.dataflows.stockstats_utils import load_ohlcv

        bars = load_ohlcv(symbol, datetime.now().strftime("%Y-%m-%d"))
        return bars, "Yahoo Finance via load_ohlcv (adjusted, ~5 years)"
    raise typer.BadParameter("source must be yfinance, alpaca or csv")


@app.command()
def backtest(
    source: str = typer.Option("yfinance", help="yfinance | alpaca | csv"),
    csv: Annotated[
        Path | None, typer.Option(help="CSV with Date,Open,High,Low,Close (for --source csv).")
    ] = None,
    capital: float = typer.Option(100_000.0, help="Starting capital per period."),
    stop_pct: float = StopPct,
):
    """Backtest vs buy-and-hold at 0.05/0.10/0.25% slippage per side; writes a report."""
    cfg = _cfg(stop_pct)
    bars, label = _load_bars(source, csv, cfg.symbol)
    report = run_report(bars, cfg, initial_capital=capital)
    text = format_report(report, cfg, label)
    out = state_dir() / "backtests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    report.to_csv(out / "results.csv", index=False)
    console.print(Markdown(text))
    console.print(f"\nSaved to {out}")


@app.command()
def verify():
    """Read-only end-to-end check against the PAPER account. Submits nothing."""
    client = _client()
    checks: list[tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            checks.append((name, False, str(exc)))

    check("trading endpoint is paper", lambda: PAPER_TRADING_URL)
    check("credentials authenticate; account is a paper account",
          lambda: f"account {mask(client.get_account()['account_number'])}")
    check("account active and not blocked", lambda: _account_summary(client.get_account()))
    check("market clock", lambda: json.dumps(client.get_clock()))
    check("positions readable", lambda: f"{len(client.list_positions())} open position(s)")
    check("open orders readable", lambda: f"{len(client.list_open_orders('SPY'))} open SPY order(s)")
    check("daily bars + signal", lambda: _signal_summary(client))
    for name, ok, detail in checks:
        mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"{mark} {name}: {detail}")
    if not all(ok for _, ok, _ in checks):
        raise typer.Exit(1)


def _account_summary(account: dict) -> str:
    if str(account.get("status")).upper() != "ACTIVE" or account.get("trading_blocked"):
        raise RuntimeError(f"status={account.get('status')} trading_blocked={account.get('trading_blocked')}")
    return f"equity {float(account['equity']):,.2f}, cash {float(account['cash']):,.2f}"


def _signal_summary(client: AlpacaPaperClient) -> str:
    from tradingagents.paper_trading.strategy import latest_signal

    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    sig = latest_signal(client.get_daily_bars("SPY", start, end), StrategyConfig())
    return (f"{sig.bar_date}: close {sig.close:.2f}, SMA20 {sig.sma_fast:.2f}, "
            f"SMA50 {sig.sma_slow:.2f}, entry={sig.entry}, exit={sig.exit}")


@app.command()
def status(stop_pct: float = StopPct):
    """Show account, position, open orders, halt state, approval and recent events."""
    cfg, directory = _cfg(stop_pct), state_dir()
    client = _client()
    account = client.get_account()
    state = TraderState.load(directory / "state.json")
    ok, why = approval_status(directory, cfg, account["account_number"])
    console.print(f"Account {mask(account['account_number'])} ({PAPER_TRADING_URL}): "
                  f"{_account_summary(account)}")
    console.print(f"Clock: {json.dumps(client.get_clock())}")
    for p in client.list_positions():
        console.print(f"Position: {p['symbol']} {p['qty']} @ {p['avg_entry_price']} "
                      f"(unrealized {p.get('unrealized_pl')})")
    for o in client.list_open_orders(cfg.symbol):
        console.print(f"Open order: {o['side']} {o.get('type')} {o['qty']} "
                      f"stop={o.get('stop_price')} limit={o.get('limit_price')} status={o['status']}")
    console.print(f"Halted: {state.halted} {state.halt_reason}")
    console.print(f"Order submission approved: {ok} ({why})")
    console.print(f"State directory: {directory}")
    for e in EventLog(directory / "events.jsonl").tail(10):
        console.print(f"  {e['ts']} {e['event']}")


@app.command("run-once")
def run_once(
    submit_orders: bool = typer.Option(False, help="Send orders (requires a recorded approval)."),
    stop_pct: float = StopPct,
):
    """One reconcile/protect/decide tick. Dry run unless --submit-orders and approved."""
    trader = PaperTrader(_client(), _cfg(stop_pct), state_dir(), submit_orders=submit_orders)
    _print_report(trader.run_once())


@app.command()
def start(
    submit_orders: bool = typer.Option(False, help="Send orders (requires a recorded approval)."),
    interval: float = typer.Option(60.0, help="Seconds between ticks."),
    stop_pct: float = StopPct,
):
    """Run ticks until `stop` is issued (or Ctrl-C)."""
    cfg, directory = _cfg(stop_pct), state_dir()
    client = _client()
    if submit_orders:
        ok, why = approval_status(directory, cfg, client.get_account()["account_number"])
        if not ok:
            console.print(f"[red]Refusing to start with --submit-orders:[/red] {why}")
            raise typer.Exit(1)
    trader = PaperTrader(client, cfg, directory, submit_orders=submit_orders)
    console.print(f"Started ({'SUBMITTING paper orders' if submit_orders else 'dry run'}); "
                  f"stop with: python -m tradingagents.paper_trading stop")
    try:
        run_loop(trader, interval=interval)
    except KeyboardInterrupt:
        console.print("Interrupted.")


@app.command()
def stop():
    """Ask a running `start` loop to exit after its current tick."""
    request_stop(state_dir())
    console.print("Stop requested. Open positions and their GTC stop orders are left in place.")


@app.command()
def approve(stop_pct: float = StopPct):
    """Record human approval to submit PAPER orders for the current settings."""
    cfg, directory = _cfg(stop_pct), state_dir()
    account = _client().get_account()
    console.print(f"Paper account {mask(account['account_number'])} at {PAPER_TRADING_URL}")
    console.print(f"Settings: {json.dumps(cfg.__dict__)}")
    console.print("This allows the workflow to submit PAPER orders (entries, exits, stops).")
    typed = typer.prompt(f"Type '{APPROVAL_PHRASE}' to approve")
    if typed.strip() != APPROVAL_PHRASE:
        console.print("Not approved.")
        raise typer.Exit(1)
    record_approval(directory, cfg, account["account_number"])
    console.print("Approved. Start with: python -m tradingagents.paper_trading start --submit-orders")


@app.command()
def revoke():
    """Remove the approval; later ticks run as dry runs."""
    removed = revoke_approval(state_dir())
    console.print("Approval revoked." if removed else "No approval was recorded.")


@app.command()
def resume(stop_pct: float = StopPct):
    """Clear a halt after you have reviewed and fixed its cause."""
    directory = state_dir()
    state = TraderState.load(directory / "state.json")
    if not state.halted:
        console.print("Not halted.")
        return
    console.print(f"Halt reason: {state.halt_reason}")
    if typer.prompt(f"Type '{RESUME_PHRASE}' to clear the halt").strip() != RESUME_PHRASE:
        raise typer.Exit(1)
    PaperTrader(_client(), _cfg(stop_pct), directory).resume()
    console.print("Halt cleared.")


@app.command()
def logs(n: int = typer.Option(30, help="Number of recent events.")):
    """Print recent events (signals, orders, fills, stops, errors, exits)."""
    for e in EventLog(state_dir() / "events.jsonl").tail(n):
        console.print_json(json.dumps(e))


if __name__ == "__main__":
    app()
