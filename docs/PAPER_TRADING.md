# SMA 20/50 paper-trading workflow (Alpaca PAPER only)

A rules-based workflow for SPY on daily bars, separate from the LLM agents.
It never talks to Alpaca's live endpoint and submits nothing until you approve.

## Rules

| Rule | Value |
|---|---|
| Symbol | SPY only (other symbols are refused in code) |
| Entry | SMA20 crosses above SMA50 on the last completed daily close → buy at the next session |
| Exit | SMA20 below SMA50 on the last completed close → sell at the next session |
| Protective stop | GTC sell stop 2% below the **actual average fill** (`--stop-pct`) |
| Sizing | Whole shares so a stop-out loses ≤ 0.5% of equity, capped by cash |
| Limits | One position, long only, no leverage, no options |

Entries are limit orders capped 3% above the prior close, so a gap up can never
make the order cost more than the account's cash. Exits are market orders.

## Setup

1. Create **paper** API keys: <https://app.alpaca.markets> → switch to *Paper
   Trading* → API keys. Paper key IDs start with `PK`.
2. Put them in `.env` at the repo root (already git-ignored):
   ```
   ALPACA_PAPER_API_KEY_ID=PK...
   ALPACA_PAPER_API_SECRET_KEY=...
   ```
   Only these names are read; generic `APCA_*` variables are ignored on purpose.
3. `pip install -e .` (no extra packages are needed).

## Commands

```bash
python -m tradingagents.paper_trading backtest                  # Yahoo data, ~5 years
python -m tradingagents.paper_trading backtest --source alpaca  # Alpaca bars from 2016
python -m tradingagents.paper_trading verify     # read-only checks; submits nothing
python -m tradingagents.paper_trading run-once   # one dry-run tick
python -m tradingagents.paper_trading approve    # type APPROVE PAPER ORDERS
python -m tradingagents.paper_trading start --submit-orders   # run; Ctrl-C or `stop`
python -m tradingagents.paper_trading status | logs | stop | revoke | resume
```

`start` without `--submit-orders` runs the same loop as a dry run and logs
`would_enter` / `would_exit` instead of sending orders. Run `start` during or
before market hours; decisions happen at the first tick after the open.

## Safety behaviour

- **Approval**: orders are sent only with `--submit-orders` *and* an approval
  recorded by `approve` for the current settings and this paper account.
  Changing any setting (for example `--stop-pct`) invalidates the approval.
- **Stop verification**: every tick checks the position has exactly one active
  stop for its full quantity at the right price, and replaces a wrong one. If
  the stop can't be confirmed (rejected, never active, or submission not
  approved), the trader **halts**: no new entries until you run `resume`.
- **Uncertain orders**: a timeout or 5xx on submission is never retried
  blindly. The order is looked up by its client order id; if it still can't
  be found, the trader halts.
- **Partial fills**: an unfilled remainder is cancelled after 90 s and the stop
  covers the filled quantity.
- **Exits**: the stop is cancelled first (it reserves the shares). If the sell
  is rejected, the stop is put back and the trader halts.
- **Unexpected state**: positions in other symbols, short positions, or sell
  stops with no position halt the trader.
- **Same bar twice**: a signal bar is acted on once, even if a lagging data
  feed serves it again.
- **Secrets**: keys go only in request headers; the event log scrubs them.

`stop` ends the loop but leaves any position and its GTC stop in place.

## Files

Under `~/.tradingagents/paper_trading/` (override with `TRADINGAGENTS_PAPER_DIR`):
`events.jsonl` (signals, orders, fills, stops, errors, exit reasons),
`state.json`, `approval.json`, and `backtests/<timestamp>/report.md` + `results.csv`.

## Backtest method

- Signals use closes up to bar *t*; trades fill at bar *t+1*'s open.
- Adverse slippage is applied to every entry and exit at 0.05%, 0.10% and
  0.25% per side, plus SEC and FINRA sell fees ($0 commission). Check current
  fee rates before relying on exact cost figures.
- Stops are checked against each bar's low, from the entry bar on. A gap below
  the stop fills at the worse open.
- Results are reported separately for development (60%), validation (20%) and
  test (20%) periods, and for the full history. Nothing is tuned on any of
  them; the 20/50/2% settings are fixed in advance.
- Buy-and-hold uses the same slippage and fees, fully invested.

## Limitations

- Each trade commits only about 25% of equity (0.5% risk / 2% stop), so
  returns aren't risk-equivalent to buy-and-hold. Compare drawdowns as well.
- An SMA20/50 rule trades rarely: about one to three round trips a year on
  SPY. Even a 10-year backtest has too few trades to separate skill from luck.
- Daily bars can't show intraday stop fills, halts or partial fills. Paper
  fills are simulated by Alpaca and are usually kinder than live fills.
- Live signals use Alpaca bars; the default backtest uses Yahoo data. Adjusted
  closes can differ slightly, which can move a crossover by a day.
- A missed session delays exits by a day; a missed cross-above day skips that
  entry.
