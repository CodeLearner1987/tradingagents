"""Run leakage-safe Pattern Memory validation against Yahoo Finance history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yfinance as yf

from tradingagents.pattern_memory.walk_forward import walk_forward_validate


def download_history(ticker: str, start: str, end: str | None, interval: str):
    """Download OHLCV history without modifying or executing trading state."""
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if frame.empty:
        raise RuntimeError(f"No historical data returned for {ticker}")
    if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
        frame.columns = frame.columns.get_level_values(0)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frame = download_history(args.ticker, args.start, args.end, args.interval)
    result = walk_forward_validate(
        frame,
        ticker=args.ticker,
        timeframe=args.interval,
        k=args.k,
        min_history=args.min_history,
    )

    report = {
        "ticker": result.ticker,
        "timeframe": result.timeframe,
        "rows": len(frame),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "predictions": result.predictions,
        "directional_hits": result.directional_hits,
        "directional_accuracy": result.directional_accuracy,
        "mae_return": result.mae,
        "insufficient_data": result.insufficient_data,
        "insufficient_rate": result.insufficient_rate,
        "execution": "NONE",
        "method": "chronological walk-forward; no future outcomes available at prediction time",
    }

    print(json.dumps(report, indent=2, default=str))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
