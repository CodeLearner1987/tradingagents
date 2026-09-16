"""Run Pattern Memory validation across a fixed asset suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yfinance as yf

from tradingagents.pattern_memory.walk_forward import walk_forward_validate

DEFAULT_TICKERS = ("BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "MSFT")


def download_history(ticker: str, start: str, end: str | None, interval: str):
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
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for ticker in args.tickers:
        try:
            frame = download_history(ticker, args.start, args.end, args.interval)
            result = walk_forward_validate(
                frame,
                ticker=ticker,
                timeframe=args.interval,
                k=args.k,
                min_history=args.min_history,
            )
            results.append(
                {
                    "ticker": ticker,
                    "rows": len(frame),
                    "predictions": result.predictions,
                    "directional_accuracy": result.directional_accuracy,
                    "baseline_directional_accuracy": result.baseline_directional_accuracy,
                    "directional_accuracy_delta_vs_baseline": result.directional_accuracy_delta_vs_baseline,
                    "mae_return": result.mae,
                    "insufficient_rate": result.insufficient_rate,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})

    valid = [item for item in results if item.get("error") is None]
    deltas = [item["directional_accuracy_delta_vs_baseline"] for item in valid if item["directional_accuracy_delta_vs_baseline"] is not None]
    beats = sum(delta > 0 for delta in deltas)
    aggregate = {
        "assets_tested": len(valid),
        "assets_beating_baseline": beats,
        "average_delta": sum(deltas) / len(deltas) if deltas else None,
        "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else None,
    }

    report = {
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "baseline": "last_observed_direction",
        "results": results,
        "aggregate": aggregate,
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
