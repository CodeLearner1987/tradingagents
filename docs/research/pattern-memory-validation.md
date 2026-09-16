# Pattern Memory Historical Validation

## Data source

The first validation uses Yahoo Finance historical OHLCV data through the repository's existing `yfinance` dependency. This is a research data source only; it is not an exchange connection and it cannot place trades.

Initial proof target:

- Asset: `BTC-USD`
- Interval: `1d`
- Start: `2021-01-01`
- Execution: none

## Feature contract

Each candle is converted into a point-in-time vector containing:

- 1-, 3-, and 5-period close returns
- high-low range as a percentage of close
- candle body as a percentage of close
- close location inside the candle range
- volume percentage change
- five-period return volatility

No future row is used to construct a feature vector.

## Walk-forward rule

At prediction row `t`, the engine may only use historical observations whose next candle has already closed. The current row is added to memory only after its next-period outcome is known. Reliability updates happen after the test outcome is known and before later predictions.

This prevents the most important form of lookahead leakage for this experiment.

## Metrics

- **Directional accuracy:** predicted and realized next-period return have the same sign.
- **MAE:** mean absolute error of predicted next-period return.
- **Insufficient rate:** fraction of evaluable rows where no usable historical match was available.

These are research measurements, not evidence that a trading strategy is profitable.

## Run

From the repository root:

```text
python scripts/validate_pattern_memory.py --ticker BTC-USD --start 2021-01-01 --interval 1d
```

Optional JSON output:

```text
python scripts/validate_pattern_memory.py --ticker BTC-USD --start 2021-01-01 --output reports/pattern-memory-btc-1d.json
```

## Interpretation

Do not integrate this engine into the live decision path based on one asset or one backtest. First compare it with simple baselines, repeat it across assets/timeframes, inspect performance by market regime, and confirm that results survive strict chronological validation.
