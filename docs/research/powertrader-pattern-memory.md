# PowerTrader Pattern-Memory Extraction

## Purpose

This document records the useful architectural idea extracted from the public
PowerTrader_AI project and how it is being implemented in TradingAgents.

PowerTrader_AI describes a multi-timeframe, instance-based historical-pattern
predictor that compares current market conditions with prior patterns, uses
what happened after those patterns as the prediction target, and updates the
reliability of historical patterns after outcomes are known.

Source: `https://github.com/garagesteve1155/PowerTrader_AI`

## What we keep

1. **Pattern memory** — retain historical market states with their subsequent
   observed outcome.
2. **Similarity matching** — find historical states that resemble the current
   state rather than relying only on a language model's generated opinion.
3. **Weighted prediction** — give more influence to closer and more reliable
   historical matches.
4. **Online evaluation** — compare predictions with realized outcomes and update
   reliability incrementally.
5. **Multi-timeframe support** — the contracts include timeframe as an explicit
   field so evidence can be evaluated separately or combined later.
6. **Traceable evidence** — predictions retain the IDs of the historical
   observations used to produce them.

## What we deliberately do not import

- Automatic trade execution.
- Dollar-cost averaging rules.
- No-stop-loss strategy assumptions.
- Position sizing or portfolio management rules.
- Exchange credentials or live-order behavior.
- Any claim that historical similarity guarantees future performance.

## TradingAgents implementation

The new `tradingagents/pattern_memory/` package is deliberately isolated from
the existing agent graph. It provides deterministic primitives that can later
be called by an analyst or research workflow without changing the existing
LLM agent contracts.

Current flow:

```text
Current feature vector
        |
        v
Historical observations --> similarity matching
        |                         |
        +-------------------------+
                  |
                  v
          reliability weighting
                  |
                  v
          weighted prediction
                  |
                  v
          traceable evidence
                  |
             outcome arrives
                  |
                  v
        prediction evaluation
                  |
                  v
        reliability update
```

The first implementation is prediction-only and has unit tests. Integration
with the TradingAgents graph should happen only after the module is validated
against real historical data and the desired feature representation is fixed.
