"""Rules-based SMA-crossover workflow for an Alpaca PAPER account.

Research and paper-trading only. The client in :mod:`.broker` refuses any
endpoint other than Alpaca's paper-trading host, and order submission is off
unless a human has recorded an approval (see :mod:`.trader`). The LangGraph
analysis pipeline does not import this package.

Run ``python -m tradingagents.paper_trading --help`` for the commands.
"""
