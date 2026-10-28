"""Offline scikit-learn experiments on daily price direction.

Research-only: nothing in this package places orders, and the LangGraph
trading pipeline does not import it. Requires the ``ml`` extra
(``pip install "tradingagents[ml]"``) for the modelling stages; feature and
target construction need only pandas/numpy/stockstats.
"""
