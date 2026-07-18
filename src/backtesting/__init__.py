"""Backtesting engine.

Provides a lightweight, event-driven backtester that replays historical
data through strategies and tracks simulated P&L.  The AI optimisation
layer calls this repeatedly while searching for optimal parameters.
"""

from src.backtesting.engine import BacktestEngine, BacktestResult

__all__ = ["BacktestEngine", "BacktestResult"]
