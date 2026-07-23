"""Backtesting engine.

Provides a lightweight, event-driven backtester that replays historical
data through strategies and tracks simulated P&L.  The AI optimisation
layer calls this repeatedly while searching for optimal parameters.
"""

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.metrics import (
    compute_metrics,
    max_drawdown,
    num_trades,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "compute_metrics",
    "max_drawdown",
    "num_trades",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "win_rate",
]
