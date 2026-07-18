"""AI/ML optimisation engine.

The optimisation layer is the core differentiator of AlgoFlow.  It
continuously analyses market conditions, backtests parameter
adjustments, and outputs tuned ``StrategyConfig`` values for each
active strategy.
"""

from src.optimization.base import Optimizer, OptimizationResult
from src.optimization.objectives import sharpe_ratio, sortino_ratio, max_drawdown

__all__ = ["Optimizer", "OptimizationResult", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
