"""AI/ML optimisation engine.

The optimisation layer is the core differentiator of AlgoFlow.  It
continuously analyses market conditions, backtests parameter
adjustments, and outputs tuned ``StrategyConfig`` values for each
active strategy.

Sub-modules are imported lazily to avoid circular dependencies with
the backtesting layer (which imports objectives from this package).
Import concrete optimisers directly when you need them::

    from src.optimization.grid_search import GridSearchOptimizer
    from src.optimization.genetic import GeneticOptimizer
    from src.optimization.selector import StrategySelector
    from src.optimization.runner import run_optimization
"""

from src.optimization.base import Optimizer, OptimizationResult
from src.optimization.objectives import sharpe_ratio, sortino_ratio, max_drawdown

__all__ = ["Optimizer", "OptimizationResult", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
