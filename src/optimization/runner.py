"""Convenience runner for strategy optimisation.

Provides a single-call entry point that loads a strategy from the
registry, builds a parameter grid, runs the chosen optimiser, and
returns the result.  Designed so end users (or automated AI pipelines)
don't have to wire up all the components manually.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.optimization.base import OptimizationResult, Optimizer
from src.optimization.grid_search import GridSearchOptimizer
from src.optimization.genetic import GeneticOptimizer
from src.optimization.objectives import sharpe_ratio
from src.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

# Sensible default parameter grids.
# Each key is a ``StrategyConfig`` field name.
_DEFAULT_PARAM_GRID: dict[str, list[float]] = {
    "entry_threshold": [0.1, 0.3, 0.5, 0.7, 1.0],
    "exit_threshold": [0.05, 0.1, 0.2, 0.3, 0.5],
    "stop_loss_pct": [0.005, 0.01, 0.02, 0.03, 0.05],
    "take_profit_pct": [0.01, 0.02, 0.05, 0.10, 0.15],
    "max_position_pct": [0.05, 0.10, 0.20],
}


def run_optimization(
    strategy_name: str,
    data: pd.DataFrame,
    method: str = "grid",
    param_grid: dict[str, list[float]] | None = None,
    objective_fn: Callable[[pd.Series], float] | None = None,
    registry: StrategyRegistry | None = None,
    backtest_engine: BacktestEngine | None = None,
    n_iter: int = 100,
    **kwargs,
) -> OptimizationResult:
    """Run full optimisation for *strategy_name* on *data*.

    Loads the strategy class from *registry*, builds a parameter grid
    (or uses the default), instantiates the chosen optimiser, and
    returns the best discovered config.

    Parameters
    ----------
    strategy_name : str
        Name of a strategy registered in *registry*.
    data : pd.DataFrame
        OHLCV bars for backtesting.
    method : {"grid", "genetic"}
        Optimisation method (default ``"grid"``).
    param_grid : dict, optional
        ``StrategyConfig`` field → list of values.  Falls back to a
        built-in default grid when ``None``.
    objective_fn : callable, optional
        Scorer for returns series.  Defaults to ``sharpe_ratio``.
    registry : StrategyRegistry, optional
        Strategy registry.  If ``None``, the module-level singleton
        ``src.strategies.registry`` is used.
    backtest_engine : BacktestEngine, optional
        Pre-configured engine (default: zero-commission, $100k capital).
    n_iter : int
        Max candidate evaluations (default 100).
    **kwargs
        Forwarded to the optimiser constructor (e.g. ``generations``,
        ``population_size`` for genetic).

    Returns
    -------
    OptimizationResult

    Raises
    ------
    KeyError
        If *strategy_name* is not found in the registry.
    ValueError
        If *method* is unrecognised.
    """
    if registry is None:
        from src.strategies import registry as _default_registry  # noqa: E402
        registry = _default_registry

    strategy_cls = registry.get(strategy_name)

    if param_grid is None:
        param_grid = _DEFAULT_PARAM_GRID

    if objective_fn is None:
        objective_fn = sharpe_ratio

    engine = backtest_engine or BacktestEngine()

    if method == "grid":
        optimizer: Optimizer = GridSearchOptimizer(
            objective_fn=objective_fn,
            backtest_engine=engine,
        )
    elif method == "genetic":
        optimizer = GeneticOptimizer(
            objective_fn=objective_fn,
            backtest_engine=engine,
            population_size=kwargs.pop("population_size", 20),
            generations=kwargs.pop("generations", 10),
            mutation_rate=kwargs.pop("mutation_rate", 0.2),
            mutation_scale=kwargs.pop("mutation_scale", 0.1),
            elitism=kwargs.pop("elitism", 2),
            tournament_size=kwargs.pop("tournament_size", 3),
        )
    else:
        raise ValueError(
            f"Unknown optimisation method: '{method}'.  "
            "Use 'grid' or 'genetic'."
        )

    logger.info(
        "Running %s optimisation for '%s' on %d bars (n_iter=%d)",
        method,
        strategy_name,
        len(data),
        n_iter,
    )

    return optimizer.optimize(strategy_cls, data, param_grid, n_iter=n_iter)
