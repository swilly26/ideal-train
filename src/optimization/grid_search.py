"""Grid-search optimiser.

Exhaustively evaluates every combination in ``param_grid`` by running
each candidate ``StrategyConfig`` through the backtesting engine and
scoring the resulting returns curve with the objective function.
"""

from __future__ import annotations

import itertools
import logging
from typing import Callable

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.optimization.base import OptimizationResult, Optimizer
from src.strategies.base import Strategy, StrategyConfig

logger = logging.getLogger(__name__)


class GridSearchOptimizer(Optimizer):
    """Exhaustive parameter-search optimiser.

    Evaluates every combination of values in *param_grid*, backtests
    each one, and returns the config that maximises *objective_fn*.

    Parameters
    ----------
    objective_fn : callable
        Scalar scorer for a returns series (higher is better).
    backtest_engine : BacktestEngine, optional
        Pre-configured engine.  A default engine with zero commission
        and $100k initial capital is used when not supplied.
    """

    def __init__(
        self,
        objective_fn: Callable[[pd.Series], float],
        backtest_engine: BacktestEngine | None = None,
    ) -> None:
        super().__init__(objective_fn)
        self._engine = backtest_engine or BacktestEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy_cls: type[Strategy],
        data: pd.DataFrame,
        param_grid: dict[str, list[float]],
        n_iter: int = 100,
    ) -> OptimizationResult:
        """Brute-force grid search over *param_grid*.

        Parameters
        ----------
        strategy_cls : type[Strategy]
            Strategy class to optimise.
        data : pd.DataFrame
            OHLCV bars for backtesting.
        param_grid : dict
            ``StrategyConfig`` field names → list of values to try.
        n_iter : int
            Ignored (grid search evaluates all combinations).  Kept
            for interface compatibility.

        Returns
        -------
        OptimizationResult
        """
        if data.empty:
            return OptimizationResult(
                best_config=StrategyConfig(),
                best_score=float("-inf"),
                all_scores=pd.DataFrame(),
            )

        if not param_grid:
            # No parameters to search — evaluate the default config.
            return self._evaluate_single(strategy_cls, data, StrategyConfig())

        keys, values = zip(*param_grid.items())
        combinations = list(itertools.product(*values))
        limit = min(len(combinations), n_iter)

        best_score = float("-inf")
        best_config = StrategyConfig()
        score_rows: list[dict] = []

        logger.info(
            "Grid search: %d combinations (%d allowed by n_iter) for strategy %s",
            len(combinations),
            limit,
            strategy_cls.__name__,
        )

        first = True
        for idx, combo in enumerate(combinations[:limit]):
            config_kwargs = dict(zip(keys, combo))
            config = StrategyConfig(**config_kwargs)
            score = self._score_config(strategy_cls, data, config)

            score_rows.append({"config": str(config_kwargs), "score": score})

            if first or score > best_score:
                best_score = score
                best_config = config
                first = False
                logger.debug(
                    "New best: score=%.4f  config=%s", best_score, config_kwargs
                )

            # Progress report every 20 evals
            if (idx + 1) % 20 == 0:
                logger.info(
                    "Grid search progress: %d/%d  best_score=%.4f",
                    idx + 1,
                    limit,
                    best_score,
                )

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            all_scores=pd.DataFrame(score_rows),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        strategy_cls: type[Strategy],
        data: pd.DataFrame,
        config: StrategyConfig,
    ) -> OptimizationResult:
        """Evaluate a single config (used when param_grid is empty)."""
        score = self._score_config(strategy_cls, data, config)
        return OptimizationResult(
            best_config=config,
            best_score=score,
            all_scores=pd.DataFrame([{"config": str(config), "score": score}]),
        )

    def _score_config(
        self,
        strategy_cls: type[Strategy],
        data: pd.DataFrame,
        config: StrategyConfig,
    ) -> float:
        """Backtest *strategy_cls* with *config* and return the objective score."""
        strategy = strategy_cls(config)
        result = self._engine.run(strategy, data)
        equity = result.equity_curve

        if len(equity) < 2:
            return float("-inf")

        returns = equity.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return float("-inf")

        return float(self.objective_fn(returns))
