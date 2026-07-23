"""Strategy selector — optimise across multiple strategies and pick the best.

Given a list of strategy names from the ``StrategyRegistry``, runs
optimisation for each and ranks them by their best objective score.
Also supports building a weighted ensemble across the top-N strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.optimization.base import OptimizationResult, Optimizer
from src.optimization.grid_search import GridSearchOptimizer
from src.strategies.base import Strategy, StrategyConfig
from src.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class StrategyRanking:
    """Result of optimising a single strategy.

    Attributes
    ----------
    name : str
        Strategy name (as registered in ``StrategyRegistry``).
    result : OptimizationResult
        Full optimisation result including best config and score table.
    rank : int
        Position in the sorted ranking (1 = best).
    """

    name: str
    result: OptimizationResult
    rank: int = 0


@dataclass
class StrategySelectionResult:
    """Aggregate output of ``StrategySelector.select_best``.

    Attributes
    ----------
    best_strategy : str
        Name of the strategy with the highest objective score.
    best_config : StrategyConfig
        Optimised config for that strategy.
    best_score : float
        Objective score of the best strategy + config.
    rankings : list[StrategyRanking]
        All evaluated strategies, ranked by score (best first).
    """

    best_strategy: str
    best_config: StrategyConfig
    best_score: float
    rankings: list[StrategyRanking] = field(default_factory=list)


@dataclass
class EnsembleResult:
    """Weighted ensemble of multiple strategies.

    Attributes
    ----------
    strategies : list[str]
        Included strategy names.
    configs : list[StrategyConfig]
        Optimised config for each strategy (with weight set).
    weights : list[float]
        Normalised weight for each strategy (sums to 1.0).
    """

    strategies: list[str] = field(default_factory=list)
    configs: list[StrategyConfig] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


class StrategySelector:
    """Select the best strategy (or ensemble) for a given market context.

    Optimises multiple strategies on the same historical data and ranks
    them by their best objective score.  Can also build a weighted
    ensemble for combined signal generation.

    Parameters
    ----------
    registry : StrategyRegistry
        Strategy registry to pull strategy classes from.
    optimizer : Optimizer
        Pre-configured optimiser instance (e.g. ``GridSearchOptimizer``).
    """

    def __init__(
        self,
        registry: StrategyRegistry,
        optimizer: Optimizer,
    ) -> None:
        self.registry = registry
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best(
        self,
        strategy_names: list[str],
        data: pd.DataFrame,
        param_grid: dict[str, list[float]],
        n_iter: int = 100,
    ) -> StrategySelectionResult:
        """Optimise each named strategy and return the best one.

        Parameters
        ----------
        strategy_names : list[str]
            Strategy names registered in *registry*.
        data : pd.DataFrame
            OHLCV bars for backtesting.
        param_grid : dict
            ``StrategyConfig`` field values to search.
        n_iter : int
            Max evaluations per strategy.

        Returns
        -------
        StrategySelectionResult
        """
        if not strategy_names:
            return StrategySelectionResult(
                best_strategy="",
                best_config=StrategyConfig(),
                best_score=float("-inf"),
            )

        rankings: list[StrategyRanking] = []

        for name in strategy_names:
            try:
                strategy_cls = self.registry.get(name)
            except KeyError:
                logger.warning("Strategy '%s' not found in registry — skipping", name)
                continue

            logger.info("Optimising strategy '%s' …", name)
            result = self.optimizer.optimize(
                strategy_cls, data, param_grid, n_iter=n_iter,
            )
            rankings.append(StrategyRanking(name=name, result=result))

        if not rankings:
            return StrategySelectionResult(
                best_strategy="",
                best_config=StrategyConfig(),
                best_score=float("-inf"),
            )

        # Sort by best_score descending
        rankings.sort(key=lambda r: r.result.best_score, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1

        best = rankings[0]
        return StrategySelectionResult(
            best_strategy=best.name,
            best_config=best.result.best_config,
            best_score=best.result.best_score,
            rankings=rankings,
        )

    def combine_strategies(
        self,
        strategy_names: list[str],
        data: pd.DataFrame,
        param_grid: dict[str, list[float]],
        top_n: int = 3,
        n_iter: int = 100,
    ) -> EnsembleResult:
        """Build a weighted ensemble from the top-N strategies.

        Each strategy is optimised separately, then weights are assigned
        proportional to their best objective scores (softmax-style).

        Parameters
        ----------
        strategy_names : list[str]
            Candidate strategy names.
        data : pd.DataFrame
            OHLCV bars.
        param_grid : dict
            Parameter search grid.
        top_n : int
            How many of the top-ranked strategies to include.
        n_iter : int
            Max evaluations per strategy.

        Returns
        -------
        EnsembleResult
        """
        selection = self.select_best(strategy_names, data, param_grid, n_iter=n_iter)

        top_rankings = selection.rankings[:top_n]
        if not top_rankings:
            return EnsembleResult()

        # Extract scores and compute softmax-style weights
        scores = [r.result.best_score for r in top_rankings]

        # Guard against all scores being -inf or NaN
        finite_scores = [s for s in scores if s != float("-inf") and not (s != s)]  # s != s catches NaN
        if not finite_scores:
            # All scores are -inf or NaN — uniform weights
            weights = [1.0 / len(top_rankings)] * len(top_rankings)
        else:
            # Shift scores so the minimum is 0 (avoid negative weights)
            min_score = min(finite_scores)
            adjusted = [max(s - min_score, 0.0) for s in scores]
            total = sum(adjusted)
            if total == 0:
                weights = [1.0 / len(top_rankings)] * len(top_rankings)
            else:
                weights = [a / total for a in adjusted]

        configs: list[StrategyConfig] = []
        for ranking, w in zip(top_rankings, weights):
            cfg = ranking.result.best_config
            cfg.weight = w
            configs.append(cfg)

        return EnsembleResult(
            strategies=[r.name for r in top_rankings],
            configs=configs,
            weights=weights,
        )
