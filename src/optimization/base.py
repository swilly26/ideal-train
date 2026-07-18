"""Abstract base for strategy optimisers.

Optimisers take a strategy + historical data and search the parameter
space defined by ``StrategyConfig`` to maximise a user-chosen objective
(Sharpe ratio, Sortino, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.strategies.base import Strategy, StrategyConfig


@dataclass
class OptimizationResult:
    """The output of a single optimisation run.

    Attributes
    ----------
    best_config : StrategyConfig
        The parameter set that produced the highest objective score.
    best_score : float
        Value of the objective function for the best config.
    all_scores : pd.DataFrame
        Full table of config → score for every candidate evaluated.
    """

    best_config: StrategyConfig
    best_score: float
    all_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


class Optimizer(ABC):
    """Abstract base for parameter-search optimisers.

    Subclasses implement ``optimize`` using grid search, Bayesian
    optimisation, genetic algorithms, etc.
    """

    def __init__(
        self,
        objective_fn: Callable[[pd.Series], float],
    ) -> None:
        """
        Parameters
        ----------
        objective_fn : callable
            A function that takes a returns series and returns a scalar
            (higher is better).  E.g. ``sharpe_ratio``.
        """
        self.objective_fn = objective_fn

    @abstractmethod
    def optimize(
        self,
        strategy_cls: type[Strategy],
        data: pd.DataFrame,
        param_grid: dict[str, list[float]],
        n_iter: int = 100,
    ) -> OptimizationResult:
        """Search for the best ``StrategyConfig`` for *strategy_cls*.

        Parameters
        ----------
        strategy_cls : type[Strategy]
            Strategy class to optimise (the optimiser will instantiate
            it with candidate configs).
        data : pd.DataFrame
            Historical OHLCV data for backtesting each candidate.
        param_grid : dict
            Mapping of ``StrategyConfig`` field names to lists of values
            to try, e.g. ``{"stop_loss_pct": [0.01, 0.02, 0.03]}``.
        n_iter : int
            Maximum number of candidate evaluations.

        Returns
        -------
        OptimizationResult
        """
        ...
