"""Genetic-algorithm optimiser.

Population-based search that evolves ``StrategyConfig`` parameters
through selection, crossover, and mutation over multiple generations.
More efficient than grid search for large parameter spaces.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.optimization.base import OptimizationResult, Optimizer
from src.strategies.base import Strategy, StrategyConfig

logger = logging.getLogger(__name__)


def _config_to_array(config: StrategyConfig, keys: list[str]) -> np.ndarray:
    """Extract numeric fields from a ``StrategyConfig`` into a 1-d array."""
    return np.array([getattr(config, k) for k in keys], dtype=float)


def _array_to_config(arr: np.ndarray, keys: list[str]) -> StrategyConfig:
    """Build a ``StrategyConfig`` from an array of values for *keys*."""
    kwargs = dict(zip(keys, arr))
    return StrategyConfig(**kwargs)


class GeneticOptimizer(Optimizer):
    """Genetic-algorithm parameter optimiser.

    Evolves a population of ``StrategyConfig`` candidates over multiple
    generations, using tournament selection, uniform crossover, and
    Gaussian mutation.

    Parameters
    ----------
    objective_fn : callable
        Scalar scorer for a returns series (higher is better).
    backtest_engine : BacktestEngine, optional
        Pre-configured engine.  Default: zero-commission, $100k capital.
    population_size : int
        Number of candidates per generation (default 20).
    generations : int
        Number of generations to evolve (default 10).
    mutation_rate : float
        Probability of mutating each gene (default 0.2).
    mutation_scale : float
        Standard deviation of Gaussian noise added during mutation,
        relative to the parameter's current value (default 0.1, i.e. 10 %).
    elitism : int
        Number of best candidates carried unchanged into the next
        generation (default 2).
    tournament_size : int
        Number of candidates in each tournament for selection (default 3).
    """

    def __init__(
        self,
        objective_fn: Callable[[pd.Series], float],
        backtest_engine: BacktestEngine | None = None,
        *,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.1,
        elitism: int = 2,
        tournament_size: int = 3,
    ) -> None:
        super().__init__(objective_fn)
        self._engine = backtest_engine or BacktestEngine()
        self.population_size = max(population_size, 4)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism = max(elitism, 1)
        self.tournament_size = tournament_size

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
        """Evolve a population to find the best ``StrategyConfig``.

        Parameters
        ----------
        strategy_cls : type[Strategy]
            Strategy class to optimise.
        data : pd.DataFrame
            OHLCV bars for backtesting.
        param_grid : dict
            ``StrategyConfig`` field names → list of values to try.
            The first and last value per key define the search range;
            intermediate values are used as sampling hints.
        n_iter : int
            Maximum number of candidate evaluations.  If the
            population × generations exceeds this, generations are
            capped.

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
            return OptimizationResult(
                best_config=StrategyConfig(),
                best_score=self._score_config(strategy_cls, data, StrategyConfig()),
                all_scores=pd.DataFrame(),
            )

        keys = list(param_grid.keys())
        bounds = self._compute_bounds(param_grid, keys)

        # Cap generations so we don't exceed n_iter evaluations
        max_gen = max(1, n_iter // self.population_size)
        generations = min(self.generations, max_gen)

        # Initialise random population
        rng = np.random.default_rng()
        population = self._init_population(keys, bounds, rng)

        best_config = StrategyConfig()
        best_score = float("-inf")
        all_rows: list[dict] = []
        eval_count = 0
        first_eval = True

        for gen in range(generations):
            # Evaluate every individual in the population
            scores: list[float] = []
            for individual in population:
                config = _array_to_config(individual, keys)
                score = self._score_config(strategy_cls, data, config)
                scores.append(score)
                eval_count += 1

                all_rows.append({"generation": gen, "config": str(config), "score": score})

                if first_eval or score > best_score:
                    best_score = score
                    best_config = config
                    first_eval = False

            logger.info(
                "Gen %d/%d  best=%.4f  avg=%.4f  evals=%d",
                gen + 1,
                generations,
                best_score,
                float(np.mean(scores)),
                eval_count,
            )

            if gen == generations - 1:
                break

            # Select next generation
            population = self._evolve(population, scores, keys, bounds, rng)

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            all_scores=pd.DataFrame(all_rows),
        )

    # ------------------------------------------------------------------
    # Population machinery
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bounds(
        param_grid: dict[str, list[float]], keys: list[str]
    ) -> dict[str, tuple[float, float]]:
        """Extract (min, max) bounds for each parameter from *param_grid*."""
        bounds: dict[str, tuple[float, float]] = {}
        for k in keys:
            vals = param_grid[k]
            bounds[k] = (float(min(vals)), float(max(vals)))
        return bounds

    def _init_population(
        self,
        keys: list[str],
        bounds: dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Create a random initial population within *bounds*."""
        population: list[np.ndarray] = []
        for _ in range(self.population_size):
            individual = np.array(
                [rng.uniform(*bounds[k]) for k in keys], dtype=float
            )
            population.append(individual)
        return population

    def _evolve(
        self,
        population: list[np.ndarray],
        scores: list[float],
        keys: list[str],
        bounds: dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Produce the next generation via selection, crossover, and mutation."""
        # Elitism — keep the best
        elite_indices = np.argsort(scores)[::-1][: self.elitism]
        next_gen = [population[i].copy() for i in elite_indices]

        # Breed the rest
        while len(next_gen) < self.population_size:
            parent1 = self._tournament_select(population, scores, rng)
            parent2 = self._tournament_select(population, scores, rng)
            child = self._crossover(parent1, parent2, rng)
            child = self._mutate(child, keys, bounds, rng)
            next_gen.append(child)

        return next_gen[: self.population_size]

    def _tournament_select(
        self,
        population: list[np.ndarray],
        scores: list[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select one individual via tournament selection (higher score wins)."""
        indices = rng.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = indices[0]
        best_s = scores[best_idx]
        for i in indices[1:]:
            if scores[i] > best_s:
                best_idx = i
                best_s = scores[i]
        return population[best_idx].copy()

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Uniform crossover: each gene randomly from either parent."""
        mask = rng.random(len(parent1)) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(
        self,
        individual: np.ndarray,
        keys: list[str],
        bounds: dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply Gaussian mutation to a random subset of genes."""
        for i, key in enumerate(keys):
            if rng.random() < self.mutation_rate:
                lo, hi = bounds[key]
                noise = rng.normal(0, self.mutation_scale * abs(individual[i]))
                individual[i] = np.clip(individual[i] + noise, lo, hi)
        return individual

    # ------------------------------------------------------------------
    # Shared scoring helper
    # ------------------------------------------------------------------

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
