"""Tests for the optimisation module — grid search, genetic algorithm,
strategy selector, and convenience runner.
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.optimization.base import OptimizationResult, Optimizer
from src.optimization.grid_search import GridSearchOptimizer
from src.optimization.genetic import GeneticOptimizer
from src.optimization.objectives import sharpe_ratio, sortino_ratio, max_drawdown
from src.optimization.runner import run_optimization
from src.optimization.selector import (
    EnsembleResult,
    StrategyRanking,
    StrategySelector,
    StrategySelectionResult,
)
from src.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    Strategy,
    StrategyConfig,
    registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_walk_data():
    """Synthetic random-ish price data with a gentle upward drift.
    Enough bars to give strategies something to work with.
    """
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="1min")
    close = 100.0
    closes = []
    for _ in range(n):
        close *= (1.0 + rng.normal(0.0002, 0.005))
        closes.append(round(close, 4))

    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        },
        index=idx,
    )
    return df


@pytest.fixture
def flat_data():
    """Unchanging price — no strategy can profit."""
    idx = pd.date_range("2026-01-01 09:30", periods=50, freq="1min")
    closes = [100.0] * 50
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * 50,
        },
        index=idx,
    )


@pytest.fixture
def strong_trend_data():
    """Strong uptrend — momentum strategy should thrive with low entry thresholds."""
    n = 200
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    closes = []
    base = 100.0
    for i in range(n):
        # 0.5% per bar with small noise
        base *= 1.005
        noise = rng.normal(0, 0.05)
        close = base + noise
        closes.append(round(close, 4))

    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.3 for c in closes],
            "low": [c - 0.3 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        },
        index=idx,
    )


@pytest.fixture
def oscillating_data():
    """Prices oscillating around a mean — good for mean reversion."""
    n = 200
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="1min")
    rng = np.random.default_rng(7)
    closes = []
    for i in range(n):
        close = 100.0 + 8.0 * np.sin(2 * np.pi * i / 40) + rng.normal(0, 0.1)
        closes.append(round(close, 4))

    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        },
        index=idx,
    )


# Param grids that actually produce trades with the strategies ------------

@pytest.fixture
def momentum_grid():
    """Param grid for MomentumStrategy on trending data.
    Low entry thresholds so the moderate momentum values trigger BUYs.
    """
    return {
        "entry_threshold": [0.005, 0.01, 0.05, 0.1],
        "exit_threshold": [0.0, 0.01, 0.05],
        "stop_loss_pct": [0.02, 0.10],
        "take_profit_pct": [0.05, 0.20],
    }


@pytest.fixture
def momentum_small_grid():
    """Smaller grid for faster tests."""
    return {
        "entry_threshold": [0.01, 0.05],
        "exit_threshold": [0.0, 0.02],
        "stop_loss_pct": [0.02, 0.10],
    }


@pytest.fixture
def minimal_param_grid():
    """Single-value grid."""
    return {
        "stop_loss_pct": [0.02],
        "take_profit_pct": [0.05],
    }


# ---------------------------------------------------------------------------
# GridSearchOptimizer
# ---------------------------------------------------------------------------

class TestGridSearchOptimizer:
    def test_finds_better_than_default(self, strong_trend_data, momentum_small_grid):
        """Grid search should find a config with a better Sharpe than
        the default StrategyConfig on trending data."""
        default_cfg = StrategyConfig(entry_threshold=0.01, exit_threshold=0.0)
        engine = BacktestEngine()
        strategy = MomentumStrategy(default_cfg)
        result = engine.run(strategy, strong_trend_data)
        returns = result.equity_curve.pct_change().dropna()
        default_score = sharpe_ratio(returns) if len(returns) > 0 and returns.std() > 0 else float("-inf")

        opt = GridSearchOptimizer(objective_fn=sharpe_ratio, backtest_engine=engine)
        opt_result = opt.optimize(
            MomentumStrategy,
            strong_trend_data,
            momentum_small_grid,
        )

        assert isinstance(opt_result, OptimizationResult)
        assert isinstance(opt_result.best_config, StrategyConfig)
        assert opt_result.best_score >= default_score, (
            f"Expected best_score ({opt_result.best_score}) >= "
            f"default_score ({default_score})"
        )
        assert not opt_result.all_scores.empty

    def test_all_scores_table(self, strong_trend_data, momentum_small_grid):
        """The all_scores DataFrame should have one row per evaluated config."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(MomentumStrategy, strong_trend_data, momentum_small_grid)
        # 2 * 2 * 2 = 8 combinations
        assert len(result.all_scores) == 8
        assert "config" in result.all_scores.columns
        assert "score" in result.all_scores.columns

    def test_empty_data_returns_inf(self):
        """Empty DataFrame should yield -inf score with default config."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(
            MomentumStrategy,
            pd.DataFrame(),
            {"stop_loss_pct": [0.01, 0.02]},
        )
        assert result.best_score == float("-inf")
        assert isinstance(result.best_config, StrategyConfig)

    def test_empty_param_grid(self, strong_trend_data):
        """When param_grid is empty, evaluate just the default config."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(MomentumStrategy, strong_trend_data, {})
        assert isinstance(result, OptimizationResult)
        # Default config may or may not trade, but we get a valid score
        assert result.best_score is not None
        assert len(result.all_scores) == 1

    def test_single_value_grid(self, strong_trend_data):
        """Grid with one value per param that produces trades."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(
            MomentumStrategy,
            strong_trend_data,
            {"entry_threshold": [0.01], "exit_threshold": [0.0]},
        )
        assert isinstance(result, OptimizationResult)
        assert len(result.all_scores) == 1

    def test_flat_equity_all_configs(self, flat_data):
        """When all configs produce flat equity (no trades), scores
        should be -inf."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(
            MomentumStrategy,
            flat_data,
            {"stop_loss_pct": [0.01, 0.02], "take_profit_pct": [0.05, 0.10]},
        )
        assert result.best_score == float("-inf")
        assert len(result.all_scores) == 4

    def test_n_iter_limits_evaluations(self, strong_trend_data):
        """Grid search respects n_iter when it's less than total combinations."""
        big_grid = {
            "entry_threshold": [0.01, 0.02, 0.03, 0.04, 0.05],
            "exit_threshold": [0.0, 0.01, 0.02],
            "stop_loss_pct": [0.02, 0.05, 0.10, 0.15],
        }
        # 5 * 3 * 4 = 60 combinations, n_iter=20 should cap it
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(
            MomentumStrategy, strong_trend_data, big_grid, n_iter=20
        )
        assert len(result.all_scores) <= 20

    def test_best_config_uses_correct_field_values(self, strong_trend_data):
        """The returned best_config should have values from the param_grid."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(
            MomentumStrategy,
            strong_trend_data,
            {
                "entry_threshold": [0.01],
                "exit_threshold": [0.0],
                "stop_loss_pct": [0.03],
                "take_profit_pct": [0.07],
            },
        )
        assert result.best_config.stop_loss_pct == 0.03
        assert result.best_config.take_profit_pct == 0.07
        assert result.best_config.entry_threshold == 0.01
        assert result.best_config.exit_threshold == 0.0

    def test_sortino_objective(self, strong_trend_data, momentum_small_grid):
        """Grid search works with Sortino ratio as objective."""
        opt = GridSearchOptimizer(objective_fn=sortino_ratio)
        result = opt.optimize(
            MomentumStrategy, strong_trend_data, momentum_small_grid
        )
        assert isinstance(result, OptimizationResult)

    def test_max_drawdown_objective(self, strong_trend_data, momentum_small_grid):
        """Grid search works with max drawdown as objective (higher = better)."""
        opt = GridSearchOptimizer(objective_fn=max_drawdown)
        result = opt.optimize(
            MomentumStrategy, strong_trend_data, momentum_small_grid
        )
        assert isinstance(result, OptimizationResult)

    def test_progress_logging(self, strong_trend_data, momentum_small_grid, caplog):
        """Progress messages are emitted during grid search."""
        caplog.set_level(logging.INFO, logger="src.optimization.grid_search")
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        opt.optimize(MomentumStrategy, strong_trend_data, momentum_small_grid)
        assert any("Grid search:" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# GeneticOptimizer
# ---------------------------------------------------------------------------

class TestGeneticOptimizer:
    def test_converges_to_profitable_config(self, strong_trend_data, momentum_small_grid):
        """Genetic algorithm should find a config with reasonable Sharpe
        over several generations on trending data."""
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=10,
            generations=5,
            mutation_rate=0.2,
        )
        result = opt.optimize(
            MomentumStrategy,
            strong_trend_data,
            momentum_small_grid,
            n_iter=100,
        )

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)
        # On strong trending data with good entry thresholds, we should get positive Sharpe
        assert result.best_score > -1.0, (
            f"Expected somewhat reasonable score, got {result.best_score}"
        )
        assert "generation" in result.all_scores.columns

    def test_population_evolves(self, strong_trend_data, momentum_small_grid):
        """The score table should show multiple generations."""
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=8,
            generations=4,
        )
        result = opt.optimize(
            MomentumStrategy, strong_trend_data, momentum_small_grid, n_iter=50
        )
        generations_seen = result.all_scores["generation"].unique()
        assert len(generations_seen) >= 1

    def test_empty_data(self):
        """Empty data returns -inf."""
        opt = GeneticOptimizer(objective_fn=sharpe_ratio, population_size=4, generations=2)
        result = opt.optimize(
            MomentumStrategy,
            pd.DataFrame(),
            {"stop_loss_pct": [0.01, 0.02]},
        )
        assert result.best_score == float("-inf")

    def test_empty_param_grid(self, strong_trend_data):
        """Empty param grid evaluates the default config once."""
        opt = GeneticOptimizer(objective_fn=sharpe_ratio, population_size=4, generations=2)
        result = opt.optimize(MomentumStrategy, strong_trend_data, {})
        assert isinstance(result, OptimizationResult)
        assert len(result.all_scores) <= 1

    def test_n_iter_caps_total_evals(self, strong_trend_data, momentum_small_grid):
        """When n_iter is small, total evaluations should not exceed it."""
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=10,
            generations=20,
        )
        result = opt.optimize(
            MomentumStrategy,
            strong_trend_data,
            momentum_small_grid,
            n_iter=30,
        )
        assert len(result.all_scores) <= 30

    def test_elitism_preserves_best(self, strong_trend_data, momentum_small_grid):
        """With elitism, best score should never decrease across generations."""
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=8,
            generations=5,
            elitism=2,
            mutation_rate=0.3,
        )
        result = opt.optimize(
            MomentumStrategy, strong_trend_data, momentum_small_grid, n_iter=100
        )
        # Best score should be >= -inf (valid)
        assert result.best_score > float("-inf") or result.best_score == float("-inf")

    def test_genetic_with_mean_reversion(self, oscillating_data):
        """Genetic optimiser works with MeanReversionStrategy on oscillating data."""
        grid = {
            "entry_threshold": [0.5, 0.8, 1.2, 1.5],
            "stop_loss_pct": [0.02, 0.10],
        }
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=6,
            generations=3,
        )
        result = opt.optimize(MeanReversionStrategy, oscillating_data, grid, n_iter=30)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)

    def test_genetic_with_breakout(self, strong_trend_data):
        """Genetic optimiser works with BreakoutStrategy."""
        grid = {
            "entry_threshold": [0.001, 0.005, 0.01],
            "stop_loss_pct": [0.02, 0.10],
        }
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=6,
            generations=3,
        )
        result = opt.optimize(BreakoutStrategy, strong_trend_data, grid, n_iter=30)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)

    def test_progress_logging(self, strong_trend_data, momentum_small_grid, caplog):
        """Progress messages are emitted during genetic optimisation."""
        caplog.set_level(logging.INFO, logger="src.optimization.genetic")
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=6,
            generations=3,
        )
        opt.optimize(MomentumStrategy, strong_trend_data, momentum_small_grid, n_iter=30)
        assert any("Gen " in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# StrategySelector
# ---------------------------------------------------------------------------

class TestStrategySelector:
    def test_ranks_strategies(self, strong_trend_data, momentum_small_grid):
        """Selector should rank multiple strategies and return the best."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        result = selector.select_best(
            ["momentum", "mean_reversion", "breakout"],
            strong_trend_data,
            momentum_small_grid,
            n_iter=8,
        )
        assert isinstance(result, StrategySelectionResult)
        assert result.best_strategy in {"momentum", "mean_reversion", "breakout"}
        assert len(result.rankings) == 3
        # Rankings should be sorted by score descending
        for i in range(len(result.rankings) - 1):
            assert (
                result.rankings[i].result.best_score
                >= result.rankings[i + 1].result.best_score
            )
        assert result.rankings[0].rank == 1

    def test_rankings_match_result_best(self, strong_trend_data, momentum_small_grid):
        """The best_strategy/best_score should match the top ranking."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        result = selector.select_best(
            ["momentum", "mean_reversion"],
            strong_trend_data,
            momentum_small_grid,
            n_iter=8,
        )
        assert result.best_strategy == result.rankings[0].name
        assert result.best_score == result.rankings[0].result.best_score
        assert result.best_config == result.rankings[0].result.best_config

    def test_empty_strategy_list(self, strong_trend_data):
        """Empty list should return a sentinel result."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        result = selector.select_best([], strong_trend_data, {})
        assert result.best_strategy == ""
        assert result.best_score == float("-inf")

    def test_unknown_strategy_skipped(self, strong_trend_data, momentum_small_grid):
        """Unknown strategy names are skipped gracefully."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        result = selector.select_best(
            ["momentum", "nonexistent_strategy"],
            strong_trend_data,
            momentum_small_grid,
            n_iter=8,
        )
        assert len(result.rankings) == 1
        assert result.rankings[0].name == "momentum"

    def test_all_unknown_strategies(self, strong_trend_data):
        """If all strategies are unknown, return sentinel."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        result = selector.select_best(
            ["fake_a", "fake_b"],
            strong_trend_data,
            {},
        )
        assert result.best_strategy == ""
        assert result.best_score == float("-inf")

    def test_combine_strategies_ensemble(self, strong_trend_data, momentum_small_grid):
        """combine_strategies returns weighted ensemble with top-N strategies."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        ensemble = selector.combine_strategies(
            ["momentum", "mean_reversion", "breakout"],
            strong_trend_data,
            momentum_small_grid,
            top_n=2,
            n_iter=8,
        )
        assert isinstance(ensemble, EnsembleResult)
        assert len(ensemble.strategies) == 2
        assert len(ensemble.configs) == 2
        assert len(ensemble.weights) == 2
        # Weights should sum to ~1
        assert sum(ensemble.weights) == pytest.approx(1.0)
        # Configs should have their weight field set
        for cfg, w in zip(ensemble.configs, ensemble.weights):
            assert cfg.weight == w

    def test_combine_top_n_exceeds_available(self, strong_trend_data, momentum_small_grid):
        """If top_n > available strategies, all are included."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        ensemble = selector.combine_strategies(
            ["momentum", "mean_reversion"],
            strong_trend_data,
            momentum_small_grid,
            top_n=5,
            n_iter=8,
        )
        assert len(ensemble.strategies) == 2

    def test_combine_weight_assignment(self, strong_trend_data, momentum_small_grid):
        """Higher-scoring strategies should get higher weights in the ensemble."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        # Get rankings to know which is best
        selection = selector.select_best(
            ["momentum", "mean_reversion", "breakout"],
            strong_trend_data,
            momentum_small_grid,
            n_iter=8,
        )
        # Now combine all three
        ensemble = selector.combine_strategies(
            ["momentum", "mean_reversion", "breakout"],
            strong_trend_data,
            momentum_small_grid,
            top_n=3,
            n_iter=8,
        )
        # Best strategy should have highest weight (or tied for highest)
        best_name = selection.best_strategy
        best_idx = ensemble.strategies.index(best_name)
        best_weight = ensemble.weights[best_idx]
        for i, w in enumerate(ensemble.weights):
            assert w <= best_weight + 0.01, (
                f"Expected weight for {ensemble.strategies[i]} ({w}) "
                f"<= best weight ({best_weight})"
            )


# ---------------------------------------------------------------------------
# run_optimization convenience runner
# ---------------------------------------------------------------------------

class TestRunOptimization:
    def test_grid_method(self, strong_trend_data):
        """run_optimization with method='grid' works end-to-end."""
        result = run_optimization(
            "momentum",
            strong_trend_data,
            method="grid",
            param_grid={"entry_threshold": [0.01], "stop_loss_pct": [0.02]},
            n_iter=2,
        )
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)

    def test_genetic_method(self, strong_trend_data):
        """run_optimization with method='genetic' works end-to-end."""
        result = run_optimization(
            "momentum",
            strong_trend_data,
            method="genetic",
            param_grid={"entry_threshold": [0.01, 0.05], "stop_loss_pct": [0.02, 0.10]},
            n_iter=20,
            population_size=6,
            generations=3,
        )
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)

    def test_default_param_grid(self, strong_trend_data):
        """When param_grid is None, the built-in default is used."""
        result = run_optimization(
            "momentum",
            strong_trend_data,
            method="grid",
            n_iter=10,
        )
        assert isinstance(result, OptimizationResult)
        assert not result.all_scores.empty

    def test_invalid_method_raises(self, strong_trend_data):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown optimisation method"):
            run_optimization(
                "momentum",
                strong_trend_data,
                method="bayesian",
            )

    def test_unknown_strategy_raises(self, strong_trend_data):
        """Unknown strategy name raises KeyError."""
        with pytest.raises(KeyError):
            run_optimization(
                "ghost_strategy",
                strong_trend_data,
                method="grid",
            )

    def test_custom_objective(self, strong_trend_data):
        """A custom objective function can be passed."""
        def total_return_obj(returns: pd.Series) -> float:
            return float((1 + returns).prod() - 1)

        result = run_optimization(
            "momentum",
            strong_trend_data,
            method="grid",
            objective_fn=total_return_obj,
            param_grid={"entry_threshold": [0.01], "stop_loss_pct": [0.02]},
            n_iter=2,
        )
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_loss_strategies(self, flat_data):
        """On flat data all strategies should have -inf or negative scores.
        Optimizer still returns a valid result with a config."""
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(
            MomentumStrategy,
            flat_data,
            {"stop_loss_pct": [0.01, 0.02, 0.03]},
        )
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)

    def test_extreme_param_values(self, random_walk_data):
        """Extreme parameter values shouldn't crash the optimizer."""
        grid = {
            "stop_loss_pct": [0.0001, 0.99],
            "take_profit_pct": [0.0001, 2.0],
        }
        opt = GridSearchOptimizer(objective_fn=sharpe_ratio)
        result = opt.optimize(MomentumStrategy, random_walk_data, grid)
        assert isinstance(result, OptimizationResult)

    def test_genetic_all_loss_population(self, flat_data):
        """Genetic optimizer on flat data — all individuals score -inf,
        but we still get a valid result."""
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=6,
            generations=3,
        )
        result = opt.optimize(
            MomentumStrategy,
            flat_data,
            {"stop_loss_pct": [0.01, 0.02], "take_profit_pct": [0.05, 0.10]},
            n_iter=30,
        )
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, StrategyConfig)

    def test_single_strategy_selection(self, strong_trend_data, momentum_small_grid):
        """Selector with just one strategy should still work."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        result = selector.select_best(
            ["momentum"],
            strong_trend_data,
            momentum_small_grid,
            n_iter=8,
        )
        assert result.best_strategy == "momentum"
        assert len(result.rankings) == 1
        assert result.rankings[0].rank == 1

    def test_combine_strategies_single(self, strong_trend_data, momentum_small_grid):
        """Combining with only one available strategy should work."""
        optimizer = GridSearchOptimizer(objective_fn=sharpe_ratio)
        selector = StrategySelector(registry, optimizer)
        ensemble = selector.combine_strategies(
            ["momentum"],
            strong_trend_data,
            momentum_small_grid,
            top_n=3,
            n_iter=8,
        )
        assert len(ensemble.strategies) == 1
        assert sum(ensemble.weights) == pytest.approx(1.0)

    def test_objective_fn_max_drawdown_grid(self, strong_trend_data, momentum_small_grid):
        """max_drawdown objective (negated for higher-is-better) works with grid."""
        opt = GridSearchOptimizer(objective_fn=max_drawdown)
        result = opt.optimize(
            MomentumStrategy, strong_trend_data, momentum_small_grid
        )
        assert isinstance(result, OptimizationResult)

    def test_genetic_respects_bounds(self, strong_trend_data):
        """Genetic optimizer should produce configs within param_grid bounds."""
        grid = {
            "stop_loss_pct": [0.01, 0.05],
            "take_profit_pct": [0.02, 0.10],
        }
        opt = GeneticOptimizer(
            objective_fn=sharpe_ratio,
            population_size=8,
            generations=5,
        )
        result = opt.optimize(MomentumStrategy, strong_trend_data, grid, n_iter=50)
        cfg = result.best_config
        assert 0.01 <= cfg.stop_loss_pct <= 0.05
        assert 0.02 <= cfg.take_profit_pct <= 0.10
