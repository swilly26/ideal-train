"""Tests for the optimisation module."""

import numpy as np
import pandas as pd
import pytest

from src.optimization import OptimizationResult, Optimizer, sharpe_ratio, sortino_ratio, max_drawdown
from src.strategies import Strategy, StrategyConfig


class DummyOptimizer(Optimizer):
    """An optimiser that just returns the first config in the grid."""

    def optimize(self, strategy_cls, data, param_grid, n_iter=100):
        cfg = StrategyConfig()
        for key, values in param_grid.items():
            setattr(cfg, key, values[0])
        return OptimizationResult(best_config=cfg, best_score=1.5)


class TestObjectiveFunctions:
    def test_sharpe_positive_returns(self):
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02])
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_flat_returns(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        assert sharpe_ratio(returns) == 0.0

    def test_sortino_positive(self):
        returns = pd.Series([0.01, -0.005, 0.02, 0.01, -0.002])
        sr = sortino_ratio(returns)
        assert sr > 0

    def test_max_drawdown(self):
        returns = pd.Series([0.01, -0.10, 0.02])
        mdd = max_drawdown(returns)
        assert mdd > 0  # positive because we negated it
        assert mdd == pytest.approx(0.10, abs=0.02)


class TestOptimizer:
    def test_dummy_optimizer_returns_result(self):
        opt = DummyOptimizer(objective_fn=sharpe_ratio)
        idx = pd.date_range("2026-01-01", periods=10, freq="1min")
        data = pd.DataFrame({"close": np.linspace(100, 110, 10)}, index=idx)
        result = opt.optimize(
            strategy_cls=Strategy,
            data=data,
            param_grid={"stop_loss_pct": [0.01, 0.02]},
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_score == 1.5
        assert isinstance(result.best_config, StrategyConfig)
