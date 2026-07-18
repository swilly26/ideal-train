"""Tests for the backtesting engine."""

import pandas as pd
import numpy as np

from src.backtesting import BacktestEngine, BacktestResult
from src.strategies import Strategy, Signal, SignalType


class NoOpStrategy(Strategy):
    """Strategy that generates no signals."""

    def generate_signals(self, data):
        return []


class TestBacktestResult:
    def test_result_has_equity_curve(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1min")
        curve = pd.Series(100_000.0, index=idx)
        result = BacktestResult(equity_curve=curve)
        assert len(result.equity_curve) == 5
        assert result.metrics == {}


class TestBacktestEngine:
    def test_run_returns_result(self):
        engine = BacktestEngine(initial_capital=50_000.0)
        idx = pd.date_range("2026-01-01", periods=10, freq="1min")
        data = pd.DataFrame(
            {"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
            index=idx,
        )
        strategy = NoOpStrategy()
        result = engine.run(strategy, data)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity_curve, pd.Series)
        assert result.metrics["sharpe"] == 0.0

    def test_default_initial_capital(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 100_000.0
        assert engine.commission == 0.0

    def test_custom_commission(self):
        engine = BacktestEngine(commission=0.001)
        assert engine.commission == 0.001
