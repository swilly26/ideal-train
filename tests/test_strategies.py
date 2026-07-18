"""Tests for the strategy module."""

import pandas as pd
import pytest

from src.strategies import Signal, SignalType, Strategy, StrategyConfig, StrategyRegistry


class MockStrategy(Strategy):
    """A strategy that always emits a single BUY signal."""

    def generate_signals(self, data):
        return [
            Signal(
                symbol="TEST",
                timestamp=data.index[0] if len(data) > 0 else pd.Timestamp.now(),
                signal_type=SignalType.BUY,
                confidence=0.9,
            )
        ]


class TestStrategyConfig:
    def test_default_values(self):
        cfg = StrategyConfig()
        assert cfg.entry_threshold == 0.5
        assert cfg.stop_loss_pct == 0.02
        assert cfg.max_position_pct == 0.10
        assert cfg.weight == 1.0

    def test_custom_values(self):
        cfg = StrategyConfig(
            stop_loss_pct=0.05,
            max_position_pct=0.20,
            extra={"foo": "bar"},
        )
        assert cfg.stop_loss_pct == 0.05
        assert cfg.max_position_pct == 0.20
        assert cfg.extra == {"foo": "bar"}


class TestSignal:
    def test_signal_creation(self):
        ts = pd.Timestamp("2026-01-01 10:00")
        sig = Signal(symbol="SPY", timestamp=ts, signal_type=SignalType.SELL, confidence=0.7)
        assert sig.symbol == "SPY"
        assert sig.signal_type == SignalType.SELL
        assert sig.confidence == 0.7


class TestStrategy:
    def test_strategy_accepts_config(self):
        cfg = StrategyConfig(stop_loss_pct=0.10)
        strat = MockStrategy(config=cfg)
        assert strat.config.stop_loss_pct == 0.10

    def test_strategy_default_config(self):
        strat = MockStrategy()
        assert isinstance(strat.config, StrategyConfig)
        assert strat.config.stop_loss_pct == 0.02

    def test_generate_signals_returns_list(self):
        strat = MockStrategy()
        idx = pd.date_range("2026-01-01", periods=5, freq="1min")
        data = pd.DataFrame({"close": [100, 101, 102, 101, 103]}, index=idx)
        signals = strat.generate_signals(data)
        assert isinstance(signals, list)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY


class TestStrategyRegistry:
    def test_register_and_get(self):
        registry = StrategyRegistry()
        registry.register("mock", MockStrategy)
        assert registry.get("mock") is MockStrategy

    def test_register_duplicate_raises(self):
        registry = StrategyRegistry()
        registry.register("mock", MockStrategy)
        with pytest.raises(KeyError, match="already registered"):
            registry.register("mock", MockStrategy)

    def test_get_missing_raises(self):
        registry = StrategyRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_strategies(self):
        registry = StrategyRegistry()
        registry.register("a", MockStrategy)
        registry.register("b", MockStrategy)
        names = registry.list()
        assert "a" in names
        assert "b" in names

    def test_clear(self):
        registry = StrategyRegistry()
        registry.register("x", MockStrategy)
        registry.clear()
        assert registry.list() == []
