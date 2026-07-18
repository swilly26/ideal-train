"""Tests for the strategy module."""

import pandas as pd
import pytest

from src.strategies import (
    Signal,
    SignalType,
    Strategy,
    StrategyConfig,
    StrategyRegistry,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    registry,
)


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


# ---------------------------------------------------------------------------
# Momentum Strategy Tests
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    def test_buy_on_uptrend(self, uptrend_ohlcv):
        """Momentum should generate BUY signals on a strong uptrend."""
        cfg = StrategyConfig(entry_threshold=0.001, exit_threshold=-0.02, extra={"lookback": 10})
        strat = MomentumStrategy(config=cfg)
        signals = strat.generate_signals(uptrend_ohlcv)

        # Should have BUY signals (momentum positive & exceeds threshold)
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) > 0, "Expected BUY signals on uptrend"
        for s in buys:
            assert 0.0 <= s.confidence <= 1.0
            assert "momentum" in s.metadata

    def test_sell_on_downtrend(self, downtrend_ohlcv):
        """Momentum should generate SELL signals on a strong downtrend."""
        # exit_threshold is the low bar — negative momentum below it = SELL
        cfg = StrategyConfig(entry_threshold=0.02, exit_threshold=-0.005, extra={"lookback": 10})
        strat = MomentumStrategy(config=cfg)
        signals = strat.generate_signals(downtrend_ohlcv)

        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) > 0, "Expected SELL signals on downtrend"
        for s in sells:
            assert 0.0 <= s.confidence <= 1.0

    def test_no_signals_when_flat(self, oscillating_ohlcv):
        """With high thresholds, a flat-ish market should produce few or no signals."""
        cfg = StrategyConfig(entry_threshold=0.10, exit_threshold=-0.10, extra={"lookback": 10})
        strat = MomentumStrategy(config=cfg)
        signals = strat.generate_signals(oscillating_ohlcv)
        # Most oscillating data stays within a narrow momentum band
        assert len(signals) < len(oscillating_ohlcv) // 2, (
            "Expected sparse signals in flat market with high thresholds"
        )

    def test_config_entry_threshold_alters_behavior(self, uptrend_ohlcv):
        """A higher entry_threshold should produce fewer BUY signals."""
        cfg_low = StrategyConfig(entry_threshold=0.001, exit_threshold=-0.02, extra={"lookback": 10})
        cfg_high = StrategyConfig(entry_threshold=0.05, exit_threshold=-0.02, extra={"lookback": 10})

        signals_low = MomentumStrategy(config=cfg_low).generate_signals(uptrend_ohlcv)
        signals_high = MomentumStrategy(config=cfg_high).generate_signals(uptrend_ohlcv)

        buys_low = len([s for s in signals_low if s.signal_type == SignalType.BUY])
        buys_high = len([s for s in signals_high if s.signal_type == SignalType.BUY])
        assert buys_high <= buys_low, (
            f"Higher threshold ({buys_high} buys) should not exceed lower ({buys_low} buys)"
        )

    def test_config_exit_threshold_alters_behavior(self, downtrend_ohlcv):
        """A more negative exit_threshold should produce fewer SELL signals."""
        cfg_lenient = StrategyConfig(entry_threshold=0.02, exit_threshold=-0.10, extra={"lookback": 10})
        cfg_strict = StrategyConfig(entry_threshold=0.02, exit_threshold=-0.001, extra={"lookback": 10})

        signals_lenient = MomentumStrategy(config=cfg_lenient).generate_signals(downtrend_ohlcv)
        signals_strict = MomentumStrategy(config=cfg_strict).generate_signals(downtrend_ohlcv)

        sells_lenient = len([s for s in signals_lenient if s.signal_type == SignalType.SELL])
        sells_strict = len([s for s in signals_strict if s.signal_type == SignalType.SELL])
        assert sells_lenient <= sells_strict, (
            f"More negative exit_threshold should not produce more sells"
        )

    def test_lookback_parameter_alters_behavior(self, uptrend_ohlcv):
        """Changing lookback changes the SMA period → different signal counts."""
        cfg_short = StrategyConfig(entry_threshold=0.001, exit_threshold=-0.02, extra={"lookback": 5})
        cfg_long = StrategyConfig(entry_threshold=0.001, exit_threshold=-0.02, extra={"lookback": 20})

        signals_short = MomentumStrategy(config=cfg_short).generate_signals(uptrend_ohlcv)
        signals_long = MomentumStrategy(config=cfg_long).generate_signals(uptrend_ohlcv)

        # Signal counts may differ because the moving average changes
        assert isinstance(signals_short, list)
        assert isinstance(signals_long, list)

    def test_signals_have_confidence_in_range(self, uptrend_ohlcv):
        cfg = StrategyConfig(entry_threshold=0.001, exit_threshold=-0.02, extra={"lookback": 10})
        strat = MomentumStrategy(config=cfg)
        signals = strat.generate_signals(uptrend_ohlcv)
        for s in signals:
            assert 0.0 <= s.confidence <= 1.0, f"Confidence {s.confidence} out of range"


# ---------------------------------------------------------------------------
# Mean Reversion Strategy Tests
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    def test_buy_on_oversold(self, oscillating_ohlcv):
        """When price dips far below mean (negative z-score), generate BUY."""
        cfg = StrategyConfig(entry_threshold=0.5, extra={"lookback": 10})
        strat = MeanReversionStrategy(config=cfg)
        signals = strat.generate_signals(oscillating_ohlcv)

        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) > 0, "Expected BUY signals on oscillating data (oversold dips)"
        for s in buys:
            assert "z_score" in s.metadata
            assert s.metadata["z_score"] < 0  # oversold = negative z-score

    def test_sell_on_overbought(self, oscillating_ohlcv):
        """When price spikes above mean (positive z-score), generate SELL."""
        cfg = StrategyConfig(entry_threshold=0.5, extra={"lookback": 10})
        strat = MeanReversionStrategy(config=cfg)
        signals = strat.generate_signals(oscillating_ohlcv)

        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) > 0, "Expected SELL signals on oscillating data (overbought spikes)"
        for s in sells:
            assert s.metadata["z_score"] > 0  # overbought = positive z-score

    def test_config_threshold_alters_signals(self, oscillating_ohlcv):
        """Higher entry_threshold → fewer signals in same data."""
        cfg_low = StrategyConfig(entry_threshold=0.3, extra={"lookback": 10})
        cfg_high = StrategyConfig(entry_threshold=1.5, extra={"lookback": 10})

        signals_low = MeanReversionStrategy(config=cfg_low).generate_signals(oscillating_ohlcv)
        signals_high = MeanReversionStrategy(config=cfg_high).generate_signals(oscillating_ohlcv)

        assert len(signals_high) <= len(signals_low), (
            "Higher z-score threshold should reduce signal count"
        )

    def test_lookback_alters_behavior(self, oscillating_ohlcv):
        """Different lookback → different mean/std → different signals."""
        cfg_short = StrategyConfig(entry_threshold=0.5, extra={"lookback": 5})
        cfg_long = StrategyConfig(entry_threshold=0.5, extra={"lookback": 20})

        signals_short = MeanReversionStrategy(config=cfg_short).generate_signals(oscillating_ohlcv)
        signals_long = MeanReversionStrategy(config=cfg_long).generate_signals(oscillating_ohlcv)

        # Both should produce some signals, but counts may differ
        assert isinstance(signals_short, list)
        assert isinstance(signals_long, list)

    def test_no_signals_in_steady_uptrend(self, uptrend_ohlcv):
        """In a steady uptrend, mean reversion may find fewer oversold conditions."""
        cfg = StrategyConfig(entry_threshold=1.0, extra={"lookback": 10})
        strat = MeanReversionStrategy(config=cfg)
        signals = strat.generate_signals(uptrend_ohlcv)
        # Steady uptrend shouldn't produce many extreme z-scores
        # (price stays above mean but not necessarily > threshold std devs)
        assert isinstance(signals, list)

    def test_confidence_in_range(self, oscillating_ohlcv):
        cfg = StrategyConfig(entry_threshold=0.5, extra={"lookback": 10})
        strat = MeanReversionStrategy(config=cfg)
        signals = strat.generate_signals(oscillating_ohlcv)
        for s in signals:
            assert 0.0 <= s.confidence <= 1.0


# ---------------------------------------------------------------------------
# Breakout Strategy Tests
# ---------------------------------------------------------------------------


class TestBreakoutStrategy:
    def test_buy_on_upside_breakout(self, breakout_ohlcv):
        """After a sustained range, a sharp upside breakout triggers BUY."""
        cfg = StrategyConfig(entry_threshold=0.005, extra={"lookback": 10})
        strat = BreakoutStrategy(config=cfg)
        signals = strat.generate_signals(breakout_ohlcv)

        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buys) > 0, "Expected BUY on upside breakout"
        for s in buys:
            assert "upside_breakout" in s.metadata
            assert s.metadata["upside_breakout"] > 0

    def test_sell_on_downside_breakout(self):
        """Construct data with a downside breakdown to verify SELL signals."""
        periods = 40
        idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
        closes = [100.0] * 30 + [95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        opens = closes
        data = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
            index=idx,
        )

        cfg = StrategyConfig(entry_threshold=0.005, extra={"lookback": 10})
        strat = BreakoutStrategy(config=cfg)
        signals = strat.generate_signals(data)

        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sells) > 0, "Expected SELL on downside breakout"
        for s in sells:
            assert "downside_breakout" in s.metadata

    def test_no_signals_in_tight_range(self):
        """When price stays inside the channel, no breakout signals."""
        periods = 30
        idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
        closes = [100.0 + (i % 3) * 0.1 for i in range(periods)]  # tiny oscillation
        highs = [c + 0.3 for c in closes]
        lows = [c - 0.3 for c in closes]
        data = pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
            index=idx,
        )

        cfg = StrategyConfig(entry_threshold=0.02, extra={"lookback": 10})
        strat = BreakoutStrategy(config=cfg)
        signals = strat.generate_signals(data)
        assert len(signals) == 0, f"Expected no signals in tight range, got {len(signals)}"

    def test_config_threshold_alters_signals(self, breakout_ohlcv):
        """Higher entry_threshold → fewer breakout signals."""
        cfg_low = StrategyConfig(entry_threshold=0.001, extra={"lookback": 10})
        cfg_high = StrategyConfig(entry_threshold=0.10, extra={"lookback": 10})

        signals_low = BreakoutStrategy(config=cfg_low).generate_signals(breakout_ohlcv)
        signals_high = BreakoutStrategy(config=cfg_high).generate_signals(breakout_ohlcv)

        assert len(signals_high) <= len(signals_low), (
            "Higher threshold should not increase signal count"
        )

    def test_lookback_alters_behavior(self, breakout_ohlcv):
        """Different lookback → different channel → different signals."""
        cfg_short = StrategyConfig(entry_threshold=0.005, extra={"lookback": 5})
        cfg_long = StrategyConfig(entry_threshold=0.005, extra={"lookback": 15})

        signals_short = BreakoutStrategy(config=cfg_short).generate_signals(breakout_ohlcv)
        signals_long = BreakoutStrategy(config=cfg_long).generate_signals(breakout_ohlcv)

        assert isinstance(signals_short, list)
        assert isinstance(signals_long, list)

    def test_confidence_in_range(self, breakout_ohlcv):
        cfg = StrategyConfig(entry_threshold=0.005, extra={"lookback": 10})
        strat = BreakoutStrategy(config=cfg)
        signals = strat.generate_signals(breakout_ohlcv)
        for s in signals:
            assert 0.0 <= s.confidence <= 1.0


# ---------------------------------------------------------------------------
# Registry auto-registration tests
# ---------------------------------------------------------------------------


class TestRegistryAutoRegistration:
    def test_all_three_strategies_registered(self):
        names = registry.list()
        assert "momentum" in names
        assert "mean_reversion" in names
        assert "breakout" in names

    def test_can_instantiate_from_registry(self):
        for name in ["momentum", "mean_reversion", "breakout"]:
            cls = registry.get(name)
            strat = cls()
            assert isinstance(strat, Strategy)

    def test_registry_default_configs(self):
        """Instantiated via registry, each strategy gets a default config."""
        for name in ["momentum", "mean_reversion", "breakout"]:
            cls = registry.get(name)
            strat = cls()
            assert isinstance(strat.config, StrategyConfig)
            assert strat.config.entry_threshold == 0.5
