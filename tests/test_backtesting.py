"""Tests for the backtesting engine."""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    BacktestEngine,
    BacktestResult,
    compute_metrics,
    max_drawdown as metric_max_dd,
    sharpe_ratio as metric_sharpe,
    sortino_ratio as metric_sortino,
    total_return as metric_total_return,
    win_rate as metric_win_rate,
    profit_factor as metric_profit_factor,
    num_trades as metric_num_trades,
)
from src.strategies import Signal, SignalType, Strategy, StrategyConfig


# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------


class NoOpStrategy(Strategy):
    """Strategy that produces no signals."""

    def generate_signals(self, data):
        return []


class FixedSignalStrategy(Strategy):
    """Produces a single BUY and later SELL at fixed indices."""

    def __init__(self, buy_idx=5, sell_idx=15, config=None):
        super().__init__(config)
        self.buy_idx = buy_idx
        self.sell_idx = sell_idx

    def generate_signals(self, data):
        signals = []
        for i in range(len(data)):
            ts = data.index[i]
            if i == self.buy_idx:
                signals.append(
                    Signal(symbol="TEST", timestamp=ts, signal_type=SignalType.BUY)
                )
            if i == self.sell_idx:
                signals.append(
                    Signal(symbol="TEST", timestamp=ts, signal_type=SignalType.SELL)
                )
        return signals


class CustomSignalStrategy(Strategy):
    """Strategy driven by an explicit list of (index, SignalType) tuples."""

    def __init__(self, plan, config=None):
        super().__init__(config)
        self.plan = plan  # list of (bar_index, SignalType)

    def generate_signals(self, data):
        signals = []
        for i, st in self.plan:
            if i < len(data):
                ts = data.index[i]
                signals.append(Signal(symbol="TEST", timestamp=ts, signal_type=st))
        return signals


# ---------------------------------------------------------------------------
# Synthetic OHLCV helpers
# ---------------------------------------------------------------------------


def _make_data(prices, freq="1min"):
    """Build an OHLCV DataFrame from a list of (open, high, low, close)."""
    n = len(prices)
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq=freq)
    opens, highs, lows, closes = zip(*prices)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000},
        index=idx,
    )


def _trending_up(n=20, start=100.0, step=1.0):
    """Steady uptrend with small spread."""
    prices = []
    for i in range(n):
        c = start + i * step
        prices.append((c - 0.1, c + 0.2, c - 0.2, c))
    return _make_data(prices)


def _trending_down(n=20, start=100.0, step=1.0):
    """Steady downtrend."""
    prices = []
    for i in range(n):
        c = start - i * step
        prices.append((c - 0.1, c + 0.2, c - 0.2, c))
    return _make_data(prices)


def _flat(n=20, level=100.0):
    """Absolutely flat prices (no volatility)."""
    prices = [(level, level, level, level) for _ in range(n)]
    return _make_data(prices)


def _volatile(n=40, seed=42):
    """Random walk with noise."""
    rng = np.random.default_rng(seed)
    c = 100.0
    prices = []
    for _ in range(n):
        c += rng.normal(0, 0.8)
        prices.append((c - 0.3, c + 0.5, c - 0.5, c))
    return _make_data(prices)


# ---------------------------------------------------------------------------
# Existing tests (kept as-is to avoid regressions)
# ---------------------------------------------------------------------------


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
        assert result.metrics["sharpe_ratio"] == 0.0

    def test_default_initial_capital(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 100_000.0
        assert engine.commission == 0.0

    def test_custom_commission(self):
        engine = BacktestEngine(commission=0.001)
        assert engine.commission == 0.001


# ---------------------------------------------------------------------------
# New tests — simulation behavior
# ---------------------------------------------------------------------------


class TestSimulationNoSignals:
    """When there are zero signals the equity curve should be flat."""

    def test_flat_equity_with_no_signals(self):
        engine = BacktestEngine(initial_capital=100_000)
        data = _trending_up(20)
        result = engine.run(NoOpStrategy(), data)
        # All equity values should equal initial_capital
        assert result.equity_curve.iloc[0] == 100_000.0
        assert (result.equity_curve == 100_000.0).all()
        assert result.trades.empty
        assert result.metrics["num_trades"] == 0
        assert result.metrics["total_return"] == 0.0

    def test_flat_equity_empty_data(self):
        engine = BacktestEngine()
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )
        result = engine.run(NoOpStrategy(), empty)
        assert len(result.equity_curve) == 1
        assert result.equity_curve.iloc[0] == 100_000.0
        assert result.trades.empty


class TestSimulationBasicRoundTrip:
    """BUY → SELL round-trip in trending markets."""

    def test_buy_then_sell_in_uptrend(self):
        """Buy early in uptrend, sell later — should make a profit."""
        engine = BacktestEngine(initial_capital=100_000)
        data = _trending_up(20, start=100.0, step=0.5)
        result = engine.run(FixedSignalStrategy(buy_idx=3, sell_idx=17), data)

        assert result.metrics["num_trades"] == 1
        assert result.metrics["total_return"] > 0  # profit
        assert result.trades.iloc[0]["pnl"] > 0
        assert len(result.equity_curve) == len(data) + 1  # initial + N bars

    def test_buy_then_sell_in_downtrend(self):
        """Buy early in downtrend — should lose money on the exit."""
        engine = BacktestEngine(initial_capital=100_000)
        data = _trending_down(20, start=100.0, step=0.5)
        result = engine.run(FixedSignalStrategy(buy_idx=3, sell_idx=17), data)

        assert result.metrics["num_trades"] == 1
        assert result.metrics["total_return"] < 0
        assert result.trades.iloc[0]["pnl"] < 0

    def test_equity_follows_price_while_in_position(self):
        """While holding a position, equity should track mark-to-market."""
        engine = BacktestEngine()
        data = _trending_up(10, start=100.0, step=1.0)
        # Buy at bar 2, sell at bar 8
        result = engine.run(FixedSignalStrategy(buy_idx=2, sell_idx=8), data)

        # Equity increases from bar 2 to bar 8 because price rises
        eq = result.equity_curve
        # Bar 2 corresponds to index position 3 in equity_curve (0=initial, 1=bar0, 2=bar1, 3=bar2)
        eq_at_entry = eq.iloc[3]
        eq_at_exit = eq.iloc[9]  # bar 8 → index 9
        assert eq_at_exit > eq_at_entry


class TestStopLoss:
    """Stop-loss behaviour."""

    def test_stop_loss_triggers_on_low(self):
        """When bar low crosses the stop-loss level, exit at stop price."""
        config = StrategyConfig(stop_loss_pct=0.03, take_profit_pct=0.50)
        # Buy at bar 2 (price ~102), then bar 4 has a low deep enough to trigger SL
        prices = [
            (100.0, 100.2, 99.8, 100.0),  # bar 0
            (101.0, 101.2, 100.8, 101.0),  # bar 1
            (102.0, 102.2, 101.8, 102.0),  # bar 2 ← BUY at close=102
            (101.0, 101.2, 100.8, 101.0),  # bar 3
            (100.0, 100.2, 98.5, 99.0),  # bar 4 ← low=98.5 < 102*0.97=98.94
        ]
        data = _make_data(prices)

        strategy = CustomSignalStrategy([(2, SignalType.BUY)], config=config)
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 1
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == "stop_loss"
        # Exit price should be the stop-loss price: 102 * 0.97 = 98.94
        assert trade["exit_price"] == pytest.approx(102.0 * 0.97, rel=1e-6)
        assert trade["pnl"] < 0  # lost money

    def test_stop_loss_not_triggered_if_low_above_sl(self):
        """Stop-loss should NOT trigger when low stays above stop level."""
        config = StrategyConfig(stop_loss_pct=0.02)
        prices = [
            (100.0, 100.2, 99.8, 100.0),  # bar 0
            (101.0, 101.2, 100.8, 101.0),  # bar 1
            (102.0, 102.2, 101.8, 102.0),  # bar 2 ← BUY at 102 (SL = 99.96)
            (101.5, 101.7, 100.5, 101.5),  # bar 3 ← low=100.5 > 99.96 → no SL
            (102.5, 102.7, 101.5, 102.5),  # bar 4 ← no SL
        ]
        data = _make_data(prices)

        strategy = CustomSignalStrategy(
            [(2, SignalType.BUY), (4, SignalType.SELL)], config=config
        )
        engine = BacktestEngine()
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 1
        # Should have exited via signal, not stop-loss
        assert result.trades.iloc[0]["exit_reason"] == "signal"


class TestTakeProfit:
    """Take-profit behaviour."""

    def test_take_profit_triggers_on_high(self):
        """When bar high crosses take-profit level, exit at tp price."""
        config = StrategyConfig(take_profit_pct=0.05, stop_loss_pct=0.50)
        prices = [
            (100.0, 100.2, 99.8, 100.0),  # bar 0
            (101.0, 101.2, 100.8, 101.0),  # bar 1
            (100.0, 100.2, 99.8, 100.0),  # bar 2 ← BUY at 100 (TP = 105)
            (102.0, 102.2, 101.8, 102.0),  # bar 3
            (104.0, 105.5, 103.5, 104.0),  # bar 4 ← high=105.5 >= 105 → TP!
        ]
        data = _make_data(prices)

        strategy = CustomSignalStrategy([(2, SignalType.BUY)], config=config)
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 1
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == "take_profit"
        assert trade["exit_price"] == pytest.approx(100.0 * 1.05, rel=1e-6)
        assert trade["pnl"] > 0


class TestStopLossBeforeTakeProfit:
    """When both SL and TP are possible in the same bar, SL wins (conservative)."""

    def test_sl_before_tp_same_bar(self):
        """Bar has both low below SL and high above TP — SL triggers first."""
        config = StrategyConfig(stop_loss_pct=0.02, take_profit_pct=0.10)
        # Buy at 100, SL = 98, TP = 110
        prices = [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.0),  # bar 1 ← BUY at 100
            (100.0, 112.0, 97.0, 100.0),  # bar 2 ← high=112 > 110, low=97 < 98
        ]
        data = _make_data(prices)

        strategy = CustomSignalStrategy([(1, SignalType.BUY)], config=config)
        engine = BacktestEngine()
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 1
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == "stop_loss"
        assert trade["exit_price"] == pytest.approx(98.0, rel=1e-6)

    def test_tp_only_when_sl_not_hit(self):
        """When only TP level is breached in a bar, TP triggers."""
        config = StrategyConfig(stop_loss_pct=0.02, take_profit_pct=0.10)
        # Buy at 100, SL = 98, TP = 110
        prices = [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.0),  # bar 1 ← BUY at 100
            (100.0, 112.0, 99.0, 100.0),  # bar 2 ← high=112 > 110, low=99 > 98 → TP
        ]
        data = _make_data(prices)

        strategy = CustomSignalStrategy([(1, SignalType.BUY)], config=config)
        engine = BacktestEngine()
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 1
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == "take_profit"
        assert trade["exit_price"] == pytest.approx(110.0, rel=1e-6)


class TestCommission:
    """Commission reduces net returns."""

    def test_commission_reduces_returns(self):
        """Higher commission → lower (or more negative) total return."""
        data = _trending_up(20, start=100.0, step=0.5)

        engine_no_comm = BacktestEngine(initial_capital=100_000, commission=0.0)
        result_no = engine_no_comm.run(FixedSignalStrategy(buy_idx=3, sell_idx=17), data)

        engine_comm = BacktestEngine(initial_capital=100_000, commission=0.01)
        result_comm = engine_comm.run(FixedSignalStrategy(buy_idx=3, sell_idx=17), data)

        assert result_comm.metrics["total_return"] < result_no.metrics["total_return"]

    def test_commission_applied_on_both_sides(self):
        """Both entry and exit incur commission costs."""
        config = StrategyConfig(stop_loss_pct=0.50, take_profit_pct=0.50)
        data = _flat(10, level=100.0)

        # Buy at bar 2, sell at bar 7 — zero price change, so only cost is commission
        strategy = CustomSignalStrategy(
            [(2, SignalType.BUY), (7, SignalType.SELL)], config=config
        )
        engine = BacktestEngine(initial_capital=100_000, commission=0.005)
        result = engine.run(strategy, data)

        # Equity should decrease due to commission
        assert result.metrics["total_return"] < 0
        trade = result.trades.iloc[0]
        assert trade["pnl"] < 0  # only cost is commission on both sides


class TestPositionSizing:
    """Position sizes respect max_position_pct."""

    def test_position_sizing_respects_max_pct(self):
        """Position notional should not exceed max_position_pct * equity."""
        config = StrategyConfig(max_position_pct=0.05)
        data = _trending_up(20, start=100.0, step=0.5)

        strategy = CustomSignalStrategy(
            [(3, SignalType.BUY), (15, SignalType.SELL)], config=config
        )
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(strategy, data)

        trade = result.trades.iloc[0]
        notional = trade["quantity"] * trade["entry_price"]
        # At entry time equity ≈ 100_000, so max notional = 5_000
        assert notional <= 100_000 * 0.05 + 1e-9

    def test_position_sizing_uses_current_equity(self):
        """After a losing trade, next position should be sized on reduced equity."""
        config = StrategyConfig(max_position_pct=0.10)
        # Downtrend: first trade loses money, second should be smaller
        data = _trending_down(30, start=100.0, step=0.5)

        plan = [
            (2, SignalType.BUY),
            (10, SignalType.SELL),
            (15, SignalType.BUY),
            (25, SignalType.SELL),
        ]
        strategy = CustomSignalStrategy(plan, config=config)
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 2
        t1_notional = result.trades.iloc[0]["quantity"] * result.trades.iloc[0]["entry_price"]
        t2_notional = result.trades.iloc[1]["quantity"] * result.trades.iloc[1]["entry_price"]
        # Both should respect max_position_pct
        assert t1_notional <= 100_000 * 0.10 + 1e-9
        # Second trade notional should be smaller because equity decreased
        assert t2_notional < t1_notional

    def test_min_position_pct_not_enforced_by_engine(self):
        """Engine doesn't enforce min_position_pct (that's a strategy concern)."""
        config = StrategyConfig(max_position_pct=0.01, min_position_pct=0.001)
        data = _trending_up(10, start=100.0, step=0.5)

        strategy = CustomSignalStrategy(
            [(2, SignalType.BUY), (8, SignalType.SELL)], config=config
        )
        engine = BacktestEngine(initial_capital=100)
        result = engine.run(strategy, data)

        # Works fine even with tiny capital — position is small
        assert result.metrics["num_trades"] == 1
        assert result.trades.iloc[0]["quantity"] > 0


class TestMultipleRoundTrips:
    """Multiple buy/sell cycles."""

    def test_multiple_profitable_trades(self):
        """Multiple trades in an oscillating market."""
        # Oscillating: 100 → 105 → 95 → 105 (buy dips, sell peaks)
        prices = [
            (100.0, 100.2, 99.8, 100.0),   # 0
            (102.0, 102.2, 101.8, 102.0),   # 1
            (104.0, 104.2, 103.8, 104.0),   # 2
            (103.0, 103.2, 102.8, 103.0),   # 3
            (101.0, 101.2, 100.8, 101.0),   # 4 ← BUY
            (99.0, 99.2, 98.8, 99.0),       # 5
            (97.0, 97.2, 96.8, 97.0),       # 6
            (99.0, 99.2, 98.8, 99.0),       # 7
            (101.0, 101.2, 100.8, 101.0),   # 8
            (103.0, 103.2, 102.8, 103.0),   # 9 ← SELL (profit)
            (101.0, 101.2, 100.8, 101.0),   # 10
            (99.0, 99.2, 98.8, 99.0),       # 11 ← BUY
            (97.0, 97.2, 96.8, 97.0),       # 12
            (99.0, 99.2, 98.8, 99.0),       # 13
            (101.0, 101.2, 100.8, 101.0),   # 14 ← SELL (profit)
            (103.0, 103.2, 102.8, 103.0),   # 15
        ]
        data = _make_data(prices)

        plan = [
            (4, SignalType.BUY),
            (9, SignalType.SELL),
            (11, SignalType.BUY),
            (14, SignalType.SELL),
        ]
        config = StrategyConfig(stop_loss_pct=0.50, take_profit_pct=0.50)
        strategy = CustomSignalStrategy(plan, config=config)
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(strategy, data)

        assert result.metrics["num_trades"] == 2
        assert result.metrics["win_rate"] == 1.0
        assert result.metrics["total_return"] > 0


class TestMetrics:
    """Metrics calculations are correct."""

    def test_sharpe_zero_for_flat_equity(self):
        curve = pd.Series([100.0] * 100, index=pd.date_range("2026-01-01", periods=100, freq="1min"))
        assert metric_sharpe(curve) == 0.0

    def test_sharpe_positive_for_uptrend(self):
        curve = pd.Series(
            [100.0 + i * 0.1 for i in range(100)],
            index=pd.date_range("2026-01-01", periods=100, freq="1min"),
        )
        assert metric_sharpe(curve) > 0

    def test_sortino_zero_for_flat_equity(self):
        curve = pd.Series([100.0] * 100, index=pd.date_range("2026-01-01", periods=100, freq="1min"))
        assert metric_sortino(curve) == 0.0

    def test_max_drawdown_captures_peak_to_trough(self):
        # Price goes up 10%, then down 20%
        values = [100.0]
        for i in range(10):
            values.append(values[-1] * 1.01)   # up
        for i in range(10):
            values.append(values[-1] * 0.98)   # down
        curve = pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="1min"))
        dd = metric_max_dd(curve)
        assert dd > 0.0
        assert dd < 1.0

    def test_total_return_correct(self):
        curve = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0],
            index=pd.date_range("2026-01-01", periods=5, freq="1min"),
        )
        assert metric_total_return(curve) == pytest.approx(0.04, rel=1e-6)

    def test_total_return_zero_for_single_point(self):
        curve = pd.Series([100.0], index=pd.DatetimeIndex(["2026-01-01"]))
        assert metric_total_return(curve) == 0.0

    def test_win_rate_correct(self):
        trades = pd.DataFrame({
            "entry_time": pd.date_range("2026-01-01", periods=4, freq="1min"),
            "exit_time": pd.date_range("2026-01-01 00:01", periods=4, freq="1min"),
            "symbol": ["A"] * 4,
            "side": ["BUY"] * 4,
            "entry_price": [100.0] * 4,
            "exit_price": [101.0] * 4,
            "quantity": [10.0] * 4,
            "pnl": [10.0, -5.0, 15.0, -2.0],
            "pnl_pct": [0.01, -0.005, 0.015, -0.002],
            "exit_reason": ["signal"] * 4,
        })
        assert metric_win_rate(trades) == 0.5  # 2 out of 4

    def test_win_rate_empty_trades(self):
        empty = pd.DataFrame()
        assert metric_win_rate(empty) == 0.0

    def test_profit_factor(self):
        trades = pd.DataFrame({
            "entry_time": pd.date_range("2026-01-01", periods=4, freq="1min"),
            "exit_time": pd.date_range("2026-01-01 00:01", periods=4, freq="1min"),
            "symbol": ["A"] * 4,
            "side": ["BUY"] * 4,
            "entry_price": [100.0] * 4,
            "exit_price": [101.0] * 4,
            "quantity": [10.0] * 4,
            "pnl": [10.0, -5.0, 15.0, -2.0],
            "pnl_pct": [0.01, -0.005, 0.015, -0.002],
            "exit_reason": ["signal"] * 4,
        })
        pf = metric_profit_factor(trades)
        assert pf == pytest.approx(25.0 / 7.0, rel=1e-6)

    def test_profit_factor_no_losers(self):
        trades = pd.DataFrame({
            "entry_time": [pd.Timestamp("2026-01-01")],
            "exit_time": [pd.Timestamp("2026-01-01 00:01")],
            "symbol": ["A"],
            "side": ["BUY"],
            "entry_price": [100.0],
            "exit_price": [101.0],
            "quantity": [10.0],
            "pnl": [10.0],
            "pnl_pct": [0.01],
            "exit_reason": ["signal"],
        })
        assert metric_profit_factor(trades) == float("inf")

    def test_profit_factor_empty(self):
        assert metric_profit_factor(pd.DataFrame()) == 0.0

    def test_num_trades(self):
        trades = pd.DataFrame({"pnl": [1, 2, 3]})
        assert metric_num_trades(trades) == 3

    def test_num_trades_empty(self):
        assert metric_num_trades(pd.DataFrame()) == 0


class TestEndToEnd:
    """Smoke tests matching the lead's definition-of-done criteria."""

    def test_equity_curve_goes_up_and_down_based_on_signals(self, uptrend_ohlcv):
        """Equity curve should NOT be flat when there are signals."""
        engine = BacktestEngine(initial_capital=50_000)
        result = engine.run(FixedSignalStrategy(buy_idx=5, sell_idx=30), uptrend_ohlcv)
        eq = result.equity_curve.values
        # Must have at least some variation
        assert eq.std() > 0, "Equity curve is flat; should vary with price"

    def test_metrics_in_result_are_populated(self):
        engine = BacktestEngine()
        data = _trending_up(30, start=100.0, step=0.5)
        result = engine.run(FixedSignalStrategy(buy_idx=3, sell_idx=25), data)
        required_keys = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "total_return", "win_rate", "profit_factor", "num_trades",
        }
        for k in required_keys:
            assert k in result.metrics, f"Missing metric: {k}"

    def test_trades_dataframe_has_correct_columns(self):
        engine = BacktestEngine()
        data = _trending_up(20, start=100.0, step=0.5)
        result = engine.run(FixedSignalStrategy(buy_idx=3, sell_idx=15), data)

        expected_cols = [
            "entry_time", "exit_time", "symbol", "side",
            "entry_price", "exit_price", "quantity",
            "pnl", "pnl_pct", "exit_reason",
        ]
        for col in expected_cols:
            assert col in result.trades.columns, f"Missing column: {col}"


class TestComputeMetricsIntegration:
    """Integration-style metrics tests that exercise compute_metrics directly."""

    def test_compute_metrics_returns_dict(self):
        curve = pd.Series(
            [100_000 + i * 10 for i in range(100)],
            index=pd.date_range("2026-01-01", periods=100, freq="1min"),
        )
        trades = pd.DataFrame({
            "entry_time": [pd.Timestamp("2026-01-01")],
            "exit_time": [pd.Timestamp("2026-01-02")],
            "symbol": ["A"],
            "side": ["BUY"],
            "entry_price": [100.0],
            "exit_price": [105.0],
            "quantity": [100.0],
            "pnl": [500.0],
            "pnl_pct": [0.05],
            "exit_reason": ["signal"],
        })
        metrics = compute_metrics(curve, trades)
        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "num_trades" in metrics

    def test_compute_metrics_all_trades_winning(self):
        curve = pd.Series(
            [100_000 + i * 10 for i in range(100)],
            index=pd.date_range("2026-01-01", periods=100, freq="1min"),
        )
        trades = pd.DataFrame({
            "entry_time": pd.date_range("2026-01-01", periods=3, freq="1h"),
            "exit_time": pd.date_range("2026-01-01 02:00", periods=3, freq="1h"),
            "symbol": ["A", "B", "C"],
            "side": ["BUY"] * 3,
            "entry_price": [100.0, 200.0, 300.0],
            "exit_price": [105.0, 210.0, 315.0],
            "quantity": [10.0, 5.0, 3.0],
            "pnl": [50.0, 50.0, 45.0],
            "pnl_pct": [0.05, 0.05, 0.05],
            "exit_reason": ["signal"] * 3,
        })
        metrics = compute_metrics(curve, trades)
        assert metrics["win_rate"] == 1.0
        assert metrics["profit_factor"] == float("inf")
        assert metrics["num_trades"] == 3
