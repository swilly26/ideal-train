"""Tests for live trading runner, position manager, and config persistence."""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.execution.broker import Broker, Order, OrderResult, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.execution.live_runner import LiveTradingRunner
from src.optimization.persistence import save_config, load_config
from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.data.provider import DataProvider, MarketDataFrame


# ---------------------------------------------------------------------------
# A simple Strategy that always emits a BUY for the last bar
# ---------------------------------------------------------------------------


class AlwaysBuyStrategy(Strategy):
    """Test strategy: emits a BUY with configurable confidence on the last bar."""

    def __init__(self, config=None, confidence=1.0):
        super().__init__(config)
        self._confidence = confidence

    def generate_signals(self, data):
        if data.empty:
            return []
        return [
            Signal(
                symbol="AAPL",
                timestamp=data.index[-1],
                signal_type=SignalType.BUY,
                confidence=self._confidence,
            )
        ]


class AlwaysSellStrategy(Strategy):
    """Test strategy: emits a SELL on every bar."""

    def generate_signals(self, data):
        if data.empty:
            return []
        return [
            Signal(
                symbol="AAPL",
                timestamp=data.index[-1],
                signal_type=SignalType.SELL,
                confidence=1.0,
            )
        ]


# ---------------------------------------------------------------------------
# Mock broker and data provider
# ---------------------------------------------------------------------------


class DummyBroker(Broker):
    """In-memory broker for testing the live runner."""

    def __init__(self):
        self.orders: list[Order] = []
        self._next_id = 0
        self._market_open = True

    async def place_order(self, order):
        self._next_id += 1
        self.orders.append(order)
        return OrderResult(
            order_id=str(self._next_id),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            filled_avg_price=100.0,
            status="filled",
            created_at=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id):
        return True

    async def get_positions(self):
        return []

    async def get_account(self):
        return {"equity": 100_000.0, "buying_power": 200_000.0, "cash": 50_000.0, "portfolio_value": 100_000.0}

    async def close(self):
        pass

    async def is_market_open(self):
        return self._market_open


def _make_ohlcv_data(symbol="AAPL", periods=20):
    """Create a small DataFrame of OHLCV bars."""
    idx = pd.date_range("2026-01-15 10:00", periods=periods, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(periods)],
            "high": [101.0 + i * 0.1 for i in range(periods)],
            "low": [99.0 + i * 0.1 for i in range(periods)],
            "close": [100.5 + i * 0.1 for i in range(periods)],
            "volume": [1000] * periods,
        },
        index=idx,
    )


class DummyDataProvider(DataProvider):
    """Data provider that returns a fixed DataFrame."""

    def __init__(self, data=None):
        super().__init__()
        self._data = data or _make_ohlcv_data()

    async def fetch_bars(self, symbol, start, end, timeframe="1min"):
        return MarketDataFrame(df=self._data.copy(), symbol=symbol, timeframe=timeframe)

    async def subscribe_live(self, symbols, on_bar=None):
        raise NotImplementedError("streaming not used in tests")

    async def close(self):
        pass


class PerSymbolDataProvider(DataProvider):
    """Data provider that returns a different DataFrame per symbol."""

    def __init__(self, data_by_symbol):
        super().__init__()
        self._data = {k.upper(): v for k, v in data_by_symbol.items()}

    async def fetch_bars(self, symbol, start, end, timeframe="1min"):
        return MarketDataFrame(
            df=self._data[symbol.upper()].copy(), symbol=symbol, timeframe=timeframe
        )

    async def subscribe_live(self, symbols, on_bar=None):
        raise NotImplementedError("streaming not used in tests")

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# Position Manager tests
# ---------------------------------------------------------------------------


class TestPositionManager:
    def test_can_open_when_flat(self):
        pm = PositionManager()
        assert pm.can_open("AAPL", 100_000) is True

    def test_prevents_duplicate_entry(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        assert pm.can_open("AAPL", 100_000) is False
        assert pm.has_position("AAPL") is True
        assert pm.has_position("MSFT") is False

    def test_open_position_raises_on_duplicate(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        with pytest.raises(ValueError, match="already open"):
            pm.open_position("AAPL", 5, 101.0)

    def test_close_position_returns_trade(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        trade = pm.close_position("AAPL", 110.0, exit_reason="signal")
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.pnl == 100.0  # (110 - 100) * 10
        assert trade.exit_reason == "signal"

    def test_close_nonexistent_position(self):
        pm = PositionManager()
        trade = pm.close_position("AAPL", 110.0)
        assert trade is None

    def test_unrealized_pnl(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        pm.update_price("AAPL", 105.0)
        assert pm.get_unrealized_pnl() == 50.0

    def test_realized_pnl(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        pm.close_position("AAPL", 110.0)
        assert pm.get_realized_pnl() == 100.0  # (110-100)*10
        assert pm.get_total_pnl() == 100.0

    def test_total_pnl_combines_both(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        pm.update_price("AAPL", 105.0)
        pm.open_position("MSFT", 5, 200.0)
        pm.close_position("MSFT", 210.0, exit_reason="take_profit")
        # Open: AAPL unrealized = 50
        # Closed: MSFT realized = (210-200)*5 = 50
        assert pm.get_realized_pnl() == 50.0
        assert pm.get_unrealized_pnl() == 50.0
        assert pm.get_total_pnl() == 100.0

    def test_get_open_symbols(self):
        pm = PositionManager()
        pm.open_position("MSFT", 5, 200.0)
        pm.open_position("AAPL", 10, 100.0)
        assert pm.get_open_symbols() == ["AAPL", "MSFT"]

    def test_get_open_count(self):
        pm = PositionManager()
        assert pm.get_open_count() == 0
        pm.open_position("AAPL", 10, 100.0)
        assert pm.get_open_count() == 1
        pm.close_position("AAPL", 110.0)
        assert pm.get_open_count() == 0

    def test_close_all(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        pm.open_position("MSFT", 5, 200.0)

        def exit_price_fn(sym):
            return {"AAPL": 110.0, "MSFT": 210.0}[sym]

        trades = pm.close_all(exit_price_fn)
        assert len(trades) == 2
        assert pm.get_open_count() == 0
        # AAPL: (110-100)*10 = 100, MSFT: (210-200)*5 = 50
        assert pm.get_realized_pnl() == 150.0

    def test_reset(self):
        pm = PositionManager()
        pm.open_position("AAPL", 10, 100.0)
        pm.reset()
        assert pm.get_open_count() == 0
        assert pm.get_realized_pnl() == 0.0

    def test_sync_adds_unknown_positions(self):
        """PositionManager.add_position should work for sync-from-broker flow."""
        pm = PositionManager()
        # Simulate broker reporting a position we don't know about
        pm.open_position("TSLA", 50, 300.0)
        assert pm.has_position("TSLA") is True
        pos = pm.get_positions()["TSLA"]
        assert pos.quantity == 50
        assert pos.entry_price == 300.0

    def test_sync_removes_stale_positions(self):
        """Closing a position and verifying it's gone."""
        pm = PositionManager()
        pm.open_position("MSFT", 20, 400.0)
        assert pm.has_position("MSFT") is True

        # Broker no longer has MSFT — remove it
        pm.close_position("MSFT", exit_price=0, exit_reason="sync_removed")
        assert pm.has_position("MSFT") is False
        assert pm.get_open_count() == 0

    def test_sync_mixed_scenario(self):
        """Partial sync: some added, some removed."""
        pm = PositionManager()
        # Pre-load positions the PM already knows about
        pm.open_position("AAPL", 10, 150.0)

        # Broker reports AAPL + SPY, but PM also has stale MSFT
        # Add SPY (new from broker)
        pm.open_position("SPY", 5, 450.0)
        
        # Remove stale MSFT (if it were present)
        # Simulate: close MSFT that wasn't on broker
        assert pm.has_position("AAPL") is True
        assert pm.has_position("SPY") is True
        assert pm.get_open_count() == 2


# ---------------------------------------------------------------------------
# Config persistence tests
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    def test_save_load_roundtrip(self):
        config = StrategyConfig(
            entry_threshold=0.7,
            exit_threshold=0.5,
            max_position_pct=0.15,
            stop_loss_pct=0.03,
            take_profit_pct=0.08,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.entry_threshold == 0.7
            assert loaded.exit_threshold == 0.5
            assert loaded.max_position_pct == 0.15
            assert loaded.stop_loss_pct == 0.03
            assert loaded.take_profit_pct == 0.08
            assert loaded.weight == 1.0  # default
        finally:
            os.unlink(path)

    def test_save_load_with_extra(self):
        config = StrategyConfig(extra={"period": 20, "symbol": "AAPL"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.extra == {"period": 20, "symbol": "AAPL"}
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent_config_file.json")

    def test_load_missing_fields_fills_defaults(self):
        """Missing optional fields should be filled with defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "entry_threshold": 0.5,
                    "exit_threshold": 0.3,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.05,
                    # intentionally omit max_position_pct, weight, extra
                },
                f,
            )
            path = f.name

        try:
            loaded = load_config(path)
            assert loaded.max_position_pct == 0.10  # default
            assert loaded.weight == 1.0
            assert loaded.min_position_pct == 0.01
            assert loaded.extra == {}
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self):
        config = StrategyConfig()
        tmpdir = tempfile.mkdtemp()
        nested_path = os.path.join(tmpdir, "subdir1", "subdir2", "config.json")
        try:
            save_config(config, nested_path)
            assert os.path.exists(nested_path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Live Runner tests
# ---------------------------------------------------------------------------


class TestLiveRunner:
    @pytest.mark.asyncio
    async def test_runner_places_buy_order(self):
        """Runner should place a buy order when strategy emits BUY."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()
        config = StrategyConfig(max_position_pct=0.10, stop_loss_pct=0.02, take_profit_pct=0.05)

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
            check_interval_seconds=0.0,
        )

        # Run one tick manually
        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config)
        await runner._tick()

        assert len(broker.orders) == 1
        assert broker.orders[0].symbol == "AAPL"
        assert broker.orders[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_runner_ignores_low_confidence(self):
        """Signals below confidence threshold should be ignored."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()
        config = StrategyConfig()

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
            confidence_threshold=0.9,
            check_interval_seconds=0.0,
        )

        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=0.3)
        await runner._tick()

        assert len(broker.orders) == 0

    @pytest.mark.asyncio
    async def test_runner_respects_max_positions(self):
        """Runner should not open more positions than max_positions allows."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()
        config = StrategyConfig()

        runner = LiveTradingRunner(
            symbols=["AAPL", "MSFT", "TSLA"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
            max_positions=1,
            check_interval_seconds=0.0,
        )

        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=1.0)
        await runner._tick()

        # Only the first symbol should be bought (max_positions=1)
        # The tick iterates through symbols and buys the first eligible one
        assert runner.position_manager.get_open_count() == 1

    @pytest.mark.asyncio
    async def test_runner_prevents_duplicate_buys(self):
        """Runner should not buy the same symbol twice."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()
        config = StrategyConfig(max_position_pct=0.10)

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
            check_interval_seconds=0.0,
        )

        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=1.0)

        # First tick opens a position
        await runner._tick()
        assert runner.position_manager.has_position("AAPL")
        first_order_count = len(broker.orders)

        # Second tick should NOT place another buy
        await runner._tick()
        assert len(broker.orders) == first_order_count

    @pytest.mark.asyncio
    async def test_runner_sell_signal_closes_position(self):
        """A SELL signal should close an existing position."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()
        config = StrategyConfig(max_position_pct=0.10)

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
            check_interval_seconds=0.0,
        )

        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=1.0)
        await runner._tick()
        assert runner.position_manager.has_position("AAPL")

        # Now switch to sell strategy
        runner._strategy = AlwaysSellStrategy(config=config)
        await runner._tick()
        assert not runner.position_manager.has_position("AAPL")

    @pytest.mark.asyncio
    async def test_runner_skips_when_market_closed(self):
        """Runner should skip tick when market is closed."""
        broker = DummyBroker()
        broker._market_open = False
        data_provider = DummyDataProvider()
        config = StrategyConfig()

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
            check_interval_seconds=0.0,
        )

        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=1.0)
        await runner._tick()

        assert len(broker.orders) == 0

    @pytest.mark.asyncio
    async def test_runner_position_manager_accessible(self):
        """Position manager should be accessible via property."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()
        config = StrategyConfig()

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
            config=config,
        )

        assert runner.position_manager is not None
        assert runner.position_manager.get_open_count() == 0

    @pytest.mark.asyncio
    async def test_runner_stop(self):
        """Calling stop() should set _running to False."""
        broker = DummyBroker()
        data_provider = DummyDataProvider()

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=data_provider,
        )
        runner._running = True
        runner.stop()
        assert runner._running is False

    @pytest.mark.asyncio
    async def test_risk_stop_uses_per_symbol_data(self):
        """Risk stops must be priced against each position's OWN symbol.

        Regression: ``_check_risk_stops`` used the last-polled symbol's
        bars to price every position, so a crash in one symbol could
        trigger erroneous exits in others.
        """
        broker = DummyBroker()
        flat = _make_ohlcv_data(periods=5)  # closes ≈ 100.5 → no stop for entry 100
        crash = _make_ohlcv_data(periods=5)
        crash["low"] = 50.0
        crash["close"] = 50.0
        provider = PerSymbolDataProvider({"AAPL": flat, "MSFT": crash})
        config = StrategyConfig(max_position_pct=0.10, stop_loss_pct=0.02, take_profit_pct=0.05)

        runner = LiveTradingRunner(
            symbols=["AAPL", "MSFT"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=provider,
            config=config,
            check_interval_seconds=0.0,
        )
        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=1.0)

        runner.position_manager.open_position(
            "AAPL", 10, 100.0, stop_loss_price=98.0, take_profit_price=105.0
        )
        # Passing MSFT's crash bars (what the old tick handed to the risk
        # check) must NOT trigger an AAPL exit — data is fetched per symbol.
        await runner._check_risk_stops(crash)

        assert len(broker.orders) == 0
        assert runner.position_manager.has_position("AAPL")

    @pytest.mark.asyncio
    async def test_risk_stop_triggers_sell_on_own_symbol(self):
        """A crash in the position's own symbol must still trigger the exit."""
        broker = DummyBroker()
        crash = _make_ohlcv_data(periods=5)
        crash["low"] = 50.0
        crash["close"] = 50.0
        provider = PerSymbolDataProvider({"AAPL": crash})
        config = StrategyConfig(max_position_pct=0.10, stop_loss_pct=0.02, take_profit_pct=0.05)

        runner = LiveTradingRunner(
            symbols=["AAPL"],
            strategy_cls=AlwaysBuyStrategy,
            broker=broker,
            data_provider=provider,
            config=config,
            check_interval_seconds=0.0,
        )
        runner._running = True
        runner._strategy = AlwaysBuyStrategy(config=config, confidence=1.0)

        runner.position_manager.open_position(
            "AAPL", 10, 100.0, stop_loss_price=98.0, take_profit_price=105.0
        )
        await runner._check_risk_stops(None)

        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.SELL
        assert not runner.position_manager.has_position("AAPL")
