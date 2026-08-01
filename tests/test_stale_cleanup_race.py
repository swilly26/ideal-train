"""Regression tests for safe cancellation and rejected liquidation orders."""
from datetime import datetime, timezone

import pytest

from src.execution.broker import OrderResult, OrderSide
from src.execution.live_runner import LiveTradingRunner
from tests.test_live_runner import DummyBroker, DummyDataProvider, AlwaysSellStrategy


class RejectSellBroker(DummyBroker):
    async def place_order(self, order):
        self.orders.append(order)
        return OrderResult(
            order_id="rejected", symbol=order.symbol, side=order.side,
            quantity=order.quantity, filled_quantity=0,
            filled_avg_price=None, status="rejected", created_at=datetime.now(timezone.utc),
        )


@pytest.mark.asyncio
async def test_rejected_sell_keeps_position_tracked():
    broker = RejectSellBroker()
    runner = LiveTradingRunner(
        symbols=["AAPL"], strategy_cls=AlwaysSellStrategy, broker=broker,
        data_provider=DummyDataProvider(), check_interval_seconds=0,
    )
    runner.position_manager.open_position("AAPL", 10, 100)
    await runner._execute_signal(AlwaysSellStrategy().generate_signals(DummyDataProvider()._data)[0], DummyDataProvider()._data)
    assert runner.position_manager.has_position("AAPL")
    assert len(broker.orders) == 1
