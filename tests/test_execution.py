"""Tests for the execution / broker module."""

import pytest
from datetime import datetime

from src.execution import Broker, Order, OrderResult, OrderSide, OrderType


class MockBroker(Broker):
    """A broker that simulates execution (always fills at market)."""

    def __init__(self):
        super().__init__()
        self.orders: list[Order] = []
        self._next_id = 1

    async def place_order(self, order):
        self.orders.append(order)
        return OrderResult(
            order_id=str(self._next_id),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            filled_avg_price=100.0,
            status="filled",
            created_at=datetime.now(),
        )
        self._next_id += 1

    async def cancel_order(self, order_id):
        return True

    async def get_positions(self):
        return []

    async def get_account(self):
        return {"equity": 100_000.0, "buying_power": 200_000.0}

    async def close(self):
        pass


class TestOrder:
    def test_market_buy_order(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None

    def test_limit_order(self):
        order = Order(
            symbol="SPY",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=450.0,
        )
        assert order.limit_price == 450.0
        assert order.order_type == OrderType.LIMIT


class TestBroker:
    @pytest.mark.asyncio
    async def test_place_order_returns_result(self):
        broker = MockBroker()
        order = Order(symbol="TSLA", side=OrderSide.BUY, quantity=50)
        result = await broker.place_order(order)
        assert result.status == "filled"
        assert result.filled_quantity == 50

    @pytest.mark.asyncio
    async def test_place_order_stores_order(self):
        broker = MockBroker()
        order = Order(symbol="MSFT", side=OrderSide.SELL, quantity=20)
        await broker.place_order(order)
        assert len(broker.orders) == 1
        assert broker.orders[0].symbol == "MSFT"

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        broker = MockBroker()
        assert await broker.cancel_order("123") is True

    @pytest.mark.asyncio
    async def test_get_account(self):
        broker = MockBroker()
        acct = await broker.get_account()
        assert acct["equity"] == 100_000.0
