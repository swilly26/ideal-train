"""Regression tests for the turbo protective-stop placement bug.

The bug: after a buy fills, the turbo trader immediately submitted a SELL
stop while the entry BUY was still open.  Alpaca rejected it with
"potential wash trade detected ... opposite side market/stop order exists"
and the position ran the whole session without a broker-side stop.

The fix has two halves, both tested here:
1. ``_handle_buy`` waits for the entry order to fill (via the broker's
   ``wait_for_order_fill``) before recording the position and placing the stop.
2. ``_place_protective_stop`` retries with backoff and re-checks the
   symbol's open orders before each attempt, so a transient wash-trade
   rejection no longer leaves the position unprotected.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import turbo_trader
from src.execution.broker import OrderResult, OrderSide
from src.execution.position_manager import PositionManager

WASH_TRADE_ERROR = Exception(
    '{"code":40310000,"message":"potential wash trade detected. use complex orders",'
    '"reject_reason":"opposite side market/stop order exists"}'
)


class FakeBroker:
    """In-memory broker stand-in for the turbo stop placement paths."""

    def __init__(self, open_orders_seq=None, stop_errors=(), fill_result=None,
                 buy_status="accepted"):
        self._open_orders_seq = list(open_orders_seq or [])
        self._open_orders = []
        self._stop_errors = list(stop_errors)
        self.fill_result = fill_result
        self.buy_status = buy_status
        self.stop_requests = []  # (symbol, qty, stop_price, client_id)
        self.buy_orders = []

    async def get_account(self):
        return {"equity": 200_000.0, "buying_power": 400_000.0,
                "cash": 100_000.0, "portfolio_value": 200_000.0}

    async def get_open_orders(self, symbol=None):
        if self._open_orders_seq:
            view = self._open_orders_seq.pop(0)
            if view is not None:
                self._open_orders = view
        return [o for o in self._open_orders if o.symbol == symbol]

    async def place_stop_order(self, symbol, qty, stop_price, client_id=None):
        if self._stop_errors:
            raise self._stop_errors.pop(0)
        self.stop_requests.append((symbol, qty, stop_price, client_id))
        return MagicMock(status="new")

    async def place_order(self, order):
        self.buy_orders.append(order)
        return OrderResult(
            order_id="order-1",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0.0,
            filled_avg_price=None,
            status=self.buy_status,
            created_at=datetime.now(timezone.utc),
        )

    async def wait_for_order_fill(self, order_id, timeout=8.0, poll_interval=0.25):
        return self.fill_result


def _make_trader(broker: FakeBroker):
    """Build a TurboTrader with only the attributes the tested paths touch."""
    trader = object.__new__(turbo_trader.TurboTrader)
    trader.broker = broker
    trader.pm = PositionManager(turbo_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader._ls_symbols = set()
    return trader


def _buy_order_mock(symbol="FNGU"):
    o = MagicMock()
    o.symbol = symbol
    o.side = "buy"
    return o


def _sell_order_mock(symbol="FNGU"):
    o = MagicMock()
    o.symbol = symbol
    o.side = "sell"
    return o


def _filled_order_mock(qty="100", price="32.50"):
    o = MagicMock()
    o.status = "filled"
    o.qty = qty
    o.filled_avg_price = price
    return o


# ---------------------------------------------------------------------------
# _place_protective_stop — retry / backoff / race-safe pre-checks
# ---------------------------------------------------------------------------


class TestPlaceProtectiveStop:
    @pytest.mark.asyncio
    async def test_retries_after_wash_trade_rejection(self):
        """A wash-trade rejection must be retried, not swallowed."""
        broker = FakeBroker(stop_errors=[WASH_TRADE_ERROR])
        trader = _make_trader(broker)

        ok = await trader._place_protective_stop(
            "FNGU", 100, 32.0, max_attempts=2, initial_delay=0.0,
        )

        assert ok is True
        assert len(broker.stop_requests) == 1
        symbol, qty, stop_price, _ = broker.stop_requests[0]
        assert symbol == "FNGU"
        assert qty == 100
        assert stop_price == round(32.0 * (1 - turbo_trader.STRATEGY_CONFIG.stop_loss_pct), 2)

    @pytest.mark.asyncio
    async def test_waits_while_entry_buy_still_open(self):
        """While the entry BUY is open, the stop must wait, not submit."""
        broker = FakeBroker(
            open_orders_seq=[[_buy_order_mock()], None],  # buy open, then gone
        )
        trader = _make_trader(broker)

        ok = await trader._place_protective_stop(
            "FNGU", 100, 32.0, max_attempts=2, initial_delay=0.0,
        )

        assert ok is True
        assert len(broker.stop_requests) == 1

    @pytest.mark.asyncio
    async def test_skips_when_sell_order_already_exists(self):
        """Never submit a duplicate stop when a sell order is already open."""
        broker = FakeBroker(open_orders_seq=[[_sell_order_mock()]])
        trader = _make_trader(broker)

        ok = await trader._place_protective_stop("FNGU", 100, 32.0)

        assert ok is True
        assert broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_reports_failure_after_exhausting_attempts(self):
        broker = FakeBroker(stop_errors=[WASH_TRADE_ERROR, WASH_TRADE_ERROR])
        trader = _make_trader(broker)

        ok = await trader._place_protective_stop(
            "FNGU", 100, 32.0, max_attempts=2, initial_delay=0.0,
        )

        assert ok is False
        assert len(broker.stop_requests) == 0  # both submits rejected

    @pytest.mark.asyncio
    async def test_skips_position_too_small(self):
        broker = FakeBroker()
        trader = _make_trader(broker)

        ok = await trader._place_protective_stop("FNGU", 0.4, 32.0)

        assert ok is False
        assert broker.stop_requests == []


# ---------------------------------------------------------------------------
# _handle_buy — wait for the entry fill before attaching the stop
# ---------------------------------------------------------------------------


class TestHandleBuy:
    @pytest.mark.asyncio
    async def test_waits_for_fill_then_places_stop_at_fill_price(self):
        """The position must be recorded with the fill price/qty, then the
        stop attached — never while the entry is still open."""
        broker = FakeBroker(fill_result=_filled_order_mock(qty="100", price="32.50"))
        trader = _make_trader(broker)

        await trader._handle_buy("FNGU", 32.0, 1.0, strategy="mean_reversion")

        pos = trader.pm.get_positions().get("FNGU")
        assert pos is not None
        assert pos.quantity == 100.0
        assert pos.entry_price == 32.50

        assert len(broker.stop_requests) == 1
        symbol, qty, stop_price, _ = broker.stop_requests[0]
        assert symbol == "FNGU"
        assert qty == 100
        assert stop_price == round(32.50 * 0.94, 2)

    @pytest.mark.asyncio
    async def test_does_not_open_position_when_entry_dies(self):
        """If the entry order dies after submission, don't track a phantom
        position and don't place a stop for shares we don't hold."""
        broker = FakeBroker(fill_result=False)
        trader = _make_trader(broker)

        await trader._handle_buy("FNGU", 32.0, 1.0)

        assert not trader.pm.has_position("FNGU")
        assert broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_opens_defensively_when_fill_unconfirmed(self):
        """If the fill can't be confirmed in time, still open the position
        (in-process risk checks protect it) and attempt the stop."""
        broker = FakeBroker(fill_result=None)  # timeout
        trader = _make_trader(broker)

        await trader._handle_buy("FNGU", 32.0, 1.0)

        pos = trader.pm.get_positions().get("FNGU")
        assert pos is not None
        assert pos.entry_price == 32.0  # fell back to the signal price
        assert len(broker.stop_requests) == 1
