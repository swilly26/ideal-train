"""Regression tests for verified-fill close accounting (P1 phantom-fill fix).

The engine previously booked realised P&L from the order-submission response
alone: a cleanup MARKET SELL that sat NEW/ACCEPTED pre-open was treated as a
fill, logged "Closed ... P&L=...", and claimed "N position(s) liquidated" while
the broker realised thousands more in losses (incidents 2026-09-08/09).

This suite pins the new contract:

* P&L is booked ONLY after the broker confirms a fill (``filled`` /
  ``done_for_day``), at the broker's average fill price — never the last mark.
* An unconfirmed close stays ``pending``: position tracked, no P&L booked,
  and a second close attempt for the same symbol is NOT re-submitted.
* When a tracked position vanishes between syncs, the position sync books it
  at ``broker.get_last_fill_price`` when one exists, and otherwise drops it
  WITHOUT booking P&L (never a fabricated loss).
"""
import pytest
from datetime import datetime, timezone

from src.execution.broker import OrderResult, OrderSide
from src.execution.position_manager import PositionManager
from src.execution.verified_close import CloseOutcome, close_position_verified

import live_trader
import turbo_trader


# ---------------------------------------------------------------------------
# Configurable in-memory broker
# ---------------------------------------------------------------------------


class FakeBroker:
    """Broker stand-in whose order lifecycle is scripted per test.

    ``status`` is the status returned by ``place_order``; ``wait_result``
    controls ``wait_for_order_fill`` ("filled" / "terminal" / "timeout").
    """

    def __init__(self, status="filled", fill_price=105.0, error_message=None,
                 wait_result="timeout"):
        self.status = status
        self.fill_price = fill_price
        self.error_message = error_message
        self.wait_result = wait_result
        self.orders: list = []
        self._next_id = 1

    async def place_order(self, order):
        self.orders.append(order)
        oid = f"ord_{self._next_id}"
        self._next_id += 1
        filled = self.status in ("filled", "done_for_day")
        return OrderResult(
            order_id=oid,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity if filled else 0,
            filled_avg_price=self.fill_price if filled else None,
            status=self.status,
            error_message=self.error_message,
            created_at=datetime.now(timezone.utc),
        )

    async def wait_for_order_fill(self, order_id, timeout=8.0, poll_interval=0.25):
        if self.wait_result == "filled":
            last = self.orders[-1]
            return OrderResult(
                order_id=order_id, symbol=last.symbol, side=last.side,
                quantity=last.quantity, filled_quantity=last.quantity,
                filled_avg_price=self.fill_price, status="filled",
                created_at=datetime.now(timezone.utc),
            )
        if self.wait_result == "terminal":
            return False
        return None  # timeout — still pending

    # Protective-stop interface (live_trader._handle_sell cancels stops
    # before closing; this stand-in holds no stop orders, so the cancel is
    # a trivially-confirmed no-op).
    async def get_open_orders(self, symbol=None):
        return []

    async def cancel_order_and_wait(self, order_id, timeout=10.0, poll_interval=0.25):
        return True


def _pm_with(symbol, qty, entry):
    pm = PositionManager()
    pm.open_position(symbol, qty, entry)
    return pm


# ---------------------------------------------------------------------------
# (a) Pending close: no P&L, position tracked, no duplicate sell
# ---------------------------------------------------------------------------


class TestPendingClose:
    @pytest.mark.asyncio
    async def test_accepted_no_fill_books_no_pnl_and_stays_tracked(self):
        broker = FakeBroker(status="accepted", wait_result="timeout")
        pm = _pm_with("NVDA", 10, 100.0)

        outcome = await close_position_verified(pm, broker, "NVDA", wait_timeout=0.5)

        assert outcome.status == "pending"
        assert outcome.confirmed is False
        assert outcome.fill_price is None
        assert pm.has_position("NVDA") is True
        assert pm.get_realized_pnl() == 0.0
        assert pm.has_pending_close("NVDA") is True
        assert len(broker.orders) == 1

    @pytest.mark.asyncio
    async def test_pending_close_not_resubmitted(self):
        broker = FakeBroker(status="new", wait_result="timeout")
        pm = _pm_with("NVDA", 10, 100.0)

        first = await close_position_verified(pm, broker, "NVDA", wait_timeout=0.5)
        assert first.status == "pending"

        second = await close_position_verified(pm, broker, "NVDA", wait_timeout=0.5)

        assert second.status == "pending"
        assert "not resubmitting" in second.message
        assert len(broker.orders) == 1  # no duplicate sell

    @pytest.mark.asyncio
    async def test_terminal_failure_keeps_position_tracked(self):
        broker = FakeBroker(status="accepted", wait_result="terminal")
        pm = _pm_with("NVDA", 10, 100.0)

        outcome = await close_position_verified(pm, broker, "NVDA", wait_timeout=0.5)

        assert outcome.status == "rejected"
        assert outcome.confirmed is False
        assert pm.has_position("NVDA") is True
        assert pm.get_realized_pnl() == 0.0

    @pytest.mark.asyncio
    async def test_submission_rejection_books_no_pnl(self):
        broker = FakeBroker(status="rejected", error_message="insufficient buying power")
        pm = _pm_with("NVDA", 10, 100.0)

        outcome = await close_position_verified(pm, broker, "NVDA")

        assert outcome.status == "rejected"
        assert outcome.confirmed is False
        assert pm.has_position("NVDA") is True
        assert pm.get_realized_pnl() == 0.0


# ---------------------------------------------------------------------------
# (b) Confirmed fill: P&L booked at the broker's fill price, never the mark
# ---------------------------------------------------------------------------


class TestConfirmedFill:
    @pytest.mark.asyncio
    async def test_filled_on_submission_books_pnl_at_fill_price(self):
        broker = FakeBroker(status="filled", fill_price=107.5)
        pm = _pm_with("NVDA", 10, 100.0)
        pm.update_price("NVDA", 103.0)  # the mark — must NOT be used

        outcome = await close_position_verified(pm, broker, "NVDA")

        assert outcome.status == "filled"
        assert outcome.confirmed is True
        assert outcome.fill_price == 107.5
        assert not pm.has_position("NVDA")
        # (107.5 - 100) * 10, not (103 - 100) * 10
        assert pm.get_realized_pnl() == pytest.approx(75.0)
        assert not pm.has_pending_close("NVDA")

    @pytest.mark.asyncio
    async def test_filled_via_poll_books_pnl_at_fill_price(self):
        broker = FakeBroker(status="accepted", wait_result="filled", fill_price=106.0)
        pm = _pm_with("NVDA", 10, 100.0)

        outcome = await close_position_verified(pm, broker, "NVDA", wait_timeout=1.0)

        assert outcome.status == "filled"
        assert pm.get_realized_pnl() == pytest.approx(60.0)  # (106 - 100) * 10
        assert not pm.has_position("NVDA")

    @pytest.mark.asyncio
    async def test_short_cover_books_pnl_at_fill_price(self):
        broker = FakeBroker(status="filled", fill_price=95.0)
        pm = _pm_with("LABD", -10, 100.0)  # short: entered 100, covered 95

        outcome = await close_position_verified(pm, broker, "LABD")

        assert outcome.status == "filled"
        assert outcome.trade is not None
        assert outcome.trade.quantity == -10
        assert pm.get_realized_pnl() == pytest.approx(50.0)  # (100 - 95) * 10
        assert broker.orders[0].side == OrderSide.BUY  # buy-to-cover


# ---------------------------------------------------------------------------
# (c)/(d) Position sync: vanished positions resolve via broker fill history
# ---------------------------------------------------------------------------


class _SyncBroker:
    """Broker that reports no positions and a scripted last-fill price."""

    def __init__(self, fill_price):
        self.fill_price = fill_price

    async def get_positions(self):
        return []

    async def get_last_fill_price(self, symbol):
        return self.fill_price


def _live_trader_with_pm(pm, broker):
    trader = object.__new__(live_trader.LiveTrader)
    trader.pm = pm
    trader.broker = broker
    return trader


def _turbo_trader_with_pm(pm, broker):
    trader = object.__new__(turbo_trader.TurboTrader)
    trader.pm = pm
    trader.broker = broker
    return trader


class TestSyncRemovalAccounting:
    @pytest.mark.asyncio
    async def test_sync_books_pnl_at_last_fill_price(self):
        broker = _SyncBroker(fill_price=109.0)
        trader = _live_trader_with_pm(_pm_with("META", 5, 100.0), broker)

        await trader._sync_positions_from_broker()

        assert not trader.pm.has_position("META")
        assert trader.pm.get_realized_pnl() == pytest.approx((109.0 - 100.0) * 5)

    @pytest.mark.asyncio
    async def test_sync_drops_without_pnl_when_no_fill_price(self):
        broker = _SyncBroker(fill_price=None)
        trader = _live_trader_with_pm(_pm_with("META", 5, 100.0), broker)

        await trader._sync_positions_from_broker()

        assert not trader.pm.has_position("META")
        assert trader.pm.get_realized_pnl() == 0.0  # never a fabricated loss

    @pytest.mark.asyncio
    async def test_turbo_sync_only_resolves_turbo_symbols(self):
        broker = _SyncBroker(fill_price=104.0)
        pm = PositionManager()
        pm.open_position("TQQQ", 8, 100.0)  # turbo symbol
        pm.open_position("META", 5, 100.0)  # main-trader symbol — untouchable
        trader = _turbo_trader_with_pm(pm, broker)

        await trader._sync_positions_from_broker()

        assert not trader.pm.has_position("TQQQ")
        assert trader.pm.has_position("META") is True
        assert trader.pm.get_realized_pnl() == pytest.approx((104.0 - 100.0) * 8)


# ---------------------------------------------------------------------------
# Wiring: live/turbo _handle_sell honour the verified-close contract
# ---------------------------------------------------------------------------


def _make_live_trader(broker):
    trader = object.__new__(live_trader.LiveTrader)
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader.broker = broker
    trader._entry_times = {}
    return trader


class TestLiveHandleSellWiring:
    @pytest.mark.asyncio
    async def test_pending_sell_keeps_position_and_books_nothing(self):
        broker = FakeBroker(status="accepted", wait_result="timeout")
        trader = _make_live_trader(broker)
        trader.pm.open_position("NVDA", 10, 100.0)
        trader._entry_times["NVDA"] = datetime.now(timezone.utc)

        await trader._handle_sell("NVDA", 103.0, 1.0)  # mark 103 must not leak

        assert trader.pm.has_position("NVDA") is True
        assert trader.pm.get_realized_pnl() == 0.0
        assert "NVDA" in trader._entry_times  # still held → still timed
        assert len(broker.orders) == 1

    @pytest.mark.asyncio
    async def test_confirmed_sell_books_at_fill_not_mark(self):
        broker = FakeBroker(status="filled", fill_price=107.0)
        trader = _make_live_trader(broker)
        trader.pm.open_position("NVDA", 10, 100.0)
        trader._entry_times["NVDA"] = datetime.now(timezone.utc)

        await trader._handle_sell("NVDA", 103.0, 1.0)

        assert not trader.pm.has_position("NVDA")
        assert trader.pm.get_realized_pnl() == pytest.approx(70.0)  # (107-100)*10
        assert "NVDA" not in trader._entry_times


# ---------------------------------------------------------------------------
# PositionManager pending-close marker lifecycle
# ---------------------------------------------------------------------------


class TestPendingCloseMarkers:
    def test_marker_lifecycle(self):
        pm = PositionManager()
        pm.mark_pending_close("nvda")
        assert pm.has_pending_close("NVDA") is True
        pm.clear_pending_close("NVDA")
        assert pm.has_pending_close("NVDA") is False

    def test_close_position_clears_marker(self):
        pm = _pm_with("NVDA", 10, 100.0)
        pm.mark_pending_close("NVDA")
        pm.close_position("NVDA", 105.0)
        assert pm.has_pending_close("NVDA") is False

    def test_discard_position_clears_marker(self):
        pm = _pm_with("NVDA", 10, 100.0)
        pm.mark_pending_close("NVDA")
        pm.discard_position("NVDA", reason="sync_removed_no_fill")
        assert pm.has_pending_close("NVDA") is False

    def test_reset_clears_markers(self):
        pm = _pm_with("NVDA", 10, 100.0)
        pm.mark_pending_close("NVDA")
        pm.reset()
        assert pm.has_pending_close("NVDA") is False