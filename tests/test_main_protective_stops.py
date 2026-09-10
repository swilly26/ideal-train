"""Regression tests for MAIN-trader broker-side protective stops.

The main trader previously held positions with NO broker-level stop: if every
process died (external process-group kill — watchdog, live_trader and
turbo_trader all share the session's process group), positions ran completely
unprotected until a manual relaunch.  Turbo already had this machinery; these
tests pin the port into ``live_trader.py``:

* ``_place_protective_stop`` — deterministic GTC stop at entry −6% (long) /
  entry +6% (short), wash-trade-safe retry with backoff, whole-share floor.
* ``_ensure_protective_stops`` — called at boot right after position sync;
  every inherited main symbol without an existing stop gets one placed.
* ``_handle_buy`` — attaches the stop immediately after the entry.
* ``_handle_sell`` / ``_eod_liquidate`` / ``_post_startup_cleanup`` — cancel
  the stop before closing (no orphaned GTC stops), defer when the cancel is
  not confirmed, and restore the stop if the close is rejected.
"""
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

import live_trader
from src.execution.broker import OrderResult, OrderSide
from src.execution.position_manager import PositionManager

WASH_TRADE_ERROR = Exception(
    '{"code":40310000,"message":"potential wash trade detected. use complex orders",'
    '"reject_reason":"opposite side market/stop order exists"}'
)


class FakeBroker:
    """In-memory broker for the main-trader protective-stop paths."""

    def __init__(self, open_orders_seq=None, stop_errors=(), buy_status="accepted",
                 sell_status="filled", fill_price=100.0):
        self._open_orders_seq = list(open_orders_seq or [])
        self._open_orders = []
        self._stop_errors = list(stop_errors)
        self.buy_status = buy_status
        self.sell_status = sell_status
        self.fill_price = fill_price
        self.stop_requests = []   # (symbol, qty, stop_price, client_id, side)
        self.cancelled_ids = []
        self.sell_orders = []

    # ── account / orders ────────────────────────────────────────────
    async def get_account(self):
        return {"equity": 200_000.0, "buying_power": 400_000.0,
                "cash": 100_000.0, "portfolio_value": 200_000.0}

    async def get_open_orders(self, symbol=None):
        if self._open_orders_seq:
            view = self._open_orders_seq.pop(0)
            if view is not None:
                self._open_orders = view
        if symbol is not None:
            return [o for o in self._open_orders if str(o.symbol).upper() == symbol.upper()]
        return list(self._open_orders)

    async def cancel_order_and_wait(self, order_id, timeout=10.0, poll_interval=0.25):
        self.cancelled_ids.append(order_id)
        self._open_orders = [o for o in self._open_orders if str(o.id) != str(order_id)]
        return True

    async def cancel_orders_by_client_id_prefix(self, prefix):
        return 0

    # ── order placement ─────────────────────────────────────────────
    async def place_stop_order(self, symbol, qty, stop_price, client_id=None, side="SELL"):
        if self._stop_errors:
            raise self._stop_errors.pop(0)
        self.stop_requests.append((symbol, qty, stop_price, client_id, side))
        return MagicMock(status="new")

    async def place_order(self, order):
        if getattr(order, "side", None) == OrderSide.BUY:
            status = self.buy_status
        else:
            status = self.sell_status
            self.sell_orders.append(order)
        filled = status in ("filled", "done_for_day")
        return OrderResult(
            order_id=f"order-{len(self.sell_orders) + 1}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity if filled else 0.0,
            filled_avg_price=self.fill_price if filled else None,
            status=status,
            error_message=None,
            created_at=datetime.now(timezone.utc),
        )

    async def wait_for_order_fill(self, order_id, timeout=8.0, poll_interval=0.25):
        return None  # timeout → stays pending (used by close_path tests that need pending)


def _make_trader(broker: FakeBroker):
    """Build a LiveTrader with only the attributes the tested paths touch."""
    trader = object.__new__(live_trader.LiveTrader)
    trader.broker = broker
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader.day_trades = []
    return trader


def _order_mock(symbol="AVGO", side="sell"):
    o = MagicMock()
    o.symbol = symbol
    o.side = side
    o.id = f"ord-{symbol}-{side}"
    return o


# ---------------------------------------------------------------------------
# _place_protective_stop — deterministic price, wash-trade-safe retry
# ---------------------------------------------------------------------------
class TestPlaceProtectiveStop:
    @pytest.mark.asyncio
    async def test_places_sell_stop_at_entry_minus_6pct(self):
        broker = FakeBroker(open_orders_seq=[[]])
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop("AVGO", 44, 100.0)
        assert ok is True
        assert len(broker.stop_requests) == 1
        symbol, qty, stop_price, client_id, side = broker.stop_requests[0]
        assert symbol == "AVGO"
        assert qty == 44
        assert side == "SELL"
        assert stop_price == round(100.0 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)
        assert client_id.startswith("algoflow_MAIN_AVGO_STOP_")

    @pytest.mark.asyncio
    async def test_floors_fractional_qty_to_whole_shares(self):
        broker = FakeBroker(open_orders_seq=[[]])
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop("AVGO", 44.8, 100.0)
        assert ok is True
        assert broker.stop_requests[0][1] == 44

    @pytest.mark.asyncio
    async def test_places_buy_stop_above_entry_for_short(self):
        broker = FakeBroker(open_orders_seq=[[]])
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop("TQQQ", 100, 50.0, is_short=True)
        assert ok is True
        symbol, qty, stop_price, _cid, side = broker.stop_requests[0]
        assert side == "BUY"
        assert stop_price == round(50.0 * (1 + live_trader.PROTECTIVE_STOP_PCT), 2)

    @pytest.mark.asyncio
    async def test_retries_after_wash_trade_rejection(self):
        broker = FakeBroker(open_orders_seq=[[]], stop_errors=[WASH_TRADE_ERROR])
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop(
            "AVGO", 100, 100.0, max_attempts=2, initial_delay=0.0,
        )
        assert ok is True
        assert len(broker.stop_requests) == 1  # second attempt succeeded

    @pytest.mark.asyncio
    async def test_waits_while_entry_buy_still_open(self):
        broker = FakeBroker(open_orders_seq=[[_order_mock("AVGO", "buy")], []])
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop(
            "AVGO", 100, 100.0, max_attempts=2, initial_delay=0.0,
        )
        assert ok is True
        assert len(broker.stop_requests) == 1

    @pytest.mark.asyncio
    async def test_skips_when_sell_stop_already_exists(self):
        broker = FakeBroker(open_orders_seq=[[_order_mock("AVGO", "sell")]])
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop("AVGO", 100, 100.0)
        assert ok is True
        assert broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_reports_failure_after_exhausting_attempts(self):
        broker = FakeBroker(
            open_orders_seq=[[]],
            stop_errors=[WASH_TRADE_ERROR, WASH_TRADE_ERROR],
        )
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop(
            "AVGO", 100, 100.0, max_attempts=2, initial_delay=0.0,
        )
        assert ok is False
        assert broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_skips_position_too_small(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        ok = await trader._place_protective_stop("AVGO", 0.4, 100.0)
        assert ok is False
        assert broker.stop_requests == []


# ---------------------------------------------------------------------------
# _ensure_protective_stops — boot coverage for inherited positions
# ---------------------------------------------------------------------------
class TestEnsureProtectiveStops:
    @pytest.mark.asyncio
    async def test_places_stop_for_inherited_position_without_stop(self):
        broker = FakeBroker(open_orders_seq=[[]])
        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", 44.8, 364.63)
        trader.pm.open_position("META", 24.9, 656.60)
        await trader._ensure_protective_stops()
        symbols = {r[0] for r in broker.stop_requests}
        assert symbols == {"AVGO", "META"}
        by_sym = {r[0]: r for r in broker.stop_requests}
        # floor to whole shares, stop at entry − 6%
        assert by_sym["AVGO"][1] == 44
        assert by_sym["AVGO"][2] == round(364.63 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)
        assert by_sym["META"][1] == 24
        assert by_sym["META"][2] == round(656.60 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)

    @pytest.mark.asyncio
    async def test_skips_symbol_already_covered_by_stop(self):
        broker = FakeBroker(open_orders_seq=[[_order_mock("AVGO", "sell")]])
        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", 44.8, 364.63)
        await trader._ensure_protective_stops()
        assert broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_ignores_non_main_symbols(self):
        """On the shared account, never place main stops on turbo symbols."""
        broker = FakeBroker(open_orders_seq=[[]])
        trader = _make_trader(broker)
        trader.pm.open_position("TZA", 1110.7, 42.90)   # turbo's symbol
        trader.pm.open_position("AVGO", 44.8, 364.63)   # main's symbol
        await trader._ensure_protective_stops()
        symbols = {r[0] for r in broker.stop_requests}
        assert symbols == {"AVGO"}

    @pytest.mark.asyncio
    async def test_noop_when_no_positions(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        await trader._ensure_protective_stops()
        assert broker.stop_requests == []


# ---------------------------------------------------------------------------
# _handle_buy — stop attached right after the entry is accepted
# ---------------------------------------------------------------------------
class TestHandleBuy:
    @pytest.mark.asyncio
    async def test_places_stop_after_entry_accepted(self):
        broker = FakeBroker(open_orders_seq=[[]])
        trader = _make_trader(broker)
        trader.pm.can_open = lambda symbol, equity: True  # keep sizing simple
        await trader._handle_buy("AVGO", 100.0, 0.9)
        pos = trader.pm.get_positions().get("AVGO")
        assert pos is not None
        assert len(broker.stop_requests) == 1
        symbol, qty, stop_price, _cid, side = broker.stop_requests[0]
        assert symbol == "AVGO"
        assert side == "SELL"
        assert stop_price == round(100.0 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)

    @pytest.mark.asyncio
    async def test_no_stop_when_entry_rejected(self):
        broker = FakeBroker(open_orders_seq=[[]], buy_status="rejected")
        trader = _make_trader(broker)
        await trader._handle_buy("AVGO", 100.0, 0.9)
        assert trader.pm.get_positions().get("AVGO") is None
        assert broker.stop_requests == []


# ---------------------------------------------------------------------------
# _handle_sell — cancel stop before closing; defer / restore on failure
# ---------------------------------------------------------------------------
class TestHandleSell:
    @pytest.mark.asyncio
    async def test_cancels_stop_before_closing(self):
        stop = _order_mock("AVGO", "sell")
        broker = FakeBroker(open_orders_seq=[[stop]], sell_status="filled")
        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", 44, 100.0)
        await trader._handle_sell("AVGO", 101.0, 0.9)
        assert broker.cancelled_ids == [stop.id]
        assert len(broker.sell_orders) == 1
        assert trader.pm.get_positions().get("AVGO") is None  # closed + P&L booked

    @pytest.mark.asyncio
    async def test_defers_close_when_cancellation_not_confirmed(self):
        """If the stop cannot be cancelled, do NOT close — position stays
        tracked AND protected (no naked position, no orphaned stop)."""
        stop = _order_mock("AVGO", "sell")

        class StickyBroker(FakeBroker):
            async def cancel_order_and_wait(self, order_id, timeout=10.0, poll_interval=0.25):
                return False  # cancellation never confirmed

        broker = StickyBroker(open_orders_seq=[[stop]], sell_status="filled")
        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", 44, 100.0)
        await trader._handle_sell("AVGO", 101.0, 0.9)
        assert broker.sell_orders == []
        assert trader.pm.get_positions().get("AVGO") is not None

    @pytest.mark.asyncio
    async def test_restores_stop_when_close_rejected(self):
        broker = FakeBroker(open_orders_seq=[[], []], sell_status="rejected")
        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", 44, 100.0)
        await trader._handle_sell("AVGO", 99.0, 0.9)
        # position kept AND stop re-placed
        assert trader.pm.get_positions().get("AVGO") is not None
        assert len(broker.stop_requests) == 1
        assert broker.stop_requests[0][2] == round(100.0 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)


# ---------------------------------------------------------------------------
# EOD liquidation — cancel stops before closing
# ---------------------------------------------------------------------------
class TestEodLiquidate:
    @pytest.mark.asyncio
    async def test_cancels_stops_before_eod_close(self):
        stop = _order_mock("AVGO", "sell")
        broker = FakeBroker(open_orders_seq=[[stop], []], sell_status="filled")

        async def fake_get_positions():
            return [{"symbol": "AVGO", "qty": "44.0", "avg_entry_price": "100.0"}]

        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", 44, 100.0)
        # exercise through the real method with a stubbed position source
        trader.broker.get_positions = fake_get_positions
        await trader._eod_liquidate()
        assert broker.cancelled_ids == [stop.id]
        assert len(broker.sell_orders) == 1


# ---------------------------------------------------------------------------
# run() wiring — protective stops are ensured at boot, right after sync
# ---------------------------------------------------------------------------
class TestRunWiring:
    @pytest.mark.asyncio
    async def test_run_ensures_stops_after_sync_and_cleanup(self, monkeypatch):
        trader = _make_trader(FakeBroker())
        order = []

        async def fake_startup_health_check():
            pass

        async def fake_cancel(prefix):
            return 0

        async def no_orders():
            return []

        async def fake_wait():
            raise KeyboardInterrupt

        async def fake_shutdown(close_broker=True):
            pass

        def record(name):
            async def _f(*a, **k):
                order.append(name)
            return _f

        trader.broker.startup_health_check = fake_startup_health_check
        trader.broker.cancel_orders_by_client_id_prefix = fake_cancel
        trader.broker.get_open_orders = no_orders
        trader._sync_positions_from_broker = record("sync")
        trader._post_startup_cleanup = record("cleanup")
        trader._ensure_protective_stops = record("stops")
        trader.wait_for_market_open = fake_wait
        trader._is_near_close = lambda: False
        trader.shutdown = fake_shutdown
        trader.pm.reset()
        with pytest.raises(KeyboardInterrupt):
            await trader.run()
        assert order == ["sync", "cleanup", "stops"], f"boot order was {order}"