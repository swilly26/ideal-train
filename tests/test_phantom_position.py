"""Phantom positions: never record, protect or BOOK P&L for a position the
broker never actually had (2026-09-23 open incident).

What happened live (verified against the broker's order history):

* 13:30:01.844 the broker CREATED the NVDA entry order; it only SUBMITTED it
  at 13:30:08.104 and never filled it.  TSLA (created 13:30:09.306, submitted
  13:30:18.947) and COIN (13:30:16.817 / 13:30:25.549) behaved the same way.
* The trader logged "Opened position" 6-8s BEFORE each order reached the
  broker, tried to place the protective stop against the still-working entry
  (Alpaca: 40310000 "potential wash trade detected"), and reported
  "SL=broker GTC stop x0".
* Its per-tick sync then found no such position at the broker, CANCELED the
  still-working entry orders (that is what produced `status=canceled,
  filled_qty=0`), and booked +102.90 / -80.34 / +75.22 of realised P&L priced
  from the PREVIOUS session's fills (NVDA 227.36, TSLA 376.96, COIN 200.11 --
  all filled 2026-09-22).

Every test here fails on the pre-fix code.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import live_trader
from src.execution.broker import Order, OrderResult, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.strategies.scalp.types import Direction, EntryType, ScalpSignal

NOW = datetime(2026, 9, 23, 13, 30, tzinfo=timezone.utc)


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


class FakeOrder:
    """Minimal open-order object for the protection/coverage checks."""

    def __init__(self, oid, symbol, side, order_type="stop", stop_price=None,
                 client_order_id="", status="new", qty=0):
        self.id = oid
        self.symbol = symbol
        self.side = side
        self.type = order_type
        self.stop_price = stop_price
        self.client_order_id = client_order_id
        self.status = status
        self.qty = qty


class PhantomBroker:
    """A broker that reproduces the incident: an entry that never fills."""

    def __init__(
        self,
        *,
        entry_status="accepted",
        entry_fill=None,
        poll_result=None,
        entry_order_id="entry-1",
        positions=None,
        open_orders=None,
        recent_fills=None,
        order_states=None,
        stale_fill_price=227.36,
        equity=200_000.0,
    ):
        self.entry_status = entry_status
        self.entry_fill = entry_fill        # (qty, price) when the entry DID fill
        self.poll_result = poll_result      # what wait_for_order_fill returns
        self.entry_order_id = entry_order_id
        self.positions = {p["symbol"]: p for p in (positions or [])}
        self._open_orders = list(open_orders or [])
        self.recent_fills = list(recent_fills or [])
        self.order_states = dict(order_states or {})
        self.stale_fill_price = stale_fill_price
        self.equity = equity
        self.stops = []            # (symbol, qty, stop_price, client_id, side)
        self.entries = []          # engine Orders
        self.cancelled = []
        self.order_seq = 0

    # ── account / positions / orders ────────────────────────────────
    async def get_account(self):
        return {"equity": self.equity, "buying_power": self.equity * 2,
                "cash": self.equity, "portfolio_value": self.equity,
                "available": True, "shorting_enabled": True}

    async def get_positions(self):
        return list(self.positions.values())

    async def get_open_orders(self, symbol=None):
        if symbol is not None:
            return [o for o in self._open_orders
                    if str(o.symbol).upper() == symbol.upper()]
        return list(self._open_orders)

    async def is_shortable(self, symbol):
        return True

    async def get_last_fill_price(self, symbol):
        """The PREVIOUS session's fill — the stale number that booked +102.90."""
        return self.stale_fill_price

    async def get_recent_fills(self, symbol, limit=20):
        return [o for o in self.recent_fills
                if str(getattr(o, "symbol", "")).upper() == symbol.upper()][:limit]

    async def get_order(self, order_id):
        return self.order_states.get(str(order_id))

    async def place_order(self, order: Order):
        if order.order_type == OrderType.LIMIT:
            return OrderResult(
                order_id="tp-1", symbol=order.symbol, side=order.side,
                quantity=order.quantity, filled_quantity=0.0,
                filled_avg_price=None, status="accepted",
                created_at=NOW,
            )
        self.order_seq += 1
        self.entries.append(order)
        filled_qty, avg, status = 0.0, None, self.entry_status
        if self.entry_fill is not None:
            filled_qty, avg = self.entry_fill
            status = "filled"
        return OrderResult(
            order_id=self.entry_order_id, symbol=order.symbol, side=order.side,
            quantity=order.quantity, filled_quantity=filled_qty,
            filled_avg_price=avg, status=status, created_at=NOW,
            filled_at=NOW if status == "filled" else None,
        )

    async def wait_for_order_fill(self, order_id, timeout=8.0, poll_interval=0.25):
        return self.poll_result

    async def place_stop_order(self, symbol, qty, stop_price, client_id=None, side="SELL"):
        self.stops.append((symbol, qty, stop_price, client_id, side))
        oid = f"stop-{len(self.stops)}"
        self._open_orders.append(
            FakeOrder(oid, symbol, side, order_type="stop", stop_price=stop_price,
                      client_order_id=client_id or "", qty=qty)
        )
        return MagicMock(id=oid, status="new")

    async def cancel_order_and_wait(self, order_id, timeout=10.0, poll_interval=0.25):
        self.cancelled.append(str(order_id))
        self._open_orders = [o for o in self._open_orders if str(o.id) != str(order_id)]
        return True

    async def cancel_orders_by_client_id_prefix(self, prefix):
        return 0


def _trader(broker, **state):
    trader = object.__new__(live_trader.LiveTrader)
    trader.broker = broker
    trader.provider = MagicMock()
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader._entry_times = dict(state.pop("entry_times", {}))
    trader.day_trades = []
    trader.start_equity = 0.0
    trader._main_strategy = "scalp"
    trader._scalp_init_state()
    trader._scalp_positions = dict(state.pop("scalp_positions", {}))
    for key, value in state.items():
        setattr(trader, key, value)
    return trader


def _sig(symbol, direction, entry, sl, tp, strategy="box_theory"):
    return ScalpSignal(
        symbol=symbol, timestamp=pd.Timestamp("2026-09-23 13:30:00", tz="UTC"),
        direction=direction, entry_type=EntryType.MARKET, entry_price=entry,
        stop_loss=sl, take_profit=tp, risk=abs(entry - sl), reward=abs(tp - entry),
        rr=round(abs(tp - entry) / abs(entry - sl), 4), strategy=strategy,
        breakeven_trigger_r=1.0, trailing=True, trail_distance_r=1.0,
        trail_trigger_r=1.0,
    )


def _filled_order(symbol, side, qty, price, *, oid="fill-1", when=None):
    return OrderResult(
        order_id=oid, symbol=symbol, side=side, quantity=qty,
        filled_quantity=qty, filled_avg_price=price, status="filled",
        created_at=when or NOW, filled_at=when or NOW,
    )


# ---------------------------------------------------------------------------
# A1. An entry becomes a position ONLY on a broker-verified fill
# ---------------------------------------------------------------------------
class TestEntryRequiresVerifiedFill:
    @pytest.mark.asyncio
    async def test_entry_never_filled_records_no_position_no_stop_no_pnl(
        self, caplog,
    ):
        """The 13:30 NVDA case: accepted, still working, never filled."""
        broker = PhantomBroker(entry_status="accepted", poll_result=None)
        trader = _trader(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            ok = await trader._scalp_enter(
                _sig("NVDA", Direction.SHORT, 100.0, 101.0, 94.0), 101.0)

        assert ok is False
        # no position, no slot consumed, no protective order, no P&L
        assert not trader.pm.has_position("NVDA")
        assert trader.pm.get_positions() == {}
        assert trader.pm.get_realized_pnl() == 0.0
        assert broker.stops == []
        assert trader._scalp_entries_today("NVDA") == 0
        assert trader._entry_order_ids.get("NVDA") is None
        # the working entry is cancelled, never left to fill unmanaged
        assert broker.cancelled == ["entry-1"]
        text = "\n".join(_messages(caplog))
        assert "🚨 INCIDENT SCALP NVDA: entry order entry-1" in text
        assert "NEVER FILLED" in text
        assert "Opened position" not in text

    @pytest.mark.asyncio
    async def test_entry_filled_during_the_wait_is_recorded_at_the_real_fill(
        self, caplog,
    ):
        """Still working at submission, broker confirms later → real position."""
        # requested ~40 shares; the broker fills only 37 at 101.25
        broker = PhantomBroker(
            entry_status="accepted",
            poll_result=_filled_order("NVDA", OrderSide.BUY, 37.0, 101.25),
        )
        trader = _trader(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            ok = await trader._scalp_enter(
                _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.0), 101.0)

        assert ok is True
        pos = trader.pm.get_positions()["NVDA"]
        assert pos.quantity == pytest.approx(37.0)      # the VERIFIED qty
        assert pos.entry_price == pytest.approx(101.25)  # the VERIFIED price
        # the protective order is for the verified quantity only
        assert len(broker.stops) == 1
        assert broker.stops[0][1] == 37
        assert broker.stops[0][4] == "SELL"
        assert broker.cancelled == []                   # nothing to abandon
        text = "\n".join(_messages(caplog))
        assert "broker CONFIRMED the entry fill" in text
        assert "broker-verified fill" in text

    @pytest.mark.asyncio
    async def test_entry_filling_while_being_abandoned_is_adopted_and_protected(
        self, caplog,
    ):
        """The cancel races the fill: the position IS there and must be protected."""
        broker = PhantomBroker(
            entry_status="accepted",
            poll_result=None,                       # our wait timed out
            order_states={
                "entry-1": _filled_order("NVDA", OrderSide.BUY, 40.0, 100.5),
            },
        )
        trader = _trader(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            ok = await trader._scalp_enter(
                _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.0), 100.5)

        assert ok is True
        pos = trader.pm.get_positions()["NVDA"]
        assert pos.quantity == pytest.approx(40.0)
        assert pos.entry_price == pytest.approx(100.5)
        assert len(broker.stops) == 1
        text = "\n".join(_messages(caplog))
        assert "FILLED" in text and "while it was being abandoned" in text

    @pytest.mark.asyncio
    async def test_rejected_entry_never_becomes_a_position(self):
        broker = PhantomBroker(entry_status="rejected")
        trader = _trader(broker)
        ok = await trader._scalp_enter(
            _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.0), 101.0)
        assert ok is False
        assert not trader.pm.has_position("NVDA")
        assert trader.pm.get_realized_pnl() == 0.0


# ---------------------------------------------------------------------------
# A2. "Vanished" is booked only against a verified execution
# ---------------------------------------------------------------------------
class TestPhantomVanish:
    def _tracked(self, broker, *, qty=30.0, entry=100.0, entry_order_id="entry-1"):
        trader = _trader(
            broker,
            _entry_times={"NVDA": NOW - timedelta(minutes=1)},
            _entry_order_ids={"NVDA": entry_order_id},
        )
        trader.pm.open_position("NVDA", qty, entry)
        return trader

    @pytest.mark.asyncio
    async def test_vanish_of_an_entry_that_never_filled_books_nothing(
        self, caplog,
    ):
        """The tracker believed in a position the broker NEVER HAD."""
        broker = PhantomBroker(
            positions={},
            order_states={
                "entry-1": OrderResult(
                    order_id="entry-1", symbol="NVDA", side=OrderSide.SELL,
                    quantity=30.0, filled_quantity=0.0, filled_avg_price=None,
                    status="canceled", created_at=NOW,
                ),
            },
            stale_fill_price=227.36,     # the previous session's fill
        )
        trader = self._tracked(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._scalp_sync_positions()

        assert not trader.pm.has_position("NVDA")
        assert trader.pm.get_realized_pnl() == 0.0     # NOT 227.36-priced
        assert trader.pm._closed_trades == []
        text = "\n".join(_messages(caplog))
        assert "🚨 INCIDENT SCALP NVDA" in text
        assert "a position the broker NEVER HAD" in text
        assert "NEVER FILLED" in text
        assert "entry-1" in text
        assert "Closed NVDA:" not in text

    @pytest.mark.asyncio
    async def test_vanish_without_any_execution_is_a_different_incident(
        self, caplog,
    ):
        """Entry DID fill, but nothing prices the exit → loud, no invented P&L."""
        broker = PhantomBroker(
            positions={},
            recent_fills=[],
            order_states={"entry-1": _filled_order("NVDA", OrderSide.SELL, 30.0, 100.0)},
            stale_fill_price=227.36,
        )
        trader = self._tracked(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._scalp_sync_positions()

        assert not trader.pm.has_position("NVDA")
        assert trader.pm.get_realized_pnl() == 0.0
        text = "\n".join(_messages(caplog))
        assert "🚨 INCIDENT SCALP NVDA" in text
        assert "DISAPPEARED from the broker with NO verified execution" in text
        assert "REALISED P&L NOT BOOKED" in text
        # distinguishable from the never-filled case
        assert "NEVER HAD" not in text

    @pytest.mark.asyncio
    async def test_vanish_with_a_verified_closing_fill_books_that_fill(
        self, caplog,
    ):
        """A real broker execution prices the exit — never the stale history."""
        broker = PhantomBroker(
            positions={},
            order_states={"entry-1": _filled_order("NVDA", OrderSide.BUY, 30.0, 100.0)},
            recent_fills=[
                # previous session's trade for the same symbol
                _filled_order("NVDA", OrderSide.SELL, 71.0, 227.36,
                              oid="old-fill", when=NOW - timedelta(days=1)),
                # the execution that actually closed THIS position
                _filled_order("NVDA", OrderSide.SELL, 30.0, 106.0,
                              oid="exit-fill", when=NOW + timedelta(minutes=4)),
            ],
            stale_fill_price=999.0,
        )
        trader = self._tracked(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._scalp_sync_positions()

        assert not trader.pm.has_position("NVDA")
        assert trader.pm.get_realized_pnl() == pytest.approx(30.0 * (106.0 - 100.0))
        trades = trader.pm._closed_trades
        assert len(trades) == 1 and trades[0].exit_price == pytest.approx(106.0)
        text = "\n".join(_messages(caplog))
        assert "VERIFIED exit fill $106.00" in text
        assert "INCIDENT" not in text

    @pytest.mark.asyncio
    async def test_vanish_cleanup_never_cancels_a_working_entry_order(
        self, caplog,
    ):
        """The 13:30 cancellation of the trader's own entry must not recur."""
        working_entry = FakeOrder(
            "entry-1", "NVDA", "sell", order_type="market",
            client_order_id="algoflow_MAIN_NVDA_ENTRY_SELL_10168280735420",
            status="new", qty=70,
        )
        protective = FakeOrder(
            "nvda-stop", "NVDA", "buy", order_type="stop", stop_price=107.0,
            client_order_id="algoflow_MAIN_NVDA_STOP_1", status="new", qty=70,
        )
        broker = PhantomBroker(
            positions={},
            open_orders=[working_entry, protective],
            order_states={"entry-1": _filled_order("NVDA", OrderSide.SELL, 70.0, 100.0)},
            recent_fills=[_filled_order("NVDA", OrderSide.BUY, 70.0, 99.0,
                                        oid="exit-1", when=NOW)],
        )
        trader = self._tracked(broker, qty=-70.0)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._scalp_sync_positions()

        assert broker.cancelled == ["nvda-stop"]
        remaining = {str(o.id) for o in broker._open_orders}
        assert "entry-1" in remaining, "a working ENTRY order was cancelled"
        assert "nvda-stop" not in remaining
        assert "KEEPING order entry-1" in "\n".join(_messages(caplog))


# ---------------------------------------------------------------------------
# A3. Never a protective order for an unverified / zero quantity
# ---------------------------------------------------------------------------
class TestNoZeroShareStops:
    @pytest.mark.asyncio
    async def test_stop_for_zero_shares_is_a_loud_incident_not_a_noop(
        self, caplog,
    ):
        broker = PhantomBroker()
        trader = _trader(broker)

        with caplog.at_level("INFO", logger="live_trader"):
            placed = await trader._place_protective_stop("NVDA", 0, 100.0)

        assert placed is False
        assert broker.stops == []            # nothing reached the broker
        text = "\n".join(_messages(caplog))
        assert "🚨 INCIDENT STOP NVDA" in text
        assert "refusing to submit a protective stop for qty=0" in text
        assert "zero shares protect nothing" in text

    @pytest.mark.asyncio
    async def test_an_adopted_short_gets_its_buy_stop_at_the_market(
        self,
    ):
        """Adoption must read the SIGNED broker quantity: a short needs a BUY
        stop above the market, and `int(qty)` must not be passed as negative."""
        broker = PhantomBroker(positions=[{
            "symbol": "NVDA", "qty": -30, "avg_entry_price": 100.0,
            "current_price": 101.0,
        }])
        trader = _trader(broker)

        await trader._scalp_sync_positions()

        assert trader.pm.get_positions()["NVDA"].quantity == -30
        assert len(broker.stops) == 1
        sym, qty, stop_price, _cid, side = broker.stops[0]
        assert (sym, qty, side) == ("NVDA", 30, "BUY")
        assert stop_price == pytest.approx(round(101.0 * 1.06, 2))
