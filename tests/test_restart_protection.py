"""A held position must never be unprotected across a restart.

Two defects proved themselves live on 2026-09-22/23 and are pinned here.

DEFECT 1 — positions left naked by our own restart logic (34h of exposure):
the boot-time "cancel stale orders" sweep cancelled the META and QQQ
protective stops, the broker pinned both PENDING_CANCEL (10s cancel timeout
each), and ``_ensure_protective_stops`` counted those *cancelling* orders as
coverage ("existing BUY stop found; not submitting duplicate").  The cancels
completed, nothing re-placed the stops, and both shorts sat with no
protective order at the broker until they were placed by hand.  The second
half of the same bug: anchoring a re-synced backstop on the ENTRY
(``entry * 1.06`` for a short) puts it on the wrong side of the market for a
position that has run against us (the live AVGO short: entry 337.75, market
361.10), the broker refuses it (42210000) and the fast-fail path leaves the
position bare.

Pinned behaviours:
1. coverage detection counts ONLY live, working, correctly-sided stops
   (never PENDING_CANCEL / cancelled / rejected / expired);
2. re-anchoring is side-aware and MARKET-anchored, never entry-anchored onto
   the wrong side of the market, and a rejected placement is retried rather
   than fast-failed into a naked position;
3. the startup sweep does not cancel the protective stops of inherited
   positions — the sweep and the protection check cannot disagree about the
   same order;
4. a sync pass that ends with a held position lacking a live stop emits a
   loud WARNING naming the symbol and what was attempted;
5. the "no protective stop found" warning is emitted only when a placement is
   actually attempted (it used to cry wolf on every healthy sync).

The fake broker models share RESERVATION (Alpaca holds the shares covered by
a working exit order: ``held_for_orders`` / 40310000) and the PENDING_CANCEL
lifecycle (a cancel that is not confirmed inside the 10s timeout stays
visible and keeps reserving its shares until the transition completes).
Order sides/statuses are modelled as alpaca-py str-enums
(``str(OrderSide.BUY) == "OrderSide.BUY"``) because the old coverage check
compared that string against ``"BUY"`` and therefore never matched a real
order object.  Nothing here touches the network or a real account.
"""
import pytest
from unittest.mock import MagicMock

import live_trader
from src.execution.position_manager import PositionManager


class EnumToken:
    """Mimics an alpaca-py str-enum (OrderSide.BUY, OrderStatus.NEW)."""

    def __init__(self, value, prefix="OrderSide"):
        self.value = value
        self._prefix = prefix

    def __str__(self):
        return f"{self._prefix}.{self.value.upper()}"

    def __repr__(self):
        return str(self)


def _side(value):
    return EnumToken(value, "OrderSide")


def _status(value):
    return EnumToken(value, "OrderStatus")


class FakeOrder:
    """Open-order object with a modellable PENDING_CANCEL lifecycle."""

    def __init__(self, oid, symbol, side_, order_type="stop", stop_price=None,
                 limit_price=None, client_order_id="", status_="new", qty=1):
        self.id = oid
        self.symbol = symbol
        self.side = side_ if not isinstance(side_, str) else _side(side_)
        self.type = order_type
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.client_order_id = client_order_id
        self.status = status_ if not isinstance(status_, str) else _status(status_)
        self.qty = qty


class RestartBroker:
    """In-memory broker modelling reservations and the cancel lifecycle.

    * a working order RESERVES the shares it covers (summed per symbol);
    * a stop submitted for reserved shares is refused with 40310000
      ("insufficient qty available ... held_for_orders"), exactly as Alpaca
      refuses a second exit order for the same shares;
    * a stop submitted on the wrong side of the last mark is refused with
      42210000 ("stop price must be [less|greater] than current price");
    * ``cancel_order_and_wait`` returns False (the 10s cancel timeout) and
      pins the order PENDING_CANCEL; it stays in the open-order book until
      its poll counter reaches zero, which is what kept the shares reserved
      while the protection check decided nothing had to be done.
    """

    def __init__(self, positions=None, open_orders=None):
        self.positions = {
            str(p["symbol"]).upper(): dict(p) for p in (positions or [])
        }
        self.open_orders = list(open_orders or [])
        self.stop_requests = []        # (symbol, qty, stop_price, client_id, side)
        self.cancel_calls = []
        self.rejections = []           # human-readable rejection log
        self.polls = 0
        self._pending = {}             # order id -> polls left before removal
        self._stuck = {}               # order id -> polls left (cancel timeouts)

    # ── helpers ─────────────────────────────────────────────────────
    def pin_pending_cancel(self, order, polls=2, stuck=True):
        """Model a cancel the broker pinned PENDING_CANCEL (not confirmed)."""
        order.status = _status("pending_cancel")
        if order not in self.open_orders:
            self.open_orders.append(order)
        self._pending[str(order.id)] = int(polls)
        if stuck:
            self._stuck[str(order.id)] = int(polls)
        return order

    def live_stops(self, symbol, is_short):
        return [
            o for o in self.open_orders
            if str(o.symbol).upper() == symbol.upper()
            and live_trader._order_is_protective_for(o, is_short)
        ]

    def reserved(self, symbol):
        return sum(
            int(o.qty) for o in self.open_orders
            if str(o.symbol).upper() == symbol.upper()
        )

    def _mark(self, symbol):
        row = self.positions.get(symbol.upper())
        if row is None:
            return None
        price = row.get("current_price")
        return float(price) if price else None

    # ── broker API ──────────────────────────────────────────────────
    async def get_positions(self):
        return [dict(p) for p in self.positions.values()]

    async def get_open_orders(self, symbol=None):
        self.polls += 1
        done = [oid for oid, left in self._pending.items() if left - 1 <= 0]
        for oid in done:
            self.open_orders = [o for o in self.open_orders if str(o.id) != oid]
            self._pending.pop(oid, None)
        for oid in list(self._pending):
            if oid not in done:
                self._pending[oid] -= 1
        if symbol is None:
            return list(self.open_orders)
        return [
            o for o in self.open_orders
            if str(o.symbol).upper() == symbol.upper()
        ]

    async def cancel_order_and_wait(self, order_id, timeout=10.0, poll_interval=0.25):
        self.cancel_calls.append(str(order_id))
        target = next(
            (o for o in self.open_orders if str(o.id) == str(order_id)), None,
        )
        if target is None:
            return True
        if str(order_id) in self._stuck:
            # Not confirmed inside the 10s timeout: the order stays visible as
            # PENDING_CANCEL and KEEPS RESERVING its shares until the
            # transition completes (this is what hid the naked positions).
            self.pin_pending_cancel(
                target, polls=self._stuck[str(order_id)], stuck=False,
            )
            return False
        self.open_orders = [o for o in self.open_orders if str(o.id) != str(order_id)]
        return True

    async def place_stop_order(self, symbol, qty, stop_price, client_id=None, side="SELL"):
        sym = symbol.upper()
        self.stop_requests.append((sym, int(qty), stop_price, client_id, side))
        market = self._mark(sym)
        if market is not None:
            wrong = (side == "BUY" and float(stop_price) <= market) or (
                side == "SELL" and float(stop_price) >= market
            )
            if wrong:
                self.rejections.append(
                    f"42210000 {sym} stop {stop_price} on wrong side of {market}"
                )
                raise Exception(
                    '{"code":42210000,"market_price":"%.2f","message":"stop price must '
                    'be %s than current price","stop_price":"%.2f"}'
                    % (market, "greater" if side == "BUY" else "less", float(stop_price))
                )
        row = self.positions.get(sym)
        if row is not None:
            held = abs(int(float(row.get("qty") or 0)))
            reserved = self.reserved(sym)
            if reserved > 0 and reserved + int(qty) > held:
                self.rejections.append(
                    f"40310000 {sym} qty={qty} reserved={reserved} held={held}"
                )
                raise Exception(
                    '{"available":"0","code":40310000,"existing_qty":"%d",'
                    '"held_for_orders":"%d","message":"insufficient qty available '
                    'for order (requested: %d, available: 0)","symbol":"%s"}'
                    % (held, reserved, int(qty), sym)
                )
        self.open_orders.append(FakeOrder(
            f"{sym}-stop-{len(self.stop_requests)}", sym,
            _side("buy" if side == "BUY" else "sell"),
            order_type="stop", stop_price=stop_price, client_order_id=client_id or "",
            status_="new", qty=int(qty),
        ))
        return MagicMock(status="new")


def _make_trader(broker):
    """LiveTrader with only the attributes these paths touch."""
    trader = object.__new__(live_trader.LiveTrader)
    trader.broker = broker
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader.day_trades = []
    return trader


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


# ── liveness classification ───────────────────────────────────────────────
def test_cancelling_and_terminal_statuses_are_never_working():
    for dead in ("pending_cancel", "canceled", "rejected", "expired"):
        o = FakeOrder("x", "META", "buy", status_=dead)
        assert live_trader._order_is_live_working(o) is False, dead
        assert live_trader._order_is_protective_for(o, True) is False, dead
    for alive in ("new", "accepted", "pending_new", "partially_filled"):
        o = FakeOrder("x", "META", "buy", status_=alive)
        assert live_trader._order_is_live_working(o) is True, alive
        assert live_trader._order_is_protective_for(o, True) is True, alive


def test_limit_orders_are_not_protection():
    o = FakeOrder("x", "META", "sell", order_type="limit", limit_price=800.0)
    assert live_trader._order_is_protective_for(o, False) is False


def test_market_anchored_stop_is_side_aware():
    assert live_trader._market_anchored_stop(361.10, True) == 382.77
    assert live_trader._market_anchored_stop(700.0, False) == 658.0
    assert live_trader._market_anchored_stop(None, True) is None


# ── defect 1 (a): a cancelling order is not coverage ──────────────────────
class TestCancellingOrderIsNotCoverage:
    @pytest.mark.asyncio
    async def test_pending_cancel_stop_is_not_coverage_and_the_position_is_reprotected(
        self, caplog,
    ):
        # META short -21 @ 750.06, market 736.50: the inherited BUY stop was
        # cancelled by the boot sweep and is pinned PENDING_CANCEL — it still
        # reserves the 21 shares and it is NOT protection.
        pinned = FakeOrder(
            "meta-stale-stop", "META", "buy", order_type="stop", stop_price=780.69,
            client_order_id="algoflow_MAIN_META_STOP_1", status_="pending_cancel",
            qty=21,
        )
        broker = RestartBroker(
            positions=[{"symbol": "META", "qty": -21, "avg_entry_price": 750.06,
                        "current_price": 736.50}],
        )
        broker.pin_pending_cancel(pinned, polls=3)
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._ensure_protective_stops()

        text = "\n".join(_messages(caplog))
        assert "existing BUY stop found; not submitting duplicate" not in text
        assert "already protected by live BUY stop" not in text
        # the cancelling order is named as NOT working
        assert "NOT working" in text and "PENDING_CANCEL" in text.upper()
        # a real stop WAS placed, after the reservation was released
        assert broker.rejections and "40310000" in broker.rejections[0]
        stops = broker.live_stops("META", is_short=True)
        assert len(stops) == 1
        # market 736.50 < entry 750.06, so the entry anchor (750.06 * 1.06) is
        # already on the protective side for this short — kept, not re-anchored
        assert stops[0].stop_price == pytest.approx(round(750.06 * 1.06, 2))
        assert live_trader._order_is_live_working(stops[0]) is True

    @pytest.mark.asyncio
    async def test_rejected_placement_does_not_cry_wolf_when_protection_is_live(
        self, caplog,
    ):
        live_stop = FakeOrder(
            "meta-live-stop", "META", "buy", order_type="stop", stop_price=780.69,
            client_order_id="algoflow_MAIN_META_STOP_2", status_="new", qty=21,
        )
        broker = RestartBroker(
            positions=[{"symbol": "META", "qty": -21, "avg_entry_price": 750.06,
                        "current_price": 736.50}],
            open_orders=[live_stop],
        )
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._ensure_protective_stops()

        text = "\n".join(_messages(caplog))
        assert "already protected by live BUY stop" in text
        assert "NO live protective stop" not in text
        assert "NO protective stop found" not in text
        assert broker.stop_requests == []
        assert len(broker.live_stops("META", is_short=True)) == 1


# ── defect 1 (b): market-anchored, side-aware backstop ────────────────────
class TestMarketAnchoredBackstop:
    @pytest.mark.asyncio
    async def test_short_backstop_uses_the_market_not_the_entry(self, caplog):
        # The live AVGO short: entry 337.75, market 361.10 -> entry + 6% =
        # 358.02 is BELOW the market; the broker refuses it and the old code
        # fast-failed, leaving the short with no stop at all.
        broker = RestartBroker(positions=[{
            "symbol": "AVGO", "qty": -48, "avg_entry_price": 337.745833,
            "current_price": 361.10,
        }])
        trader = _make_trader(broker)
        trader.pm.open_position("AVGO", -48, 337.745833)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._ensure_protective_stops()

        assert broker.rejections == []          # never even submitted wrongly
        stops = broker.live_stops("AVGO", is_short=True)
        assert len(stops) == 1
        assert stops[0].stop_price == pytest.approx(382.77)
        assert broker.stop_requests[0][3].startswith("algoflow_MAIN_AVGO_STOP_")
        text = "\n".join(_messages(caplog))
        assert "WRONG side of the live market" in text

    @pytest.mark.asyncio
    async def test_long_backstop_uses_the_market_not_the_entry(self):
        broker = RestartBroker(positions=[{
            "symbol": "NVDA", "qty": 12, "avg_entry_price": 750.0,
            "current_price": 700.0,
        }])
        trader = _make_trader(broker)
        trader.pm.open_position("NVDA", 12, 750.0)

        await trader._ensure_protective_stops()

        stops = broker.live_stops("NVDA", is_short=False)
        assert len(stops) == 1
        assert stops[0].stop_price == pytest.approx(658.0)

    @pytest.mark.asyncio
    async def test_invalid_level_retry_reanchors_on_the_market(self, caplog):
        """A wrong-side STRATEGY level is re-anchored instead of fast-failing."""
        broker = RestartBroker(positions=[{
            "symbol": "COIN", "qty": -10, "avg_entry_price": 200.0,
            "current_price": 220.0,
        }])
        trader = _make_trader(broker)

        with caplog.at_level("ERROR", logger="live_trader"):
            ok = await trader._place_protective_stop(
                "COIN", 10, 200.0, is_short=True, stop_price=205.0,
                initial_delay=0.0,
            )

        assert ok is True
        assert len(broker.stop_requests) == 2          # 1 rejected + 1 accepted
        assert broker.stop_requests[-1][2] == pytest.approx(233.2)  # 220 * 1.06
        text = "\n".join(_messages(caplog))
        assert "re-anchoring to $233.20" in text
        assert "must not stay unprotected" in text
        assert "COIN" not in getattr(trader, "_unprotected_symbols", set())


# ── defect 1 (c): the sweep and the protection check never disagree ───────
class TestStartupSweepKeepsProtection:
    @pytest.mark.asyncio
    async def test_sweep_keeps_stops_for_held_positions_and_cancels_only_stale_orders(
        self, caplog,
    ):
        meta_stop = FakeOrder(
            "meta-stop", "META", "buy", order_type="stop", stop_price=780.69,
            client_order_id="algoflow_MAIN_META_STOP_1", status_="new", qty=21,
        )
        stale_tp = FakeOrder(
            "tsla-tp", "TSLA", "sell", order_type="limit", limit_price=250.0,
            client_order_id="algoflow_MAIN_TSLA_TP_1", status_="new", qty=5,
        )
        manual = FakeOrder(
            "manual-stop", "QQQ", "buy", order_type="stop", stop_price=790.48,
            client_order_id="manual_QQQ_STOP_1", status_="new", qty=23,
        )
        broker = RestartBroker(
            positions=[{"symbol": "META", "qty": -21, "avg_entry_price": 750.06,
                        "current_price": 736.50}],
            open_orders=[meta_stop, stale_tp, manual],
        )
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)

        with caplog.at_level("INFO", logger="live_trader"):
            cancelled = await trader._cancel_stale_orders()

        assert cancelled == 1
        assert broker.cancel_calls == ["tsla-tp"]
        text = "\n".join(_messages(caplog))
        assert "KEEPING protective order" in text
        assert "kept 1 protective stop(s) for held positions: META" in text
        # the sweep did NOT touch the held position's stop or the manual one
        assert any(str(o.id) == "meta-stop" for o in broker.open_orders)
        assert any(str(o.id) == "manual-stop" for o in broker.open_orders)
        assert not any(str(o.id) == "tsla-tp" for o in broker.open_orders)

        # ... and the protection check agrees: nothing new is placed
        await trader._ensure_protective_stops()
        assert broker.stop_requests == []
        assert len(broker.live_stops("META", is_short=True)) == 1

    @pytest.mark.asyncio
    async def test_sweep_cancels_nothing_when_positions_cannot_be_read(self, caplog):
        class BlindBroker(RestartBroker):
            async def get_positions(self):
                raise RuntimeError("api timeout")

        stop = FakeOrder(
            "meta-stop", "META", "buy", order_type="stop", stop_price=780.69,
            client_order_id="algoflow_MAIN_META_STOP_1", status_="new", qty=21,
        )
        broker = BlindBroker(open_orders=[stop])
        trader = _make_trader(broker)

        with caplog.at_level("WARNING", logger="live_trader"):
            cancelled = await trader._cancel_stale_orders()

        assert cancelled == 0
        assert broker.cancel_calls == []
        assert "no order that looks like a protective stop will be cancelled" in \
            "\n".join(_messages(caplog))

    @pytest.mark.asyncio
    async def test_full_restart_leaves_exactly_one_live_stop_per_position(self, caplog):
        """The whole restart sequence, end to end: sweep -> sync -> protect."""
        meta_stop = FakeOrder(
            "meta-stop", "META", "buy", order_type="stop", stop_price=780.69,
            client_order_id="algoflow_MAIN_META_STOP_1", status_="new", qty=21,
        )
        qqq_cancelling = FakeOrder(
            "qqq-stop", "QQQ", "buy", order_type="stop", stop_price=790.48,
            client_order_id="algoflow_MAIN_QQQ_STOP_1", status_="pending_cancel",
            qty=23,
        )
        broker = RestartBroker(
            positions=[
                {"symbol": "META", "qty": -21, "avg_entry_price": 750.06,
                 "current_price": 736.50},
                {"symbol": "QQQ", "qty": -23, "avg_entry_price": 706.06,
                 "current_price": 745.74},
            ],
        )
        broker.pin_pending_cancel(qqq_cancelling, polls=3)
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)
        trader.pm.open_position("QQQ", -23, 706.06)

        with caplog.at_level("INFO", logger="live_trader"):
            await trader._cancel_stale_orders()      # sweep (keeps both stops)
            await trader._ensure_protective_stops()  # protection check

        for sym, market in (("META", 736.50), ("QQQ", 745.74)):
            live = broker.live_stops(sym, is_short=True)
            assert len(live) == 1, f"{sym} has {len(live)} live stops"
            assert live[0].stop_price > market, "stop must sit above a short's market"
        qqq = broker.live_stops("QQQ", is_short=True)[0]
        # QQQ's entry anchor (706.06 * 1.06 = 748.42) is still on the
        # protective side of its market (745.74), so it is kept as-is: the 6%
        # entry stop is what fires here, because this position is already past
        # it.  A market anchor is used only when the entry anchor would land
        # on the WRONG side of the market (the AVGO case above).
        assert qqq.stop_price == pytest.approx(round(706.06 * 1.06, 2))
        assert qqq.stop_price > 745.74
        # no cancelling/duplicate leftovers anywhere
        assert not [
            o for o in broker.open_orders
            if str(live_trader._order_status_token(o)) == "PENDING_CANCEL"
        ]


# ── defect 1 (d): a sync pass never ends silently naked ───────────────────
class TestProtectionAudit:
    @pytest.mark.asyncio
    async def test_audit_warns_loudly_and_heals_a_naked_position(self, caplog):
        broker = RestartBroker(positions=[{
            "symbol": "QQQ", "qty": -23, "avg_entry_price": 706.06,
            "current_price": 745.74,
        }])
        trader = _make_trader(broker)
        trader.pm.open_position("QQQ", -23, 706.06)

        with caplog.at_level("WARNING", logger="live_trader"):
            unprotected = await trader._audit_position_protection(context="position sync")

        assert unprotected == []
        text = "\n".join(_messages(caplog))
        assert "🚨 PROTECTION GAP (position sync): QQQ" in text
        assert "attempting placement now" in text
        stops = broker.live_stops("QQQ", is_short=True)
        assert len(stops) == 1
        # The audit protects the position FROM NOW, so its level is anchored on
        # the LIVE MARKET: 745.74 * 1.06 = 790.48 (owner decision, 2026-09-23).
        # The entry-anchored level (706.06 * 1.06 = 748.42) sits only 0.36%
        # above the market — it would fill on the next tick, i.e. it is an
        # immediate exit dressed up as a safety net, not protection.
        assert stops[0].stop_price == pytest.approx(round(745.74 * 1.06, 2))
        assert stops[0].stop_price > 745.74
        # ...and because this position has drifted out of its own risk band,
        # the audit says so loudly: whether to close it is a HUMAN decision,
        # while the market-anchored stop does the protecting.
        assert "BEYOND its entry-anchored" in text
        assert "a HUMAN should decide" in text

    @pytest.mark.asyncio
    async def test_audit_market_anchors_a_position_past_its_band(self, caplog):
        """A short that has run past entry * 1.06 gets a market-anchored stop.

        Its entry-anchored backstop (748.42) is BELOW the live market (800.00),
        so the broker would refuse it (42210000) and the position would stay
        naked.  The audit must anchor on the market and shout about the band.
        """
        broker = RestartBroker(positions=[{
            "symbol": "QQQ", "qty": -23, "avg_entry_price": 706.06,
            "current_price": 800.00,
        }])
        trader = _make_trader(broker)
        trader.pm.open_position("QQQ", -23, 706.06)
        with caplog.at_level("WARNING", logger="live_trader"):
            unprotected = await trader._audit_position_protection(context="position sync")
        assert unprotected == []
        text = "\n".join(_messages(caplog))
        assert "BEYOND its entry-anchored" in text
        assert "a HUMAN should decide" in text
        stops = broker.live_stops("QQQ", is_short=True)
        assert len(stops) == 1
        assert stops[0].stop_price == pytest.approx(round(800.00 * 1.06, 2))
        assert stops[0].stop_price > 800.00

    @pytest.mark.asyncio
    async def test_audit_reports_a_position_it_cannot_heal(self, caplog):
        class NoStopBroker(RestartBroker):
            async def place_stop_order(self, *a, **k):
                raise RuntimeError("broker down")

        broker = NoStopBroker(positions=[{
            "symbol": "META", "qty": -21, "avg_entry_price": 750.06,
            "current_price": 736.50,
        }])
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)

        with caplog.at_level("WARNING", logger="live_trader"):
            unprotected = await trader._audit_position_protection(context="position sync")

        assert unprotected == ["META"]
        text = "\n".join(_messages(caplog))
        assert "🚨 PROTECTION GAP (position sync): META" in text
        assert "STILL unprotected" in text

    @pytest.mark.asyncio
    async def test_audit_is_silent_when_everything_is_protected(self, caplog):
        stop = FakeOrder(
            "meta-stop", "META", "buy", order_type="stop", stop_price=780.69,
            client_order_id="algoflow_MAIN_META_STOP_1", status_="new", qty=21,
        )
        broker = RestartBroker(
            positions=[{"symbol": "META", "qty": -21, "avg_entry_price": 750.06,
                        "current_price": 736.50}],
            open_orders=[stop],
        )
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)

        with caplog.at_level("WARNING", logger="live_trader"):
            unprotected = await trader._audit_position_protection(context="position sync")

        assert unprotected == []
        assert _messages(caplog) == []
        assert broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_audit_refuses_to_place_a_stop_over_an_in_flight_order(self, caplog):
        working_close = FakeOrder(
            "meta-close", "META", "buy", order_type="market",
            client_order_id="algoflow_MAIN_META_EXIT_1", status_="new", qty=21,
        )
        broker = RestartBroker(
            positions=[{"symbol": "META", "qty": -21, "avg_entry_price": 750.06,
                        "current_price": 736.50}],
            open_orders=[working_close],
        )
        trader = _make_trader(broker)
        trader.pm.open_position("META", -21, 750.06)

        with caplog.at_level("WARNING", logger="live_trader"):
            unprotected = await trader._audit_position_protection(context="position sync")

        assert unprotected == ["META"]
        assert broker.stop_requests == []
        assert "another order is in flight" in "\n".join(_messages(caplog))
