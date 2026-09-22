"""Live exit-structure regression tests (2026-09-22).

Live defect being pinned (evidence: ``logs/trades_20260921.log``):
* 19:52:13Z META short 21 — GTC BUY stop resting for 21 shares, then the
  full-qty day-limit TP was rejected
  ``{"available":"0","code":40310000,"held_for_orders":"21"}``;
* 19:52:17Z TSLA long 42.92623498 — the stop was placed for the WHOLE-SHARE
  quantity 42 (Alpaca refuses fractional GTC stops), so the fractional TP had
  ``"available":"0.92623498"`` and was rejected the same way.

The old code reacted to that rejection with ``state["tp"] = None`` — i.e. it
threw the upside exit away, making every live position downside-only.  The
fix keeps the level and monitors it trader-side (see
``src/execution/exit_structure.py`` for the broker evidence behind the
chosen structure).

Hermetic: the in-memory fakes from ``tests/test_live_scalp_integration.py``
are reused; nothing here touches the network or an account.
"""
import asyncio
import json

import pytest

import live_trader
from src.execution.exit_structure import (
    format_exit_plan,
    plan_exit_structure,
    stop_capacity,
)
from src.execution.broker import OrderType
from src.strategies.scalp.types import Direction
from tests.test_live_scalp_integration import (
    FakeBroker,
    FakeProvider,
    _frame_df,
    _make_scalp_trader,
    _sig,
)

SYM = "TSLA"  # a live fractional long: 42.92623498 shares
FRACTIONAL_QTY = 42.92623498
FILL = 374.01
SL = 370.50
TP = 380.00

# The exact rejection Alpaca returns for a second order on shares a resting
# stop already reserves (copied from the live log, 2026-09-21 19:52:13Z).
TP_REJECTED = json.dumps({
    "available": "0", "code": 40310000, "existing_qty": "42",
    "held_for_orders": "42", "symbol": SYM,
    "message": "insufficient qty available for order (requested: 42.92623498, "
               "available: 0.92623498)",
})


class StopReservesQtyBroker(FakeBroker):
    """Paper-accurate enough: a resting GTC stop reserves whole shares.

    Alpaca reserves (``held_for_orders``) the quantity of every resting
    order, so an opposite-side DAY limit for the FULL position is rejected
    with 40310000 while our whole-share stop rests.  Reproduced live on
    META/TSLA/COIN/NVDA on 2026-09-21 and re-verified against the paper API
    on 2026-09-22.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tp_rejections = []

    async def place_order(self, order):
        if order.order_type == OrderType.LIMIT:
            held = sum(
                float(q) for (sym, q, _p, _c, _side) in self.stop_requests
                if sym.upper() == order.symbol.upper()
            )
            if held > 0:
                msg = TP_REJECTED.replace('"42"', f'"{held:g}"')
                self.tp_rejections.append(msg)
                raise Exception(msg)
        return await super().place_order(order)


class StopFailingBroker(FakeBroker):
    """No broker stop can be placed (wrong-side level, 42210000).

    Mirrors the live AVGO case (2026-09-21 19:52Z: a short whose backstop
    stop sat below the market, rejected permanently).  With nothing resting,
    a FULL-quantity DAY-limit TP can rest — the one case where the broker
    owns the upside exit.
    """

    async def place_stop_order(self, symbol, qty, stop_price, client_id=None,
                               side="SELL"):
        self.stop_requests.append((symbol, qty, stop_price, client_id, side))
        raise Exception(
            '{"code":42210000,"market_price":"363.22","message":"stop price '
            'must be greater than current price","stop_price":"358.01"}')


def _finalize(trader, qty=FRACTIONAL_QTY, direction=Direction.LONG,
              stop=None, target=None):
    """Open a tracked position and run the real bundle-finalisation path."""
    signed = -qty if direction == Direction.SHORT else qty
    sl = SL if stop is None else stop
    tp = TP if target is None else target
    if direction == Direction.SHORT:
        # valid short geometry: SL above entry, TP below
        sl = FILL + abs(sl - FILL)
        tp = FILL - abs(tp - FILL)
    trader.pm.open_position(SYM, signed, FILL)
    # the broker holds the same position (an entry fill would have created it)
    trader.broker.positions[SYM] = {
        "symbol": SYM, "qty": signed, "avg_entry_price": FILL,
    }
    sig = _sig(SYM, direction, FILL, sl, tp, "box_theory")
    asyncio.run(trader._finalize_open_bundle(SYM, qty, FILL, sig))
    return sig


def _one_min_frames(close):
    return {"1min": _frame_df([close] * 6, freq="1min")}


def _trader(broker, close, positions=None):
    broker.positions = broker.positions or {}
    if positions is not None:
        broker.positions = {str(p["symbol"]).upper(): p for p in positions}
    return _make_scalp_trader(
        broker=broker, provider=FakeProvider(_one_min_frames(close)),
    )


# ---------------------------------------------------------------------------
# 1. The planner (pure)
# ---------------------------------------------------------------------------
class TestPlanExitStructure:
    def test_whole_share_position_with_stop_reserves_everything(self):
        plan = plan_exit_structure(21.0, stop_placed=True, tp=660.80)
        assert plan.stop_qty == 21
        assert plan.broker_tp_qty == 0.0       # a second order would 40310000
        assert plan.monitored_tp is True

    def test_fractional_position_with_stop_reserves_the_position(self):
        plan = plan_exit_structure(FRACTIONAL_QTY, stop_placed=True, tp=TP)
        assert plan.stop_qty == 42             # whole-share GTC stop
        assert plan.broker_tp_qty == 0.0       # nothing left to rest a TP on
        assert plan.monitored_tp is True
        assert "40310000" in plan.reason

    def test_no_stop_resting_tries_a_broker_tp_first(self):
        plan = plan_exit_structure(FRACTIONAL_QTY, stop_placed=False, tp=TP)
        assert plan.stop_qty == 0
        assert plan.broker_tp_qty == FRACTIONAL_QTY
        assert plan.monitored_tp is True       # fallback if it is rejected

    def test_no_tp_level_means_stop_only(self):
        plan = plan_exit_structure(10.0, stop_placed=True, tp=None)
        assert plan.monitored_tp is False
        assert plan.broker_tp_qty == 0.0

    def test_stop_capacity_floors_and_never_goes_negative(self):
        assert stop_capacity(42.92623498) == 42
        assert stop_capacity(0.9) == 0
        assert stop_capacity(0) == 0
        assert stop_capacity(-3.7) == 3
        assert stop_capacity(None) == 0

    def test_format_is_readable(self):
        text = format_exit_plan(plan_exit_structure(5.0, stop_placed=True, tp=10.0))
        assert "trader-side monitor" in text
        assert "SL=broker GTC stop x5" in text


# ---------------------------------------------------------------------------
# 2. The TP level survives the stop's reservation  (fails before the fix)
# ---------------------------------------------------------------------------
class TestTpIsNeverDropped:
    def test_tp_level_kept_and_monitored_when_broker_tp_is_rejected(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=FILL)
        _finalize(trader)

        state = trader._scalp_positions[SYM]
        # Before the fix this was None — no upside exit existed at all.
        assert state["tp"] == pytest.approx(TP)
        assert state["tp_placed"] is False
        assert state["tp_monitored"] is True
        assert state["stop_placed"] is True
        # Both legs exist at the same time: the stop RESTS at the broker ...
        assert [s[0] for s in broker.stop_requests] == [SYM]
        assert broker.stop_requests[0][1] == 42       # whole shares
        # ... and the TP is owned by the monitor (no resting TP order).
        assert not [o for o in broker.orders if o.order_type == OrderType.LIMIT]

    def test_doomed_broker_tp_is_not_attempted_when_stop_reserves_all(self):
        """No pointless 40310000 on every entry: the plan short-circuits."""
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=FILL)
        _finalize(trader)
        assert broker.tp_rejections == []
        assert not [o for o in broker.orders if o.order_type == OrderType.LIMIT]

    def test_whole_share_position_next_to_a_stop_also_keeps_its_tp(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=FILL)
        _finalize(trader, qty=21.0, direction=Direction.SHORT)
        state = trader._scalp_positions[SYM]
        assert state["tp"] is not None
        assert state["tp_monitored"] is True
        assert state["stop_placed"] is True
        assert broker.stop_requests[0][1] == 21


# ---------------------------------------------------------------------------
# 3. The monitored upside exit actually fires
# ---------------------------------------------------------------------------
class TestMonitoredTpCloses:
    def test_monitor_closes_the_full_position_at_target(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=TP + 1.0)
        _finalize(trader)
        stop_id = broker._open_orders[0].id

        asyncio.run(trader._scalp_risk_pass())

        # The stop was cancelled (a close cannot co-exist with it) ...
        assert stop_id in broker.cancelled_ids
        # ... and the close was submitted for the FULL position.
        closes = [o for o in broker.orders
                  if o.order_type == OrderType.MARKET and o.symbol.upper() == SYM]
        assert closes, "monitored TP did not submit a close order"
        assert float(closes[-1].quantity) == pytest.approx(FRACTIONAL_QTY)

    def test_monitor_stays_quiet_while_a_resting_tp_order_exists(self):
        # The resting TP only exists when the stop could not be placed (a
        # resting stop would reserve the shares): then the TP order pays and
        # the monitor must NOT close a second time.
        broker = StopFailingBroker()
        trader = _trader(broker, close=TP + 1.0)
        _finalize(trader)
        state = trader._scalp_positions[SYM]
        assert state["stop_placed"] is False
        assert state["tp_placed"] is True
        assert state["tp_monitored"] is False
        tp_order_ids = [o.id for o in broker._open_orders
                        if o.type == "limit" and o.symbol.upper() == SYM]
        assert tp_order_ids

        asyncio.run(trader._scalp_risk_pass())

        # the resting TP pays the exit — no duplicate close order
        assert not [o for o in broker.orders if o.order_type == OrderType.MARKET]
        assert [o.id for o in broker._open_orders if o.type == "limit"] == tp_order_ids

    def test_monitor_does_not_fire_below_the_target(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=TP - 1.0)
        _finalize(trader)
        asyncio.run(trader._scalp_risk_pass())
        assert not [o for o in broker.orders if o.order_type == OrderType.MARKET]


# ---------------------------------------------------------------------------
# 4. A partial exit must never leave a naked residual
# ---------------------------------------------------------------------------
class TestResidualProtection:
    def test_reprotect_residual_covers_what_is_still_held(self):
        broker = StopReservesQtyBroker()      # nothing resting initially
        trader = _trader(broker, close=TP - 1.0, positions=[
            {"symbol": SYM, "qty": 10.0, "avg_entry_price": FILL}])
        trader._scalp_positions[SYM] = {"entry": FILL, "sl": SL, "tp": TP}

        ok = asyncio.run(trader._reprotect_residual(SYM, "test partial fill"))

        assert ok is True
        assert broker.stop_requests[-1][0] == SYM
        assert broker.stop_requests[-1][1] == 10      # the residual, floored
        assert broker.stop_requests[-1][2] == pytest.approx(SL)

    def test_reprotect_is_idempotent_and_never_doubles_a_stop(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=TP - 1.0, positions=[
            {"symbol": SYM, "qty": 10.0, "avg_entry_price": FILL}])
        trader._scalp_positions[SYM] = {"entry": FILL, "sl": SL, "tp": TP}
        asyncio.run(trader._reprotect_residual(SYM, "first"))
        asyncio.run(trader._reprotect_residual(SYM, "second"))
        assert len(broker.stop_requests) == 1

    def test_reprotect_noop_when_flat(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=TP - 1.0)
        assert asyncio.run(trader._reprotect_residual(SYM, "flat")) is True
        assert broker.stop_requests == []

    def test_position_sync_reprotects_a_shrunk_position(self):
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=TP - 1.0)
        _finalize(trader)                        # tracked 42.92623498, stop x42
        # A partial fill of a close order reduced the broker position.
        broker.positions[SYM]["qty"] = 12.0
        broker._open_orders = []                 # the stop was cancelled to close

        asyncio.run(trader._scalp_sync_positions())

        assert broker.stop_requests[-1][1] == 12
        # the tracked quantity is NOT silently rewritten (accounting stays
        # with the verified-close / vanish path)
        assert float(trader.pm.get_positions()[SYM].quantity) == pytest.approx(
            FRACTIONAL_QTY)

    def test_vanish_still_wins_over_the_monitor(self):
        """EOD flatten / external close: the monitor must not fight it."""
        broker = StopReservesQtyBroker()
        trader = _trader(broker, close=TP + 1.0)
        _finalize(trader)
        broker.positions.pop(SYM)                # flattened elsewhere
        before = len(broker.orders)

        asyncio.run(trader._scalp_sync_positions())
        asyncio.run(trader._scalp_risk_pass())

        assert SYM not in trader._scalp_positions
        assert len(broker.orders) == before      # no close order for a flat book
