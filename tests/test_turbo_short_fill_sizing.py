"""Tests for the turbo short-fill + buying-power sizing fixes.

Covers:
1. ``_size_entry_qty`` — notional capped at BUYING_POWER_USAGE_PCT of
   available buying power; unknown BP (None) → skip (qty 0, capped True).
2. ``_handle_buy`` — order notional never exceeds available BP (the 8/28
   TZA $49k-cost-vs-$12k-BP rejection); unknown BP → no order attempted.
3. ``_handle_short_sell`` — whole-share flooring (fractional short qty →
   floored int; < 1 share → skip); capability gate (broker reports not
   shortable → no order; shortable → order attempted; broker without
   ``is_shortable`` → legacy behavior, order attempted); BP cap applies.
4. Rejection backoff — N short rejections disable that symbol's shorts
   for the session; buy-side BP rejections disable longs past the limit.
5. ``_rejection_kind`` classification buckets.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import turbo_trader
from src.execution.broker import OrderResult, OrderSide
from src.execution.position_manager import PositionManager
from tests.test_turbo_regime_short import FakeBroker, _make_trader


def _accepted(symbol="TZA", side=OrderSide.BUY, qty=10.0, err=None):
    return OrderResult(
        order_id="order-1",
        symbol=symbol,
        side=side,
        quantity=qty,
        filled_quantity=0.0,
        filled_avg_price=None,
        status="accepted",
        created_at=datetime.now(timezone.utc),
        error_message=err,
    )


def _rejected(symbol="TZA", side=OrderSide.BUY, qty=10.0, err="insufficient buying power"):
    return OrderResult(
        order_id="",
        symbol=symbol,
        side=side,
        quantity=qty,
        filled_quantity=0.0,
        filled_avg_price=None,
        status="rejected",
        created_at=datetime.now(timezone.utc),
        error_message=err,
    )


class CapBroker(FakeBroker):
    """FakeBroker with controllable account + shortability + order results."""

    def __init__(self, equity=100_000.0, buying_power=20_000.0,
                 shortable=True, result=None, shorting_enabled=None):
        super().__init__()
        self._equity = equity
        self._bp = buying_power
        self._shortable = shortable
        self._result = result
        self._shorting_enabled = shorting_enabled
        self.shortable_calls = []

    async def get_account(self):
        acct = {"equity": self._equity, "buying_power": self._bp,
                "cash": self._equity, "portfolio_value": self._equity}
        if self._shorting_enabled is not None:
            acct["shorting_enabled"] = self._shorting_enabled
        return acct

    async def is_shortable(self, symbol):
        self.shortable_calls.append(symbol)
        return self._shortable

    async def place_order(self, order):
        self.orders.append(order)
        if self._result is not None:
            return self._result
        return _accepted(order.symbol, order.side, order.quantity)


class TestRejectionKind:
    def test_fractional_short(self):
        assert turbo_trader._rejection_kind(
            '{"code":42210000,"message":"fractional orders cannot be sold short"}'
        ) == "fractional_short"

    def test_short_not_allowed(self):
        assert turbo_trader._rejection_kind("asset XYZ cannot be sold short") == "short_not_allowed"
        assert turbo_trader._rejection_kind("shorting not allowed for this account") == "short_not_allowed"

    def test_buying_power(self):
        assert turbo_trader._rejection_kind(
            '{"code":40310000,"message":"insufficient buying power"}'
        ) == "buying_power"

    def test_other(self):
        assert turbo_trader._rejection_kind("weird broker hiccup") == "other"
        assert turbo_trader._rejection_kind(None) == "other"


class TestSizeEntryQty:
    def test_no_cap_when_bp_plentiful(self):
        qty, capped = turbo_trader._size_entry_qty(
            equity=100_000.0, buying_power=400_000.0, price=40.0,
            size_pct=0.40, bp_usage_pct=0.95,
        )
        assert capped is False
        assert qty == pytest.approx(40_000.0 / 40.0)  # 1000 shares

    def test_capped_by_buying_power(self):
        # 8/28 shape: equity implies $49k, BP only $12k → capped to 0.95*12k.
        qty, capped = turbo_trader._size_entry_qty(
            equity=123_000.0, buying_power=12_039.28, price=39.0,
            size_pct=0.40, bp_usage_pct=0.95,
        )
        assert capped is True
        assert qty * 39.0 == pytest.approx(12_039.28 * 0.95)

    def test_unknown_bp_skips(self):
        qty, capped = turbo_trader._size_entry_qty(
            equity=100_000.0, buying_power=None, price=40.0, size_pct=0.40,
        )
        assert (qty, capped) == (0.0, True)

    def test_whole_shares_floor(self):
        qty, _ = turbo_trader._size_entry_qty(
            equity=100_000.0, buying_power=400_000.0, price=117.0,
            size_pct=0.50, whole_shares=True,
        )
        assert qty == int(qty)  # floored — Alpaca rejects fractional shorts
        assert qty >= 1

    def test_zero_or_negative_inputs(self):
        assert turbo_trader._size_entry_qty(
            equity=0.0, buying_power=10_000.0, price=40.0, size_pct=0.5) == (0.0, False)
        assert turbo_trader._size_entry_qty(
            equity=100_000.0, buying_power=10_000.0, price=0.0, size_pct=0.5) == (0.0, False)


class TestBuyBuyingPowerCap:
    @pytest.mark.asyncio
    async def test_buy_capped_to_available_bp(self):
        broker = CapBroker(equity=123_000.0, buying_power=12_039.28)
        trader = _make_trader(broker)
        await trader._handle_buy("TZA", 39.0, 1.0, "mean_reversion")
        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.BUY
        assert order.quantity * 39.0 <= 12_039.28 * 0.95 + 1e-6
        # Desired uncapped would have been 0.40*123000/39 ≈ 1261 shares.
        assert order.quantity < 1261.0

    @pytest.mark.asyncio
    async def test_buy_skipped_when_bp_unknown(self):
        broker = CapBroker(equity=100_000.0, buying_power=None)
        trader = _make_trader(broker)
        await trader._handle_buy("TZA", 39.0, 1.0, "mean_reversion")
        assert broker.orders == []

    @pytest.mark.asyncio
    async def test_buy_bp_rejections_disable_longs(self, monkeypatch):
        monkeypatch.setattr(turbo_trader, "BUY_BP_REJECTIONS_DISABLE_AFTER", 2)
        broker = CapBroker(
            result=_rejected(err='{"code":40310000,"message":"insufficient buying power"}'))
        trader = _make_trader(broker)
        await trader._handle_buy("TZA", 39.0, 1.0, "mean_reversion")
        await trader._handle_buy("TZA", 39.0, 1.0, "mean_reversion")
        assert trader._buys_disabled_bp is True
        n = len(broker.orders)
        await trader._handle_buy("TZA", 39.0, 1.0, "mean_reversion")
        assert len(broker.orders) == n  # no further attempts


class TestShortGating:
    @pytest.mark.asyncio
    async def test_short_blocked_when_not_shortable(self):
        broker = CapBroker(shortable=False)
        trader = _make_trader(broker)
        await trader._handle_short_sell("SOXL", 117.0, 1.0, "mean_reversion")
        assert broker.orders == []
        assert trader._shortable_cache.get("SOXL") is False

    @pytest.mark.asyncio
    async def test_short_attempted_when_shortable(self):
        broker = CapBroker(shortable=True)
        trader = _make_trader(broker)
        await trader._handle_short_sell("TQQQ", 60.0, 1.0, "momentum")
        assert len(broker.orders) == 1
        order = broker.orders[0]
        assert order.side == OrderSide.SELL
        assert order.quantity == int(order.quantity)  # whole shares
        assert order.quantity >= 1

    @pytest.mark.asyncio
    async def test_short_attempted_when_broker_has_no_shortable_method(self):
        broker = CapBroker(shortable=True)
        delattr(type(broker), "is_shortable") if hasattr(type(broker), "is_shortable") else None
        # Simulate legacy broker: hide the method on the instance.
        broker2 = FakeBroker()
        trader = _make_trader(broker2)
        assert getattr(broker2, "is_shortable", None) is None
        await trader._handle_short_sell("TQQQ", 60.0, 1.0, "momentum")
        assert len(broker2.orders) == 1

    @pytest.mark.asyncio
    async def test_short_blocked_when_account_shorting_disabled(self):
        broker = CapBroker(shortable=True, shorting_enabled=False)
        trader = _make_trader(broker)
        await trader._handle_short_sell("TQQQ", 60.0, 1.0, "momentum")
        assert broker.orders == []
        assert broker.shortable_calls == []  # account gate fires first

    @pytest.mark.asyncio
    async def test_short_rejections_disable_symbol(self, monkeypatch):
        monkeypatch.setattr(turbo_trader, "SHORT_REJECTIONS_DISABLE_AFTER", 3)
        broker = CapBroker(
            shortable=True,
            result=_rejected(side=OrderSide.SELL,
                             err='{"code":42210000,"message":"fractional orders cannot be sold short"}'))
        trader = _make_trader(broker)
        for _ in range(3):
            await trader._handle_short_sell("SOXL", 117.0, 1.0, "mean_reversion")
        assert "SOXL" in trader._shorts_disabled
        n = len(broker.orders)
        await trader._handle_short_sell("SOXL", 117.0, 1.0, "mean_reversion")
        assert len(broker.orders) == n  # no tight-loop retry

    @pytest.mark.asyncio
    async def test_short_not_allowed_verdict_sticks_immediately(self):
        broker = CapBroker(
            shortable=True,
            result=_rejected(side=OrderSide.SELL,
                             err="asset SOXL cannot be sold short"))
        trader = _make_trader(broker)
        await trader._handle_short_sell("SOXL", 117.0, 1.0, "mean_reversion")
        assert trader._shortable_cache.get("SOXL") is False
        n = len(broker.orders)
        await trader._handle_short_sell("SOXL", 117.0, 1.0, "mean_reversion")
        # Cached False gates before any new order attempt.
        assert len(broker.orders) == n

    @pytest.mark.asyncio
    async def test_short_sized_down_to_bp(self):
        broker = CapBroker(equity=123_000.0, buying_power=12_039.28, shortable=True)
        trader = _make_trader(broker)
        await trader._handle_short_sell("TQQQ", 60.0, 1.0, "momentum")
        assert len(broker.orders) == 1
        assert broker.orders[0].quantity * 60.0 <= 12_039.28 * 0.95 + 1e-6
