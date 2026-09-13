"""Regression: live_trader must SURVIVE an unconfirmed stale-order cancel.

2026-09-13 incident: a leftover ``algoflow_MAIN_`` order was pinned
``pending cancel`` at the broker (cancel was in flight when the previous
host died on Sep 10).  Every cancel attempt returns 42210000, the order
stays in GET /orders forever, and live_trader's ``run()`` treated
"cancellation not confirmed" as FATAL — logging
"Stale order cancellation was not confirmed; deferring position cleanup"
and RETURNING, which killed the process.  The watchdog restarted it every
minute and it died every time (restart churn, stack unhealthy all weekend).

The fix mirrors the turbo trader's resilience:
1. ``cancel_order`` treats a 42210000 "order pending cancel" broker
   response as non-fatal (the cancel is already in flight — leave it alone).
2. ``run()`` logs a WARNING, skips ONLY the post-startup position cleanup,
   and CONTINUES the boot: sync positions from broker -> log inherited ->
   (cleanup skipped) -> ensure protective stops -> wait for market open.
"""
import pytest
from unittest.mock import MagicMock
import live_trader
from src.execution.alpaca_broker import AlpacaBroker
from src.execution.position_manager import PositionManager


class StuckOpenOrder:
    """Minimal stand-in for an Alpaca order object stuck in pending_cancel."""

    id = "440218e5-35a5-4392-9df0-6a48990d31f8"
    client_order_id = "algoflow_MAIN_440218e5-35a5-4392-9df0-6a48990d31f8"


class StuckCancelBroker:
    """Broker whose stale MAIN order can never be cancelled (42210000)."""

    def __init__(self):
        self.cancel_calls = 0
        self.last_prefix = None

    async def startup_health_check(self):
        return None

    async def cancel_orders_by_client_id_prefix(self, prefix):
        self.cancel_calls += 1
        self.last_prefix = prefix
        return 0  # every cancel attempt fails -> nothing confirmed cancelled

    async def get_open_orders(self):
        # The stuck order remains visible in the open-order book forever.
        return [StuckOpenOrder()]


class _ReachedMarketWait(Exception):
    """Sentinel: run() made it past startup into the wait-for-market loop."""


@pytest.mark.asyncio
async def test_run_continues_when_stale_cancel_unconfirmed():
    """run() must NOT return when stale cancellation is unconfirmed.

    It must reach position sync, skip ONLY the post-startup cleanup, still
    ensure protective stops, and enter the market-wait loop.  Pre-fix this
    test fails because run() returns at the stale-order check (the
    _ReachedMarketWait sentinel never fires, sync/stops never run).
    """
    trader = object.__new__(live_trader.LiveTrader)
    trader.broker = StuckCancelBroker()
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)

    seen = {}

    async def fake_sync():
        seen["sync"] = True

    async def fake_cleanup():
        seen["cleanup"] = True

    async def fake_stops():
        seen["stops"] = True

    async def fake_wait():
        raise _ReachedMarketWait("run() reached the wait-for-market loop")

    trader._sync_positions_from_broker = fake_sync
    trader._post_startup_cleanup = fake_cleanup
    trader._ensure_protective_stops = fake_stops
    trader.wait_for_market_open = fake_wait

    with pytest.raises(_ReachedMarketWait):
        await trader.run()

    assert trader.broker.cancel_calls == 1
    assert trader.broker.last_prefix == "algoflow_MAIN_"
    assert seen.get("sync") is True, \
        "position sync must still run when stale cancellation is unconfirmed"
    assert seen.get("stops") is True, \
        "protective stops must still be ensured when stale cancellation is unconfirmed"
    assert "cleanup" not in seen, \
        "post-startup cleanup must be skipped (deferred) when stale cancellation is unconfirmed"


# ── Broker-level: 42210000 "order pending cancel" is non-fatal ─────────


@pytest.mark.asyncio
async def test_cancel_order_treats_pending_cancel_as_handled():
    broker = AlpacaBroker(api_key="k", secret_key="s")
    client = MagicMock()
    client.cancel_order_by_id.side_effect = Exception(
        '{"code":42210000,"message":"order pending cancel"}'
    )
    broker._client = client
    assert await broker.cancel_order("440218e5-35a5-4392-9df0-6a48990d31f8") is True


@pytest.mark.asyncio
async def test_cancel_order_still_fails_on_unrelated_errors():
    broker = AlpacaBroker(api_key="k", secret_key="s")
    client = MagicMock()
    client.cancel_order_by_id.side_effect = Exception("connection reset")
    broker._client = client
    assert await broker.cancel_order("some-other-order") is False