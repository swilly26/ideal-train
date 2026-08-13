"""Tests for the Alpaca broker integration.

All API calls are mocked — no real Alpaca credentials needed.
"""

import os
import time as _time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.broker import Order, OrderSide, OrderType
from src.execution.alpaca_broker import AlpacaBroker, is_order_alive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_trading_client():
    """Return a MagicMock that stands in for ``TradingClient``."""
    client = MagicMock()
    client.submit_order = MagicMock()
    client.cancel_order_by_id = MagicMock()
    client.get_all_positions = MagicMock()
    client.get_account = MagicMock()
    client.get_clock = MagicMock()
    return client


@pytest.fixture
def broker(mock_trading_client):
    """Return an AlpacaBroker with a patched-in mock client."""
    b = AlpacaBroker(api_key="test_key", secret_key="test_secret", paper=True)
    b._client = mock_trading_client  # Inject mock directly
    return b


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------


class TestMarketOrderPlacement:
    @pytest.mark.asyncio
    async def test_place_market_order(self, broker, mock_trading_client):
        """Market order should submit a MarketOrderRequest."""
        mock_order = MagicMock()
        mock_order.id = "order-1"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.qty = "10"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.50"
        mock_order.status = "filled"
        mock_order.created_at = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)

        mock_trading_client.submit_order.return_value = mock_order

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        result = await broker.place_order(order)

        assert result.order_id == "order-1"
        assert result.symbol == "AAPL"
        assert result.side == OrderSide.BUY
        assert result.quantity == 10
        assert result.filled_quantity == 10
        assert result.filled_avg_price == 150.50
        assert result.status == "filled"

        # Verify the request was built correctly
        call_args = mock_trading_client.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.qty == 10

    @pytest.mark.asyncio
    async def test_place_limit_order(self, broker, mock_trading_client):
        """Limit order should submit a LimitOrderRequest with limit_price."""
        mock_order = MagicMock()
        mock_order.id = "order-2"
        mock_order.symbol = "MSFT"
        mock_order.side = "sell"
        mock_order.qty = "5"
        mock_order.filled_qty = "5"
        mock_order.filled_avg_price = "400.00"
        mock_order.status = "filled"
        mock_order.created_at = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)

        mock_trading_client.submit_order.return_value = mock_order

        order = Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=400.00,
        )
        result = await broker.place_order(order)

        assert result.status == "filled"
        call_args = mock_trading_client.submit_order.call_args[0][0]
        assert call_args.limit_price == 400.00

    @pytest.mark.asyncio
    async def test_place_stop_order(self, broker, mock_trading_client):
        """Stop order should submit a StopOrderRequest with stop_price."""
        mock_order = MagicMock()
        mock_order.id = "order-3"
        mock_order.symbol = "TSLA"
        mock_order.side = "sell"
        mock_order.qty = "3"
        mock_order.filled_qty = "3"
        mock_order.filled_avg_price = "200.00"
        mock_order.status = "filled"
        mock_order.created_at = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)

        mock_trading_client.submit_order.return_value = mock_order

        order = Order(
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=3,
            order_type=OrderType.STOP,
            stop_price=195.00,
        )
        result = await broker.place_order(order)

        assert result.status == "filled"
        call_args = mock_trading_client.submit_order.call_args[0][0]
        assert call_args.stop_price == 195.00

    @pytest.mark.asyncio
    async def test_limit_order_requires_limit_price(self, broker):
        """Limit order without limit_price should raise ValueError."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
        )
        with pytest.raises(ValueError, match="limit_price"):
            await broker.place_order(order)

    @pytest.mark.asyncio
    async def test_stop_order_requires_stop_price(self, broker):
        """Stop order without stop_price should raise ValueError."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.STOP,
        )
        with pytest.raises(ValueError, match="stop_price"):
            await broker.place_order(order)

    @pytest.mark.asyncio
    async def test_order_submission_failure_returns_rejected(self, broker, mock_trading_client):
        """If Alpaca raises, place_order should return a rejected OrderResult."""
        mock_trading_client.submit_order.side_effect = RuntimeError("API down")

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        result = await broker.place_order(order)

        assert result.status == "rejected"
        assert result.filled_quantity == 0.0


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestOrderCancellation:
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, broker, mock_trading_client):
        """Cancel should return True on success."""
        mock_trading_client.cancel_order_by_id.return_value = None
        result = await broker.cancel_order("order-123")
        assert result is True
        mock_trading_client.cancel_order_by_id.assert_called_once_with("order-123")

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, broker, mock_trading_client):
        """Cancel should return False when Alpaca raises."""
        mock_trading_client.cancel_order_by_id.side_effect = Exception("not found")
        result = await broker.cancel_order("bad-id")
        assert result is False


# ---------------------------------------------------------------------------
# Account & positions
# ---------------------------------------------------------------------------


class TestGetAccount:
    @pytest.mark.asyncio
    async def test_get_account_returns_expected_keys(self, broker, mock_trading_client):
        """Account dict should have buying_power, equity, cash, portfolio_value."""
        mock_acct = MagicMock()
        mock_acct.buying_power = "50000.00"
        mock_acct.equity = "100000.00"
        mock_acct.cash = "45000.00"
        mock_acct.portfolio_value = "100000.00"
        mock_trading_client.get_account.return_value = mock_acct

        result = await broker.get_account()

        assert result["buying_power"] == 50000.00
        assert result["equity"] == 100000.00
        assert result["cash"] == 45000.00
        assert result["portfolio_value"] == 100000.00

    @pytest.mark.asyncio
    async def test_get_account_failure_returns_zeros(self, broker, mock_trading_client):
        """Failed account fetch should return zeros."""
        mock_trading_client.get_account.side_effect = Exception("network error")
        result = await broker.get_account()
        assert result["buying_power"] == 0.0
        assert result["equity"] == 0.0


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_get_positions_returns_list_of_dicts(self, broker, mock_trading_client):
        """Positions should be normalised dicts with expected keys."""
        pos1 = MagicMock()
        pos1.symbol = "AAPL"
        pos1.qty = "10"
        pos1.market_value = "1500.00"
        pos1.unrealized_pl = "50.00"
        pos1.avg_entry_price = "145.00"
        pos1.current_price = "150.00"

        pos2 = MagicMock()
        pos2.symbol = "MSFT"
        pos2.qty = "5"
        pos2.market_value = "2000.00"
        pos2.unrealized_pl = "-20.00"
        pos2.avg_entry_price = "404.00"
        pos2.current_price = "400.00"

        mock_trading_client.get_all_positions.return_value = [pos1, pos2]

        result = await broker.get_positions()

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["qty"] == 10.0
        assert result[0]["unrealized_pl"] == 50.0
        assert result[1]["symbol"] == "MSFT"
        assert result[1]["unrealized_pl"] == -20.0

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, broker, mock_trading_client):
        """No open positions should return empty list."""
        mock_trading_client.get_all_positions.return_value = []
        result = await broker.get_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_positions_failure_returns_empty(self, broker, mock_trading_client):
        """Failed position fetch should return empty list."""
        mock_trading_client.get_all_positions.side_effect = Exception("error")
        result = await broker.get_positions()
        assert result == []


# ---------------------------------------------------------------------------
# Market clock
# ---------------------------------------------------------------------------


class TestMarketClock:
    @pytest.mark.asyncio
    async def test_is_market_open_returns_true(self, broker, mock_trading_client):
        """When clock says open, return True."""
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_trading_client.get_clock.return_value = mock_clock

        assert await broker.is_market_open() is True

    @pytest.mark.asyncio
    async def test_is_market_open_returns_false(self, broker, mock_trading_client):
        """When clock says closed, return False."""
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_trading_client.get_clock.return_value = mock_clock

        assert await broker.is_market_open() is False

    @pytest.mark.asyncio
    async def test_is_market_open_returns_false_on_error(self, broker, mock_trading_client):
        """When clock API fails, default to False (safe)."""
        mock_trading_client.get_clock.side_effect = Exception("timeout")
        assert await broker.is_market_open() is False


# ---------------------------------------------------------------------------
# API timeouts — the market-open hang regression
# ---------------------------------------------------------------------------


class TestApiTimeouts:
    """A stalled Alpaca connection must never hang the trader again.

    Regression for the bug where ``is_market_open()`` called
    ``asyncio.to_thread(get_clock)`` with no timeout — a dead socket read
    blocked the worker thread forever and both traders missed a session.
    """

    @staticmethod
    def _patch_timeout(monkeypatch):
        """Shrink the API timeout so tests don't wait the real 10s."""
        import src.execution.alpaca_broker as broker_mod
        monkeypatch.setattr(broker_mod, "_API_TIMEOUT_SECONDS", 0.1)

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_is_market_open_times_out_and_returns_none(self, broker, mock_trading_client, monkeypatch):
        """A hung clock call must return None (indeterminate, bounded) - the run
        loop retries on None instead of treating it as a confirmed close."""
        self._patch_timeout(monkeypatch)
        mock_trading_client.get_clock.side_effect = lambda: _time.sleep(1)
        start = _time.monotonic()
        result = await broker.is_market_open()
        elapsed = _time.monotonic() - start
        assert result is None
        assert elapsed < 1.0, f"is_market_open hung for {elapsed:.1f}s"
    @pytest.mark.asyncio
    async def test_get_account_times_out_and_returns_sentinel(self, broker, mock_trading_client, monkeypatch):
        """A hung account fetch must return the sentinel (equity None), never 0.0.
        Regression: the old zeros fallback made shutdown() print a bogus -100%."""
        self._patch_timeout(monkeypatch)
        mock_trading_client.get_account.side_effect = lambda: _time.sleep(1)
        start = _time.monotonic()
        result = await broker.get_account()
        elapsed = _time.monotonic() - start
        assert result["equity"] is None
        assert result["buying_power"] is None
        assert result["available"] is False
        assert elapsed < 1.0, f"get_account hung for {elapsed:.1f}s"
    async def test_get_positions_times_out_and_returns_empty(self, broker, mock_trading_client, monkeypatch):
        """A hung positions fetch must return [] (fail closed)."""
        self._patch_timeout(monkeypatch)
        mock_trading_client.get_all_positions.side_effect = lambda: _time.sleep(1)

        start = _time.monotonic()
        result = await broker.get_positions()
        elapsed = _time.monotonic() - start

        assert result == []
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_get_open_orders_times_out_and_propagates(self, broker, mock_trading_client, monkeypatch):
        """A hung open-orders fetch must raise (bounded), not block forever."""
        self._patch_timeout(monkeypatch)
        mock_trading_client.get_orders.side_effect = lambda flt: _time.sleep(1)

        start = _time.monotonic()
        with pytest.raises(TimeoutError):
            await broker.get_open_orders()
        elapsed = _time.monotonic() - start

        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_place_order_times_out_and_returns_rejected(self, broker, mock_trading_client, monkeypatch):
        """A hung order submission must return rejected, not hang the loop."""
        self._patch_timeout(monkeypatch)
        mock_trading_client.submit_order.side_effect = lambda req: _time.sleep(1)

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        start = _time.monotonic()
        result = await broker.place_order(order)
        elapsed = _time.monotonic() - start

        assert result.status == "rejected"
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_cancel_order_times_out_and_returns_false(self, broker, mock_trading_client, monkeypatch):
        """A hung cancel must return False, not hang the cleanup path."""
        self._patch_timeout(monkeypatch)
        mock_trading_client.cancel_order_by_id.side_effect = lambda oid: _time.sleep(1)

        start = _time.monotonic()
        result = await broker.cancel_order("order-1")
        elapsed = _time.monotonic() - start

        assert result is False
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestAlpacaBrokerConfig:
    def test_paper_mode_default_true(self):
        """Paper trading should be the default mode."""
        broker = AlpacaBroker(api_key="k", secret_key="s")
        assert broker._paper is True

    def test_uses_env_vars_when_no_args(self):
        """Credentials should come from env vars when not passed explicitly."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "env_key",
            "ALPACA_SECRET_KEY": "env_secret",
            "ALPACA_PAPER": "false",
        }):
            broker = AlpacaBroker()
            assert broker._api_key == "env_key"
            assert broker._secret_key == "env_secret"
            assert broker._paper is False

    def test_explicit_args_override_env(self):
        """Explicit constructor args should take priority over env vars."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "env_key",
            "ALPACA_SECRET_KEY": "env_secret",
            "ALPACA_PAPER": "false",
        }):
            broker = AlpacaBroker(api_key="explicit", secret_key="explicit", paper=True)
            assert broker._api_key == "explicit"
            assert broker._secret_key == "explicit"
            assert broker._paper is True


# ---------------------------------------------------------------------------
# is_order_alive
# ---------------------------------------------------------------------------


class TestIsOrderAlive:
    """Tests for the ``is_order_alive`` helper that classifies order statuses."""

    @pytest.mark.parametrize("status", [
        "pending_new",
        "accepted",
        "new",
        "partially_filled",
        "filled",
        "pending_cancel",
        "pending_replace",
        "calculated",
        "held",
        "done_for_day",
    ])
    def test_alive_statuses_return_true(self, status):
        """All non-terminal statuses should be considered alive."""
        assert is_order_alive(status) is True, f"{status} should be alive"

    @pytest.mark.parametrize("status", [
        "rejected",
        "canceled",
        "expired",
        "suspended",
        "stopped",
    ])
    def test_terminal_statuses_return_false(self, status):
        """Terminal failure statuses should be considered dead."""
        assert is_order_alive(status) is False, f"{status} should NOT be alive"

    def test_case_insensitive(self):
        """Status matching should be case-insensitive."""
        assert is_order_alive("PENDING_NEW") is True
        assert is_order_alive("Rejected") is False
        assert is_order_alive("FILLED") is True

    def test_orderstatus_prefix_stripped(self):
        """Statuses with the 'orderstatus.' prefix should also work."""
        assert is_order_alive("orderstatus.pending_new") is True
        assert is_order_alive("orderstatus.rejected") is False


# ---------------------------------------------------------------------------
# cancel_all_orders
# ---------------------------------------------------------------------------


class TestCancelAllOrders:
    @pytest.mark.asyncio
    async def test_cancel_all_empty(self, broker, mock_trading_client):
        """When no open orders, return 0."""
        mock_trading_client.get_orders.return_value = []
        result = await broker.cancel_all_orders()
        assert result == 0

    @pytest.mark.asyncio
    async def test_cancel_all_success(self, broker, mock_trading_client):
        """All open orders should be cancelled.

        Cancellation is asynchronous in Alpaca: after a successful cancel
        request the order leaves the open-orders snapshot.  The mock
        simulates that by returning an empty open-orders list on the
        follow-up polls inside ``cancel_order_and_wait``.
        """
        o1 = MagicMock()
        o1.id = "order-1"
        o2 = MagicMock()
        o2.id = "order-2"
        # call 1: initial listing; calls 2-3: post-cancel confirmation polls
        mock_trading_client.get_orders.side_effect = [[o1, o2], [], []]
        mock_trading_client.cancel_order_by_id.return_value = None

        result = await broker.cancel_all_orders()
        assert result == 2
        assert mock_trading_client.cancel_order_by_id.call_count == 2

    @pytest.mark.asyncio
    async def test_cancel_all_partial_failure(self, broker, mock_trading_client):
        """Returns count of successfully cancelled ones."""
        o1 = MagicMock()
        o1.id = "order-1"
        o2 = MagicMock()
        o2.id = "order-2"
        # call 1: initial listing; call 2: post-cancel confirmation for o1
        mock_trading_client.get_orders.side_effect = [[o1, o2], []]
        mock_trading_client.cancel_order_by_id.side_effect = [
            None,  # first succeeds
            Exception("fail"),  # second fails
        ]

        result = await broker.cancel_all_orders()
        assert result == 1

    @pytest.mark.asyncio
    async def test_cancel_all_fetch_failure(self, broker, mock_trading_client):
        """If get_orders fails, return 0."""
        mock_trading_client.get_orders.side_effect = Exception("fetch failed")
        result = await broker.cancel_all_orders()
        assert result == 0


# ---------------------------------------------------------------------------
# Idempotency keys (client_order_id)
# ---------------------------------------------------------------------------


class TestIdempotencyKeys:
    @pytest.mark.asyncio
    async def test_generates_client_order_id_when_none_provided(self, broker, mock_trading_client):
        """When Order.client_id is None, place_order generates one."""
        mock_order = MagicMock()
        mock_order.id = "alpaca-id-1"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.qty = "10"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.00"
        mock_order.status = "filled"
        mock_order.created_at = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        mock_trading_client.submit_order.return_value = mock_order

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        # client_id is None
        assert order.client_id is None

        await broker.place_order(order)

        # Verify the submitted request had a client_order_id
        call_args = mock_trading_client.submit_order.call_args[0][0]
        cid = call_args.client_order_id
        assert cid is not None
        assert cid.startswith("algoflow_AAPL_BUY_")

    @pytest.mark.asyncio
    async def test_respects_explicit_client_id(self, broker, mock_trading_client):
        """When Order.client_id is set, it should be passed through."""
        mock_order = MagicMock()
        mock_order.id = "alpaca-id-1"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.qty = "10"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.00"
        mock_order.status = "filled"
        mock_order.created_at = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        mock_trading_client.submit_order.return_value = mock_order

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, client_id="my-custom-id")

        await broker.place_order(order)

        call_args = mock_trading_client.submit_order.call_args[0][0]
        assert call_args.client_order_id == "my-custom-id"

    @pytest.mark.asyncio
    async def test_duplicate_client_order_id_treated_as_success(self, broker, mock_trading_client):
        """Duplicate client_order_id errors should return 'accepted' status."""
        mock_trading_client.submit_order.side_effect = Exception(
            "duplicate client_order_id detected"
        )

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        result = await broker.place_order(order)

        assert result.status == "accepted"
        assert result.symbol == "AAPL"
        assert result.order_id != ""  # should be the client_order_id

class TestStartupHealth:
    @pytest.mark.asyncio
    async def test_health_check_fails_without_credentials(self, mock_trading_client):
        from src.execution.alpaca_broker import BrokerAuthenticationError
        broker = AlpacaBroker(api_key="", secret_key="", paper=True)
        with pytest.raises(BrokerAuthenticationError):
            await broker.startup_health_check()

    @pytest.mark.asyncio
    async def test_health_check_fails_when_api_unauthenticated(self, broker, mock_trading_client):
        from src.execution.alpaca_broker import BrokerAuthenticationError
        mock_trading_client.get_orders.side_effect = RuntimeError("You must supply a method of authentication")
        with pytest.raises(BrokerAuthenticationError):
            await broker.startup_health_check()

    @pytest.mark.asyncio
    async def test_cancel_prefix_does_not_cancel_other_trader(self, broker, mock_trading_client):
        own = MagicMock(id="own", client_order_id="algoflow_TURBO_A")
        other = MagicMock(id="other", client_order_id="algoflow_MAIN_A")
        mock_trading_client.get_orders.side_effect = [[own, other], [other]]
        assert await broker.cancel_orders_by_client_id_prefix("algoflow_TURBO_") == 1
        mock_trading_client.cancel_order_by_id.assert_called_once_with("own")


# ---------------------------------------------------------------------------
# wait_for_order_fill — the entry-settlement barrier behind the stop-loss fix
# ---------------------------------------------------------------------------


class TestWaitForOrderFill:
    """Regression for the wash-trade bug: a SELL stop submitted while the
    entry BUY is still open is rejected by Alpaca ('opposite side market/stop
    order exists').  The entry path must wait for the fill before attaching
    the stop."""

    @pytest.mark.asyncio
    async def test_returns_order_when_filled(self, broker, mock_trading_client):
        mock_order = MagicMock()
        mock_order.status = "orderstatus.filled"
        mock_order.qty = "100"
        mock_order.filled_avg_price = "32.50"
        mock_trading_client.get_order_by_id.return_value = mock_order

        result = await broker.wait_for_order_fill("order-1", timeout=1.0)
        assert result is mock_order
        assert float(result.qty) == 100.0
        assert float(result.filled_avg_price) == 32.50

    @pytest.mark.asyncio
    async def test_returns_false_when_order_dies(self, broker, mock_trading_client):
        mock_order = MagicMock()
        mock_order.status = "rejected"
        mock_trading_client.get_order_by_id.return_value = mock_order

        result = await broker.wait_for_order_fill("order-1", timeout=1.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, broker, mock_trading_client):
        """An order stuck open (neither filled nor dead) must time out."""
        mock_order = MagicMock()
        mock_order.status = "new"
        mock_trading_client.get_order_by_id.return_value = mock_order

        start = _time.monotonic()
        result = await broker.wait_for_order_fill("order-1", timeout=0.2, poll_interval=0.05)
        elapsed = _time.monotonic() - start

        assert result is None
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_polls_until_filled(self, broker, mock_trading_client):
        """A buy that starts 'new' and later fills must be detected."""
        o_new = MagicMock()
        o_new.status = "new"
        o_filled = MagicMock()
        o_filled.status = "filled"
        o_filled.qty = "100"
        o_filled.filled_avg_price = "32.19"
        mock_trading_client.get_order_by_id.side_effect = [o_new, o_filled]

        result = await broker.wait_for_order_fill("order-1", timeout=2.0, poll_interval=0.05)

        assert result is o_filled
        assert mock_trading_client.get_order_by_id.call_count == 2


# ---------------------------------------------------------------------------
# place_stop_order — GTC protective stops used by the turbo trader
# ---------------------------------------------------------------------------


class TestPlaceStopOrder:
    @pytest.mark.asyncio
    async def test_places_gtc_stop_with_whole_shares(self, broker, mock_trading_client):
        mock_order = MagicMock()
        mock_order.id = "stop-1"
        mock_order.symbol = "FNGU"
        mock_order.side = "sell"
        mock_order.qty = "1838"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_order.status = "new"
        mock_order.created_at = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        mock_trading_client.submit_order.return_value = mock_order

        result = await broker.place_stop_order("FNGU", 1838.12, 30.25, client_id="cid-1")

        req = mock_trading_client.submit_order.call_args[0][0]
        assert req.symbol == "FNGU"
        assert int(req.qty) == 1838  # floored for GTC
        assert str(req.side.value) == "sell"
        assert str(req.time_in_force.value) == "gtc"
        assert req.stop_price == 30.25
        assert req.client_order_id == "cid-1"
        assert result.status == "new"

    @pytest.mark.asyncio
    async def test_wash_trade_rejection_raises_for_retry(self, broker, mock_trading_client):
        """A wash-trade rejection must propagate so the caller can retry."""
        mock_trading_client.submit_order.side_effect = Exception(
            '{"code":40310000,"message":"potential wash trade detected. use complex orders",'
            '"reject_reason":"opposite side market/stop order exists"}'
        )
        with pytest.raises(Exception, match="wash trade"):
            await broker.place_stop_order("FNGU", 100, 30.25)

    @pytest.mark.asyncio
    async def test_duplicate_client_id_returns_accepted(self, broker, mock_trading_client):
        """A duplicate client_order_id means the stop is already live."""
        mock_trading_client.submit_order.side_effect = Exception(
            "duplicate client_order_id detected"
        )
        result = await broker.place_stop_order("FNGU", 100, 30.25, client_id="dup-id")
        assert result.status == "accepted"


# ---------------------------------------------------------------------------
# get_open_orders symbol filter
# ---------------------------------------------------------------------------


class TestGetOpenOrdersFilter:
    @pytest.mark.asyncio
    async def test_passes_symbol_filter(self, broker, mock_trading_client):
        mock_trading_client.get_orders.return_value = []
        await broker.get_open_orders(symbol="FNGU")
        flt = mock_trading_client.get_orders.call_args[0][0]
        assert flt.symbols == ["FNGU"]

    @pytest.mark.asyncio
    async def test_no_symbol_filter_without_arg(self, broker, mock_trading_client):
        mock_trading_client.get_orders.return_value = []
        await broker.get_open_orders()
        flt = mock_trading_client.get_orders.call_args[0][0]
        assert getattr(flt, "symbols", None) is None
