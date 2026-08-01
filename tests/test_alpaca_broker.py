"""Tests for the Alpaca broker integration.

All API calls are mocked — no real Alpaca credentials needed.
"""

import os
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
        """All open orders should be cancelled."""
        o1 = MagicMock()
        o1.id = "order-1"
        o2 = MagicMock()
        o2.id = "order-2"
        mock_trading_client.get_orders.return_value = [o1, o2]
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
        mock_trading_client.get_orders.return_value = [o1, o2]
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
