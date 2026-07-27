"""Alpaca broker integration via the ``alpaca-py`` SDK.

Implements the ``Broker`` ABC using the Alpaca Trading API.  Paper trading
is the default — set ``ALPACA_PAPER=false`` for live trading.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime

from src.execution.broker import Broker, Order, OrderResult, OrderSide, OrderType

logger = logging.getLogger(__name__)

# ── Terminal (failed / dead) order statuses ───────────────────────────
_TERMINAL_FAILURE_STATUSES = frozenset({
    "rejected",
    "canceled",
    "expired",
    "suspended",
    "stopped",
})


def is_order_alive(status: str) -> bool:
    """Return ``True`` if *status* means the order is still alive / accepted.

    Terminal failure statuses (``rejected``, ``canceled``, ``expired``,
    ``suspended``, ``stopped``) return ``False``.  Everything else —
    including ``pending_new``, ``accepted``, ``new``, ``partially_filled``,
    ``filled``, ``pending_cancel``, ``pending_replace``, ``calculated``,
    ``held``, ``done_for_day`` — is considered alive.
    """
    clean = status.lower().removeprefix("orderstatus.")
    return clean not in _TERMINAL_FAILURE_STATUSES


class AlpacaBroker(Broker):
    """Broker implementation backed by the Alpaca Trading API.

    Reads credentials from environment variables:

    * ``ALPACA_API_KEY`` — API key ID.
    * ``ALPACA_SECRET_KEY`` — API secret key.
    * ``ALPACA_PAPER`` — ``"true"`` (default) for paper trading, ``"false"`` for live.

    Parameters
    ----------
    api_key : str | None
        Override the ``ALPACA_API_KEY`` env var.
    secret_key : str | None
        Override the ``ALPACA_SECRET_KEY`` env var.
    paper : bool | None
        Override paper/live mode.  Defaults to ``True``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        self._paper = paper

        if not self._api_key or not self._secret_key:
            logger.warning(
                "Alpaca API key or secret not set — broker operations will fail. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        self._client: object | None = None  # Lazily initialised TradingClient

    @property
    def _trading_client(self):
        """Lazily construct and cache the Alpaca ``TradingClient``."""
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
            )
        return self._client

    # ------------------------------------------------------------------
    # Broker ABC
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> OrderResult:
        """Convert an engine ``Order`` into an Alpaca request and submit it.

        Supports MARKET, LIMIT, and STOP order types.  Generates an
        idempotency key (``client_order_id``) if none is provided so that
        Alpaca rejects duplicate submissions from restarted sessions.
        """
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import OrderType as AlpacaType
        from alpaca.trading.enums import TimeInForce

        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        # ── Idempotency key ──────────────────────────────────────────
        client_order_id = order.client_id or self._generate_client_order_id(
            order.symbol, order.side
        )

        common = {
            "symbol": order.symbol.upper(),
            "qty": order.quantity,
            "side": side,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": client_order_id,
        }

        if order.order_type == OrderType.MARKET:
            req = MarketOrderRequest(**common)
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            req = LimitOrderRequest(limit_price=order.limit_price, **common)
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise ValueError("STOP order requires stop_price")
            req = StopOrderRequest(stop_price=order.stop_price, **common)
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        logger.info(
            "Placing %s %s %s x %s (client_id=%s)",
            order.order_type.value,
            order.side.value,
            order.symbol,
            order.quantity,
            client_order_id,
        )

        try:
            alpaca_order = await asyncio.to_thread(
                self._trading_client.submit_order, req
            )
        except Exception as exc:
            # If Alpaca rejects because this client_order_id was already used,
            # the order is already live — treat it as success rather than rejected.
            if _is_duplicate_client_order_id_error(exc):
                logger.info(
                    "Order idempotency key %s already exists — order is live",
                    client_order_id,
                )
                return OrderResult(
                    order_id=client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=0.0,
                    filled_avg_price=None,
                    status="accepted",  # order is alive, we just can't see fill yet
                    created_at=datetime.now(),
                )
            logger.error("Order submission failed: %s", exc)
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0.0,
                filled_avg_price=None,
                status="rejected",
                created_at=datetime.now(),
            )

        return self._map_order_result(alpaca_order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its Alpaca order ID."""
        logger.info("Cancelling order %s", order_id)
        try:
            await asyncio.to_thread(
                self._trading_client.cancel_order_by_id, order_id
            )
            return True
        except Exception as exc:
            logger.error("Cancel order %s failed: %s", order_id, exc)
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel **all** open orders and return the count cancelled.

        Uses ``get_orders()`` with ``status=\"open\"`` to list open orders,
        then cancels each one by its Alpaca ID.  Returns the number of
        orders that were successfully cancelled.
        """
        logger.info("Fetching open orders for cancellation…")
        try:
            open_orders = await asyncio.to_thread(
                self._trading_client.get_orders,
                {"status": "open"},
            )
        except Exception as exc:
            logger.error("Failed to fetch open orders: %s", exc)
            return 0

        if not open_orders:
            logger.info("No open orders to cancel")
            return 0

        cancelled = 0
        for o in open_orders:
            oid = str(o.id)
            if await self.cancel_order(oid):
                cancelled += 1

        logger.info("Cancelled %d / %d open orders", cancelled, len(open_orders))
        return cancelled

    async def get_positions(self) -> list[dict]:
        """Return current open positions as a list of normalised dicts."""
        try:
            positions = await asyncio.to_thread(
                self._trading_client.get_all_positions
            )
        except Exception as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return []

        result = []
        for pos in positions:
            result.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "market_value": float(pos.market_value or 0),
                "unrealized_pl": float(pos.unrealized_pl or 0),
                "avg_entry_price": float(pos.avg_entry_price or 0),
                "current_price": float(pos.current_price or 0),
            })
        return result

    async def get_account(self) -> dict:
        """Return account summary as a normalised dict."""
        try:
            acct = await asyncio.to_thread(self._trading_client.get_account)
        except Exception as exc:
            logger.error("Failed to fetch account: %s", exc)
            return {
                "buying_power": 0.0,
                "equity": 0.0,
                "cash": 0.0,
                "portfolio_value": 0.0,
            }

        return {
            "buying_power": float(acct.buying_power or 0),
            "equity": float(acct.equity or 0),
            "cash": float(acct.cash or 0),
            "portfolio_value": float(acct.portfolio_value or 0),
        }

    async def close(self) -> None:
        """Release broker resources (no persistent connections to close)."""
        self._client = None

    # ------------------------------------------------------------------
    # Alpaca-specific
    # ------------------------------------------------------------------

    async def is_market_open(self) -> bool:
        """Check whether the market is currently open via the Alpaca clock API.

        Returns ``True`` if the market is open, ``False`` otherwise.  If the
        API call fails, returns ``False`` (safe default — don't trade blind).
        """
        try:
            clock = await asyncio.to_thread(self._trading_client.get_clock)
            return bool(clock.is_open)
        except Exception as exc:
            logger.error("Market clock check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_order_result(alpaca_order) -> OrderResult:
        """Map an Alpaca ``Order`` model to our ``OrderResult`` dataclass."""
        side = (
            OrderSide.BUY
            if str(alpaca_order.side).upper() == "BUY"
            else OrderSide.SELL
        )
        created = (
            alpaca_order.created_at
            if alpaca_order.created_at
            else datetime.now()
        )
        # Ensure created_at is a datetime (could be a string from some API responses)
        if isinstance(created, str):
            from dateutil.parser import parse as dt_parse
            created = dt_parse(created)

        return OrderResult(
            order_id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            side=side,
            quantity=float(alpaca_order.qty or 0),
            filled_quantity=float(alpaca_order.filled_qty or 0),
            filled_avg_price=(
                float(alpaca_order.filled_avg_price)
                if alpaca_order.filled_avg_price
                else None
            ),
            status=str(alpaca_order.status).lower().removeprefix("orderstatus."),
            created_at=created,
        )

    @staticmethod
    def _generate_client_order_id(symbol: str, side: OrderSide) -> str:
        """Generate a unique idempotency key for an order.

        Format: ``algoflow_{SYMBOL}_{side}_{timestamp_ns}``
        """
        return f"algoflow_{symbol.upper()}_{side.value}_{time.monotonic_ns()}"


def _is_duplicate_client_order_id_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* is an Alpaca "duplicate client order id" error."""
    msg = str(exc).lower()
    return "duplicate" in msg and "client_order_id" in msg
