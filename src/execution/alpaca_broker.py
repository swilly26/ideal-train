"""Alpaca broker integration via the ``alpaca-py`` SDK.

Implements the ``Broker`` ABC using the Alpaca Trading API.  Paper trading
is the default — set ``ALPACA_PAPER=false`` for live trading.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from src.execution.broker import Broker, Order, OrderResult, OrderSide, OrderType

logger = logging.getLogger(__name__)


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

        Supports MARKET, LIMIT, and STOP order types.
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

        common = {
            "symbol": order.symbol.upper(),
            "qty": order.quantity,
            "side": side,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": order.client_id,
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
            "Placing %s %s %s x %s",
            order.order_type.value,
            order.side.value,
            order.symbol,
            order.quantity,
        )

        try:
            alpaca_order = await asyncio.to_thread(
                self._trading_client.submit_order, req
            )
        except Exception as exc:
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
            status=str(alpaca_order.status).lower(),
            created_at=created,
        )
