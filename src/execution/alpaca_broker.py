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

# ── API call protection ───────────────────────────────────────────────
# Hard cap on every blocking Alpaca SDK call.  Without this, a stalled
# API connection blocks the worker thread forever on a socket read and
# hangs the whole trader (this is what made both traders miss a session).
_API_TIMEOUT_SECONDS = 10.0

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


# ── Unavailable-account sentinel ────────────────────────────────────────
# Returned by ``get_account()`` when the fetch fails or times out.  ``equity``
# is ``None`` — NEVER a fabricated number — so callers can distinguish
# "unreadable" from a real (even zero) account.  ``available`` is an explicit
# flag for the same signal.  This kills the spurious -100% P&L bug where a
# timed-out fetch returned ``equity: 0.0`` and shutdown() computed
# ``0 - start_equity`` → -100%.
ACCOUNT_UNAVAILABLE: dict = {
    "buying_power": None,
    "equity": None,
    "cash": None,
    "portfolio_value": None,
    "available": False,
}


def account_equity(account: dict) -> float | None:
    """Extract equity from a ``get_account()`` result.

    Returns ``None`` when the fetch failed (sentinel), or when the payload is
    not a dict / lacks a numeric equity — never a fabricated number.
    """
    if not isinstance(account, dict) or account.get("available") is False:
        return None
    equity = account.get("equity")
    if equity is None:
        return None
    try:
        return float(equity)
    except (TypeError, ValueError):
        return None


class BrokerAuthenticationError(RuntimeError):
    """Raised when the broker cannot prove that its session is authenticated."""


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

    async def _run_with_timeout(self, fn, *args, timeout: float | None = None, **kwargs):
        """Run a blocking Alpaca SDK call in a worker thread with a hard timeout.

        ``asyncio.wait_for`` bounds the total call so a stalled API connection
        can never hang the event loop / trader again: after ``timeout`` seconds
        we give up and let the caller handle it (retry / fail closed).  The
        orphaned worker thread keeps blocking on the socket but no longer holds
        up the trading loop.
        """
        if timeout is None:
            timeout = _API_TIMEOUT_SECONDS
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=timeout,
        )

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
            alpaca_order = await self._run_with_timeout(
                self._trading_client.submit_order, req
            )
        except asyncio.TimeoutError as exc:
            # We never got a response — the order may or may not be live.
            # Fail closed (rejected) so the caller retries; the idempotency
            # key prevents a double submission if Alpaca actually got it.
            logger.warning(
                "Order submission timed out after %.1fs (client_id=%s) — treating as rejected",
                _API_TIMEOUT_SECONDS,
                client_order_id,
            )
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0.0,
                filled_avg_price=None,
                status="rejected",
                created_at=datetime.now(),
                error_message=str(exc),
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
                error_message=str(exc),
            )

        return self._map_order_result(alpaca_order)

    async def cancel_order(self, order_id: str) -> bool:
        """Request cancellation of an open order (completion is asynchronous)."""
        logger.info("Cancelling order %s", order_id)
        try:
            await self._run_with_timeout(
                self._trading_client.cancel_order_by_id, order_id
            )
            return True
        except Exception as exc:
            logger.error("Cancel order %s failed: %s", order_id, exc)
            return False

    async def cancel_order_and_wait(
        self, order_id: str, timeout: float = 10.0, poll_interval: float = 0.25
    ) -> bool:
        """Cancel an order and wait until it is no longer open.

        Alpaca acknowledges a cancel request before the order leaves
        ``pending_cancel``.  Callers submitting a replacement sell must wait
        for this method, otherwise the old stop can still reserve shares.
        """
        if not await self.cancel_order(order_id):
            return False
        deadline = time.monotonic() + timeout
        while True:
            try:
                open_orders = await self.get_open_orders()
                if not any(str(getattr(o, "id", "")) == str(order_id) for o in open_orders):
                    return True
                # An order may remain in the open-order snapshot while its
                # cancellation is completing; query its authoritative status.
                get_order = getattr(self._trading_client, "get_order_by_id", None)
                if get_order is not None:
                    order = await self._run_with_timeout(get_order, order_id)
                    status = getattr(order, "status", None)
                    if isinstance(status, str) and not is_order_alive(status):
                        return True
            except Exception as exc:
                logger.warning("Could not confirm cancellation of %s: %s", order_id, exc)
                return False
            if time.monotonic() >= deadline:
                logger.error("Timed out waiting for cancellation of order %s", order_id)
                return False
            await asyncio.sleep(poll_interval)

    async def get_open_orders(self, symbol: str | None = None):
        """Return open orders (optionally filtered to *symbol*).

        Failures propagate to safety-critical callers (they choose to
        fail closed).  The optional *symbol* filter keeps callers such as
        protective-stop placement from downloading the whole account's
        order book and lets them race-check a single symbol's orders.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        if symbol:
            flt = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol.upper()])
        else:
            flt = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return await self._run_with_timeout(self._trading_client.get_orders, flt)

    async def wait_for_order_fill(
        self,
        order_id: str,
        timeout: float = 8.0,
        poll_interval: float = 0.25,
    ):
        """Poll an order until it fills, dies, or the timeout elapses.

        This is the settlement barrier the entry path needs before attaching
        a protective stop: submitting a SELL stop while the entry BUY is
        still open makes Alpaca reject it as a "potential wash trade"
        (``opposite side market/stop order exists``).

        Returns
        -------
        * the order object once its status is ``filled`` (or ``done_for_day``);
        * ``False`` if it reached a terminal failure status (rejected, canceled, ...);
        * ``None`` if neither happened within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                get_order = getattr(self._trading_client, "get_order_by_id", None)
                if get_order is None:
                    return None
                order = await self._run_with_timeout(get_order, order_id)
                status = str(getattr(order, "status", "")).lower().removeprefix("orderstatus.")
                if status == "filled" or status == "done_for_day":
                    return order
                if not is_order_alive(status):
                    logger.warning(
                        "Order %s reached terminal status '%s' before filling",
                        order_id, status,
                    )
                    return False
            except Exception as exc:
                logger.warning("Could not poll order %s status: %s", order_id, exc)
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(poll_interval)

    async def place_stop_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        client_id: str | None = None,
        side: str = "SELL",
    ) -> OrderResult:
        """Place a GTC stop-loss order at the broker.

        Used for protective stops that must survive process death and
        sandbox cycling.  GTC orders require whole shares, so *qty* is
        floored to an integer.  Raises on rejection (so callers can retry
        with backoff); a duplicate ``client_order_id`` is treated as success
        (the stop is already live).

        *side* is ``"SELL"`` for long positions (stop below entry) or
        ``"BUY"`` for short positions (stop above entry).
        """
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce

        sym = symbol.upper()
        whole_qty = int(qty)
        client_order_id = client_id or self._generate_client_order_id(sym, OrderSide.SELL)
        stop_side = AlpacaSide.BUY if str(side).upper() == "BUY" else AlpacaSide.SELL

        req = StopOrderRequest(
            symbol=sym,
            qty=whole_qty,
            side=stop_side,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )

        logger.info(
            "Placing GTC STOP %s %s x %d @ $%.2f (client_id=%s)",
            stop_side.value, sym, whole_qty, stop_price, client_order_id,
        )
        try:
            alpaca_order = await self._run_with_timeout(
                self._trading_client.submit_order, req
            )
            return self._map_order_result(alpaca_order)
        except asyncio.TimeoutError:
            # We never got a response — the stop may or may not be live.
            # Raise so the caller retries; the idempotency key prevents a
            # double submission if Alpaca actually got it.
            logger.warning(
                "Stop submission timed out after %.1fs (client_id=%s) — retryable",
                _API_TIMEOUT_SECONDS, client_order_id,
            )
            raise
        except Exception as exc:
            if _is_duplicate_client_order_id_error(exc):
                logger.info("🛡️  STOP %s: already placed (duplicate client_order_id)", sym)
                return OrderResult(
                    order_id=client_order_id,
                    symbol=sym,
                    side=OrderSide.SELL,
                    quantity=float(whole_qty),
                    filled_quantity=0.0,
                    filled_avg_price=None,
                    status="accepted",  # stop is alive, we just can't see it yet
                    created_at=datetime.now(),
                )
            raise

    async def cancel_orders_by_client_id_prefix(self, prefix: str) -> int:
        """Cancel only orders owned by one trader on a shared account."""
        orders = await self.get_open_orders()
        owned = [o for o in orders if str(getattr(o, "client_order_id", "")).startswith(prefix)]
        cancelled = 0
        for order in owned:
            if await self.cancel_order_and_wait(str(order.id)):
                cancelled += 1
        return cancelled

    async def startup_health_check(self) -> None:
        """Prove authentication and account visibility before trading or cleanup."""
        if not self._api_key or not self._secret_key:
            raise BrokerAuthenticationError("Alpaca credentials are missing")
        try:
            await asyncio.gather(
                self.get_open_orders(),
                self._authenticated_positions(),
                self._run_with_timeout(self._trading_client.get_clock),
                self._run_with_timeout(self._trading_client.get_account),
            )
        except Exception as exc:
            raise BrokerAuthenticationError(f"Alpaca startup authentication check failed: {exc}") from exc

    async def _authenticated_positions(self):
        return await self._run_with_timeout(self._trading_client.get_all_positions)

    async def cancel_all_orders(self) -> int:
        """Cancel **all** open orders and return the count cancelled.

        Uses ``get_orders()`` with ``status=\"open\"`` to list open orders,
        then cancels each one by its Alpaca ID.  Returns the number of
        orders that were successfully cancelled.
        """
        logger.info("Fetching open orders for cancellation…")
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            flt = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = await self._run_with_timeout(
                self._trading_client.get_orders,
                flt,
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
            if await self.cancel_order_and_wait(oid):
                cancelled += 1

        logger.info("Cancelled %d / %d open orders", cancelled, len(open_orders))
        return cancelled

    async def get_positions(self) -> list[dict]:
        """Return current open positions as a list of normalised dicts."""
        try:
            positions = await self._run_with_timeout(
                self._trading_client.get_all_positions
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Positions fetch timed out after %.1fs — returning empty", _API_TIMEOUT_SECONDS
            )
            return []
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
        """Return account summary as a normalised dict.

        On a failed/timed-out fetch returns ``ACCOUNT_UNAVAILABLE`` — a
        sentinel with ``equity: None`` and ``available: False`` — never a
        fabricated ``0.0``.  Callers must treat the sentinel as "unknown" and
        NOT compute P&L / sizing from it.
        """
        try:
            acct = await self._run_with_timeout(self._trading_client.get_account)
        except asyncio.TimeoutError:
            logger.warning(
                "Account fetch timed out after %.1fs — equity UNKNOWN (sentinel)",
                _API_TIMEOUT_SECONDS,
            )
            return dict(ACCOUNT_UNAVAILABLE)
        except Exception as exc:
            logger.error("Failed to fetch account: %s — equity UNKNOWN (sentinel)", exc)
            return dict(ACCOUNT_UNAVAILABLE)
        return {
            "buying_power": float(acct.buying_power or 0),
            "equity": float(acct.equity or 0),
            "cash": float(acct.cash or 0),
            "portfolio_value": float(acct.portfolio_value or 0),
            "available": True,
        }
    async def close(self) -> None:
        """Release broker resources (no persistent connections to close)."""
        self._client = None

    # ------------------------------------------------------------------
    # Alpaca-specific
    # ------------------------------------------------------------------

    async def is_market_open(self) -> bool | None:
        """Check whether the market is currently open via the Alpaca clock API.

        Returns:
          * ``True``  — market is open (clock confirmed).
          * ``False`` — market is confirmed closed (clock responded).
          * ``None``  — indeterminate: the clock API failed or timed out.
            Callers MUST treat ``None`` as "unknown — retry" and never as a
            confirmed close.  Only a confirmed ``False`` may trigger
            end-of-session liquidation; an indeterminate clock must not
            (this prevents a false liquidation during an API outage).
        """
        try:
            clock = await self._run_with_timeout(self._trading_client.get_clock)
            return bool(clock.is_open)
        except asyncio.TimeoutError:
            logger.warning(
                "Market clock check timed out after %.1fs — state UNKNOWN, will retry",
                _API_TIMEOUT_SECONDS,
            )
            return None
        except Exception as exc:
            logger.error("Market clock check failed: %s — state UNKNOWN, will retry", exc)
            return None
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
