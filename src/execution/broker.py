"""Abstract broker interface for trade execution.

Concrete implementations (Alpaca, Interactive Brokers, paper-trading
simulator) inherit from ``Broker`` and implement each method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    """A trade order to be sent to the broker."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    client_id: str | None = None  # idempotency key


@dataclass
class OrderResult:
    """The broker's response after placing an order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    filled_avg_price: float | None
    status: str  # "filled", "partial", "rejected", ...
    created_at: datetime
    error_message: str | None = None
    # When the broker says the order actually EXECUTED (None while it has not).
    # Kept separate from created_at: an entry order can be created seconds
    # before it is anything other than "accepted" (2026-09-23: created
    # 13:30:01.844, submitted 13:30:08.104, never filled).
    filled_at: datetime | None = None


class Broker(ABC):
    """Abstract interface every brokerage integration must implement."""

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Send *order* to the broker and return the result."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.  Returns ``True`` on success."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return current open positions."""
        ...

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account summary (buying power, equity, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release broker resources."""
        ...

    async def is_shortable(self, symbol: str) -> bool | None:
        """Return whether *symbol* may be sold short on this account.

        ``True`` = shortable, ``False`` = not shortable, ``None`` = unknown
        (broker cannot answer — callers should treat as "no evidence it is
        blocked" and let the order attempt proceed so behavior is unchanged).
        The default implementation returns ``None`` so existing
        broker implementations keep working without modification.
        """
        return None

    async def get_order(self, order_id: str) -> "OrderResult | None":
        """Return ONE order's CURRENT state, or ``None`` when unreadable.

        Callers use this to ask the broker "did this order actually execute?"
        instead of trusting the submission response.  The default returns
        ``None`` (= "cannot tell"), which callers MUST treat as "no verified
        execution" — never as a fill.
        """
        return None

    async def get_recent_fills(self, symbol: str, limit: int = 20) -> list[OrderResult]:
        """Return the most recent CLOSED orders for *symbol* (newest first).

        Used to price a position that disappeared from the broker from the
        execution that closed it — never from "the last fill for this symbol",
        which can be the previous session's trade.  The default returns an
        empty list, so callers book NO P&L rather than a fabricated one.
        """
        return []

    async def get_last_fill_price(self, symbol: str) -> float | None:
        """Return the average fill price of the most recent FILLED order
        for *symbol*, or ``None`` if there is none / it cannot be determined.

        Used to reconcile realised P&L when a tracked position disappears
        from the broker between syncs (e.g. a pending cleanup MARKET SELL
        filled at the next open).  The default implementation returns
        ``None`` so existing broker implementations keep working without
        modification, but callers then cannot price the removal and must
        drop the position WITHOUT booking P&L (never a fabricated loss).
        """
        return None
