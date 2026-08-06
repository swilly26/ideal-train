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
