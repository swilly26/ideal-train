"""Trade execution layer.

Abstracts communication with brokerage APIs so the rest of the engine
never depends on a specific broker's SDK.
"""

from src.execution.broker import Broker, Order, OrderResult, OrderSide, OrderType
from src.execution.position_manager import PositionManager

__all__ = [
    "Broker",
    "Order",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "PositionManager",
]
