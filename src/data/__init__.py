"""Market data ingestion layer.

Provides abstract base classes for market data providers and
standardised data structures for downstream consumers.
"""

from src.data.provider import DataProvider, MarketData, MarketDataFrame

__all__ = ["DataProvider", "MarketData", "MarketDataFrame"]
