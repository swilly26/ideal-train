"""Market data ingestion layer.

Provides abstract base classes for market data providers and
standardised data structures for downstream consumers.
"""

from src.data.cache import MarketDataCache
from src.data.provider import DataProvider, MarketData, MarketDataFrame
from src.data.validation import clean_market_data, validate_market_data

__all__ = [
    "DataProvider",
    "MarketData",
    "MarketDataFrame",
    "MarketDataCache",
    "clean_market_data",
    "validate_market_data",
]
