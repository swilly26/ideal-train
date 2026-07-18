"""Abstract base class for market data providers.

All data providers (Alpaca, Polygon, Yahoo Finance, etc.) must
implement this interface so the rest of the engine can consume
market data without coupling to a specific vendor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class MarketData:
    """Normalised container for a single OHLCV bar.

    Providers must map their native format into this structure.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MarketDataFrame:
    """A DataFrame wrapper that carries metadata useful for strategies.

    Attributes
    ----------
    df : pd.DataFrame
        Columns: timestamp, open, high, low, close, volume.
        Index: timestamp (datetime).
    symbol : str
        The ticker symbol this data represents.
    timeframe : str
        Bar size, e.g. "1min", "5min", "1day".
    """

    df: pd.DataFrame
    symbol: str
    timeframe: str


class DataProvider(ABC):
    """Abstract interface every market data provider must implement."""

    def __init__(self, api_key: str | None = None, **kwargs: object) -> None:
        self._api_key = api_key

    @abstractmethod
    async def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1min",
    ) -> MarketDataFrame:
        """Fetch historical OHLCV bars for *symbol* between *start* and *end*.

        Returns
        -------
        MarketDataFrame
            Normalised bars with a datetime index.
        """
        ...

    @abstractmethod
    async def subscribe_live(
        self,
        symbols: list[str],
        on_bar: "callable | None" = None,
    ) -> None:
        """Subscribe to a real-time bar stream.

        Parameters
        ----------
        symbols : list[str]
            Symbols to subscribe to.
        on_bar : callable, optional
            Callback invoked with a ``MarketData`` instance on each bar.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...
