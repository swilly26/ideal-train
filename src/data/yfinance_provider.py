"""Yahoo Finance market data provider.

Implements ``DataProvider`` using the ``yfinance`` library, which provides free
historical OHLCV data without an API key.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.data.provider import DataProvider, MarketDataFrame

logger = logging.getLogger(__name__)

# Map of our timeframe strings to yfinance interval strings
_TIMEFRAME_MAP: dict[str, str] = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "60min": "1h",
    "1day": "1d",
    "1d": "1d",
    "1week": "1wk",
    "1wk": "1wk",
    "1month": "1mo",
    "1mo": "1mo",
}


class YFinanceProvider(DataProvider):
    """Market data provider backed by Yahoo Finance.

    Parameters
    ----------
    api_key : str | None
        Ignored — Yahoo Finance does not require an API key.
    """

    def __init__(self, api_key: str | None = None, **kwargs: object) -> None:
        super().__init__(api_key=api_key, **kwargs)

    async def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1min",
    ) -> MarketDataFrame:
        """Download historical OHLCV bars from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``, ``"SPY"``).
        start : datetime
            Start of the date range (inclusive).
        end : datetime
            End of the date range (inclusive).
        timeframe : str
            Bar size (``"1min"``, ``"5min"``, ``"1h"``, ``"1day"``, etc.).

        Returns
        -------
        MarketDataFrame
            Normalised bars with a datetime index.

        Raises
        ------
        ValueError
            If the ticker returns no data or is invalid.
        """
        yf_interval = _TIMEFRAME_MAP.get(timeframe, timeframe)

        try:
            df = await asyncio.to_thread(
                yf.download,
                tickers=symbol,
                start=start,
                end=end,
                interval=yf_interval,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
                threads=False,
            )
        except Exception as exc:
            logger.error("Yahoo Finance download failed for %s: %s", symbol, exc)
            raise ValueError(
                f"Failed to fetch data for {symbol}: {exc}"
            ) from exc

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for {symbol} between {start} and {end}. "
                f"The ticker may be invalid or the date range may have no trading days."
            )

        # yfinance returns a DataFrame with columns: Open, High, Low, Close, Volume
        # Normalise to lowercase
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)

        # Drop any extra columns (yfinance sometimes adds Adj Close)
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]

        # Ensure volume is int64
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype("int64")

        return MarketDataFrame(df=df, symbol=symbol, timeframe=timeframe)

    async def subscribe_live(
        self,
        symbols: list[str],
        on_bar: "callable | None" = None,
    ) -> None:
        """Yahoo Finance does not support real-time streaming.

        Raises
        ------
        NotImplementedError
            Always — Yahoo Finance has no live streaming API.
        """
        raise NotImplementedError(
            "Yahoo Finance does not provide a real-time streaming API. "
            "Use PolygonProvider or another provider for live data."
        )

    async def close(self) -> None:
        """No persistent connections to close."""
