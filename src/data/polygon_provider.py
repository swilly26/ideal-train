"""Polygon.io market data provider.

Implements ``DataProvider`` using the Polygon.io REST API for both historical
and real-time market data. Requires a free-tier (or paid) API key stored in
the ``POLYGON_API_KEY`` environment variable.

WebSocket streaming is documented but stubbed — see ``subscribe_live``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.data.provider import DataProvider, MarketData, MarketDataFrame

logger = logging.getLogger(__name__)

POLYGON_BASE_URL = "https://api.polygon.io"

# Mapping of our timeframe strings to Polygon's {multiplier, timespan}
_TIMEFRAME_MAP: dict[str, tuple[int, str]] = {
    "1min": (1, "minute"),
    "5min": (5, "minute"),
    "15min": (15, "minute"),
    "30min": (30, "minute"),
    "1h": (1, "hour"),
    "60min": (1, "hour"),
    "1day": (1, "day"),
    "1d": (1, "day"),
    "1week": (1, "week"),
    "1wk": (1, "week"),
    "1month": (1, "month"),
    "1mo": (1, "month"),
}


class PolygonProvider(DataProvider):
    """Market data provider backed by Polygon.io.

    Parameters
    ----------
    api_key : str | None
        Polygon.io API key. If ``None``, reads from the ``POLYGON_API_KEY``
        environment variable.
    """

    def __init__(self, api_key: str | None = None, **kwargs: object) -> None:
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        super().__init__(api_key=resolved_key, **kwargs)
        self._session = requests.Session()
        if not resolved_key:
            logger.warning(
                "No Polygon API key provided — REST calls will fail with 401. "
                "Set POLYGON_API_KEY or pass api_key=..."
            )

    def _make_url(self, path: str) -> str:
        return f"{POLYGON_BASE_URL}{path}"

    async def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1min",
    ) -> MarketDataFrame:
        """Fetch historical aggregate bars from Polygon.io.

        Uses ``GET /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}``.

        Parameters
        ----------
        symbol : str
            Ticker symbol (upper-case, e.g. ``"AAPL"``).
        start : datetime
            Start of the range (inclusive).
        end : datetime
            End of the range (inclusive).
        timeframe : str
            Bar size.

        Returns
        -------
        MarketDataFrame
            Normalised bars with a datetime index.

        Raises
        ------
        ValueError
            If the response is empty or the symbol is invalid.
        RuntimeError
            If the API key is missing or the HTTP request fails.
        """
        if not self._api_key:
            raise RuntimeError(
                "Polygon API key not set. Pass api_key=... or set POLYGON_API_KEY."
            )

        multiplier, timespan = _TIMEFRAME_MAP.get(
            timeframe, self._parse_custom_timeframe(timeframe)
        )

        # Polygon expects yyyy-MM-dd strings
        from_str = start.strftime("%Y-%m-%d")
        to_str = end.strftime("%Y-%m-%d")

        url = self._make_url(
            f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self._api_key,
        }

        try:
            resp = await asyncio.to_thread(
                self._session.get, url, params=params, timeout=30
            )
        except requests.RequestException as exc:
            logger.error("Polygon HTTP request failed: %s", exc)
            raise RuntimeError(f"Polygon API request failed: {exc}") from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "Polygon API returned 401 — your API key is invalid or expired."
            )
        if resp.status_code == 429:
            raise RuntimeError(
                "Polygon API rate limit exceeded. Wait and retry."
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Polygon API returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        results = data.get("results", [])

        if not results:
            raise ValueError(
                f"No data returned by Polygon for {symbol} "
                f"({from_str} → {to_str}). The ticker may be invalid or "
                f"there were no trades in that range."
            )

        # Polygon returns: v (volume), vw (vwap), o, c, h, l, t (timestamp ms), n
        records = []
        for bar in results:
            ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
            records.append(
                {
                    "timestamp": ts,
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": int(bar.get("v", 0)),
                }
            )

        df = pd.DataFrame(records)
        df = df.set_index("timestamp")

        return MarketDataFrame(df=df, symbol=symbol.upper(), timeframe=timeframe)

    async def subscribe_live(
        self,
        symbols: list[str],
        on_bar: "callable | None" = None,
    ) -> None:
        """Subscribe to real-time bar streaming via Polygon WebSocket.

        **Stub implementation.** The REST methods are fully functional; the
        WebSocket streaming layer is documented here for future implementation.

        Polygon WebSocket interface (``wss://socket.polygon.io/stocks``):

        1. Connect to ``wss://socket.polygon.io/stocks``.
        2. Authenticate: ``{"action":"auth","params":"<POLYGON_API_KEY>"}``.
        3. Subscribe: ``{"action":"subscribe","params":"A.<symbol>,A.<symbol>"}``
           where the ``A.`` prefix requests aggregate (minute) bars.
        4. On each bar, the server sends::

               {
                 "ev": "A",
                 "sym": "AAPL",
                 "v": 1234,      // volume
                 "av": 1234567,  // accumulated volume (for the day)
                 "op": 150.00,   // open
                 "vw": 150.50,   // volume-weighted average price
                 "o": 150.00,    // open (official)
                 "c": 151.00,    // close
                 "h": 151.50,    // high
                 "l": 149.50,    // low
                 "a": 150.75,    // average
                 "s": 1600000000000,  // start timestamp (ms)
                 "e": 1600000060000,  // end timestamp (ms)
               }

        5. Map each bar to ``MarketData`` and invoke ``on_bar``.

        Parameters
        ----------
        symbols : list[str]
            Symbols to subscribe to (e.g. ``["AAPL", "MSFT"]``).
        on_bar : callable, optional
            Callback receiving a ``MarketData`` instance per bar.
        """
        logger.info(
            "Polygon live subscription requested for %s — WebSocket streaming "
            "is not yet implemented (stub). Symbols: %s",
            symbols,
        )
        # No-op stub for now.

    async def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    @staticmethod
    def _parse_custom_timeframe(timeframe: str) -> tuple[int, str]:
        """Fallback parser for custom timeframes like ``"2min"``."""
        import re

        match = re.match(r"^(\d+)\s*(min|hour|day|week|month)s?$", timeframe)
        if not match:
            raise ValueError(
                f"Unsupported timeframe: {timeframe!r}. "
                f"Use e.g. '1min', '5min', '1h', '1day'."
            )
        multiplier = int(match.group(1))
        unit = match.group(2)
        return multiplier, unit
