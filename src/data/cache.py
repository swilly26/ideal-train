"""File-based cache for historical market data requests.

Caches ``MarketDataFrame`` objects to disk so that repeated backtesting runs
don't hammer the data provider's API. Each cache entry is keyed by a hash
of (provider, symbol, start, end, timeframe).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.provider import MarketDataFrame

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".algofLow" / "cache"


class MarketDataCache:
    """Simple file-based cache for market data.

    Each entry is stored as a pair of files in the cache directory:

    * ``<hash>.parquet`` — the OHLCV DataFrame in Parquet format.
    * ``<hash>.json`` — metadata (symbol, timeframe, created timestamp).

    Parameters
    ----------
    cache_dir : str | Path
        Directory where cache files are stored.
    ttl_seconds : int | None
        Time-to-live in seconds. Entries older than this are considered stale.
        ``None`` means no expiry.
    """

    def __init__(
        self,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
        ttl_seconds: int | None = 3600,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._ttl_seconds = ttl_seconds
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        provider_name: str,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> MarketDataFrame | None:
        """Retrieve a cached ``MarketDataFrame`` if one exists and is fresh.

        Returns ``None`` on cache miss or stale entry.
        """
        key = self._make_key(provider_name, symbol, start, end, timeframe)
        parquet_path = self._parquet_path(key)
        meta_path = self._meta_path(key)

        if not parquet_path.exists() or not meta_path.exists():
            return None

        # Check TTL
        if self._ttl_seconds is not None:
            mtime = parquet_path.stat().st_mtime
            age = time.time() - mtime
            if age > self._ttl_seconds:
                logger.debug("Cache expired for %s (age=%.0fs)", key[:12], age)
                return None

        try:
            meta = json.loads(meta_path.read_text())
            df = pd.read_parquet(parquet_path)
            return MarketDataFrame(
                df=df,
                symbol=meta["symbol"],
                timeframe=meta["timeframe"],
            )
        except Exception as exc:
            logger.warning("Failed to read cache entry %s: %s", key[:12], exc)
            return None

    def put(
        self,
        provider_name: str,
        data: MarketDataFrame,
        start: datetime,
        end: datetime,
    ) -> None:
        """Store a ``MarketDataFrame`` in the cache."""
        key = self._make_key(
            provider_name, data.symbol, start, end, data.timeframe
        )
        parquet_path = self._parquet_path(key)
        meta_path = self._meta_path(key)

        try:
            data.df.to_parquet(parquet_path, index=True)
            meta = {
                "symbol": data.symbol,
                "timeframe": data.timeframe,
                "provider": provider_name,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "created": time.time(),
            }
            meta_path.write_text(json.dumps(meta))
            logger.debug("Cached %s → %s (%d bars)", data.symbol, key[:12], len(data.df))
        except Exception as exc:
            logger.warning("Failed to write cache entry: %s", exc)

    def clear(self) -> int:
        """Remove all cached entries. Returns count of files removed."""
        count = 0
        for f in self._cache_dir.glob("*"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cache files", count)
        return count

    def stats(self) -> dict:
        """Return cache statistics: entry count and total size in bytes."""
        files = list(self._cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "entries": len(files) // 2,  # Each entry is 2 files
            "files": len(files),
            "size_bytes": total_size,
            "dir": str(self._cache_dir),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(
        self,
        provider_name: str,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> str:
        raw = f"{provider_name}|{symbol}|{start.isoformat()}|{end.isoformat()}|{timeframe}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _parquet_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _meta_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"
