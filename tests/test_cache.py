"""Tests for the market data cache."""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache import MarketDataCache
from src.data.provider import MarketDataFrame


@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def cache(temp_cache_dir):
    return MarketDataCache(cache_dir=temp_cache_dir, ttl_seconds=None)


@pytest.fixture
def sample_data():
    idx = pd.date_range("2026-01-01 09:30", periods=5, freq="1min")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 101.0, 103.0],
            "high": [102.0, 103.0, 104.0, 103.0, 105.0],
            "low": [99.0, 100.0, 101.0, 100.0, 102.0],
            "close": [101.0, 102.0, 103.0, 102.0, 104.0],
            "volume": [1000, 1100, 1200, 1100, 1300],
        },
        index=idx,
    )
    return MarketDataFrame(df=df, symbol="AAPL", timeframe="1min")


class TestMarketDataCache:
    def test_get_miss_returns_none(self, cache):
        result = cache.get(
            "yfinance",
            "AAPL",
            datetime(2026, 1, 1),
            datetime(2026, 1, 2),
            "1day",
        )
        assert result is None

    def test_put_and_get_roundtrip(self, cache, sample_data):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1)

        cache.put("yfinance", sample_data, start, end)

        result = cache.get("yfinance", "AAPL", start, end, "1min")
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.timeframe == "1min"
        assert len(result.df) == len(sample_data.df)
        assert list(result.df.columns) == list(sample_data.df.columns)

    def test_different_providers_independent(self, cache, sample_data):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1)

        cache.put("yfinance", sample_data, start, end)

        # Same parameters but different provider → miss
        result = cache.get("polygon", "AAPL", start, end, "1min")
        assert result is None

    def test_different_timeframes_independent(self, cache, sample_data):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1)

        cache.put("yfinance", sample_data, start, end)

        result = cache.get("yfinance", "AAPL", start, end, "5min")
        assert result is None

    def test_ttl_expiry(self, temp_cache_dir, sample_data):
        cache = MarketDataCache(cache_dir=temp_cache_dir, ttl_seconds=1)
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1)

        cache.put("yfinance", sample_data, start, end)

        # Should hit immediately
        result = cache.get("yfinance", "AAPL", start, end, "1min")
        assert result is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        result = cache.get("yfinance", "AAPL", start, end, "1min")
        assert result is None

    def test_clear_removes_all_entries(self, cache, sample_data):
        cache.put("yfinance", sample_data, datetime(2026, 1, 1), datetime(2026, 1, 2))
        cache.put("polygon", sample_data, datetime(2026, 1, 1), datetime(2026, 1, 2))

        stats_before = cache.stats()
        assert stats_before["entries"] >= 2

        removed = cache.clear()
        assert removed > 0

        stats_after = cache.stats()
        assert stats_after["entries"] == 0

    def test_stats_reports_counts(self, cache, sample_data):
        cache.put("yfinance", sample_data, datetime(2026, 1, 1), datetime(2026, 1, 2))
        stats = cache.stats()

        assert "entries" in stats
        assert "files" in stats
        assert "size_bytes" in stats
        assert stats["entries"] >= 1
        assert stats["files"] >= 2  # .parquet + .json

    def test_overwrite_same_key(self, cache, sample_data):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1)

        cache.put("yfinance", sample_data, start, end)

        # Create different data for same key
        idx = pd.date_range("2026-01-01 09:30", periods=3, freq="1min")
        new_df = pd.DataFrame(
            {
                "open": [200.0] * 3,
                "high": [201.0] * 3,
                "low": [199.0] * 3,
                "close": [200.5] * 3,
                "volume": [500] * 3,
            },
            index=idx,
        )
        new_data = MarketDataFrame(df=new_df, symbol="AAPL", timeframe="1min")
        cache.put("yfinance", new_data, start, end)

        result = cache.get("yfinance", "AAPL", start, end, "1min")
        assert result is not None
        assert len(result.df) == 3  # New data, not old
        assert result.df.iloc[0]["open"] == 200.0

    def test_none_ttl_never_expires(self, temp_cache_dir, sample_data):
        cache = MarketDataCache(cache_dir=temp_cache_dir, ttl_seconds=None)
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1)

        cache.put("yfinance", sample_data, start, end)
        time.sleep(0.5)

        result = cache.get("yfinance", "AAPL", start, end, "1min")
        assert result is not None
