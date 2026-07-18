"""Tests for the Yahoo Finance data provider."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.provider import MarketDataFrame
from src.data.yfinance_provider import YFinanceProvider


@pytest.fixture
def provider():
    return YFinanceProvider()


class TestYFinanceProviderInit:
    def test_creates_without_api_key(self):
        p = YFinanceProvider()
        assert p is not None

    def test_ignores_api_key_kwarg(self):
        p = YFinanceProvider(api_key="ignored")
        assert p is not None


class TestYFinanceProviderFetchBars:
    @pytest.mark.asyncio
    async def test_fetches_real_data_for_valid_ticker(self, provider):
        """Integration test: fetches recent daily bars for AAPL."""
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=7)

        result = await provider.fetch_bars(
            "AAPL", start, end, timeframe="1day"
        )

        assert isinstance(result, MarketDataFrame)
        assert result.symbol == "AAPL"
        assert result.timeframe == "1day"
        assert len(result.df) > 0
        assert list(result.df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(result.df.index, pd.DatetimeIndex)

    @pytest.mark.asyncio
    async def test_fetches_1min_data(self, provider):
        """Integration test: fetches 1-minute bars for a valid ticker.

        Uses a known working date during market hours to avoid issues
        with after-hours / weekend data availability.
        """
        # Use a recent trading day during US market hours (9:30 AM – 4:00 PM ET)
        # 2026-07-17 was a Friday and a trading day
        start = datetime(2026, 7, 17, 14, 0)  # 2:00 PM UTC ≈ 10:00 AM ET
        end = datetime(2026, 7, 17, 15, 0)    # 3:00 PM UTC ≈ 11:00 AM ET

        result = await provider.fetch_bars(
            "SPY", start, end, timeframe="1min"
        )

        assert isinstance(result, MarketDataFrame)
        assert result.timeframe == "1min"
        assert len(result.df) > 0

    @pytest.mark.asyncio
    async def test_raises_for_invalid_ticker(self, provider):
        """A clearly invalid ticker should raise ValueError."""
        end = datetime.now()
        start = end - timedelta(days=30)

        with pytest.raises(ValueError, match="No data returned"):
            await provider.fetch_bars(
                "ZZZZ_INVALID_TICKER_XYZ", start, end, timeframe="1day"
            )

    @pytest.mark.asyncio
    async def test_raises_for_future_date_range(self, provider):
        """Asking for data entirely in the future should fail."""
        future_start = datetime.now() + timedelta(days=365)
        future_end = future_start + timedelta(days=7)

        with pytest.raises(ValueError):
            await provider.fetch_bars(
                "AAPL", future_start, future_end, timeframe="1day"
            )

    @pytest.mark.asyncio
    async def test_market_dataframe_has_metadata(self, provider):
        """Verify MarketDataFrame carries correct symbol and timeframe."""
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=7)

        result = await provider.fetch_bars("MSFT", start, end, timeframe="1day")
        assert result.symbol == "MSFT"
        assert result.timeframe == "1day"

    @pytest.mark.asyncio
    async def test_volume_is_int(self, provider):
        """Volume column should be int64."""
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=7)

        result = await provider.fetch_bars("AAPL", start, end, timeframe="1day")
        assert result.df["volume"].dtype == "int64"

    @pytest.mark.asyncio
    async def test_no_adj_close_column(self, provider):
        """Ensure extra columns like Adj Close are stripped."""
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=7)

        result = await provider.fetch_bars("AAPL", start, end, timeframe="1day")
        for col in result.df.columns:
            assert col in ["open", "high", "low", "close", "volume"]


class TestYFinanceProviderSubscribeLive:
    @pytest.mark.asyncio
    async def test_raises_not_implemented(self, provider):
        with pytest.raises(NotImplementedError):
            await provider.subscribe_live(["AAPL"])


class TestYFinanceProviderClose:
    @pytest.mark.asyncio
    async def test_close_does_not_raise(self, provider):
        await provider.close()
