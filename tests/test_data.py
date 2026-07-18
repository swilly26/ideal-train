"""Tests for the data provider module."""

import pandas as pd
import pytest

from src.data import DataProvider, MarketData, MarketDataFrame


class MockDataProvider(DataProvider):
    """Concrete provider that returns canned data for testing."""

    async def fetch_bars(self, symbol, start, end, timeframe="1min"):
        idx = pd.date_range(start, end, freq="1min")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            },
            index=idx,
        )
        return MarketDataFrame(df=df, symbol=symbol, timeframe=timeframe)

    async def subscribe_live(self, symbols, on_bar=None):
        pass

    async def close(self):
        pass


class TestMarketData:
    def test_market_data_creation(self):
        md = MarketData(
            symbol="AAPL",
            timestamp=pd.Timestamp("2026-01-01 10:00"),
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=5000,
        )
        assert md.symbol == "AAPL"
        assert md.close == 150.5
        assert md.volume == 5000


class TestDataProvider:
    @pytest.mark.asyncio
    async def test_fetch_bars_returns_dataframe(self):
        provider = MockDataProvider()
        result = await provider.fetch_bars(
            "AAPL",
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-01 00:05"),
        )
        assert isinstance(result, MarketDataFrame)
        assert result.symbol == "AAPL"
        assert len(result.df) == 6  # 5 min range, 1-min bars
        assert list(result.df.columns) == ["open", "high", "low", "close", "volume"]

    @pytest.mark.asyncio
    async def test_subscribe_live_does_not_raise(self):
        provider = MockDataProvider()
        await provider.subscribe_live(["AAPL"])

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        provider = MockDataProvider()
        await provider.close()
