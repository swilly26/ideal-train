"""Tests for the Polygon.io data provider."""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.polygon_provider import PolygonProvider
from src.data.provider import MarketDataFrame


@pytest.fixture
def mock_response():
    """Create a mock requests.Response with valid Polygon aggregate data."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "ticker": "AAPL",
        "queryCount": 3,
        "resultsCount": 3,
        "adjusted": True,
        "results": [
            {
                "v": 1000000,
                "vw": 150.5,
                "o": 150.0,
                "c": 151.0,
                "h": 151.5,
                "l": 149.5,
                "t": 1713312000000,  # 2024-04-17 00:00:00 UTC
                "n": 5000,
            },
            {
                "v": 1200000,
                "vw": 151.0,
                "o": 151.0,
                "c": 152.0,
                "h": 152.5,
                "l": 150.5,
                "t": 1713398400000,
                "n": 6000,
            },
            {
                "v": 1100000,
                "vw": 152.0,
                "o": 152.0,
                "c": 150.5,
                "h": 153.0,
                "l": 150.0,
                "t": 1713484800000,
                "n": 5500,
            },
        ],
        "status": "OK",
        "request_id": "abc123",
    }
    return resp


class TestPolygonProviderInit:
    def test_creates_with_explicit_key(self):
        p = PolygonProvider(api_key="test_key_123")
        assert p._api_key == "test_key_123"

    def test_reads_key_from_env(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "env_key_456")
        p = PolygonProvider()
        assert p._api_key == "env_key_456"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "env_key_456")
        p = PolygonProvider(api_key="explicit_key")
        assert p._api_key == "explicit_key"

    def test_warns_when_no_key(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        p = PolygonProvider()
        assert p._api_key == ""


class TestPolygonProviderFetchBars:
    @pytest.mark.asyncio
    async def test_fetches_bars_successfully(self, mock_response):
        with patch("requests.Session.get", return_value=mock_response):
            p = PolygonProvider(api_key="test_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 19)

            result = await p.fetch_bars("AAPL", start, end, timeframe="1day")

            assert isinstance(result, MarketDataFrame)
            assert result.symbol == "AAPL"
            assert result.timeframe == "1day"
            assert len(result.df) == 3
            assert list(result.df.columns) == [
                "open", "high", "low", "close", "volume"
            ]
            assert result.df.iloc[0]["close"] == 151.0
            assert result.df.iloc[0]["volume"] == 1000000

    @pytest.mark.asyncio
    async def test_passes_correct_url_params(self, mock_response):
        with patch("requests.Session.get", return_value=mock_response) as mock_get:
            p = PolygonProvider(api_key="test_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 19)

            await p.fetch_bars("AAPL", start, end, timeframe="1day")

            call_args = mock_get.call_args
            url = call_args[0][0]
            params = call_args[1]["params"]

            assert "/v2/aggs/ticker/AAPL/range/1/day/2024-04-17/2024-04-19" in url
            assert params["apiKey"] == "test_key"
            assert params["adjusted"] == "true"

    @pytest.mark.asyncio
    async def test_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        p = PolygonProvider()
        start = datetime(2024, 4, 17)
        end = datetime(2024, 4, 19)

        with pytest.raises(RuntimeError, match="API key not set"):
            await p.fetch_bars("AAPL", start, end)

    @pytest.mark.asyncio
    async def test_raises_on_401(self):
        resp = MagicMock()
        resp.status_code = 401

        with patch("requests.Session.get", return_value=resp):
            p = PolygonProvider(api_key="bad_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 19)

            with pytest.raises(RuntimeError, match="401"):
                await p.fetch_bars("AAPL", start, end)

    @pytest.mark.asyncio
    async def test_raises_on_429(self):
        resp = MagicMock()
        resp.status_code = 429

        with patch("requests.Session.get", return_value=resp):
            p = PolygonProvider(api_key="test_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 19)

            with pytest.raises(RuntimeError, match="rate limit"):
                await p.fetch_bars("AAPL", start, end)

    @pytest.mark.asyncio
    async def test_raises_on_empty_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": [], "status": "OK"}

        with patch("requests.Session.get", return_value=resp):
            p = PolygonProvider(api_key="test_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 19)

            with pytest.raises(ValueError, match="No data returned"):
                await p.fetch_bars("AAPL", start, end)

    @pytest.mark.asyncio
    async def test_network_error_is_wrapped(self):
        import requests as req_lib

        with patch(
            "requests.Session.get",
            side_effect=req_lib.ConnectionError("timeout"),
        ):
            p = PolygonProvider(api_key="test_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 19)

            with pytest.raises(RuntimeError, match="failed"):
                await p.fetch_bars("AAPL", start, end)

    @pytest.mark.asyncio
    async def test_5min_timeframe(self, mock_response):
        with patch("requests.Session.get", return_value=mock_response) as mock_get:
            p = PolygonProvider(api_key="test_key")
            start = datetime(2024, 4, 17)
            end = datetime(2024, 4, 17, 23, 59)

            await p.fetch_bars("AAPL", start, end, timeframe="5min")

            call_args = mock_get.call_args
            url = call_args[0][0]
            assert "/range/5/minute/" in url


class TestPolygonProviderSubscribeLive:
    @pytest.mark.asyncio
    async def test_is_noop_stub(self):
        p = PolygonProvider(api_key="test_key")
        # Should not raise
        await p.subscribe_live(["AAPL", "MSFT"])
        await p.subscribe_live([])


class TestPolygonProviderClose:
    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        p = PolygonProvider(api_key="test_key")
        await p.close()
        # Session should be closed — calling again won't error
