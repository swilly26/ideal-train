"""Tests for the MAIN trader regime gate (live_trader.py).

The main trader is pure mean reversion and previously bought oversold dips
UNCONDITIONALLY — the #1 bleeding source on down days (catching falling
knives).  This suite verifies the new `_regime_gate_allows_long` trend
filter and that it actually gates entries in the live `_tick` loop.

Covers:
1. `_regime_gate_allows_long` blocks a LONG in a confirmed downtrend
   (price < MA10 AND RSI < 40), allows otherwise, and allows on
   insufficient data (fail-open).
2. End-to-end `_tick`: a mean-reversion BUY in a downtrend is skipped
   (never reaches _handle_buy); in an uptrend the BUY fires; with the gate
   disabled the BUY fires even in a downtrend (flag preserves old behavior).
"""
import pandas as pd
import pytest

import live_trader
from src.execution.position_manager import PositionManager
from src.strategies.base import Signal, SignalType


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _ohlcv(closes):
    idx = pd.date_range("2026-01-01 09:30", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


def _downtrend_df(n=40, start=100.0, step=-0.5):
    """Monotonic decline → price below MA10 and RSI ≈ 0 at the last bar."""
    return _ohlcv([start + i * step for i in range(n)])


def _uptrend_df(n=40, start=90.0, step=0.3):
    """Monotonic rise → price above MA10 and RSI ≈ 100 at the last bar."""
    return _ohlcv([start + i * step for i in range(n)])


class FakeProvider:
    def __init__(self, df):
        self._df = df

    async def fetch_bars(self, *a, **k):
        return type("MDF", (), {"df": self._df})()


def _make_trader(data):
    trader = object.__new__(live_trader.LiveTrader)
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader.day_trades = []
    trader.start_equity = 0.0
    trader.strategy = type("S", (), {})()
    trader.strategy.generate_signals = lambda data_: [
        Signal(symbol="NVDA", timestamp=pd.Timestamp("2026-01-01 10:00"),
               signal_type=SignalType.BUY, confidence=1.0,
               metadata={"z_score": -3.0}),
    ]
    trader.provider = FakeProvider(data)
    return trader


# ---------------------------------------------------------------------------
# 1. Regime gate function
# ---------------------------------------------------------------------------
class TestRegimeGateFunction:
    def test_blocks_long_in_downtrend(self):
        allow, reason = live_trader._regime_gate_allows_long(_downtrend_df())
        assert allow is False
        assert "downtrend" in reason

    def test_allows_long_in_uptrend(self):
        allow, _ = live_trader._regime_gate_allows_long(_uptrend_df())
        assert allow is True

    def test_allows_long_when_rsi_recovering_despite_below_ma(self):
        # Price dips below the MA but RSI stays strong → not a downtrend.
        closes = [100.0] * 30 + [98.0] * 10
        allow, _ = live_trader._regime_gate_allows_long(_ohlcv(closes))
        assert allow is True

    def test_allows_long_on_insufficient_data(self):
        # Fail-open: too few bars → don't block trading.
        allow, _ = live_trader._regime_gate_allows_long(_ohlcv([100.0, 101.0]))
        assert allow is True


# ---------------------------------------------------------------------------
# 2. End-to-end _tick gating
# ---------------------------------------------------------------------------
class TestLiveTickGate:
    @pytest.mark.asyncio
    async def test_tick_skips_buy_in_downtrend(self):
        """Mean-reversion BUY in a confirmed downtrend must NOT reach
        _handle_buy (the gate stops the falling-knife buy)."""
        trader = _make_trader(_downtrend_df())
        bought = []

        async def fake_handle_buy(symbol, price, confidence):
            bought.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        assert live_trader.ENABLE_REGIME_GATE is True
        await trader._tick(1)
        assert bought == []

    @pytest.mark.asyncio
    async def test_tick_buys_in_uptrend(self):
        """Same BUY signal fires normally in an uptrend (gate allows)."""
        trader = _make_trader(_uptrend_df())
        bought = []

        async def fake_handle_buy(symbol, price, confidence):
            bought.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        await trader._tick(1)
        assert bought == live_trader.SYMBOLS  # all symbols' dips get bought

    @pytest.mark.asyncio
    async def test_tick_buys_when_gate_disabled(self, monkeypatch):
        """With the gate off, old unconditional dip-buying is preserved
        (the flag retains the previous behavior exactly)."""
        trader = _make_trader(_downtrend_df())
        bought = []

        async def fake_handle_buy(symbol, price, confidence):
            bought.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        monkeypatch.setattr(live_trader, "ENABLE_REGIME_GATE", False)
        await trader._tick(1)
        assert bought == live_trader.SYMBOLS  # gate off → all dips bought
