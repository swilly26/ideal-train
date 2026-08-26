"""Tests for the turbo trader MEAN-REVERSION SHORT (down-side) path.

On a confirmed down day (price below MA10 AND RSI < 40) the regime gate
blocks LONGs.  This suite verifies that instead of just doing nothing, the
turbo engine OPENS A SHORT on the gated mean-reversion oversold dip (the
falling knife) so the down-tape is monetized rather than only avoided, and
that momentum longs are also prevented from fighting the trend.

Covers:
1. Gated MR LONG → SHORT opens (via _handle_short_sell).
2. ENABLE_MEAN_REVERSION_SHORT=False → gated MR LONG is purely skipped
   (no short, no buy) — flag preserves pre-change behavior.
3. An existing SHORT is never doubled up — a BUY while short goes to the
   COVER path (buy-to-cover), not a new short.
4. A momentum LONG in a confirmed downtrend is also gated (no trend-fighting).
5. Flag default is on.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import turbo_trader
from src.strategies.base import Signal, SignalType

# Reuse the canonical helpers from the PR #23 suite so both suites agree.
from tests.test_turbo_regime_short import (
    FakeBroker,
    _downtrend_df,
    _make_trader,
)


def _mr_buy_signal(symbol="SOXL", confidence=1.0):
    return Signal(
        symbol=symbol,
        timestamp=pd.Timestamp("2026-01-01 10:00"),
        signal_type=SignalType.BUY,
        confidence=confidence,
        metadata={"z_score": -3.0, "strategy": "mean_reversion"},
    )


def _momentum_buy_signal(symbol="SOXL", confidence=1.0):
    return Signal(
        symbol=symbol,
        timestamp=pd.Timestamp("2026-01-01 10:00"),
        signal_type=SignalType.BUY,
        confidence=confidence,
        metadata={"strategy": "momentum", "rsi": 50.0},
    )


class TestMeanReversionShort:
    def test_flag_defaults_on(self):
        assert turbo_trader.ENABLE_MEAN_REVERSION_SHORT is True

    @pytest.mark.asyncio
    async def test_gated_mr_long_opens_short(self, monkeypatch):
        """A mean-reversion BUY (oversold dip) in a confirmed downtrend must
        route to _handle_short_sell — monetizing the knife drawdown."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        shorts = []

        async def fake_short_sell(symbol, price, confidence, strategy=""):
            shorts.append((symbol, price, confidence, strategy))

        async def fake_check_risk_stops():
            pass

        trader._handle_short_sell = fake_short_sell
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = [_mr_buy_signal()]
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(turbo_trader, "_generate_momentum_signals",
                            lambda *a, **k: [])

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        monkeypatch.setattr(turbo_trader, "SYMBOLS", ["SOXL"])
        await trader._tick(1)
        assert len(shorts) == 1
        assert shorts[0][0] == "SOXL"
        assert shorts[0][3] == "mean_reversion"  # strategy tag preserved

    @pytest.mark.asyncio
    async def test_gated_mr_long_skipped_when_short_flag_disabled(self, monkeypatch):
        """Flag off → pure skip: no short AND no buy (old safe behavior)."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        shorts = []
        bought = []

        async def fake_short_sell(symbol, price, confidence, strategy=""):
            shorts.append(symbol)

        async def fake_handle_buy(symbol, price, confidence, strategy=""):
            bought.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_short_sell = fake_short_sell
        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = [_mr_buy_signal()]
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(turbo_trader, "_generate_momentum_signals",
                            lambda *a, **k: [])

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        monkeypatch.setattr(turbo_trader, "ENABLE_MEAN_REVERSION_SHORT", False)
        monkeypatch.setattr(turbo_trader, "SYMBOLS", ["SOXL"])
        await trader._tick(1)
        assert shorts == []
        assert bought == []

    @pytest.mark.asyncio
    async def test_no_double_up_when_already_short(self, monkeypatch):
        """A BUY signal while already SHORT must go to COVER (buy-to-cover),
        never double down — even when the gate would otherwise short."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        covers = []

        # Open an existing short position.
        trader.pm.open_position("SOXL", -100.0, 40.0)
        trader._entry_times["SOXL"] = datetime.now(timezone.utc)

        async def fake_handle_sell(symbol, price, confidence, strategy=""):
            covers.append((symbol, price))

        async def fake_short_sell(symbol, price, confidence, strategy=""):
            pytest.fail("must not open a second short")

        async def fake_check_risk_stops():
            pass

        trader._handle_sell = fake_handle_sell
        trader._handle_short_sell = fake_short_sell
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = [_mr_buy_signal()]
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(turbo_trader, "_generate_momentum_signals",
                            lambda *a, **k: [])

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        monkeypatch.setattr(turbo_trader, "SYMBOLS", ["SOXL"])
        await trader._tick(1)
        # BUY while short → cover instead of double-down / MR-short.
        assert len(covers) == 1
        assert covers[0][0] == "SOXL"


class TestMomentumLongGated:
    @pytest.mark.asyncio
    async def test_momentum_long_gated_in_downtrend(self, monkeypatch):
        """A momentum LONG must also be suppressed in a confirmed downtrend
        (no strategy fights the trend) — and must NOT trigger an MR-short
        (the MR-short path is reserved for mean-reversion dips)."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        shorts = []
        bought = []

        async def fake_short_sell(symbol, price, confidence, strategy=""):
            shorts.append(symbol)

        async def fake_handle_buy(symbol, price, confidence, strategy=""):
            bought.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_short_sell = fake_short_sell
        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = []
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        # Inject a momentum BUY (normally wouldn't generate below the MA, but
        # verify the gate catches the trend-fighting LONG regardless).
        monkeypatch.setattr(
            turbo_trader, "_generate_momentum_signals",
            lambda data, symbol, **k: [_momentum_buy_signal(symbol)],
        )

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        monkeypatch.setattr(turbo_trader, "SYMBOLS", ["SOXL"])
        await trader._tick(1)
        assert bought == []      # momentum long gated
        assert shorts == []      # not an MR dip → no MR-short
