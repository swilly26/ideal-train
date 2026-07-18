"""Shared pytest fixtures and configuration."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv():
    """Return a small OHLCV DataFrame for use across test modules."""
    idx = pd.date_range("2026-01-01 09:30", periods=20, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(20)],
            "high": [101.0 + i * 0.1 for i in range(20)],
            "low": [99.0 + i * 0.1 for i in range(20)],
            "close": [100.5 + i * 0.1 for i in range(20)],
            "volume": [1000] * 20,
        },
        index=idx,
    )


@pytest.fixture
def uptrend_ohlcv():
    """Steadily rising prices — good for momentum and upside breakout."""
    periods = 40
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(42)
    base = 100.0
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        base += 0.5  # steady uptrend
        noise = rng.normal(0, 0.2)
        close = base + noise
        high = close + abs(rng.normal(0.15, 0.05))
        low = close - abs(rng.normal(0.15, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def downtrend_ohlcv():
    """Steadily falling prices — should trigger momentum SELL."""
    periods = 40
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(99)
    base = 100.0
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        base -= 0.5  # steady downtrend
        noise = rng.normal(0, 0.2)
        close = base + noise
        high = close + abs(rng.normal(0.15, 0.05))
        low = close - abs(rng.normal(0.15, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def oscillating_ohlcv():
    """Prices oscillating around a fixed mean — ideal for mean reversion."""
    periods = 60
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(7)
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        # Sine wave around 100 with amplitude 5, plus small noise
        close = 100.0 + 5.0 * np.sin(2 * np.pi * i / 20) + rng.normal(0, 0.2)
        high = close + abs(rng.normal(0.15, 0.05))
        low = close - abs(rng.normal(0.15, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def breakout_ohlcv():
    """Range-bound for most of the period, then a sharp upside breakout."""
    periods = 40
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(13)
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        if i < 30:
            # Tight range between 99 and 101
            close = 100.0 + rng.normal(0, 0.3)
            high = close + abs(rng.normal(0.2, 0.05))
            low = close - abs(rng.normal(0.2, 0.05))
        else:
            # Sharp breakout upward
            close = 105.0 + (i - 30) * 1.0 + rng.normal(0, 0.2)
            high = close + abs(rng.normal(0.3, 0.05))
            low = close - abs(rng.normal(0.3, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )
