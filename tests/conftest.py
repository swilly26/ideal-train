"""Shared pytest fixtures and configuration."""

import pytest


@pytest.fixture
def sample_ohlcv():
    """Return a small OHLCV DataFrame for use across test modules."""
    import pandas as pd

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
