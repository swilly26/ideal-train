"""Tests for shared technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.indicators import (
    sma,
    rolling_high,
    rolling_low,
    bollinger_bands,
    z_score,
    momentum,
    breakout_distance,
)


class TestSMA:
    def test_sma_calculation(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(prices, period=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert result.iloc[3] == 3.0  # (2+3+4)/3
        assert result.iloc[4] == 4.0  # (3+4+5)/3

    def test_sma_single_value(self):
        prices = pd.Series([10.0])
        result = sma(prices, period=1)
        assert result.iloc[0] == 10.0

    def test_sma_period_longer_than_data(self):
        prices = pd.Series([1.0, 2.0, 3.0])
        result = sma(prices, period=5)
        assert result.isna().all()


class TestRollingHighLow:
    def test_rolling_high(self):
        prices = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        result = rolling_high(prices, period=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 3.0  # max(1,3,2)
        assert result.iloc[3] == 5.0  # max(3,2,5)
        assert result.iloc[4] == 5.0  # max(2,5,4)

    def test_rolling_low(self):
        prices = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        result = rolling_low(prices, period=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 1.0  # min(1,3,2)
        assert result.iloc[3] == 2.0  # min(3,2,5)
        assert result.iloc[4] == 2.0  # min(2,5,4)


class TestBollingerBands:
    def test_bollinger_bands_structure(self):
        prices = pd.Series(range(1, 21), dtype=float)
        middle, upper, lower = bollinger_bands(prices, period=5, std_dev_multiplier=2.0)
        assert len(middle) == 20
        assert len(upper) == 20
        assert len(lower) == 20
        # First 4 should be NaN
        assert middle.iloc[:4].isna().all()
        # After warmup, upper > middle > lower
        valid = middle.dropna()
        assert (upper.loc[valid.index] >= middle.loc[valid.index]).all()
        assert (middle.loc[valid.index] >= lower.loc[valid.index]).all()

    def test_bollinger_bands_constant_price(self):
        """Constant price → std = 0 → bands collapse to middle."""
        prices = pd.Series([100.0] * 10)
        middle, upper, lower = bollinger_bands(prices, period=5)
        valid = middle.dropna()
        assert (upper.loc[valid.index] == 100.0).all()
        assert (lower.loc[valid.index] == 100.0).all()


class TestZScore:
    def test_z_score_calculation(self):
        # Values: mean=3, std≈1.58 for [1,2,3,4,5]
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = z_score(prices, period=5)
        assert np.isnan(result.iloc[:4]).all()
        # For [1,2,3,4,5]: mean=3, std≈1.5811
        # z5 = (5-3)/1.5811 ≈ 1.265
        assert 1.2 < result.iloc[4] < 1.3

    def test_z_score_zero_for_constant(self):
        prices = pd.Series([100.0] * 10)
        result = z_score(prices, period=5)
        valid = result.dropna()
        # With near-zero std, z-score should be 0 (handles div-by-zero)
        assert (valid == 0.0).all()


class TestMomentum:
    def test_momentum_calculation(self):
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        result = momentum(prices, period=5)
        # Last bar: (104 - 102) / 102 ≈ 0.0196
        assert result.iloc[4] == pytest.approx((104.0 - 102.0) / 102.0)

    def test_momentum_positive_and_negative(self):
        # Up then down
        prices = pd.Series([100.0, 102.0, 104.0, 102.0, 100.0])
        result = momentum(prices, period=5)
        # After 5 bars, mean=101.6, last=100, momentum = (100-101.6)/101.6 < 0
        assert result.iloc[4] < 0


class TestBreakoutDistance:
    def test_breakout_distance_inside_channel(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="1min")
        close = pd.Series([100.0] * 10, index=idx)
        high = pd.Series([101.0] * 10, index=idx)
        low = pd.Series([99.0] * 10, index=idx)
        upside, downside = breakout_distance(close, high, low, period=5)
        valid = upside.dropna()
        # close == high → upside == 0; close == low → downside == 0
        assert (valid <= 0).all()
        valid_d = downside.dropna()
        assert (valid_d <= 0).all()

    def test_breakout_distance_above_channel(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="1min")
        close = pd.Series([100.0] * 9 + [110.0], index=idx)
        high = pd.Series([101.0] * 10, index=idx)
        low = pd.Series([99.0] * 10, index=idx)
        upside, downside = breakout_distance(close, high, low, period=5)
        # Last bar: (110 - 101) / 101 ≈ 0.089
        assert upside.iloc[9] > 0.08
