"""Tests for the data validation module."""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    clean_market_data,
    detect_outliers,
    ensure_required_columns,
    normalize_columns,
    validate_market_data,
)


def make_df(data_dict: dict | None = None) -> pd.DataFrame:
    """Create a minimal valid OHLCV DataFrame for testing."""
    idx = pd.date_range("2026-01-01 09:30", periods=5, freq="1min")
    defaults = {
        "open": [100.0, 101.0, 102.0, 101.0, 103.0],
        "high": [102.0, 103.0, 104.0, 103.0, 105.0],
        "low": [99.0, 100.0, 101.0, 100.0, 102.0],
        "close": [101.0, 102.0, 103.0, 102.0, 104.0],
        "volume": [1000, 1100, 1200, 1100, 1300],
    }
    if data_dict:
        defaults.update(data_dict)
    return pd.DataFrame(defaults, index=idx)


class TestNormalizeColumns:
    def test_lowercase_unchanged(self):
        df = make_df()
        result = normalize_columns(df)
        assert list(result.columns) == list(df.columns)

    def test_uppercase_renamed(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0], "Close": [1.5], "Volume": [100]}
        )
        result = normalize_columns(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_single_letter_renamed(self):
        df = pd.DataFrame(
            {"o": [1], "h": [2], "l": [0], "c": [1.5], "v": [100]}
        )
        result = normalize_columns(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_mixed_columns(self):
        df = pd.DataFrame({"Open": [1], "l": [0], "close": [1.5]})
        result = normalize_columns(df)
        assert "open" in result.columns


class TestEnsureRequiredColumns:
    def test_passes_with_all_columns(self):
        df = make_df()
        ensure_required_columns(df)  # Should not raise

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            ensure_required_columns(df)


class TestValidateMarketData:
    def test_clean_data_has_no_issues(self):
        df = make_df()
        issues = validate_market_data(df)
        assert issues == []

    def test_detects_nan(self):
        df = make_df({"close": [101.0, np.nan, 103.0, 102.0, 104.0]})
        issues = validate_market_data(df)
        assert any("NaN" in i for i in issues)

    def test_detects_negative_prices(self):
        df = make_df({"low": [99.0, 100.0, -1.0, 100.0, 102.0]})
        issues = validate_market_data(df)
        assert any("Negative low" in i for i in issues)

    def test_detects_high_less_than_low(self):
        df = make_df({
            "high": [102.0, 99.0, 104.0, 103.0, 105.0],  # bar 1: high < low (100)
        })
        issues = validate_market_data(df)
        assert any("High < Low" in i for i in issues)

    def test_detects_open_outside_range(self):
        df = make_df({"open": [200.0, 101.0, 102.0, 101.0, 103.0]})
        issues = validate_market_data(df)
        assert any("Open outside" in i for i in issues)

    def test_detects_close_outside_range(self):
        df = make_df({"close": [200.0, 102.0, 103.0, 102.0, 104.0]})
        issues = validate_market_data(df)
        assert any("Close outside" in i for i in issues)

    def test_detects_negative_volume(self):
        df = make_df({"volume": [1000, -500, 1200, 1100, 1300]})
        issues = validate_market_data(df)
        assert any("Negative volume" in i for i in issues)

    def test_detects_zero_volume(self):
        df = make_df({"volume": [1000, 0, 1200, 1100, 1300]})
        issues = validate_market_data(df)
        assert any("Zero volume" in i for i in issues)

    def test_detects_gaps(self):
        idx = [
            pd.Timestamp("2026-01-01 09:30"),
            pd.Timestamp("2026-01-01 09:31"),
            pd.Timestamp("2026-01-01 10:00"),  # 29-min gap
            pd.Timestamp("2026-01-01 10:01"),
            pd.Timestamp("2026-01-01 10:02"),
        ]
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000] * 5,
            },
            index=idx,
        )
        issues = validate_market_data(df, max_gap_minutes=10)
        assert any("gaps larger than 10 min" in i for i in issues)

    def test_no_gap_issue_when_under_limit(self):
        idx = [
            pd.Timestamp("2026-01-01 09:30"),
            pd.Timestamp("2026-01-01 09:35"),  # 5-min gap
            pd.Timestamp("2026-01-01 09:40"),
        ]
        df = pd.DataFrame(
            {
                "open": [100.0] * 3,
                "high": [101.0] * 3,
                "low": [99.0] * 3,
                "close": [100.5] * 3,
                "volume": [1000] * 3,
            },
            index=idx,
        )
        issues = validate_market_data(df, max_gap_minutes=10)
        # 5-min gaps within 10-min limit — no issue
        assert not any("gap" in i.lower() for i in issues)

    def test_allow_nan_suppresses_nan_warning(self):
        df = make_df({"close": [101.0, np.nan, 103.0, 102.0, 104.0]})
        issues = validate_market_data(df, allow_nan=True)
        assert not any("NaN" in i for i in issues)


class TestDetectOutliers:
    def test_no_outliers_in_normal_data(self):
        idx = pd.date_range("2026-01-01 09:30", periods=100, freq="1min")
        df = pd.DataFrame(
            {
                "open": 100.0 + np.random.randn(100) * 0.5,
                "high": 101.0 + np.random.randn(100) * 0.5,
                "low": 99.0 + np.random.randn(100) * 0.5,
                "close": 100.5 + np.random.randn(100) * 0.5,
                "volume": np.random.randint(5000, 15000, 100),
            },
            index=idx,
        )
        mask = detect_outliers(df)
        # With reasonable thresholds, almost no bars should be flagged
        assert mask.any(axis=1).sum() <= 5

    def test_flags_extreme_price_move(self):
        idx = pd.date_range("2026-01-01 09:30", periods=20, freq="1min")
        closes = [100.5] * 20
        closes[10] = 500.0  # Huge outlier
        df = pd.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [101.0] * 20,
                "low": [99.0] * 20,
                "close": closes,
                "volume": [1000] * 20,
            },
            index=idx,
        )
        mask = detect_outliers(df, price_zscore=3.0)
        assert mask.loc[mask.index[10], "close"]


class TestCleanMarketData:
    def test_removes_nan_rows(self):
        df = make_df({"close": [101.0, np.nan, 103.0, 102.0, 104.0]})
        cleaned = clean_market_data(df)
        assert len(cleaned) == 4
        assert not cleaned.isna().any().any()

    def test_swaps_high_low(self):
        df = make_df({
            "high": [102.0, 99.0, 104.0, 103.0, 105.0],
            "low": [99.0, 100.0, 101.0, 100.0, 102.0],
        })
        cleaned = clean_market_data(df)
        # Bar 1: high was 99, low was 100 → after swap high=100, low=99
        assert cleaned.loc[cleaned.index[1], "high"] == 100.0
        assert cleaned.loc[cleaned.index[1], "low"] == 99.0

    def test_clamps_open_close(self):
        df = make_df({
            "open": [200.0, 101.0, 102.0, 101.0, 103.0],
            "close": [50.0, 102.0, 103.0, 102.0, 104.0],
        })
        cleaned = clean_market_data(df)
        # Bar 0: open=200 should be clamped to high=102
        assert cleaned.loc[cleaned.index[0], "open"] == 102.0
        # Bar 0: close=50 should be clamped to low=99
        assert cleaned.loc[cleaned.index[0], "close"] == 99.0

    def test_removes_zero_volume(self):
        df = make_df({"volume": [1000, 0, 1200, 1100, 1300]})
        cleaned = clean_market_data(df)
        assert len(cleaned) == 4
        assert (cleaned["volume"] > 0).all()

    def test_normalizes_uppercase_columns(self):
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=pd.date_range("2026-01-01 09:30", periods=2, freq="1min"),
        )
        cleaned = clean_market_data(df)
        assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]
