"""Tests for the walk-forward / IS-OOS-Holdout split machinery."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.walk_forward import (
    Split,
    split_is_oos_holdout,
    unique_days,
    walk_forward_windows,
)


def _index(days: int = 20, bars_per_day: int = 12) -> pd.DatetimeIndex:
    """Synthetic intraday index: *days* unique dates × *bars_per_day* bars."""
    start = pd.Timestamp("2026-07-01 09:30")
    parts = []
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        parts.append(day + pd.timedelta_range(0, periods=bars_per_day, freq="5min"))
    return pd.DatetimeIndex(np.concatenate(parts))


def test_unique_days_dedupes_and_sorts():
    idx = _index(5, 12)
    days = unique_days(idx)
    assert len(days) == 5
    assert days[0] == pd.Timestamp("2026-07-01").normalize()
    assert list(days) == sorted(days)


def test_split_partitions_all_positions():
    idx = _index(20, 12)
    split = split_is_oos_holdout(idx, is_frac=0.5, oos_frac=0.25)

    all_positions = np.concatenate([split.is_idx, split.oos_idx, split.holdout_idx])
    assert sorted(all_positions) == list(range(len(idx)))
    # Windows are chronological, no overlap.
    assert split.is_idx.max() < split.oos_idx.min() or len(split.oos_idx) == 0
    assert split.oos_idx.max() < split.holdout_idx.min() or len(split.holdout_idx) == 0


def test_split_respects_day_boundaries():
    idx = _index(10, 12)
    split = split_is_oos_holdout(idx, is_frac=0.5, oos_frac=0.25)
    # IS = 5 days, OOS = 2.5→3 days (round), holdout = remaining.
    assert len(split.is_days) == 5
    assert sorted(set(split.is_days)) == split.is_days
    # A window boundary never splits a day: last IS bar and first OOS bar
    # belong to different normalized days.
    is_last_ts = pd.Timestamp(idx[split.is_idx.max()]).normalize()
    oos_first_ts = pd.Timestamp(idx[split.oos_idx.min()]).normalize()
    assert is_last_ts < oos_first_ts


def test_split_slicing_helpers():
    idx = _index(6, 8)
    df = pd.DataFrame({"close": np.arange(len(idx))}, index=idx)
    split = split_is_oos_holdout(idx, is_frac=0.5, oos_frac=0.25)
    assert len(split.is_data(df)) == len(split.is_idx)
    assert len(split.oos_data(df)) == len(split.oos_idx)
    assert len(split.holdout_data(df)) == len(split.holdout_idx)
    # All points accounted for exactly once (compare normalized timestamps).
    slices = [split.is_data(df), split.oos_data(df), split.holdout_data(df)]
    recovered = sorted(ts.normalize() for s in slices for ts in s.index)
    original = sorted([ts.normalize() for ts in idx])
    assert recovered == original
    # And the row values match the original frame exactly.
    assert df.loc[split.is_data(df).index, "close"].tolist() == \
        df["close"].iloc[split.is_idx].tolist()


def test_split_too_few_days_raises():
    idx = _index(3, 5)
    with pytest.raises(ValueError, match="Too few unique days"):
        split_is_oos_holdout(idx, is_frac=0.5, oos_frac=0.25)


def test_split_invalid_fractions_raise():
    idx = _index(10, 5)
    with pytest.raises(ValueError, match="Invalid fractions"):
        split_is_oos_holdout(idx, is_frac=0.0, oos_frac=0.25)
    with pytest.raises(ValueError, match="Invalid fractions"):
        split_is_oos_holdout(idx, is_frac=0.5, oos_frac=0.6)


def test_split_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        split_is_oos_holdout(pd.DatetimeIndex([]))


def test_walk_forward_windows_growing_train():
    idx = _index(12, 8)
    windows = walk_forward_windows(idx, n_windows=3, train_frac=0.5)
    assert len(windows) == 3
    # Training sets grow monotonically.
    assert len(windows[0].train_idx) < len(windows[1].train_idx) < len(windows[2].train_idx)
    # Test sets are disjoint and after their own train set.
    for w in windows:
        assert w.train_idx.max() < w.test_idx.min()
    assert windows[0].test_idx.max() < windows[1].test_idx.min()
    assert len(windows[0].test_days) > 0


def test_walk_forward_too_few_days_raises():
    idx = _index(4, 8)
    with pytest.raises(ValueError, match="cannot form"):
        walk_forward_windows(idx, n_windows=3, train_frac=0.5)


def test_split_summary_dict():
    idx = _index(10, 8)
    split = split_is_oos_holdout(idx, is_frac=0.5, oos_frac=0.25)
    s = split.summary()
    assert s["is"] == len(split.is_idx)
    assert s["holdout"] == len(split.holdout_idx)
    assert Split.is_data  # dataclass with methods is constructible