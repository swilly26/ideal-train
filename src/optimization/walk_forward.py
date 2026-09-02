"""Walk-forward / IS-OOS-Holdout split machinery.

The single biggest failure mode in parameter optimisation is over-fitting to
the window the parameters were tuned on.  This module implements the
discipline that guards against it:

* a **chronological split** of any dated series into
  *in-sample (IS)* — the window the parameter search is allowed to see,
  *out-of-sample (OOS/validation)* — a later window used to grade each
  candidate and measure degradation, and an **untouched holdout** — the
  newest data, never used for search or selection, reported only at the end.
* an **anchored walk-forward** iterator that produces multiple
  (train, test) windows for rolling re-estimation.

Splits are made on **unique-day boundaries**, not arbitrary bar counts, so a
split never slices a trading day in half.  Positions in the returned split
are integer array indices into the original index, so callers can slice any
DataFrame/Series aligned to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


def unique_days(index: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> np.ndarray:
    """Return sorted unique calendar days (as ``datetime64``) for *index*.

    Used as the granularity for chronological splits so that a split
    boundary never falls inside a trading day.
    """
    days = pd.Series(pd.to_datetime(index)).dt.normalize()
    return days.unique()


def _day_positions(index: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> pd.Series:
    """Map each position in *index* to its normalized calendar day."""
    return pd.Series(pd.to_datetime(pd.Index(index))).dt.normalize().reset_index(drop=True)


@dataclass
class Split:
    """A chronological IS/OOS/holdout partition of a dated series.

    Attributes
    ----------
    is_idx : np.ndarray
        Integer positions belonging to the in-sample (training) window.
    oos_idx : np.ndarray
        Integer positions belonging to the out-of-sample (validation) window.
    holdout_idx : np.ndarray
        Integer positions belonging to the untouched holdout window.
    is_days / oos_days / holdout_days : list[str]
        ISO date strings of the unique trading days in each window.
    """

    is_idx: np.ndarray
    oos_idx: np.ndarray
    holdout_idx: np.ndarray
    is_days: list[str] = field(default_factory=list)
    oos_days: list[str] = field(default_factory=list)
    holdout_days: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def is_data(self, df: pd.DataFrame | pd.Series):
        """Return the slice of *df* belonging to the in-sample window."""
        return df.iloc[self.is_idx]

    def oos_data(self, df: pd.DataFrame | pd.Series):
        """Return the slice of *df* belonging to the OOS (validation) window."""
        return df.iloc[self.oos_idx]

    def holdout_data(self, df: pd.DataFrame | pd.Series):
        """Return the slice of *df* belonging to the holdout window."""
        return df.iloc[self.holdout_idx]

    @property
    def has_holdout(self) -> bool:
        return len(self.holdout_idx) > 0

    def summary(self) -> dict:
        """Compact summary dict for reports (counts of positions per window)."""
        return {
            "is": int(len(self.is_idx)),
            "oos": int(len(self.oos_idx)),
            "holdout": int(len(self.holdout_idx)),
            "is_days": len(self.is_days),
            "oos_days": len(self.oos_days),
            "holdout_days": len(self.holdout_days),
        }


def split_is_oos_holdout(
    index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    is_frac: float = 0.50,
    oos_frac: float = 0.25,
    min_days_per_window: int = 1,
) -> Split:
    """Split *index* chronologically into IS / OOS / holdout windows.

    The split is made on unique-day boundaries.  Days are sorted
    chronologically; the first ``round(n_days * is_frac)`` unique days form
    the in-sample window, the next ``round(n_days * oos_frac)`` form the
    out-of-sample (validation) window, and everything newer is the untouched
    holdout.

    Parameters
    ----------
    index : array-like of timestamps
        Anything ``pd.to_datetime`` accepts (DatetimeIndex, timestamps list).
    is_frac : float
        Fraction of unique trading days used for in-sample search.
    oos_frac : float
        Fraction of unique trading days used for out-of-sample validation.
    min_days_per_window : int
        Guard: raise if any window ends up with fewer than this many days.

    Returns
    -------
    Split
        Position indices (into *index*) plus day lists per window.

    Raises
    ------
    ValueError
        If the index has too few unique days to form all three windows, or
        if the fractions are not in ``(0, 1)``.
    """
    sorted_days = unique_days(index)
    n_days = len(sorted_days)
    if n_days == 0:
        raise ValueError("Cannot split an empty index")
    if not (0.0 < is_frac < 1.0 and 0.0 < oos_frac < 1.0 and is_frac + oos_frac < 1.0):
        raise ValueError(f"Invalid fractions is_frac={is_frac}, oos_frac={oos_frac}")

    is_n = max(1, int(round(n_days * is_frac)))
    oos_n = max(1, int(round(n_days * oos_frac)))
    # Never let the OOS window end beyond the last day.
    oos_n = min(oos_n, n_days - is_n - 1) if n_days - is_n > 1 else 0
    # The holdout must remain untouched; if we cannot leave at least
    # min_days_per_window days for it (and an OOS window, when requested),
    # refuse to fabricate a split from too-little data.
    if oos_n < 1 or (n_days - is_n - oos_n) < min_days_per_window:
        raise ValueError(
            f"Too few unique days ({n_days}) for is_frac={is_frac} "
            f"oos_frac={oos_frac}: need >= {is_n + oos_n + min_days_per_window} "
            "days for IS + OOS + holdout. Collect more history or lower the fractions."
        )

    is_cut = sorted_days[is_n - 1]
    oos_cut = sorted_days[is_n + oos_n - 1] if oos_n > 0 else is_cut

    day_pos = _day_positions(index)
    is_idx = np.where(day_pos <= is_cut)[0]
    if oos_n > 0:
        oos_idx = np.where((day_pos > is_cut) & (day_pos <= oos_cut))[0]
        holdout_idx = np.where(day_pos > oos_cut)[0]
    else:
        oos_idx = np.array([], dtype=int)
        holdout_idx = np.where(day_pos > is_cut)[0]

    def _day_list(positions: np.ndarray) -> list[str]:
        if len(positions) == 0:
            return []
        days = day_pos.iloc[positions].unique()
        return [pd.Timestamp(d).isoformat()[:10] for d in sorted(days.tolist())]

    return Split(
        is_idx=is_idx,
        oos_idx=oos_idx,
        holdout_idx=holdout_idx,
        is_days=_day_list(is_idx),
        oos_days=_day_list(oos_idx),
        holdout_days=_day_list(holdout_idx),
    )


@dataclass
class Window:
    """One (train, test) pair produced by :func:`walk_forward_windows`."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_days: list[str] = field(default_factory=list)
    test_days: list[str] = field(default_factory=list)

    def train_data(self, df: pd.DataFrame | pd.Series):
        return df.iloc[self.train_idx]

    def test_data(self, df: pd.DataFrame | pd.Series):
        return df.iloc[self.test_idx]


def walk_forward_windows(
    index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    n_windows: int = 3,
    train_frac: float = 0.5,
    min_test_days: int = 1,
) -> list[Window]:
    """Produce anchored walk-forward (train, test) windows.

    Each window uses a **growing** training set anchored at the start of the
    data (the standard anchored walk-forward scheme), with the next block of
    unseen days as the test set.  The last window ends at the final day.

    Parameters
    ----------
    index : array-like of timestamps
        Chronologically ordered (or sortable) dated index.
    n_windows : int
        Number of (train, test) pairs to produce.
    train_frac : float
        Fraction of unique days used as the initial training set.
    min_test_days : int
        Minimum unique days required in each test set.

    Returns
    -------
    list[Window]
    """
    sorted_days = unique_days(index)
    n_days = len(sorted_days)
    if n_days < n_windows + 1:
        raise ValueError(
            f"Need at least {n_windows + 1} unique days for {n_windows} walk-forward "
            f"windows, got {n_days}."
        )
    day_pos = _day_positions(index)

    train_n = max(1, int(round(n_days * train_frac)))
    # Distribute the remaining days as evenly as possible.
    test_n = (n_days - train_n) // n_windows
    if test_n < min_test_days:
        raise ValueError(
            f"Remaining days ({n_days - train_n}) cannot form {n_windows} test sets "
            f"of >= {min_test_days} days each."
        )

    windows: list[Window] = []
    for w in range(n_windows):
        train_end = train_n + w * test_n
        test_end = train_end + test_n
        train_days = sorted_days[:train_end]
        test_days = sorted_days[train_end:test_end]

        train_pos = np.where(day_pos <= train_days[-1])[0]
        test_pos = np.where(
            (day_pos > train_days[-1]) & (day_pos <= test_days[-1])
        )[0]

        def _dl(positions: np.ndarray) -> list[str]:
            return [pd.Timestamp(d).isoformat()[:10] for d in np.unique(day_pos.iloc[positions])]

        windows.append(
            Window(
                train_idx=train_pos,
                test_idx=test_pos,
                train_days=_dl(train_pos),
                test_days=_dl(test_pos),
            )
        )
    return windows