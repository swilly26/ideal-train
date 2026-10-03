"""Future-blindness harness (lead decisions §E4, §R3-7).

A family's entry/exit arrays at bar *t* may depend only on bars at or before
*t*.  A rolling feature that is normalised with the *whole session's* std or
range, or a gate set at a whole-window quantile, breaks that — and it breaks it
silently, because the arrays still look plausible.

The harness rebuilds the whole pipeline (features → family) on a copy of the
data in which **every bar after a cut is replaced**, once by garbage and once by
its own mirror, and asserts the entry/exit arrays are byte-for-byte unchanged at
and before the cut.  Any dependence on a later bar — through a whole-session
aggregate, a re-indexed future value, or a reversed slice — shows up immediately.

**A single cut misses the failure mode** (R3-7 E4).  :func:`dense_cuts` returns
the **first and last bar of every calendar month, plus the first two bars of
every session** (never 09:30 — the open bar's own close *is* the session open, so
a cut there tests nothing about reading forward), and
:func:`assert_blind_to_the_future_dense` runs every one of them.  On top of that
goes the mechanical invariant

    ``build_features(df.iloc[:t]) == build_features(df).iloc[:t]``

for every sampled *t*: that is what makes "causal/expanding only" **checkable**
rather than promised.  The one documented exception is the opening-range columns
(see ``WHOLE_WINDOW_COLS``): their value inside the OR window really does depend
on the session's own later bars, which is why a family may only read them behind
a ``ready`` gate — and why the *signal* comparison still covers them in full.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PRICE_COLS = ("open", "high", "low", "close")
#: Opening-range columns are the one *declared* whole-window aggregate: their
#: value on every bar of the session is only final at the end of the OR window
#: (10:00 / 10:30).  A family may only read them behind a ``ready`` gate, which
#: is exactly what the entry/exit comparison below checks — so they are excluded
#: from the column-by-column feature comparison and never from the signal check.
WHOLE_WINDOW_COLS = ("or30_hi", "or30_lo", "or60_hi", "or60_lo")
#: A cut exactly at the open is not a cut: the 09:30 bar's close *is* the session
#: open, so replacing everything after it cannot change what the open told us.
OPEN_MIN = 9 * 60 + 30


def mutate_after(frames: Mapping[str, pd.DataFrame], cut: pd.Timestamp,
                 mode: str) -> dict[str, pd.DataFrame]:
    """Copy *frames* with every bar after *cut* rebuilt (``garbage``/``mirror``)."""
    if mode not in ("garbage", "mirror"):
        raise ValueError(f"unknown mutation mode {mode!r}")
    out: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        d = df.copy()
        after = d.index > cut
        if not after.any():
            out[sym] = d
            continue
        if mode == "garbage":
            for col in PRICE_COLS:
                d.loc[after, col] = d.loc[after, col].to_numpy() * 1000.0 + 500.0
            d.loc[after, "volume"] = (d.loc[after, "volume"].to_numpy() * 977.0
                                     + 1.0)
        else:
            cut_close = float(df.loc[df.index <= cut, "close"].iloc[-1])
            o = d.loc[after, "open"].to_numpy(dtype=float)
            h = d.loc[after, "high"].to_numpy(dtype=float)
            low = d.loc[after, "low"].to_numpy(dtype=float)
            c = d.loc[after, "close"].to_numpy(dtype=float)
            d.loc[after, "open"] = 2.0 * cut_close - o
            d.loc[after, "high"] = 2.0 * cut_close - low
            d.loc[after, "low"] = 2.0 * cut_close - h
            d.loc[after, "close"] = 2.0 * cut_close - c
            d.loc[after, "volume"] = d.loc[after, "volume"].to_numpy()[::-1]
        out[sym] = d
    return out


def signal_arrays(family: str, spec: dict, frames: Mapping[str, pd.DataFrame]):
    """Run the whole pipeline and return ``(market, aligned, arrays)``."""
    from src.backtesting.strategy_search import families as fam
    from src.backtesting.strategy_search.engine import Market
    from src.backtesting.strategy_search.features import FeatureBook

    market = Market(frames)
    book = FeatureBook(frames)
    aligned = book.align(market.axis)
    _cfg, instruments, _resolved = fam.build(family, spec, market, aligned)
    arrays = {k: (inst.entry_dir.copy(), inst.exit_now.copy())
              for k, inst in instruments.items()}
    return market, aligned, arrays


def assert_blind_to_the_future(family: str, spec: dict,
                               frames: Mapping[str, pd.DataFrame],
                               cut: pd.Timestamp) -> dict:
    """Assert the family reads no bar after *cut*; returns the cut index."""
    market, aligned, base = signal_arrays(family, spec, frames)
    cut_index = int(market.axis.get_loc(cut))
    for mode in ("garbage", "mirror"):
        m2, aligned2, arrays = signal_arrays(
            family, spec, mutate_after(frames, cut, mode))
        assert m2.axis.equals(market.axis), f"{mode}: the bar axis moved"
        for key, (e0, x0) in base.items():
            e1, x1 = arrays[key]
            assert np.array_equal(e1[:cut_index + 1], e0[:cut_index + 1]), (
                f"{family}/{key}: entry_dir changed at or before the cut when "
                f"every later bar was replaced ({mode}) — the family reads the "
                f"future")
            assert np.array_equal(x1[:cut_index + 1], x0[:cut_index + 1]), (
                f"{family}/{key}: exit_now changed at or before the cut when "
                f"every later bar was replaced ({mode}) — the family reads the "
                f"future")
        for sym, frame in aligned.items():
            other = aligned2[sym]
            for col in frame.columns:
                if col in WHOLE_WINDOW_COLS:
                    continue
                a = frame[col].to_numpy(dtype=float)[:cut_index + 1]
                b = other[col].to_numpy(dtype=float)[:cut_index + 1]
                assert np.array_equal(a, b, equal_nan=True), (
                    f"{family}/{sym}/{col}: feature changed at or before the "
                    f"cut when every later bar was replaced ({mode})")
    return {"cut_index": cut_index, "keys": sorted(base)}


# ── the dense cut set and the causal-features invariant (R3-7 E4) ──────


def dense_cuts(axis: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """The dense cut set: month edges plus every session's first two bars.

    * the **first and last bar of every calendar month** — a leak that only
      appears across a month boundary (a whole-window normaliser, a quantile
      gate) has to be sampled there;
    * the **first two bars of every session**, never at 09:30 — the open bar's
      own close *is* the session open, so a cut there tests nothing, while
      09:31 is the bar immediately after the open, which is where a
      whole-session normaliser or an ungated opening-range read shows up.

    Deduped and sorted, so a caller can cut on each element once.
    """
    axis = pd.DatetimeIndex(axis)
    if not len(axis):
        return []
    minute_of = np.array([int(t.hour) * 60 + int(t.minute) for t in axis])
    pos = pd.Series(np.arange(len(axis)), index=axis)
    month = pd.Series(axis, index=axis).dt.strftime("%Y-%m")
    cuts: dict[pd.Timestamp, None] = {}

    def add(p: int) -> None:
        # the 09:30 exclusion applies to every source: a cut at the open bar
        # cannot change what the open told us, so it tests nothing.
        if minute_of[p] != OPEN_MIN:
            cuts[axis[p]] = None

    for _, grp in pos.groupby(month.to_numpy()):
        add(int(grp.iloc[0]))
        add(int(grp.iloc[-1]))
    day = pd.Series(axis.normalize(), index=axis)
    for _, grp in pos.groupby(day.to_numpy()):
        for p in np.asarray(grp.iloc[:2]).tolist():
            add(int(p))
    return sorted(cuts)


def assert_features_causal(frames: Mapping[str, pd.DataFrame],
                           cuts: Sequence[pd.Timestamp]) -> dict:
    """``build_features(df.iloc[:t]) == build_features(df).iloc[:t]`` at every cut.

    This is the mechanical form of "causal/expanding only": a column computed
    from the whole session (or the whole frame) disagrees with the same column
    rebuilt on the bars up to *t*.  The declared opening-range columns are the
    one exception and are named in ``WHOLE_WINDOW_COLS`` with the reason.
    """
    from src.backtesting.strategy_search.features import build_features

    checked = 0
    for sym, df in frames.items():
        full = build_features(df)
        cols = [c for c in full.columns if c not in WHOLE_WINDOW_COLS]
        for cut in cuts:
            t = int(df.index.searchsorted(cut, side="right"))
            if t <= 0 or t >= len(df):
                continue                      # nothing before the cut to check
            prefix = build_features(df.iloc[:t])
            for col in cols:
                a = full[col].to_numpy(dtype=float)[:t]
                b = prefix[col].to_numpy(dtype=float)
                assert np.array_equal(a, b, equal_nan=True), (
                    f"{sym}/{col}: build_features(df.iloc[:{t}]) differs from "
                    f"build_features(df).iloc[:{t}] at cut {cut} — a feature is "
                    f"computed from bars after the cut (a whole-session "
                    f"aggregate, a reversed slice, or a forward fill)")
            checked += 1
    return {"checked": checked}


def assert_blind_to_the_future_dense(family: str, spec: dict,
                                     frames: Mapping[str, pd.DataFrame],
                                     feature_invariant: bool = True) -> dict:
    """Every dense cut, both mutation modes, plus the features invariant."""
    axis = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    cuts = dense_cuts(axis)
    assert cuts, "the dense cut set is empty: nothing was tested"
    keys: list[str] = []
    for cut in cuts:
        out = assert_blind_to_the_future(family, spec, frames, cut)
        keys = out["keys"]
    inv = assert_features_causal(frames, cuts) if feature_invariant else {}
    return {"cuts": len(cuts), "cut_times": [str(c) for c in cuts], "keys": keys,
            "feature_rows_checked": inv.get("checked", 0)}
