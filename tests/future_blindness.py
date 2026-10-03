"""Future-blindness harness (lead decisions §E4).

A family's entry/exit arrays at bar *t* may depend only on bars at or before
*t*.  A rolling feature that is normalised with the *whole session's* std or
range, or a gate set at a whole-window quantile, breaks that — and it breaks it
silently, because the arrays still look plausible.

The harness rebuilds the whole pipeline (features → family) on a copy of the
data in which **every bar after a cut is replaced**, once by garbage and once by
its own mirror, and asserts the entry/exit arrays are byte-for-byte unchanged at
and before the cut.  Any dependence on a later bar — through a whole-session
aggregate, a re-indexed future value, or a reversed slice — shows up immediately.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

PRICE_COLS = ("open", "high", "low", "close")
#: Opening-range columns are the one *declared* whole-window aggregate: their
#: value on every bar of the session is only final at the end of the OR window
#: (10:00 / 10:30).  A family may only read them behind a ``ready`` gate, which
#: is exactly what the entry/exit comparison below checks — so they are excluded
#: from the column-by-column feature comparison and never from the signal check.
WHOLE_WINDOW_COLS = ("or30_hi", "or30_lo", "or60_hi", "or60_lo")


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
