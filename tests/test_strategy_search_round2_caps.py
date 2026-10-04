"""The round-2 cap re-pin (brief §R3-11.1): four slots at $25k, not two at $50k.

Source: ``/home/team/shared/ROUND2_BUILDER_AUDIT.md`` §6.1 (ranked finding 1,
BLOCKING) and ``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-11.1, which governs.

**The defect these tests are written against.**  ``R2_UNIVERSE`` is also the
instrument insertion order and the cap is applied in that order.  ``gap`` is
session-constant, so in family E every symbol signals on the **same bar**: with
``max_positions=2``, SOXL and TQQQ always took both slots and SPXL and SPY could
never trade.  Every E number would have been a two-symbol, two-correlated-3x-
leveraged-ETF number, and any neighbour/robustness evidence would have been about
a grid that was never the declared grid.

Every test in this file fails on the pre-fix commit ``4a6347d`` for the reason
its own docstring names; the recorded run against a scratch worktree at that
commit is ``docs/ROUND2_CAPREPIN_PREFIX_PYTEST.txt`` with its header in
``docs/ROUND2_CAPREPIN_PREFIX_EVIDENCE.md``.

No P&L is read here: the run uses ``CostModel.baseline()`` only to open and
flatten the book, and only position counts, symbols and gross notional are
asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import families  # noqa: E402
from src.backtesting.strategy_search.engine import run_search  # noqa: E402
from tests.test_strategy_search_round2_families import (  # noqa: E402
    D2,
    cell,
    entry_indices,
    flat_bars,
    frames_2day,
    market_of,
)

#: A 0.05 % gap on a flat prior session: ``gap_atr = 1.25`` (see the E-family
#: tests), so the ``gap_threshold = 0.5`` cell fires on **all four** symbols on
#: the same bar — the exact shape that made the cap decide the strategy.
GAP_UP = 100.0 * 1.0005
SIGNAL_MIN = 9 * 60 + 31
FILL_MIN = 9 * 60 + 32


def pinned_caps_cell(family: str = "E1", index: int = 0):
    """One grid cell built at the pinned caps, with its instruments."""
    frames = frames_2day(flat_bars(GAP_UP))
    m, aligned = market_of(frames)
    cfg, instruments, resolved = families.build(family, cell(family, index),
                                                m, aligned)
    return frames, m, aligned, cfg, instruments, resolved


def test_all_four_universe_symbols_open_at_the_pinned_caps():
    """Four slots means SPXL and SPY are reachable; four symbols, gross 100k.

    Pre-fix this showed exactly two positions (SOXL, TQQQ) and SPXL/SPY absent:
    ``skipped["max_positions"] == 2``.
    """
    _frames, m, _aligned, cfg, instruments, _resolved = pinned_caps_cell()
    for sym in families.R2_UNIVERSE:
        idx = entry_indices(instruments[sym])
        assert len(idx) == 1, f"{sym}: {len(idx)} signals, expected one"
        assert int(m.minute[idx[0]]) == SIGNAL_MIN, \
            f"{sym}: the gap signal is on the session's first in-window bar"

    res = run_search(m, instruments, cfg, CostModel.baseline())
    traded = list(res.trades["instrument"])
    assert traded == list(families.R2_UNIVERSE), (
        "four symbols signal on the same bar, so the pinned cap must open all "
        f"four, in the declared universe order — got {traded}")
    assert "SPXL" in traded and "SPY" in traded, \
        "the two symbols the pre-fix cap of 2 made unreachable"
    assert res.stats["skipped"]["max_positions"] == 0, \
        "at cap = |universe| the cap can never bind"
    assert res.stats["entries"] == 4
    for _, tr in res.trades.iterrows():
        assert tr["entry_time"] == pd.Timestamp(D2) + pd.Timedelta(minutes=FILL_MIN)
        assert tr["exit_reason"] == "eod"


def test_the_pinned_caps_are_asserted_by_value():
    """A silent re-pin must fail a test, not silently change a whole family."""
    assert families.R2_CAPS["max_positions"] == 4
    assert families.R2_CAPS["notional_usd"] == 25_000.0
    assert families.R2_CAPS["initial_equity"] == 100_000.0
    assert (4 * families.R2_CAPS["notional_usd"]
            == families.R2_CAPS["initial_equity"]), \
        "four slots at the pinned notional are exactly the equity ceiling"

    for family in families.ROUND2_FAMILIES:
        for spec in families.GRIDS[family]:
            p = spec["params"]
            assert p["max_positions"] == 4, f"{family}/{spec['name']}"
            assert p["notional_usd"] == 25_000.0, f"{family}/{spec['name']}"

    # and the value the engine actually runs with, through build()
    _frames, _m, _a, cfg, _insts, _r = pinned_caps_cell()
    assert cfg.max_positions == 4
    assert cfg.notional_usd == 25_000.0
    assert families.R2_CAPS == {"initial_equity": 100_000.0,
                                "notional_usd": 25_000.0,
                                "max_positions": 4,
                                "max_entries_per_session": 1,
                                "min_minutes_between_entries": 0}


def test_the_four_symbol_book_never_exceeds_the_equity_ceiling():
    """Every symbol signals at once and the book sits **at** the ceiling, never
    above it: 4 x 25k = 100k = ``initial_equity``, the declared 100 % limit.

    Pre-fix the book was also 100k gross, but as 2 x 50k — so this test fails on
    the per-position notional (``gross_notional == 50_000``), which is the pinned
    literal the re-pin changes.  The engine exposes no per-bar book-gross series:
    the peak is the all-four-open bar, which is the entry bar of all four trips.
    """
    _frames, m, _aligned, cfg, instruments, _r = pinned_caps_cell()
    res = run_search(m, instruments, cfg, CostModel.baseline())
    assert len(res.trades) == 4
    per_trip = res.trades["gross_notional"].to_numpy(dtype=float)
    assert np.allclose(per_trip, 25_000.0), \
        f"each position is the pinned 25k notional, got {per_trip}"
    book_gross = float(per_trip.sum())
    ceiling = float(families.R2_CAPS["initial_equity"])
    assert book_gross == pytest.approx(ceiling), \
        "the four-symbol book is exactly at the 100 % equity ceiling"
    assert book_gross <= ceiling
    assert res.stats["skipped"]["gross_leverage"] == 0, \
        "no basket is refused by the gross-leverage gate at the pinned caps"
    assert res.stats["gross_limit_usd"] == ceiling
