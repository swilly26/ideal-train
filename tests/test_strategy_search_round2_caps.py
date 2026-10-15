"""The round-2 cap re-pins: four slots, and headroom for the fourth fill's costs.

Source: ``/home/team/shared/ROUND2_BUILDER_AUDIT.md`` §6.1 (ranked finding 1,
BLOCKING) and ``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-11.1 for the first re-pin
(2 x 50k -> 4 x 25k), and ``/home/team/shared/ROUND2_TAPE_POLICY_RULING.md``
**L17** for the second (25k -> **24k**, so 4 x 24k = 96k of the 100k equity
leaves 4k of headroom for the fourth fill's own costs).  The first re-pin made
SPXL/SPY reachable; the second removes the **mechanical** funding refusal that
4 x 25k = exactly 100k produced at every full-cash moment.

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

import dataclasses
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
    """A silent re-pin must fail a test, not silently change a whole family.

    L17 pins the **24 000 / 96 000 / 4 000** relationship, not the old
    "4 x 25k is exactly the equity ceiling": the headroom is the whole point.
    """
    assert families.R2_CAPS["max_positions"] == 4
    assert families.R2_CAPS["notional_usd"] == 24_000.0
    assert families.R2_CAPS["initial_equity"] == 100_000.0
    # L17 — the fourth fill's own costs must fit the residual cash, so the
    # deployed notional is strictly *below* the equity and the headroom is the
    # declared 4 000 (4 % of equity), never 0.
    assert 4 * families.R2_CAPS["notional_usd"] < families.R2_CAPS["initial_equity"], \
        "4 x notional must leave headroom for the fourth fill's costs (L17)"
    assert (families.R2_CAPS["initial_equity"]
            - 4 * families.R2_CAPS["notional_usd"]) == 4_000.0, \
        "the L17 headroom is the pinned 4 000"

    for family in families.ROUND2_FAMILIES:
        for spec in families.GRIDS[family]:
            p = spec["params"]
            assert p["max_positions"] == 4, f"{family}/{spec['name']}"
            assert p["notional_usd"] == 24_000.0, f"{family}/{spec['name']}"

    # and the value the engine actually runs with, through build()
    _frames, _m, _a, cfg, _insts, _r = pinned_caps_cell()
    assert cfg.max_positions == 4
    assert cfg.notional_usd == 24_000.0
    assert families.R2_CAPS == {"initial_equity": 100_000.0,
                                "notional_usd": 24_000.0,
                                "max_positions": 4,
                                "max_entries_per_session": 1,
                                "min_minutes_between_entries": 0}


def test_the_four_symbol_book_stays_under_the_equity_ceiling():
    """Every symbol signals at once and the book sits **below** the ceiling:
    4 x 24k = 96k of the 100k equity, leaving the pinned 4k headroom (L17).

    The first re-pin moved the per-position notional 50k -> 25k (so this test
    failed on ``gross_notional == 50_000``); L17 moves it 25k -> 24k, so the
    book is no longer *exactly* at the ceiling — the headroom is the point, and
    a re-pin back to 25k fails here as well as in the by-value test above.  The
    engine exposes no per-bar book-gross series: the peak is the all-four-open
    bar, which is the entry bar of all four trips.
    """
    _frames, m, _aligned, cfg, instruments, _r = pinned_caps_cell()
    res = run_search(m, instruments, cfg, CostModel.baseline())
    assert len(res.trades) == 4
    per_trip = res.trades["gross_notional"].to_numpy(dtype=float)
    assert np.allclose(per_trip, 24_000.0), \
        f"each position is the pinned 24k notional, got {per_trip}"
    book_gross = float(per_trip.sum())
    ceiling = float(families.R2_CAPS["initial_equity"])
    assert book_gross == pytest.approx(4 * families.R2_CAPS["notional_usd"])
    assert book_gross < ceiling, \
        "the four-symbol book must leave headroom under the equity ceiling (L17)"
    assert ceiling - book_gross == pytest.approx(4_000.0), \
        "the headroom is the pinned 4 000"
    assert res.stats["skipped"]["gross_leverage"] == 0, \
        "no basket is refused by the gross-leverage gate at the pinned caps"
    assert res.stats["gross_limit_usd"] == ceiling


def test_the_re_pin_lets_the_fourth_fill_pay_its_own_costs():
    """L17, proved by measurement on the same four-symbol book.

    The only cost level that charges a commission is ``pessimistic``
    ($0.005/share, ``replay_costs.py``).  At the L17 caps all four symbols open
    and **no** cash refusal fires.  At the old literal, on the identical book and
    cost model, the fourth symbol's outlay (24/25k of notional **plus** its fee)
    exceeds the residual cash and the engine refuses it — which is the
    mechanical full-cash refusal L17 removes, and why the re-pin cannot be
    reverted without failing this test.
    """
    _frames, m, _aligned, cfg, instruments, _r = pinned_caps_cell()
    costs = CostModel.pessimistic()
    assert costs.commission_per_share > 0, "this test needs a fee-charging level"

    res = run_search(m, instruments, cfg, costs)
    assert res.stats["entries"] == 4, "the fourth fill must still open at L17"
    assert res.stats["skipped"]["cash"] == 0, \
        "the mechanical full-cash refusal must be gone at the L17 caps"
    assert res.stats["cash_refusals_cost_only"] == 0
    assert res.stats["skipped"]["max_positions"] == 0

    old = dataclasses.replace(cfg, notional_usd=25_000.0)
    res_old = run_search(m, instruments, old, costs)
    assert res_old.stats["skipped"]["cash"] >= 1, \
        "the old 25k literal must refuse the fourth fill — that is the L17 defect"
    assert res_old.stats["entries"] == 3
    # The old refusal fired with the book at full deployment, not after a loss:
    # the residual cash (100k − 3×25k − fees) was still **above** the L17
    # notional, so the only thing that refused the fourth fill was the cost
    # boundary — which is exactly the mechanical refusal L17 removes.
    old_refusal = res_old.stats["cash_refusals_detail"][0]
    assert old_refusal["cash"] >= 24_000.0, (
        "the old refusal was not drawdown-driven: cash still covered the L17 "
        f"notional — {old_refusal}")
    assert old_refusal["cash_covered_notional"] is False, \
        "at 25k the residual cash sits inside the fee band below the notional"
