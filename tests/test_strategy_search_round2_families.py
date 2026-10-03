"""The four round-2 family builders: E1/E2 (gap), G (opening-range reversal), H (VWAP).

Each family is checked against the **declared** rule, recomputed independently in
this file from the raw bars — that is the round-1 lesson applied to a builder: a
declared parameter the engine silently does not execute is the failure class that
has cost this team twice (the ``z_window`` the artefact never recorded; the
``exit_now`` comparison that never ran).

Pinned by these tests, per family:

* a **full 3x3 factorial** — 9 cells, exactly two axes, 3 levels each, so every
  cell has exactly two neighbours per axis and the neighbour rule is mechanical
  (R3-1);
* the **family literals** — universe, caps, entry window, ``eod_flat_min = 15:30``,
  ``allow_short = True``, one signal per session (R3-6 / R3-7);
* the **entry rule itself**, recomputed from ``squeeze``-free raw features;
* the **causal normaliser** (``atr_prev_pct`` / ``atr_prev_abs``: the prior
  completed session's ATR, never today's);
* **future-blindness at every dense cut** for all four families.

No P&L, no screen: this file builds arrays only.
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
from src.backtesting.strategy_search import families, screen  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    EOD_PINNED_MIN,
    Market,
    run_search,
)
from src.backtesting.strategy_search.features import FeatureBook  # noqa: E402
from tests.future_blindness import assert_blind_to_the_future_dense  # noqa: E402
from tests.test_strategy_search_engine import mk_frame  # noqa: E402

OPEN = 9 * 60 + 30
D1, D2 = "2025-01-02", "2025-01-03"
#: A flat day of 1-minute bars at *px* with a ±0.02 range: TR = 0.04, so day 2's
#: ``atr_prev_abs`` is exactly 0.04 and ``atr_prev_pct`` is 0.04 / px.
RANGE = 0.02


def flat_bars(px: float, n: int = 390, rng: float = RANGE) -> list:
    return [(px, px + rng, px - rng, px) for _ in range(n)]


def frames_2day(day2: list, day1: list | None = None,
                symbols=families.R2_UNIVERSE, n: int = 390) -> dict:
    """Day 1 flat at 100 (the normaliser) plus whatever *day2* declares."""
    d1 = flat_bars(100.0, n) if day1 is None else day1
    return {s: mk_frame({D1: list(d1), D2: list(day2)}) for s in symbols}


def market_of(frames) -> tuple[Market, dict]:
    m = Market(frames)
    return m, FeatureBook(frames).align(m.axis)


def build_family(family: str, spec: dict, frames):
    m, aligned = market_of(frames)
    cfg, instruments, resolved = families.build(family, spec, m, aligned)
    return m, aligned, cfg, instruments, resolved


def cell(family: str, index: int, symbols=None) -> dict:
    spec = dict(families.GRIDS[family][index])
    spec["params"] = dict(spec["params"])
    spec["signal"] = dict(spec["signal"])
    spec["extras"] = dict(spec.get("extras", {}))
    if symbols is not None:
        spec["symbols"] = list(symbols)
    return spec


def declared_first_per_session(m: Market, mask: np.ndarray,
                               start_min: int, end_min: int) -> np.ndarray:
    """Independent re-implementation of "the first qualifying bar of a session"."""
    out = np.zeros(m.n(), dtype=np.int8)
    in_window = (m.minute >= start_min) & (m.minute <= end_min)
    mask = np.asarray(mask, dtype=bool) & in_window
    session = 0
    for k in range(m.n()):
        if m.day_change[k]:
            session += 1
            found = False
        if mask[k] and not found:
            out[k] = 1
            found = True
    return out


def entry_indices(inst) -> list[int]:
    return [int(i) for i in np.flatnonzero(inst.entry_dir)]


# ── the grids: full factorials, pinned literals ────────────────────────


def test_the_four_round2_groups_are_nine_full_factorial_cells_each():
    assert families.ROUND2_FAMILIES == ("E1", "E2", "G", "H")
    total = 0
    for family in families.ROUND2_FAMILIES:
        grid = families.GRIDS[family]
        assert len(grid) == 9, f"{family}: {len(grid)} cells, expected 9"
        axes = screen.declared_axes(family, grid)
        assert len(axes) == 2, f"{family} declares {sorted(axes)} — not two axes"
        for axis, values in axes.items():
            assert len(values) == 3, f"{family}/{axis}: {len(values)} levels"
        total += len(grid)
    assert total == 36, "N = 4 groups x 9 cells"


def test_the_neighbour_rule_is_mechanical_for_every_round2_cell():
    """Every cell has exactly two neighbours per axis (R3-1)."""
    for family in families.ROUND2_FAMILIES:
        grid = families.GRIDS[family]
        for spec in grid:
            nb = screen.neighbour_cells(family, grid, spec["name"])
            assert len(nb) == 2, f"{family}/{spec['name']} axes: {sorted(nb)}"
            for axis, cells in nb.items():
                assert len(cells) == 2, \
                    f"{family}/{spec['name']}/{axis}: {cells} — need exactly 2"


def test_every_round2_cell_declares_the_pinned_literals():
    for family in families.ROUND2_FAMILIES:
        start, end = families.R2_ENTRY_WINDOWS[family]
        for spec in families.GRIDS[family]:
            p = spec["params"]
            assert spec["symbols"] == list(families.R2_UNIVERSE)
            assert "FNGU" not in spec["symbols"], "FNGU has no bars before 2025-02"
            assert p["allow_short"] is True
            assert p["eod_flat_min"] == EOD_PINNED_MIN == 15 * 60 + 30
            assert (p["entry_start_min"], p["entry_end_min"]) == (start, end)
            assert p["entry_end_min"] <= p["eod_flat_min"] - 2, "zero-hold guard"
            assert p["initial_equity"] == families.R2_CAPS["initial_equity"]
            assert p["notional_usd"] == families.R2_CAPS["notional_usd"]
            assert p["max_positions"] == families.R2_CAPS["max_positions"] == 2
            assert p["max_entries_per_session"] == 1
            assert p["min_minutes_between_entries"] == 0
            assert p["stop_pct"] is None and p["target_pct"] is None
            assert p["trail_pct"] is None
            assert spec["extras"] == {}


def test_all_thirty_six_cells_build_and_resolve():
    frames = _blind_frames()
    m, aligned = market_of(frames)
    seen = set()
    for family in families.ROUND2_FAMILIES:
        for spec in families.GRIDS[family]:
            cfg, instruments, resolved = families.build(family, spec, m, aligned)
            assert sorted(instruments) == sorted(families.R2_UNIVERSE)
            assert len({id(i) for i in instruments.values()}) == 4
            assert resolved["signal"], "the resolved signal dict may not be empty"
            seen.add(cfg.name)
            assert cfg.name == families.resolved_name(family, resolved)
            assert cfg.max_positions == 2
    assert len(seen) == 36, "every cell resolves to its own config name"


@pytest.mark.parametrize("family,bad", [
    ("E1", dict(params_over={"allow_short": False})),
    ("E2", dict(params_over={"allow_short": False})),
    ("G", dict(params_over={"allow_short": False})),
    ("H", dict(params_over={"allow_short": False})),
    ("E1", dict(params_over={"entry_start_min": 10 * 60})),
    ("G", dict(params_over={"entry_end_min": 15 * 60})),
    ("H", dict(params_over={"eod_flat_min": 15 * 60 + 25})),
    ("E1", dict(params_over={"max_positions": 4})),
    ("G", dict(params_over={"notional_usd": 25_000.0})),
    ("H", dict(params_over={"max_entries_per_session": 2})),
])
def test_a_cell_off_the_pinned_literals_is_refused(family, bad):
    frames = _blind_frames()
    m, aligned = market_of(frames)
    spec = cell(family, 0)
    spec["params"].update(bad["params_over"])
    with pytest.raises(families.SpecError):
        families.build(family, spec, m, aligned)


def test_a_cell_outside_the_pinned_universe_is_refused():
    frames = _blind_frames(symbols=tuple(families.R2_UNIVERSE) + ("FNGU",))
    m, aligned = market_of(frames)
    spec = cell("E1", 0)
    spec["symbols"] = ["SOXL", "TQQQ", "SPXL", "FNGU"]
    with pytest.raises(families.SpecError, match="universe"):
        families.build("E1", spec, m, aligned)


@pytest.mark.parametrize("family,signal", [
    ("E1", {"gap_threshold": 0.75}),
    ("E2", {"gap_threshold": 2.0}),
    ("G", {"breach_atr": 0.75}),
    ("H", {"dev_threshold": 0.25, "rvol_min": 0.0005}),
    ("H", {"dev_threshold": 1.0, "rvol_min": 0.0020}),
])
def test_a_signal_level_off_the_declared_grid_is_refused(family, signal):
    """A 37th hypothesis must not be reachable by editing a cell."""
    frames = _blind_frames()
    m, aligned = market_of(frames)
    spec = cell(family, 0)
    spec["signal"] = dict(signal)
    with pytest.raises(families.SpecError):
        families.build(family, spec, m, aligned)


# ── E1 / E2: the gap families ──────────────────────────────────────────


def test_the_gap_families_reproduce_the_declared_rule():
    """Recompute ``gap_atr`` and the first-bar rule from the raw features."""
    for gap_pct in (0.0005, -0.0005, 0.0001):
        frames = frames_2day(flat_bars(100.0 * (1.0 + gap_pct)))
        m, aligned = market_of(frames)
        f = aligned["SOXL"]
        gap = ((f["sess_open"] / f["prev_close"] - 1.0)
               / f["atr_prev_pct"]).to_numpy(dtype=float)
        for family, sign in (("E1", +1), ("E2", -1)):
            start, end = families.R2_ENTRY_WINDOWS[family]
            for spec in families.GRIDS[family]:
                g = float(spec["signal"]["gap_threshold"])
                mask = np.isfinite(gap) & (np.abs(gap) >= g)
                want = declared_first_per_session(m, mask, start, end)
                _m, _a, _cfg, insts, _r = build_family(family, spec, frames)
                got = insts["SOXL"].entry_dir
                idx = entry_indices(insts["SOXL"])
                assert len(idx) <= 1, "one signal per session, not a running mask"
                for i in idx:
                    assert got[i] == sign * int(np.sign(gap[i])), \
                        f"{family} {spec['name']}: direction vs gap_atr {gap[i]}"
                    assert want[i] == 1, \
                        f"{family} {spec['name']}: signalled a bar the rule excludes"


def test_the_gap_threshold_axis_selects_the_sessions():
    """A 1.25-ATR gap trades at g = 0.5 and 1.0, and not at g = 1.5."""
    frames = frames_2day(flat_bars(100.0 * 1.0005))       # gap_atr = 1.25
    fired = {}
    for spec in families.GRIDS["E1"]:
        _m, _a, _cfg, insts, _r = build_family("E1", spec, frames)
        fired[float(spec["signal"]["gap_threshold"])] = bool(
            entry_indices(insts["SOXL"]))
    assert fired[0.5] and fired[1.0] and not fired[1.5], fired


def test_a_gap_below_every_threshold_trades_nothing():
    frames = frames_2day(flat_bars(100.0 * 1.0001))        # gap_atr = 0.25
    for family in ("E1", "E2"):
        for spec in families.GRIDS[family]:
            _m, _a, _cfg, insts, _r = build_family(family, spec, frames)
            assert entry_indices(insts["SOXL"]) == [], spec["name"]


def test_e2_is_the_exact_mirror_of_e1_on_every_cell():
    frames = frames_2day(flat_bars(100.0 * 1.0005))       # gap_atr = 1.25
    traded = 0
    for s1, s2 in zip(families.GRIDS["E1"], families.GRIDS["E2"]):
        _m, _a, _c1, i1, _r1 = build_family("E1", s1, frames)
        _m2, _a2, _c2, i2, _r2 = build_family("E2", s2, frames)
        a = i1["SOXL"].entry_dir
        b = i2["SOXL"].entry_dir
        assert np.array_equal(a, -b), f"{s1['name']} vs {s2['name']}"
        traded += int(bool(a.any()))
    assert traded == 6, "the 0.5 and 1.0 gap cells fire on a 1.25-ATR gap"


def test_the_gap_entry_fills_at_the_next_bar_open_and_flat_by_the_close():
    frames = frames_2day(flat_bars(100.0 * 1.0005))
    m, _a, cfg, insts, _r = build_family("E1", cell("E1", 0), frames)
    res = run_search(m, insts, cfg, CostModel.baseline())
    assert len(res.trades) == cfg.max_positions == 2, \
        "four symbols signal, the pinned cap opens two"
    for _, tr in res.trades.iterrows():
        assert tr["entry_time"] == pd.Timestamp(D2) + pd.Timedelta(minutes=OPEN + 2)
        assert tr["exit_time"] == pd.Timestamp(D2) + pd.Timedelta(minutes=15 * 60 + 30)
        assert tr["exit_reason"] == "eod"
        assert tr["hold_minutes"] >= 1.0
        assert tr["entry_price"] > 100.0, "a gap up enters long"
    assert res.stats["entries"] + res.stats["skipped_total"] == res.stats["signals"]
    assert res.stats["skipped"]["max_positions"] == 2


def test_a_gap_down_is_traded_short_not_dropped():
    """``allow_short`` deletes half the family if it is mis-set."""
    frames = frames_2day(flat_bars(100.0 * 0.9995))
    m, _a, cfg, insts, _r = build_family("E1", cell("E1", 0), frames)
    assert cfg.allow_short is True
    res = run_search(m, insts, cfg, CostModel.baseline())
    assert len(res.trades) == 2
    for _, tr in res.trades.iterrows():
        assert tr["entry_price"] < 100.0
        assert tr["exit_reason"] == "eod"
    assert int(cfg.max_entries_per_session) == 1


# ── G: opening-range reversal ──────────────────────────────────────────


def or_day(spike_at: int, up: bool, n: int = 390) -> list:
    """30 flat bars (the OR), then a breach at *spike_at* and flat after."""
    bars = flat_bars(100.0, 30)
    for k in range(30, n):
        if k == spike_at:
            bars.append((100.0, 100.05 if up else 100.02,
                         99.98 if up else 99.95, 100.0))
        else:
            bars.append((100.0, 100.02, 99.98, 100.0))
    return bars


def test_g_reproduces_the_declared_rule_and_enters_counter_to_the_breach():
    for up in (True, False):
        frames = frames_2day(or_day(30, up))
        m, aligned = market_of(frames)
        f = aligned["SOXL"]
        hi = f["or30_hi"].to_numpy(dtype=float)
        lo = f["or30_lo"].to_numpy(dtype=float)
        high = f["high"].to_numpy(dtype=float)
        low = f["low"].to_numpy(dtype=float)
        atr = f["atr_prev_abs"].to_numpy(dtype=float)
        d2 = int(np.argmax(np.asarray(m.day == pd.Timestamp(D2).date())))
        assert atr[d2 + 30] == pytest.approx(2 * RANGE)
        start, end = families.R2_ENTRY_WINDOWS["G"]
        for spec in families.GRIDS["G"]:
            b = float(spec["signal"]["breach_atr"])
            mask = (high >= hi + b * atr) | (low <= lo - b * atr)
            want = declared_first_per_session(m, mask, start, end)
            _m, _a, _cfg, insts, _r = build_family("G", spec, frames)
            for i in entry_indices(insts["SOXL"]):
                assert want[i] == 1, f"{spec['name']}: bar {i} excluded by the rule"
                expected = -1 if up else +1
                assert insts["SOXL"].entry_dir[i] == expected, \
                    f"{spec['name']}: a breach {'up' if up else 'down'} must enter " \
                    f"counter to it"


def test_the_breach_axis_selects_which_cells_fire():
    """A 0.05 spike beyond the OR band fires at b = 0.25 and 0.5, not at b = 1.0."""
    for up in (True, False):
        frames = frames_2day(or_day(30, up))
        by_b: dict[float, list[bool]] = {}
        for spec in families.GRIDS["G"]:
            fired = bool(entry_indices(build_family("G", spec, frames)[3]["SOXL"]))
            by_b.setdefault(float(spec["signal"]["breach_atr"]), []).append(fired)
        assert by_b[0.25] == [True, True, True], by_b
        assert by_b[0.5] == [True, True, True], by_b
        assert by_b[1.0] == [False, False, False], by_b


def test_g_does_not_trade_a_breach_inside_the_opening_range():
    """10:00 is the first bar the OR high is final on (``ready``)."""
    frames = frames_2day(or_day(15, True))          # the spike is at 09:45
    for spec in families.GRIDS["G"]:
        _m, _a, _cfg, insts, _r = build_family("G", spec, frames)
        assert entry_indices(insts["SOXL"]) == [], spec["name"]


def test_g_enters_at_1000_at_the_earliest():
    frames = frames_2day(or_day(30, True))
    spec = cell("G", 0)
    m, _a, _cfg, insts, _r = build_family("G", spec, frames)
    idx = entry_indices(insts["SOXL"])
    assert idx == [m.axis.get_loc(pd.Timestamp(D2) + pd.Timedelta(minutes=OPEN + 30))]
    assert m.minute[idx[0]] == 10 * 60


# ── H: session-VWAP reversion behind the regime gate ───────────────────


def vwap_day(jump_to: float | None, n: int = 390, wobble: float = 0.0) -> list:
    """30 bars around 100 (optionally wobbling), then a jump to *jump_to*."""
    bars = []
    for k in range(30):
        px = 100.0 + (wobble if k % 2 else 0.0)
        bars.append((px, px + RANGE, px - RANGE, px))
    rest = 100.0 if jump_to is None else jump_to
    for _ in range(30, n):
        bars.append((rest, rest + RANGE, rest - RANGE, rest))
    return bars


def test_h_reproduces_the_declared_rule_on_every_cell():
    """dev, the regime gate and the counter-entry, recomputed independently."""
    for day2 in (vwap_day(100.06), vwap_day(100.60), vwap_day(None, wobble=0.08)):
        frames = frames_2day(day2)
        m, aligned = market_of(frames)
        f = aligned["SOXL"]
        c = f["close"].to_numpy(dtype=float)
        vwap = f["vwap"].to_numpy(dtype=float)
        atrp = f["atr_prev_pct"].to_numpy(dtype=float)
        rvol = f["rvol30"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = (c - vwap) / (atrp * c)
        start, end = families.R2_ENTRY_WINDOWS["H"]
        for spec in families.GRIDS["H"]:
            d = float(spec["signal"]["dev_threshold"])
            th = float(spec["signal"]["rvol_min"])
            mask = ((np.abs(dev) >= d) & (rvol >= th))
            want = declared_first_per_session(m, mask, start, end)
            _m, _a, _cfg, insts, _r = build_family("H", spec, frames)
            got_dir = insts["SOXL"].entry_dir
            for i in entry_indices(insts["SOXL"]):
                assert want[i] == 1, f"{spec['name']}: bar {i} excluded by the rule"
                expected = -1 if dev[i] > 0 else +1
                assert got_dir[i] == expected, \
                    f"{spec['name']}: a deviation of {dev[i]:+.2f} ATR must enter " \
                    f"counter to it"


def test_h_exits_when_the_deviation_returns_or_flips():
    """``dev >= -0.25`` exits a long; ``dev <= +0.25`` exits a short."""
    frames = frames_2day(vwap_day(100.60))
    m, aligned = market_of(frames)
    f = aligned["SOXL"]
    c = f["close"].to_numpy(dtype=float)
    dev = ((c - f["vwap"].to_numpy(dtype=float))
           / (f["atr_prev_pct"].to_numpy(dtype=float) * c))
    _m, _a, _cfg, insts, _r = build_family("H", cell("H", 0), frames)
    exit_now = insts["SOXL"].exit_now
    deep_pos = int(np.nanargmax(dev))
    assert dev[deep_pos] > 1.5
    assert exit_now[deep_pos] == 1, "a deep positive deviation exits a long"
    inside = int(np.argmax(np.abs(dev) <= 0.25))
    assert np.abs(dev[inside]) <= 0.25
    assert exit_now[inside] == 2, "inside the band either side may exit"


def test_h_exit_is_not_armed_at_the_entry_bar_of_a_long():
    """A long enters deep below VWAP where ``dev >= -0.25`` is false."""
    frames = frames_2day([(100.0, 100.02, 99.98, 100.0)] * 30
                         + [(100.0, 100.02, 99.98, 100.0)] * 30
                         + [(99.40, 99.42, 99.38, 99.40)] * 330)
    m, aligned = market_of(frames)
    f = aligned["SOXL"]
    c = f["close"].to_numpy(dtype=float)
    dev = ((c - f["vwap"].to_numpy(dtype=float))
           / (f["atr_prev_pct"].to_numpy(dtype=float) * c))
    _m, _a, _cfg, insts, _r = build_family("H", cell("H", 0), frames)
    idx = entry_indices(insts["SOXL"])
    assert idx, "the fixture must produce a long entry"
    for i in idx:
        assert insts["SOXL"].entry_dir[i] == 1, "below VWAP enters long"
        assert dev[i] <= -0.5
    assert insts["SOXL"].exit_now[idx[0]] == -1, \
        "at a deep negative deviation the *short* exit is armed, not the long's"


def test_the_regime_gate_switches_cells_off():
    """``rvol30`` is a causal gate: a quiet session trades no cell above it."""
    quiet = frames_2day(vwap_day(100.06))
    m, aligned = market_of(quiet)
    rvol = aligned["SOXL"]["rvol30"].to_numpy(dtype=float)
    hi = float(np.nanmax(rvol[np.asarray(m.minute) >= 10 * 60]))
    assert hi < 0.0002, f"the quiet fixture is not quiet enough (rvol30={hi:.6f})"
    for spec in families.GRIDS["H"]:
        _m, _a, _cfg, insts, _r = build_family("H", spec, quiet)
        assert entry_indices(insts["SOXL"]) == [], spec["name"]


# ── the causal normaliser ──────────────────────────────────────────────


def test_the_normalisers_are_the_prior_completed_session():
    frames = frames_2day(flat_bars(100.0))
    m, aligned = market_of(frames)
    f = aligned["SOXL"]
    day1 = np.asarray(m.day == pd.Timestamp(D1).date())
    day2 = np.asarray(m.day == pd.Timestamp(D2).date())
    abs_ = f["atr_prev_abs"].to_numpy(dtype=float)
    pct_ = f["atr_prev_pct"].to_numpy(dtype=float)
    assert np.isnan(abs_[day1]).all(), "day 1 has no completed prior session"
    # a flat tape: true range = high - low = 2 * RANGE, so ATR(14) = 2 * RANGE
    assert np.allclose(abs_[day2], 2 * RANGE)
    assert np.allclose(pct_[day2], 2 * RANGE / 100.0), \
        "the prior session's ATR over that session's own close"


# ── future-blindness at every dense cut, for all four families ──────────


def _blind_frames(n: int = 120, symbols=None) -> dict:
    """Five sessions across two calendar months, four symbols, both directions."""
    days = ("2025-01-30", "2025-01-31", "2025-02-03", "2025-02-04", "2025-02-05")
    frames: dict[str, pd.DataFrame] = {}
    for sym in (symbols or families.R2_UNIVERSE):
        sessions = {}
        for i, day in enumerate(days):
            drift = 0.5 if i < 3 else -0.4
            base = 100.0 + 2.0 * i
            sessions[day] = [(base + drift * k, base + drift * k + 0.05,
                              base + drift * k - 0.05, base + drift * k + 0.02)
                             for k in range(n)]
        frames[sym] = mk_frame(sessions)
    return frames


@pytest.mark.parametrize("family", ["E1", "E2", "G", "H"])
def test_every_round2_family_is_blind_to_the_future_at_every_dense_cut(family):
    frames = _blind_frames()
    out = assert_blind_to_the_future_dense(family, cell(family, 4), frames)
    assert out["cuts"] >= 5, out
    assert out["keys"] == sorted(families.R2_UNIVERSE)
    assert out["feature_rows_checked"] > 0


@pytest.mark.parametrize("family", ["E1", "E2", "G", "H"])
def test_the_last_cell_of_every_round2_grid_is_also_future_blind(family):
    """The far corner of each grid: the largest thresholds and the longest stop."""
    frames = _blind_frames()
    out = assert_blind_to_the_future_dense(family, cell(family, 8), frames)
    assert out["cuts"] >= 5
