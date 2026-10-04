"""The builder audit's missing assertions (§2.4, §5, §6.3, §6.4; brief §R3-11).

Source: ``/home/team/shared/ROUND2_BUILDER_AUDIT.md`` and
``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-11, which governs.

* **§2.4 / R3-11.3 — the OR ready gate.**  ``or30_hi``/``or30_lo`` are a
  whole-window maximum written onto every bar of the session (the declared sole
  exception to the prefix-causal invariant, kept on purpose).  The test that
  claimed to stand in for the gate was vacuous: its fixture ``or_day(15, True)``
  never places the 09:45 spike its own docstring names.  This file's fixture
  places them, and asserts ``entry_dir == 0`` on every bar with
  ``sess_min < G_OR_MINUTES`` — while the same day *does* fire at 10:00, so the
  zeros are not vacuous.
* **§5 / §6.4 — the literals asserted by value.**  The per-family time stops,
  the sizing mode and the sizing percentages were pinned only as "3 distinct
  levels" / "constant": ``H_TIME_STOP_MINUTES = 600`` or a family-wide switch to
  ``equity_fraction`` would have changed every cell and failed nothing.
* **§3.4 — the ``rvol30`` warm-up.**  H's 10:00 window start is load-bearing
  because ``rvol30`` (a 30-bar rolling std) does not exist before it.  That was
  a coincidence of two constants in two files; it is asserted here instead.
* **§6.4 — G's both-bands tie-break** (a bar whose range breaches both bands
  enters short, by rule) had no test.

No P&L and no screen: arrays only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_search import families  # noqa: E402
from tests.test_strategy_search_round2_families import (  # noqa: E402
    D2,
    RANGE,
    build_family,
    cell,
    flat_bars,
    frames_2day,
    market_of,
    vwap_day,
)


def or_day_custom(overrides: dict[int, tuple], n: int = 390) -> list:
    """30 flat bars (the OR) and flat after, with per-bar OHLC overrides.

    Unlike the older ``or_day``, an override may sit **inside** the opening
    range (indices < 30): the whole point of R3-11.3's fixture.
    """
    bars = flat_bars(100.0, n)
    for k, ohlc in overrides.items():
        bars[k] = ohlc
    return bars


def day2_index(m, day: str = D2) -> int:
    return int(np.flatnonzero(np.asarray(m.day == pd.Timestamp(day).date()))[0])


#: 09:45 and 09:59 breaches **inside** the opening range, plus a 10:00 bar that
#: breaches the band once it is final.  Against the band as it would be known at
#: each bar (a running max of the bars before it) the 09:45 spike (100.10) beats
#: the pre-spike high (100.02) and the 09:59 spike (100.20) beats 100.10 — both
#: clear the hardest declared threshold, 1.0 x ATR = 0.04.  Against the
#: whole-window band (100.20) neither can breach, which is why the gate is
#: enforced by the window too (R3-11.3).
IN_RANGE_SPIKES = {
    15: (100.00, 100.10, 99.98, 100.00),    # 09:45
    29: (100.00, 100.20, 99.98, 100.00),    # 09:59
    30: (100.20, 100.30, 99.98, 100.25),    # 10:00 — breaches the final band
}


def test_g_entry_dir_is_zero_on_every_bar_before_the_or_window_closes():
    """The fixture places its own spikes; ``entry_dir`` is 0 before 10:00.

    Pre-fix (R3-11.3) no test could fail on this property: the old fixture never
    placed the spike, and the OR band is a whole-window maximum, so an in-window
    bar can never breach it.  This test is the declaration's price: it fails if
    the OR band is ever exposed before it is final (see the mutation note in
    ``docs/ROUND2_AUDIT_ASSERTIONS_PREFIX_EVIDENCE.md``), and it no longer
    pretends the fixture contains a breach it does not.
    """
    day2 = or_day_custom(IN_RANGE_SPIKES)
    frames = frames_2day(day2)
    m, aligned = market_of(frames)

    # (i) the spikes are real: each beats the band known at that bar, by the
    #     hardest declared margin (breach_atr = 1.0 x atr_prev_abs = 0.04).
    hardest = max(families.G_BREACH_LEVELS) * 2 * RANGE
    running_high = max(b[1] for b in day2[:15])
    assert day2[15][1] >= running_high + hardest, "the 09:45 spike must breach"
    running_high = max(b[1] for b in day2[:29])
    assert day2[29][1] >= running_high + hardest, "the 09:59 spike must breach"
    # (ii) and the shipped band is the whole-window maximum — the declared leak.
    f = aligned["SOXL"]
    i0 = day2_index(m)
    assert f["or30_hi"].to_numpy(dtype=float)[i0 + 15] == day2[29][1]

    sess_min = f["sess_min"].to_numpy(dtype=float)
    for spec in families.GRIDS["G"]:
        _m, _a, _cfg, insts, _r = build_family("G", spec, frames)
        for sym in families.R2_UNIVERSE:
            ent = insts[sym].entry_dir
            pre = sess_min < families.G_OR_MINUTES
            assert not ent[pre].any(), (
                f"{spec['name']}/{sym}: fired on a bar with sess_min < "
                f"{families.G_OR_MINUTES}")
            assert ent[i0 + 15] == 0 and ent[i0 + 29] == 0, \
                f"{spec['name']}/{sym}: fired inside the opening range"
            # the same day does trade, at 10:00, counter to the breach — so the
            # zeros above are a gate, not an absence of signal
            assert list(np.flatnonzero(ent)) == [i0 + 30], spec["name"]
            assert ent[i0 + 30] == -1, spec["name"]


def test_g_or_minutes_is_exactly_the_entry_window_open():
    """The gate and the window are the same barrier; keep them one constant."""
    assert families.G_OR_MINUTES == 30
    assert families.G_OR_MINUTES == (families.R2_ENTRY_WINDOWS["G"][0]
                                     - families.RTH_OPEN_MIN), \
        "G's OR must close exactly when its entry window opens"


def test_the_family_time_stop_literals_are_asserted_by_value():
    """§5/§6.4: ``{None,10,20}`` or ``H_TIME_STOP_MINUTES = 600`` must fail."""
    assert families.E_EXIT_LEVELS == (("eod", None), ("t60", 60), ("t120", 120))
    assert families.G_STOP_LEVELS == (15, 30, 60)
    assert families.H_TIME_STOP_MINUTES == 60
    assert families.H_DEV_EXIT_ATR == 0.25
    assert families.E_GAP_LEVELS == (0.5, 1.0, 1.5)
    assert families.G_BREACH_LEVELS == (0.25, 0.5, 1.0)
    assert families.H_DEV_LEVELS == (0.5, 1.0, 1.5)
    assert families.H_RVOL_LEVELS == (0.0002, 0.0005, 0.0010)

    want = {"E1": {None, 60, 120}, "E2": {None, 60, 120},
            "G": {15, 30, 60}, "H": {60}}
    for family, levels in want.items():
        got = {spec["params"]["time_exit_minutes"] for spec in
               families.GRIDS[family]}
        assert got == levels, f"{family}: time stops {got} != {levels}"
        for spec in families.GRIDS[family]:
            assert spec["params"]["time_exit_minutes"] in levels, spec["name"]

    # the cell's *name* carries its own label; a name/value mismatch is a
    # declared cell that is not the cell it claims to be
    labels = {label: tstop for label, tstop in families.E_EXIT_LEVELS}
    for family in ("E1", "E2"):
        for spec in families.GRIDS[family]:
            label = spec["name"].rsplit("_", 1)[1]
            assert label in labels, spec["name"]
            assert spec["params"]["time_exit_minutes"] == labels[label], spec["name"]
    for spec in families.GRIDS["G"]:
        assert (int(spec["name"].rsplit("_t", 1)[1])
                == spec["params"]["time_exit_minutes"]), spec["name"]

    # and through build(): the config the engine runs with
    for family, index, minutes in (("E1", 0, None), ("E1", 1, 60),
                                   ("E1", 2, 120), ("G", 0, 15),
                                   ("G", 8, 60), ("H", 0, 60), ("H", 8, 60)):
        frames = frames_2day(flat_bars(100.0 * 1.0005))
        _m, _a, cfg, _i, _r = build_family(family, cell(family, index), frames)
        assert cfg.time_exit_minutes == minutes, f"{family} cell {index}"


def test_every_round2_cell_pins_the_sizing_literals():
    """§6.4: ``sizing``/``position_size_pct``/``bp_usage_pct`` were unasserted —
    switching every cell to another mode would have failed no test."""
    for family in families.ROUND2_FAMILIES:
        for spec in families.GRIDS[family]:
            p = spec["params"]
            assert p["sizing"] == "fixed_notional", spec["name"]
            assert p["position_size_pct"] == 0.50, spec["name"]
            assert p["bp_usage_pct"] == 0.95, spec["name"]
    frames = frames_2day(flat_bars(100.0 * 1.0005))
    for family in families.ROUND2_FAMILIES:
        _m, _a, cfg, _i, _r = build_family(family, cell(family, 0), frames)
        assert cfg.sizing == "fixed_notional", family
        assert cfg.position_size_pct == 0.50, family
        assert cfg.bp_usage_pct == 0.95, family


def test_h_regime_gate_is_decidable_exactly_when_its_window_opens():
    """§3.4: ``rvol30`` is a 30-bar rolling std, so H's 10:00 start is the first
    bar at which the regime gate *can* be evaluated.  Two constants in two
    files must not drift apart unnoticed."""
    frames = frames_2day(vwap_day(100.06))
    m, aligned = market_of(frames)
    rvol = aligned["SOXL"]["rvol30"].to_numpy(dtype=float)
    sess_min = (np.asarray(m.minute) - families.RTH_OPEN_MIN).astype(int)
    day2 = np.asarray(m.day == pd.Timestamp(D2).date())

    warmup = families.R2_ENTRY_WINDOWS["H"][0] - families.RTH_OPEN_MIN
    assert warmup == 30, "rvol30's rolling window is 30 one-minute returns"
    assert not np.isfinite(rvol[day2 & (sess_min < warmup)]).any(), \
        "rvol30 must be undefined before it has 30 returns"
    finite = np.flatnonzero(day2 & np.isfinite(rvol))
    assert len(finite) > 0, "the fixture must have a decidable regime gate"
    first = int(finite[0])
    assert int(m.minute[first]) == families.R2_ENTRY_WINDOWS["H"][0] == 10 * 60
    assert int(sess_min[first]) == warmup
    # every H cell's gate therefore *can* be evaluated on its first possible bar
    _m, _a, _cfg, insts, _r = build_family("H", cell("H", 0), frames)
    assert int(np.asarray(m.minute)[first]) >= int(_cfg.entry_start_min)


def test_g_both_band_breach_takes_the_short_side():
    """§6.4/R3-11.5: a bar whose range breaches both bands enters **short**, and
    a lower-band-only breach still enters long."""
    both = or_day_custom({30: (100.00, 100.30, 99.70, 100.00)})
    frames = frames_2day(both)
    m, aligned = market_of(frames)
    i0 = day2_index(m)
    f = aligned["SOXL"]
    high = f["high"].to_numpy(dtype=float)
    low = f["low"].to_numpy(dtype=float)
    hi = f["or30_hi"].to_numpy(dtype=float)
    lo = f["or30_lo"].to_numpy(dtype=float)
    atr = f["atr_prev_abs"].to_numpy(dtype=float)
    for spec in families.GRIDS["G"]:
        b = float(spec["signal"]["breach_atr"])
        k = i0 + 30
        assert high[k] >= hi[k] + b * atr[k], "the upper band must be breached"
        assert low[k] <= lo[k] - b * atr[k], "the lower band must be breached"
        _m, _a, _cfg, insts, _r = build_family("G", spec, frames)
        for sym in families.R2_UNIVERSE:
            ent = insts[sym].entry_dir
            assert list(np.flatnonzero(ent)) == [k], spec["name"]
            assert ent[k] == -1, \
                f"{spec['name']}/{sym}: a both-band breach is a short, by rule"

    down_only = or_day_custom({30: (100.00, 100.02, 99.70, 100.00)})
    frames2 = frames_2day(down_only)
    m2, aligned2 = market_of(frames2)
    f2 = aligned2["SOXL"]
    for spec in families.GRIDS["G"]:
        b = float(spec["signal"]["breach_atr"])
        _m, _a, _cfg, insts, _r = build_family("G", spec, frames2)
        k = day2_index(m2) + 30
        assert (f2["low"].to_numpy(dtype=float)[k]
                <= f2["or30_lo"].to_numpy(dtype=float)[k] - b * f2["atr_prev_abs"]
                .to_numpy(dtype=float)[k])
        assert insts["SOXL"].entry_dir[k] == 1, \
            f"{spec['name']}: a lower-band-only breach enters long"
