"""The round-2 engine correctness gate (lead decisions §D2: E1–E6, §P2).

Every test in this file fails on the pre-gate engine (commit 69d5d8e) and pins
one of the failures that killed round 1's survivor:

* **E1** — fills keyed by symbol let two legs on one symbol collapse into one
  fill, so the cash cancels to exactly zero toll while ``cost_drag`` still books
  both legs' toll: a fabricated *profit* equal to the cancelled toll, and the
  screen's fold-breadth rule runs on that zero-cost number.
* **E3** — silent drops and delayed fills (a pending entry filled at a later
  bar's open; a position flatten that fails silently and carries overnight).
* **E4** — a feature name outside the declared set reading as all-NaN, which
  turns a typo into "killed: too few trips".
* **E5/E6** — a multi-leg config declaring a stop it never runs; a hold counted
  in bars instead of minutes.

The harness helpers live in ``tests.test_strategy_search_engine`` so this file
runs unchanged against the previous commit.
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
from src.backtesting.strategy_search import engine, families, screen  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Instrument,
    Leg,
    Market,
    run_search,
)
from src.backtesting.strategy_search.features import FeatureBook  # noqa: E402
from tests.test_strategy_search_engine import (  # noqa: E402
    OPEN_MIN,
    flat_session,
    mk_frame,
    simple_cfg,
)

FLAT_PX = 100.0
SLIP = CostModel.baseline().slip_per_share(FLAT_PX)      # 2 bps of $100 → 2¢


def built2(*args, **kwargs):
    """``families.build`` unpacked for the first two values (2- or 3-tuple)."""
    out = families.build(*args, **kwargs)
    return out[0], out[1]


def flat_market(symbols, n: int = 390, px: float = FLAT_PX):
    bars = flat_session("2025-01-02", px, n)
    return Market({s: mk_frame({"2025-01-02": bars}) for s in symbols})


def flat_trip(legs, key, direction: int = 1, exit_value: int = 2, n: int = 390,
              costs: CostModel | None = None, kind: str = "pair"):
    """One trip on a flat tape with *legs*; *exit_value* is the family's signal."""
    m = flat_market(sorted({l.symbol for l in legs}), n=n)
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[5] = direction
    inst = Instrument(key=key, legs=legs, entry_dir=entry,
                      exit_now=np.full(m.n(), exit_value, dtype=np.int8),
                      valid=m.valid(legs), kind=kind)
    cfg = simple_cfg(allow_short=True, max_positions=4)
    return run_search(m, {key: inst}, cfg,
                      costs if costs is not None else CostModel.baseline())


# ── E1: fills are keyed by (leg_index, symbol) ─────────────────────────


LEG_SETS = {
    "two_symbols_equal": (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0)),
    "two_symbols_unequal": (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 2.0)),
    "one_symbol_equal": (Leg("SOXL", 1, 1.0), Leg("SOXL", -1, 1.0)),
    "one_symbol_unequal": (Leg("SOXL", 1, 1.0), Leg("SOXL", -1, 2.0)),
}


@pytest.mark.parametrize("label", sorted(LEG_SETS))
def test_a_flat_pair_can_never_book_a_profit(label):
    """A flat tape: spread P&L is exactly zero, so only the toll may be booked.

    Two legs on one symbol used to collapse into a single fill per symbol, and
    ``pnl_gross + cost_drag`` then came out **positive** by the cancelled toll —
    a fabricated profit, which is the mechanism that made round 1's survivor
    look positive.  Both shapes must lose exactly the four adverse fills.
    """
    legs = LEG_SETS[label]
    res = flat_trip(legs, key=f"pair-{label}")
    assert len(res.trades) == 1, "the fixture signals exactly one trip"
    tr = res.trades.iloc[0]
    toll = 2.0 * SLIP * (tr["gross_notional"] / FLAT_PX)
    assert tr["pnl_gross"] + tr["slip_drag"] == pytest.approx(0.0, abs=1e-6), \
        "identical price paths: the strategy's own spread P&L is exactly zero"
    assert tr["pnl_gross"] + tr["cost_drag"] - tr["fees"] == pytest.approx(0.0, abs=1e-6), \
        "the same statement in the split's other form: gross + slippage = 0"
    assert tr["pnl_after_costs"] == pytest.approx(-toll, rel=1e-3), \
        "a flat pair must lose exactly one adverse fill per leg per side"
    assert tr["pnl_after_costs"] < 0, "never a positive P&L on a flat tape"


@pytest.mark.parametrize("label", sorted(LEG_SETS))
def test_the_flat_pair_toll_is_reported_honestly(label):
    """``cost_drag`` must equal what the fills actually charged — slippage plus
    fees, each reported on its own line."""
    legs = LEG_SETS[label]
    tr = flat_trip(legs, key=f"pair-{label}").trades.iloc[0]
    assert tr["cost_drag"] == pytest.approx(
        2.0 * SLIP * (tr["gross_notional"] / FLAT_PX), rel=1e-3)
    assert tr["cost_drag"] == pytest.approx(tr["slip_drag"] + tr["fees"], rel=1e-12)
    assert tr["fees"] == 0.0
    assert tr["slip_drag"] == pytest.approx(tr["cost_drag"], rel=1e-12)


@pytest.mark.parametrize("direction", [1, -1])
def test_a_same_symbol_pair_mirrors_like_a_distinct_one(direction):
    """The mirror of a same-symbol pair pays the same toll as the original."""
    same = flat_trip(LEG_SETS["one_symbol_equal"], key="SOXL/SOXL",
                     direction=direction).trades.iloc[0]
    distinct = flat_trip(LEG_SETS["two_symbols_equal"], key="SOXL/TQQQ",
                         direction=direction).trades.iloc[0]
    assert same["cost_drag"] == pytest.approx(distinct["cost_drag"], rel=1e-12)
    assert same["pnl_after_costs"] < 0


def test_every_trip_carries_the_accounting_identity():
    """``pnl_after_costs == Σ leg.side·(exit_fill − entry_fill)·qty − fees``."""
    tr = flat_trip(LEG_SETS["one_symbol_unequal"], key="SOXL/SOXL").trades.iloc[0]
    assert tr["identity_residual"] == pytest.approx(0.0, abs=1e-9)
    assert tr["identity_check"] == pytest.approx(tr["pnl_after_costs"], rel=1e-12)
    assert tr["leg_detail"].count("|") >= 10, "both legs are recorded"


def test_the_zero_cost_run_reconciles_with_the_reconstruction():
    """A real zero-cost replay must equal ``pnl_gross + cost_drag − fees``.

    This is the number the screen's fold-breadth rule reads; if the engine's own
    reconstruction of it disagrees with a replay that actually charges nothing,
    the zero-cost column is not measuring a P&L at all.
    """
    legs = LEG_SETS["one_symbol_unequal"]
    m = flat_market(["SOXL"])
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[5] = 1
    inst = Instrument(key="SOXL/SOXL", legs=legs, entry_dir=entry,
                      exit_now=np.full(m.n(), 2, dtype=np.int8),
                      valid=m.valid(legs), kind="pair")
    cfg = simple_cfg(allow_short=True, max_positions=4)
    base = run_search(m, {"SOXL/SOXL": inst}, cfg, CostModel.baseline())
    zero = run_search(m, {"SOXL/SOXL": inst}, cfg, CostModel.zero_cost())
    rec = engine.reconcile_zero_cost(base, zero)
    assert rec["same_trips"] and rec["same_fill_timestamps"]
    assert rec["within_tolerance"], rec["mismatches"]
    assert rec["zero_cost_net"] == pytest.approx(0.0, abs=1e-6), \
        "a flat tape has no gross edge at all with costs switched off"


def test_the_builder_refuses_a_universe_too_small_for_its_legs():
    """``2k <= |U| - 1``: a two-leg instrument needs a universe of five symbols."""
    syms = ["SOXL", "TQQQ", "SPY"]
    bars = flat_session("2025-01-02", FLAT_PX, 390)
    m = Market({s: mk_frame({"2025-01-02": bars}) for s in syms})
    book = FeatureBook({s: mk_frame({"2025-01-02": bars}) for s in syms})
    spec = dict(families.GRID_D[0])
    spec["signal"] = dict(spec["signal"])
    with pytest.raises(ValueError, match=r"2k|universe"):
        built2("D", spec, m, book.align(m.axis))


# ── E2: the recorded artefact is the resolved config ────────────────────


def bars_frame(day: str = "2025-01-02", n: int = 390, px: float = FLAT_PX):
    return mk_frame({day: flat_session(day, px, n)})


def gap_frame(mins, day: str = "2025-01-02", px: float = FLAT_PX):
    """A frame with bars only at the given minute offsets from the open."""
    idx = [pd.Timestamp(day) + pd.Timedelta(minutes=OPEN_MIN + m) for m in mins]
    rows = [(px, px + 0.05, px - 0.05, px, 1000.0)] * len(mins)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex(idx, name="ts"))


def family_a_spec(**param_over):
    spec = dict(families.GRID_A[3])
    spec["params"] = dict(spec["params"])
    spec["signal"] = dict(spec["signal"])
    spec["extras"] = dict(spec.get("extras", {}))
    spec["symbols"] = ["SOXL"]
    spec["params"].update(param_over)
    return spec


def a_market(n: int = 60):
    frames = {"SOXL": bars_frame(n=n)}
    m = Market(frames)
    return m, FeatureBook(frames).align(m.axis)


def five_symbol_market(n: int = 60):
    frames = {s: bars_frame(n=n) for s in ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")}
    m = Market(frames)
    return m, FeatureBook(frames).align(m.axis)


def test_a_spec_must_declare_every_behaviour_changing_param():
    m, aligned = a_market()
    spec = family_a_spec()
    del spec["params"]["eod_flat_min"]
    with pytest.raises(families.SpecError, match="required"):
        families.build("A", spec, m, aligned)


def test_the_builder_refuses_an_unknown_param_key():
    m, aligned = a_market()
    spec = family_a_spec(notional_usdz=1.0)
    with pytest.raises(families.SpecError, match="unknown param"):
        families.build("A", spec, m, aligned)


def test_the_builder_refuses_an_unknown_extras_key():
    m, aligned = a_market()
    spec = family_a_spec()
    spec["extras"] = {"stop_atr_mul": 1.0}
    with pytest.raises(families.SpecError, match="unknown extras"):
        families.build("A", spec, m, aligned)


def test_the_resolved_spec_is_returned_and_names_the_config():
    """``z_window`` is the knob the stage-1 artefact silently never recorded."""
    m, aligned = five_symbol_market()
    spec = dict(families.GRID_D[0])
    cfg, _inst, resolved = families.build("D", spec, m, aligned)
    assert resolved["signal"]["z_window"] == 30, "no .get() default: resolved"
    assert cfg.name == families.resolved_name("D", resolved)
    spec2 = dict(spec)
    spec2["signal"] = dict(spec["signal"], z_window=20)
    cfg2, _inst2, resolved2 = families.build("D", spec2, m, aligned)
    assert cfg2.name != cfg.name, "the name is generated from the resolved params"
    assert families.canonical_hash(resolved2) != families.canonical_hash(resolved)


def test_the_resolved_hash_is_window_independent():
    """W1's hash must equal W2's: the same resolved config, two bar sets.

    This replaces a **tautological** version that built the same frames twice
    and compared two evaluations of a deterministic function — it could not
    fail.  It now resolves the cell against two genuinely different windows
    (different dates, bar counts and price levels).  The runner-level versions
    of this — and the W1/W2 *mismatch* the guard must refuse — are in
    ``tests/test_strategy_search_round2_fourfixes.py`` (A4).
    """
    w1 = {s: bars_frame(day="2025-09-15", n=60, px=100.0)
          for s in ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")}
    w2 = {s: bars_frame(day="2024-09-16", n=90, px=250.0)
          for s in ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")}
    spec = dict(families.GRID_D[0])
    hashes, names = [], []
    for frames in (w1, w2):
        m = Market(frames)
        cfg, _inst, resolved = families.build(
            "D", spec, m, FeatureBook(frames).align(m.axis))
        hashes.append(families.canonical_hash(resolved))
        names.append(cfg.name)
    assert hashes[0] == hashes[1], "the hash is a function of the config, not the bars"
    assert names[0] == names[1]


def test_the_recorded_run_set_must_equal_the_declared_set():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_strategy_search as runner_mod

    key = runner_mod.run_key("C", "c1", "W1", "baseline", "fixed_notional")
    assert key == "C|c1|W1|baseline|fixed_notional"
    runner_mod.assert_exact_run_set({key}, {key}, "test")
    with pytest.raises(RuntimeError, match="does not equal the declared set"):
        runner_mod.assert_exact_run_set(
            {key}, {key, "C|c2|W1|baseline|fixed_notional"}, "test")


# ── E3: silent drops and delayed fills ─────────────────────────────────


def test_a_pending_entry_is_discarded_not_filled_later():
    """A fill bar the instrument does not have means *dropped*, not delayed.

    Filling at a later bar's open hands the config a free option — and for a
    gap/reversal family it is systematically flattering.
    """
    soxl = gap_frame(list(range(0, 11)) + list(range(21, 31)))
    spy = gap_frame(list(range(0, 31)))
    m = Market({"SOXL": soxl, "SPY": spy})
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[10] = 1                                     # fills on minute 10+1
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(m.n(), dtype=np.int8),
                      valid=m.valid((Leg("SOXL", 1, 1.0),)))
    res = run_search(m, {"SOXL": inst}, simple_cfg(max_positions=1),
                     CostModel.baseline())
    assert res.stats["skipped"]["stale_entry"] == 1
    assert len(res.trades) == 0, "the entry must be dropped, not filled later"
    assert res.stats["signals"] == 1
    assert res.stats["entries"] + res.stats["skipped_total"] == res.stats["signals"]


@pytest.mark.parametrize("kind", ["single", "pair"])
def test_a_missing_bar_at_the_flatten_time_is_fatal(kind):
    """A failed EOD flatten is fatal, never a position carried into tomorrow."""
    soxl = bars_frame(n=360)                          # 09:30 .. 15:29 only
    other = bars_frame(n=361)                         # has the 15:30 bar
    m = Market({"SOXL": soxl, "TQQQ": other})
    legs = ((Leg("SOXL", 1, 1.0),) if kind == "single"
            else (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0)))
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[60] = 1
    inst = Instrument(key="SOXL" if kind == "single" else "SOXL/TQQQ", legs=legs,
                      entry_dir=entry, exit_now=np.zeros(m.n(), dtype=np.int8),
                      valid=m.valid(legs),
                      kind="single" if kind == "single" else "pair")
    cfg = simple_cfg(max_positions=4, allow_short=True)
    with pytest.raises(RuntimeError, match="flatten"):
        run_search(m, {inst.key: inst}, cfg, CostModel.baseline())


def test_every_drop_is_counted_and_the_accounting_holds():
    """A short refused by ``allow_short`` is counted, not silently vanished."""
    m = Market({"SOXL": bars_frame()})
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[60] = -1
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(m.n(), dtype=np.int8),
                      valid=m.valid((Leg("SOXL", 1, 1.0),)))
    res = run_search(m, {"SOXL": inst}, simple_cfg(), CostModel.baseline())
    assert res.stats["skipped"]["allow_short"] == 1
    assert res.stats["entries"] == 0
    assert res.stats["signals"] == 1
    assert res.stats["entries"] + res.stats["skipped_total"] == res.stats["signals"]


# ── E4: lookahead and undeclared feature names ─────────────────────────


def test_an_unknown_feature_name_raises_instead_of_reading_as_nan():
    """A typo used to trade zero trips and read as 'killed: too few trips'."""
    frames = {"SOXL": bars_frame(n=60)}
    m = Market(frames)
    aligned = FeatureBook(frames).align(m.axis)
    with pytest.raises(ValueError, match="declared feature set"):
        families._col(aligned, "SOXL", "or99_hi", m.axis)
    with pytest.raises(ValueError, match="declared feature set"):
        families._col_feat(aligned["SOXL"], "z_window")


def test_an_atr_skip_is_counted_as_no_atr_not_min_qty():
    m = Market({"SOXL": bars_frame()})
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[60] = 1
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(m.n(), dtype=np.int8),
                      valid=m.valid((Leg("SOXL", 1, 1.0),)), atr_pct=None)
    cfg = simple_cfg()
    cfg = engine.SearchConfig(**{**cfg.params(), "extras": {"stop_atr_mult": 1.0}})
    res = run_search(m, {"SOXL": inst}, cfg, CostModel.baseline())
    assert res.stats["skipped"]["no_atr"] == 1
    assert res.stats["skipped"]["min_qty"] == 0


def test_family_c_is_blind_to_the_future():
    from tests.future_blindness import assert_blind_to_the_future

    frames = _blind_frames()
    spec = dict(families.GRID_C[1])
    spec["symbols"] = ["SOXL"]
    out = assert_blind_to_the_future("C", spec, frames,
                                     pd.Timestamp("2025-01-03 10:00"))
    assert out["keys"] == ["SOXL"]


def test_family_a_is_blind_to_the_future():
    from tests.future_blindness import assert_blind_to_the_future

    frames = _blind_frames()
    spec = dict(families.GRID_A[0])
    spec["symbols"] = ["SOXL"]
    out = assert_blind_to_the_future("A", spec, frames,
                                     pd.Timestamp("2025-01-03 10:00"))
    assert out["keys"] == ["SOXL"]


def _blind_frames():
    s1 = [(100.0 + 0.05 * i, 100.1 + 0.05 * i, 99.9 + 0.05 * i, 100.05 + 0.05 * i)
          for i in range(120)]
    s2 = [(120.0 - 0.02 * i, 120.1 - 0.02 * i, 119.9 - 0.02 * i, 120.0 - 0.02 * i)
          for i in range(120)]
    df = mk_frame({"2025-01-02": s1, "2025-01-03": s2})
    return {"SOXL": df, "SPY": df.copy()}


# ── E5 / E6: declared-but-unexecuted exits, and reporting honesty ───────


def test_a_multi_leg_config_may_not_declare_a_stop():
    m, aligned = five_symbol_market()
    spec = dict(families.GRID_D[0])
    spec["params"] = dict(spec["params"], stop_pct=0.01)
    with pytest.raises(families.SpecError, match="more than one leg"):
        families.build("D", spec, m, aligned)


def test_a_time_exit_is_measured_in_minutes_not_bars():
    mins = list(range(0, 6)) + list(range(30, 41))
    df = gap_frame(mins)
    m = Market({"SOXL": df})
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[3] = 1                                      # filled on minute 4
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(m.n(), dtype=np.int8),
                      valid=m.valid((Leg("SOXL", 1, 1.0),)))
    res = run_search(m, {"SOXL": inst}, simple_cfg(time_exit_minutes=5),
                     CostModel.baseline())
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "time"
    assert tr["entry_time"] == m.axis[4]
    assert tr["exit_time"] == m.axis[6], \
        "the first bar at or after 5 *minutes*, not the fifth bar"
    assert tr["hold_minutes"] >= 5.0


def test_eod_flat_min_must_be_a_declared_value():
    m, aligned = a_market()
    spec = family_a_spec(eod_flat_min=15 * 60 + 59)
    with pytest.raises(families.SpecError, match="eod_flat_min"):
        families.build("A", spec, m, aligned)


def test_the_power_floor_is_150_trips():
    ok, why = screen.gate_power_and_net({"trips": 149, "net_bps_per_trip": 5.0})
    assert not ok and "150" in why
    ok, _why = screen.gate_power_and_net({"trips": 150, "net_bps_per_trip": 5.0})
    assert ok


def test_a_zero_trip_month_fails_the_stability_rule():
    good = [{"month": f"2025-{m:02d}", "round_trips": 20, "net_bps_per_trip": 1.0}
            for m in range(1, 13)]
    assert screen.fold_stability(good)[0]
    hole = list(good)
    hole[4] = {"month": "2025-05", "round_trips": 0, "net_bps_per_trip": 0.0}
    ok, why, zero = screen.fold_stability(hole)
    assert not ok and zero == 1 and "month" in why


def test_the_ranking_is_net_then_drawdown_then_trips_then_costs_then_stability():
    def row(net, dd, trips, drag, folds):
        return {"net_pct_equal_weighted": net, "max_drawdown_worst": dd,
                "trips_w1": trips, "trips_w2": trips,
                "cost_drag_bps_w1": drag, "cost_drag_bps_w2": drag,
                "folds_positive_min": folds}

    deep = row(0.05, -0.40, 200, 4.0, 9)
    shallow = row(0.05, -0.10, 200, 4.0, 9)
    assert screen.rank_row(shallow) > screen.rank_row(deep)
    fewer = row(0.05, -0.10, 160, 4.0, 9)
    assert screen.rank_row(shallow) > screen.rank_row(fewer)
    pricey = row(0.05, -0.10, 200, 9.0, 9)
    assert screen.rank_row(shallow) > screen.rank_row(pricey)
    narrow = row(0.05, -0.10, 200, 4.0, 7)
    assert screen.rank_row(shallow) > screen.rank_row(narrow)
    assert screen.rank_row(row(0.06, -0.90, 1, 99.0, 0)) > screen.rank_row(shallow)


def test_the_neighbour_rule_applies_per_axis():
    both_ok = {"n1": True, "n2": True, "n3": True}
    ok, why = screen.neighbour_verdict({"z_entry": ["n1"], "or_minutes": ["n2", "n3"]},
                                       both_ok)
    assert ok, why
    ok, why = screen.neighbour_verdict({"z_entry": ["n1"]}, {"n1": False})
    assert not ok and "z_entry" in why
    ok, why = screen.neighbour_verdict({"or_minutes": ["n2", "n3"]}, {"n2": True})
    assert not ok and "need 2" in why
    ok, why = screen.neighbour_verdict({"z_entry": [], "or_minutes": ["n2"]},
                                       {"n2": True})
    assert not ok and "z_entry" in why


def test_declared_axes_come_from_the_resolved_grid():
    axes = screen.declared_axes("C", families.GRID_C)
    assert "or_minutes" in axes and "stop_atr_mult" in axes
    assert "z_window" not in axes


def test_neighbour_cells_exclude_the_baseline_and_dedupe_by_hash():
    specs = [
        dict(name="c0", params=dict(families.GRID_C[1]["params"]),
             extras=dict(stop_atr_mult=1.0, target_atr_mult=2.0, trail_atr_mult=0.0),
             signal=dict(or_minutes=30, allow_short=False)),
        dict(name="c1", params=dict(families.GRID_C[1]["params"]),
             extras=dict(stop_atr_mult=1.5, target_atr_mult=2.0, trail_atr_mult=0.0),
             signal=dict(or_minutes=30, allow_short=False)),
        dict(name="c2", params=dict(families.GRID_C[1]["params"]),
             extras=dict(stop_atr_mult=1.5, target_atr_mult=2.0, trail_atr_mult=0.0),
             signal=dict(or_minutes=30, allow_short=False)),
        # varies two axes at once, so it is a neighbour of neither
        dict(name="c3", params=dict(families.GRID_C[1]["params"]),
             extras=dict(stop_atr_mult=2.0, target_atr_mult=2.0, trail_atr_mult=0.0),
             signal=dict(or_minutes=60, allow_short=False)),
    ]
    cells = screen.neighbour_cells("C", specs, "c0")
    assert "c0" not in [c for v in cells.values() for c in v]
    assert cells["stop_atr_mult"] == ["c1"], \
        "c2 resolves to the same config as c1 and is deduped"
    assert cells.get("or_minutes") == [], "an axis with no neighbour stays listed"
