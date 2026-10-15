"""The round-2 screen **path**: the declared grid, under the declared rule.

Source: ``ROUND2_PREREG_CONSISTENCY_AUDIT.md`` items 1, 2, 9, 10, 11 + the lead
rulings L1 (round 2's screen path IS the both-windows grid, ``fixed_notional``
sole gating sizing), L6 (fold breadth at BASE cost) and L16 (two assertions,
two names); ``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-3(iii)/(iv), §R3-10;
``ROUND2_GATE_ORDER_SPEC.md`` §1–§2.

What is pinned here, and why it is not cosmetic:

* **The wired path.**  ``scripts/run_strategy_search.py --screen`` used to call
  ``screen.screen_family``, which admits a config if **either** sizing mode
  clears W1 (so the grid's N was 72, not the declared 36), kills on W1 only,
  and cannot emit the declared gating cost level.  A grid run under that
  function is not the grid the protocol declares, so its numbers cannot be
  reported.  The test at the top of this file runs the round-2 entry point over
  the real declared family grids with a stubbed replay and asserts what was
  called: every declared cell, both windows, base cost, ``fixed_notional`` only
  — and that ``screen_family`` was never entered.
* **Either window kills.**  Round 2 is not stage 1: a cell dies when it fails
  ``W1`` **or** ``W2`` (R3-3(iii)/(iv)); a cell with no record on a window is a
  hard error, never a quietly smaller grid.
* **Fold breadth at base cost.**  The brief's kill criteria are stated at base
  cost; the round-1 ``breadth()`` reads a zero-cost record.
* **L16.**  ``assert_screen_recorded`` means *run-set equality*; the gate-block
  requirement is a second, differently named assertion.

Every test here fails on the pre-change tree (``9d31dad``): the new names it
calls do not exist yet.  The recorded pre-change run is the committed artefact
``docs/ROUND2_SCREENPATH_PREFIX_PYTEST.txt``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.backtesting.strategy_search import families, screen  # noqa: E402
import run_strategy_search as runner_mod  # noqa: E402


def spec(name: str, **kw) -> dict:
    return {"name": name, "params": {}, "signal": {}, **kw}


def grid(*names: str) -> list[dict]:
    return [spec(n) for n in names]


def rec(cell: str, window: str, cost: str = "baseline",
        sizing: str = "fixed_notional", **kw) -> dict:
    """A minimal record carrying the fields the two non-gate rules read."""
    out = {"family": "X", "cell_id": cell, "config": cell, "window": window,
           "cost_label": cost, "sizing": sizing, "trips": 200,
           "net_pnl": 100.0, "net_pct": 0.01, "net_bps_per_trip": 3.0,
           "folds": [{"month": f"2025-{m:02d}", "round_trips": 10,
                      "net_bps_per_trip": 1.0} for m in range(1, 13)]}
    out.update(kw)
    return out


# ── item 1: the wired path is the declared one ─────────────────────────
def test_the_round2_screen_path_is_the_both_windows_grid_not_screen_family(
        tmp_path, monkeypatch):
    """``--screen`` on a round-2 family runs ``screen_grid_both_windows``.

    The replay is stubbed, so this asserts the *declaration* the runner
    executes: 36 cells x 2 windows = 72 runs, all at base cost and
    ``fixed_notional`` — and not one of them through ``screen_family``, whose
    either-sizing admission would make N = 72.
    """
    def forbidden(*_a, **_k):
        raise AssertionError(
            "screen_family is ROUND 1's rule (L1): the round-2 --screen path "
            "must run screen_grid_both_windows over declared_round2_run_set")

    monkeypatch.setattr(screen, "screen_family", forbidden)

    runner = runner_mod.Runner(cache_dir=Path("/nonexistent-cache"),
                               out_dir=tmp_path / "out", verbose=False)
    calls: list[tuple[str, str, str, str, str]] = []

    def fake_run(family, sp, window, cost, sizing):
        calls.append((family, sp["name"], window, cost, sizing))
        record = rec(sp["name"], window, cost, sizing, family=family, trips=5,
                     net_bps_per_trip=-9.0, net_pnl=-100.0, folds=[])
        runner.recorded[runner_mod.run_key(family, sp["name"], window, cost,
                                           sizing)] = record
        return record

    runner.run = fake_run
    payloads = runner_mod.screen_round2(
        runner, list(families.ROUND2_FAMILIES), out_dir=tmp_path / "out",
        log=lambda *_: None)

    declared = [(f, s["name"], w)
                for f in families.ROUND2_FAMILIES for s in families.GRIDS[f]
                for w in screen.ROUND2_WINDOWS]
    assert len(declared) == 2 * screen.ROUND2_N_CONFIGS == 72
    assert [(f, c, w) for f, c, w, _cost, _sz in calls] == declared, \
        "every declared cell must be run on both windows, in the declared order"
    assert {cost for *_x, cost, _sz in calls} == {screen.ROUND2_BASE_COST}, \
        "the declared 72 base records are at base cost"
    assert {sz for *_x, sz in calls} == {screen.ROUND2_SIZING}, \
        "fixed_notional is the sole gating sizing (L1)"
    assert {p["status"] for p in payloads} == {"COLLECTED_NOT_JUDGED"}, \
        "the round-2 path collects the declared records; the gates are the " \
        "next delegation's job, so it must not write a verdict"
    assert all("verdict" not in p for p in payloads)


def test_the_declared_round2_grid_is_36_cells_with_z_bar_3_196():
    """N = 36 (4 groups x 9) and z_bar = 3.196 are declarations, not prose."""
    decl = screen.assert_round2_declaration()
    assert decl["n_configs"] == screen.ROUND2_N_CONFIGS == 36
    assert decl["z_bar"] == screen.ROUND2_Z_BAR == 3.196
    assert tuple(decl["families"]) == tuple(families.ROUND2_FAMILIES), \
        "the declaration covers exactly the four round-2 groups"
    assert len(set(decl["cells"])) == 36, \
        "two cells under one name is a silent drop (R3-3)"


def test_the_second_sizing_mode_can_be_recorded_but_can_never_admit_a_cell():
    """R3-3(ii): ``equity_fraction`` is a reported column, not a hypothesis.

    It may still be run — but it must land outside the gating record set
    (``runs``), so no admission test can ever read it.  Here the same grid is
    collected with ``equity_fraction`` requested for reporting and the gating
    set still holds exactly two windows per cell.
    """
    def run(sp, window, cost, sizing):
        return rec(sp["name"], window, cost, sizing)

    out = screen.screen_grid_both_windows(
        "X", grid("a", "b"), run, log=lambda *_: None,
        report_sizings=("equity_fraction",))
    assert set(out["runs"]) == {("a", "W1"), ("a", "W2"),
                                ("b", "W1"), ("b", "W2")}, \
        "the reporting sizing cannot enter the gating record set"
    assert set(out["reported_runs"]) == {
        ("a", "W1", "equity_fraction"), ("a", "W2", "equity_fraction"),
        ("b", "W1", "equity_fraction"), ("b", "W2", "equity_fraction")}
    assert all(r["sizing"] == "fixed_notional" for r in out["runs"].values())


def test_the_declared_round2_run_set_never_declares_the_reporting_sizing():
    expected = runner_mod.declared_round2_run_set("H", grid("a", "b", "c"))
    assert len(expected) == 6
    assert {k.split("|")[-1] for k in expected} == {"fixed_notional"}
    assert {k.split("|")[-2] for k in expected} == {"baseline"}


# ── item 2: a cell dies on EITHER window ───────────────────────────────
def test_a_cell_is_dead_if_it_fails_either_window():
    """The round-2 kill is conjunctive across windows (R3-3(iii)/(iv)).

    ``a`` clears W1 and loses W2; ``b`` clears both.  Failing W2 alone must be
    enough to kill ``a`` — the stage-1 rule ("kill on the development window")
    would have kept it.
    """
    runs = {
        ("a", "W1"): rec("a", "W1"),
        ("a", "W2"): rec("a", "W2", trips=5, net_bps_per_trip=-1.0),
        ("b", "W1"): rec("b", "W1"),
        ("b", "W2"): rec("b", "W2"),
    }
    fates = screen.round2_cell_fates(runs)
    assert fates["dead"] == ["a"], "a cell that fails W2 is dead even if W1 clears"
    assert fates["alive"] == ["b"]
    assert "W2" in fates["cells"]["a"]["why"]


def test_a_cell_that_fails_w1_is_dead_even_if_w2_clears():
    """The other direction of the same conjunction."""
    runs = {
        ("a", "W1"): rec("a", "W1", trips=0, net_bps_per_trip=0.0),
        ("a", "W2"): rec("a", "W2"),
    }
    fates = screen.round2_cell_fates(runs)
    assert fates["dead"] == ["a"] and fates["alive"] == []
    assert "W1" in fates["cells"]["a"]["why"]


def test_a_missing_window_record_is_a_hard_error_not_a_silent_drop():
    """A cell-window that produced no record cannot be scored as a loser."""
    runs = {("a", "W1"): rec("a", "W1")}
    with pytest.raises(screen.GridCellFailure, match="W2"):
        screen.round2_cell_fates(runs)


def test_the_stage1_kill_is_round_one_only_and_round_two_kills_on_either():
    """The two kill rules are declared separately, by name (L1).

    ``screen_family``'s kill runs on W1 and is **round 1's** rule; round 2's
    kill runs on both windows and is a different function.  The routing is
    asserted too: a round-2 family never reaches ``screen_family``.
    """
    assert screen.STAGE1_KILL_WINDOWS == ("W1",)
    assert screen.STAGE1_KILL_SCOPE == "round_1_only"
    assert screen.ROUND2_KILL_WINDOWS == ("W1", "W2")
    assert set(families.ROUND2_FAMILIES) <= set(runner_mod.ROUND2_SCREEN_FAMILIES), \
        "round-2 families must be routed to the round-2 path, not screen_family"
    assert not set(families.ROUND2_FAMILIES) & {"A", "B", "C", "D"}


# ── item 4: fold breadth at BASE cost ─────────────────────────────────
def test_fold_breadth_is_read_at_base_cost_not_zero_cost():
    """L6: the brief's kill criteria are stated at base cost.

    ``a``'s base record fails the 12-month rule while its zero-cost record
    passes; the round-2 breadth conjunct must read the base record, so ``a``
    fails.  Reading the other record is the defect (audit item 10).
    """
    thin = [{"month": f"2025-{m:02d}", "round_trips": 1,
             "net_bps_per_trip": 1.0} for m in range(1, 13)]
    runs = {
        ("a", "W1"): rec("a", "W1", folds=thin),
        ("a", "W2"): rec("a", "W2", folds=thin),
        ("z", "W1"): rec("z", "W1", cost="zero_cost"),
        ("z", "W2"): rec("z", "W2", cost="zero_cost"),
    }
    out = screen.round2_fold_breadth(runs)
    assert out["pass"] == [], "a thin base-cost month fails the config"
    assert "base" in out["cells"]["a"]["why"], out["cells"]["a"]


def test_a_zero_cost_record_is_refused_as_the_breadth_input():
    """The cost level is checked, not assumed: passing a zero-cost record is an error."""
    with pytest.raises(ValueError, match="base"):
        screen.breadth_at_base(rec("a", "W1", cost="zero_cost"),
                               cost_label="zero_cost")
    runs = {("a", "W1"): rec("a", "W1", cost="zero_cost"),
            ("a", "W2"): rec("a", "W2", cost="zero_cost")}
    with pytest.raises(ValueError, match="base"):
        screen.round2_fold_breadth(runs)


def test_the_round2_breadth_cost_is_the_declared_base_level():
    assert screen.ROUND2_BREADTH_COST == screen.ROUND2_BASE_COST == "baseline"


# ── item 5 (L16): two assertions, two names ───────────────────────────
def test_assert_gate_blocks_present_names_every_missing_block():
    """The gate-block assertion fails closed and names what is missing."""
    empty = rec("a", "W1")
    with pytest.raises(screen.GateBlocksMissing) as exc:
        screen.assert_gate_blocks_present(empty)
    msg = str(exc.value)
    for block in screen.ROUND2_GATE_BLOCKS:
        assert block in msg, f"the failure must name the missing block {block!r}"
    assert "a" in msg and "W1" in msg, "the failure must name the record"

    full = rec("a", "W1", benchmark={}, fat_tail={}, concentration={})
    screen.assert_gate_blocks_present(full)          # no raise
    with pytest.raises(screen.GateBlocksMissing, match="fat_tail"):
        screen.assert_gate_blocks_present(rec("a", "W1", benchmark={},
                                              concentration={}))


def test_the_run_set_assertion_and_the_gate_block_assertion_are_different_checks(
        tmp_path):
    """``assert_screen_recorded`` = run-set equality; L16's new name = blocks.

    A record with the right key and no gate block at all passes the run-set
    assertion — the split is the point: the run-set assertion cannot be cited
    as evidence that the gates were written.
    """
    runner = runner_mod.Runner(cache_dir=Path("/nonexistent-cache"),
                               out_dir=tmp_path / "out", verbose=False)
    specs = grid("a")
    expected = runner_mod.declared_screen_run_set("X", specs)
    for key in expected:
        cell, window, cost, sizing = key.split("|")[1:]
        runner.recorded[key] = rec(cell, window, cost, sizing)
    runner.assert_screen_recorded("X", specs, {}, "split")   # run-set only
    for key in expected:
        with pytest.raises(screen.GateBlocksMissing):
            screen.assert_gate_blocks_present(runner.recorded[key])
    for key in expected:
        runner.recorded[key].update({"benchmark": {}, "fat_tail": {},
                                     "concentration": {}})
        screen.assert_gate_blocks_present(runner.recorded[key])
    runner.assert_screen_recorded("X", specs, {}, "split")   # still only keys


def test_the_declared_gate_blocks_are_the_three_gate_docs_blocks():
    assert tuple(screen.ROUND2_GATE_BLOCKS) == ("benchmark", "fat_tail",
                                                "concentration")
