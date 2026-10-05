"""R3-3(iii)/(iv): the round-2 grid is **every** declared cell on **both** windows.

Source: ``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-3(iii)–(iv) and
``ROUND2_BUILDER_AUDIT.md`` §B4.  The rule the round-2 screen runs under is not
stage 1's: ``screen.screen_family`` kills a family at W1 (the round-1 rule, still
pinned by ``tests/test_strategy_search_engine.py::
test_a_family_with_no_w1_survivor_is_killed_without_touching_w2``), so a cell
that loses W1 licenses no W2 row and the 36 x 2 = 72 base records R3-3 declares
could not exist.  ``screen.screen_grid_both_windows`` is the round-2 rule:

* **no early kill** — every declared cell runs on both windows regardless of its
  W1 number;
* **no silent drop** — a cell that raises, or returns no record, is collected and
  raised as a :class:`screen.GridCellFailure` **after** every cell was attempted,
  naming all of them.  A grid that quietly shrinks to the cells that happened to
  work is the failure mode this rule exists to make impossible, because the
  report would still say N = 36.

Every test here fails on the pre-change commit ``2146ed4`` (the fixtures are
new): the both-windows path had no test at all — a grep of ``tests/`` found only
the definition, never a caller.

The last test pins the two readings **side by side**, so a later session cannot
silently "fix" one into the other: flipping ``screen_family`` to round-2's
unconditional rule (or the reverse) breaks it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_search import screen  # noqa: E402


def spec(name: str) -> dict:
    return {"name": name, "params": {}, "signal": {}}


def grid(*names: str) -> list[dict]:
    return [spec(n) for n in names]


# ── the loud, whole-grid failure (R3-3(iii)) ───────────────────────────
def test_a_failing_cell_does_not_stop_the_grid_and_every_failure_is_named():
    """The loop must not stop at the first casualty.

    Three cells x two windows = six runs.  Two of them fail, on *different*
    cells and different windows.  ``GridCellFailure`` must name **both**, and
    all six must have been attempted — a stop-at-first-casualty loop leaves the
    report scoring a grid it never ran.
    """
    attempted: list[tuple[str, str]] = []
    broken = {("b", "W1"), ("c", "W2")}

    def run(sp, window, cost, sizing):
        attempted.append((sp["name"], window))
        if (sp["name"], window) in broken:
            raise RuntimeError(f"no bar for {sp['name']} on {window}")
        return {"config": sp["name"], "window": window}

    with pytest.raises(screen.GridCellFailure) as exc:
        screen.screen_grid_both_windows("X", grid("a", "b", "c"), run,
                                       log=lambda *_: None)
    msg = str(exc.value)
    assert len(attempted) == 6, "every cell-window run must be attempted"
    assert set(attempted) == {(c, w) for c in "abc" for w in ("W1", "W2")}
    assert "2 of 6 declared cell-window runs" in msg, msg
    assert "b[W1]" in msg and "c[W2]" in msg, \
        "the message must name every failure, not just the first one"
    assert "RuntimeError" in msg, msg


def test_a_run_that_returns_nothing_is_a_failure_not_a_silently_recorded_cell():
    """An empty return is a cell that did not run, not a cell that ran and lost.

    ``screen_family``'s callable contract returns a record; if the runner hands
    back ``None`` (or ``{}``) the honest reading is "this cell did not run",
    which under R3-3(iii) is a hard failure — recording it as an empty dict
    would let a report count it as a scored cell.
    """
    def run(sp, window, cost, sizing):
        empty = {("a", "W2"): None, ("b", "W1"): {}}.get((sp["name"], window))
        return empty if empty is not None or (sp["name"], window) == ("a", "W2") \
            else {"config": sp["name"], "window": window}

    with pytest.raises(screen.GridCellFailure) as exc:
        screen.screen_grid_both_windows("X", grid("a", "b"), run,
                                       log=lambda *_: None)
    msg = str(exc.value)
    assert "2 of 4 declared cell-window runs" in msg, msg
    assert "returned no record" in msg, \
        "an empty return must be named as 'no record', not recorded as a cell"
    assert "a[W2]" in msg and "b[W1]" in msg, msg
    assert "a[W1]" not in msg and "b[W2]" not in msg, \
        "a cell-window that returned a record must not be listed as a failure"


def test_the_grid_is_not_scored_if_any_cell_window_failed():
    """Fail closed: a partial grid returns **nothing**, not a smaller grid."""
    rows: list[dict] = []

    def run(sp, window, cost, sizing):
        if sp["name"] == "b":
            raise ValueError("unresolvable cell")
        return {"config": sp["name"], "window": window}

    with pytest.raises(screen.GridCellFailure):
        rows.append(screen.screen_grid_both_windows("X", grid("a", "b"), run,
                                                    log=lambda *_: None))
    assert rows == [], "a failed grid must not hand back a scoreable payload"


def test_duplicate_cell_names_in_a_declared_grid_are_refused_before_any_run():
    """Two cells under one name is a silent drop waiting to happen."""
    calls: list[tuple[str, str]] = []

    def run(sp, window, cost, sizing):
        calls.append((sp["name"], window))
        return {"config": sp["name"]}

    with pytest.raises(ValueError, match="duplicate cell"):
        screen.screen_grid_both_windows("X", grid("a", "a"), run,
                                       log=lambda *_: None)
    assert calls == [], "the grid must be validated before any run is attempted"


# ── the happy path ─────────────────────────────────────────────────────
def test_every_cell_of_the_grid_is_recorded_on_both_windows():
    """36 cells declare 72 base records; here, 2 cells declare 4."""
    seen: list[tuple[str, str, str, str]] = []

    def run(sp, window, cost, sizing):
        seen.append((sp["name"], window, cost, sizing))
        return {"config": sp["name"], "window": window, "trips": 200}

    out = screen.screen_grid_both_windows("X", grid("a", "b"), run,
                                         log=lambda *_: None)
    assert set(out["runs"]) == {("a", "W1"), ("a", "W2"),
                                ("b", "W1"), ("b", "W2")}
    assert out["cells"] == ["a", "b"]
    assert out["windows"] == ["W1", "W2"], "both windows, in the declared order"
    # R3-3(ii)/(iii): round 2's one gating sizing mode, at the base cost level
    assert out["sizing"] == "fixed_notional" and out["cost"] == "baseline"
    assert set(seen) == {("a", "W1", "baseline", "fixed_notional"),
                         ("b", "W2", "baseline", "fixed_notional"),
                         ("a", "W2", "baseline", "fixed_notional"),
                         ("b", "W1", "baseline", "fixed_notional")}
    assert out["runs"][("b", "W2")]["trips"] == 200


def test_the_round2_run_set_is_every_cell_on_both_windows():
    """The runner-level mirror of the same rule (``scripts/run_strategy_search``).

    ``declared_round2_run_set`` must be the stage-1 set's *unconditional*
    sibling: 3 cells declare 6 keys, not 3.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_strategy_search as runner_mod

    expected = runner_mod.declared_round2_run_set("H", grid("a", "b", "c"))
    assert expected == {
        "H|a|W1|baseline|fixed_notional", "H|a|W2|baseline|fixed_notional",
        "H|b|W1|baseline|fixed_notional", "H|b|W2|baseline|fixed_notional",
        "H|c|W1|baseline|fixed_notional", "H|c|W2|baseline|fixed_notional"}
    runner = runner_mod.Runner(cache_dir=Path("/nonexistent-cache"),
                               out_dir=Path("/tmp") / "r2_bothwindows_out",
                               verbose=False)
    runner.recorded = {k: {} for k in expected}
    runner.assert_round2_grid_recorded("H", grid("a", "b", "c"), "test")
    del runner.recorded["H|b|W2|baseline|fixed_notional"]
    with pytest.raises(RuntimeError, match="missing"):
        runner.assert_round2_grid_recorded("H", grid("a", "b", "c"), "test")


# ── the rule boundary: two rules, both live, deliberately different ─────
def test_the_stage1_early_kill_is_not_the_round2_rule():
    """Both readings, side by side, on one fixture.

    Stage 1 (round 1's kill rule) stops at W1: a grid with no W1 survivor gets
    a verdict and **no W2 run at all**.  Round 2 (R3-3(iii)) is unconditional:
    the same grid, where every cell returns a *losing* record, still gets its
    full W1/W2 pair for every cell.

    The two must not collapse into one.  Re-pinning ``screen_family`` to the
    round-2 rule breaks the first half; re-pinning
    ``screen_grid_both_windows`` to the stage-1 rule breaks the second.
    """
    calls: list[tuple[str, str]] = []

    def losing(sp, window, cost, sizing):
        """Never clears the power floor — every cell loses on W1 too."""
        calls.append((sp["name"], window))
        return {"family": "X", "config": sp["name"], "window": window,
                "sizing": sizing, "trips": 5, "net_pnl": -100.0,
                "net_pct": -0.01, "net_bps_per_trip": -9.0,
                "net_bps_t_stat": -1.0, "cost_drag_per_trip": 20.0,
                "notional_per_trip": 50_000.0, "win_rate": 0.4,
                "break_even_win_rate": 0.5, "folds_positive": 0,
                "folds_total": 12, "avg_hold_minutes": 30.0,
                "trips_per_session": 0.1, "folds": []}

    # stage 1's rule: killed on W1, W2 never touched
    calls.clear()
    verdict = screen.screen_family("X", grid("a", "b"), losing,
                                   log=lambda *_: None)
    assert verdict["verdict"] == "KILLED_ON_W1"
    assert not [w for _, w in calls if w == "W2"], \
        "stage 1's early kill must not touch W2 (round-1 rule, still live)"
    assert not [k for k in verdict["runs"] if k.startswith("W2|")]

    # round 2's rule: same cells, same losing records, both windows anyway
    calls.clear()
    out = screen.screen_grid_both_windows("X", grid("a", "b"), losing,
                                         log=lambda *_: None)
    assert set(out["runs"]) == {("a", "W1"), ("a", "W2"),
                                ("b", "W1"), ("b", "W2")}
    assert [w for _, w in calls].count("W2") == 2, \
        "round 2 must run every cell on W2 even when every cell loses W1"
    assert out["runs"][("a", "W2")]["net_bps_per_trip"] == -9.0
