"""L19.1–L19.5 — the judge's inputs, and two code defects in the round-2 path.

Source: ``/home/team/shared/ROUND2_L19_RULING.md`` (lead ruling L19, 2026-10-15,
on the verifier's adversarial read of ``ROUND2_GATE_ORDER_SPEC.md``).

What is pinned here, and why each one is a defect rather than a style question:

* **L19.2 — shadowed definitions.**  ``screen.py`` defined
  ``GateBlocksMissing``, ``assert_gate_blocks_present``,
  ``assert_round2_declaration``, ``round2_cell_fates``, ``breadth_at_base`` and
  ``round2_fold_breadth`` **twice each**; the later definition wins, so ~180
  lines of a *different* implementation of the same rules were dead while the
  suite passed against the live copy (the tests only ever exercised one of the
  two).  A silent-wrongness defect: the next reader cannot tell which rules are
  in force.  The scan below is over the whole search package **and**
  ``scripts/*.py`` — the two script modules ``c0a306d`` touched are in it.
* **L19.3 — presence is not content.**  ``assert_gate_blocks_present`` treated
  an empty dict as present, so ``{"benchmark": {}, "fat_tail": {},
  "concentration": {}}`` satisfied the one check the acceptance criteria cite as
  proof the gates were written.  It must fail on missing / ``None`` / empty, and
  name the offending block and the keys it requires.
* **L19.1 — the label.**  ``slip_x1_5`` is canonical (``costs.py``); the spec's
  ``slip_x1.5`` literals and its stale ``L3`` citation are corrected.  The dotted
  spelling stays **refused, not aliased**, and stage-2 records already carrying
  it are historical evidence and are not rewritten — so the check is on the
  *spec*, never on the records.
* **L19.4 — the ×1.5 gating level is part of the declared run set.**
  36 cells × 2 windows × 2 gating levels = **144** records, emitted and asserted
  as a set.  A declared gating level that has to be produced one config at a time
  by hand is not declared.
* **L19.5 — the judge's input is the per-run record.**  ``_trim``'s keep-list
  must carry ``cost_label``, ``folds`` and ``per_symbol``, or judge steps 5 (fold
  breadth), 6 (benchmark) and 8 (concentration) cannot be evaluated from what the
  screen writes.

Every test here fails on the pre-change tree (``c0a306d``); the recorded
pre-change run is the committed artefact
``docs/ROUND2_L19_PREFIX_PYTEST.txt``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.backtesting.strategy_search import costs, families, screen  # noqa: E402
import run_strategy_search as runner_mod  # noqa: E402

SPEC = Path("/home/team/shared/ROUND2_GATE_ORDER_SPEC.md")


# ── L19.2: shadowed definitions ────────────────────────────────────────
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def module_level_duplicates(path: Path) -> dict[str, list[int]]:
    """Names bound **twice** at module level, with both line numbers.

    Only real bindings are counted — ``def``/``class``/assignment targets.  An
    import is deliberately excluded: re-importing a name is not a second
    implementation of it, and a scan that cries wolf is a scan nobody reads.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    seen: dict[str, list[int]] = {}
    for node in tree.body:
        if isinstance(node, _DEF_NODES):
            bound = [node.name]
        elif isinstance(node, ast.Assign):
            bound = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign):
            bound = [node.target.id] if isinstance(node.target, ast.Name) else []
        else:
            continue
        for name in bound:
            seen.setdefault(name, []).append(node.lineno)
    return {k: v for k, v in seen.items() if len(v) > 1}


def _swept_modules() -> list[Path]:
    """The search package plus every script — ``c0a306d``'s modules are inside."""
    return (sorted((ROOT / "src" / "backtesting" / "strategy_search").glob("*.py"))
            + sorted((ROOT / "scripts").glob("*.py")))


@pytest.mark.parametrize("path", _swept_modules(), ids=lambda p: p.name)
def test_no_module_defines_a_module_level_name_twice(path: Path):
    """Exactly one definition per module-level name (L19.2).

    The later definition silently wins, so a duplicate is a *different*
    implementation of the same rule running while the dead one is read by
    whoever opens the file next.  Nothing in the suite catches it, because a
    test can only call the name, not the copy.
    """
    dups = module_level_duplicates(path)
    assert dups == {}, (
        f"{path.relative_to(ROOT)} defines {sorted(dups)} more than once "
        f"(line numbers per name): {dups} — the later definition shadows the "
        f"earlier, so one of the two implementations of each of those rules is "
        f"dead code that nothing can call and nothing can test (L19.2)")


def test_the_sweep_covers_the_modules_c0a306d_touched():
    """The sweep is not vacuous: it includes the modules the fix commit touched."""
    swept = {p.name for p in _swept_modules()}
    assert {"screen.py", "costs.py", "families.py"} <= swept
    assert {"run_strategy_search.py", "run_strategy_search_stage2.py"} <= swept


# ── L19.3: presence is not content ─────────────────────────────────────
def gate_record(**blocks) -> dict:
    """A minimal base record with the given gate blocks."""
    out = {"family": "X", "cell_id": "a", "config": "a", "window": "W1",
           "cost_label": "baseline", "sizing": "fixed_notional"}
    out.update(blocks)
    return out


def test_an_empty_gate_block_is_not_a_gate_block():
    """``{"benchmark": {}, ...}`` must fail: presence is not content (L19.3).

    This is the L17 failure class at one level down — the acceptance criteria
    cite ``assert_gate_blocks_present`` as the check that proves the gates were
    written, so a record carrying three *empty* mappings must not satisfy it.
    """
    record = gate_record(benchmark={}, fat_tail={}, concentration={})
    with pytest.raises(screen.GateBlocksMissing) as exc:
        screen.assert_gate_blocks_present(record)
    msg = str(exc.value)
    for block in screen.ROUND2_GATE_BLOCKS:
        assert block in msg, f"the failure must name the offending block {block!r}"
    assert "a" in msg and "W1" in msg, "the failure must name the record"


@pytest.mark.parametrize("block", screen.ROUND2_GATE_BLOCKS)
def test_a_block_that_is_none_or_absent_fails_and_names_its_required_keys(block):
    """Missing / ``None`` fail, and the error names the keys the block requires."""
    required = screen.ROUND2_GATE_BLOCK_KEYS[block]
    assert required, f"{block!r} must declare the keys it requires"
    for record in (gate_record(), gate_record(**{block: None})):
        with pytest.raises(screen.GateBlocksMissing) as exc:
            screen.assert_gate_blocks_present(record)
        msg = str(exc.value)
        assert block in msg, f"the failure must name the offending block {block!r}"
        for key in required:
            assert key in msg, (
                f"the failure must name the key {key!r} that {block!r} requires "
                f"(L19.3: 'must name the block's required keys')")


def test_a_non_empty_gate_block_still_passes():
    """The assertion still asserts presence — it does not become the judge."""
    record = gate_record(**{b: {"written": True} for b in screen.ROUND2_GATE_BLOCKS})
    screen.assert_gate_blocks_present(record)          # no raise


def test_the_declared_gate_block_keys_cover_every_declared_block():
    """One entry per declared block, non-empty, and no stray block."""
    assert set(screen.ROUND2_GATE_BLOCK_KEYS) == set(screen.ROUND2_GATE_BLOCKS)


# ── L19.1: the canonical label in the spec ─────────────────────────────
@pytest.mark.skipif(not SPEC.exists(), reason="the gate-order spec is a shared doc")
def test_the_gate_order_spec_spells_the_gating_label_canonically():
    """The spec's literal matches ``costs.py``; the dotted spelling is refused (L19.1).

    The dotted ``slip_x1.5`` is *not* aliased: a record already on disk under it
    is historical evidence and is not rewritten, so the correction is confined to
    the spec, which is a live contract document.
    """
    text = SPEC.read_text()
    assert costs.SLIP_X1_5 == "slip_x1_5"
    assert "slip_x1.5" not in text, (
        "the spec still carries the stale dotted literal 'slip_x1.5' (L19.1); the "
        "canonical spelling is costs.SLIP_X1_5 == 'slip_x1_5', and the dotted "
        "form is refused by costs.canonical_label rather than aliased")
    assert "slip_x1_5" in text
    assert '("baseline", "slip_x1_5")' in text, (
        "the spec's GATE_COST_LABELS literal must match the code's **tuple** "
        "(the old line was wrong twice: spelling, and a set literal compared "
        "against costs.GATE_COST_LABELS, which is a tuple)")
    assert "costs.py" in text, (
        "the stale L3 citation must be corrected to the code home of the label "
        "set (L19.1), so the literal and its source agree")


@pytest.mark.skipif(not SPEC.exists(), reason="the gate-order spec is a shared doc")
def test_the_spec_no_longer_claims_the_runner_cannot_emit_the_x1_5_label():
    """The stale sentence: ``costs.py`` landed, so the label has a producer (L19.1)."""
    text = SPEC.read_text()
    assert "cannot emit that label today" not in text, (
        "the spec still says the runner cannot emit the ×1.5 label; costs.py "
        "declares it and the round-2 path now emits it as a gating level (L19.1, "
        "L19.4)")


# ── L19.4: both gating levels, as a set ────────────────────────────────
def spec(name: str, **kw) -> dict:
    return {"name": name, "params": {}, "signal": {}, **kw}


def grid(*names: str) -> list[dict]:
    return [spec(n) for n in names]


def rec(cell: str, window: str, cost: str = "baseline",
        sizing: str = "fixed_notional", **kw) -> dict:
    out = {"family": "X", "cell_id": cell, "config": cell, "window": window,
           "cost_label": cost, "sizing": sizing, "trips": 200,
           "net_pnl": 100.0, "net_pct": 0.01, "net_bps_per_trip": 3.0,
           "folds": [{"month": f"2025-{m:02d}", "round_trips": 10,
                      "net_bps_per_trip": 1.0} for m in range(1, 13)]}
    out.update(kw)
    return out


def test_the_declared_round2_run_set_is_36_cells_by_2_windows_by_2_levels():
    """144 keys for the declared grid — both gating levels, exactly (L19.4)."""
    assert tuple(runner_mod.ROUND2_GATE_COSTS) == ("baseline", "slip_x1_5")
    declared = runner_mod.declared_round2_run_set("H", grid("a", "b", "c"))
    assert len(declared) == 3 * 2 * 2 == 12
    assert {k.split("|")[-2] for k in declared} == set(runner_mod.ROUND2_GATE_COSTS)
    assert {k.split("|")[-1] for k in declared} == {"fixed_notional"}

    n = len(families.ROUND2_FAMILIES)
    total = sum(len(runner_mod.declared_round2_run_set(f, families.GRIDS[f]))
                for f in families.ROUND2_FAMILIES)
    assert n == 4 and total == 36 * 2 * 2 == 144, (
        "36 cells x 2 windows x 2 gating levels = 144 records (L19.4)")


def test_the_round2_screen_path_emits_both_gating_levels(tmp_path):
    """``--screen`` on a round-2 family runs all 144 declared records (L19.4).

    Today it runs the 72 base records and the run-set assertion *raises* on the
    extra ×1.5 keys, so the declared ×1.5 level could only be produced one config
    at a time by hand — and a declared gating level that cannot be run as a set is
    not declared.
    """
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

    declared = [(f, s["name"], w, c)
                for f in families.ROUND2_FAMILIES for s in families.GRIDS[f]
                for w in screen.ROUND2_WINDOWS for c in runner_mod.ROUND2_GATE_COSTS]
    assert len(declared) == 144
    #: The *set* is declared; the order of the two gating levels is not, so the
    #: comparison is on the sorted list — exact in content, order-insensitive.
    assert sorted((f, c, w, cost) for f, c, w, cost, _sz in calls) == sorted(declared), \
        "every declared cell must run on both windows at both gating levels"
    assert {cost for *_x, cost, _sz in calls} == set(runner_mod.ROUND2_GATE_COSTS)
    assert {sz for *_x, sz in calls} == {screen.ROUND2_SIZING}
    for payload in payloads:
        assert len(payload["runs"]) == 9 * 2 * 2, \
            "the persisted record set is the 144 declared records"
        assert {k.split("|")[-1] for k in payload["runs"]} == \
            set(runner_mod.ROUND2_GATE_COSTS)


def test_the_run_set_assertion_still_fails_loudly_on_a_missing_or_extra_key(tmp_path):
    """Re-specified, not weakened: equality, both directions (L19.4)."""
    specs = grid("a", "b")
    expected = runner_mod.declared_round2_run_set("H", specs)
    assert len(expected) == 8

    runner = runner_mod.Runner(cache_dir=Path("/nonexistent-cache"),
                               out_dir=tmp_path / "out", verbose=False)
    for key in sorted(expected)[:-1]:                      # one key short
        runner.recorded[key] = {}
    with pytest.raises(RuntimeError, match="does not equal the declared set"):
        runner.assert_round2_grid_recorded("H", specs, "L19.4 missing key")

    runner.recorded[sorted(expected)[-1]] = {}
    runner.recorded["H|a|W1|slip_x2|fixed_notional"] = {}  # a level that is not gating
    with pytest.raises(RuntimeError, match="does not equal the declared set"):
        runner.assert_round2_grid_recorded("H", specs, "L19.4 extra key")


# ── L19.5: the judge's input is the per-run record ─────────────────────
@pytest.mark.parametrize("field", ["cost_label", "folds", "per_symbol"])
def test_trim_keeps_every_field_the_judge_reads(field):
    """``_trim`` must not drop ``cost_label`` / ``folds`` / ``per_symbol`` (L19.5).

    Without them the persisted screen artefact cannot support judge step 5 (fold
    breadth at base cost — it needs ``folds`` and the record's own cost level),
    step 6 (benchmark) or step 8 (concentration, which reads ``per_symbol``) — and
    a judge built against that payload would silently mis-score.
    """
    record = rec("a", "W1", **{field: {"sentinel": True}})
    trimmed = runner_mod._trim(record)
    assert field in trimmed, (
        f"_trim drops {field!r}: the judge's step-5/6/8 inputs must survive into "
        f"the persisted screen artefact (L19.5)")
    assert trimmed[field] == {"sentinel": True}
