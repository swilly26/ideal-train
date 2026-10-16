"""The declared cost levels: ONE canonical label per level (audit item 4 / L3).

Source: ``ROUND2_PREREG_CONSISTENCY_AUDIT.md`` item 4 (the runner's
``SCREEN_COSTS = ("baseline", "zero_cost")`` cannot emit the gating stress level
``×1.5``, and the label is spelled two ways across the tree);
``ROUND2_GATE_ORDER_SPEC.md`` §1 step 4 and §4 (gating set exactly
``{baseline, slip_x1_5}``); lead rulings L3 (one spelling, gating levels
commission-free) and L1.

Pinned here:

* the gating set is exactly two labels and the ×1.5 level is 1.5 × **both**
  slippage knobs off ``CostModel.baseline()``;
* the gating levels charge **no commission**, which is what makes the cap-4
  precondition hold where it gates (L3);
* the runner can emit ``slip_x1_5`` (the CLI's ``--cost`` accepts it and the
  round-2 path names it as a gating level);
* the stale dotted spelling ``slip_x1.5`` appears nowhere in ``src/`` or
  ``scripts/`` as a label — the clash the audit found cannot come back.

Added by the verifier's cost-module audit (``ROUND2_COSTS_AUDIT_FINDINGS.md``,
defects 1, 3 and 4; no coefficient moves):

* **one label is listed once** — ``REPORT_ONLY_COST_LABELS`` used to carry
  ``slip_x1_5``, which put the gating label in ``CANONICAL_LABELS`` twice and
  made the ``:64`` comment ("every level that is reported and never gates")
  false for it; the gating and report-only sets are now a partition of the
  canonical set (defect 3);
* **the declared invariant can fail** — positive controls hand
  ``assert_gate_cost_labels`` duplicate gate labels, a commission-bearing gate
  model, a half-spread-bearing gate model and a ×1.5 that is not strictly
  harder, and each one must raise (defect 1: the guard had no test anywhere in
  the tree, L15);
* **the stress levels' values are pinned** — ``slip_x2`` appeared in no test,
  ``pessimistic_x2`` in none at all, and ``costs.cost_model("pessimistic")``
  was not pinned to the turbo variant (defect 4);
* **the price bands are pinned** — baseline < ×1.5 < ×2 at probe prices on
  both sides of the $50 crossover, where the 1¢ floor binds below it (defect
  4).

Every test in this file fails on the pre-change tree (``9d31dad``): the module
it reads does not exist and ``scripts/run_strategy_search_stage2.py`` still
emits the dotted spelling.  The recorded pre-change run is the committed
artefact ``docs/ROUND2_SCREENPATH_PREFIX_PYTEST.txt``.  The tests added by the
costs audit fail on the tree before that audit's fix — the recorded run is
``docs/ROUND2_COSTS_GUARD_PREFIX_PYTEST.txt``.
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.backtesting.replay_costs import COST_FIELDS, CostModel  # noqa: E402
from src.backtesting.strategy_search import costs as cost_levels  # noqa: E402
import run_strategy_search as runner_mod  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_strategy_search_stage2.py"
SRC = ROOT / "src" / "backtesting" / "strategy_search" / "costs.py"

#: The probe prices the guard and the ordering test share: both sides of the $50
#: crossover, where ``max(p * slippage_pct, slippage_abs)`` swaps its binding term
#: (audit defect 4 — "the 1¢ floor binds below $50").
PROBE_PRICES = (4.75, 12.50, 49.99, 50.00, 500.00)


def _fields(model: CostModel) -> tuple:
    """The five cost knobs, in ``COST_FIELDS`` order — the whole model."""
    return tuple(float(getattr(model, f)) for f in COST_FIELDS)


def test_the_gating_cost_set_is_exactly_base_and_slip_x1_5():
    """L3: exactly two gating levels, under one spelling."""
    assert tuple(cost_levels.GATE_COST_LABELS) == ("baseline", "slip_x1_5")
    assert cost_levels.SLIP_X1_5 == "slip_x1_5"
    assert cost_levels.BASE_COST == "baseline"


def test_the_x1_5_level_is_one_and_a_half_times_both_slippage_knobs():
    base = CostModel.baseline()
    x15 = cost_levels.cost_model("slip_x1_5")
    assert x15.label == "slip_x1_5"
    assert x15.slippage_pct == pytest.approx(base.slippage_pct * 1.5)
    assert x15.slippage_abs == pytest.approx(base.slippage_abs * 1.5)
    assert x15.half_spread_pct == base.half_spread_pct
    assert x15.commission_per_share == base.commission_per_share
    assert x15.commission_pct == base.commission_pct


def test_the_gating_levels_charge_no_commission():
    """L3: gating is slippage-only, so the cap-4 precondition holds where it gates."""
    for label in cost_levels.GATE_COST_LABELS:
        cm = cost_levels.cost_model(label)
        assert cm.commission_per_share == 0.0 and cm.commission_pct == 0.0, \
            f"{label} must be commission-free: at 4 x notional + a fee the " \
            f"fourth slot is refused mechanically (audit item 6)"


def test_the_report_only_levels_are_declared_and_stay_out_of_the_gating_set():
    """×2 / pessimistic / zero-cost are reported, never gating (R3-5)."""
    for label in ("zero_cost", "slip_x2", "pessimistic", "pessimistic_x2"):
        assert label in cost_levels.REPORT_ONLY_COST_LABELS
        assert label not in cost_levels.GATE_COST_LABELS
        assert cost_levels.cost_model(label).label == label


def test_no_canonical_label_is_listed_twice():
    """Defect 3: the report-only set used to carry the gating label ``slip_x1_5``.

    ``REPORT_ONLY_COST_LABELS = (ZERO_COST,) + STRESS_COST_LABELS`` put the
    gating label in ``CANONICAL_LABELS`` twice — safe for ``argparse`` today
    and silently collapsing for any consumer that builds a mapping from the
    tuple.  One label, one entry.
    """
    canonical = tuple(cost_levels.CANONICAL_LABELS)
    assert len(set(canonical)) == len(canonical), (
        f"a canonical label is listed twice: {canonical!r} — REPORT_ONLY must "
        f"exclude the gating labels (audit defect 3)")
    assert canonical == ("baseline", "slip_x1_5", "zero_cost", "slip_x2",
                         "pessimistic", "pessimistic_x2"), canonical


def test_the_gating_and_report_only_sets_partition_the_canonical_labels():
    """Defect 3: gating or report-only — never both, never neither.

    ``slip_x1_5`` is *reported* and *gates*, so it stays in
    ``STRESS_COST_LABELS`` (the set the stage-2 runner reports) and in
    ``GATE_COST_LABELS``; what it must not be is report-**only**.
    """
    gating = set(cost_levels.GATE_COST_LABELS)
    report_only = set(cost_levels.REPORT_ONLY_COST_LABELS)
    canonical = set(cost_levels.CANONICAL_LABELS)
    assert gating & report_only == set(), (
        "a gating label is also in REPORT_ONLY_COST_LABELS — the 'reported and "
        "never gates' comment is then false for it (audit defect 3)")
    assert gating | report_only == canonical, (
        f"every canonical label is exactly one of gating / report-only; "
        f"gating={sorted(gating)}, report_only={sorted(report_only)}, "
        f"canonical={sorted(canonical)}")
    assert cost_levels.SLIP_X1_5 in cost_levels.STRESS_COST_LABELS, (
        "×1.5 is still part of the reported stress set (R3-5/R3-7); it is the "
        "report-ONLY *set* that must not carry it")


def test_the_stress_level_values_are_pinned():
    """Defect 4: ×2 appeared in no test and ``pessimistic_x2`` in none at all.

    Exact equality, not ``approx``: the point of these pins is that a
    coefficient cannot move unnoticed (the module's headline property is that
    nothing drifted from the extracted model).
    """
    assert _fields(cost_levels.cost_model("slip_x2")) == (0.0004, 0.02, 0.0, 0.0, 0.0)
    assert cost_levels.cost_model("slip_x2").label == "slip_x2"
    assert _fields(cost_levels.cost_model("pessimistic_x2")) == (
        0.0004, 0.02, 0.0001, 0.005, 0.0)
    assert cost_levels.cost_model("pessimistic_x2").label == "pessimistic_x2"
    assert _fields(cost_levels.cost_model("zero_cost")) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_the_pessimistic_wrapper_is_the_turbo_variant():
    """Defect 4 / parity: ``costs.cost_model("pessimistic")`` **is** the turbo variant.

    ``tests/test_replay_costs_parity.py`` pins ``CostModel.pessimistic()`` to
    ``scripts.run_turbo_backtest.build_config("classic", "pessimistic")``; this
    file's wrapper is the third copy of the same numbers, so it is pinned to
    the same variant here rather than left to the reader.
    """
    from scripts.run_turbo_backtest import build_config
    from src.backtesting.replay_costs import cost_fields_match_config

    cfg = build_config("classic", "pessimistic")
    model = cost_levels.cost_model("pessimistic")
    assert cost_fields_match_config(model, cfg)
    for f in COST_FIELDS:
        assert float(getattr(model, f)) == float(getattr(cfg, f)), f
    assert _fields(model) == (0.0002, 0.01, 0.0001, 0.005, 0.0)


def test_the_slippage_levels_are_strictly_ordered_across_the_price_bands():
    """Defect 4: baseline < ×1.5 < ×2 at every probe price, floor band included.

    ``slip_per_share(p) = max(|p| * slippage_pct, slippage_abs)``: the 1¢ floor
    binds below $50 and the percentage term above it, and both levels scale
    **both** terms, so the strict ordering must hold on either side of the
    crossover.
    """
    assert tuple(cost_levels.GATE_PROBE_PRICES) == PROBE_PRICES, (
        "the guard's probe prices must span both bands (audit fix 3b)")
    base = cost_levels.cost_model("baseline")
    x15 = cost_levels.cost_model("slip_x1_5")
    x2 = cost_levels.cost_model("slip_x2")
    for price in PROBE_PRICES:
        b, h, d = (base.slip_per_share(price), x15.slip_per_share(price),
                   x2.slip_per_share(price))
        assert b < h < d, (
            f"at ${price} the slippage levels are not strictly ordered: "
            f"baseline={b}, x1.5={h}, x2={d} — ×1.5 must be strictly harder "
            f"than baseline at every price (audit fix 3b)")


def test_the_one_cent_floor_binds_below_fifty_dollars():
    """Defect 4: "the floor binds under $50", pinned as a fact.

    At $49.99 the percentage term is 0.009998 < 0.01, so the floor is what is
    charged; at $50.00 the two are equal.  This is why the ×1.5 level must
    scale ``slippage_abs`` too — a level that scaled only ``slippage_pct``
    would be *identical* to baseline everywhere below $50.
    """
    base = cost_levels.cost_model("baseline")
    assert 49.99 * 0.0002 < 0.01
    assert base.slip_per_share(4.75) == 0.01
    assert base.slip_per_share(49.99) == 0.01
    assert base.slip_per_share(50.00) == pytest.approx(50.00 * 0.0002)
    assert base.slip_per_share(500.00) == pytest.approx(0.10)
    assert cost_levels.cost_model("slip_x1_5").slip_per_share(49.99) == pytest.approx(0.015)
    assert cost_levels.cost_model("slip_x2").slip_per_share(49.99) == pytest.approx(0.02)


def test_the_invariant_is_called_at_import():
    """Defect 1: the guard runs where the edit is written, not inside a screen."""
    src = SRC.read_text()
    assert re.search(r"^assert_gate_cost_labels\(\)$", src, re.M), (
        "costs.py must call assert_gate_cost_labels() at module level, so a "
        "bad gating level fails at import (audit defect 1)")


def test_the_guard_refuses_duplicate_gating_labels(monkeypatch):
    """Defect 1, positive control (a): two gate labels that are the same label."""
    monkeypatch.setattr(cost_levels, "GATE_COST_LABELS",
                        (cost_levels.BASE_COST, cost_levels.BASE_COST))
    with pytest.raises(AssertionError):
        cost_levels.assert_gate_cost_labels()


def test_the_guard_refuses_a_gating_level_that_charges_a_commission(monkeypatch):
    """Defect 1, positive control (a): a gate label handed a commission-bearing model."""
    def fake(label: str) -> CostModel:
        return dataclasses.replace(CostModel.baseline(),
                                   commission_per_share=0.005, label=label)

    monkeypatch.setattr(cost_levels, "cost_model", fake)
    with pytest.raises(AssertionError, match="commission"):
        cost_levels.assert_gate_cost_labels()


def test_the_guard_refuses_a_gating_level_that_carries_a_half_spread(monkeypatch):
    """Defect 1, positive control: the invariant checked commissions only.

    A gating level carrying a 1 bp half-spread passed the old import check
    while adding ~2 bps per single-leg round trip — the level would no longer
    be slippage-only.
    """
    def fake(label: str) -> CostModel:
        return dataclasses.replace(CostModel.baseline(),
                                   half_spread_pct=0.0001, label=label)

    monkeypatch.setattr(cost_levels, "cost_model", fake)
    with pytest.raises(AssertionError, match="half-spread"):
        cost_levels.assert_gate_cost_labels()


def test_the_guard_checks_every_gating_label_not_only_the_first(monkeypatch):
    """Defect 1: the invariant loops over the gating set, and it is pinned here."""
    plain = CostModel.baseline()

    def fake(label: str) -> CostModel:
        if label == cost_levels.SLIP_X1_5:
            return dataclasses.replace(plain, commission_pct=0.0001, label=label)
        return plain

    monkeypatch.setattr(cost_levels, "cost_model", fake)
    with pytest.raises(AssertionError, match="commission"):
        cost_levels.assert_gate_cost_labels()


def test_the_guard_refuses_an_x1_5_that_is_not_strictly_harder(monkeypatch):
    """Defect 1, positive control (b): ×1.5 that does not beat baseline.

    The declared tuple still reads ``("baseline", "slip_x1_5")``, so only the
    strictness check can catch a multiplier that collapses to 1.0 — which is
    exactly the edit the old guard let through.
    """
    monkeypatch.setattr(cost_levels, "SLIP_X1_5_MULTIPLIER", 1.0)
    with pytest.raises(AssertionError, match="strictly harder"):
        cost_levels.assert_gate_cost_labels()


def test_the_dotted_spelling_is_refused_loudly_rather_than_aliased():
    """One spelling, or a loud refusal — never a silent second spelling."""
    with pytest.raises(ValueError) as exc:
        cost_levels.canonical_label("slip_x1.5")
    msg = str(exc.value)
    assert "slip_x1_5" in msg and "slip_x1.5" in msg, msg


def test_unknown_cost_labels_are_refused():
    with pytest.raises(ValueError, match="unknown cost label"):
        cost_levels.cost_model("free_lunch")


def test_the_runner_can_emit_the_gating_cost_level():
    """``--cost slip_x1_5`` is a real choice, and the round-2 gate set is declared."""
    assert set(cost_levels.GATE_COST_LABELS) <= set(runner_mod.COST_CHOICES)
    assert runner_mod.ROUND2_GATE_COSTS == tuple(cost_levels.GATE_COST_LABELS)
    # the stage-2 stress set is the same declaration, under the one spelling
    assert set(cost_levels.stress_costs()) == set(cost_levels.STRESS_COST_LABELS)
    assert cost_levels.SLIP_X1_5 in cost_levels.stress_costs()


def test_no_stress_costs_are_defined_twice():
    """One definition: the stage-2 script delegates, it does not re-declare."""
    src = SCRIPT.read_text()
    body = src.split("def stress_costs():", 1)[1].split("\ndef ", 1)[0]
    assert "dataclasses.replace(" not in body, \
        "stress_costs() must live in src/backtesting/strategy_search/costs.py " \
        "alone; a second construction is how the label clash survived"
    assert "stress_levels()" in body, \
        "the stage-2 runner must read the one canonical stress set"


def test_the_dotted_spelling_appears_nowhere_as_a_label():
    """The stale spelling is gone from the engine and the scripts (L3).

    ``costs.py`` names it only to *refuse* it, so it is excluded.
    """
    allowed = {ROOT / "src" / "backtesting" / "strategy_search" / "costs.py"}
    hits = []
    for base in (ROOT / "src", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            if path in allowed or "__pycache__" in path.parts:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"slip_x1\.5", line):
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert hits == [], ("the dotted spelling is stale (L3): one label, "
                        f"slip_x1_5 — found {hits}")
