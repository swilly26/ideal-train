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

Every test here fails on the pre-change tree (``9d31dad``): the module it reads
does not exist and ``scripts/run_strategy_search_stage2.py`` still emits the
dotted spelling.  The recorded pre-change run is the committed artefact
``docs/ROUND2_SCREENPATH_PREFIX_PYTEST.txt``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import costs as cost_levels  # noqa: E402
import run_strategy_search as runner_mod  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_strategy_search_stage2.py"


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
    for label in ("zero_cost", "slip_x2", "pessimistic"):
        assert label in cost_levels.REPORT_ONLY_COST_LABELS
        assert label not in cost_levels.GATE_COST_LABELS
        assert cost_levels.cost_model(label).label == label


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
