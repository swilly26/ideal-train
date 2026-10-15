"""The declared cost levels — **one canonical label per level** (L3, audit item 4).

Why this module exists
----------------------
Round 2 gates on exactly two cost levels: ``baseline`` and slippage ×1.5
(``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-5, "cost gates are exactly **base and
slippage ×1.5**").  Before this module that ×1.5 level existed only inside a
*script* (``scripts/run_strategy_search_stage2.py``), spelled ``slip_x1.5``
there while the funding probe's own records spelled it ``slip_x1_5``; the screen
runner could emit neither — its cost set was ``("baseline", "zero_cost")``.  A
gating level no code path can produce is not a gate, and two labels for one
level is how a reader ends up comparing a record to a column that is not the
same thing.

The canonical spelling is **``slip_x1_5``**: the probe records already on disk
under ``/home/team/shared`` are written with it, and a record is evidence and is
not rewritten.  The dotted spelling is not aliased — :func:`canonical_label`
*refuses* it, because a silent alias is how the clash survived.
``ROUND2_GATE_ORDER_SPEC.md`` §4 spells it ``slip_x1.5``; L3 ("ONE label
spelling used in docs and code") governs and that citation is stale.

Declared levels
---------------
=================  ====================================================  =======
label              construction                                         gating?
=================  ====================================================  =======
``baseline``       ``CostModel.baseline()``: 2 bps or 1¢, no fees,      **yes**
                   no half-spread
``slip_x1_5``      1.5 × ``slippage_pct`` **and** 1.5 ×                 **yes**
                   ``slippage_abs`` off baseline, no fees,
                   no half-spread
``zero_cost``      every cost off                                       no
``slip_x2``        2 × both slippage knobs off baseline                no
``pessimistic``    2 bps/1¢ + 1 bp half-spread + $0.005/share           no
``pessimistic_x2`` ``pessimistic`` with 4 bps / 2¢                      no
=================  ====================================================  =======

The gating levels charge **no commission** (L3).  With cap = |universe| = 4 and
4 × 24 000 = 96 000 of the 100 000 equity, the fourth slot is affordable only
while the fill's own cost fits the 4 000 of headroom; a commission-bearing level
would gate on a 3-symbol basket while base gated on 4 — a different portfolio at
a gating level (audit item 6).  :func:`assert_gate_cost_labels` pins both facts
and runs at import.
"""
from __future__ import annotations

import dataclasses

from src.backtesting.replay_costs import CostModel

BASE_COST = "baseline"
ZERO_COST = "zero_cost"
SLIP_X1_5 = "slip_x1_5"
SLIP_X2 = "slip_x2"
PESSIMISTIC = "pessimistic"
PESSIMISTIC_X2 = "pessimistic_x2"

#: The two gating labels, in the declared order (L3).
GATE_COST_LABELS: tuple[str, ...] = (BASE_COST, SLIP_X1_5)
#: The stress set the stage-2 runner reports (R3-5/R3-7) — the same key set it
#: used before this module existed, under the canonical ×1.5 spelling.
STRESS_COST_LABELS: tuple[str, ...] = (SLIP_X1_5, SLIP_X2, PESSIMISTIC,
                                       PESSIMISTIC_X2)
#: Every level that is reported and never gates (R3-5).
REPORT_ONLY_COST_LABELS: tuple[str, ...] = (ZERO_COST,) + STRESS_COST_LABELS
CANONICAL_LABELS: tuple[str, ...] = GATE_COST_LABELS + REPORT_ONLY_COST_LABELS

#: Stale spellings of a canonical label: refused, never silently rewritten.
REJECTED_SPELLINGS: dict[str, str] = {"slip_x1.5": SLIP_X1_5}

#: The multiplier the ×1.5 level applies to **both** slippage knobs.
SLIP_X1_5_MULTIPLIER = 1.5
SLIP_X2_MULTIPLIER = 2.0


def canonical_label(label: str) -> str:
    """The canonical spelling of a declared level, or a loud refusal (L3).

    The refusal is the point: mapping ``slip_x1.5`` onto ``slip_x1_5`` here
    would leave the dotted form legal in a caller, which is exactly the state
    the audit found.
    """
    if label in REJECTED_SPELLINGS:
        raise ValueError(
            f"{label!r} is the stale spelling of {REJECTED_SPELLINGS[label]!r}: "
            f"L3 declares ONE label per level, so the canonical spelling is "
            f"{REJECTED_SPELLINGS[label]!r} — the dotted form is refused rather "
            f"than aliased, because a silent alias is how the clash survived")
    if label not in CANONICAL_LABELS:
        raise ValueError(f"unknown cost label {label!r}: canonical labels are "
                         f"{CANONICAL_LABELS}")
    return label


def _scaled(label: str, multiplier: float) -> CostModel:
    """Baseline slippage × ``multiplier`` — ``slippage_pct`` **and** ``slippage_abs``."""
    base = CostModel.baseline()
    return dataclasses.replace(
        base,
        slippage_pct=base.slippage_pct * multiplier,
        slippage_abs=base.slippage_abs * multiplier,
        label=label)


def cost_model(label: str) -> CostModel:
    """The declared :class:`CostModel` for one canonical label."""
    label = canonical_label(label)
    if label == BASE_COST:
        return CostModel.baseline()
    if label == ZERO_COST:
        return CostModel.zero_cost()
    if label == PESSIMISTIC:
        return CostModel.pessimistic()
    if label == SLIP_X1_5:
        return _scaled(label, SLIP_X1_5_MULTIPLIER)
    if label == SLIP_X2:
        return _scaled(label, SLIP_X2_MULTIPLIER)
    if label == PESSIMISTIC_X2:
        return dataclasses.replace(CostModel.pessimistic(),
                                   slippage_pct=0.0004, slippage_abs=0.02,
                                   label=PESSIMISTIC_X2)
    raise ValueError(f"no construction declared for {label!r}")   # pragma: no cover


def stress_costs() -> dict[str, CostModel]:
    """The stress set the stage-2 runner reports, keyed by canonical label."""
    return {label: cost_model(label) for label in STRESS_COST_LABELS}


def gate_costs() -> dict[str, CostModel]:
    """The two gating levels, keyed by label — the set the judge reads (L3)."""
    return {label: cost_model(label) for label in GATE_COST_LABELS}


def assert_gate_cost_labels() -> None:
    """The declared invariant: exactly two gating levels, both commission-free.

    Runs at import, so a future edit that adds a gating label or gives one a
    commission fails where it is written, not in the middle of a screen.
    """
    if tuple(GATE_COST_LABELS) != (BASE_COST, SLIP_X1_5):
        raise AssertionError(
            f"the gating cost set is exactly ({BASE_COST!r}, {SLIP_X1_5!r}) "
            f"(L3); got {GATE_COST_LABELS!r}")
    for label in GATE_COST_LABELS:
        cm = cost_model(label)
        if cm.commission_per_share or cm.commission_pct:
            raise AssertionError(
                f"gating level {label!r} charges a commission "
                f"(${cm.commission_per_share}/share, {cm.commission_pct:.4%}): "
                f"at 4 × notional + a fee the fourth slot is refused "
                f"mechanically, so the level would gate on a 3-symbol basket "
                f"while base gated on 4 (audit item 6 / L3)")


assert_gate_cost_labels()
