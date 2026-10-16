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

The gating levels charge **no commission and no half-spread** (L3), and what
holds that is the declaration itself — not a cash mechanism.  Post-L17 the
fixed notional is 24 000 with cap = |universe| = 4, so 4 × 24 000 = 96 000 of
the 100 000 equity: cash after three fills is ~$27 99x, which exceeds one more
24 000 plus its own fee by ~$4 000 against a fee of a few dollars, so a
commission-bearing gating level would still fund the fourth fill and would
**not** gate on a smaller basket (``ROUND2_DECLARATIONS_AUDIT.md`` item 6,
``ROUND2_PREREG_CONSISTENCY_AUDIT.md`` §6).  The pre-L17 arithmetic this module
used to quote — 4 × 25 000 = 100 000, so any fee refuses slot 4 — is falsified
by its own numbers (at exact full cash the engine's ``outlay > cash`` test
passes) and it protects nothing that is still true.  What protects the invariant
is the R3-5 declaration this module implements: gating is slippage-only, so the
gated basket is the same 4 × 24 000 at both gating levels (audit item 6).
:func:`assert_gate_cost_labels` pins that declaration, and runs at import.
"""
from __future__ import annotations

import dataclasses

from src.backtesting.replay_costs import COST_FIELDS, CostModel

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
#: Every level that is reported and **never** gates (R3-5).  The gating labels
#: are excluded by construction: ``slip_x1_5`` is *reported* (it stays in
#: ``STRESS_COST_LABELS``, the key set the stage-2 runner writes) **and** gates,
#: so listing it here too would put it in ``CANONICAL_LABELS`` twice — an entry
#: ``argparse`` tolerates and any consumer that builds a mapping from the tuple
#: silently collapses (``ROUND2_COSTS_AUDIT_FINDINGS.md`` defect 3).
REPORT_ONLY_COST_LABELS: tuple[str, ...] = (ZERO_COST,) + tuple(
    label for label in STRESS_COST_LABELS if label not in GATE_COST_LABELS)
CANONICAL_LABELS: tuple[str, ...] = GATE_COST_LABELS + REPORT_ONLY_COST_LABELS

#: Stale spellings of a canonical label: refused, never silently rewritten.
REJECTED_SPELLINGS: dict[str, str] = {"slip_x1.5": SLIP_X1_5}

#: The multiplier the ×1.5 level applies to **both** slippage knobs.
SLIP_X1_5_MULTIPLIER = 1.5
SLIP_X2_MULTIPLIER = 2.0

#: The probe prices the gating invariant checks the ×1.5 level at — both sides
#: of the $50 crossover, where ``max(price × slippage_pct, slippage_abs)`` swaps
#: its binding term: the 1¢ floor binds **below** $50, and a level that scaled
#: only ``slippage_pct`` would be identical to baseline there.
GATE_PROBE_PRICES: tuple[float, ...] = (4.75, 12.50, 49.99, 50.00, 500.00)


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
    """The declared invariant: exactly two gating levels, both slippage-only.

    Runs at import, so a future edit that adds a gating label, gives one a
    commission or a half-spread, or lets the ×1.5 level stop being strictly
    harder than baseline fails where it is written, not in the middle of a
    screen.  Every branch here is exercised by a positive control in
    ``tests/test_strategy_search_round2_cost_labels.py`` — nothing proved this
    guard could fail before that (``ROUND2_COSTS_AUDIT_FINDINGS.md`` defect 1).
    """
    if tuple(GATE_COST_LABELS) != (BASE_COST, SLIP_X1_5):
        raise AssertionError(
            f"the gating cost set is exactly ({BASE_COST!r}, {SLIP_X1_5!r}) "
            f"(L3); got {GATE_COST_LABELS!r}")
    if len(set(GATE_COST_LABELS)) != len(GATE_COST_LABELS):
        raise AssertionError(
            f"the gating labels must be distinct — one label per level (L3); "
            f"got {GATE_COST_LABELS!r}")
    for label in GATE_COST_LABELS:
        cm = cost_model(label)
        if cm.commission_per_share or cm.commission_pct:
            raise AssertionError(
                f"gating level {label!r} charges a commission "
                f"(${cm.commission_per_share}/share, {cm.commission_pct:.4%}): "
                f"the gating levels are slippage-only by **declaration** "
                f"(R3-5 / L3; ROUND2_DECLARATIONS_AUDIT.md item 6).  The "
                f"declaration is what keeps the gated basket the same 4 × 24 000 "
                f"at every gating level — not a cash mechanism, since post-L17 "
                f"the fourth fill is funded at full cash (~$4 000 of headroom "
                f"against a fee of a few dollars), so a commission would no "
                f"longer refuse the slot and would change nothing a gate could "
                f"see")
        if cm.half_spread_pct:
            raise AssertionError(
                f"gating level {label!r} carries a half-spread "
                f"({cm.half_spread_pct:.4%}): a half-spread is a cost the "
                f"gating levels must not have — the levels gate on slippage "
                f"only (L3), and 1 bp of half-spread adds ~2 bps per "
                f"single-leg round trip to every trip the level gates")
    base = cost_model(BASE_COST)
    harder = cost_model(SLIP_X1_5)
    for price in GATE_PROBE_PRICES:
        if not harder.slip_per_share(price) > base.slip_per_share(price):
            raise AssertionError(
                f"the ×1.5 gating level must be strictly harder than baseline "
                f"at every price: at ${price} it charges "
                f"{harder.slip_per_share(price)} against baseline's "
                f"{base.slip_per_share(price)}.  Both slippage knobs are scaled "
                f"(L3) — scaling ``slippage_pct`` alone leaves the level "
                f"identical to baseline below $50, where the 1¢ floor binds")
    if all(getattr(base, field) == getattr(harder, field) for field in COST_FIELDS):
        raise AssertionError(
            f"the two gating levels must not be the same model; {BASE_COST!r} "
            f"and {SLIP_X1_5!r} both resolve to {base!r}")


assert_gate_cost_labels()
