"""ROUND-2 JUDGE — screened records in, ranked verdicts out.

Implements ``ROUND2_GATE_ORDER_SPEC.md`` §1 steps 0–11 and the rulings that
bind it (``ROUND2_L19_RULING.md`` L12–L16, ``ROUND2_L19_SUPPLEMENT_RULING.md``
C1–C10).  It is the *consumer* of the round-2 contract, not a second copy of
it: every decision helper it needs already exists in
:mod:`src.backtesting.strategy_search.screen` and is imported rather than
re-implemented —

* step 2 — :func:`screen.round2_preconditions` (the shared fail-closed
  preconditions; ``why`` opens with the slug of the refusal),
* step 3 — :func:`screen.session_block_lower_bound` (the session-block
  bootstrap, pinned by its call sequence; returns the bound **and** ``B``),
* step 5 — :func:`screen.round2_fold_breadth` (base-cost fold breadth),
* step 9 — :func:`screen.round2_neighbour_stage` (``neighbour_cells`` →
  ``neighbour_verdict``; clone cells excluded per L9),
* step 11 — :func:`screen.round2_bucket` (the labelling order, the total
  vocabulary, one spelling for a precondition refusal),
* the two benchmark trip-count identities — :func:`screen.assert_benchmark_paired_identity`,
* the block-presence contract — :func:`screen.assert_gate_blocks_present`,
* the ranking — :func:`screen.rank_row`.

What this module *does* own is the **order of operations** and the **two
passes**: every gate is evaluated and recorded for every (cell, window) at both
gating cost levels even after a cell is already dead (L12), and the bucket is
assigned in a **second pass** after every cell has been scored, because step 9's
neighbour rule is not per-record (C1).  The bucket is the *first* failure in the
**labelling** order — headline → benchmark → fat-tail → concentration → fold
breadth → neighbour — which is deliberately **not** the order §1 evaluates the
steps in.  A cell that fails nothing is ``alive``; a cell that fails step 2 is
``not_evaluated``, one spelling, the reason in ``why`` (C2), with the single
exception that the ``paired_floor`` slug is ``headline_rejected`` (L19.7 / C10).

**Input.** The judge reads the **untrimmed per-run record files** (the
JSON projections ``run_strategy_search.Runner.run`` writes), not the family
payload: the trip table under :data:`screen.ROUND2_TRIP_TABLE_KEY`, the caps
under ``config_params`` restricted to :data:`families.R2_CAPS`, and the three
gate blocks (``benchmark``, ``fat_tail``, ``concentration``) the gate build
writes into every record.  ``_trim`` is **not** extended for the judge (L19-S
implementation notes): the family payload stays a summary.

**What is re-derived rather than trusted.** The pre-registration's standing rule
is "a gate that cannot be re-derived from committed bytes is not a gate", so the
judge re-derives, and raises :class:`GateBlockInconsistent` when a persisted
number contradicts its own inputs:

* the headline lower bound and ``B`` from the trip table, under the exact call
  sequence of §1 step 3 (a persisted ``bootstrap_block_p0.0694``/``B`` that
  disagrees is a defect, not a rounding difference);
* the four declared headline literals (seed, resamples, ``z_bar``, ``n_configs``)
  — a record that re-declares any of them is refused (L19.12: ``N = 36`` and
  ``z_bar = 3.196`` are fixed for the round-2 verdict);
* the per-trip mean ``mean(y_i)`` from the benchmark block's **own** ``per_trip``
  rows (L19.10 / C5), the pooled ``delta_bps = config − benchmark`` identity and
  the per-symbol decomposition ``pooled = Σ_s (n_s/N)·delta_s`` (R3-6.1 §1);
* each fat-tail and concentration rule's pass/fail from its own recorded
  ``value`` against the protocol literal, cross-checked against the published
  ``pass`` — so a block whose verdict contradicts its numbers cannot be scored;
* the concentration table from its own ``per_symbol`` rows
  (``Σ n_s == n_trips``, ``Σ P_s == Total``, ``max_share``, the ≥ 25-trip
  minimum over *traded* symbols, zero-trip rows as ``N/A``).

**What is asserted rather than assumed.** I2/L3: the two gating levels must
carry a commission-free fee schedule and the ×1.5 level must be exactly 1.5 ×
**both** slippage knobs of the base level, or the judge refuses to score
(:class:`CostLevelError`).  R3-6.3 §1: the caps in force are read out of the
concentration block and a block computed under another caps set refuses a
verdict (``not_evaluated``, slug ``caps``).

**Recorded, never scored** (L19.9 / L19.10): ``B`` — the entry-date block count
— is recorded per (cell, window) and reported, and **no verdict path reads it**
(no fifth or sixth gate exists for it); ``benchmark_fills_per_trip`` and
``delta_seed`` are declarations — recorded, flagged when they disagree with the
declaration, never scored; ``z_bar`` is the normal-approximation *reference* for
the same bar and is **not applied** by any gate (C7) — the decision number is
the empirical 0.0694th percentile of the bootstrap draws.

**The residual** (R3-6.2 §6.1 / the residual ruling's literal 12) is reported,
never gating: the four fields are quoted beside every survivor together with the
verbatim sentence in :data:`RESIDUAL_SENTENCE`.

**Both verdict shapes** (C9) are carried: the per-cell bucket (and
``cells_by_bucket``) **and** the three gate docs' own per-window rejection lists
(``benchmark_rejected``, ``fat_tail_rejected``, ``concentration_rejected``), plus
the three block-less stages' rows (``headline_rejected``, ``fold_breadth_rejected``,
``neighbour_rejected``).  A gate's rows are recorded whenever that gate failed —
even when the cell's *bucket* is an earlier stage's — so no failure hides behind
another's reason string (I3 / L12).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.backtesting.strategy_search import costs as cost_levels
from src.backtesting.strategy_search import families, screen

#: The judge's own version tag, recorded in every verdict so a payload can be
#: matched to the code that produced it.
ROUND2_JUDGE_VERSION = "R2-judge-1"

#: The two gating cost levels, in the declared order, taken from ``costs`` —
#: never re-spelled here (L19.1: one label per level, the dotted form refused).
GATING_COSTS: tuple[str, ...] = tuple(cost_levels.GATE_COST_LABELS)
BASE_COST = GATING_COSTS[0]
SLIP_COST = GATING_COSTS[1]

#: The four declared headline fields (spec §3) whose values are **literals**, not
#: measurements.  A record that disagrees is re-declaring the pre-registration;
#: ``n_configs``/``z_bar`` in particular may never move after the data is seen
#: (L19.12 — a forking path).
HEADLINE_DECLARED: tuple[tuple[str, float], ...] = (
    ("bootstrap_seed", screen.ROUND2_BOOTSTRAP_SEED),
    ("bootstrap_resamples", screen.ROUND2_BOOTSTRAP_RESAMPLES),
    ("z_bar", screen.ROUND2_Z_BAR),
    ("n_configs", screen.ROUND2_N_CONFIGS),
)
#: The two headline fields that are **measurements** and must therefore equal the
#: trip table's own re-derivation.
HEADLINE_DERIVED = ("bootstrap_block_p0.0694", "B")

#: The protocol literals each fat-tail rule is re-derived against (R3-6.2 §1:
#: strict ``> 0`` for the two remainders, ``<= 0.25`` for the session share).
FAT_TAIL_THRESHOLDS: Mapping[str, tuple[str, float]] = {
    "best_month": (">", 0.0),
    "trim": (">", 0.0),
    "session": ("<=", 0.25),
}
#: R3-6.3 §1: ``max_share <= 0.50`` (equality passes) and ``n_s >= 25`` per
#: traded symbol (equality passes).
CONCENTRATION_THRESHOLDS: Mapping[str, tuple[str, float]] = {
    "share": ("<=", 0.50),
    "min_trips": (">=", 25.0),
}
#: The five caps R3-6.3 requires inside every concentration block (and which
#: :func:`screen.round2_preconditions` checks under ``config_params``).
CAP_KEYS: tuple[str, ...] = tuple(families.R2_CAPS)

#: The residual's four fields (R3-6.2 §5, quoted beside every survivor) and the
#: verbatim sentence the residual ruling (§6) requires them to travel with.
RESIDUAL_FIELDS: tuple[str, ...] = ("remainder_after_best_k_bps",
                                    "best_session_share",
                                    "worst_session_date",
                                    "worst_session_symbols")
RESIDUAL_SENTENCE = (
    "Three-session residual, stated and not closed (R3-6.2 §6.1): this gate "
    "forbids one session above 25% of the window's dollars and does not forbid "
    "three at ~20% each; if this survivor's edge sits in three sessions then "
    "`remainder_after_best_k_bps` above, not `net_bps_per_trip`, is its honest "
    "size, because the same three session blocks are what the 0.0694th-"
    "percentile headline bootstrap's left tail is computed on, and a cell "
    "carrying 60% of its dollars in three sessions cannot clear that bar "
    "without a positive base of its own."
)

#: The one precondition slug that is **not** ``not_evaluated`` (L19.7 / C10).
PAIRED_FLOOR_SLUG = "paired_floor"


# ── refusals ───────────────────────────────────────────────────────────
class JudgeError(RuntimeError):
    """The judge cannot issue a verdict on the records it was handed."""


class RecordSetError(JudgeError):
    """The record set is ambiguous, incomplete or carries a stale spelling."""


class GateBlockInconsistent(JudgeError):
    """A gate block contradicts its own numbers or a declared literal.

    Not a strategy failure: a block whose published verdict disagrees with the
    values it publishes, or whose persisted number is not the one its own inputs
    re-derive, cannot be scored at all.  Failing loudly here is what stops a
    mis-priced block from being read as a strategy result (the stage-2 audit's
    "1 of 32 configs re-derived" failure class).
    """


class CostLevelError(JudgeError):
    """The two gating levels are not base and exactly 1.5 × slippage (I2 / L3)."""


# ── small helpers ──────────────────────────────────────────────────────
def _num(value, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise GateBlockInconsistent(f"{what} is not a number ({value!r})") from None


def _same_number(a, b) -> bool:
    """Numeric equality that survives a JSON round trip (``4`` vs ``4.0``)."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _isclose(a, b, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def _rule_holds(op: str, value: float, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    raise JudgeError(f"unknown comparison {op!r} — the thresholds are literals "
                     f"and are never re-spelled")


def _require(block: Mapping, key: str, where: str):
    if key not in block:
        raise GateBlockInconsistent(
            f"{where} carries no {key!r}: the block is not re-derivable without "
            f"it (spec §2, 'a gate that cannot be re-derived from committed "
            f"bytes is not a gate')")
    return block[key]


def _triple(stage: str, result: Mapping) -> tuple[str, bool, str]:
    return (stage, bool(result["pass"]), str(result["why"]))


def _result(ok: bool, why: str, **detail) -> dict:
    return {"pass": bool(ok), "why": str(why), "detail": dict(detail)}


# ── the record set ─────────────────────────────────────────────────────
def load_records(paths: Iterable[Path | str]) -> list[dict]:
    """Read the untrimmed per-run record files the judge consumes.

    A thin JSON reader: the judge's input is the run record on disk, and the
    files are named by ``run_strategy_search.record_filename`` (identity plus
    both hashes).  A file that is not a JSON object is a :class:`RecordSetError`
    — a half-written record must not be scored as a small one.
    """
    out: list[dict] = []
    for path in paths:
        text = Path(path).read_text()
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RecordSetError(f"{path}: not a JSON run record ({exc})") from exc
        if not isinstance(record, dict):
            raise RecordSetError(f"{path}: the run record is a "
                                 f"{type(record).__name__}, not an object")
        out.append(record)
    return out


def index_records(records: Iterable[Mapping]) -> dict[tuple[str, str, str, str],
                                                     Mapping]:
    """``(cell, window, cost_label, sizing)`` → record, refusing ambiguity.

    * the cost label is put through :func:`costs.canonical_label`, so the stale
      dotted spelling is **refused, not aliased** (L19.1) — a record whose level
      cannot be named cannot be placed in the gating set;
    * two records for one key is a :class:`RecordSetError`: two answers to one
      question, and picking one silently is how a re-run hides the first run's
      evidence;
    * every record's ``family`` must be a declared round-2 family
      (:data:`families.ROUND2_FAMILIES`) and every record in one call must be
      the same family — the neighbour rule and the declared grid are per family.
    """
    index: dict[tuple[str, str, str, str], Mapping] = {}
    family: str | None = None
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RecordSetError(f"record {i} is a {type(record).__name__}, not "
                                 f"a mapping")
        fam = record.get("family")
        if fam not in families.ROUND2_FAMILIES:
            raise RecordSetError(
                f"record {i} declares family {fam!r}; the round-2 judge scores "
                f"only {families.ROUND2_FAMILIES}")
        if family is None:
            family = str(fam)
        elif fam != family:
            raise RecordSetError(
                f"the record set mixes families {family!r} and {fam!r}: the "
                f"neighbour rule and the declared grid are per family, so one "
                f"verdict cannot cover both")
        try:
            label = cost_levels.canonical_label(str(record.get("cost_label")))
        except ValueError as exc:
            raise RecordSetError(f"record {i}: {exc}") from exc
        key = (str(record.get("cell_id") or record.get("config") or ""),
               str(record.get("window")), label,
               str(record.get("sizing")))
        if key[0] == "" or key[1] == "" or key[3] == "":
            raise RecordSetError(
                f"record {i} has no cell/window/sizing identity: "
                f"cell={record.get('cell_id')!r} window={record.get('window')!r} "
                f"sizing={record.get('sizing')!r}")
        if key in index:
            raise RecordSetError(
                f"two run records for {key}: a re-run must replace the file it "
                f"supersedes, never sit beside it — two answers to one question "
                f"is not a record set")
        index[key] = record
    return index


def _cell_windows(index: Mapping, cell: str) -> set[str]:
    return {window for (c, window, _cost, _sizing) in index if c == cell}


def _gating(index: Mapping, cell: str, window: str, cost: str) -> Mapping | None:
    """The **gating** record: the declared sizing only (L1 / R3-3(ii)).

    An ``equity_fraction`` record is a reported column and can never gate, so it
    is invisible here — which is what makes "only an ``equity_fraction`` record
    exists" a *refusal* (``not_evaluated``) rather than an admission.
    """
    return index.get((cell, window, cost, screen.ROUND2_SIZING))


# ── step 2: the shared preconditions (screen's, plus the block-level caps) ─
def cell_preconditions(record: Mapping) -> tuple[bool, str]:
    """Step 2 for one gating record: :func:`screen.round2_preconditions` + caps.

    The helper does the five conjuncts the spec names (trips floor, trip table,
    ``legs == 1``, uniform notional, caps under ``config_params``, and the paired
    floor read from this level's benchmark block).  R3-6.3 §1 adds one thing it
    cannot see: the caps **inside the concentration block**, which is the set the
    share was computed under — a block computed under another caps set may never
    be quoted beside a run under these, so a mismatch refuses a verdict
    (``not_evaluated``, slug ``caps``, one spelling — C2).

    ``why`` opens with the refusal's slug either way, because the judge's bucket
    pass routes on that slug (``paired_floor`` → ``headline_rejected``; anything
    else → ``not_evaluated``).
    """
    ok, why = screen.round2_preconditions(record)
    if not ok:
        return ok, why
    block = record.get("concentration")
    caps = block.get("caps") if isinstance(block, Mapping) else None
    if not isinstance(caps, Mapping):
        return False, ("caps: the concentration block carries no caps in force "
                       "(R3-6.3 §1), so the share it publishes cannot be quoted")
    wrong = sorted(k for k, v in families.R2_CAPS.items()
                   if k not in caps or not _same_number(caps[k], v))
    if wrong:
        return False, (f"caps: the concentration block's caps {wrong} are not the "
                       f"pre-registered {dict(families.R2_CAPS)} — a share "
                       f"computed under another caps set is not this run's "
                       f"(R3-6.3 §1)")
    return True, "ok"


def _precondition_stage(ok: bool, why: str) -> tuple[tuple, tuple | None]:
    """Map step 2's refusal onto the bucket vocabulary (L19.7 / C10 / C2).

    Returns ``(preconditions_triple, headline_override)``.  The ``paired_floor``
    slug is the **one** exception: ``config_trips >= 150`` with
    ``paired_trips < 150`` is a *headline* failure, never "the benchmark gate
    failed closed", so it is reported as ``headline_rejected`` — and it is
    decided **by the slug**, not by the stage, which is why this mapping lives in
    the judge and not in the helper.
    """
    if ok:
        return (screen.ROUND2_PRECONDITION_STAGE, True, "ok"), None
    slug = str(why).split(":", 1)[0].strip()
    if slug == PAIRED_FLOOR_SLUG:
        return ((screen.ROUND2_PRECONDITION_STAGE, True, "ok"),
                (False, str(why)))
    return (screen.ROUND2_PRECONDITION_STAGE, False, str(why)), None


# ── I2 / L3: the two gating levels really are base and 1.5 × slippage ───
def assert_cost_levels(base: Mapping, slip: Mapping) -> None:
    """Assert the ×1.5 level's fee schedule and multiplier (I2, L3).

    The gate docs forbid the judge from *assuming* this.  Two things are
    asserted, because each closes a distinct way the gate set can be wrong:

    * **commission-free** at both gating levels — a commission-bearing ×1.5 level
      can change which basket funds, so the concentration verdict (base) and the
      benchmark verdict (×1.5) would describe different baskets, invisibly (I2);
    * the ×1.5 level is exactly :data:`costs.SLIP_X1_5_MULTIPLIER` × **both**
      ``slippage_pct`` **and** ``slippage_abs`` of the base level, with the
      half-spread unchanged — the 1¢ floor binds below $50, so a level that
      scaled only the percentage would be identical to base there (L3).
    """
    models = {}
    for label, record in ((BASE_COST, base), (SLIP_COST, slip)):
        model = record.get("cost_model")
        if not isinstance(model, Mapping):
            raise CostLevelError(
                f"the {label} record carries no 'cost_model': the judge must "
                f"assert the fee schedule rather than assume it (I2/L3), and it "
                f"cannot assert what the record does not publish")
        if model.get("label") != label:
            raise CostLevelError(
                f"the {label} record's cost_model declares label "
                f"{model.get('label')!r} — a level that misnames itself cannot "
                f"be placed in the gating set (L19.1)")
        for knob in ("commission_per_share", "commission_pct"):
            if not _same_number(model.get(knob, 0.0), 0.0):
                raise CostLevelError(
                    f"the {label} gating level is not commission-free "
                    f"({knob}={model.get(knob)!r}); the gating set is "
                    f"slippage-only by declaration (L3), and a "
                    f"commission-bearing level can change which basket funds "
                    f"(I2)")
        models[label] = model
    b, s = models[BASE_COST], models[SLIP_COST]
    for knob in ("slippage_pct", "slippage_abs"):
        want = float(b.get(knob, 0.0)) * cost_levels.SLIP_X1_5_MULTIPLIER
        if not _isclose(s.get(knob, 0.0), want, tol=1e-12):
            raise CostLevelError(
                f"the ×1.5 level's {knob} is {s.get(knob)!r}, not "
                f"{cost_levels.SLIP_X1_5_MULTIPLIER} × the base level's "
                f"{b.get(knob)!r} (= {want!r}) — the level is not the declared "
                f"stress (L3), and bps that do not scale with the level are not "
                f"a stress test")
    if not _isclose(s.get("half_spread_pct", 0.0), b.get("half_spread_pct", 0.0),
                    tol=1e-12):
        raise CostLevelError(
            "the ×1.5 level's half_spread_pct differs from the base level's: "
            "the declared stress scales slippage only (L3)")


# ── step 3 + 4: the headline statistic and the ×1.5 net rule ───────────
def _table_bps(table) -> list[float]:
    """Per-trip net bps from the trip table, in either accepted shape."""
    if hasattr(table, "columns"):                      # the engine's own frame
        pnl = list(table["pnl_after_costs"])
        notional = list(table["notional"])
    else:
        rows = list(table)
        pnl = [row["pnl_after_costs"] for row in rows]
        notional = [row["notional"] for row in rows]
    out = []
    for p, n in zip(pnl, notional):
        out.append(float(p) / float(n) * 1e4 if float(n) else 0.0)
    return out


def headline_verdict(base: Mapping, slip: Mapping | None) -> dict:
    """Spec §1 steps 3–4 for one window: the bootstrap bar and the ×1.5 net rule.

    The decision number is the **empirical 0.0694th percentile** of the
    session-block bootstrap of mean net bps/trip at base, computed by
    :func:`screen.session_block_lower_bound` under the declared call sequence
    (seed :data:`screen.ROUND2_BOOTSTRAP_SEED`, :data:`screen.ROUND2_BOOTSTRAP_RESAMPLES`
    resamples).  ``z_bar`` is recorded and **not applied** (C7).  The ×1.5 rule
    is ``net_bps_per_trip > 0``; a missing ×1.5 record **fails closed**.

    The persisted headline fields are cross-checked, not trusted: a record whose
    ``bootstrap_block_p0.0694``/``B`` is not the one its own trip table
    re-derives, or which re-declares the seed, the resample count, ``z_bar`` or
    ``n_configs``, raises :class:`GateBlockInconsistent` (L19.8 / L19.12).
    """
    table = base.get(screen.ROUND2_TRIP_TABLE_KEY)
    if table is None:
        raise GateBlockInconsistent(
            f"{base.get('cell_id')!r} {base.get('window')!r}: the base record "
            f"carries no {screen.ROUND2_TRIP_TABLE_KEY!r}, so the headline's "
            f"lower bound cannot be re-derived (spec §3, L4)")
    try:
        lower, blocks = screen.session_block_lower_bound(table)
    except ValueError as exc:
        raise GateBlockInconsistent(
            f"{base.get('cell_id')!r} {base.get('window')!r}: the trip table "
            f"cannot produce the headline bootstrap ({exc})") from exc

    for key, expected in HEADLINE_DECLARED:
        if key in base and not _same_number(base[key], expected):
            raise GateBlockInconsistent(
                f"{base.get('cell_id')!r} {base.get('window')!r}: the record "
                f"declares {key}={base[key]!r}, but the pre-registration fixes "
                f"it at {expected!r} — a re-declared literal after the data is "
                f"seen is a forking path (L19.12, C7)")
    persisted_lb = base.get(HEADLINE_DERIVED[0])
    if persisted_lb is not None and not _isclose(persisted_lb, lower):
        raise GateBlockInconsistent(
            f"{base.get('cell_id')!r} {base.get('window')!r}: the persisted "
            f"{HEADLINE_DERIVED[0]}={persisted_lb!r} is not the {lower!r} the "
            f"trip table re-derives under the declared call sequence (L19.8) — "
            f"a persisted number that cannot be re-derived is not evidence")
    persisted_b = base.get(HEADLINE_DERIVED[1])
    if persisted_b is not None and not _same_number(persisted_b, blocks):
        raise GateBlockInconsistent(
            f"{base.get('cell_id')!r} {base.get('window')!r}: the persisted B="
            f"{persisted_b!r} is not the {blocks} distinct entry-date blocks the "
            f"trip table has (L19.9)")

    bps = _table_bps(table)
    base_mean = sum(bps) / len(bps) if bps else 0.0
    if "net_bps_per_trip" in base and not _isclose(base["net_bps_per_trip"],
                                                   base_mean, tol=1e-6):
        raise GateBlockInconsistent(
            f"{base.get('cell_id')!r} {base.get('window')!r}: net_bps_per_trip="
            f"{base['net_bps_per_trip']!r} is not the trip table's mean "
            f"{base_mean!r} — the record and its own trips disagree")

    detail = {
        "lower_bound": lower,
        "B": blocks,
        "net_bps_per_trip": base_mean,
        "percentile": screen.ROUND2_HEADLINE_PERCENTILE,
        "seed": screen.ROUND2_BOOTSTRAP_SEED,
        "resamples": screen.ROUND2_BOOTSTRAP_RESAMPLES,
        "z_bar_reference": screen.ROUND2_Z_BAR,
        "z_bar_applied": False,
        "n_configs": screen.ROUND2_N_CONFIGS,
        "trip_table_bps_mean": base_mean,
    }
    problems = []
    if not lower > 0:
        problems.append(f"base lower bound {lower:+.6f} bps not > 0 "
                        f"(0.0694th percentile of the session-block bootstrap)")
    if slip is None:
        problems.append(f"no {SLIP_COST} record for this window: the ×1.5 net "
                        f"rule fails closed (spec §1 step 4)")
        detail["net_bps_per_trip_x1_5"] = None
    else:
        x15 = float(slip.get("net_bps_per_trip", 0.0))
        detail["net_bps_per_trip_x1_5"] = x15
        table15 = slip.get(screen.ROUND2_TRIP_TABLE_KEY)
        if table15 is not None:
            bps15 = _table_bps(table15)
            mean15 = sum(bps15) / len(bps15) if bps15 else 0.0
            if not _isclose(x15, mean15, tol=1e-6):
                raise GateBlockInconsistent(
                    f"{base.get('cell_id')!r} {base.get('window')!r}: the ×1.5 "
                    f"net_bps_per_trip={x15!r} is not its own trip table's mean "
                    f"{mean15!r}")
            detail["x1_5_trip_table_bps_mean"] = mean15
        if not x15 > 0:
            problems.append(f"×1.5 net bps/trip {x15:+.6f} not > 0")
    why = "; ".join(problems) if problems else "ok"
    return _result(not problems, why, **detail)


# ── step 6: the matched benchmark gate (R3-6.1) ────────────────────────
def benchmark_verdict(base: Mapping, slip: Mapping | None) -> dict:
    """Spec §1 step 6 for one window: ``LB_delta > 0`` at base, ``delta_bps > 0``
    at ×1.5.

    Read **after** step 2 (which is where an unpriceable-EOD *shortfall* is
    decided) and after :func:`screen.assert_benchmark_paired_identity` closes
    both trip-count identities at **each** gating level — before any
    ``delta_bps`` is read.  Then the block is re-derived from its own published
    numbers:

    * ``mean(y_i)`` over the block's own ``per_trip`` rows equals
      ``benchmark_net_bps_per_trip`` (L19.10 / C5 — the per-trip equality is
      checkable from the block itself);
    * ``delta_bps == config_net_bps_per_trip − benchmark_net_bps_per_trip``
      exactly (R3-6.1 §1: no path ambiguity);
    * the per-symbol decomposition ``pooled = Σ_s (n_s/N)·delta_s`` (1e-9);
    * ``benchmark_cost_drag_per_trip_bps > 0`` at base — the **only** cost
      assertion, because ``benchmark_fills_per_trip`` and ``delta_seed`` are
      declarations (a constant compared with a constant is not an assertion,
      L19.10) and are recorded here, never scored.

    A multi-leg or trips=0 cell fails closed here; a missing ×1.5 record fails
    closed.  Zero-trip symbol rows are ``N/A`` and are not conjunctive.
    """
    blocks = {}
    for label, record in ((BASE_COST, base), (SLIP_COST, slip)):
        if record is None:
            continue
        block = record.get("benchmark")
        if not isinstance(block, Mapping) or not block:
            raise GateBlockInconsistent(
                f"{base.get('cell_id')!r} {base.get('window')!r}: the {label} "
                f"record carries no benchmark block — step 6 has nothing to read "
                f"(L16 / R3-6.1 §4)")
        if block.get("sizing") not in (None, screen.ROUND2_SIZING):
            raise GateBlockInconsistent(
                f"the benchmark block is declared at sizing {block.get('sizing')!r},"
                f" not the gating {screen.ROUND2_SIZING!r} (R3-3(ii))")
        if block.get("cost_label") not in (None, label):
            raise GateBlockInconsistent(
                f"the benchmark block declares cost_label "
                f"{block.get('cost_label')!r} inside the {label} record")
        screen.assert_benchmark_paired_identity(block)
        if "config_trips" in block and not _same_number(block["config_trips"],
                                                        record.get("trips")):
            raise GateBlockInconsistent(
                f"the benchmark block's config_trips={block['config_trips']!r} "
                f"is not the record's trips={record.get('trips')!r}: the "
                f"benchmark was not built from this run's own trips (R3-6.1 §1)")
        blocks[label] = block

    b = blocks[BASE_COST]
    if not _same_number(_require(b, "benchmark_fills_per_trip", "the benchmark "
                                 "block"), 2):
        # recorded, never scored (L19.10) — but a declaration that disagrees with
        # the declaration is worth saying out loud.
        pass
    y = [float(row["y_i"]) for row in (b.get("per_trip") or [])
         if row.get("y_i") is not None]
    paired = float(b.get("paired_trips", 0))
    if len(y) != int(paired):
        raise GateBlockInconsistent(
            f"the benchmark block's per_trip rows carry {len(y)} priced y_i "
            f"values but paired_trips={paired:g}: the per-trip equality cannot "
            f"be checked against the count it claims (L19.10 / C5)")
    mean_y = sum(y) / len(y) if y else 0.0
    if y and not _isclose(mean_y, _require(b, "benchmark_net_bps_per_trip",
                                          "the benchmark block")):
        raise GateBlockInconsistent(
            f"mean(y_i) over the block's own per_trip rows is {mean_y!r} but "
            f"benchmark_net_bps_per_trip is "
            f"{b['benchmark_net_bps_per_trip']!r} — the published benchmark "
            f"expectancy has drifted from the rows it claims to be the mean of "
            f"(F10 / L19.10)")
    delta = _num(_require(b, "delta_bps", "the benchmark block"), "delta_bps")
    if not _isclose(delta, float(b["config_net_bps_per_trip"]) - mean_y):
        raise GateBlockInconsistent(
            f"delta_bps={delta!r} is not config_net_bps_per_trip − "
            f"benchmark_net_bps_per_trip = "
            f"{float(b['config_net_bps_per_trip']) - mean_y!r} (R3-6.1 §1: "
            f"delta_bps = mean(x) − mean(y) exactly, no path ambiguity)")

    # per-symbol decomposition: pooled = Σ_s (n_s/N)·delta_s
    per_symbol = b.get("per_symbol")
    worst_symbol = None
    if per_symbol is None:
        raise GateBlockInconsistent(
            "the benchmark block carries no per_symbol table: the pooled delta "
            "is then an undecomposed number, which R3-6.1 §1 bans (the "
            "per-symbol table is the decomposition of the one gate number)")
    rows = list(per_symbol)
    counts = {row.get("symbol"): int(row.get("config_trips", 0) or 0)
              for row in rows}
    total_n = sum(counts.values())
    if total_n and not _same_number(total_n, b.get("config_trips")):
        raise GateBlockInconsistent(
            f"the per_symbol config_trips sum to {total_n} but the block's "
            f"config_trips is {b.get('config_trips')!r} (R3-6.1 §1)")
    pooled = 0.0
    for row in rows:
        n_s = int(row.get("config_trips", 0) or 0)
        if n_s == 0:
            if row.get("status") not in (None, "N/A"):
                raise GateBlockInconsistent(
                    f"the zero-trip symbol {row.get('symbol')!r} is written "
                    f"status={row.get('status')!r}; a symbol that could not "
                    f"trade is N/A, never a pass or a fail (R3-6.1 §2.3)")
            continue
        pooled += (n_s / total_n) * float(row.get("delta_bps", 0.0))
        if worst_symbol is None or float(row.get("delta_bps", 0.0)) < worst_symbol[1]:
            worst_symbol = (row.get("symbol"), float(row.get("delta_bps", 0.0)))
    if total_n and not _isclose(pooled, delta):
        raise GateBlockInconsistent(
            f"the per-symbol decomposition gives {pooled!r} but the pooled "
            f"delta_bps is {delta!r} — the per-symbol table is supposed to be "
            f"the decomposition of the one gate number (R3-6.1 §1, 1e-9)")

    cost_drag = _num(_require(b, "benchmark_cost_drag_per_trip_bps",
                              "the benchmark block"),
                     "benchmark_cost_drag_per_trip_bps")
    if not cost_drag > 0:
        raise GateBlockInconsistent(
            f"the base benchmark's cost drag is {cost_drag!r} bps/trip, not > 0: "
            f"a benchmark that pays no costs while the config pays is banned "
            f"(R3-6.1 §5.2 / §2.5)")

    detail = {
        "delta_bps": delta,
        "lb_delta": _num(_require(b, "delta_lower_bound_p0.0694",
                                  "the benchmark block"), "delta_lower_bound"),
        "config_net_bps_per_trip": float(b["config_net_bps_per_trip"]),
        "benchmark_net_bps_per_trip": mean_y,
        "paired_trips": int(paired),
        "benchmark_unpriced": float(b.get("benchmark_unpriced", 0.0)),
        "benchmark_fills_per_trip": b.get("benchmark_fills_per_trip"),
        "delta_seed": b.get("delta_seed"),
        "benchmark_cost_drag_per_trip_bps": cost_drag,
        "worst_symbol": worst_symbol[0] if worst_symbol else None,
        "worst_symbol_delta_bps": worst_symbol[1] if worst_symbol else None,
        "per_symbol": rows,
    }
    problems = []
    if not detail["lb_delta"] > 0:
        problems.append(f"base LB_delta {detail['lb_delta']:+.6f} not > 0")
    if slip is None:
        problems.append(f"no {SLIP_COST} record: the ×1.5 delta rule fails closed")
        detail["delta_bps_x1_5"] = None
    else:
        delta15 = _num(_require(blocks[SLIP_COST], "delta_bps",
                                "the ×1.5 benchmark block"), "delta_bps")
        detail["delta_bps_x1_5"] = delta15
        if not delta15 > 0:
            problems.append(f"×1.5 delta_bps {delta15:+.6f} not > 0")
    why = "; ".join(problems) if problems else "ok"
    return _result(not problems, why, **detail)


# ── step 7: the fat-tail (tail-dependence) gate, base only (R3-6.2) ─────
def fat_tail_verdict(record: Mapping) -> dict:
    """Spec §1 step 7 for one window: R1 > 0, R2 > 0, R3 ≤ 0.25 — base only.

    A **tail-dependence** rule, deliberately weaker than a fat-tail test (R3-6.2
    §1b): the reference is the two protocol constants ``0`` and ``0.25``, never a
    quantile or fitted statistic of the judged data.  The block's own ``rules``
    are re-derived from their recorded ``value`` and cross-checked against the
    published ``pass`` — a block whose verdict contradicts its numbers cannot be
    scored.  ``x1_5`` columns are reported and never gate; a survivor whose
    ×1.5 remainder is ≤ 0 is **flagged**.
    """
    block = record.get("fat_tail")
    if not isinstance(block, Mapping) or not block:
        raise GateBlockInconsistent(
            f"{record.get('cell_id')!r} {record.get('window')!r}: no fat_tail "
            f"block (R3-6.2 §5) — step 7 has nothing to read")
    status = block.get("status")
    if status != "ok":
        return _result(False, f"status {status!r}: {block.get('why')!r}",
                       status=status, not_evaluated=True)
    rules = block.get("rules")
    if not isinstance(rules, Mapping):
        raise GateBlockInconsistent("the fat_tail block carries no rules mapping")
    detail = {
        "k_trips_removed": block.get("k_trips_removed"),
        "remainder_after_best_k_bps": block.get("remainder_after_best_k_bps"),
        "remainder_after_best_month_bps": block.get("remainder_after_best_month_bps"),
        "worst_case_removed_month": block.get("worst_case_removed_month"),
        "best_session_share": block.get("best_session_share"),
        "worst_session_date": block.get("worst_session_date"),
        "worst_session_symbols": block.get("worst_session_symbols"),
        "x1_5": block.get("x1_5"),
        "label": "weaker — tail-dependence only",
    }
    problems = []
    derived: dict[str, bool] = {}
    for rule, (op, threshold) in FAT_TAIL_THRESHOLDS.items():
        entry = rules.get(rule)
        if not isinstance(entry, Mapping) or "value" not in entry:
            raise GateBlockInconsistent(
                f"the fat_tail block's rule {rule!r} carries no value: the rule "
                f"cannot be re-derived from its own numbers (R3-6.2 §5)")
        value = _num(entry["value"], f"fat_tail.rules.{rule}.value")
        ok = _rule_holds(op, value, threshold)
        if "pass" in entry and bool(entry["pass"]) != ok:
            raise GateBlockInconsistent(
                f"the fat_tail rule {rule!r} publishes pass={entry['pass']!r} "
                f"with value {value!r} against the literal {op} {threshold} — "
                f"the block's verdict contradicts its own numbers")
        derived[rule] = ok
        if not ok:
            problems.append(f"{rule}: {value:+.6f} not {op} {threshold}")
    conjunct = all(derived.values())
    if "pass" in block and bool(block["pass"]) != conjunct:
        raise GateBlockInconsistent(
            f"the fat_tail block publishes pass={block['pass']!r} but its three "
            f"rules are {derived}")
    if not float(block.get("total_net_pnl_usd", 0.0)) > 0:
        problems.append("total not positive — share undefined")
    detail["rules"] = {k: {"pass": v, "value": rules[k]["value"],
                           "threshold": FAT_TAIL_THRESHOLDS[k][1],
                           "comparison": FAT_TAIL_THRESHOLDS[k][0]}
                       for k, v in derived.items()}
    x15 = block.get("x1_5") or {}
    detail["x1_5_flagged"] = bool(
        isinstance(x15, Mapping)
        and float(x15.get("remainder_after_best_k_bps", 1.0)) <= 0)
    why = "; ".join(problems) if problems else "ok"
    return _result(not problems, why, **detail)


# ── step 8: the concentration gate, base only (R3-6.3) ─────────────────
def concentration_verdict(record: Mapping) -> dict:
    """Spec §1 step 8 for one window: ``max_share ≤ 0.50`` and ``n_s ≥ 25`` per
    traded symbol — base only.

    Dollars, not trips: ``share_s = P_s / Total`` over ``pnl_after_costs``, with
    the plain trip share published only as a diagnostic (R3-6.3 §1).  The whole
    table is re-derived from the block's own ``per_symbol`` rows: ``Σ n_s ==
    n_trips`` (so an ``N/A`` row cannot hide trips), ``Σ P_s == Total`` (so the
    share is a decomposition of the one headline number), the published
    ``max_symbol``/``max_symbol_share`` and both rule verdicts.  A zero-trip
    symbol is ``N/A`` — neither pass nor fail; ``Total ≤ 0`` fails closed.
    """
    block = record.get("concentration")
    if not isinstance(block, Mapping) or not block:
        raise GateBlockInconsistent(
            f"{record.get('cell_id')!r} {record.get('window')!r}: no "
            f"concentration block (R3-6.3 §5) — step 8 has nothing to read")
    status = block.get("status")
    if status != "ok":
        return _result(False, f"status {status!r}: {block.get('why')!r}",
                       status=status, not_evaluated=True)
    rows = block.get("per_symbol")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise GateBlockInconsistent(
            "the concentration block carries no per_symbol table: the share "
            "cannot be re-derived, and a share quoted without its table is a "
            "banned number (R3-6.3 §5)")
    rows = list(rows)
    n_trips = sum(int(row.get("trips", 0) or 0) for row in rows)
    if not _same_number(n_trips, block.get("n_trips")):
        raise GateBlockInconsistent(
            f"the per_symbol trips sum to {n_trips} but the block's n_trips is "
            f"{block.get('n_trips')!r} — an N/A row cannot hide trips "
            f"(R3-6.3 §2.2)")
    if not _same_number(n_trips, record.get("trips")):
        raise GateBlockInconsistent(
            f"the per_symbol trips sum to {n_trips} but the record ran "
            f"{record.get('trips')!r} trips")
    total = sum(float(row.get("net_pnl_usd", 0.0) or 0.0) for row in rows)
    if not _isclose(total, block.get("total_net_pnl_usd"), tol=1e-6):
        raise GateBlockInconsistent(
            f"the per_symbol net_pnl_usd sums to {total!r} but the block's "
            f"total_net_pnl_usd is {block.get('total_net_pnl_usd')!r} "
            f"(R3-6.3 §1)")
    if not _isclose(total, record.get("net_pnl"), tol=1e-6):
        raise GateBlockInconsistent(
            f"the concentration total {total!r} is not the record's net_pnl "
            f"{record.get('net_pnl')!r}: the share must decompose the one "
            f"headline number (R3-6.3 §1)")

    per_symbol = []
    traded = []
    max_row = None
    for row in rows:
        n_s = int(row.get("trips", 0) or 0)
        p_s = float(row.get("net_pnl_usd", 0.0) or 0.0)
        share = (p_s / total) if total > 0 else None
        published = row.get("net_pnl_share")
        if share is not None and published is not None and not _isclose(
                published, share):
            raise GateBlockInconsistent(
                f"symbol {row.get('symbol')!r}: published net_pnl_share "
                f"{published!r} is not P_s/Total = {share!r} (R3-6.3 §1)")
        if n_s >= 1:
            if row.get("status") not in (None, "ok"):
                raise GateBlockInconsistent(
                    f"symbol {row.get('symbol')!r} traded {n_s} trips but is "
                    f"written status={row.get('status')!r}")
            traded.append(row.get("symbol"))
            if max_row is None or (share or 0.0) > (max_row[1] or 0.0):
                max_row = (row.get("symbol"), share)
        else:
            if row.get("status") not in (None, "N/A"):
                raise GateBlockInconsistent(
                    f"the zero-trip symbol {row.get('symbol')!r} is written "
                    f"status={row.get('status')!r}: a symbol that could not "
                    f"trade is N/A, never a pass or a fail (R3-6.3 §2.2)")
            if p_s:
                raise GateBlockInconsistent(
                    f"the zero-trip symbol {row.get('symbol')!r} carries "
                    f"net_pnl_usd {p_s!r}")
        per_symbol.append({"symbol": row.get("symbol"), "trips": n_s,
                           "net_pnl_usd": p_s, "net_pnl_share": share,
                           "trip_share": (n_s / n_trips) if n_trips else None,
                           "status": "ok" if n_s >= 1 else "N/A"})

    max_share = max_row[1] if max_row else None
    min_traded = min((int(r.get("trips", 0) or 0) for r in rows
                      if int(r.get("trips", 0) or 0) >= 1), default=None)
    published_max = block.get("max_symbol_share")
    if max_share is not None and published_max is not None and not _isclose(
            published_max, max_share):
        raise GateBlockInconsistent(
            f"published max_symbol_share {published_max!r} is not the table's "
            f"{max_share!r} (R3-6.3 §1)")
    if max_row is not None and block.get("max_symbol") not in (None, max_row[0]):
        raise GateBlockInconsistent(
            f"published max_symbol {block.get('max_symbol')!r} is not the "
            f"table's {max_row[0]!r} (R3-6.3 §1)")

    rules = block.get("rules")
    if not isinstance(rules, Mapping):
        raise GateBlockInconsistent("the concentration block carries no rules")
    derived = {}
    if total <= 0:
        derived["share"] = False
    else:
        derived["share"] = _rule_holds("<=", float(max_share),
                                       CONCENTRATION_THRESHOLDS["share"][1])
    derived["min_trips"] = (min_traded is not None
                            and _rule_holds(">=", float(min_traded),
                                            CONCENTRATION_THRESHOLDS["min_trips"][1]))
    for rule, ok in derived.items():
        entry = rules.get(rule)
        if isinstance(entry, Mapping) and "pass" in entry and bool(entry["pass"]) != ok:
            raise GateBlockInconsistent(
                f"the concentration rule {rule!r} publishes pass="
                f"{entry['pass']!r} but the table re-derives {ok!r} — the "
                f"block's verdict contradicts its own numbers")
    conjunct = all(derived.values())
    if "pass" in block and bool(block["pass"]) != conjunct:
        raise GateBlockInconsistent(
            f"the concentration block publishes pass={block['pass']!r} but its "
            f"two rules are {derived}")

    problems = []
    if total <= 0:
        problems.append("total not positive — share undefined")
    elif not derived["share"]:
        problems.append(f"max share {max_share:.6f} > "
                        f"{CONCENTRATION_THRESHOLDS['share'][1]}")
    if not derived["min_trips"]:
        problems.append(f"thinnest traded symbol has {min_traded} trips < "
                        f"{CONCENTRATION_THRESHOLDS['min_trips'][1]:g}")
    detail = {
        "caps": block.get("caps"),
        "n_trips": n_trips,
        "total_net_pnl_usd": total,
        "max_symbol": max_row[0] if max_row else None,
        "max_symbol_share": max_share,
        "n_symbols_traded": len(traded),
        "traded_symbols": traded,
        "min_traded_symbol_trips": min_traded,
        "per_symbol": per_symbol,
        "rules": {k: {"pass": v, "threshold": CONCENTRATION_THRESHOLDS[k][1],
                      "comparison": CONCENTRATION_THRESHOLDS[k][0]}
                  for k, v in derived.items()},
    }
    why = "; ".join(problems) if problems else "ok"
    return _result(not problems, why, **detail)


# ── the per-cell scoring pass (spec §1 steps 2,3/4,6,7,8,5) ────────────
def score_cell(cell: str, base: Mapping, slip: Mapping | None,
               folds: Mapping[str, Mapping]) -> dict:
    """Score one cell's two gating records; **no bucketing** (that is pass two).

    Evaluation order is the spec's (preconditions → headline → benchmark →
    fat-tail → concentration → fold breadth), deliberately *not* the labelling
    order: the bucket is assigned later, in a second pass, from the same stage
    results (C1).  Every stage is evaluated even when an earlier one failed
    (L12) — a headline-failing cell is still scored by the other gates, and each
    gate's own rows are recorded as evidence (C9 / I3).
    """
    ok, why = cell_preconditions(base)
    pre_triple, headline_override = _precondition_stage(ok, why)
    slip_pre_ok, slip_pre_why = ((True, "ok") if slip is None
                                 else cell_preconditions(slip))
    if not slip_pre_ok:
        # C10: the paired floor applies at EACH gating level's benchmark block.
        pre_triple, override = _precondition_stage(False, slip_pre_why)
        if override is not None:
            headline_override = override
        else:
            pre_triple = (screen.ROUND2_PRECONDITION_STAGE, False,
                          f"{slip_pre_why} [at {SLIP_COST}]")
    if slip is not None:
        assert_cost_levels(base, slip)

    window = str(base.get("window"))
    if not pre_triple[1]:
        # A precondition refusal means the cell **refuses a verdict** (C2): the
        # four gates consume step 2's result, so there is nothing to score them
        # on.  They are recorded as not-evaluated with the slug's reason rather
        # than read off a record that has already been refused — and the bucket
        # pass returns ``not_evaluated`` before it walks the order.
        reason = f"not evaluated ({pre_triple[2]})"
        refused = {name: _result(False, reason, not_evaluated=True)
                   for name in ("headline", "benchmark", "fat_tail",
                                "concentration")}
        refused["preconditions"] = {"pass": False, "why": pre_triple[2],
                                    "detail": {"level": BASE_COST}}
        fold = folds.get(window) or {"pass": False,
                                     "why": "no fold-breadth result"}
        refused["fold_breadth"] = {"pass": bool(fold.get("pass")),
                                   "why": str(fold.get("why")),
                                   "detail": dict(fold)}
        return {"cell": cell, "window": window, "stages": refused,
                "residual": {k: None for k in RESIDUAL_FIELDS}}
    # L16 / L19.3: every gating record must carry the three blocks with content
    # (the ×1.5 record too — C10 reads the paired floor at each level).  This is
    # the landed assertion the acceptance criteria cite, called where the spec
    # puts it: before any gate is read, so a record whose block was never written
    # fails loudly instead of being scored on a `None`.
    screen.assert_gate_blocks_present(base, screen.ROUND2_GATE_BLOCKS)
    if slip is not None:
        screen.assert_gate_blocks_present(slip, screen.ROUND2_GATE_BLOCKS)
    stages = {
        "preconditions": {"pass": pre_triple[1], "why": pre_triple[2],
                          "detail": {"level": BASE_COST}},
        "headline": headline_verdict(base, slip),
        "benchmark": benchmark_verdict(base, slip),
        "fat_tail": fat_tail_verdict(base),
        "concentration": concentration_verdict(base),
    }
    if headline_override is not None:
        # L19.7 / C10: the paired-floor shortfall IS the headline failure.
        stages["headline"] = _result(False, headline_override[1],
                                     paired_floor=True,
                                     lower_bound=None,
                                     net_bps_per_trip=None)
    fold = folds.get(window) or {"pass": False, "why": "no fold-breadth result"}
    stages["fold_breadth"] = {"pass": bool(fold.get("pass")),
                              "why": str(fold.get("why")),
                              "detail": dict(fold)}
    residual = {k: (base.get("fat_tail") or {}).get(k) for k in RESIDUAL_FIELDS}
    return {"cell": cell, "window": window, "stages": stages,
            "residual": residual}


# ── step 11: ranking rows ──────────────────────────────────────────────
def ranked_row(cell: str, cell_detail: Mapping) -> dict:
    """One ``ranked`` row: the gate docs' required columns, from the verdict.

    Every field the three gate docs require on a ranked row (R3-6.1 §4,
    R3-6.2 §5, R3-6.3 §5) plus the five :func:`screen.rank_row` inputs, so the
    ranking is a function of the row rather than of a side channel.
    """
    w1 = cell_detail["windows"][screen.ROUND2_WINDOWS[0]]
    w2 = cell_detail["windows"][screen.ROUND2_WINDOWS[1]]
    b1, b2 = w1["records"]["base"], w2["records"]["base"]
    s1, s2 = w1["stages"], w2["stages"]
    net_pct = (float(b1.get("net_pct", 0.0)) + float(b2.get("net_pct", 0.0))) / 2.0
    row = {
        "config": cell, "cell": cell, "sizing": screen.ROUND2_SIZING,
        "net_pct_w1": float(b1.get("net_pct", 0.0)),
        "net_pct_w2": float(b2.get("net_pct", 0.0)),
        "net_pct_equal_weighted": net_pct,
        "net_bps_w1": float(b1.get("net_bps_per_trip", 0.0)),
        "net_bps_w2": float(b2.get("net_bps_per_trip", 0.0)),
        "net_bps_equal_weighted": (float(b1.get("net_bps_per_trip", 0.0))
                                   + float(b2.get("net_bps_per_trip", 0.0))) / 2.0,
        "trips_w1": int(b1.get("trips", 0)), "trips_w2": int(b2.get("trips", 0)),
        "t_stat_w1": float(b1.get("net_bps_t_stat", 0.0)),
        "t_stat_w2": float(b2.get("net_bps_t_stat", 0.0)),
        "max_drawdown_w1": float(b1.get("max_drawdown", 0.0)),
        "max_drawdown_w2": float(b2.get("max_drawdown", 0.0)),
        "max_drawdown_worst": min(float(b1.get("max_drawdown", 0.0)),
                                  float(b2.get("max_drawdown", 0.0))),
        "cost_drag_bps_w1": float(b1.get("cost_drag_per_trip_bps", 0.0)),
        "cost_drag_bps_w2": float(b2.get("cost_drag_per_trip_bps", 0.0)),
        "folds_positive_min": min(int(b1.get("folds_positive", 0)),
                                  int(b2.get("folds_positive", 0))),
        # headline (steps 3/4)
        "headline_lower_bound_w1": s1["headline"]["detail"].get("lower_bound"),
        "headline_lower_bound_w2": s2["headline"]["detail"].get("lower_bound"),
        "headline_B_w1": s1["headline"]["detail"].get("B"),
        "headline_B_w2": s2["headline"]["detail"].get("B"),
        "headline_net_bps_x1_5_w1": s1["headline"]["detail"].get("net_bps_per_trip_x1_5"),
        "headline_net_bps_x1_5_w2": s2["headline"]["detail"].get("net_bps_per_trip_x1_5"),
        # benchmark (step 6)
        "benchmark_pass": cell_detail["stages"]["benchmark"]["pass"],
        "benchmark_delta_bps_w1": s1["benchmark"]["detail"].get("delta_bps"),
        "benchmark_delta_bps_w2": s2["benchmark"]["detail"].get("delta_bps"),
        "benchmark_lb_delta_w1": s1["benchmark"]["detail"].get("lb_delta"),
        "benchmark_lb_delta_w2": s2["benchmark"]["detail"].get("lb_delta"),
        "benchmark_net_bps_w1": s1["benchmark"]["detail"].get("benchmark_net_bps_per_trip"),
        "benchmark_net_bps_w2": s2["benchmark"]["detail"].get("benchmark_net_bps_per_trip"),
        "benchmark_worst_symbol": _worst_symbol(s1, s2),
        # fat tail (step 7)
        "fat_tail_pass": cell_detail["stages"]["fat_tail"]["pass"],
        "fat_tail_k_trips_removed": s1["fat_tail"]["detail"].get("k_trips_removed"),
        "fat_tail_remainder_best_k_bps_w1": s1["fat_tail"]["detail"].get("remainder_after_best_k_bps"),
        "fat_tail_remainder_best_k_bps_w2": s2["fat_tail"]["detail"].get("remainder_after_best_k_bps"),
        "fat_tail_remainder_best_month_bps_w1": s1["fat_tail"]["detail"].get("remainder_after_best_month_bps"),
        "fat_tail_remainder_best_month_bps_w2": s2["fat_tail"]["detail"].get("remainder_after_best_month_bps"),
        "fat_tail_worst_month_w1": s1["fat_tail"]["detail"].get("worst_case_removed_month"),
        "fat_tail_worst_month_w2": s2["fat_tail"]["detail"].get("worst_case_removed_month"),
        "fat_tail_worst_session_share_w1": s1["fat_tail"]["detail"].get("best_session_share"),
        "fat_tail_worst_session_share_w2": s2["fat_tail"]["detail"].get("best_session_share"),
        "fat_tail_worst_session_date_w1": s1["fat_tail"]["detail"].get("worst_session_date"),
        "fat_tail_worst_session_date_w2": s2["fat_tail"]["detail"].get("worst_session_date"),
        "fat_tail_x1_5_flagged": bool(s1["fat_tail"]["detail"].get("x1_5_flagged")
                                      or s2["fat_tail"]["detail"].get("x1_5_flagged")),
        # concentration (step 8)
        "concentration_pass": cell_detail["stages"]["concentration"]["pass"],
        "concentration_max_symbol_w1": s1["concentration"]["detail"].get("max_symbol"),
        "concentration_max_symbol_w2": s2["concentration"]["detail"].get("max_symbol"),
        "concentration_max_symbol_share_w1": s1["concentration"]["detail"].get("max_symbol_share"),
        "concentration_max_symbol_share_w2": s2["concentration"]["detail"].get("max_symbol_share"),
        "concentration_min_traded_symbol_trips_w1": s1["concentration"]["detail"].get("min_traded_symbol_trips"),
        "concentration_min_traded_symbol_trips_w2": s2["concentration"]["detail"].get("min_traded_symbol_trips"),
        "concentration_n_symbols_traded_w1": s1["concentration"]["detail"].get("n_symbols_traded"),
        "concentration_n_symbols_traded_w2": s2["concentration"]["detail"].get("n_symbols_traded"),
        "concentration_caps": dict(families.R2_CAPS),
        # fold breadth + neighbour (steps 5, 9)
        "fold_breadth_pass": cell_detail["stages"]["fold_breadth"]["pass"],
        "neighbour_pass": cell_detail["stages"]["neighbour"]["pass"],
        "neighbour_why": cell_detail["stages"]["neighbour"]["why"],
        # the residual (reported, never gating)
        "residual": cell_detail["residual"],
    }
    row["rank"] = list(screen.rank_row(row))
    return row


def _worst_symbol(*stage_details) -> str | None:
    worst = None
    for stages in stage_details:
        detail = stages["benchmark"]["detail"]
        symbol = detail.get("worst_symbol")
        delta = detail.get("worst_symbol_delta_bps")
        if symbol is None or delta is None:
            continue
        if worst is None or float(delta) < worst[1]:
            worst = (symbol, float(delta))
    return worst[0] if worst else None


# ── the judge: steps 0–11, both passes ─────────────────────────────────
def judge_round2(records: Iterable[Mapping], specs: Sequence[dict] | None = None,
                 windows: Sequence[str] = screen.ROUND2_WINDOWS,
                 clone_census: Mapping | None = None) -> dict:
    """Score a family's round-2 records and return the ranked verdict.

    ``records`` are the **untrimmed per-run record files** (one mapping per
    (cell, window, gating level)); ``specs`` is the declared grid of that
    family, used by the neighbour rule only (:func:`families.GRIDS` when
    omitted).  ``clone_census`` is recorded verbatim as a **census artifact**
    and is never read by a verdict path: clone discovery excludes cells from the
    neighbour set (inside :func:`screen.neighbour_cells`) and changes nothing
    else — ``N = 36`` and ``z_bar = 3.196`` are fixed for the round-2 verdict
    (L19.12).

    Returns the verdict payload: per-cell buckets and stage results (pass one),
    the three gate docs' per-window rejection lists verbatim (C9), the ranked
    survivors, and the residual.
    """
    cost_levels.assert_gate_cost_labels()
    declaration = screen.assert_round2_declaration()      # N = 36, z_bar fixed
    index = index_records(records)
    if not index:
        raise RecordSetError("no round-2 records were handed in")
    family = str(next(iter(index.values())).get("family"))
    if specs is None:
        specs = families.GRIDS[family]
    cells = sorted({cell for (cell, _w, _c, _s) in index})

    # ── step 1: record selection by sizing (fail closed, L1) ────────────
    gating: dict[str, dict] = {}
    refused: dict[str, str] = {}
    for cell in cells:
        base: dict[str, Mapping] = {}
        slip: dict[str, Mapping] = {}
        for window in windows:
            b = _gating(index, cell, window, BASE_COST)
            s = _gating(index, cell, window, SLIP_COST)
            if b is None:
                if _cell_windows(index, cell) & {window}:
                    refused[cell] = (
                        f"sizing: no {screen.ROUND2_SIZING!r} record for "
                        f"{window} — only a reported sizing exists, and a "
                        f"reported sizing cannot gate (L1 / R3-3(ii))")
                    break
                raise screen.GridCellFailure(
                    f"round-2 cell {cell!r} has no run record for {window}: a "
                    f"cell that failed to run on either window is a hard error, "
                    f"never a silent drop (R3-3(iii)) — the grid did not run")
            base[window], slip[window] = b, s
        if cell in refused:
            continue
        gating[cell] = {"base": base, "slip": slip}

    # ── step 5's producer, at base cost, over the whole grid (L6) ───────
    base_runs = {(cell, window): rec
                 for cell, levels in gating.items()
                 for window, rec in levels["base"].items()}
    breadth = screen.round2_fold_breadth(base_runs) if base_runs else {"cells": {}}

    # ── pass one: every gate, every (cell, window), both levels (L12) ───
    scored: dict[str, dict] = {}
    for cell, levels in gating.items():
        per_window = {}
        for window in windows:
            folds = breadth["cells"].get(cell, {}).get("windows", {})
            per_window[window] = score_cell(cell, levels["base"][window],
                                            levels["slip"].get(window), folds)
            per_window[window]["records"] = {
                "base": levels["base"][window],
                "slip": levels["slip"].get(window),
            }
        stages = {}
        for stage in ("headline", "benchmark", "fat_tail", "concentration"):
            failures = [(w, per_window[w]["stages"][stage]["why"])
                        for w in windows
                        if not per_window[w]["stages"][stage]["pass"]]
            stages[stage] = {
                "pass": not failures,
                "why": "; ".join(f"{w}: {why}" for w, why in failures) or "ok",
                "windows": {w: per_window[w]["stages"][stage] for w in windows},
            }
        pre_windows = {w: per_window[w]["stages"]["preconditions"] for w in windows}
        pre_failures = [(w, pre_windows[w]["why"]) for w in windows
                        if not pre_windows[w]["pass"]]
        stages["preconditions"] = {
            "pass": not pre_failures,
            "why": "; ".join(f"{w}: {why}" for w, why in pre_failures) or "ok",
            "windows": pre_windows,
        }
        fold = breadth["cells"].get(cell, {"pass": False,
                                           "why": "no fold-breadth result",
                                           "windows": {}})
        stages["fold_breadth"] = {
            "pass": bool(fold.get("pass")), "why": str(fold.get("why")),
            "windows": fold.get("windows", {}),
        }
        scored[cell] = {"cell": cell, "windows": per_window, "stages": stages,
                        "residual": per_window[windows[0]]["residual"]}

    # ── step 9: the neighbour rule, after every cell is scored ──────────
    positive_both = {}
    for cell, detail in scored.items():
        positive_both[cell] = all(
            float(detail["windows"][w]["records"]["base"].get(
                "net_bps_per_trip", 0.0)) > 0 for w in windows)
    for cell, detail in scored.items():
        ok, why = screen.round2_neighbour_stage(family, specs, cell, positive_both)
        detail["stages"]["neighbour"] = {
            "pass": bool(ok), "why": str(why),
            "detail": {"positive_both": dict(positive_both),
                       "rule": "a cell is positive on both windows when its "
                               "base-cost net_bps_per_trip > 0 on each"}}

    # ── pass two: the bucket, in the LABELLING order (L19.6 / C1) ───────
    verdict_cells: dict[str, dict] = {}
    for cell in cells:
        if cell in refused:
            verdict_cells[cell] = {
                "cell": cell, "bucket": screen.ROUND2_NOT_EVALUATED,
                "why": f"{cell}: not evaluated ({refused[cell]})",
                "stages": {"preconditions": {"pass": False, "why": refused[cell]}},
                "windows": {}, "residual": {}, "refused": True,
            }
            continue
        detail = scored[cell]
        stages = detail["stages"]
        triples = [
            _triple(screen.ROUND2_PRECONDITION_STAGE, stages["preconditions"]),
            _triple("headline", stages["headline"]),
            _triple("benchmark", stages["benchmark"]),
            _triple("fat_tail", stages["fat_tail"]),
            _triple("concentration", stages["concentration"]),
            _triple("fold_breadth", stages["fold_breadth"]),
            _triple("neighbour", stages["neighbour"]),
        ]
        bucket, why = screen.round2_bucket(cell, *triples)
        verdict_cells[cell] = {
            "cell": cell, "bucket": bucket, "why": why,
            "stages": stages, "windows": detail["windows"],
            "residual": detail["residual"], "refused": False,
        }

    # ── the two verdict shapes (C9): buckets AND per-window rows ────────
    cells_by_bucket: dict[str, list[str]] = {
        name: [] for name in (screen.ROUND2_ALIVE,) + screen.ROUND2_BUCKET_NAMES
    }
    cells_by_bucket[screen.ROUND2_NOT_EVALUATED] = []
    for cell in cells:
        cells_by_bucket[verdict_cells[cell]["bucket"]].append(cell)

    headline_rejected, benchmark_rejected = [], []
    fat_tail_rejected, concentration_rejected = [], []
    fold_breadth_rejected, neighbour_rejected = [], []
    for cell in cells:
        detail = verdict_cells[cell]
        if detail["refused"]:
            continue
        for window in windows:
            stages = detail["windows"][window]["stages"]
            if not stages["headline"]["pass"]:
                d = stages["headline"]["detail"]
                headline_rejected.append({
                    "config": cell, "window": window,
                    "lower_bound": d.get("lower_bound"),
                    "net_bps_x1_5": d.get("net_bps_per_trip_x1_5"),
                    "B": d.get("B"), "why": stages["headline"]["why"]})
            if not stages["benchmark"]["pass"]:
                d = stages["benchmark"]["detail"]
                benchmark_rejected.append({
                    "config": cell, "window": window,
                    "delta_bps": d.get("delta_bps"),
                    "lb_delta": d.get("lb_delta"),
                    "why": stages["benchmark"]["why"]})
            if not stages["fat_tail"]["pass"]:
                rules = stages["fat_tail"]["detail"].get("rules") or {}
                failed = [(rule, entry) for rule, entry in rules.items()
                          if not entry["pass"]] or [("status", {"value": None,
                                                                "threshold": None})]
                for rule, entry in failed:
                    fat_tail_rejected.append({
                        "config": cell, "window": window, "rule_failed": rule,
                        "value": entry.get("value"),
                        "threshold": entry.get("threshold"),
                        "why": stages["fat_tail"]["why"]})
            if not stages["concentration"]["pass"]:
                rules = stages["concentration"]["detail"].get("rules") or {}
                failed = [(rule, entry) for rule, entry in rules.items()
                          if not entry["pass"]]
                if not failed:
                    failed = [("status", {"value": None, "threshold": None})]
                for rule, entry in failed:
                    concentration_rejected.append({
                        "config": cell, "window": window, "rule_failed": rule,
                        "value": entry.get("value"),
                        "threshold": entry.get("threshold"),
                        "why": stages["concentration"]["why"]})
            if not stages["fold_breadth"]["pass"]:
                fold_breadth_rejected.append({
                    "config": cell, "window": window,
                    "folds_positive": stages["fold_breadth"]["detail"].get(
                        "folds_positive"),
                    "why": stages["fold_breadth"]["why"]})
        if not detail["stages"]["neighbour"]["pass"]:
            neighbour_rejected.append({
                "config": cell, "why": detail["stages"]["neighbour"]["why"]})

    # ── step 11: rank, and only the living are ranked ───────────────────
    ranked = [ranked_row(cell, verdict_cells[cell]) for cell in cells
              if verdict_cells[cell]["bucket"] == screen.ROUND2_ALIVE]
    ranked.sort(key=lambda row: tuple(row["rank"]), reverse=True)

    return {
        "family": family,
        "judge_version": ROUND2_JUDGE_VERSION,
        "windows": list(windows),
        "gating_costs": list(GATING_COSTS),
        "gating_sizing": screen.ROUND2_SIZING,
        "n_configs": int(declaration["n_configs"]),
        "z_bar": float(declaration["z_bar"]),
        "z_bar_role": "normal-approximation reference for the same bar; applied "
                      "by no gate (C7) — the decision number is the empirical "
                      f"{screen.ROUND2_HEADLINE_PERCENTILE}th percentile",
        "bootstrap": {"seed": screen.ROUND2_BOOTSTRAP_SEED,
                      "resamples": screen.ROUND2_BOOTSTRAP_RESAMPLES,
                      "percentile": screen.ROUND2_HEADLINE_PERCENTILE},
        "B_annotations": list(screen.ROUND2_B_ANNOTATIONS),
        "cells": verdict_cells,
        "buckets": {cell: verdict_cells[cell]["bucket"] for cell in cells},
        "cells_by_bucket": cells_by_bucket,
        "alive": cells_by_bucket[screen.ROUND2_ALIVE],
        "not_evaluated": cells_by_bucket[screen.ROUND2_NOT_EVALUATED],
        "dead": [cell for cell in cells
                 if verdict_cells[cell]["bucket"] not in (screen.ROUND2_ALIVE,
                                                          screen.ROUND2_NOT_EVALUATED)],
        "ranked": ranked,
        "verdict": "SURVIVORS" if ranked else "NO_SURVIVOR",
        "positive_both": positive_both,
        "headline_rejected": headline_rejected,
        "benchmark_rejected": benchmark_rejected,
        "fat_tail_rejected": fat_tail_rejected,
        "concentration_rejected": concentration_rejected,
        "fold_breadth_rejected": fold_breadth_rejected,
        "neighbour_rejected": neighbour_rejected,
        "residual": {"sentence": RESIDUAL_SENTENCE, "fields": list(RESIDUAL_FIELDS),
                     "gating": False},
        "clone_census": dict(clone_census) if clone_census is not None else None,
        "clone_census_role": "census artifact, not a judge input: clone discovery "
                             "excludes cells from the neighbour set only and "
                             "never re-declares N, z_bar or any other input "
                             "(L19.12)",
    }
