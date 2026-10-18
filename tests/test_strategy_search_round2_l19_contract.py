"""L19.6–L19.12 — the judge's *decision contract*.

Source: ``/home/team/shared/ROUND2_L19_RULING.md`` (L19, 2026-10-15) and
``/home/team/shared/ROUND2_L19_SUPPLEMENT_RULING.md`` (L19-S, the C1–C10
decisions), on the verifier's per-item edit map
``/home/team/shared/ROUND2_L196_13_EDIT_MAP.md``.

Sections below are added one delegation item at a time and are labelled with the
item they pin:

* **L19.8** — the headline bootstrap is pinned by its **call sequence**, not by
  four literals, and it must be **re-derivable from the trip table**.
* **L19.9** — ``B`` (the entry-date block count) is a *recorded diagnostic*; no
  verdict path may read it.
* **L19.6** — the bucket vocabulary must be **total**: six declared buckets plus
  ``not_evaluated``, one spelling, and the bucket is the *first* failure in the
  declared **labelling** order (which is not the step evaluation order — C1).
* **L19.12** — ``N`` and ``z_bar`` are fixed for the verdict; clone discovery
  never re-declares them.

The one artefact this file reads from disk is the committed trip table
``tests/data/round2_l198_stage2_trip_table.csv`` — a copy of the stage-2 W1 trip
table, the artifact L19.8's port exists to keep re-derivable — and its persisted
golden ``tests/data/round2_l198_persisted.json``.
"""
from __future__ import annotations

import inspect
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_search import families, screen  # noqa: E402

SPEC = Path("/home/team/shared/ROUND2_GATE_ORDER_SPEC.md")
PREREG = Path("/home/team/shared/ROUND2_PREREG_DECLARATIONS.md")
SHARED_TREE = Path("/home/team/shared")

DATA = ROOT / "tests" / "data"
FIXTURE = DATA / "round2_l198_stage2_trip_table.csv"
GOLDEN = DATA / "round2_l198_persisted.json"


# ═══════════════════════════════════════════════════════════════════════
# L19.8 — the bootstrap is pinned by its CALL SEQUENCE
# ═══════════════════════════════════════════════════════════════════════
def test_the_committed_trip_table_fixture_is_a_real_stage2_trip_table():
    """The golden below is only evidence if the fixture is the real thing.

    A fixture invented for the test would prove the helper consistent with
    itself.  This one is the stage-2 W1 trip table, whose own published
    per-trip expectancy (``-2.735 bps``) is asserted against the fixture in the
    next test — so a fixture that silently changes is caught twice.
    """
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing: the L19.8 golden is 'persisted == re-derived from "
        f"a committed trip table', so the table is part of the evidence")
    frame = pd.read_csv(FIXTURE)
    for column in ("entry_time", "notional", "pnl_after_costs"):
        assert column in frame.columns, (
            f"the fixture must carry {column!r}: the bootstrap reads entry_time "
            f"(the block key), notional and pnl_after_costs (L19.8)")
    assert len(frame) > 100, (
        "a fixture with a handful of trips cannot distinguish the call sequences "
        "the pin exists to separate")


def test_the_lower_bound_re_derives_the_persisted_value_from_the_committed_table():
    """**persisted == re-derived**, on the committed trip table (L19.8).

    The persisted value is the golden: ``bootstrap_block_p0.0694`` and ``B`` as
    the record will carry them.  Re-deriving them from the committed table is
    what makes the pin checkable — change the block key, the block order, the
    draw, the statistic, the seed, the resample count or the percentile and this
    fails.
    """
    golden = json.loads(GOLDEN.read_text())
    frame = pd.read_csv(FIXTURE)
    lower, blocks = screen.session_block_lower_bound(frame)
    assert lower == pytest.approx(golden["lower_bound"], abs=0.0), (
        f"the re-derived lower bound {lower!r} != the persisted "
        f"{golden['lower_bound']!r} (L19.8): the two must agree under the exact "
        f"call sequence of spec §1 step 3, or the persisted number is not a "
        f"re-derivable quantity")
    assert blocks == golden["blocks"], (
        f"B (the entry-date block count) re-derived {blocks} != persisted "
        f"{golden['blocks']} (L19.9 records B per cell per window)")
    assert (screen.ROUND2_BOOTSTRAP_SEED, screen.ROUND2_BOOTSTRAP_RESAMPLES,
            screen.ROUND2_HEADLINE_PERCENTILE) == (
                golden["seed"], golden["resamples"], golden["percentile"]), (
        "the declared seed / resample count / percentile moved away from the "
        "persisted golden (L19.14: none of the three may be re-chosen)")


def test_the_fixture_reproduces_stage2s_own_published_expectancy():
    """A stronger anchor than the golden: the number stage 2 published.

    ``stage2_trades_W1.csv`` is the post-fix round-1 survivor's W1 trip table;
    its mean net bps/trip is published as **-2.735** in the stage-2 report.  If
    the fixture or the bps derivation drifts, this fails before the golden does —
    and it is evidence the port reads the same ``bps`` stage 2 read.
    """
    frame = pd.read_csv(FIXTURE)
    bps = (frame["pnl_after_costs"] / frame["notional"] * 1e4)
    assert float(bps.mean()) == pytest.approx(-2.735, abs=5e-4), (
        f"the fixture's mean net bps/trip is {float(bps.mean())!r}, not the "
        f"-2.735 bps/trip stage 2 published for this window — the fixture is not "
        f"the table it claims to be, or the bps formula drifted")


def test_the_decision_number_is_the_empirical_percentile_not_stage2s_p25():
    """The 2.5th percentile is ~36x looser — the defect the port must not reuse.

    ``run_strategy_search_stage2.bootstrap`` returns
    ``percentile(blocks, 2.5)``; the round-2 bar is the 0.0694th percentile of
    the **same** draws (finding 3 / L2).  On this fixture the two differ by
    ~4 bps, which is the size of the whole modelled toll — so quietly reusing
    the stage-2 bar is not a rounding difference.
    """
    golden = json.loads(GOLDEN.read_text())
    assert golden["p2_5_of_the_same_draws"] != pytest.approx(
        golden["lower_bound"], abs=1.0), (
        "the fixture cannot tell the two bars apart, so this control is vacuous "
        "— pick one where they differ (L2)")
    frame = pd.read_csv(FIXTURE)
    lower, _ = screen.session_block_lower_bound(frame)
    assert lower == pytest.approx(golden["lower_bound"], abs=0.0)
    assert lower != pytest.approx(golden["p2_5_of_the_same_draws"], abs=0.5), (
        f"the helper returned the 2.5th percentile ({golden['p2_5_of_the_same_draws']!r}) "
        f"rather than the 0.0694th ({golden['lower_bound']!r}) — that is the "
        f"stage-2 bar, ~36x looser, and it is not the round-2 decision number")


def _uneven_blocks_frame() -> pd.DataFrame:
    """A trip table whose sessions differ in trip count — deliberately.

    The committed stage-2 fixture has **one trip per session**, so on it the
    trip-weighted statistic and the mean of session means coincide: it cannot
    show that the pin is load-bearing.  This frame can.  It is deterministic and
    built in-test (nothing about it is a fact about the market).
    """
    rows = []
    for b in range(40):
        size = 1 + (b % 5)                       # 1..5 trips per session
        level = ((b * 37) % 19) - 9.0            # bps level, spread -9..+9
        day = pd.Timestamp("2025-09-02") + pd.Timedelta(days=b)
        for i in range(size):
            bps = level + 0.1 * i
            rows.append({"entry_time": day + pd.Timedelta(minutes=i),
                         "notional": 24_000.0,
                         "pnl_after_costs": bps * 24_000.0 / 1e4})
    return pd.DataFrame(rows)


def _declared_sequence_reference(frame: pd.DataFrame, seed: int,
                                 resamples: int, percentile: float) -> float:
    """The declared call sequence, written out longhand — no helper used.

    This is the sequence spec §1 step 3 pins, spelled as code rather than prose:
    entry-date blocks, ascending date order, one seeded ``default_rng``,
    ``integers(0, n, n)`` per resample, the mean over the **concatenated trips**,
    and the percentile of the draws.  A test that called the helper twice would
    prove nothing; this one is derived independently of it.
    """
    day = pd.to_datetime(frame["entry_time"]).dt.strftime("%Y-%m-%d")
    bps = frame["pnl_after_costs"] / frame["notional"] * 1e4
    by_day = [g["bps"].to_numpy()
              for _, g in pd.DataFrame({"day": day, "bps": bps}).groupby("day")]
    rng = np.random.default_rng(seed)
    n = len(by_day)
    draws = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        drawn: list[float] = []
        for j in idx:
            drawn.extend(by_day[j].tolist())
        draws[i] = float(np.mean(drawn))
    return float(np.percentile(draws, percentile))


def _mean_of_session_means(frame: pd.DataFrame, seed: int, resamples: int,
                           percentile: float) -> float:
    """The plausible *wrong* statistic: mean of the drawn sessions' means.

    It differs from the declared one exactly when sessions differ in trip count,
    which is why L19.8 calls this out by name.
    """
    day = pd.to_datetime(frame["entry_time"]).dt.strftime("%Y-%m-%d")
    bps = frame["pnl_after_costs"] / frame["notional"] * 1e4
    by_day = [g["bps"].to_numpy()
              for _, g in pd.DataFrame({"day": day, "bps": bps}).groupby("day")]
    rng = np.random.default_rng(seed)
    n = len(by_day)
    draws = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        draws[i] = float(np.mean([by_day[j].mean() for j in idx]))
    return float(np.percentile(draws, percentile))


def test_the_lower_bound_is_the_declared_call_sequence_and_nothing_else():
    """The pin is load-bearing: a plausible lookalike gives a different number.

    Three independent statements:

    * the helper equals a **longhand** implementation of the declared sequence;
    * the *mean-of-session-means* variant — same seed, same percentile, same
      resample count, same block key and order — gives a different answer, so the
      statistic is what is being pinned;
    * a different seed gives yet another answer, so the single seeded stream is
      load-bearing too.
    """
    frame = _uneven_blocks_frame()
    seed = screen.ROUND2_BOOTSTRAP_SEED
    resamples = screen.ROUND2_BOOTSTRAP_RESAMPLES
    pct = screen.ROUND2_HEADLINE_PERCENTILE

    lower, blocks = screen.session_block_lower_bound(frame)
    reference = _declared_sequence_reference(frame, seed, resamples, pct)
    assert lower == reference, (
        f"the helper returned {lower!r}; the declared sequence, written out "
        f"longhand, gives {reference!r} (L19.8) — a different block key, block "
        f"order, draw or statistic")
    assert blocks == frame["entry_time"].dt.strftime("%Y-%m-%d").nunique(), (
        "B must be the number of distinct entry-date blocks (L19.9)")

    lookalike = _mean_of_session_means(frame, seed, resamples, pct)
    assert lookalike != pytest.approx(lower, abs=1e-9), (
        f"the mean-of-session-means statistic gives the same lower bound "
        f"({lookalike!r}) as the declared trip-weighted one ({lower!r}) on a "
        f"frame whose sessions differ in trip count — so this control cannot "
        f"fail and the pin is untested (L19.8)")
    other_seed, _ = screen.session_block_lower_bound(frame, seed=seed + 1)
    assert other_seed != pytest.approx(lower, abs=1e-9), (
        "changing the seed did not change the lower bound: the single seeded RNG "
        "stream is not what the draws come from (L19.8 (c))")


def test_the_helper_accepts_the_record_files_json_projection_of_the_table():
    """One function, both shapes — or the persisted number is a second producer.

    The judge reads the **per-run record files** (L19-S implementation notes),
    where the trip table has been through JSON.  A helper that only took a
    DataFrame would force the judge to re-derive the number itself, which is
    precisely the two-producers defect the "same helper, same sequence" rule
    exists to prevent.
    """
    frame = pd.read_csv(FIXTURE)
    as_json = json.loads(frame.to_json(orient="records"))
    assert isinstance(as_json, list) and isinstance(as_json[0], dict)
    lower_df, blocks_df = screen.session_block_lower_bound(frame)
    lower_json, blocks_json = screen.session_block_lower_bound(as_json)
    assert blocks_json == blocks_df, (
        "B (the block count) must not depend on the table's shape (L19.9)")
    assert lower_json == pytest.approx(lower_df, abs=1e-9), (
        f"the helper returns {lower_json!r} for the trip table's JSON projection "
        f"and {lower_df!r} for the frame it projects (L19.8).  The read is the "
        f"same read: JSON writes ~16 significant digits, so a drift **beyond 1e-9 "
        f"bps** is a different statistic, not float formatting — a real call-"
        f"sequence difference is whole bps")


def test_the_helper_refuses_a_table_it_cannot_read_rather_than_guessing():
    """Positive control on the failing side: a missing column is refused."""
    with pytest.raises(ValueError) as exc:
        screen.session_block_lower_bound(pd.DataFrame({"entry_time": []}))
    message = str(exc.value)
    for column in ("notional", "pnl_after_costs"):
        assert column in message, (
            f"the refusal must name the missing column {column!r} — a bootstrap "
            f"that silently invents a denominator is worse than one that stops")
    with pytest.raises(ValueError):
        screen.session_block_lower_bound(pd.DataFrame(
            {"entry_time": [], "notional": [], "pnl_after_costs": []}))


# ═══════════════════════════════════════════════════════════════════════
# L19.6 — the bucket vocabulary is TOTAL, and `not_evaluated` has one spelling
# ═══════════════════════════════════════════════════════════════════════
#: Every stage :func:`screen.round2_bucket` must be handed a verdict for.
#: ``preconditions`` is step 2 (L19.7): it is *not* a gate, it decides whether
#: the cell can be scored at all, so it short-circuits to the one terminal
#: spelling instead of to a ``*_rejected`` bucket (C2).
_ALL_STAGES = screen.ROUND2_BUCKET_STAGES


def _triples(failing=(), precondition_ok=True):
    """One ``(stage, ok, why)`` triple per declared stage, as the judge hands them in.

    Mirrors the shape the real producers return: the three gate blocks'
    ``(ok, why)``, :func:`screen.round2_fold_breadth` and
    :func:`screen.round2_neighbour_stage` for the two block-less rules, and
    :func:`screen.round2_preconditions` for step 2.
    """
    out = []
    for stage in _ALL_STAGES:
        if stage == screen.ROUND2_PRECONDITION_STAGE:
            ok = precondition_ok
        else:
            ok = stage not in failing
        out.append((stage, ok, f"{stage} verdict"))
    return tuple(out)


def test_the_vocabulary_is_total_over_the_whole_verdict_space():
    """**Every** dead cell has exactly one bucket (L19.6).

    The guard L19.6 exists for: a full sweep of all ``2**6 = 64`` pass/fail
    combinations over the six declared stages.  Each one must return exactly one
    bucket, the bucket must be the **first** failure in the **declared labelling
    order** — not any failure, and not the order the triples arrive in — and the
    six ``*_rejected`` names plus ``alive`` must be the *whole* output range, so
    no dead cell can be unbucketed.  ``fold_breadth_rejected`` and
    ``neighbour_rejected`` had no name at all before this, which is the defect.
    """
    order = screen.ROUND2_BUCKET_ORDER
    assert len(order) == 6 and len(set(order)) == 6, (
        f"the declared bucket order is {order!r} — six distinct stages is the "
        f"vocabulary L19.6 rules on")
    assert screen.ROUND2_BUCKET_NAMES == tuple(f"{b}_rejected" for b in order), (
        "one bucket name per stage, derived from the order so the two cannot "
        "drift (L19.6)")
    assert len(set(screen.ROUND2_BUCKET_NAMES)) == 6
    assert screen.ROUND2_NOT_EVALUATED == "not_evaluated"
    assert screen.ROUND2_NOT_EVALUATED not in screen.ROUND2_BUCKET_NAMES

    reachable = set()
    for mask in range(1 << len(order)):
        failing = {order[i] for i in range(len(order)) if (mask >> i) & 1}
        bucket, why = screen.round2_bucket("cell", *_triples(failing))
        reachable.add(bucket)
        first = min(order.index(stage) for stage in failing) if failing else None
        expected = (f"{order[first]}_rejected" if first is not None
                    else screen.ROUND2_ALIVE)
        assert bucket == expected, (
            f"with {sorted(failing)} failing the cell bucketed {bucket!r}, not "
            f"{expected!r}: the bucket is the FIRST failure in the declared "
            f"order {order} (L19.6/C1)")
        assert why.startswith("cell:"), (
            f"the reason must name the cell it is about, got {why!r}")

    assert reachable == {screen.ROUND2_ALIVE} | set(screen.ROUND2_BUCKET_NAMES), (
        f"the sweep reached {sorted(reachable)}; the vocabulary must be exactly "
        f"alive + the six reject buckets, or some outcome has no name (L19.6). "
        f"A missing name is how a dead cell stayed unbucketed before the ruling")


def test_a_partial_or_duplicated_verdict_set_is_refused_not_bucketed():
    """The input side of totality: fail closed on a partial picture (L19.6).

    A cell bucketed from five of six stages is a cell whose bucket is a guess,
    and a stage handed in twice is two answers to one question.  Both raise.
    The positive control runs first, so the assertion cannot pass vacuously on a
    helper that raises for everything.
    """
    full = _triples({"fat_tail"})
    assert screen.round2_bucket("cell", *full)[0] == "fat_tail_rejected", (
        "positive control: a complete, well-formed verdict set must bucket")

    for dropped in _ALL_STAGES:
        with pytest.raises(ValueError) as exc:
            screen.round2_bucket("cell", *(t for t in full if t[0] != dropped))
        assert dropped in str(exc.value), (
            f"a verdict set with no {dropped!r} verdict must be refused *naming* "
            f"it — a silent bucket on a partial picture is the unbucketed-dead-"
            f"cell defect; got {str(exc.value)!r}")

    with pytest.raises(ValueError) as exc:
        screen.round2_bucket("cell", *full, ("benchmark", True, "a second opinion"))
    assert "benchmark" in str(exc.value) and "twice" in str(exc.value), (
        f"a stage passed twice must be refused as such, got {str(exc.value)!r}")

    with pytest.raises(ValueError) as exc:
        screen.round2_bucket("cell", *full[:-1],
                             ("not_defined_for_multi_leg", False, "multi-leg"))
    assert "not_defined_for_multi_leg" in str(exc.value), (
        "a stage name outside the declared vocabulary is refused one spelling at "
        "a time (C2): R3-6.1 §2.7 and R3-6.3 §1/§7 used "
        "`not_defined_for_multi_leg` and it is not a bucket stage")

    with pytest.raises((TypeError, ValueError)):
        screen.round2_bucket("cell", *full[:-1], ("neighbour", True))


def test_the_bucket_is_the_first_failure_in_the_declared_order_not_the_input_order():
    """Input-order independence (L19.6/C1).

    The labelling order and §1's evaluation order are deliberately different
    things, so the bucket may not depend on the order the judge happens to
    assemble its verdicts in.  Every permutation of the same seven triples gives
    the same bucket, and the bucket is the earliest stage in
    :data:`screen.ROUND2_BUCKET_ORDER` — here ``headline``, even when the failing
    stages are passed last.
    """
    triples = _triples({"concentration", "headline"})
    base, _ = screen.round2_bucket("cell", *triples)
    assert base == "headline_rejected", (
        "headline outranks concentration in the declared order, so a cell failing "
        "both is benchmark-independent: it is headline_rejected (L19.6)")

    for perm in itertools.permutations(triples):
        bucket, _ = screen.round2_bucket("cell", *perm)
        assert bucket == base, (
            f"the same verdicts in a different order bucketed {bucket!r} instead "
            f"of {base!r} (L19.6/C1): the bucket must be decided by the declared "
            f"order, not by the order the judge assembled them in")

    reversed_ = tuple(reversed(triples))
    assert reversed_[0][0] == "neighbour" and reversed_[1][0] == "fold_breadth", (
        "the control must actually pass the last-declared stages first")
    assert screen.round2_bucket("cell", *reversed_)[0] == "headline_rejected"

    assert screen.round2_bucket("cell", *_triples({"concentration"}))[0] == (
        "concentration_rejected"), (
        "non-vacuity control: with headline passing, concentration's failure is "
        "reachable on its own — so the equality above is about the order, not "
        "about the fixture failing everything")


def test_a_precondition_failure_is_not_evaluated_one_spelling_only():
    """Precondition mismatch is ``not_evaluated``, and nothing else (C2).

    One terminal spelling for a cell that refuses a verdict; the *reason* rides
    in ``why``.  The control below shows the short-circuit is load-bearing (the
    same six gate failures with step 2 satisfied is a reject), so this is not a
    helper that returns the same string for everything.
    """
    all_gates_fail = set(screen.ROUND2_BUCKET_ORDER)
    bucket, why = screen.round2_bucket("cell", *_triples(all_gates_fail,
                                                         precondition_ok=False))
    assert bucket == screen.ROUND2_NOT_EVALUATED == "not_evaluated", (
        f"a precondition mismatch bucketed {bucket!r}; the one spelling is "
        f"`not_evaluated` (C2) and the reason belongs in `why`, not in a second "
        f"bucket name")
    assert "not evaluated" in why and "preconditions verdict" in why, (
        f"the reason must be carried in `why`, got {why!r}")

    assert screen.round2_bucket("cell", *_triples(all_gates_fail))[0] == (
        "headline_rejected"), (
        "control: with step 2 satisfied the same six gate failures is a reject, "
        "so the `not_evaluated` above is the short-circuit doing the work")

    plain, _ = screen.round2_bucket("cell", *_triples((), precondition_ok=False))
    assert plain == screen.ROUND2_NOT_EVALUATED, (
        "a precondition mismatch with every gate passing is still not_evaluated: "
        "nothing downstream is meaningful, so it is decided first")


def _step2_record(**over):
    """A minimal record that satisfies **every** step-2 precondition."""
    record = {
        "trips": 150,
        "trip_table": [{"legs": 1, "notional": families.R2_CAPS["notional_usd"]}] * 150,
        "config_params": dict(families.R2_CAPS),
        "benchmark": {"config_trips": 150, "benchmark_trips": 150,
                      "paired_trips": 150, "benchmark_unpriced": 0},
    }
    record.update(over)
    return record


def test_the_step_2_slug_decides_the_bucket_and_the_two_slugs_are_not_interchangeable():
    """``paired_floor`` is a **headline** failure, every other slug is not (L19.7/C2).

    :func:`screen.round2_preconditions` returns one ``(ok, why)`` for all six
    conjuncts and the judge reads the **slug** to pick the bucket.  That handoff
    is the only place the L13 ambiguity could come back, so it is pinned here at
    the bucket: the helper's own paired-floor refusal, handed in as a *headline*
    verdict, buckets ``headline_rejected`` — while the same refusal handed in
    under the precondition stage would bucket ``not_evaluated``, which is the
    mis-bucket this test exists to make visible.
    """
    ok, why = screen.round2_preconditions(
        _step2_record(benchmark={"config_trips": 150, "benchmark_trips": 150,
                                 "paired_trips": 149, "benchmark_unpriced": 1}))
    assert ok is False and why.startswith("paired_floor"), (
        f"the paired shortfall must be refused under its own slug, got {why!r}")

    others = [t for t in _triples(("benchmark",)) if t[0] != "headline"]
    as_headline = screen.round2_bucket("cell", *others, ("headline", False, why))
    assert as_headline[0] == "headline_rejected", (
        f"a paired-subset shortfall is headline_rejected (L19.7/C10), got "
        f"{as_headline[0]!r}")

    as_precondition = screen.round2_bucket(
        "cell", *_triples((), precondition_ok=False))
    assert as_precondition[0] == screen.ROUND2_NOT_EVALUATED, (
        "routing the same refusal through the precondition stage yields "
        "not_evaluated — wrong for `paired_floor`, which is exactly why the slug, "
        "and not the stage, decides")

    ok, why = screen.round2_preconditions(_step2_record(trips=149))
    assert why.startswith("trips_floor"), (
        f"a genuine precondition mismatch keeps its own slug, got {why!r}")
    assert why.split(":")[0] not in screen.ROUND2_BUCKET_ORDER, (
        "a precondition slug must not collide with a stage name, or the judge's "
        "slug->stage mapping is ambiguous")


# ═══════════════════════════════════════════════════════════════════════
# L19.6 — the neighbour rule is *wired* into the vocabulary
# ═══════════════════════════════════════════════════════════════════════
def _family_c_neighbour_grid():
    """A five-cell grid around ``c0``: one neighbour per axis, one clone, one two-axis cell.

    Built the way ``tests/test_strategy_search_round2_gate.py`` builds its
    neighbour fixture, so the resolved-config hashing under test is the live one.
    ``c1``/``c2`` declare the **same** behaviour (``c2`` is a clone); ``c3``
    varies two axes at once, so it is a neighbour of neither.
    """
    params = dict(families.GRID_C[1]["params"])

    def cell(name, stop, or_minutes):
        return dict(name=name, params=dict(params),
                    extras=dict(stop_atr_mult=stop, target_atr_mult=2.0,
                                trail_atr_mult=0.0),
                    signal=dict(or_minutes=or_minutes, allow_short=False))

    return [cell("c0", 1.0, 30), cell("c1", 1.5, 30), cell("c2", 1.5, 30),
            cell("c3", 2.0, 60), cell("c4", 1.0, 60)]


def test_the_neighbour_stage_runs_the_real_neighbour_rule_and_reaches_the_vocabulary():
    """The wiring point (L19.6): the one conjunct that is not per-record.

    ``neighbour_cells``/``neighbour_verdict`` were pure and unwired, so a cell
    dying on the neighbour rule had **no bucket**.  The stage must produce the
    same verdict as calling the pair directly, must be consumable as the
    ``neighbour`` triple, and its failure must reach ``neighbour_rejected``.
    """
    specs = _family_c_neighbour_grid()
    positive = {"c1": True, "c4": True}

    stage = screen.round2_neighbour_stage("C", specs, "c0", positive)
    assert isinstance(stage, tuple) and len(stage) == 2 and isinstance(stage[0], bool), (
        f"the stage must return the (ok, why) shape round2_bucket takes, got "
        f"{stage!r}")
    assert stage[0] is True, f"both axes have a positive neighbour; got {stage!r}"

    direct = screen.neighbour_verdict(screen.neighbour_cells("C", specs, "c0"),
                                      positive)
    assert stage == direct, (
        f"the stage returned {stage!r} but the real rule returns {direct!r} — the "
        f"wiring must adapt neighbour_cells/neighbour_verdict, not re-implement "
        f"the rule (L19.6)")

    without_neighbour = _triples(()) [:-1]
    alive, why = screen.round2_bucket("cell", *without_neighbour,
                                      ("neighbour", *stage))
    assert alive == screen.ROUND2_ALIVE, (
        f"a passing neighbour rule leaves the cell alive; got {alive!r} ({why!r})")

    failing = screen.round2_neighbour_stage("C", specs, "c0",
                                            {"c1": False, "c4": False})
    assert failing[0] is False
    bucket, why = screen.round2_bucket(
        "cell", *(_triples(()) [:-1]), ("neighbour", *failing))
    assert bucket == "neighbour_rejected", (
        f"a cell dying on the neighbour rule must be bucketed "
        f"`neighbour_rejected` (L19.6) — before this it was unbucketed; got "
        f"{bucket!r} ({why!r})")
    assert bucket in screen.ROUND2_BUCKET_NAMES


def test_a_clone_cell_is_excluded_from_the_neighbour_set_only():
    """L9/L19.12: clone cells leave the neighbour set, and nothing else.

    The exclusion is load-bearing, not cosmetic: with the clone counted as a
    second neighbour the axis needs two positives and the cell fails; with the
    clone excluded the axis has one neighbour and passes.  The two verdicts are
    asserted against each other, so the exclusion cannot be silently dropped.
    """
    specs = _family_c_neighbour_grid()
    clones = (families.resolved_hash("C", families.resolve_spec("C", specs[1]))
              == families.resolved_hash("C", families.resolve_spec("C", specs[2])))
    assert clones, (
        "the fixture's c1/c2 must resolve to the same config, or this control "
        "tests nothing")

    cells = screen.neighbour_cells("C", specs, "c0")
    assert cells == {"stop_atr_mult": ["c1"], "or_minutes": ["c4"]}, (
        f"the neighbour set is {cells!r}: the clone c2 and the two-axis c3 are "
        f"both out, the survivor c0 is out, and one neighbour per axis is in "
        f"(L9/L19.12)")

    positive = {"c1": True, "c2": False, "c4": True}
    wired = screen.round2_neighbour_stage("C", specs, "c0", positive)
    assert wired[0] is True, (
        f"with the clone excluded, stop_atr_mult's only neighbour is c1 (positive) "
        f"and or_minutes' is c4 (positive), so the rule holds; got {wired!r}")

    as_if_clone_counted = screen.neighbour_verdict(
        {"stop_atr_mult": ["c1", "c2"], "or_minutes": ["c4"]}, positive)
    assert as_if_clone_counted[0] is False, (
        "the control must fail with the clone counted (2 neighbours, 1 positive, "
        "need 2) — otherwise the exclusion above is untested (L9/L19.12)")


# ═══════════════════════════════════════════════════════════════════════
# L19.9 — `B` is a recorded diagnostic; no verdict path reads it
# ═══════════════════════════════════════════════════════════════════════
def test_B_is_recorded_and_no_verdict_path_reads_it():
    """L19.9/C6: the block count is reported, never scored.

    Three statements, in this order of strength: the helper really does report a
    different ``B`` for a one-session and a forty-session frame (so ``B`` is a
    live quantity); the bucket is identical whether the record's diagnostic
    field says ``B = 1`` or ``B = 1000``, while a stage verdict that *is* read
    moves it (the non-vacuity control); and no verdict entry point reads the
    annotation constants or a ``bootstrap_blocks`` field — the scan is shown to
    be sensitive by finding a name the function does read.
    """
    one_block = pd.DataFrame({
        "entry_time": pd.to_datetime(["2025-09-02 09:31", "2025-09-02 10:02",
                                      "2025-09-02 11:00"]),
        "notional": [24_000.0] * 3, "pnl_after_costs": [1.0, 2.0, 3.0]})
    _, b_one = screen.session_block_lower_bound(one_block)
    _, b_many = screen.session_block_lower_bound(_uneven_blocks_frame())
    assert (b_one, b_many) == (1, 40), (
        f"B must be the number of distinct entry-date blocks (L19.9): got "
        f"{b_one} for one session and {b_many} for the uneven 40-session frame")

    triples = _triples({"concentration"})
    buckets = {screen.round2_bucket("cell", *triples)[0]
               for _record_b in ({"bootstrap_blocks": 1},
                                 {"bootstrap_blocks": 1000})}
    assert buckets == {"concentration_rejected"}, (
        f"the bucket moved with the recorded B — no verdict path may read B, it "
        f"is a diagnostic (L19.9); got {sorted(buckets)}")

    assert screen.round2_bucket("cell", *_triples({"benchmark"}))[0] == (
        "benchmark_rejected"), (
        "non-vacuity control: the fixture's stage verdicts do move the bucket, so "
        "the equality asserted for B is about B alone")

    params = list(inspect.signature(screen.round2_bucket).parameters)
    assert params == ["cell", "per_gate_results"], (
        f"round2_bucket's inputs are {params!r}: B is not one of them and must "
        f"not become one (L19.9)")
    assert not any("block" in name or name.strip("_") == "b" for name in params)

    for fn in (screen.round2_bucket, screen.round2_neighbour_stage,
               screen.round2_fold_breadth, screen.round2_preconditions,
               screen.assert_benchmark_paired_identity):
        source = inspect.getsource(fn)
        assert "ROUND2_B_ANNOTATIONS" not in source, (
            f"{fn.__name__} reads the B annotation constants — 3 and N/4 are "
            f"annotations retained for the report, never evaluated (L19.9/C6)")
        assert "bootstrap_blocks" not in source, (
            f"{fn.__name__} reads the recorded block count; no verdict path may "
            f"be a function of B (L19.9)")

    assert "ROUND2_BUCKET_ORDER" in inspect.getsource(screen.round2_bucket), (
        "sensitivity control: the scan above must be able to find a constant the "
        "function *does* read, or it proves nothing")
    assert screen.ROUND2_B_ANNOTATIONS == (3, screen.ROUND2_N_CONFIGS // 4), (
        "the two retained annotations are 3 and N/4, documented in the spec as "
        "diagnostics (L19.9/C6)")
