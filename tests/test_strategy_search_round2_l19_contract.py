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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_search import screen  # noqa: E402

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
