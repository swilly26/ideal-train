"""The ROUND-2 JUDGE — synthetic fixture records only (no screens, no runs).

Source: ``/home/team/shared/ROUND2_GATE_ORDER_SPEC.md`` §1 (evaluation order) and
§2 (I1–I7, L12–L16), ``ROUND2_L19_RULING.md``, ``ROUND2_L19_SUPPLEMENT_RULING.md``
(C1–C10), and the three gate definitions
``ROUND2_BENCHMARK_GATE_R3-6.1.md`` / ``ROUND2_FAT_TAIL_GATE_R3-6.2.md`` /
``ROUND2_CONCENTRATION_GATE_R3-6.3.md``.

**What this file is.** The judge
(:mod:`src.backtesting.strategy_search.judge`) turns the untrimmed per-run
records into ranked verdicts.  The gate *producers* (the code that writes the
``benchmark``/``fat_tail``/``concentration`` blocks into a run record) are a
different build; here every block is a **fixture**, computed from a synthetic
trip table by the arithmetic in the three gate docs, so the judge's decision
paths can be driven exactly — one gate failing at a time — without a screen, a
144-record run, or a frozen grid.

**The standard each test is held to.** Every gate has a **reject path** and an
**alive path**, and each fixture is built so the *other* gates pass: a benchmark
failure has a positive headline and a spread concentration; a concentration
failure has every other rule satisfied.  That is what makes a constant stub fail
— always-True fails the reject tests, always-False fails the alive test.  The
guards that read a persisted number (the headline's re-derivation, the
benchmark's per-trip mean, the two rule tables, the cost levels) each get a test
that breaks the number and watches the judge refuse; the mutation battery in
``docs/ROUND2_JUDGE_MUTATIONS.txt`` is the other half of that evidence.

The one artefact read from disk is the residual ruling's **verbatim sentence**
(``ROUND2_FAT_TAIL_RESIDUAL_RULING.md`` §6), because a paraphrase in the code is
not the sentence the ruling requires beside every survivor.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_search import (  # noqa: E402
    costs as cost_levels,
    families,
    judge,
    screen,
)

SHARED = Path("/home/team/shared")
RESIDUAL_RULING = SHARED / "ROUND2_FAT_TAIL_RESIDUAL_RULING.md"

NOTIONAL = families.R2_CAPS["notional_usd"]
UNIVERSE = tuple(families.R2_UNIVERSE)
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
TRIPS_PER_MONTH = 14
N_TRIPS = len(MONTHS) * TRIPS_PER_MONTH          # 168, > the 150-trip floor
WINDOW_DATES = {"W1": "2025", "W2": "2024"}


# ═══════════════════════════════════════════════════════════════════════
# the fixture builder — a trip table, and the three blocks computed from it
# ═══════════════════════════════════════════════════════════════════════
def _session(month: str, i: int, year: str) -> tuple[str, str]:
    """One session per trip: ``i`` trips into ``month`` land on distinct days."""
    month_num = int(month.split("-")[1])
    day = 2 + i
    entry = f"{year}-{month_num:02d}-{day:02d} 09:32:00"
    exit_ = f"{year}-{month_num:02d}-{day:02d} 15:30:00"
    return entry, exit_


def trip_table(bps: list[float], year: str, per_session: int = 4) -> list[dict]:
    """The L4 trip table: one row per trip, one session per trip.

    ``bps[i]`` is trip ``i``'s net bps; ``pnl_after_costs`` is derived from it at
    the pinned notional, so the table's own mean **is** the record's
    ``net_bps_per_trip`` and the headline bootstrap is re-derivable from it.

    ``per_session`` trips share one session (the pinned cap is 4), so ``B`` — the
    entry-date block count the bootstrap resamples — is ``N / per_session``.
    """
    rows = []
    for i, value in enumerate(bps):
        month = MONTHS[i // TRIPS_PER_MONTH]
        entry, exit_ = _session(month, (i % TRIPS_PER_MONTH) // per_session, year)
        pnl = value / 1e4 * NOTIONAL
        rows.append({
            "instrument": UNIVERSE[i % len(UNIVERSE)],
            "symbols": UNIVERSE[i % len(UNIVERSE)],
            "legs": 1,
            "side": 1,
            "notional": NOTIONAL,
            "entry_time": entry,
            "signal_time": entry,
            "exit_time": exit_,
            "net_bps": value,
            "pnl_after_costs": pnl,
            "pnl_gross": pnl,
            "exit_reason": "eod",
        })
    return rows


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _fat_tail_block(rows: list[dict], cost: str, x1_5=None) -> dict:
    """R3-6.2 §1 arithmetic, computed from the table (no shortcuts).

    ``k = max(10, ceil(0.05N))``; R1 = the leave-one-month-out mean that hurts
    most; R2 = the mean of the N−k lowest bps (ties broken by pnl desc, exit asc,
    instrument asc); R3 = the largest session's share of the window's dollars.
    """
    n = len(rows)
    k = max(10, math.ceil(0.05 * n))
    by_bps = sorted(rows, key=lambda r: (-float(r["net_bps"]),
                                         -float(r["pnl_after_costs"]),
                                         str(r["exit_time"]),
                                         str(r["instrument"])))
    removed, kept = by_bps[:k], by_bps[k:]
    r2 = _mean(float(r["net_bps"]) for r in kept)
    months = sorted({str(r["exit_time"])[:7] for r in rows})
    r1, worst_month = None, None
    for month in months:
        rest = [float(r["net_bps"]) for r in rows
                if str(r["exit_time"])[:7] != month]
        value = _mean(rest) if rest else float("-inf")
        if r1 is None or value < r1:
            r1, worst_month = value, month
    sessions: dict[str, float] = {}
    symbols_by_session: dict[str, set] = {}
    for r in rows:
        day = str(r["entry_time"])[:10]
        sessions[day] = sessions.get(day, 0.0) + float(r["pnl_after_costs"])
        symbols_by_session.setdefault(day, set()).add(r["instrument"])
    total = sum(sessions.values())
    worst_session = max(sessions, key=lambda d: sessions[d]) if sessions else None
    share = (sessions[worst_session] / total) if total > 0 and worst_session else 0.0
    best_trip = max((float(r["pnl_after_costs"]) for r in rows), default=0.0)
    rules = {
        "best_month": {"pass": r1 > 0, "value": r1},
        "trim": {"pass": r2 > 0, "value": r2},
        "session": {"pass": share <= 0.25, "value": share},
    }
    block = {
        "version": "R3-6.2", "window": None, "config_name_resolved": None,
        "resolved_hash": None, "sizing": screen.ROUND2_SIZING,
        "cost_label": cost,
        "n_trips": n, "n_sessions": len(sessions), "n_months": len(months),
        "k_trips_removed": k, "k_formula": "max(10, ceil(0.05N))",
        "removed_fraction": k / n, "removal_cutoff_bps": float(kept[0]["net_bps"]) if kept else None,
        "total_net_pnl_usd": total, "net_bps_per_trip": _mean(float(r["net_bps"]) for r in rows),
        "remainder_after_best_k_bps": r2,
        "remainder_after_best_10_bps": _mean(
            float(r["net_bps"]) for r in sorted(
                rows, key=lambda r: -float(r["net_bps"]))[10:]),
        "best_k_trips_pnl_usd": sum(float(r["pnl_after_costs"]) for r in removed),
        "worst_case_removed_month": worst_month,
        "remainder_after_best_month_bps": r1,
        "best_month_by_total_pnl": worst_month,
        "remainder_after_that_month_bps": r1,
        "worst_session_date": worst_session,
        "worst_session_symbols": len(symbols_by_session.get(worst_session, ())),
        "worst_session_net_pnl_usd": sessions.get(worst_session, 0.0),
        "best_session_share": share,
        "best_single_trip_share": (best_trip / total) if total > 0 else None,
        "notional_uniform": True, "total_matches_record": True,
        "x1_5": x1_5 or {"remainder_after_best_k_bps": r2,
                         "remainder_after_best_month_bps": r1,
                         "best_session_share": share},
        "status": "ok",
        "pass": all(rule["pass"] for rule in rules.values()) and total > 0,
        "why": "ok",
        "rules": rules,
    }
    return block


def _concentration_block(rows: list[dict], cost: str) -> dict:
    """R3-6.3 §1 arithmetic: dollar shares per symbol, zero-trip rows ``N/A``."""
    per_symbol, traded = [], []
    total = sum(float(r["pnl_after_costs"]) for r in rows)
    n = len(rows)
    for symbol in UNIVERSE:
        mine = [r for r in rows if r["instrument"] == symbol]
        p_s = sum(float(r["pnl_after_costs"]) for r in mine)
        share = (p_s / total) if total > 0 else None
        per_symbol.append({
            "symbol": symbol, "trips": len(mine), "net_pnl_usd": p_s,
            "net_pnl_share": share,
            "trip_share": len(mine) / n if n else None,
            "mean_net_bps": _mean(float(r["net_bps"]) for r in mine),
            "status": "ok" if mine else "N/A",
            "why": None if mine else "no trips",
        })
        if mine:
            traded.append((symbol, len(mine), share))
    priced = [(sym, sh) for sym, _n, sh in traded if sh is not None]
    max_symbol, max_share = (max(priced, key=lambda t: t[1])
                             if priced else (None, None))
    min_trips = min((n_s for _s, n_s, _sh in traded), default=None)
    rules = {
        "share": {"pass": total > 0 and max_share is not None and max_share <= 0.50,
                  "value": max_share, "threshold": 0.50, "comparison": "<="},
        "min_trips": {"pass": min_trips is not None and min_trips >= 25,
                      "value": min_trips, "threshold": 25, "comparison": ">="},
    }
    return {
        "version": "R3-6.3", "window": None, "config_name_resolved": None,
        "resolved_hash": None, "sizing": screen.ROUND2_SIZING,
        "cost_label": cost,
        "caps": dict(families.R2_CAPS),
        "symbol_priority": "universe_insertion_order",
        "universe": list(UNIVERSE), "n_trips": n, "total_net_pnl_usd": total,
        "symbol_unit": "single_leg_instrument_key==symbol", "symbol_unit_ok": True,
        "shares_are_dollar_based": True, "decomposition_max_abs_residual": 0.0,
        "max_symbol": max_symbol, "max_symbol_share": max_share,
        "max_symbol_net_pnl_usd": (max_share * total) if max_share is not None else None,
        "n_symbols_traded": len(traded),
        "traded_symbols": [s for s, _n, _sh in traded],
        "min_traded_symbol": min(((s, n_s) for s, n_s, _sh in traded),
                                 key=lambda t: t[1])[0] if traded else None,
        "min_traded_symbol_trips": min_trips,
        "per_symbol": per_symbol, "rules": rules, "status": "ok",
        "pass": all(rule["pass"] for rule in rules.values()),
        "why": "ok",
    }


def _benchmark_block(rows: list[dict], delta_bps: float, lb_delta: float,
                    cost: str = cost_levels.BASE_COST) -> dict:
    """R3-6.1 §1: one matched benchmark trip per config trip, paired identity.

    ``y_i = config_bps_i − delta_bps`` so ``delta_bps = mean(x) − mean(y)``
    exactly, and the block's own ``per_trip`` rows are what the per-trip equality
    is checked against (L19.10 / C5).
    """
    per_trip = [{"entry_time": r["entry_time"],
                 "qty": NOTIONAL / 100.0,
                 "y_i": float(r["net_bps"]) - delta_bps,
                 "symbol": r["instrument"], "side": int(r["side"])}
                for r in rows]
    bench_mean = _mean(row["y_i"] for row in per_trip)
    per_symbol = []
    for symbol in UNIVERSE:
        mine = [r for r in rows if r["instrument"] == symbol]
        per_symbol.append({
            "symbol": symbol, "config_trips": len(mine),
            "benchmark_trips": len(mine), "paired": len(mine),
            "config_net_bps": _mean(float(r["net_bps"]) for r in mine),
            "benchmark_net_bps": _mean(float(r["net_bps"]) - delta_bps for r in mine),
            "delta_bps": delta_bps if mine else None,
            "delta_lower_bound_p0.0694": lb_delta if mine else None,
            "status": "ok" if mine else "N/A",
            "why": None if mine else "no trips",
        })
    n = len(rows)
    return {
        "variant": "matched_long_same_entry", "source": "realised_trips",
        "eod_flat_min": 930,
        "entry_convention": "t+1 open, config's own entry bar, buy",
        "exit_convention": "close of first session bar >= eod_flat_min, sell",
        "cost_label": cost, "sizing": screen.ROUND2_SIZING,
        "notional_usd": NOTIONAL,
        "config_trips": n, "benchmark_trips": n, "paired_trips": n,
        "benchmark_unpriced": 0,
        "config_net_bps_per_trip": _mean(float(r["net_bps"]) for r in rows),
        "benchmark_net_bps_per_trip": bench_mean,
        "delta_bps": delta_bps,
        "delta_lower_bound_p0.0694": lb_delta,
        "delta_resamples": screen.ROUND2_BOOTSTRAP_RESAMPLES,
        "delta_seed": screen.ROUND2_BOOTSTRAP_SEED,
        "config_cost_drag_per_trip_bps": 4.24,
        "benchmark_cost_drag_per_trip_bps": 4.24,
        "benchmark_fills_per_trip": 2,
        "pass": lb_delta > 0 and delta_bps > 0, "why": "ok",
        "per_symbol": per_symbol, "per_trip": per_trip,
    }


def build_record(cell: str, window: str, cost: str, bps: list[float],
                 *, delta_bps: float = 2.0, lb_delta: float = 1.0,
                 per_session: int = 4) -> dict:
    """One per-run record, in the shape ``Runner.run`` writes (JSON-safe)."""
    year = WINDOW_DATES[window]
    rows = trip_table(bps, year, per_session)
    lower, blocks = screen.session_block_lower_bound(rows)
    folds = []
    for month in sorted({str(r["exit_time"])[:7] for r in rows}):
        mine = [float(r["net_bps"]) for r in rows if str(r["exit_time"])[:7] == month]
        folds.append({"month": month, "round_trips": len(mine),
                      "net_bps_per_trip": _mean(mine),
                      "pnl_after_costs": sum(float(r["pnl_after_costs"])
                                             for r in rows
                                             if str(r["exit_time"])[:7] == month)})
    net_pnl = sum(float(r["pnl_after_costs"]) for r in rows)
    sd = (lambda v: (sum((x - _mean(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
          if len(v) > 1 else 0.0)([float(r["net_bps"]) for r in rows])
    fat_tail = _fat_tail_block(rows, cost)
    fat_tail["window"] = window
    concentration = _concentration_block(rows, cost)
    concentration["window"] = window
    benchmark = _benchmark_block(rows, delta_bps, lb_delta, cost)
    return {
        "family": "E1", "cell_id": cell, "config": cell,
        "config_name_resolved": f"E1|{cell}|{cost}",
        "window": window, "window_dates": (
            ["2025-09-01", "2026-09-01"] if window == "W1"
            else ["2024-09-01", "2025-09-01"]),
        "cost_label": cost, "cost_model": cost_levels.cost_model(cost).as_dict(),
        "sizing": screen.ROUND2_SIZING,
        "engine_sha": "0" * 40, "resolved_hash": "a" * 64,
        "identity_key": f"E1|{cell}|{screen.ROUND2_SIZING}",
        "config_params": dict(families.R2_CAPS),
        "notional_usd": NOTIONAL,
        "trips": len(rows),
        "net_pnl": net_pnl,
        "net_pct": net_pnl / families.R2_CAPS["initial_equity"],
        "max_drawdown": -0.10,
        "net_bps_per_trip": _mean(float(r["net_bps"]) for r in rows),
        "net_bps_per_trip_sd": sd,
        "net_bps_t_stat": (_mean(float(r["net_bps"]) for r in rows) / sd
                           * math.sqrt(len(rows))) if sd else 0.0,
        "cost_drag_per_trip_bps": 4.24,
        "folds_positive": sum(1 for f in folds if f["net_bps_per_trip"] > 0
                              and f["round_trips"] >= screen.MIN_TRIPS_PER_MONTH),
        "folds_total": len(folds),
        "folds": folds,
        "per_symbol": [],
        "trip_table": rows,
        "bootstrap_block_p0.0694": lower,
        "bootstrap_seed": screen.ROUND2_BOOTSTRAP_SEED,
        "bootstrap_resamples": screen.ROUND2_BOOTSTRAP_RESAMPLES,
        "z_bar": screen.ROUND2_Z_BAR,
        "n_configs": screen.ROUND2_N_CONFIGS,
        "B": blocks,
        "benchmark": benchmark,
        "fat_tail": fat_tail,
        "concentration": concentration,
    }


def build_grid(*, specs=None, bps=None, per_cell=None, mutate=None,
               family: str = "E1", per_session: int = 4) -> list[dict]:
    """A whole family's records: every cell × both windows × both gating levels.

    ``bps`` is the default per-trip bps sequence; ``per_cell`` overrides one cell;
    ``mutate(cell, window, cost, record)`` runs last, so a test can break exactly
    one number in exactly one record.
    """
    specs = specs if specs is not None else families.GRIDS[family]
    bps = bps if bps is not None else [5.0] * N_TRIPS
    per_cell = per_cell or {}
    records = []
    for spec in specs:
        cell = spec["name"]
        cell_bps = per_cell.get(cell, bps)
        for window in screen.ROUND2_WINDOWS:
            for cost in judge.GATING_COSTS:
                record = build_record(cell, window, cost, list(cell_bps),
                                      per_session=per_session)
                record["family"] = family
                if mutate is not None:
                    mutate(cell, window, cost, record)
                records.append(record)
    return records


def cell_records(records, cell):
    return [r for r in records if r["cell_id"] == cell]


# ═══════════════════════════════════════════════════════════════════════
# the fixture itself: a grid the judge must admit, and one it must not
# ═══════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def clean_grid():
    return build_grid()


@pytest.fixture()
def grid(clean_grid):
    return copy.deepcopy(clean_grid)


def _judge(records, **kw):
    return judge.judge_round2(records, **kw)


# ── the alive path: every gate passes, so a constant stub cannot pass here ─
def test_a_grid_whose_every_gate_passes_is_ranked_with_no_rejection_row(grid):
    """The alive path for all six stages at once.

    Every cell of the declared 3×3 grid passes the headline, the benchmark, the
    fat-tail, the concentration and the neighbour rule on both windows, so the
    verdict is ``SURVIVORS``, every cell is ``alive`` and **no** gate's
    rejection list has a row.  An always-False gate fails here.
    """
    verdict = _judge(grid)
    assert verdict["verdict"] == "SURVIVORS"
    assert verdict["n_configs"] == 36 and verdict["z_bar"] == screen.ROUND2_Z_BAR
    assert len(verdict["cells"]) == len(families.GRIDS["E1"]) == 9
    assert verdict["dead"] == [] and verdict["not_evaluated"] == []
    assert sorted(verdict["alive"]) == sorted(c["name"] for c in families.GRIDS["E1"])
    assert len(verdict["ranked"]) == 9
    for key in ("headline_rejected", "benchmark_rejected", "fat_tail_rejected",
                "concentration_rejected", "fold_breadth_rejected",
                "neighbour_rejected"):
        assert verdict[key] == [], f"{key} has rows although every gate passed"
    for cell in verdict["alive"]:
        assert verdict["buckets"][cell] == screen.ROUND2_ALIVE
        assert verdict["cells"][cell]["stages"]["headline"]["pass"] is True
        assert verdict["cells"][cell]["stages"]["benchmark"]["pass"] is True
        assert verdict["cells"][cell]["stages"]["fat_tail"]["pass"] is True
        assert verdict["cells"][cell]["stages"]["concentration"]["pass"] is True
        assert verdict["cells"][cell]["stages"]["fold_breadth"]["pass"] is True
        assert verdict["cells"][cell]["stages"]["neighbour"]["pass"] is True


def test_the_ranked_rows_carry_the_fields_each_gate_doc_requires(grid):
    """R3-6.1 §4 / R3-6.2 §5 / R3-6.3 §5's ranked-row fields, all present."""
    row = _judge(grid)["ranked"][0]
    for key in ("benchmark_pass", "benchmark_delta_bps_w1", "benchmark_delta_bps_w2",
                "benchmark_lb_delta_w1", "benchmark_lb_delta_w2",
                "benchmark_net_bps_w1", "benchmark_net_bps_w2",
                "benchmark_worst_symbol",
                "fat_tail_pass", "fat_tail_k_trips_removed",
                "fat_tail_remainder_best_k_bps_w1", "fat_tail_remainder_best_k_bps_w2",
                "fat_tail_remainder_best_month_bps_w1",
                "fat_tail_remainder_best_month_bps_w2",
                "fat_tail_worst_month_w1", "fat_tail_worst_month_w2",
                "fat_tail_worst_session_share_w1", "fat_tail_worst_session_share_w2",
                "fat_tail_worst_session_date_w1", "fat_tail_worst_session_date_w2",
                "fat_tail_x1_5_flagged",
                "concentration_pass", "concentration_max_symbol_w1",
                "concentration_max_symbol_w2", "concentration_max_symbol_share_w1",
                "concentration_max_symbol_share_w2",
                "concentration_min_traded_symbol_trips_w1",
                "concentration_min_traded_symbol_trips_w2",
                "concentration_n_symbols_traded_w1",
                "concentration_n_symbols_traded_w2", "concentration_caps",
                "headline_lower_bound_w1", "headline_lower_bound_w2",
                "headline_B_w1", "headline_B_w2"):
        assert key in row, f"the ranked row lost {key}"
    assert row["concentration_caps"] == families.R2_CAPS
    assert row["benchmark_pass"] is True
    assert row["fat_tail_pass"] is True
    assert row["concentration_pass"] is True


def test_the_residual_is_quoted_beside_every_survivor_and_gates_nothing(grid):
    """R3-6.2 §6.1 / the residual ruling: the four fields and the sentence.

    The sentence is compared against the ruling **file**, not against the code's
    own copy of it: a paraphrase in the code is not the sentence the ruling
    requires, and the check has to be able to notice.
    """
    verdict = _judge(grid)
    assert verdict["residual"]["gating"] is False
    for row in verdict["ranked"]:
        for field in judge.RESIDUAL_FIELDS:
            assert field in row["residual"], f"a survivor lost {field}"
    doc = " ".join(RESIDUAL_RULING.read_text().split())
    assert " ".join(judge.RESIDUAL_SENTENCE.split()) in doc, (
        "the residual sentence the judge emits is not the one the ruling "
        "requires verbatim")


def test_n_and_z_bar_are_fixed_for_the_verdict_and_a_clone_census_changes_nothing(grid):
    """L19.12: N = 36 and z_bar = 3.196 are fixed; clones are a census artifact.

    Clone discovery excludes cells from the neighbour set only (L9) and never
    re-declares ``N``, ``z_bar`` or any other judge input, so passing a census
    must leave every bucket, every rejection row and the ranking byte-identical.
    """
    plain = _judge(grid)
    with_census = _judge(copy.deepcopy(grid),
                         clone_census={"clones": [["a", "b"]], "found": 1})
    assert plain["n_configs"] == 36 and plain["z_bar"] == pytest.approx(3.196)
    assert with_census["n_configs"] == plain["n_configs"]
    assert with_census["z_bar"] == plain["z_bar"]
    assert with_census["buckets"] == plain["buckets"]
    assert with_census["ranked"] == plain["ranked"]
    assert with_census["clone_census"] == {"clones": [["a", "b"]], "found": 1}


# ── step 2: the shared preconditions, and the slug that routes a bucket ─
def test_the_paired_floor_slug_is_headline_rejected_not_not_evaluated(grid):
    """L19.7 / C10 — the one routing exception, pinned at the judge.

    ``config_trips >= 150`` with ``paired_trips < 150`` is a **headline**
    failure.  The helper reports it under the slug ``paired_floor``;
    ``screen.round2_precondition_stage`` is the one place that slug becomes the
    ``headline`` stage, and the judge calls it.  The block still closes both
    identities (``140 + 28 == 168``), so the shortfall is a paired-subset
    shortfall and not a broken block.
    """
    cell = families.GRIDS["E1"][0]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["paired_trips"] = 140
            record["benchmark"]["benchmark_unpriced"] = 28
            record["benchmark"]["per_trip"] = record["benchmark"]["per_trip"][:140]

    records = build_grid(mutate=mutate)
    target = [r for r in records if r["cell_id"] == cell
              and r["window"] == "W1" and r["cost_label"] == judge.BASE_COST][0]
    ok, why = screen.round2_preconditions(target)
    assert not ok and why.startswith("paired_floor"), (
        f"the helper must report the slug, the judge maps it: got {why!r}")
    verdict = _judge(records)
    assert verdict["buckets"][cell] == "headline_rejected", (
        "the paired floor is a headline failure (L19.7); bucketing it "
        "not_evaluated is the L13 ambiguity the ruling removed")
    assert cell not in verdict["not_evaluated"]
    assert any(row["config"] == cell for row in verdict["headline_rejected"])
    assert not any(row["config"] == cell for row in verdict["benchmark_rejected"]), (
        "the paired floor is NOT the benchmark gate failing closed (L19.7)")


def test_an_absent_or_malformed_benchmark_block_is_not_evaluated_not_a_verdict(grid):
    """The contract audit's item A, at the judge: a malformed input is not a result.

    An absent benchmark block and a genuine ``paired_trips < 150`` shortfall used
    to emit the same ``paired_floor`` slug, so a literal slug rule reported "the
    block was never written" as ``headline_rejected`` — indistinguishable from a
    real headline failure.  ``benchmark_block`` now covers the absent block and
    every non-finite count (``nan < 150`` is ``False``, so NaN used to pass), and
    the cell refuses a verdict instead.
    """
    absent = families.GRIDS["E1"][1]["name"]
    nan_cell = families.GRIDS["E1"][2]["name"]

    def mutate(c, window, cost, record):
        if c == absent and cost == judge.BASE_COST:
            record["benchmark"] = None
        if c == nan_cell and cost == judge.BASE_COST:
            record["benchmark"]["paired_trips"] = float("nan")

    verdict = _judge(build_grid(mutate=mutate))
    for cell, label in ((absent, "absent"), (nan_cell, "NaN paired_trips")):
        assert verdict["buckets"][cell] == screen.ROUND2_NOT_EVALUATED, (
            f"a {label} benchmark block is a malformed input, not a strategy "
            f"result — got {verdict['buckets'][cell]!r}")
        assert "benchmark_block" in verdict["cells"][cell]["why"]
        assert cell not in verdict["cells_by_bucket"]["headline_rejected"], (
            f"the {label} block must not be reported as a headline failure: the "
            f"audit's trap is that both cases emitted the paired_floor slug")


def test_a_trips_floor_refusal_is_not_evaluated_with_the_slug_in_why(grid):
    """C2: one spelling for a precondition refusal — ``not_evaluated`` + slug."""
    cell = families.GRIDS["E1"][1]["name"]

    def mutate(c, window, cost, record):
        if c == cell:
            record["trips"] = 140
            record["trip_table"] = record["trip_table"][:140]

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == screen.ROUND2_NOT_EVALUATED
    assert "trips_floor" in verdict["cells"][cell]["why"]
    assert cell not in verdict["ranked"]
    assert cell not in verdict["cells_by_bucket"]["headline_rejected"]


def test_a_record_whose_trip_table_is_shorter_than_its_trip_count_is_refused(grid):
    """The contract audit's item B: a claimed count and a table must agree.

    Step 2 checks the **claimed** ``trips`` and then the headline iterates
    whatever table it is handed, so a record claiming 150 trips with a 12-row
    table passed step 2 and produced a headline from 12 trips.  A corrupt record
    is not a strategy result: the judge refuses loudly rather than bucketing it.
    """
    cell = families.GRIDS["E1"][0]["name"]

    def mutate(c, window, cost, record):
        if c == cell and window == "W1" and cost == judge.BASE_COST:
            record["trips"] = 150                 # still >= the 150-trip floor
            record["trip_table"] = record["trip_table"][:12]

    with pytest.raises(judge.RecordSetError, match="claims 150 trips"):
        _judge(build_grid(mutate=mutate))


def test_a_benchmark_block_counted_for_another_runs_trips_is_refused(grid):
    """The contract audit's item C: ``config_trips`` is pinned to the record.

    :func:`screen.assert_benchmark_paired_identity` checks the identities
    *inside* the block and nothing compared ``config_trips`` to the record, so a
    block written for another run's trip count could be quoted beside this one.
    """
    cell = families.GRIDS["E1"][0]["name"]

    def mutate(c, window, cost, record):
        if c == cell and window == "W1" and cost == judge.BASE_COST:
            record["benchmark"]["config_trips"] = 150

    with pytest.raises(judge.RecordSetError, match="config_trips"):
        _judge(build_grid(mutate=mutate))


def test_the_same_trip_table_object_reaches_step_2_and_the_bootstrap(grid,
                                                                    monkeypatch):
    """The audit's item B, second half: **one** table object, not two reads.

    Passing the record to step 2 and to the bootstrap leaves two independent
    reads of the same key; anything between them (a `_trim`, a JSON round trip, a
    caller that rebuilds the record) can make them different tables without
    either check noticing.  The judge passes the object it pinned, so this spies
    on both calls and asserts every table they see is an object from the record
    set itself — identity, not equality.
    """
    tables = [record["trip_table"] for record in grid]
    seen = []
    real_pre = screen.round2_preconditions
    real_boot = screen.session_block_lower_bound

    def spy_pre(record, table=None):
        seen.append(("preconditions", table))
        return real_pre(record, table=table)

    def spy_boot(trades, *args, **kwargs):
        seen.append(("bootstrap", trades))
        return real_boot(trades, *args, **kwargs)

    monkeypatch.setattr(screen, "round2_preconditions", spy_pre)
    monkeypatch.setattr(screen, "session_block_lower_bound", spy_boot)
    _judge(grid)

    kinds = {kind for kind, _table in seen}
    assert kinds == {"preconditions", "bootstrap"}, (
        f"both calls must be exercised for this to be evidence, got {kinds}")
    for kind, table in seen:
        assert any(table is known for known in tables), (
            f"the {kind} call did not receive a table object from the record "
            f"set — a copy is a second table, and the count step 2 checked then "
            f"describes rows the bootstrap never saw")


def test_a_cell_with_only_a_reported_sizing_is_not_evaluated(grid):
    """L1 / R3-3(ii): ``equity_fraction`` is a column and can never gate."""
    cell = families.GRIDS["E1"][2]["name"]
    records = [r for r in grid if r["cell_id"] != cell]
    for window in screen.ROUND2_WINDOWS:
        report = build_record(cell, window, judge.BASE_COST, [5.0] * N_TRIPS)
        report["sizing"] = "equity_fraction"
        records.append(report)
    verdict = _judge(records)
    assert verdict["buckets"][cell] == screen.ROUND2_NOT_EVALUATED
    assert "sizing" in verdict["cells"][cell]["why"]
    assert verdict["cells"][cell]["refused"] is True


def test_a_missing_gating_window_record_aborts_the_grid(grid):
    """R3-3(iii): a cell that did not run is a grid that did not run."""
    cell = families.GRIDS["E1"][3]["name"]
    records = [r for r in grid if not (r["cell_id"] == cell
                                       and r["window"] == "W2")]
    with pytest.raises(screen.GridCellFailure) as exc:
        _judge(records)
    assert cell in str(exc.value)


def test_a_concentration_block_under_another_caps_set_refuses_a_verdict(grid):
    """R3-6.3 §1: a share computed under other caps is not this run's."""
    cell = families.GRIDS["E1"][4]["name"]

    def mutate(c, window, cost, record):
        if c == cell:
            record["concentration"]["caps"] = {
                **families.R2_CAPS, "max_positions": 2}

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == screen.ROUND2_NOT_EVALUATED
    assert "caps" in verdict["cells"][cell]["why"]


# ── step 3/4: the headline statistic and the ×1.5 net rule ─────────────
def test_the_headline_gate_fails_a_cell_whose_x1_5_net_is_negative(grid):
    """Step 4's reject path, isolated: the base bar still clears.

    The ×1.5 trip table and its ``net_bps_per_trip`` are moved together, so the
    record still agrees with its own trips and the only failing conjunct is the
    ×1.5 net rule — which is exactly the "cost gate" the protocol declares.
    """
    cell = families.GRIDS["E1"][5]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.SLIP_COST:
            for row in record["trip_table"]:
                row["net_bps"] = -1.0
                row["pnl_after_costs"] = -1.0 / 1e4 * NOTIONAL
            record["net_bps_per_trip"] = -1.0

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "headline_rejected"
    detail = verdict["cells"][cell]["stages"]["headline"]
    assert detail["pass"] is False
    assert "×1.5" in detail["why"]
    # every other gate still passes: the kill is the cost gate, not the edge
    for stage in ("benchmark", "fat_tail", "concentration", "fold_breadth",
                  "neighbour"):
        assert verdict["cells"][cell]["stages"][stage]["pass"] is True, stage
    assert cell not in verdict["ranked"]
    assert any(row["config"] == cell and row["window"] == "W1"
               for row in verdict["headline_rejected"])


def test_the_headline_gate_fails_a_cell_whose_base_lower_bound_is_negative(grid):
    """Step 3's reject path: a positive mean with a negative 0.0694th percentile.

    Twenty sessions at +50 bps and twenty at −40 bps average +5 bps/trip, but the
    bootstrap's left tail is negative — the stage-2 finding that the mean is not
    the decision statistic.  The cell's other gates are recorded as well (L12),
    and the bucket is still the headline's because it is first in the order.
    """
    cell = families.GRIDS["E1"][6]["name"]
    bps = []
    for i in range(N_TRIPS):
        bps.append(50.0 if (i // 4) % 2 == 0 else -40.0)
    verdict = _judge(build_grid(per_cell={cell: bps}))
    detail = verdict["cells"][cell]["stages"]["headline"]["windows"]["W1"]
    assert detail["detail"]["net_bps_per_trip"] == pytest.approx(5.0)
    assert detail["detail"]["lower_bound"] < 0
    assert detail["pass"] is False
    assert verdict["buckets"][cell] == "headline_rejected"


def test_a_persisted_headline_bound_the_trip_table_does_not_re_derive_is_refused(grid):
    """L19.8: a persisted number that cannot be re-derived is not evidence."""
    cell = families.GRIDS["E1"][7]["name"]

    def mutate(c, window, cost, record):
        if c == cell:
            record["bootstrap_block_p0.0694"] += 1.0

    with pytest.raises(judge.GateBlockInconsistent) as exc:
        _judge(build_grid(mutate=mutate))
    assert "re-derive" in str(exc.value)


def test_a_record_that_re_declares_n_or_z_bar_is_refused(grid):
    """L19.12: ``N`` and ``z_bar`` are never re-declared after the data is seen."""
    cell = families.GRIDS["E1"][8]["name"]
    for key, value in (("n_configs", 34), ("z_bar", 2.5)):
        def mutate(c, window, cost, record, key=key, value=value):
            if c == cell:
                record[key] = value
        with pytest.raises(judge.GateBlockInconsistent) as exc:
            _judge(build_grid(mutate=mutate))
        assert key in str(exc.value)


def test_the_headline_bootstrap_is_read_from_the_trip_table_not_a_persisted_field(grid):
    """The lower bound in the verdict is the trip table's, under the call sequence."""
    cell = families.GRIDS["E1"][0]["name"]
    verdict = _judge(grid)
    detail = verdict["cells"][cell]["stages"]["headline"]["windows"]["W1"]["detail"]
    base = [r for r in grid if r["cell_id"] == cell and r["window"] == "W1"
            and r["cost_label"] == judge.BASE_COST][0]
    lower, blocks = screen.session_block_lower_bound(base["trip_table"])
    assert detail["lower_bound"] == pytest.approx(lower, abs=0)
    assert detail["B"] == blocks
    assert detail["seed"] == screen.ROUND2_BOOTSTRAP_SEED
    assert detail["resamples"] == screen.ROUND2_BOOTSTRAP_RESAMPLES
    assert detail["z_bar_applied"] is False


# ── step 6: the matched benchmark gate ─────────────────────────────────
def test_the_benchmark_gate_fails_a_cell_that_does_not_beat_its_benchmark(grid):
    """R3-6.1: ``LB_delta > 0`` is strict — zero fails, and so does a negative."""
    cell = families.GRIDS["E1"][1]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["delta_lower_bound_p0.0694"] = -0.5

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "benchmark_rejected"
    assert verdict["cells"][cell]["stages"]["headline"]["pass"] is True, (
        "the benchmark kill must not be the headline's")
    row = [r for r in verdict["benchmark_rejected"] if r["config"] == cell]
    assert row and row[0]["window"] == "W1" and row[0]["lb_delta"] == -0.5
    assert row[0]["delta_bps"] == pytest.approx(2.0)


def test_the_benchmark_gate_fails_a_cell_whose_x1_5_delta_is_not_positive(grid):
    """The ×1.5 half of R3-6.1's pass rule (a different level, a different row)."""
    cell = families.GRIDS["E1"][2]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.SLIP_COST:
            record["benchmark"]["delta_bps"] = 0.0
            record["benchmark"]["pass"] = False

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "benchmark_rejected"
    assert "×1.5" in verdict["cells"][cell]["stages"]["benchmark"]["why"]


def test_a_benchmark_block_whose_trip_counts_do_not_close_is_refused(grid):
    """L19.7: both identities, to 1e-9, before any ``delta_bps`` is read."""
    cell = families.GRIDS["E1"][3]["name"]

    def mutate(c, window, cost, record):
        if c == cell:
            record["benchmark"]["benchmark_unpriced"] = 1

    with pytest.raises(screen.BenchmarkIdentityError):
        _judge(build_grid(mutate=mutate))


def test_a_benchmark_whose_per_trip_rows_do_not_mean_its_published_expectancy_is_refused(grid):
    """F10 / L19.10: ``mean(y_i)`` must be the published benchmark expectancy."""
    cell = families.GRIDS["E1"][4]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["per_trip"][0]["y_i"] += 10.0

    with pytest.raises(judge.GateBlockInconsistent) as exc:
        _judge(build_grid(mutate=mutate))
    assert "mean(y_i)" in str(exc.value)


def test_a_benchmark_whose_per_symbol_table_does_not_decompose_the_pooled_delta_is_refused(grid):
    """R3-6.1 §1: pooled = Σ_s (n_s/N)·delta_s, asserted to 1e-9."""
    cell = families.GRIDS["E1"][5]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["per_symbol"][0]["delta_bps"] += 1.0

    with pytest.raises(judge.GateBlockInconsistent) as exc:
        _judge(build_grid(mutate=mutate))
    assert "decomposition" in str(exc.value)


def test_a_benchmark_that_pays_no_costs_is_refused(grid):
    """R3-6.1 §2.5 / L19.10: the only cost assertion that survives."""
    cell = families.GRIDS["E1"][6]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["benchmark_cost_drag_per_trip_bps"] = 0.0

    with pytest.raises(judge.GateBlockInconsistent) as exc:
        _judge(build_grid(mutate=mutate))
    assert "cost drag" in str(exc.value)


# ── step 7: the fat-tail (tail-dependence) gate ────────────────────────
def test_the_fat_tail_gate_fails_a_cell_whose_edge_is_ten_winning_trips(grid):
    """R3-6.2's FT_fail_trips shape: the top-``k`` trim removes the whole edge.

    Ten trips at +200 bps, one in each of the first ten months, the rest at
    −1 bps: R1 (leave-one-month-out) still passes and R3 (session share) passes,
    so the **trim rule alone** fires — which is what isolates it.
    """
    cell = families.GRIDS["E1"][7]["name"]
    bps = [-1.0] * N_TRIPS
    for i in range(10):
        bps[i * TRIPS_PER_MONTH] = 200.0
    verdict = _judge(build_grid(per_cell={cell: bps}))
    assert verdict["buckets"][cell] == "fat_tail_rejected"
    detail = verdict["cells"][cell]["stages"]["fat_tail"]["windows"]["W1"]["detail"]
    assert detail["remainder_after_best_k_bps"] == pytest.approx(-1.0)
    assert detail["rules"]["trim"]["pass"] is False
    assert detail["rules"]["best_month"]["pass"] is True
    assert detail["rules"]["session"]["pass"] is True
    assert detail["label"] == "weaker — tail-dependence only"
    rows = [r for r in verdict["fat_tail_rejected"] if r["config"] == cell]
    assert rows and rows[0]["rule_failed"] == "trim"


def test_the_fat_tail_gate_fails_a_cell_whose_edge_is_one_session(grid):
    """R3-6.2's FT_fail_session shape: one session above 25% of the dollars.

    Three trips at +100 bps share **one** session (the pinned cap is 4) among 165
    at +1 bps, so that session holds 0.647 of the window's dollars while both
    remainders stay positive: the session rule alone fires, which is what
    isolates it from the trim and the worst-month rule.
    """
    cell = families.GRIDS["E1"][8]["name"]
    bps = [1.0] * N_TRIPS
    for i in range(3):
        bps[i] = 100.0
    verdict = _judge(build_grid(per_cell={cell: bps}))
    detail = verdict["cells"][cell]["stages"]["fat_tail"]["windows"]["W1"]["detail"]
    assert detail["rules"]["session"]["pass"] is False
    assert detail["rules"]["best_month"]["pass"] is True
    assert detail["rules"]["trim"]["pass"] is True
    assert detail["best_session_share"] == pytest.approx(301.0 / 465.0)
    assert verdict["buckets"][cell] == "fat_tail_rejected"
    rows = [r for r in verdict["fat_tail_rejected"] if r["config"] == cell]
    assert rows and rows[0]["rule_failed"] == "session"
    assert rows[0]["threshold"] == 0.25


def test_a_fat_tail_block_whose_verdict_contradicts_its_rule_value_is_refused(grid):
    """A block whose ``pass`` disagrees with its own numbers cannot be scored."""
    cell = families.GRIDS["E1"][0]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["fat_tail"]["rules"]["trim"]["value"] = -1.0

    with pytest.raises(judge.GateBlockInconsistent) as exc:
        _judge(build_grid(mutate=mutate))
    assert "contradicts" in str(exc.value)


def test_a_fat_tail_block_that_is_not_evaluated_fails_closed(grid):
    """R3-6.2 §1: a ``not_evaluated`` block is a fail, never a silent pass."""
    cell = families.GRIDS["E1"][1]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["fat_tail"]["status"] = "not_evaluated"
            record["fat_tail"]["pass"] = False
            record["fat_tail"]["why"] = "multi_leg"

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "fat_tail_rejected"
    assert verdict["cells"][cell]["stages"]["fat_tail"]["pass"] is False


# ── step 8: the concentration gate ────────────────────────────────────
def test_the_concentration_gate_fails_a_cell_whose_edge_is_one_symbol(grid):
    """R3-6.3's CT_fail_share shape: one symbol above 50% of the dollars.

    SOXL's trips pay +100 bps, the other three +12 bps, so every symbol keeps
    ≥ 25 trips (the trip rule passes) and the **share rule alone** fires.
    """
    cell = families.GRIDS["E1"][2]["name"]
    bps = []
    for i in range(N_TRIPS):
        bps.append(100.0 if UNIVERSE[i % 4] == "SOXL" else 12.0)
    verdict = _judge(build_grid(per_cell={cell: bps}))
    assert verdict["buckets"][cell] == "concentration_rejected"
    detail = verdict["cells"][cell]["stages"]["concentration"]["windows"]["W1"]["detail"]
    assert detail["max_symbol"] == "SOXL"
    assert detail["max_symbol_share"] > 0.50
    assert detail["rules"]["min_trips"]["pass"] is True, (
        "the share rule must fire alone: every symbol keeps its trips")
    rows = [r for r in verdict["concentration_rejected"] if r["config"] == cell]
    assert rows and rows[0]["rule_failed"] == "share"
    assert verdict["cells"][cell]["stages"]["benchmark"]["pass"] is True


def test_the_concentration_gate_fails_a_cell_with_a_thin_traded_symbol(grid):
    """R3-6.3's CT_fail_trips shape: a traded symbol under 25 trips."""
    cell = families.GRIDS["E1"][3]["name"]
    bps = [5.0] * N_TRIPS
    symbols = ["SOXL"] * 48 + ["TQQQ"] * 48 + ["SPY"] * 48 + ["SPXL"] * 24
    verdict = _judge(build_grid(per_cell={cell: bps},
                                mutate=_symbol_assignment(cell, symbols)))
    detail = verdict["cells"][cell]["stages"]["concentration"]["windows"]["W1"]["detail"]
    assert detail["min_traded_symbol_trips"] == 24
    assert detail["max_symbol_share"] <= 0.50, (
        "the trip rule must fire alone: no share exceeds 50%")
    assert verdict["buckets"][cell] == "concentration_rejected"
    rows = [r for r in verdict["concentration_rejected"] if r["config"] == cell]
    assert rows and rows[0]["rule_failed"] == "min_trips"


def _symbol_assignment(cell, symbols):
    """Reassign one cell's trips to symbols, keeping the table consistent."""
    def mutate(c, window, cost, record):
        if c != cell:
            return
        for row, symbol in zip(record["trip_table"], symbols):
            row["instrument"] = symbol
            row["symbols"] = symbol
        rows = record["trip_table"]
        record["concentration"] = _concentration_block(rows, cost)
        record["concentration"]["window"] = window
        record["benchmark"] = _benchmark_block(rows, 2.0, 1.0, cost)
    return mutate


def test_a_concentration_block_whose_share_contradicts_its_table_is_refused(grid):
    """R3-6.3 §1: ``share_s`` is re-derived from ``per_symbol``, not trusted."""
    cell = families.GRIDS["E1"][4]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["concentration"]["per_symbol"][0]["net_pnl_usd"] += 1.0

    with pytest.raises(judge.GateBlockInconsistent) as exc:
        _judge(build_grid(mutate=mutate))
    assert "total_net_pnl_usd" in str(exc.value) or "share" in str(exc.value)


def test_a_concentration_block_that_is_not_evaluated_fails_closed(grid):
    """R3-6.3 §2.1: ``not_evaluated`` is a fail, never a silent pass."""
    cell = families.GRIDS["E1"][5]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["concentration"]["status"] = "not_evaluated"
            record["concentration"]["pass"] = False
            record["concentration"]["why"] = "multi_leg"

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "concentration_rejected"


# ── step 5: fold breadth, at base cost ────────────────────────────────
def test_the_fold_breadth_gate_fails_a_cell_with_a_thin_month(grid):
    """L6 / R3-8: 12 months, ≥ 5 trips each, ≥ 7 of 12 positive — at base cost."""
    cell = families.GRIDS["E1"][6]["name"]

    def mutate(c, window, cost, record):
        if c == cell:
            record["folds"][0]["round_trips"] = 4

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "fold_breadth_rejected"
    assert any(row["config"] == cell for row in verdict["fold_breadth_rejected"])
    assert verdict["cells"][cell]["stages"]["headline"]["pass"] is True


def test_the_fold_breadth_gate_fails_a_cell_with_too_few_positive_months(grid):
    """The 7-of-12 half of the same rule, isolated from the thin-month half."""
    cell = families.GRIDS["E1"][7]["name"]

    def mutate(c, window, cost, record):
        if c == cell:
            for fold in record["folds"][:6]:
                fold["net_bps_per_trip"] = -5.0

    verdict = _judge(build_grid(mutate=mutate))
    assert verdict["buckets"][cell] == "fold_breadth_rejected"
    assert "months positive" in verdict["cells"][cell]["stages"]["fold_breadth"]["why"]


def test_fold_breadth_is_read_at_base_cost_and_a_non_base_record_is_refused(grid):
    """L6: reading the zero-cost column here is round 1's rule, not round 2's."""
    record = copy.deepcopy(grid[0])
    record["cost_label"] = cost_levels.ZERO_COST
    with pytest.raises(ValueError):
        screen.breadth_at_base(record)


# ── step 9: the neighbour rule, after every cell is scored ────────────
def test_the_neighbour_gate_fails_a_cell_whose_axis_neighbours_are_negative(grid):
    """P2 / L19.6: an axis needs two positive neighbours when it has two.

    Both of one axis's neighbours are made net-negative on both windows — their
    own headlines fail, which is irrelevant to the *survivor's* rule: what the
    rule reads is ``positive_both``, and it must be False for them.
    """
    survivor = families.GRIDS["E1"][4]["name"]
    specs = families.GRIDS["E1"]
    axes = screen.neighbour_cells("E1", specs, survivor)
    axis = sorted(axes)[0]
    losers = axes[axis]
    assert len(losers) >= 2, "the 3x3 grid gives every axis two neighbours"
    records = build_grid(per_cell={cell: [-5.0] * N_TRIPS for cell in losers})
    verdict = _judge(records)
    assert verdict["positive_both"][survivor] is True
    assert all(verdict["positive_both"][cell] is False for cell in losers)
    assert verdict["buckets"][survivor] == "neighbour_rejected"
    assert verdict["cells"][survivor]["stages"]["neighbour"]["pass"] is False
    assert axis in verdict["cells"][survivor]["stages"]["neighbour"]["why"]
    rows = [r for r in verdict["neighbour_rejected"] if r["config"] == survivor]
    assert rows


def test_the_neighbour_rule_is_applied_after_every_cell_is_scored(grid):
    """Step 9 is not per-record: its input is the whole grid's both-window record."""
    verdict = _judge(grid)
    rule = verdict["cells"][verdict["alive"][0]]["stages"]["neighbour"]["detail"]["rule"]
    assert "net_bps_per_trip" in rule
    assert set(verdict["positive_both"]) == {c["name"] for c in families.GRIDS["E1"]}


# ── step 11: the bucket, the labelling order, and both verdict shapes ──
def test_the_bucket_is_the_first_failure_in_the_labelling_order(grid):
    """C1 / I3: benchmark is ranked second, concentration fourth — benchmark wins.

    The cell fails **both** the benchmark and the concentration rule; the bucket
    is ``benchmark_rejected``, and the concentration gate's own verdict and row
    are still in the payload (C9: both shapes retained, so no failure hides
    behind another's reason string).
    """
    cell = families.GRIDS["E1"][0]["name"]
    bps = []
    for i in range(N_TRIPS):
        bps.append(100.0 if UNIVERSE[i % 4] == "SOXL" else 12.0)

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["delta_lower_bound_p0.0694"] = -1.0

    verdict = _judge(build_grid(per_cell={cell: bps}, mutate=mutate))
    assert verdict["buckets"][cell] == "benchmark_rejected"
    assert any(row["config"] == cell for row in verdict["benchmark_rejected"])
    assert any(row["config"] == cell for row in verdict["concentration_rejected"]), (
        "the concentration gate's own evidence must survive the earlier bucket "
        "(C9 / I3)")
    assert verdict["cells"][cell]["stages"]["concentration"]["pass"] is False


def test_every_gate_is_evaluated_and_recorded_even_after_the_cell_is_dead(grid):
    """L12: all stages are evaluated; a dead cell's other failures stay recorded.

    A cell whose every trip loses 5 bps fails the headline (negative lower
    bound), the fat-tail (both remainders negative, total negative) and the
    concentration gate (total not positive).  All three must be present with
    their own reasons, and the bucket must be the headline's.
    """
    cell = families.GRIDS["E1"][1]["name"]
    verdict = _judge(build_grid(per_cell={cell: [-5.0] * N_TRIPS}))
    stages = verdict["cells"][cell]["stages"]
    assert stages["headline"]["pass"] is False
    assert stages["fat_tail"]["pass"] is False
    assert stages["concentration"]["pass"] is False
    assert stages["benchmark"]["pass"] in (True, False)
    assert verdict["buckets"][cell] == "headline_rejected"
    assert any(row["config"] == cell for row in verdict["fat_tail_rejected"])
    assert any(row["config"] == cell for row in verdict["concentration_rejected"])
    assert any(row["config"] == cell for row in verdict["headline_rejected"])


def test_the_bucket_does_not_depend_on_the_order_the_records_are_handed_in(grid):
    """C1: the bucket is the labelling order's first failure, not assembly order."""
    forward = _judge(grid)
    backward = _judge(list(reversed(copy.deepcopy(grid))))
    assert forward["buckets"] == backward["buckets"]
    assert forward["ranked"] == backward["ranked"]
    assert forward["cells_by_bucket"] == backward["cells_by_bucket"]


def test_only_a_cell_that_fails_nothing_is_ranked(grid):
    """A dead cell never enters ``ranked``; exactly one bucket per cell, total."""
    cell = families.GRIDS["E1"][2]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.BASE_COST:
            record["benchmark"]["delta_lower_bound_p0.0694"] = -1.0

    verdict = _judge(build_grid(mutate=mutate))
    assert cell not in [row["config"] for row in verdict["ranked"]]
    assert len(verdict["ranked"]) == len(families.GRIDS["E1"]) - 1
    for name in (screen.ROUND2_ALIVE,) + screen.ROUND2_BUCKET_NAMES + (
            screen.ROUND2_NOT_EVALUATED,):
        assert name in verdict["cells_by_bucket"]
    seen = [cell for cells in verdict["cells_by_bucket"].values() for cell in cells]
    assert sorted(seen) == sorted(verdict["buckets"]), (
        "every cell has exactly one bucket, dead or alive (L19.6)")


# ── the record set: refused spellings, ambiguity, and the file path ────
def test_the_dotted_cost_spelling_is_refused_not_aliased(grid):
    """L19.1: one label per level; the stale dotted form is refused."""
    record = copy.deepcopy(grid[0])
    record["cost_label"] = "slip_x1.5"
    with pytest.raises(judge.RecordSetError) as exc:
        judge.index_records([record])
    assert "slip_x1.5" in str(exc.value)


def test_two_records_for_one_key_are_refused(grid):
    """Two answers to one question is not a record set."""
    with pytest.raises(judge.RecordSetError) as exc:
        judge.index_records([copy.deepcopy(grid[0]), copy.deepcopy(grid[0])])
    assert "two run records" in str(exc.value)


def test_a_record_set_that_mixes_families_is_refused(grid):
    record = copy.deepcopy(grid[0])
    record["family"] = "H"
    with pytest.raises(judge.RecordSetError) as exc:
        judge.index_records([copy.deepcopy(grid[1]), record])
    assert "mixes families" in str(exc.value)


def test_a_commission_bearing_gating_level_is_refused(grid):
    """I2 / L3: the judge asserts the gating set is slippage-only."""
    cell = families.GRIDS["E1"][3]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.SLIP_COST:
            record["cost_model"]["commission_per_share"] = 0.005

    with pytest.raises(judge.CostLevelError) as exc:
        _judge(build_grid(mutate=mutate))
    assert "commission" in str(exc.value)


def test_the_x1_5_level_must_be_one_and_a_half_times_the_base_slippage(grid):
    """I2 / L3: a level that did not scale is not the declared stress."""
    cell = families.GRIDS["E1"][4]["name"]

    def mutate(c, window, cost, record):
        if c == cell and cost == judge.SLIP_COST:
            record["cost_model"]["slippage_abs"] = 0.01     # the base level's own

    with pytest.raises(judge.CostLevelError) as exc:
        _judge(build_grid(mutate=mutate))
    assert "×1.5" in str(exc.value)


def test_b_is_recorded_and_never_scored(grid):
    """L19.9 / C6: ``B`` is a recorded diagnostic — no verdict path reads it.

    Two otherwise identical grids differ only in how many trips share a session,
    so ``B`` (the entry-date block count the bootstrap resamples) differs by a
    factor of four.  The buckets must be **identical**: a fifth gate on ``B`` is
    refused, and the two annotation literals (3, N/4) are never evaluated.
    """
    many = _judge(build_grid(per_session=1))
    few = _judge(build_grid(per_session=4))
    cell = families.GRIDS["E1"][0]["name"]
    b_many = many["cells"][cell]["stages"]["headline"]["windows"]["W1"]["detail"]["B"]
    b_few = few["cells"][cell]["stages"]["headline"]["windows"]["W1"]["detail"]["B"]
    assert b_many == 168 and b_few == 48, (b_many, b_few)
    assert many["buckets"] == few["buckets"]
    assert many["cells_by_bucket"] == few["cells_by_bucket"]
    assert many["B_annotations"] == list(screen.ROUND2_B_ANNOTATIONS)
    assert b_few >= 3, "the annotation's own bound holds on this fixture"


def test_the_judge_reads_the_untrimmed_record_files_from_disk(grid, tmp_path):
    """The input is the per-run record file, JSON round trip included."""
    paths = []
    for i, record in enumerate(grid):
        path = tmp_path / f"rec_{i:03d}.json"
        path.write_text(json.dumps(record, default=str))
        paths.append(path)
    on_disk = judge.load_records(paths)
    assert on_disk == grid
    assert _judge(on_disk)["buckets"] == _judge(grid)["buckets"]


def test_a_half_written_record_file_is_not_scored_as_a_small_one(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(judge.RecordSetError):
        judge.load_records([path])
