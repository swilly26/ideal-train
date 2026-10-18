"""The stage-1 screen: a mechanical, pre-registered pass/fail rule.

The point of this module is that **no config's fate is decided by hand**.  Every
threshold below was written into the brief before any run, and the same function
reads every run record, so a family is killed or kept by the code.

Pre-registered rule (brief §5)
------------------------------
A config clears stage 1 only if all of:

1. **Power floor** — at least :data:`POWER_FLOOR` round trips in the window.
2. **Net positive** — net-of-cost expectancy per trip > 0 on W1, and then, and
   only then, also > 0 on W2.
3. **Gross breadth** — the zero-cost run's expectancy is positive in at least
   :data:`MIN_POSITIVE_FOLDS` of the window's 12 monthly folds, in **both**
   windows.

**Kill rule (ROUND 1 ONLY — see L1 below).**  For each family, screen its coarse
grid on W1 at realistic costs first; if no config clears the power floor and
net-positive rule on W1, stop the family there: no W2 run, no zero-cost run, no
grid widening.  The kill is reported with the family's best W1 numbers and the
mechanism that killed it.  This rule is **round 1's**; round 2's screen path is
:func:`screen_grid_both_windows`, which runs every declared cell on both windows
and kills a cell that fails **either** window (R3-3(iii)/(iv)).

**Two sizing modes (ROUND 1 ONLY).**  Every config is run at fixed $50k notional
*and* at the engine's 50 %-of-equity/95 %-of-cash mode.  A config clears W1 if
**either** mode clears it, and the W2 confirmation is then required in **the
same mode** that cleared W1 — so a config cannot survive by switching sizing
between windows.  Round 2 declares ``fixed_notional`` the **sole gating** sizing
(L1); the other mode is a reported column and can never admit a cell.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Sequence

POWER_FLOOR = 150
#: Reinstated power floor, plus the pinned per-month minimum: a calendar month
#: with fewer than :data:`MIN_TRIPS_PER_MONTH` trips fails the config outright
#: (P2).  Pinned at **5**, not 1: the round-1 rule was a floor of one trip, which
#: a config can clear on a single trade — a month is then "positive" on a coin
#: flip, and the 7-of-12 stability guard becomes 7 coin flips.  The lead's
#: revision-3 correction (R3-8 / open ambiguity answer 3) is this number; it is
#: declared **once**, here, and every reader of the rule uses this constant.
MIN_TRIPS_PER_MONTH = 5
MIN_POSITIVE_FOLDS = 7
MONTHS_PER_WINDOW = 12
SIZING_MODES = ("fixed_notional", "equity_fraction")

#: The window(s) stage 1's early kill runs on — **ROUND 1's rule, by name** (L1,
#: audit item 9).  ``screen_family`` stops a family at W1 and never touches W2.
#: Round 2 declares a different rule in :data:`ROUND2_KILL_WINDOWS`; the two must
#: never be read as one, which is why each is a declared constant rather than a
#: line of prose inside a function.
STAGE1_KILL_WINDOWS: tuple[str, ...] = ("W1",)
#: The scope of stage 1's kill rule.  Round 2 must not apply it: a round-2 cell
#: that loses W1 is still run on W2 (R3-3(iii)) and dies on either window.
STAGE1_KILL_SCOPE = "round_1_only"


def positive_months(folds: Iterable[Mapping]) -> int:
    """Months with at least one trip and a positive mean bps/trip (pinned)."""
    return sum(1 for f in (folds or [])
               if int(f.get("round_trips", 0)) >= MIN_TRIPS_PER_MONTH
               and float(f.get("net_bps_per_trip", 0.0)) > 0)


def fold_stability(folds: Iterable[Mapping]) -> tuple[bool, str, int]:
    """The **one** pinned sub-period stability rule (P2).

    A fixed config's sub-period stability is the **mean bps/trip per calendar
    month**: it passes when the window has all
    :data:`MONTHS_PER_WINDOW` months, no month has fewer than
    :data:`MIN_TRIPS_PER_MONTH` trips (a thin month fails the config), and
    at least :data:`MIN_POSITIVE_FOLDS` of those months have a positive mean
    bps/trip.  This is a *concentration guard, not validation* — nothing is
    re-fitted, so it is never called walk-forward.

    Returns ``(pass, why, months_with_zero_trips)``.
    """
    rows = list(folds or [])
    total = len(rows)
    zero_trip = sum(1 for f in rows
                    if int(f.get("round_trips", 0)) < MIN_TRIPS_PER_MONTH)
    pos = positive_months(rows)
    if zero_trip:
        return (False, f"{zero_trip} calendar month(s) with fewer than "
                       f"{MIN_TRIPS_PER_MONTH} trips", zero_trip)
    if total < MONTHS_PER_WINDOW:
        return (False, f"only {total} month(s) with trades "
                       f"(< {MONTHS_PER_WINDOW})", zero_trip)
    if pos < MIN_POSITIVE_FOLDS:
        return (False, f"{pos}/{total} months positive (< {MIN_POSITIVE_FOLDS})",
                zero_trip)
    return (True, f"{pos}/{total} months positive", zero_trip)


def gate_power_and_net(rec: Mapping) -> tuple[bool, str]:
    """Rules 1 + 2 on one window's realistic-cost run."""
    trips = int(rec.get("trips", 0))
    if trips < POWER_FLOOR:
        return False, f"trips {trips} < {POWER_FLOOR}"
    bps = float(rec.get("net_bps_per_trip", 0.0))
    if not bps > 0:
        return False, f"net bps/trip {bps:+.2f} not > 0"
    return True, "ok"


def breadth(rec: Mapping) -> tuple[bool, str]:
    """Rule 3: the pinned per-month stability rule on the zero-cost run."""
    ok, why, _zero = fold_stability(rec.get("folds") or [])
    if not ok:
        return False, f"zero-cost {why}"
    return True, f"zero-cost {why}"


def screen_family(family: str,
                  specs: Sequence[dict],
                  run: Callable[[dict, str, str, str], Mapping],
                  log: Callable[[str], None] = print) -> dict:
    """Run the pre-registered screen for one family and return its verdict.

    ``run(spec, window, cost_label, sizing)`` must return a machine-readable run
    record; this function only applies the rules to those records.
    """
    W1, W2 = "W1", "W2"
    runs: dict[str, dict] = {}
    w1_pass: dict[str, list[str]] = {}
    for spec in specs:
        name = spec["name"]
        for sizing in SIZING_MODES:
            rec = dict(run(spec, W1, "baseline", sizing))
            runs[f"{W1}|baseline|{sizing}|{name}"] = rec
            ok, why = gate_power_and_net(rec)
            rec["w1_gate"] = {"pass": ok, "why": why}
            if ok:
                w1_pass.setdefault(name, []).append(sizing)
            log(f"  [W1] {name} {sizing}: trips={rec['trips']} "
                f"net={rec['net_pnl']:,.0f} netbps={rec['net_bps_per_trip']:+.2f} "
                f"-> {'PASS' if ok else 'fail'} ({why})")

    if not w1_pass:
        best = _best_w1(runs, W1, "baseline")
        return {"family": family, "verdict": "KILLED_ON_W1",
                "w1_survivors": [], "w2_survivors": [], "ranked": [],
                "runs": runs, "best_w1": best,
                "w1_pass_pairs": [], "w2_confirmed_pairs": [],
                "kill_reason": _kill_mechanism(best)}

    confirmed: dict[str, list[str]] = {}
    for name, sizings in w1_pass.items():
        spec = next(s for s in specs if s["name"] == name)
        for sizing in sizings:
            rec = dict(run(spec, W2, "baseline", sizing))
            runs[f"{W2}|baseline|{sizing}|{name}"] = rec
            ok, why = gate_power_and_net(rec)
            rec["w2_gate"] = {"pass": ok, "why": why}
            if ok:
                confirmed.setdefault(name, []).append(sizing)
            log(f"  [W2] {name} {sizing}: trips={rec['trips']} "
                f"net={rec['net_pnl']:,.0f} netbps={rec['net_bps_per_trip']:+.2f} "
                f"-> {'CONFIRMED' if ok else 'fail'} ({why})")

    ranked: list[dict] = []
    for name, sizings in confirmed.items():
        spec = next(s for s in specs if s["name"] == name)
        for sizing in sizings:
            zero_ok = True
            zero_why = []
            for window in (W1, W2):
                zrec = dict(run(spec, window, "zero_cost", sizing))
                runs[f"{window}|zero_cost|{sizing}|{name}"] = zrec
                ok, why = breadth(zrec)
                zero_ok &= ok
                zero_why.append(f"{window}: {why}")
                log(f"  [{window}] {name} {sizing} zero-cost breadth: "
                    f"{'PASS' if ok else 'fail'} ({why})")
            if not zero_ok:
                continue
            r1 = runs[f"{W1}|baseline|{sizing}|{name}"]
            r2 = runs[f"{W2}|baseline|{sizing}|{name}"]
            net_pct = (float(r1["net_pct"]) + float(r2["net_pct"])) / 2.0
            bps = (float(r1["net_bps_per_trip"]) + float(r2["net_bps_per_trip"])) / 2.0
            folds_pos = min(int(r1["folds_positive"]), int(r2["folds_positive"]))
            row = {
                "config": name, "sizing": sizing,
                "net_pct_w1": float(r1["net_pct"]), "net_pct_w2": float(r2["net_pct"]),
                "net_pct_equal_weighted": net_pct,
                "net_bps_w1": float(r1["net_bps_per_trip"]),
                "net_bps_w2": float(r2["net_bps_per_trip"]),
                "net_bps_equal_weighted": bps,
                "trips_w1": int(r1["trips"]), "trips_w2": int(r2["trips"]),
                "t_stat_w1": float(r1.get("net_bps_t_stat", 0.0)),
                "t_stat_w2": float(r2.get("net_bps_t_stat", 0.0)),
                "max_drawdown_w1": float(r1.get("max_drawdown", 0.0)),
                "max_drawdown_w2": float(r2.get("max_drawdown", 0.0)),
                "max_drawdown_worst": min(float(r1.get("max_drawdown", 0.0)),
                                          float(r2.get("max_drawdown", 0.0))),
                "cost_drag_bps_w1": float(r1.get("cost_drag_per_trip_bps", 0.0)),
                "cost_drag_bps_w2": float(r2.get("cost_drag_per_trip_bps", 0.0)),
                "folds_positive_min": folds_pos,
                "zero_cost_breadth": zero_why,
            }
            row["rank"] = rank_row(row)
            ranked.append(row)
    ranked.sort(key=lambda r: r["rank"], reverse=True)
    verdict = "SURVIVORS" if ranked else (
        "NO_W2_SURVIVOR" if not confirmed else "FAILED_BREADTH")
    return {"family": family, "verdict": verdict,
            "w1_survivors": sorted(w1_pass), "w2_survivors": sorted(confirmed),
            "ranked": ranked, "runs": runs,
            #: The (cell, sizing) pairs each stage actually ran, so the runner
            #: can assert the recorded run set against what the protocol
            #: declares *in this branch* (A3) — a survivor licenses W2 and both
            #: zero-cost runs, a W1 loser licenses neither.
            "w1_pass_pairs": sorted([n, s] for n, ss in w1_pass.items() for s in ss),
            "w2_confirmed_pairs": sorted([n, s] for n, ss in confirmed.items()
                                         for s in ss)}


def rank_row(row: Mapping) -> tuple:
    """The ranked order the deliverable promises (P2/E6).

    Lexicographic, in this order: **net** (equal-weighted net %) → **drawdown**
    (the shallower of the two windows' max drawdowns) → **trips** (more evidence
    is better) → **costs** (lower cost drag in bps is better) → **most positive
    months** (the pinned stability count).  ``conservative_score =
    min(net_pct, bps/100)`` mixed a percentage with a bps-scaled number and
    collapsed to ``bps/100`` whenever bps was the smaller of the two; it is gone.
    """
    return (
        round(float(row.get("net_pct_equal_weighted", 0.0)), 8),
        round(float(row.get("max_drawdown_worst", 0.0)), 8),
        int(row.get("trips_w1", 0)) + int(row.get("trips_w2", 0)),
        -round(float(row.get("cost_drag_bps_w1", 0.0))
               + float(row.get("cost_drag_bps_w2", 0.0)), 8),
        int(row.get("folds_positive_min", 0)),
    )


def declared_axes(family: str, specs: Sequence[dict]) -> dict[str, tuple]:
    """Every axis declared inside the grid: a knob with >= 2 distinct values.

    Values are read from the **resolved** spec, so an axis that only exists
    because two cells silently resolved to different values is not missed.
    """
    from src.backtesting.strategy_search import families as _families
    seen: dict[str, set] = {}
    for spec in specs:
        res = _families.resolve_spec(family, spec)
        flat = {**res["params"], **res["signal"], **res["extras"]}
        for key, val in flat.items():
            seen.setdefault(key, set()).add(_families.canonical_json(val))
    return {k: v for k, v in seen.items() if len(v) >= 2}


def neighbour_cells(family: str, specs: Sequence[dict], survivor: str) -> dict:
    """Per-axis neighbour sets of the *survivor* cell (P2).

    The neighbour set of axis *a* is the cells that differ from the survivor in
    exactly that axis, deduped by resolved-config hash, **the survivor itself
    excluded**.  The criterion then applies *per axis*: an axis with >= 2
    neighbours needs two of them positive on both windows; an axis with exactly
    one neighbour needs that one positive on both windows.
    """
    from src.backtesting.strategy_search import families as _families
    axes = declared_axes(family, specs)
    resolved = {s["name"]: _families.resolve_spec(family, s) for s in specs}
    survivors = [s for s in specs if s["name"] == survivor]
    if len(survivors) != 1:
        raise ValueError(f"survivor {survivor!r} is not exactly one cell")
    base = resolved[survivor]
    base_flat = {**base["params"], **base["signal"], **base["extras"]}
    base_hash = _families.resolved_hash(family, base)
    out: dict[str, list[str]] = {}
    for axis in axes:
        neighbours: dict[str, str] = {}
        for spec in specs:
            if spec["name"] == survivor:
                continue
            res = resolved[spec["name"]]
            flat = {**res["params"], **res["signal"], **res["extras"]}
            if _families.resolved_hash(family, res) == base_hash:
                continue                       # a duplicate resolved config
            if all(flat.get(k) == base_flat.get(k) for k in flat
                   if k != axis) and flat.get(axis) != base_flat.get(axis):
                neighbours.setdefault(_families.resolved_hash(family, res),
                                      spec["name"])
        out[axis] = sorted(neighbours.values())
    return out


def neighbour_verdict(neighbours: Mapping[str, Sequence[str]],
                      positive_both: Mapping[str, bool]) -> tuple[bool, str]:
    """Apply the per-axis neighbour rule (P2).

    ``positive_both[cell]`` is True when the cell is net-positive on both
    windows.  An axis needs two neighbours positive on both windows when it has
    at least two, and its single neighbour to be positive on both when it has
    exactly one.  An axis with no neighbour at all is a *failure*: a declared
    axis that cannot be perturbed is not tested.
    """
    problems: list[str] = []
    if not neighbours:
        return False, "no axis has a neighbour cell"
    for axis, cells in sorted(neighbours.items()):
        if not cells:
            problems.append(f"{axis}: declared axis with no neighbour cell "
                            f"(a declared axis that cannot be perturbed is not "
                            f"tested)")
            continue
        pos = [c for c in cells if positive_both.get(c, False)]
        if len(cells) == 1 and not pos:
            problems.append(f"{axis}: its only neighbour {cells[0]} is not "
                            f"positive on both windows")
        elif len(cells) >= 2 and len(pos) < 2:
            problems.append(f"{axis}: only {len(pos)}/{len(cells)} neighbours "
                            f"positive on both windows (need 2)")
    if problems:
        return False, "; ".join(problems)
    return True, f"per-axis neighbours hold ({len(neighbours)} axes)"


# ── R3-3(iii)/(iv): the *round-2* grid rule ────────────────────────────
#: R3-3(iii): every declared cell runs on **both** windows.  This is **not**
#: stage 1's rule: ``screen_family`` stops a family at W1 (round 1's kill, see
#: :data:`STAGE1_KILL_SCOPE`; still pinned by
#: ``tests/test_strategy_search_engine.py::test_a_family_with_no_w1_survivor_is_killed_without_touching_w2``),
#: so a cell that loses W1 never gets a W2 row and the 36 x 2 = 72 base records
#: R3-3 declares cannot exist.  The two rules are kept apart on purpose, by
#: name (:data:`STAGE1_KILL_WINDOWS` vs :data:`ROUND2_KILL_WINDOWS`) and by test
#: (``test_the_stage1_kill_is_round_one_only_and_round_two_kills_on_either``),
#: because a round-2 cell that is missing from one window is not a killed cell,
#: it is a cell the report cannot score.
ROUND2_WINDOWS: tuple[str, ...] = ("W1", "W2")
#: R3-3(ii): the one sizing mode that gates (the other is a reported column).
ROUND2_SIZING = "fixed_notional"
#: R3-3(iii): base cost for the declared 36 x 2 = 72 records.
ROUND2_BASE_COST = "baseline"
#: R3-1/R3-3: the declared N of the round-2 grid — four groups (E1, E2, G, H) of
#: a full 3 x 3 factorial, 36 cells.  Declared here so the *number* is a name a
#: test can read, not a claim in prose: :func:`assert_round2_declaration` checks
#: the tree's own grids against it.
ROUND2_N_CONFIGS = 36
#: R3-2: the Bonferroni bar for N = 36 (alpha = 0.05 two-sided, alpha/2N =
#: 6.944e-4).  Recorded, never re-derived at read time.
ROUND2_Z_BAR = 3.196
#: L6: round-2 fold breadth is read at **base** cost — the brief's kill criteria
#: are stated there.  Stage 1's :func:`breadth` reads a zero-cost record, which
#: is round 1's rule and must not be carried into round 2 (audit item 10).
ROUND2_BREADTH_COST = ROUND2_BASE_COST
#: L1: round 2's kill runs on either window — the stage-1 W1-only rule does not.
ROUND2_KILL_WINDOWS: tuple[str, ...] = ROUND2_WINDOWS
#: L19.8 / L19-S: the **declared** session-block bootstrap.  These three are the
#: literals the pre-registration pins (§4's ``Statistic`` line); they are declared
#: here so :func:`session_block_lower_bound` cannot drift from them, and they are
#: the *only* place the round-2 bootstrap reads a number.  L19.14: nothing in L19
#: changes any of them — a percentile, a seed and a resample count are part of the
#: verdict and may not be re-chosen after a P&L number is seen.
ROUND2_BOOTSTRAP_SEED = 20260930
ROUND2_BOOTSTRAP_RESAMPLES = 4000
#: The decision number is the **empirical** 0.0694th percentile of the bootstrap
#: draws (alpha/2N = 6.944e-4 at N = 36).  :data:`ROUND2_Z_BAR` = 3.196 is the
#: normal-approximation *reference* for the same bar and is **never applied** by
#: the judge (C7) — the two must not be conflated in the report.
ROUND2_HEADLINE_PERCENTILE = 0.0694
#: L19.9 / C6: ``B`` (the number of distinct entry-date blocks) is a **recorded
#: diagnostic**, not a gate.  3 and ``N / 4`` bound the bootstrap's *block count*
#: (not fold breadth — that is R3-6.2/R3-6.3's business), they are inert as
#: conditions, and no verdict path may read them.  ``B`` is recorded per (cell,
#: window) and reported; if a future re-pin lets ``B`` fall, the bootstrap
#: degenerates toward the sample mean and *that is stated in the report* rather
#: than policed by a fifth gate.
ROUND2_B_ANNOTATIONS: tuple[int, ...] = (3, ROUND2_N_CONFIGS // 4)
#: L16: the blocks every round-2 base record must carry, as the gate docs name
#: them (R3-6.1 benchmark, R3-6.2 fat_tail, R3-6.3 concentration).  The headline
#: statistic's own fields (L2) are record fields, not a block, and are the next
#: delegation's work; this constant is the *contract* the judge asserts.
ROUND2_GATE_BLOCKS: tuple[str, ...] = ("benchmark", "fat_tail", "concentration")

#: L19.3: the keys each declared gate block must carry, so that the failure for a
#: block that is absent or empty can **name what that block requires**.  The lists
#: are the gate docs' own published block fields (R3-6.1 §4, R3-6.2 §5, R3-6.3 §5);
#: the blocks' *values* are the gate build's contract (L19.6-L19.13), and this
#: assertion never evaluates them.  One entry per declared block.
ROUND2_GATE_BLOCK_KEYS: Mapping[str, tuple[str, ...]] = {
    "benchmark": (
        "variant", "source", "eod_flat_min", "entry_convention",
        "exit_convention", "cost_label", "sizing", "notional_usd",
        "config_trips", "benchmark_trips", "paired_trips", "benchmark_unpriced",
        "config_net_bps_per_trip", "benchmark_net_bps_per_trip", "delta_bps",
        "delta_lower_bound_p0.0694", "delta_resamples", "delta_seed",
        "config_cost_drag_per_trip_bps", "benchmark_cost_drag_per_trip_bps",
        "benchmark_fills_per_trip", "pass", "why", "per_symbol", "per_trip",
    ),
    "fat_tail": (
        "version", "window", "config_name_resolved", "resolved_hash", "sizing",
        "cost_label", "n_trips", "n_sessions", "n_months", "k_trips_removed",
        "k_formula", "removed_fraction", "removal_cutoff_bps",
        "total_net_pnl_usd", "net_bps_per_trip", "remainder_after_best_k_bps",
        "remainder_after_best_10_bps", "best_k_trips_pnl_usd",
        "worst_case_removed_month", "remainder_after_best_month_bps",
        "best_month_by_total_pnl", "remainder_after_that_month_bps",
        "worst_session_date", "worst_session_symbols",
        "worst_session_net_pnl_usd", "best_session_share",
        "best_single_trip_share", "notional_uniform", "total_matches_record",
        "x1_5", "status", "pass", "why", "rules",
    ),
    "concentration": (
        "version", "window", "config_name_resolved", "resolved_hash", "sizing",
        "cost_label", "caps", "symbol_priority", "universe", "n_trips",
        "total_net_pnl_usd", "symbol_unit", "symbol_unit_ok",
        "shares_are_dollar_based", "decomposition_max_abs_residual",
        "max_symbol", "max_symbol_share", "max_symbol_net_pnl_usd",
        "n_symbols_traded", "traded_symbols", "min_traded_symbol",
        "min_traded_symbol_trips", "per_symbol", "rules", "status", "pass",
        "why",
    ),
}


class GridCellFailure(RuntimeError):
    """A cell of the frozen grid could not be resolved or run (R3-3(iii)).

    Raised *after* every cell has been attempted, so the message lists **all**
    of them: a grid that quietly shrinks to the cells that happen to work is
    the one failure mode this rule exists to make impossible, because the
    report would still say N = 36.
    """


def screen_grid_both_windows(family: str, specs: Sequence[dict],
                             run: Callable[[dict, str, str, str], Mapping],
                             log: Callable[[str], None] = print,
                             windows: Sequence[str] = ROUND2_WINDOWS,
                             cost: str = ROUND2_BASE_COST,
                             sizing: str = ROUND2_SIZING,
                             report_sizings: Sequence[str] = ()) -> dict:
    """Run **every** declared cell on **both** windows (R3-3(iii)–(iv)).

    ``run(spec, window, cost, sizing)`` is the same callable ``screen_family``
    takes.  Two declared properties, neither of which stage 1 has:

    * **no early kill** — a cell's W2 run does not depend on its W1 number, so
      the W1/W2 pair of rows exists for every cell;
    * **no silent drop** — a cell that cannot be resolved or run (the callable
      raises, or returns nothing) is collected and raised as a
      :class:`GridCellFailure` naming the family, the cell, the window and the
      underlying error.  Every remaining cell is still attempted, so the
      failure text is the whole picture and not just the first casualty.

    ``report_sizings`` runs additional sizing modes **for reporting only**
    (R3-3(ii): the second mode "may still be run and recorded but must never
    admit a cell").  Those records land in ``reported_runs``, keyed
    ``(cell, window, sizing)``, never in ``runs`` — so no admission test can
    read one, and the declared record set is still two keys per cell.  A
    reported run that fails is listed in ``reported_failures`` and does not
    abort the grid, because it gates nothing.

    Nothing here reads a P&L: it returns the records it was handed, keyed by
    ``(cell, window)``.
    """
    expected = [(spec["name"], window) for spec in specs for window in windows]
    if len({name for name, _ in expected}) != len(specs):
        raise ValueError(f"family {family}: the declared grid has duplicate cell "
                         f"names ({[s['name'] for s in specs]}) — two cells under "
                         f"one name is a silent drop waiting to happen")
    for extra in report_sizings:
        if extra == sizing:
            raise ValueError(f"family {family}: {extra!r} is the gating sizing — "
                             f"a reported-only column cannot be the gating one")
    runs: dict[tuple[str, str], dict] = {}
    reported: dict[tuple[str, str, str], dict] = {}
    failures: list[tuple[str, str, str]] = []
    report_failures: list[tuple[str, str, str, str]] = []
    for name, window in expected:
        spec = next(s for s in specs if s["name"] == name)
        try:
            rec = run(spec, window, cost, sizing)
            if not rec:
                raise ValueError("the run returned no record")
            runs[(name, window)] = dict(rec)
        except Exception as exc:                                # noqa: BLE001
            failures.append((name, window, f"{type(exc).__name__}: {exc}"))
            log(f"  ! {family} {name} [{window}/{cost}/{sizing}]: "
                f"{type(exc).__name__}: {exc}")
            continue
        log(f"  {family} {name} [{window}/{cost}/{sizing}]: recorded")
    if failures:
        detail = "; ".join(f"{n}[{w}]: {why}" for n, w, why in failures)
        raise GridCellFailure(
            f"family {family}: {len(failures)} of {len(expected)} declared "
            f"cell-window runs could not be resolved — the grid is a hard "
            f"failure, not a smaller grid ({detail})")
    # R3-3(ii)/L1: a non-gating sizing may still be run and recorded, but it
    # lands here, outside ``runs`` — so no admission test can ever read it, and
    # the declared 2-keys-per-cell record set is unchanged by running it.  A
    # reported run that fails is recorded loudly and is *not* fatal: it gates
    # nothing, so it must not be able to abort the declared grid.
    for extra in report_sizings:
        for name, window in expected:
            spec = next(s for s in specs if s["name"] == name)
            try:
                rec = run(spec, window, cost, extra)
                if not rec:
                    raise ValueError("the run returned no record")
                reported[(name, window, extra)] = dict(rec)
            except Exception as exc:                            # noqa: BLE001
                report_failures.append((name, window, extra,
                                        f"{type(exc).__name__}: {exc}"))
                log(f"  ! {family} {name} [{window}/{cost}/{extra}] (reported "
                    f"only, never gating): {type(exc).__name__}: {exc}")
                continue
            log(f"  {family} {name} [{window}/{cost}/{extra}] (reported only)")
    return {"family": family, "windows": list(windows), "cost": cost,
            "sizing": sizing, "cells": [s["name"] for s in specs],
            "runs": runs, "reported_runs": reported,
            "reported_failures": report_failures}


def _best_w1(runs: Mapping[str, Mapping], window: str, cost: str) -> dict:
    """The family's best W1 config, by the screen's own ordering (bps, then net)."""
    cands = [r for k, r in runs.items() if k.startswith(f"{window}|{cost}|")]
    if not cands:
        return {}
    best = max(cands, key=lambda r: (float(r.get("net_bps_per_trip", 0.0)),
                                     float(r.get("net_pnl", 0.0))))
    return {"config": best["config"], "sizing": best["sizing"],
            "trips": int(best["trips"]), "net_pnl": float(best["net_pnl"]),
            "net_pct": float(best["net_pct"]),
            "net_bps_per_trip": float(best["net_bps_per_trip"]),
            "net_bps_t_stat": float(best.get("net_bps_t_stat", 0.0)),
            "cost_drag_per_trip": float(best.get("cost_drag_per_trip", 0.0)),
            "win_rate": float(best.get("win_rate", 0.0)),
            "break_even_win_rate": float(best.get("break_even_win_rate", 0.0)),
            "folds_positive": int(best.get("folds_positive", 0)),
            "folds_total": int(best.get("folds_total", 0)),
            "avg_hold_minutes": float(best.get("avg_hold_minutes", 0.0)),
            "trips_per_session": float(best.get("trips_per_session", 0.0)),
        }


def _kill_mechanism(best: Mapping) -> str:
    """One line: what killed the family, from its own best numbers."""
    if not best:
        return "no run produced a record"
    trips = int(best.get("trips", 0))
    bps = float(best.get("net_bps_per_trip", 0.0))
    drag = float(best.get("cost_drag_per_trip", 0.0))
    notional = float(best.get("notional_per_trip", 0.0)) or 50000.0
    drag_bps = drag / notional * 1e4 if notional else 0.0
    if trips < POWER_FLOOR:
        return (f"best config only reached {trips} trips (< {POWER_FLOOR}): the "
                f"selection is too tight to clear the power floor")
    return (f"best config nets {bps:+.2f} bps/trip over {trips} trips while paying "
            f"{drag:,.2f} $/trip ({drag_bps:.2f} bps) of cost drag")


# ── L16: the gate-block assertion, a name of its own ───────────────────
class GateBlocksMissing(RuntimeError):
    """A round-2 base record does not carry a declared gate block (L16).

    Separate from :class:`GridCellFailure` and from the runner's run-set
    assertion on purpose: "the record is not there" and "the gate was never
    written into the record that is there" are different failures, and the
    acceptance criteria cite the second one by name.
    """


def assert_gate_blocks_present(record: Mapping,
                               gates: Sequence[str] = ROUND2_GATE_BLOCKS) -> None:
    """Every declared gate block is present **and carries content** (L16, L19.3).

    ``record`` is one round-2 base run record; ``gates`` is the set of block
    names the pre-registration requires (default :data:`ROUND2_GATE_BLOCKS`).
    A block fails when it is **missing**, ``None``, **or an empty mapping**: an
    empty dict used to count as present, which made
    ``{"benchmark": {}, "fat_tail": {}, "concentration": {}}`` satisfy the one
    check the acceptance criteria cite as proof that the gates were written —
    the L17 "presence is not content" failure class, one level down (L19.3).
    The failure names the offending block and the keys that block must carry
    (:data:`ROUND2_GATE_BLOCK_KEYS`).

    It still does **not** evaluate a gate: a non-empty block passes whatever its
    values say.  What a block's fields must *contain* is the gate build's
    contract (L19.6-L19.13); this asserts only that there is a block to check.

    The counterpart of :meth:`run_strategy_search.Runner.assert_screen_recorded`,
    which asserts run-set equality and nothing else.  A record whose gate block
    was never written cannot be scored, and a run-set assertion must not be
    quoted as evidence that it was.
    """
    problems: list[str] = []
    for gate in gates:
        block = record.get(gate)
        required = list(ROUND2_GATE_BLOCK_KEYS.get(gate, ()))
        if block is None:
            problems.append(f"{gate}: absent — it must carry {required}")
        elif not isinstance(block, Mapping) or not block:
            problems.append(f"{gate}: present but empty ({block!r}) — it must "
                            f"carry {required}")
    if not problems:
        return
    where = record.get("identity_key") or "|".join(
        str(record.get(k, "?")) for k in ("family", "cell_id", "config", "window",
                                          "cost_label", "sizing"))
    raise GateBlocksMissing(
        f"{where}: the round-2 base record has no gate block for "
        f"{'; '.join(problems)} (L16/L19.3, required blocks {tuple(gates)}) — a "
        f"gate that was never written, or written empty, cannot be scored, and "
        f"the run-set assertion does not check it")


# ── L19.7: step 2 (the shared fail-closed preconditions), executable ─────
#: The key under which the per-(cell, window) **trip table** (L4) reaches the
#: judge.  Step 2's ``legs == 1 ∀i`` and ``|notional_i / notional_usd − 1| < 1e-6
#: ∀i`` are statements about *trips*, so they are not readable from the aggregate
#: record: the judge hands the table in here, as either the engine's own trip
#: frame (a ``DataFrame`` with ``legs``/``notional`` columns) or its JSON
#: projection (a sequence of mappings with those keys).
ROUND2_TRIP_TABLE_KEY = "trip_table"


class BenchmarkIdentityError(RuntimeError):
    """A benchmark block's two trip-count identities do not hold (L19.7).

    The identities are ``benchmark_trips == config_trips`` (``benchmark_trips``
    is the **generated** count) and
    ``paired_trips + benchmark_unpriced == benchmark_trips``.  L19.7 replaces
    L13's ``benchmark_trips + benchmark_unpriced == config_trips``, which made a
    generated count out of a paired one and left the paired floor inside the
    benchmark gate.
    """


def _trip_column(table, column: str):
    """One column of the trip table in either accepted shape; ``None`` if absent.

    Accepting two shapes is not tolerance for its own sake: the round-2 judge
    reads the **per-run record files** (L19-S implementation notes), where the
    table has been through JSON, while the same helper is called in-process
    against the engine's own frame.  Both are the same table.
    """
    if table is None:
        return None
    if hasattr(table, "columns"):                       # the engine's trip frame
        if column not in table.columns:
            return None
        return list(table[column])
    if isinstance(table, (str, bytes)) or not isinstance(table, Iterable):
        return None
    out = []
    for row in list(table):
        if not isinstance(row, Mapping) or column not in row:
            return None
        out.append(row[column])
    return out


def _same_number(a, b) -> bool:
    """Numeric equality that survives a JSON round trip (``4`` vs ``4.0``)."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def round2_preconditions(record: Mapping) -> tuple[bool, str]:
    """Spec §1 step 2 — the shared fail-closed preconditions, in one place (L19.7).

    All conjunctive, all computed before any gate is read: the 150-trip floor on
    the **generated** count, single-leg trips, uniform notional, the caps in
    force equal to the pinned caps, and the paired-subset floor (L19.7's move: it
    is a step-2 precondition so a cell can no longer be either
    ``headline_rejected`` or "fails closed" depending on the reader).

    Returns ``(ok, why)``.  ``why`` is ``"ok"`` when the record passes and
    otherwise **starts with the slug of the precondition that refused it**, so
    the judge's bucket pass reads the verdict off the reason instead of
    re-deriving it:

    * ``trips_floor`` / ``trip_table`` / ``legs`` / ``notional`` / ``caps`` — a
      precondition mismatch: the cell refuses a verdict (``not_evaluated``), one
      spelling (C2);
    * ``paired_floor`` — ``config_trips >= 150`` with ``paired_trips < 150`` — the
      cell is ``headline_rejected``, unambiguously (L19.7 / C10).  The floor is
      read from *this* record's benchmark block, i.e. at each gating level.

    It **evaluates no gate**: what it reads is either :data:`POWER_FLOOR` or a
    pinned literal.
    """
    trips = int(record.get("trips", 0) or 0)
    if trips < POWER_FLOOR:
        return False, f"trips_floor: trips {trips} < {POWER_FLOOR}"
    table = record.get(ROUND2_TRIP_TABLE_KEY)
    legs = _trip_column(table, "legs")
    if legs is None:
        return False, (f"trip_table: no per-trip 'legs' column (the L4 trip table "
                       f"is handed in under {ROUND2_TRIP_TABLE_KEY!r})")
    for i, n in enumerate(legs):
        if int(n) != 1:
            return False, (f"legs: trip {i} has {int(n)} legs — round 2 is "
                           f"single-leg only (R3-6.1 §2.7)")
    notional = _trip_column(table, "notional")
    if notional is None:
        return False, (f"trip_table: no per-trip 'notional' column (the L4 trip "
                       f"table is handed in under {ROUND2_TRIP_TABLE_KEY!r})")
    from src.backtesting.strategy_search import families as _families
    # The reference is the pinned literal: the caps check below requires the run's
    # own notional_usd to equal it, so this cannot disagree with the run.
    ref = float(_families.R2_CAPS["notional_usd"])
    for i, n in enumerate(notional):
        try:
            uniform = abs(float(n) / ref - 1.0) < 1e-6
        except (TypeError, ValueError):
            uniform = False
        if not uniform:
            return False, (f"notional: trip {i} notional {n!r} — "
                           f"|notional_i/notional_usd - 1| must be < 1e-6 at "
                           f"notional_usd {ref:g}")
    params = record.get("config_params")
    if not isinstance(params, Mapping):
        return False, ("caps: the record carries no 'config_params', so the caps "
                       "in force cannot be read — refuse a verdict rather than "
                       "score a run whose caps are unknown")
    wrong = [k for k, v in _families.R2_CAPS.items()
             if k not in params or not _same_number(params[k], v)]
    if wrong:
        return False, (f"caps: {', '.join(sorted(wrong))} do not equal the pinned "
                       f"caps {dict(_families.R2_CAPS)}")
    block = record.get("benchmark")
    if not isinstance(block, Mapping) or "paired_trips" not in block:
        return False, ("paired_floor: the record carries no benchmark block with "
                       "paired_trips — the paired floor is a step-2 precondition "
                       "(L19.7), so an absent block cannot be left to the gate")
    paired = float(block["paired_trips"])
    if paired < POWER_FLOOR:
        return False, (f"paired_floor: paired_trips {paired:g} < {POWER_FLOOR} "
                       f"with config_trips {trips} >= {POWER_FLOOR} → "
                       f"headline_rejected (L19.7)")
    return True, "ok"


def assert_benchmark_paired_identity(block: Mapping) -> None:
    """Both benchmark trip-count identities hold, **to 1e-9** (L19.7).

    ``benchmark_trips == config_trips`` — the benchmark's trip count is the
    **generated** count, one benchmark trip per realised config trip (R3-6.1
    §2.4); and ``paired_trips + benchmark_unpriced == benchmark_trips`` — the
    unpriceable-EOD trips are *inside* the generated count, not extra (§2.7).
    This is what L19.7 puts in place of L13's
    ``benchmark_trips + benchmark_unpriced == config_trips``.

    Called from the judge **before any ``delta_bps`` is read**: a block whose
    counts do not close cannot be scored, and a silently mismatched pair makes
    every gate number read off it incomparable with the trip table it claims to
    describe.
    """
    missing = [k for k in ("config_trips", "benchmark_trips", "paired_trips",
                           "benchmark_unpriced") if k not in block]
    if missing:
        raise BenchmarkIdentityError(
            f"the benchmark block carries no {', '.join(missing)}, so neither "
            f"identity is checkable (L19.7)")
    config_trips = float(block["config_trips"])
    benchmark_trips = float(block["benchmark_trips"])
    paired = float(block["paired_trips"])
    unpriced = float(block["benchmark_unpriced"])
    if (abs(benchmark_trips - config_trips) <= 1e-9
            and abs((paired + unpriced) - benchmark_trips) <= 1e-9):
        return
    raise BenchmarkIdentityError(
        f"the benchmark block's trip counts do not close: "
        f"benchmark_trips==config_trips is {benchmark_trips:g} vs "
        f"{config_trips:g}; "
        f"paired_trips+benchmark_unpriced==benchmark_trips is {paired:g}+"
        f"{unpriced:g} vs {benchmark_trips:g} (L19.7; both checked to 1e-9)")


# ── the headline statistic's producer (L19.8) ──────────────────────────
def _bootstrap_trip_frame(trades) -> "object":
    """The trip table as a frame with the three columns the bootstrap reads.

    Accepts the same two shapes :func:`_trip_column` accepts — the engine's own
    trip frame (``SearchResult.trades``) and its JSON projection (the
    ``trip_table`` of a per-run record file) — so persisted and in-process
    numbers come from one function rather than two.
    """
    import pandas as _pd
    if hasattr(trades, "columns"):
        frame = trades
    else:
        rows = list(trades)
        frame = _pd.DataFrame(rows)
    missing = [c for c in ("entry_time", "notional", "pnl_after_costs")
               if c not in frame.columns]
    if missing:
        raise ValueError(
            f"the trip table cannot produce the session-block bootstrap: it has "
            f"no {', '.join(missing)} (L19.8 reads entry_time, notional and "
            f"pnl_after_costs, the same three the trip frame carries)")
    return frame


def session_block_lower_bound(trades, seed: int = ROUND2_BOOTSTRAP_SEED,
                              resamples: int = ROUND2_BOOTSTRAP_RESAMPLES
                              ) -> tuple[float, int]:
    """The round-2 headline: the 0.0694th percentile of the session-block
    bootstrap of mean net bps/trip, and ``B``, the number of blocks (L19.8/L19.9).

    **Pinned by its CALL SEQUENCE, not by four literals** (L19.8).  Two
    implementations that agree on seed, percentile and resample count but differ
    on any of the five points below return different lower bounds, and the
    difference is material whenever sessions differ in trip count.  The sequence,
    in order:

    (a) each trip's **block** is its entry date, ``entry_time.date()``;
    (b) the blocks are ordered **ascending by date** (``DataFrame.groupby``'s
        default sort — *not* first appearance in the frame, which is what
        ``Series.unique()`` would give and is a different draw);
    (c) ``rng = numpy.random.default_rng(seed)`` — **one** stream, seeded once;
    (d) per resample, ``idx = rng.integers(0, n_blocks, n_blocks)`` and the
        drawn sessions' trips are concatenated **in draw order**;
    (e) the statistic is the mean of ``net_bps`` over the **concatenated trips** —
        i.e. a trip-weighted mean — *not* the mean of the drawn sessions' means;
    (f) that draw is repeated ``resamples`` times and **no other number is drawn
        from the stream**;
    (g) the decision number is the ``ROUND2_HEADLINE_PERCENTILE``-th percentile of
        the **empirical draws**.  ``ROUND2_Z_BAR`` is a normal-approximation
        *reference* for the same bar and is not applied here or by any gate (C7).

    This is a **verbatim port** of the block half of
    ``scripts/run_strategy_search_stage2.py::bootstrap`` — the same key, the same
    groupby order, the same single seeded ``default_rng``, the same
    ``integers(0, len(by_day), len(by_day))`` draw and the same
    ``np.concatenate([...]).mean()`` statistic — so stage 2's published block
    numbers stay re-derivable from round 2's own code.  Two deliberate
    differences, both stated rather than implicit:

    * stage 2 draws a second, i.i.d. ``rng.choice`` cross-check from the **same**
      stream; that draw is **not** ported.  Every block draw is therefore the
      first ``resamples`` draws of the stream, exactly as in stage 2's block half,
      and a future caller that re-adds the i.i.d. draw must state that its order
      now depends on it;
    * ``np.percentile(blocks, 0.0694)`` replaces stage 2's ``percentile(_, 2.5)``
      — the 2.5th percentile is ~36x looser (finding 3 / L2) and is not the
      round-2 bar.

    Returns ``(lower_bound, B)``.  ``B`` is the number of **distinct entry-date
    blocks**, recorded per (cell, window) as a diagnostic (L19.9); no gate reads
    it.  It degenerates toward the sample mean as ``B`` falls.
    """
    import numpy as _np
    import pandas as _pd
    frame = _bootstrap_trip_frame(trades)
    if len(frame) == 0:
        raise ValueError(
            "the session-block bootstrap needs at least one trip (L19.8); this "
            "trip table is empty, so the headline has no denominator at all")
    day = _pd.to_datetime(frame["entry_time"]).dt.strftime("%Y-%m-%d")
    bps = frame["pnl_after_costs"] / frame["notional"] * 1e4
    # (b): groupby's default sort is ascending by date — the block order the draw
    # uses.  `Series.unique()` would give first-appearance order instead.
    by_day = [g.to_numpy()
              for _, g in _pd.DataFrame({"day": day, "bps": bps}).groupby("day")]
    n_blocks = len(by_day)
    if n_blocks < 1:
        raise ValueError("the session-block bootstrap found no entry-date block")
    rng = _np.random.default_rng(seed)                       # (c): one stream
    blocks = _np.empty(int(resamples), dtype=float)
    for i in range(int(resamples)):
        idx = rng.integers(0, n_blocks, n_blocks)            # (d)
        blocks[i] = _np.concatenate([by_day[j] for j in idx]).mean()   # (e)
    lower = float(_np.percentile(blocks, ROUND2_HEADLINE_PERCENTILE))  # (g)
    return lower, n_blocks


# ── the round-2 declared grid, by value ────────────────────────────────
def assert_round2_declaration(n_configs: int = ROUND2_N_CONFIGS,
                              z_bar: float = ROUND2_Z_BAR) -> dict:
    """The tree's own round-2 grids declare N = 36 cells and z_bar = 3.196.

    N is a **condition** in R3-3, not an assumption: it is 36 only while each
    group is a full factorial over exactly two 3-level axes and no cell is added
    after the first P&L number is seen.  Reading the declared grids back and
    comparing the count to the pre-registered literal is what makes ``z_bar``
    checkable rather than asserted.  Returns the declaration it verified.
    """
    from src.backtesting.strategy_search import families as _families
    fams = list(_families.ROUND2_FAMILIES)
    cells: list[str] = []
    per_family: dict[str, int] = {}
    for fam in fams:
        names = [s["name"] for s in _families.GRIDS[fam]]
        per_family[fam] = len(names)
        cells.extend(names)
    if len(set(cells)) != len(cells):
        dupes = sorted({c for c in cells if cells.count(c) > 1})
        raise ValueError(f"the declared round-2 grid has duplicate cell names "
                         f"({dupes}) — two cells under one name is a silent drop")
    if len(cells) != n_configs:
        raise ValueError(
            f"the declared round-2 grid has {len(cells)} cells "
            f"({per_family}) but the pre-registration declares N = {n_configs} "
            f"(R3-3(i)/(ii)); z_bar = {z_bar} is computed from N and would be "
            f"wrong")
    return {"families": fams, "per_family": per_family, "cells": cells,
            "n_configs": len(cells), "z_bar": float(z_bar)}


# ── the two round-2 rules that need no gate block ──────────────────────
def round2_cell_fates(runs: Mapping[tuple[str, str], Mapping],
                      fails_on: Callable[[Mapping, str], tuple[bool, str]]
                      = lambda rec, window: gate_power_and_net(rec),
                      windows: Sequence[str] = ROUND2_KILL_WINDOWS) -> dict:
    """A cell is dead if it fails **either** window (R3-3(iii)/(iv)).

    ``runs`` is :func:`screen_grid_both_windows`'s ``runs`` mapping, keyed
    ``(cell, window)``.  A cell-window with **no record** is a
    :class:`GridCellFailure`, never a silent drop: under R3-3(iii) a cell that
    did not run on a window is a grid that did not run.

    ``fails_on(record, window) -> (pass, why)`` is the conjunct applied to each
    window's record.  It defaults to the one conjunction round 2's screen can
    apply without any gate block — the power floor and net-positive expectancy
    of :func:`gate_power_and_net`.  The four gates of
    ``ROUND2_GATE_ORDER_SPEC.md`` §1 and the ranking are the judge's job, the
    next delegation's; ``fails_on`` is injectable so that judge can reuse this
    function rather than restate the either-window rule.
    """
    cells = sorted({cell for cell, _ in runs})
    out: dict[str, dict] = {}
    dead: list[str] = []
    alive: list[str] = []
    for cell in cells:
        detail: dict[str, dict] = {}
        for window in windows:
            rec = runs.get((cell, window))
            if rec is None:
                raise GridCellFailure(
                    f"round-2 cell {cell!r} has no run record for {window}: a "
                    f"cell that failed to run on either window is a hard error, "
                    f"never a silent drop (R3-3(iii)) — the grid did not run")
            ok, why = fails_on(rec, window)
            detail[window] = {"pass": bool(ok), "why": str(why)}
        is_dead = any(not d["pass"] for d in detail.values())
        why = "; ".join(f"{w}: {detail[w]['why']}" for w in windows
                        if not detail[w]["pass"]) or "clears every window"
        out[cell] = {"dead": is_dead, "windows": detail, "why": why}
        (dead if is_dead else alive).append(cell)
    return {"cells": out, "dead": dead, "alive": alive,
            "windows": list(windows),
            "rule": "a cell is dead if it fails EITHER window (R3-3(iii)/(iv)); "
                    "stage 1's W1-only kill is round 1's rule and is not applied "
                    "to round 2"}


def breadth_at_base(record: Mapping,
                    cost_label: Optional[str] = None) -> tuple[bool, str]:
    """L6: round-2 fold breadth read off the **base-cost** record.

    The cost level is *checked*, not assumed: handing this a record whose
    ``cost_label`` is anything but :data:`ROUND2_BREADTH_COST` raises, because
    reading the zero-cost column here is exactly the defect (stage 1's
    :func:`breadth` does read zero cost, and that is round 1's rule).
    """
    label = cost_label if cost_label is not None else record.get("cost_label")
    if label != ROUND2_BREADTH_COST:
        raise ValueError(
            f"round-2 fold breadth is a base-cost statistic (L6): this record's "
            f"cost level is {label!r}, expected {ROUND2_BREADTH_COST!r}")
    ok, why, _zero = fold_stability(record.get("folds") or [])
    return (ok, f"base-cost {why}")


def round2_fold_breadth(runs: Mapping[tuple[str, str], Mapping],
                        windows: Sequence[str] = ROUND2_WINDOWS) -> dict:
    """Per-cell fold breadth at base cost, conjunctive over the windows (L6).

    A cell passes only if **both** windows' base-cost records pass
    :func:`fold_stability` (all 12 months present, every month at least
    :data:`MIN_TRIPS_PER_MONTH` trips, at least :data:`MIN_POSITIVE_FOLDS`
    positive).  A missing window record is a :class:`GridCellFailure`; a record
    from a non-base cost level is a :class:`ValueError`.
    """
    cells = sorted({cell for cell, _ in runs})
    out: dict[str, dict] = {}
    passed: list[str] = []
    for cell in cells:
        detail: dict[str, dict] = {}
        for window in windows:
            rec = runs.get((cell, window))
            if rec is None:
                raise GridCellFailure(
                    f"round-2 cell {cell!r} has no run record for {window}: the "
                    f"base-cost fold breadth cannot be read (R3-3(iii), L6)")
            ok, why = breadth_at_base(rec)
            detail[window] = {"pass": bool(ok), "why": str(why)}
        ok_all = all(d["pass"] for d in detail.values())
        why = "; ".join(f"{w}: {detail[w]['why']}" for w in windows
                        if not detail[w]["pass"]) or "base-cost breadth holds"
        out[cell] = {"pass": ok_all, "windows": detail, "why": why}
        if ok_all:
            passed.append(cell)
    return {"cells": out, "pass": passed, "cost": ROUND2_BREADTH_COST,
            "windows": list(windows)}
