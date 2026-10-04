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

**Kill rule** — for each family, screen its coarse grid on W1 at realistic costs
first; if no config clears the power floor and net-positive rule on W1, stop the
family there: no W2 run, no zero-cost run, no grid widening.  The kill is
reported with the family's best W1 numbers and the mechanism that killed it.

**Two sizing modes.**  Every config is run at fixed $50k notional *and* at the
engine's 50 %-of-equity/95 %-of-cash mode.  A config clears W1 if **either**
mode clears it, and the W2 confirmation is then required in **the same mode**
that cleared W1 — so a config cannot survive by switching sizing between
windows.  Both modes' numbers are reported for every config either way.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

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
