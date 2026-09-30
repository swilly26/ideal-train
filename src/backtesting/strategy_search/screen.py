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
MIN_POSITIVE_FOLDS = 7
SIZING_MODES = ("fixed_notional", "equity_fraction")


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
    """Rule 3: zero-cost expectancy positive in >= 7 of the window's folds."""
    folds = rec.get("folds") or []
    total = len(folds)
    pos = sum(1 for f in folds if float(f.get("net_bps_per_trip", 0.0)) > 0)
    if total < 12:
        return False, f"only {total} folds with trades (< 12)"
    if pos < MIN_POSITIVE_FOLDS:
        return False, f"{pos}/{total} zero-cost folds positive < {MIN_POSITIVE_FOLDS}"
    return True, f"{pos}/{total} zero-cost folds positive"


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
            ranked.append({
                "config": name, "sizing": sizing,
                "net_pct_w1": float(r1["net_pct"]), "net_pct_w2": float(r2["net_pct"]),
                "net_pct_equal_weighted": net_pct,
                "net_bps_w1": float(r1["net_bps_per_trip"]),
                "net_bps_w2": float(r2["net_bps_per_trip"]),
                "net_bps_equal_weighted": bps,
                "trips_w1": int(r1["trips"]), "trips_w2": int(r2["trips"]),
                "t_stat_w1": float(r1.get("net_bps_t_stat", 0.0)),
                "t_stat_w2": float(r2.get("net_bps_t_stat", 0.0)),
                "folds_positive_min": folds_pos,
                "conservative_score": min(net_pct, bps / 100.0),
                "zero_cost_breadth": zero_why,
            })
    ranked.sort(key=lambda r: r["conservative_score"], reverse=True)
    verdict = "SURVIVORS" if ranked else (
        "NO_W2_SURVIVOR" if not confirmed else "FAILED_BREADTH")
    return {"family": family, "verdict": verdict,
            "w1_survivors": sorted(w1_pass), "w2_survivors": sorted(confirmed),
            "ranked": ranked, "runs": runs}


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
