#!/usr/bin/env python3
"""Aggregate round-1 turbo replay artifacts into the report tables."""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path("/tmp/scratch/r1out")
REPORT = Path("/tmp/scratch/r1_tables.md")

# tag -> (lever, cost model label)
RUNS = [
    ("classic_baseline",            "baseline (live classic)", "$"),
    ("classic_zero_cost",           "baseline (live classic)", "0"),
    ("classic_baseline__A_latest",  "A window_tie=latest", "$"),
    ("classic_zero_cost__A_latest", "A window_tie=latest", "0"),
    ("classic_baseline__B_linear",  "B mr_conf_mode=linear", "$"),
    ("classic_zero_cost__B_linear", "B mr_conf_mode=linear", "0"),
    ("classic_baseline__C_age5",    "C max_signal_age_bars=5", "$"),
    ("classic_zero_cost__C_age5",   "C max_signal_age_bars=5", "0"),
    ("classic_baseline__D_cap2",    "D max_entries_per_session=2", "$"),
    ("classic_zero_cost__D_cap2",   "D max_entries_per_session=2", "0"),
    ("classic_baseline__D_cap5",    "D max_entries_per_session=5", "$"),
    ("classic_zero_cost__D_cap5",   "D max_entries_per_session=5", "0"),
    ("classic_baseline__E1_tp2",    "E1 take_profit_pct=0.02", "$"),
    ("classic_zero_cost__E1_tp2",   "E1 take_profit_pct=0.02", "0"),
    ("classic_baseline__E2_hold10", "E2 max_hold_minutes=10", "$"),
    ("classic_zero_cost__E2_hold10", "E2 max_hold_minutes=10", "0"),
    ("classic_baseline__F_minbars55", "F min_bars=55 (~10:25 ET)", "$"),
    ("classic_zero_cost__F_minbars55", "F min_bars=55 (~10:25 ET)", "0"),
    ("classic_baseline__F_minbars85", "F min_bars=85 (~10:55 ET)", "$"),
    ("classic_zero_cost__F_minbars85", "F min_bars=85 (~10:55 ET)", "0"),
]


def load(tag):
    p = OUT / f"turbo_stats_{tag}.json"
    if not p.exists():
        return None
    st = json.loads(p.read_text())
    folds = OUT / f"turbo_folds_monthly_{tag}.csv"
    st["_folds"] = pd.read_csv(folds) if folds.exists() else pd.DataFrame()
    return st


def main() -> int:
    runs = {}
    for tag, lever, cost in RUNS:
        st = load(tag)
        if st is None:
            print(f"MISSING {tag}", file=sys.stderr)
            continue
        runs[tag] = (lever, cost, st)
    if not runs:
        print("nothing to aggregate", file=sys.stderr)
        return 1

    base = runs["classic_baseline"][2]
    lines: list[str] = []

    def w(s=""):
        lines.append(s)

    # ── headline table ─────────────────────────────────────────────────
    w("### Headline (12 months, 2025-09-01..2026-09-01, base-4, $100k, EOD flatten ON)")
    w()
    w("| variant | costs | round trips | trips/session | net $ | net % | PF | win % vs break-even | max DD | cost drag $ | drag $/trip | exits (sig/time/eod/stop/target) | expectancy $/trip | pos folds |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, (lever, cost, st) in runs.items():
        ex = st["exits"]
        exs = "/".join(str(ex.get(k, 0)) for k in ("signal", "time", "eod", "stop", "target"))
        drag = st["cost_drag_same_fills"]
        per = drag / st["round_trips"] if st["round_trips"] else 0.0
        w(f"| {lever} | {cost} | {st['round_trips']:,} | {st['trades_per_session']:.2f} | "
          f"{st['pnl_after_costs']:,.0f} | {st['total_return']:+.1%} | {st['profit_factor']:.2f} | "
          f"{st['win_rate']:.1%} vs {st['break_even_win_rate']:.1%} | {st['max_drawdown']:.1%} | "
          f"{drag:,.0f} | {per:.2f} | {exs} | {st['expectancy']:,.1f} | "
          f"{st['folds_positive']}/{st['folds_total']} |")
    w()
    w("Baseline sanity: round trips, net %, PF, win rate, max DD and cost drag must match the "
      "published baseline (6,477 / −80.4% / 0.76 / 43.5% / 81.0% / $74,620) for the $ rows and "
      "(−5.7% / 0.99 / 50.6% / 32.5%) for the zero-cost row.")
    w()

    # ── lever ranking ─────────────────────────────────────────────────
    w("### Lever ranking (change vs baseline, per round trip and per session)")
    w()
    w("| lever | costs | trips Δ | net $ Δ | expectancy $/trip (Δ vs base) | drag $/trip (Δ) | pos folds Δ |")
    w("|---|---|---|---|---|---|---|")
    for cost in ("$", "0"):
        b = next(st for (lv, c, st) in runs.values() if c == cost and lv.startswith("baseline"))
        b_exp = b["expectancy"]
        b_drag = b["cost_drag_same_fills"] / b["round_trips"] if b["round_trips"] else 0.0
        for tag, (lever, c, st) in runs.items():
            if c != cost or lever.startswith("baseline"):
                continue
            drag = st["cost_drag_same_fills"] / st["round_trips"] if st["round_trips"] else 0.0
            w(f"| {lever} | {c} | {st['round_trips'] - b['round_trips']:+,} | "
              f"{st['pnl_after_costs'] - b['pnl_after_costs']:+,.0f} | "
              f"{st['expectancy']:,.1f} ({st['expectancy'] - b_exp:+,.1f}) | "
              f"{drag:.2f} ({drag - b_drag:+.2f}) | "
              f"{st['folds_positive'] - b['folds_positive']:+d} |")
    w()

    # ── compounding-free per-trade edge (the honest "is there an edge" test) ──
    w("### Per-trade expectancy and its significance (compounding-free)")
    w()
    w("Sizing is 50% of *equity*, so the equity path differs between variants and a `net %` column "
      "mixes signal quality with position size. `mean ret/trade` is `(exit_fill/entry_fill - 1)` "
      "averaged over the round trips, which has no compounding in it; the t-statistic is "
      "`mean / (sd/sqrt(n))` over round trips (treating them as independent, which they are not — "
      "overlapping positions, so treat |t| as an upper bound on significance).")
    w()
    w("| variant | costs | trips | mean ret/trade (bps) | sd (bps) | t | win % vs break-even |")
    w("|---|---|---|---|---|---|---|")
    for tag, (lever, cost, st) in runs.items():
        tr = OUT / f"turbo_trades_{tag}.csv"
        if not tr.exists():
            continue
        t = pd.read_csv(tr)
        if not len(t):
            continue
        r = t["ret_pct"].to_numpy(dtype=float) * 1e4
        m, sd = float(r.mean()), float(r.std(ddof=1))
        n = len(r)
        tstat = m / (sd / n ** 0.5) if sd and n > 1 else float("nan")
        w(f"| {lever} | {cost} | {n:,} | {m:+.2f} | {sd:,.0f} | {tstat:+.2f} | "
          f"{st['win_rate']:.1%} vs {st['break_even_win_rate']:.1%} |")
    w()

    # ── fold matrix ───────────────────────────────────────────────────
    for cost in ("$", "0"):
        w(f"### Fold-by-fold net % per monthly fold (costs = {cost})")
        w()
        months: list[str] = []
        for tag, (lever, c, st) in runs.items():
            if c != cost:
                continue
            f = st["_folds"]
            if len(f):
                months = [m for m in f["month"]]
                break
        if not months:
            w("_no folds_")
            continue
        w("| variant | " + " | ".join(m[-2:] for m in months) + " | pos |")
        w("|---" * (len(months) + 2) + "|")
        for tag, (lever, c, st) in runs.items():
            if c != cost:
                continue
            f = st["_folds"].set_index("month")["return_pct"] if len(st["_folds"]) else pd.Series(dtype=float)
            cells = []
            for m in months:
                if m in f.index:
                    v = float(f.loc[m])
                    cells.append(f"{v:+.1%}")
                else:
                    cells.append("—")
            w(f"| {lever} | " + " | ".join(cells) + f" | {st['folds_positive']}/{st['folds_total']} |")
        w()
        # per-fold per-trade expectancy in $ helps see whether a fold is one lucky trade
        w(f"#### Fold detail, trips and $ per fold (costs = {cost})")
        w()
        w("| variant | " + " | ".join(m[-2:] for m in months) + " |")
        w("|---" * (len(months) + 1) + "|")
        for tag, (lever, c, st) in runs.items():
            if c != cost:
                continue
            f = st["_folds"].set_index("month") if len(st["_folds"]) else pd.DataFrame()
            cells = []
            for m in months:
                if len(f) and m in f.index:
                    cells.append(f"{int(f.loc[m, 'trades'])}t/{f.loc[m, 'pnl']:+,.0f}")
                else:
                    cells.append("—")
            w(f"| {lever} | " + " | ".join(cells) + " |")
        w()

    # ── counters ──────────────────────────────────────────────────────
    w("### Counters (skips and exits) per run")
    w()
    w("| variant | costs | skipped | exits |")
    w("|---|---|---|---|")
    for tag, (lever, cost, st) in runs.items():
        w(f"| {lever} | {cost} | {st['skipped']} | {st['exits']} |")
    w()
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
