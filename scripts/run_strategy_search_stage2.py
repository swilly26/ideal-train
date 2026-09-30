"""Stage-2 deep validation of the one stage-1 survivor (brief §1–§4).

Everything measurement-only: no adoption, no live stack.  One replay at a time.

  nice -n 19 .venv/bin/python scripts/run_strategy_search_stage2.py

Writes JSON/CSV artifacts under /home/team/shared/strategy_search/stage2/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel            # noqa: E402
from src.backtesting.strategy_search import families          # noqa: E402
from src.backtesting.strategy_search.engine import (          # noqa: E402
    Market, SearchConfig, run_search)
from src.backtesting.strategy_search.features import (        # noqa: E402
    DEFAULT_CACHE, FeatureBook, coverage, load_window)        # noqa: E402

WINDOWS = {"W1": ("2025-09-01", "2026-09-01"),      # development
           "W2": ("2024-09-01", "2025-09-01")}      # confirmation
SYMBOLS = ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")
SURVIVOR = "d_pair_z15_soxl_tqqq"
OUT = Path("/home/team/shared/strategy_search/stage2")
SEED = 20260930
N_BOOT = 4000


def spec_for(name: str) -> dict:
    return next(s for s in families.GRIDS["D"] if s["name"] == name)


def variant(base: dict, **signal_over) -> dict:
    s = json.loads(json.dumps(base))
    s["signal"].update(signal_over)
    return s


def params_variant(base: dict, **params_over) -> dict:
    s = json.loads(json.dumps(base))
    s["params"].update(params_over)
    return s


class Windows:
    def __init__(self, cache: Path | str, verbose: bool = True):
        self.cache = Path(cache)
        self.verbose = verbose
        self._c: dict[str, dict] = {}

    def get(self, w: str) -> dict:
        if w not in self._c:
            start, end = WINDOWS[w]
            frames = load_window(SYMBOLS, start, end, self.cache)
            market = Market(frames)
            aligned = FeatureBook(frames).align(market.axis)
            self._c[w] = {"market": market, "aligned": aligned,
                          "dates": [start, end], "coverage": coverage(frames)}
            if self.verbose:
                print(f"[{w}] {market.n():,} bars {market.axis[0]} .. "
                      f"{market.axis[-1]}", flush=True)
        return self._c[w]

    def run(self, spec: dict, w: str, sizing: str = "fixed_notional",
            costs: CostModel | None = None, tag: str = "") -> tuple:
        win = self.get(w)
        cfg, insts = families.build("D", dict(spec), win["market"],
                                    win["aligned"])
        cfg = dataclasses.replace(cfg, sizing=sizing)
        costs = costs if costs is not None else CostModel.baseline()
        t0 = time.time()
        res = run_search(win["market"], insts, cfg, costs)
        if self.verbose and tag:
            print(f"  {tag:<34} {w} trips={res.stats['round_trips']:>4} "
                  f"bps={res.stats['net_bps_per_trip']:+.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return res, cfg, costs


def bps_per_trip(trades: pd.DataFrame) -> float:
    if not len(trades):
        return 0.0
    return float((trades["pnl_after_costs"] / trades["notional"] * 1e4).mean())


def summary(res, cfg, costs) -> dict:
    s = res.stats
    t = res.trades
    out = {
        "trips": int(s["round_trips"]),
        "net_pnl": float(s["pnl_after_costs"]),
        "net_pct": float(s["total_return"]),
        "net_bps_per_trip": float(s["net_bps_per_trip"]),
        "net_bps_t_stat": float(s["net_bps_t_stat"]),
        "net_bps_sd": float(s["net_bps_per_trip_sd"]),
        "profit_factor": float(s["profit_factor"]),
        "win_rate": float(s["win_rate"]),
        "break_even_win_rate": float(s["break_even_win_rate"]),
        "max_drawdown": float(s["max_drawdown"]),
        "cost_drag_per_trip": float(s["cost_drag_per_trip"]),
        "cost_drag_per_trip_bps": float(s["cost_drag_per_trip"]
                                        / (s["notional_per_trip"] or 1.0) * 1e4),
        "avg_hold_minutes": float(s["avg_hold_minutes"]),
        "exit_reasons": dict(s["exits"]),
        "sessions": int(s["sessions"]),
        "cost_model": costs.as_dict(),
        "sizing": cfg.sizing,
    }
    # the strategy's own (zero-cost) P&L, on the same fills
    own = t["pnl_gross"] + t["cost_drag"] if len(t) else pd.Series(dtype=float)
    out["zero_cost_pnl"] = float(own.sum())
    out["zero_cost_bps_per_trip"] = (float((own / t["notional"] * 1e4).mean())
                                     if len(t) else 0.0)
    if len(t):
        out["direction_split"] = (
            t.assign(d=np.where(t["symbols"].str.contains("SOXL\\+"), 1, -1))
             .groupby("d").agg(trips=("net_bps", "size"),
                               net_bps=("net_bps", "mean")).to_dict("index"))
    # break-even toll multiple: net = own - k * drag  ->  k* = mean(own)/mean(drag)
    if len(t) and t["cost_drag"].mean() > 0:
        out["break_even_toll_multiple"] = float(
            (t["pnl_gross"] + t["cost_drag"]).mean() / t["cost_drag"].mean())
    else:
        out["break_even_toll_multiple"] = None
    return out


def folds_table(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.strftime("%Y-%m")
    t["own"] = t["pnl_gross"] + t["cost_drag"]
    t["net_bps"] = t["pnl_after_costs"] / t["notional"] * 1e4
    t["own_bps"] = t["own"] / t["notional"] * 1e4
    g = t.groupby("month", sort=True)
    return pd.DataFrame({
        "trips": g.size(),
        "net_pnl": g["pnl_after_costs"].sum(),
        "gross_pnl": g["pnl_gross"].sum(),
        "zero_cost_pnl": g["own"].sum(),
        "net_bps_per_trip": g["net_bps"].mean(),
        "zero_cost_bps_per_trip": g["own_bps"].mean(),
    }).reset_index()


def bootstrap(trades: pd.DataFrame, seed: int = SEED, n: int = N_BOOT) -> dict:
    """Session-block bootstrap (resample whole sessions) + i.i.d. trip bootstrap."""
    t = trades.copy()
    t["day"] = pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m-%d")
    t["bps"] = t["pnl_after_costs"] / t["notional"] * 1e4
    rng = np.random.default_rng(seed)
    days = t["day"].unique()
    by_day = [g["bps"].to_numpy() for _, g in t.groupby("day")]
    blocks = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(by_day), len(by_day))
        blocks[i] = np.concatenate([by_day[j] for j in idx]).mean()
    x = t["bps"].to_numpy()
    iid = np.array([rng.choice(x, size=len(x), replace=True).mean()
                    for _ in range(n)])
    return {
        "design": "session-block bootstrap over whole sessions; i.i.d. trip "
                  "bootstrap as a cross-check",
        "seed": seed, "resamples": n,
        "trips": int(len(t)), "sessions": int(len(days)),
        "mean_bps": float(x.mean()),
        "block_ci95": [float(np.percentile(blocks, 2.5)),
                       float(np.percentile(blocks, 97.5))],
        "iid_ci95": [float(np.percentile(iid, 2.5)),
                     float(np.percentile(iid, 97.5))],
        "block_frac_positive": float((blocks > 0).mean()),
    }


def walk_forward(trades: pd.DataFrame, test_months: int = 3) -> dict:
    """Rolling-origin stability: the sign of the edge over each 3-month block."""
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M")
    t["net_bps"] = t["pnl_after_costs"] / t["notional"] * 1e4
    t["own_bps"] = (t["pnl_gross"] + t["cost_drag"]) / t["notional"] * 1e4
    months = sorted(t["month"].unique())
    blocks = []
    step = 1
    for i in range(0, max(len(months) - test_months + 1, 0), step):
        sel = months[i:i + test_months]
        sub = t[t["month"].isin(sel)]
        if not len(sub):
            continue
        blocks.append({
            "test_months": f"{sel[0]}..{sel[-1]}", "trips": int(len(sub)),
            "net_bps_per_trip": float(sub["net_bps"].mean()),
            "zero_cost_bps_per_trip": float(sub["own_bps"].mean()),
            "net_pnl": float(sub["pnl_after_costs"].sum()),
        })
    pos = sum(1 for b in blocks if b["net_bps_per_trip"] > 0)
    return {"design": f"rolling {test_months}-month test blocks, rolled monthly, "
                      f"no fitting (the config is fixed)",
            "blocks": blocks, "blocks_positive": pos, "blocks_total": len(blocks)}


def concentration(trades: pd.DataFrame) -> dict:
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.strftime("%Y-%m")
    t["net_bps"] = t["pnl_after_costs"] / t["notional"] * 1e4
    tot = float(t["pnl_after_costs"].sum())
    g = t.groupby("month")["pnl_after_costs"].sum().sort_values()
    best, worst = g.index[-1], g.index[0]
    top10 = t.nlargest(10, "pnl_after_costs")["pnl_after_costs"].sum()
    drop = lambda m: (float(t[t["month"] != m]["pnl_after_costs"].sum()),
                      float(t[t["month"] != m]["net_bps"].mean()))
    return {
        "total_net_pnl": tot, "total_net_bps_per_trip": float(t["net_bps"].mean()),
        "best_month": {"month": best, "net_pnl": float(g.loc[best]),
                       "share_of_total": float(g.loc[best] / tot)},
        "worst_month": {"month": worst, "net_pnl": float(g.loc[worst]),
                        "share_of_total": float(g.loc[worst] / tot)},
        "top10_trips_pnl": float(top10),
        "top10_share_of_total": float(top10 / tot),
        "minus_best_month": drop(best), "minus_worst_month": drop(worst),
        "net_folds_positive": int((g > 0).sum()), "folds_total": int(len(g)),
    }


def stress_costs():
    base = CostModel.baseline()
    out = {}
    out["pessimistic"] = CostModel.pessimistic()
    out["slip_x1.5"] = dataclasses.replace(
        base, slippage_pct=base.slippage_pct * 1.5,
        slippage_abs=base.slippage_abs * 1.5, label="slip_x1.5")
    out["slip_x2"] = dataclasses.replace(
        base, slippage_pct=base.slippage_pct * 2.0,
        slippage_abs=base.slippage_abs * 2.0, label="slip_x2")
    out["pessimistic_x2"] = dataclasses.replace(
        CostModel.pessimistic(), slippage_pct=0.0004, slippage_abs=0.02,
        label="pessimistic_x2")
    return out


def borrow_cost(trades: pd.DataFrame, rate: float) -> float:
    """Intraday-only short financing: one short leg per pair, calendar-time."""
    if not len(trades):
        return 0.0
    short_notional = trades["notional"] / 2.0        # one leg of the two
    hours = trades["hold_minutes"] / 60.0
    return float((short_notional * rate * hours / (24 * 365)).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-neighbours", action="store_true")
    ap.add_argument("--skip-stress", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wins = Windows(args.cache_dir)
    base = spec_for(SURVIVOR)
    report: dict = {"survivor": SURVIVOR, "seed": SEED,
                    "windows": WINDOWS, "symbols": list(SYMBOLS)}

    # ── 1. re-derivation (post-fix) + zero-cost + folds + stats ─────────
    rebuild: dict = {}
    for w in ("W1", "W2"):
        res, cfg, costs = wins.run(base, w, "fixed_notional", tag="rebuild")
        res_z, _, _ = wins.run(base, w, "fixed_notional",
                               CostModel.zero_cost(), tag="rebuild zero-cost")
        rec = summary(res, cfg, costs)
        rec["config_params"] = cfg.params()
        rec["signal_params"] = base["signal"]
        rebuild[w] = rec
        f = folds_table(res.trades)
        f.to_csv(out / f"stage2_folds_{w}.csv", index=False)
        res.trades.to_csv(out / f"stage2_trades_{w}.csv", index=False)
        rebuild[w]["folds"] = json.loads(f.to_json(orient="records"))
        rebuild[w]["bootstrap"] = bootstrap(res.trades)
        rebuild[w]["walk_forward"] = walk_forward(res.trades)
        rebuild[w]["concentration"] = concentration(res.trades)
        rebuild[w]["zero_cost_run"] = {
            "net_pnl": float(res_z.stats["pnl_after_costs"]),
            "net_pct": float(res_z.stats["total_return"]),
            "net_bps_per_trip": float(res_z.stats["net_bps_per_trip"]),
            "trips": int(res_z.stats["round_trips"])}
        print(f"  [{w}] rebuild netbps={rec['net_bps_per_trip']:+.3f} "
              f"t={rec['net_bps_t_stat']:+.2f} exits={rec['exit_reasons']}",
              flush=True)
    report["rebuild"] = rebuild

    # pooled bootstrap across both windows
    pooled = pd.concat([pd.read_csv(out / f"stage2_trades_{w}.csv")
                        for w in ("W1", "W2")], ignore_index=True)
    report["pooled_bootstrap"] = bootstrap(pooled)

    # ── 2. neighbour sweep (declared axes only, one at a time) ──────────
    if not args.skip_neighbours:
        axes: list[tuple[str, dict, dict]] = []
        for z in (1.25, 1.5, 1.75, 2.0):
            axes.append((f"z_entry={z}", variant(base, z_entry=z), {}))
        for ze in (0.0, 0.25, 0.5):
            axes.append((f"z_exit={ze}", variant(base, z_exit=ze), {}))
        for zw in (20, 30, 45):
            axes.append((f"z_window={zw}", variant(base, z_window=zw), {}))
        for g in (9 * 60 + 45, 10 * 60, 10 * 60 + 30):
            axes.append((f"gate_start={g // 60}:{g % 60:02d}",
                         params_variant(base, entry_start_min=g), {}))
        for cap in (1, 2):
            axes.append((f"max_entries={cap}",
                         params_variant(base, max_entries_per_session=cap), {}))
        axes.append(("sizing=equity_fraction", json.loads(json.dumps(base)), {}))
        rows = []
        for name, spec, _ in axes:
            sizing = ("equity_fraction" if name.startswith("sizing=")
                      else "fixed_notional")
            row = {"neighbour": name, "sizing": sizing}
            for w in ("W1", "W2"):
                res, cfg, costs = wins.run(spec, w, sizing, tag=name)
                s = summary(res, cfg, costs)
                row[f"{w}_trips"] = s["trips"]
                row[f"{w}_net_bps"] = s["net_bps_per_trip"]
                row[f"{w}_t"] = s["net_bps_t_stat"]
                row[f"{w}_net_pct"] = s["net_pct"]
                row[f"{w}_net_folds_positive"] = int(
                    (folds_table(res.trades)["net_bps_per_trip"] > 0).sum())
                row[f"{w}_zero_cost_bps"] = s["zero_cost_bps_per_trip"]
            rows.append(row)
        report["neighbours"] = rows
        (out / "stage2_neighbours.json").write_text(
            json.dumps(rows, indent=2, default=str))
        print(f"  neighbours: {sum(1 for r in rows if r['W1_net_bps'] > 0)}/"
              f"{len(rows)} positive on W1, "
              f"{sum(1 for r in rows if r['W2_net_bps'] > 0)}/{len(rows)} on W2",
              flush=True)

    # ── 3. cost stress ──────────────────────────────────────────────────
    if not args.skip_stress:
        stress = {}
        for label, cm in stress_costs().items():
            rec = {}
            for w in ("W1", "W2"):
                res, cfg, costs = wins.run(base, w, "fixed_notional", cm,
                                           tag=f"cost {label}")
                rec[w] = summary(res, cfg, costs)
                rec[w]["break_even_toll_multiple"] = rec[w][
                    "break_even_toll_multiple"]
            stress[label] = rec
            print(f"  cost {label:<16} W1={rec['W1']['net_bps_per_trip']:+.3f} "
                  f"W2={rec['W2']['net_bps_per_trip']:+.3f}", flush=True)
        # short borrow / financing on the short leg, intraday only
        borrow = {}
        for rate in (0.0, 0.01):
            rec = {}
            for w in ("W1", "W2"):
                t = pd.read_csv(out / f"stage2_trades_{w}.csv")
                bc = borrow_cost(t, rate)
                base_bps = float((t["pnl_after_costs"] / t["notional"] * 1e4).mean())
                bc_bps = bc / t["notional"].sum() / len(t) * 1e4 if len(t) else 0.0
                rec[w] = {"borrow_cost_total": bc, "borrow_bps_per_trip": bc_bps,
                          "net_bps_after_borrow": base_bps - bc_bps}
            borrow[f"rate={rate:.0%}"] = rec
        stress["short_borrow"] = borrow
        report["cost_stress"] = stress

    (out / "stage2_rebuild.json").write_text(json.dumps(report, indent=2,
                                                        default=str))
    print("wrote", out / "stage2_rebuild.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
