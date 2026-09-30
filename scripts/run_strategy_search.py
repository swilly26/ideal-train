#!/usr/bin/env python3
"""Stage-1 strategy search: run the declared grids through the screen.

    # one config, one window (a single replay)
    .venv/bin/python scripts/run_strategy_search.py --family A \
        --config a_trend_trail50_reg --window W1 --sizing fixed_notional

    # the full pre-registered screen for one or more families (order A, C, D, B)
    .venv/bin/python scripts/run_strategy_search.py --screen --family A

Every run writes one machine-readable record to ``--out-dir``; the screen writes
a family verdict JSON next to them.  Nothing is adopted here: stage 1 only
decides which configs deserve stage 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import families, screen  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Market,
    per_symbol,
    monthly_folds,
    run_search,
)
from src.backtesting.strategy_search.features import (  # noqa: E402
    DEFAULT_CACHE,
    FeatureBook,
    coverage,
    load_window,
)

WINDOWS = {
    "W1": ("2025-09-01", "2026-09-01"),   # development
    "W2": ("2024-09-01", "2025-09-01"),   # confirmation
}
DEFAULT_OUT = Path("/home/team/shared/strategy_search/stage1")
FAMILY_SYMBOLS = ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")
FAMILY_ORDER = ("A", "C", "D", "B")


class Runner:
    """Loads a window's bars + features once, then replays configs against it."""

    def __init__(self, cache_dir: Path | str, out_dir: Path, verbose: bool = True
                 ) -> None:
        self.cache_dir = Path(cache_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._cache: dict[str, dict] = {}

    def _window(self, window: str) -> dict:
        if window not in self._cache:
            start, end = WINDOWS[window]
            frames = load_window(FAMILY_SYMBOLS, start, end, self.cache_dir)
            missing = [s for s in FAMILY_SYMBOLS if s not in frames]
            if missing:
                raise SystemExit(f"no cached bars for {missing} under {self.cache_dir}")
            market = Market(frames)
            book = FeatureBook(frames)
            aligned = book.align(market.axis)
            self._cache[window] = {
                "frames": frames, "market": market, "aligned": aligned,
                "cov": coverage(frames), "dates": [start, end],
            }
            if self.verbose:
                print(f"[{window}] axis={market.n():,} bars "
                      f"{market.axis[0]} .. {market.axis[-1]} "
                      f"symbols={sorted(frames)}", flush=True)
        return self._cache[window]

    def run(self, family: str, spec: dict, window: str, cost_label: str,
            sizing: str) -> dict:
        w = self._window(window)
        market: Market = w["market"]
        spec = dict(spec)
        spec.setdefault("symbols", list(FAMILY_SYMBOLS[:-1]))
        cfg, instruments = families.build(family, spec, market, w["aligned"])
        cfg = dataclasses.replace(cfg, sizing=sizing)
        costs = CostModel.baseline() if cost_label == "baseline" else CostModel.zero_cost()
        costs = dataclasses.replace(costs, label=cost_label)
        t0 = time.time()
        res = run_search(market, instruments, cfg, costs)
        elapsed = time.time() - t0
        stats = res.stats
        folds = monthly_folds(res.trades)
        persym = per_symbol(res.trades)
        stats["folds_positive"] = int((folds["net_bps_per_trip"] > 0).sum()) \
            if len(folds) else 0
        stats["folds_total"] = int(len(folds))
        folds = folds[[c for c in folds.columns]]
        record = {
            "family": family,
            "config": spec["name"],
            "window": window,
            "window_dates": w["dates"],
            "cost_model": costs.as_dict(),
            "sizing": sizing,
            "notional_usd": cfg.notional_usd,
            "position_size_pct": cfg.position_size_pct,
            "config_params": cfg.params(),
            "signal_params": spec["signal"],
            "symbols": list(spec["symbols"]),
            "coverage": w["cov"],
            "trips": stats["round_trips"],
            "trips_per_session": stats["trips_per_session"],
            "net_pnl": stats["pnl_after_costs"],
            "net_pct": stats["total_return"],
            "final_equity": stats["final_equity"],
            "gross_same_fills": stats["pnl_gross_same_fills"],
            "zero_cost_same_fills": stats["pnl_zero_cost_same_fills"],
            "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"],
            "break_even_win_rate": stats["break_even_win_rate"],
            "max_drawdown": stats["max_drawdown"],
            "net_bps_per_trip": stats["net_bps_per_trip"],
            "net_bps_per_trip_sd": stats["net_bps_per_trip_sd"],
            "net_bps_t_stat": stats["net_bps_t_stat"],
            "cost_drag_per_trip": stats["cost_drag_per_trip"],
            "cost_drag_per_trip_bps": (stats["cost_drag_per_trip"]
                                       / stats["notional_per_trip"] * 1e4
                                       if stats["notional_per_trip"] else 0.0),
            "cost_drag_total": stats["cost_drag_total"],
            "fees_paid": stats["fees_paid"],
            "notional_per_trip": stats["notional_per_trip"],
            "avg_hold_minutes": stats["avg_hold_minutes"],
            "median_hold_minutes": stats["median_hold_minutes"],
            "folds_positive": stats["folds_positive"],
            "folds_total": stats["folds_total"],
            "folds": json.loads(folds.to_json(orient="records")),
            "per_symbol": json.loads(persym.to_json(orient="records")),
            "exits": stats["exits"],
            "skipped": stats["skipped"],
            "entries": stats["entries"],
            "signals": stats["signals"],
            "runtime_s": round(elapsed, 2),
        }
        name = f"{family}__{spec['name']}__{window}__{cost_label}__{sizing}.json"
        (self.out_dir / name).write_text(json.dumps(record, indent=2, default=str))
        if self.verbose:
            print(f"  {spec['name']:<26} {window} {cost_label:<10} {sizing:<16} "
                  f"trips={stats['round_trips']:>5} net=${stats['pnl_after_costs']:>12,.0f} "
                  f"({stats['total_return']:+.1%}) bps={stats['net_bps_per_trip']:+.2f} "
                  f"PF={stats['profit_factor']:.2f} {elapsed:.1f}s", flush=True)
        return record


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="A", help="A, C, D, B or a comma list")
    ap.add_argument("--config", default=None, help="single config name (single run)")
    ap.add_argument("--window", default="W1", choices=sorted(WINDOWS))
    ap.add_argument("--sizing", default="fixed_notional", choices=screen.SIZING_MODES)
    ap.add_argument("--cost", default="baseline", choices=("baseline", "zero_cost"))
    ap.add_argument("--screen", action="store_true",
                    help="run the pre-registered screen for the whole family")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    runner = Runner(args.cache_dir, Path(args.out_dir), verbose=not args.quiet)
    fams = [f.strip().upper() for f in args.family.split(",") if f.strip()]
    if not args.screen:
        fam = fams[0]
        specs = families.GRIDS[fam]
        if args.config:
            specs = [s for s in specs if s["name"] == args.config]
            if not specs:
                raise SystemExit(f"no config {args.config!r} in family {fam}")
        for spec in specs:
            runner.run(fam, spec, args.window, args.cost, args.sizing)
        return 0

    verdicts = []
    for fam in fams:
        specs = families.GRIDS[fam]
        print(f"=== family {fam}: screening {len(specs)} configs on W1 ===", flush=True)
        verdict = screen.screen_family(
            fam, specs,
            lambda spec, window, cost, sizing, fam=fam:
                runner.run(fam, spec, window, cost, sizing),
            log=print if not args.quiet else (lambda *_: None))
        verdict["runs"] = {k: _trim(v) for k, v in verdict["runs"].items()}
        (Path(args.out_dir) / f"screen_{fam}.json").write_text(
            json.dumps(verdict, indent=2, default=str))
        print(f"=== family {fam}: {verdict['verdict']} "
              f"(w1: {verdict['w1_survivors']}, w2: {verdict['w2_survivors']}) ===",
              flush=True)
        verdicts.append(verdict)
    summary = [{k: v for k, v in ver.items() if k != "runs"} for ver in verdicts]
    (Path(args.out_dir) / "screen_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return 0


def _trim(rec: dict) -> dict:
    """Keep the screen's verdicts and headline numbers in the family JSON."""
    keep = ("family", "config", "window", "sizing", "trips", "net_pnl", "net_pct",
            "net_bps_per_trip", "net_bps_t_stat", "cost_drag_per_trip",
            "cost_drag_per_trip_bps", "profit_factor", "win_rate",
            "break_even_win_rate", "max_drawdown", "folds_positive", "folds_total",
            "avg_hold_minutes", "trips_per_session", "w1_gate", "w2_gate",
            "signal_params", "config_params")
    return {k: v for k, v in rec.items() if k in keep}


if __name__ == "__main__":
    raise SystemExit(main())
