#!/usr/bin/env python3
"""Run the turbo-trader replay (classic / current logic) over cached 1m bars.

Writes, per (logic, variant):
    <out-dir>/turbo_trades_<logic>_<variant>.csv      one row per round trip
    <out-dir>/turbo_equity_<logic>_<variant>.csv      portfolio mark-to-market per 1m bar
    <out-dir>/turbo_per_symbol_<logic>_<variant>.csv  P&L / win rate / PF per symbol
    <out-dir>/turbo_folds_monthly_<logic>_<variant>.csv  per-calendar-month slice
    <out-dir>/turbo_stats_<logic>_<variant>.json      config + metrics + counters
    <out-dir>/turbo_summary_<logic>_<variant>.md      human-readable dump

Usage (from the tree under test)::

    env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY \\
        .venv/bin/python scripts/run_turbo_backtest.py \\
        --logic classic --variant baseline \\
        --start 2025-09-01 --end 2026-09-01 --out-dir data/backtest_out

``--window`` restricts trading to a short period while still loading history
so the indicator windows are warm (used for the live-fidelity check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.turbo_engine import (  # noqa: E402
    TurboConfig,
    load_frames,
    replay,
)

LOGICS = ("classic", "current")
VARIANTS = ("baseline", "zero_cost", "pessimistic")


def build_config(logic: str, variant: str) -> TurboConfig:
    from dataclasses import replace
    cfg = TurboConfig.classic() if logic == "classic" else TurboConfig.current_base4()
    if variant == "baseline":
        return cfg
    if variant == "zero_cost":
        return cfg.zero_cost()
    if variant == "pessimistic":
        return replace(cfg, slippage_pct=0.0002, slippage_abs=0.01,
                       half_spread_pct=0.0001, commission_per_share=0.005)
    raise SystemExit(f"unknown variant: {variant}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logic", choices=LOGICS, default="classic")
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    ap.add_argument("--symbols", default="SOXL,TQQQ,FNGU,SPXL")
    ap.add_argument("--start", default="2025-09-01", help="first bar loaded (warmup included)")
    ap.add_argument("--end", default="2026-09-01", help="exclusive end of loaded bars")
    ap.add_argument("--window-start", default=None,
                    help="first TRADED timestamp (defaults to --start)")
    ap.add_argument("--window-end", default=None,
                    help="last TRADED timestamp (defaults to --end)")
    ap.add_argument("--out-dir", default="data/backtest_out")
    ap.add_argument("--tag", default=None, help="override the artifact filename tag")
    args = ap.parse_args(argv)

    cfg = build_config(args.logic, args.variant)
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    cfg = cfg.__class__(**{**vars(cfg), "symbols": symbols})

    t0 = time.time()
    frames = load_frames(symbols, start=args.start, end=args.end)
    missing = [s for s in symbols if s not in frames]
    if missing:
        print(f"FATAL: no cached bars for {missing} (run fetch_scalpset_history.py)",
              file=sys.stderr)
        return 2
    print(f"loaded {len(frames)} symbols in {time.time()-t0:.1f}s: "
          f"{ {s: len(f) for s, f in frames.items()} }", flush=True)

    window = None
    if args.window_start or args.window_end:
        window = (args.window_start or args.start, args.window_end or args.end)

    t1 = time.time()
    res = replay(frames, cfg, trade_window=window)
    elapsed = time.time() - t1

    tag = args.tag or f"{args.logic}_{args.variant}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / f"turbo_trades_{tag}.csv", index=False)
    res.equity_curve.to_csv(out / f"turbo_equity_{tag}.csv")
    if len(res.per_symbol):
        res.per_symbol.to_csv(out / f"turbo_per_symbol_{tag}.csv")
    if len(res.folds_monthly):
        res.folds_monthly.to_csv(out / f"turbo_folds_monthly_{tag}.csv", index=False)

    stats = dict(res.stats)
    stats["config"] = {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in vars(cfg).items()}
    stats["window"] = list(window) if window else [args.start, args.end]
    stats["replay_seconds"] = round(elapsed, 1)
    (out / f"turbo_stats_{tag}.json").write_text(json.dumps(stats, indent=2, default=str))

    md = [f"# turbo replay — logic={args.logic} variant={args.variant}", "",
          f"window: {stats['window'][0]} .. {stats['window'][1]} | "
          f"bars={stats['bars']} sessions={stats['sessions']} | "
          f"replay {elapsed:.1f}s", ""]
    md.append("## headline")
    md.append(f"- round trips: **{stats['round_trips']}** ({stats['trades_per_session']:.2f}/session)")
    md.append(f"- net P&L: **${stats['pnl_after_costs']:,.0f}** "
              f"({stats['total_return']:+.2%} on ${stats['initial_equity']:,.0f})")
    md.append(f"- gross P&L (same fills, no costs): ${stats['pnl_gross']:,.0f}")
    md.append(f"- profit factor: {stats['profit_factor']:.2f} | win rate "
              f"{stats['win_rate']:.1%} vs break-even {stats['break_even_win_rate']:.1%}")
    md.append(f"- max drawdown: {stats['max_drawdown']:.1%} | session Sharpe "
              f"{stats['session_sharpe']:.2f} | expectancy ${stats['expectancy']:,.0f}/trade")
    md.append(f"- monthly folds: {stats['folds_positive']}/{stats['folds_total']} positive")
    md.append(f"- exits: {stats['exits']} | skipped: {stats['skipped']} | "
              f"fees ${stats['fees_paid']:,.0f}")
    md.append("")
    md.append("## per symbol")
    md.append(_df_md(res.per_symbol) if len(res.per_symbol) else "_no trades_")
    md.append("")
    md.append("## monthly folds")
    md.append(_df_md(res.folds_monthly) if len(res.folds_monthly) else "_no trades_")
    md.append("")
    (out / f"turbo_summary_{tag}.md").write_text("\n".join(md))

    print(res.summary())
    print(f"artifacts -> {out}/turbo_*_{tag}.* (replay {elapsed:.1f}s)")
    return 0


def _df_md(df: pd.DataFrame) -> str:
    """Hand-rolled markdown table (the engine venv has no tabulate)."""
    if df is None or len(df) == 0:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
