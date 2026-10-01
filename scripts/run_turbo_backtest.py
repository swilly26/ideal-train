#!/usr/bin/env python3
"""Run the turbo-trader replay (classic / current logic) over cached 1m bars.

Writes, per (logic, variant):
    <out-dir>/turbo_trades_<tag>.csv      one row per round trip
    <out-dir>/turbo_equity_<tag>.csv      portfolio mark-to-market per 1m bar
    <out-dir>/turbo_per_symbol_<tag>.csv  P&L / win rate / PF per symbol
    <out-dir>/turbo_folds_monthly_<tag>.csv  per-calendar-month slice
    <out-dir>/turbo_stats_<tag>.json      config + metrics + counters
    <out-dir>/turbo_summary_<tag>.md      human-readable dump

``<tag>`` is built by :func:`artifact_tag` and always carries the logic, the
cost-model variant, every ``--set`` knob and the traded window (and the
``--tag`` label first, when given)::

    [<label>__]<logic>__<variant>[__<knob>-<value>...][__<start>_<end>]

Round-1 bug this fixes: the ``--tag`` value used to be the WHOLE filename, so
a ``--tag Vcap2`` *baseline* run silently overwrote the *zero-cost* run's
stats/trades/folds and the second half of an experiment was lost.

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
import dataclasses
import json
import sys
import time
from collections.abc import Iterable
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


def apply_overrides(cfg: TurboConfig, pairs: list[str]) -> TurboConfig:
    """Apply ``--set KEY=VALUE`` overrides to *cfg* (typed by the current value).

    Round 1 of the edge search varies one behaviour at a time, so the knobs are
    set from the command line rather than by editing a config constructor: the
    artifact then carries exactly what was run (the stats JSON holds the full
    config), and the file name says it too.
    """
    known = {f.name: f for f in dataclasses.fields(cfg)}
    updates: dict = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        if key not in known:
            raise SystemExit(f"--set: unknown TurboConfig field {key!r}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            updates[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            updates[key] = int(raw)
        elif isinstance(cur, float):
            updates[key] = float(raw)
        elif isinstance(cur, tuple):
            updates[key] = tuple(x.strip() for x in raw.split(",") if x.strip())
        else:
            updates[key] = raw
    return dataclasses.replace(cfg, **updates) if updates else cfg


def _slug(text: str) -> str:
    """Filename-safe form of *text* (keep alphanumerics, ``-``, ``_``, ``.``)."""
    out = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in str(text))
    return out.strip("-") or "x"


def artifact_tag(logic: str, variant: str, tag: str | None = None,
                 overrides: Iterable[str] = (),
                 window: tuple[str | None, str | None] | None = None) -> str:
    """Build the artifact filename tag from everything that identifies the run.

    ``<label>__<logic>__<variant>[__<knob>-<value>...][__<start>_<end>]`` —
    every part is always present (when it exists), so two runs that differ in
    *any* of logic, cost model, knobs or window can never write the same file.
    The part order is fixed and the pieces are deterministic (knobs sorted), so
    the same run always lands on the same name.
    """
    parts: list[str] = []
    if tag:
        parts.append(_slug(tag))
    parts.append(_slug(logic))
    parts.append(_slug(variant))
    if overrides:
        parts.append("__".join(sorted(_slug(o.replace("=", "-")) for o in overrides)))
    if window:
        start, end = window
        if start or end:
            parts.append(f"{_slug(start or 'start')}_{_slug(end or 'end')}")
    return "__".join(parts)


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
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override a TurboConfig field, repeatable "
                         "(e.g. --set window_tie=latest --set max_entries_per_session=2)")
    args = ap.parse_args(argv)

    cfg = build_config(args.logic, args.variant)
    cfg = apply_overrides(cfg, args.overrides)
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    cfg = cfg.__class__(**{**vars(cfg), "symbols": symbols})
    if args.overrides:
        print("knob overrides: " + " ".join(args.overrides), flush=True)

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

    tag = artifact_tag(args.logic, args.variant, tag=args.tag,
                       overrides=args.overrides, window=window or (args.start, args.end))
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
    stats["knob_overrides"] = list(args.overrides)
    stats["inert_flags"] = list(cfg.inert_flags())
    stats["variant"] = args.variant
    stats["logic"] = args.logic
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
