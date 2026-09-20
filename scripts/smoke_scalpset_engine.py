#!/usr/bin/env python3
"""Smoke run for the ScalpSet replay engine — one symbol, one month.

Usage::

    .venv/bin/python scripts/smoke_scalpset_engine.py --symbol NVDA \
        --start 2026-08-01 --end 2026-09-01

Reads the cached 1m history (``data/history``), replays the live ScalpSet
(ICT IFVG / Box Theory / VolProfile+FIB) with live semantics and prints the
execution summary + performance metrics.  Network-free (cache only): this is
the end-to-end proof that the engine runs; the full multi-symbol backtest and
its report are a separate step.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.scalpset_engine import ScalpSetConfig, replay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--warmup-days", type=int, default=75)
    ap.add_argument("--cost-variant", action="store_true",
                    help="apply the pessimistic cost model (2bps + half-spread + fees)")
    ap.add_argument("--trades-csv", default=None, help="optional path for the trades table")
    args = ap.parse_args()

    cfg = ScalpSetConfig.cost_variant() if args.cost_variant else ScalpSetConfig()
    t0 = time.time()
    res = replay([args.symbol], start=args.start, end=args.end,
                 warmup_days=args.warmup_days, config=cfg)
    elapsed = time.time() - t0
    s = res.stats
    m = res.metrics()

    print(f"ScalpSet replay — {args.symbol} {args.start}..{args.end}"
          f" ({'cost variant' if args.cost_variant else 'no-cost baseline'})")
    print(f"  bars={s['bars']}  sessions={s['sessions']}  elapsed={elapsed:.1f}s")
    print(f"  signals by module : {s['signals_by_module']}")
    print(f"  entries           : {s['entries']} "
          f"(market={s['market_entries']}, limit={s['limit_orders_placed']}) "
          f"by module={s['entries_by_module']} by side={s['entries_by_side']}")
    print(f"  limit fills       : filled={s['limit_orders_filled']} "
          f"expired={s['limit_orders_expired']}")
    print(f"  exits             : {s['exits']}")
    print(f"  skipped           : {s['skipped']}")
    print(f"  TRADES            : {len(res.trades)}")
    print(f"  net P&L           : ${res.trades['pnl_after_costs'].sum():,.2f} "
          f"(fees ${s['fees_paid']:,.2f})")
    print(f"  final equity      : ${s['final_equity']:,.2f}")
    print(f"  total_return={m['total_return']:.2%} win_rate={m['win_rate']:.1%} "
          f"profit_factor={m['profit_factor']:.2f} max_dd={m['max_drawdown']:.2%} "
          f"sharpe={m['sharpe_ratio']:.2f}")
    if not res.trades.empty:
        by_reason = res.trades.groupby("exit_reason")["pnl_after_costs"].agg(["count", "sum"])
        print("  by exit reason:")
        for reason, row in by_reason.iterrows():
            print(f"    {reason:<4} n={int(row['count'])} pnl=${row['sum']:,.2f}")
    if args.trades_csv:
        res.trades.to_csv(args.trades_csv, index=False)
        print(f"  trades written to {args.trades_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
