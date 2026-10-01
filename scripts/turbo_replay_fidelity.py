#!/usr/bin/env python3
"""Fidelity check: replay trades vs the real broker fills for the same window.

This is the gate for trusting a long replay: if a short replay of the period
the live trader actually traded does not look like what the broker did, the
long number is not evidence.

Inputs
------
``--fills``        broker FILL activities as JSON (Alpaca
                   ``/v2/account/activities?activity_types=FILL``).  Fills are
                   first collapsed to *orders* (one BUY order = one entry)
                   because one order arrives as many partial fills; FIFO is
                   then run over order-level vwaps.
``--replay-trades``  ``turbo_trades_*.csv`` from ``run_turbo_backtest.py``.
``--live-csv``     optional: reuse a previously written live round-trip CSV
                   instead of ``--fills``.

Outputs a markdown side-by-side plus, with ``--write-live-csv``, the live
round-trip CSV so the comparison can be re-checked without the raw fills.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

ET_OFFSET_H = 4  # EDT (UTC-4) — the whole live turbo window is in summer time


def order_level_round_trips(fills: list[dict]) -> pd.DataFrame:
    """Collapse partial fills per order, then FIFO into long round trips."""
    orders: dict[str, dict] = {}
    for f in fills:
        oid = f["order_id"]
        qty = float(f["qty"])
        px = float(f["price"])
        o = orders.setdefault(oid, {"symbol": f["symbol"], "side": f["side"],
                                    "qty": 0.0, "notional": 0.0,
                                    "t0": f["transaction_time"],
                                    "t1": f["transaction_time"]})
        o["qty"] += qty
        o["notional"] += qty * px
        o["t0"] = min(o["t0"], f["transaction_time"])
        o["t1"] = max(o["t1"], f["transaction_time"])
    for o in orders.values():
        o["price"] = o["notional"] / o["qty"] if o["qty"] else 0.0

    bysym: dict[str, list[dict]] = collections.defaultdict(list)
    for o in orders.values():
        bysym[o["symbol"]].append(o)

    rows = []
    for sym, os_ in bysym.items():
        os_ = sorted(os_, key=lambda z: z["t0"])
        longs: list[list] = []
        shorts: list[list] = []
        for o in os_:
            rem = o["qty"]
            opp = shorts if o["side"] == "buy" else longs
            while rem > 1e-9 and opp:
                lot = opp[0]
                take = min(lot[0], rem)
                direction = "long" if opp is longs else "short"
                oqty, oprice, ot = lot[0], lot[1], lot[2]
                if direction == "long" and o["side"] == "sell":
                    pnl = (o["price"] - oprice) * take
                elif direction == "short" and o["side"] == "buy":
                    pnl = (oprice - o["price"]) * take
                else:  # same side — cannot close; ignore
                    raise AssertionError(f"unexpected fifo pairing for {sym}")
                rows.append(dict(symbol=sym, dir=direction, qty=take,
                                 entry=oprice, exit=o["price"], pnl=pnl,
                                 t_in=ot, t_out=o["t1"]))
                lot[0] -= take
                rem -= take
                if lot[0] <= 1e-9:
                    opp.pop(0)
            if rem > 1e-9:
                (longs if o["side"] == "buy" else shorts).append([rem, o["price"], o["t0"], o["t1"]])
    df = pd.DataFrame(rows)
    if len(df):
        df["t_in"] = pd.to_datetime(df["t_in"], utc=True)
        df["t_out"] = pd.to_datetime(df["t_out"], utc=True)
        # broker timestamps are UTC; convert to exchange (ET) naive local time
        for c in ("in", "out"):
            df[f"et_{c}"] = (df[f"t_{c}"] - pd.Timedelta(hours=ET_OFFSET_H)).dt.tz_localize(None)
        df["hold_minutes"] = (df["t_out"] - df["t_in"]).dt.total_seconds() / 60.0
        df = df.sort_values("t_out").reset_index(drop=True)
    return df


def _minutes(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.hour * 60 + pd.to_datetime(series).dt.minute


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def compare(live: pd.DataFrame, rep: pd.DataFrame, *, window: tuple[str, str],
            out_md: Path) -> str:
    lw = live[(live["et_out"].dt.strftime("%Y-%m-%d") >= window[0])
              & (live["et_out"].dt.strftime("%Y-%m-%d") <= window[1])]
    rw = rep.copy()
    if len(rw):
        rw["et_in"] = pd.to_datetime(rw["entry_time"])
        rw["et_out"] = pd.to_datetime(rw["exit_time"])
        rw = rw[(rw["et_out"].dt.strftime("%Y-%m-%d") >= window[0])
                & (rw["et_out"].dt.strftime("%Y-%m-%d") <= window[1])]

    live_sessions = lw["et_out"].dt.normalize().nunique() if len(lw) else 0
    rows = []

    def add(metric, a, b, note=""):
        rows.append((metric, a, b, note))

    add("round trips", f"{len(lw)}", f"{len(rw)}",
        "order-level FIFO vs replay trades")
    add("per session", f"{len(lw)/live_sessions:.1f}" if live_sessions else "—",
        f"{len(rw)/max(live_sessions,1):.1f}", f"live sessions with fills: {live_sessions}")
    add("net P&L", f"${lw['pnl'].sum():,.0f}" if len(lw) else "—",
        f"${rw['pnl_after_costs'].sum():,.0f}" if len(rw) else "—", "broker fills vs replay")
    add("win rate",
        _pct(float((lw["pnl"] > 0).mean())) if len(lw) else "—",
        _pct(float((rw["pnl_after_costs"] > 0).mean())) if len(rw) else "—", "")
    add("median hold (min)",
        f"{lw['hold_minutes'].median():.0f}" if len(lw) else "—",
        f"{rw['hold_minutes'].median():.0f}" if len(rw) else "—",
        "same-session live holds only" if len(lw) else "")
    add("mean hold (same session)",
        f"{lw.loc[lw['hold_minutes'] < 390, 'hold_minutes'].mean():.0f}" if len(lw) else "—",
        f"{rw.loc[rw['hold_minutes'] < 390, 'hold_minutes'].mean():.0f}" if len(rw) else "—",
        "multi-session live trades excluded")
    add("median entry notional",
        f"${(lw['qty']*lw['entry']).median():,.0f}" if len(lw) else "—",
        f"${(rw['qty']*rw['entry_price']).median():,.0f}" if len(rw) else "—",
        "live entry price is the fill vwap")
    add("entries before 10:15 ET",
        f"{int((_minutes(lw['et_in']) < 615).sum())} / {len(lw)}" if len(lw) else "—",
        f"{int((_minutes(rw['et_in']) < 615).sum())} / {len(rw)}" if len(rw) else "—",
        "min_bars=25 gate")
    add("entries after 15:30 ET",
        f"{int((_minutes(lw['et_in']) >= 930).sum())}" if len(lw) else "—",
        f"{int((_minutes(rw['et_in']) >= 930).sum())}" if len(rw) else "—",
        "EOD flatten must prevent these")

    sym_rows = []
    for sym in sorted(set(list(lw["symbol"]) + list(rw["symbol"]))):
        a = lw[lw["symbol"] == sym]
        b = rw[rw["symbol"] == sym]
        sym_rows.append((sym, len(a), f"${a['pnl'].sum():,.0f}" if len(a) else "$0",
                         len(b), f"${b['pnl_after_costs'].sum():,.0f}" if len(b) else "$0"))

    # entry-time-of-day buckets (30 min, ET)
    buckets = []
    for lo in range(9 * 60 + 30, 16 * 60, 30):
        hi = lo + 30
        lbl = f"{lo//60:02d}:{lo%60:02d}-{hi//60:02d}:{hi%60:02d}"
        a = int(((_minutes(lw["et_in"]) >= lo) & (_minutes(lw["et_in"]) < hi)).sum()) if len(lw) else 0
        b = int(((_minutes(rw["et_in"]) >= lo) & (_minutes(rw["et_in"]) < hi)).sum()) if len(rw) else 0
        if a or b:
            buckets.append((lbl, a, b))

    hold_buckets = []
    edges = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 390), (390, 1e9)]
    names = ["<5m", "5-15m", "15-30m", "30-60m", "60m-1session", ">1 session"]
    for (lo, hi), nm in zip(edges, names):
        a = int(((lw["hold_minutes"] >= lo) & (lw["hold_minutes"] < hi)).sum()) if len(lw) else 0
        b = int(((rw["hold_minutes"] >= lo) & (rw["hold_minutes"] < hi)).sum()) if len(rw) else 0
        hold_buckets.append((nm, a, b))

    md = ["| metric | live (broker fills) | replay (classic) | note |",
          "|---|---|---|---|"]
    for m, a, b, n in rows:
        md.append(f"| {m} | {a} | {b} | {n} |")
    md += ["", "| symbol | live trips | live P&L | replay trips | replay P&L |",
           "|---|---|---|---|---|"]
    for sym, a, ap, b, bp in sym_rows:
        md.append(f"| {sym} | {a} | {ap} | {b} | {bp} |")
    md += ["", "| entry bucket (ET) | live | replay |", "|---|---|---|"]
    for lbl, a, b in buckets:
        md.append(f"| {lbl} | {a} | {b} |")
    md += ["", "| hold bucket | live | replay |", "|---|---|---|"]
    for nm, a, b in hold_buckets:
        md.append(f"| {nm} | {a} | {b} |")
    text = "\n".join(md)
    out_md.write_text(text + "\n")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills")
    ap.add_argument("--live-csv")
    ap.add_argument("--replay-trades", required=True)
    ap.add_argument("--start", default="2026-07-30")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--out", default="/tmp/turbo_fidelity.md")
    ap.add_argument("--write-live-csv", default=None)
    ap.add_argument("--symbols", default=None, help="restrict the live set (CSV)")
    args = ap.parse_args(argv)

    if args.live_csv:
        live = pd.read_csv(args.live_csv)
        for c in ("t_in", "t_out", "et_in", "et_out"):
            if c in live:
                live[c] = pd.to_datetime(live[c])
    else:
        if not args.fills:
            raise SystemExit("need --fills or --live-csv")
        fills = json.loads(Path(args.fills).read_text())
        live = order_level_round_trips(fills)
        if args.symbols:
            keep = {s.strip().upper() for s in args.symbols.split(",")}
            live = live[live["symbol"].isin(keep)].reset_index(drop=True)
        if args.write_live_csv:
            live.to_csv(args.write_live_csv, index=False)
    rep = pd.read_csv(args.replay_trades)
    text = compare(live, rep, window=(args.start, args.end), out_md=Path(args.out))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
