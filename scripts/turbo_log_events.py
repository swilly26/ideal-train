#!/usr/bin/env python3
"""Extract the turbo trader's own decisions from its daily logs.

The logs are UTC-stamped and use a handful of fixed log lines for entries and
exits.  This turns them into a CSV of decisions (entry / exit with reason,
strategy, confidence) so they can be lined up against a replay.  Log *P&L*
lines are deliberately NOT extracted: before PR #30 they booked the last mark
rather than the fill price, so quoted log P&L was never realised P&L.

Usage::

    .venv/bin/python scripts/turbo_log_events.py --log-dir logs \\
        --start 2026-07-30 --end 2026-08-13 --out /tmp/turbo_log_events.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

BUY = re.compile(
    r"🚀 BUY\s+(?P<symbol>[A-Z]+): (?P<qty>[\d.]+) shares @ \$(?P<price>[\d.]+).*?"
    r"conf=(?P<conf>[\d.]+)(?: \[(?P<strategy>[a-z_]+)\])?")
SELL = re.compile(
    r"(?P<icon>📉|📗) SELL (?P<symbol>[A-Z]+): (?P<qty>[\d.]+) shares @ \$(?P<price>[\d.]+)"
    r".*?\[(?P<reason>[a-z_]+)\]")
EXIT_TAGS = (
    ("⏰ TIME-EXIT", "time"),
    ("🛑 STOP-LOSS", "stop"),
    ("🎯 TAKE-PROFIT", "target"),
    ("⏰ EOD closing", "eod"),
    ("⏰ Mandatory EOD liquidation", "eod_sweep"),
)
TS = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")


def parse(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        m = TS.match(line)
        if not m:
            continue
        ts = m.group("ts")
        mb = BUY.search(line)
        if mb:
            rows.append(dict(ts=ts, kind="entry", symbol=mb.group("symbol"),
                             qty=float(mb.group("qty")), price=float(mb.group("price")),
                             conf=float(mb.group("conf")),
                             strategy=mb.group("strategy") or "", reason=""))
            continue
        ms = SELL.search(line)
        if ms:
            rows.append(dict(ts=ts, kind="exit", symbol=ms.group("symbol"),
                             qty=float(ms.group("qty")), price=float(ms.group("price")),
                             conf=0.0, strategy="", reason=ms.group("reason")))
            continue
        for tag, reason in EXIT_TAGS:
            if tag in line and ("[A-Z]" in line or "positions closed" in line):
                sym = ""
                m2 = re.search(r"[A-Z]{2,5}", line.split(tag, 1)[1])
                if m2:
                    sym = m2.group(0)
                rows.append(dict(ts=ts, kind="exit_tag", symbol=sym, qty=0.0,
                                 price=0.0, conf=0.0, strategy="", reason=reason))
                break
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--start", default="2026-07-30")
    ap.add_argument("--end", default="2026-08-13")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    rows: list[dict] = []
    for path in sorted(Path(args.log_dir).glob("turbo_*.log")):
        m = re.search(r"(\d{8})", path.name)
        if not m:
            continue
        day = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        if not (start <= day <= end):
            continue
        for r in parse(path):
            r["log_day"] = day.isoformat()
            rows.append(r)
    rows.sort(key=lambda r: r["ts"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ts", "log_day", "kind", "symbol", "qty",
                                          "price", "conf", "strategy", "reason"])
        w.writeheader()
        w.writerows(rows)
    n_entry = sum(1 for r in rows if r["kind"] == "entry")
    n_exit = sum(1 for r in rows if r["kind"] == "exit")
    print(f"{len(rows)} decisions ({n_entry} entries, {n_exit} exits) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
