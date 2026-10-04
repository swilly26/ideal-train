#!/usr/bin/env python3
"""Round-2 **signal census** — the builder only, no P&L, no cost model (R3-8).

For every declared round-2 cell this runs ``families.build`` and nothing else:
no engine, no ``CostModel``, no fills, no P&L.  It counts, per cell, per window
and per symbol, the bars whose ``entry_dir != 0`` **that pass the declared entry
gate** (``entry_start_min <= minute <= entry_end_min``) — i.e. the signals the
engine would actually consider.

Why it comes before any replay: the 150-trip power floor has to be reachable
*by construction*.  Without a census, a cell that trades 12 times because its
threshold is too tight is indistinguishable from a family that never fires —
and "killed: too few trips" would be a statement about arithmetic rather than
about the strategy.  Cells that cannot reach the floor are dropped **before the
grid is frozen**.

Because each round-2 family emits at most one signal per symbol per session,
the count of gated signal bars is also the maximum number of trips that symbol
can produce (``max_entries_per_session = 1`` per symbol; the re-pinned
``max_positions = 4`` equals the universe size and so can never bind), which is
why the census is a trip-count bound.

    .venv/bin/python scripts/round2_signal_census.py [--windows W1,W2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.backtesting.strategy_search import families  # noqa: E402
from src.backtesting.strategy_search.engine import Market  # noqa: E402
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
DEFAULT_OUT = Path("/home/team/shared/strategy_search/round2")
POWER_FLOOR = 150


def engine_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def census_cell(family: str, spec: dict, market: Market, aligned: dict) -> dict:
    """Count the gated signals of one cell, per symbol — builder only."""
    cfg, instruments, resolved = families.build(family, spec, market, aligned)
    per_symbol: dict[str, dict] = {}
    total = 0
    for sym, inst in sorted(instruments.items()):
        idx = np.flatnonzero(inst.entry_dir)
        gate = ((market.minute[idx] >= int(cfg.entry_start_min))
                & (market.minute[idx] <= int(cfg.entry_end_min)))
        gated = idx[gate]
        dirs = inst.entry_dir[gated]
        sessions = sorted({str(market.day[i]) for i in gated})
        per_symbol[sym] = {
            "signals": int(len(gated)),
            "long": int((dirs > 0).sum()),
            "short": int((dirs < 0).sum()),
            "signals_outside_gate": int(len(idx) - len(gated)),
            "sessions": len(sessions),
            "first_signal": str(market.axis[gated[0]]) if len(gated) else None,
            "last_signal": str(market.axis[gated[-1]]) if len(gated) else None,
        }
        total += int(len(gated))
    return {
        "family": family,
        "cell_id": spec["name"],
        "config_name": cfg.name,
        "resolved_hash": families.resolved_hash(family, resolved),
        "resolved_signal": resolved["signal"],
        "entry_window": [int(cfg.entry_start_min), int(cfg.entry_end_min)],
        "allow_short": bool(cfg.allow_short),
        "symbols": list(resolved["symbols"]),
        "signals_total": total,
        "max_trips_upper_bound": total,
        "per_symbol": per_symbol,
    }


def run_window(window: str, cache_dir: Path | str, families_wanted,
               verbose: bool = True) -> dict:
    start, end = WINDOWS[window]
    frames = load_window(families.R2_UNIVERSE, start, end, cache_dir)
    missing = [s for s in families.R2_UNIVERSE if s not in frames]
    if missing:
        raise SystemExit(f"no cached bars for {missing} under {cache_dir}")
    market = Market(frames)
    aligned = FeatureBook(frames).align(market.axis)
    cov = coverage(frames)
    if verbose:
        print(f"[{window}] {market.n():,} bars "
              f"{market.axis[0]} .. {market.axis[-1]} "
              f"symbols={sorted(frames)}", flush=True)
    cells = []
    for family in families_wanted:
        for spec in families.GRIDS[family]:
            t0 = time.time()
            rec = census_cell(family, spec, market, aligned)
            rec["runtime_s"] = round(time.time() - t0, 3)
            cells.append(rec)
    del frames, market, aligned
    return {"window": window, "dates": [start, end], "coverage": cov,
            "cells": cells}


def print_table(per_window: dict, families_wanted) -> None:
    symbols = list(families.R2_UNIVERSE)
    head = f"{'family':<4} {'cell':<22} " + " ".join(f"{s:>6}" for s in symbols) \
        + f" {'W1':>6} {'W2':>6}  verdict"
    print(head)
    print("-" * len(head))
    for family in families_wanted:
        for spec in families.GRIDS[family]:
            row = {}
            for window, data in per_window.items():
                cell = next(c for c in data["cells"]
                            if c["family"] == family and c["cell_id"] == spec["name"])
                row[window] = cell
            w1, w2 = row.get("W1"), row.get("W2")
            counts = [w1["per_symbol"][s]["signals"] if s in w1["per_symbol"] else 0
                      for s in symbols]
            low = [w for w in ("W1", "W2")
                   if row[w] and row[w]["signals_total"] < POWER_FLOOR]
            verdict = "keep" if not low else f"DROP (<{POWER_FLOOR} on {','.join(low)})"
            print(f"{family:<4} {spec['name']:<22} "
                  + " ".join(f"{c:>6}" for c in counts)
                  + f" {w1['signals_total']:>6} {w2['signals_total']:>6}  {verdict}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default="W1,W2")
    ap.add_argument("--families", default=",".join(families.ROUND2_FAMILIES))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(DEFAULT_OUT / "signal_census.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    for f in wanted:
        if f not in families.ROUND2_FAMILIES:
            raise SystemExit(f"unknown round-2 family {f!r}")
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]

    per_window: dict[str, dict] = {}
    for window in windows:
        per_window[window] = run_window(window, args.cache_dir, wanted,
                                        verbose=not args.quiet)
    out = {
        "script": "scripts/round2_signal_census.py",
        "engine_sha": engine_sha(),
        "cache_dir": str(args.cache_dir),
        "definition": ("bars whose entry_dir != 0 and whose minute is inside the "
                       "cell's declared [entry_start_min, entry_end_min]; the "
                       "builder only — no engine, no cost model, no P&L"),
        "power_floor": POWER_FLOOR,
        "universes": list(families.R2_UNIVERSE),
        "grid_sizes": families.ROUND2_GRID_SIZES,
        "windows": per_window,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    if not args.quiet:
        print()
        print_table({w: d for w, d in per_window.items()}, wanted)
        totals = {w: sum(c["signals_total"] for c in d["cells"])
                  for w, d in per_window.items()}
        print(f"\ntotal gated signals: {totals}")
        print(f"artefact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
