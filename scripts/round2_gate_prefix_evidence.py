#!/usr/bin/env python3
"""E1 evidence: the same-symbol flat pair on the pre-gate engine vs the gate.

Run it against **two trees** to reproduce the round-2 blocking finding:

    .venv/bin/python scripts/round2_gate_prefix_evidence.py /home/team/shared/wt-prefix
    .venv/bin/python scripts/round2_gate_prefix_evidence.py /home/team/shared/wt-r2

On the pre-gate engine (commit 69d5d8e) two legs on one symbol collapse into one
fill, so a flat tape books ``pnl_gross + cost_drag`` **positive** — a fabricated
profit equal to the cancelled toll, which is the zero-cost number the screen's
fold-breadth rule runs on.  On the gated engine it is exactly zero and the trip
loses one adverse fill per leg per side.
"""
from __future__ import annotations

import sys
from pathlib import Path

TREE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import engine  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Instrument, Leg, Market, run_search,
)

PX = 100.0


def flat_frame(n: int = 390) -> pd.DataFrame:
    idx = [pd.Timestamp("2025-01-02") + pd.Timedelta(minutes=9 * 60 + 30 + i)
           for i in range(n)]
    rows = [(PX, PX + 0.02, PX - 0.02, PX, 1000.0)] * n
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex(idx, name="ts"))


def run(legs, key):
    m = Market({"SOXL": flat_frame(), "TQQQ": flat_frame()})
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[5] = 1
    inst = Instrument(key=key, legs=legs, entry_dir=entry,
                      exit_now=np.full(m.n(), 2, dtype=np.int8),
                      valid=m.valid(legs), kind="pair")
    cfg = engine.SearchConfig(family="D", name="e1-evidence", sizing="fixed_notional",
                              notional_usd=50_000.0, initial_equity=100_000.0,
                              max_positions=4, entry_start_min=9 * 60 + 30,
                              entry_end_min=15 * 60 + 29, max_entries_per_session=1,
                              allow_short=True, eod_flat_min=15 * 60 + 30)
    return run_search(m, {key: inst}, cfg, CostModel.baseline()).trades.iloc[0]


def main() -> int:
    print(f"tree: {TREE}")
    for label, legs, key in (
            ("two symbols (control)", (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0)),
             "SOXL/TQQQ"),
            ("one symbol, equal weights", (Leg("SOXL", 1, 1.0), Leg("SOXL", -1, 1.0)),
             "SOXL/SOXL"),
            ("one symbol, unequal weights", (Leg("SOXL", 1, 1.0), Leg("SOXL", -1, 2.0)),
             "SOXL/SOXL-21")):
        tr = run(legs, key)
        print(f"  {label:<26} pnl_gross={tr['pnl_gross']:+10.4f} "
              f"cost_drag={tr['cost_drag']:+10.4f} "
              f"gross+drag={tr['pnl_gross'] + tr['cost_drag']:+10.4f} "
              f"net={tr['pnl_after_costs']:+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
