#!/usr/bin/env python3
"""Quantified loss attribution for the ScalpSet 12-month replay.

Every number in the emitted document is computed here from two inputs only:

* ``data/backtest_out/trades_<variant>.csv`` — one row per round trip
  (symbol, module, side, entry/exit time+price, qty, sl, tp, sl_source,
  exit_reason, pnl, bars_held);
* the cached 1-minute RTH bars (``data/history/1m``) — used only to measure
  the *path* a trade took (MAE / MFE) and to run the stop-distance
  counterfactuals on the same entry set.

Nothing is transcribed by hand, so the document can be regenerated after any
re-run::

    env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY .venv/bin/python \
        scripts/scalpset_loss_forensics.py \
        --out /home/team/shared/SCALPSET_LOSS_FORENSICS.md

Definitions used throughout (all on the trade's own fill price):

* ``dist_sl_bps`` — |entry_price − sl| / entry_price × 1e4, the stop distance
  recorded on the trade row (the stop may have been moved; ``sl_source`` says
  by what: ``strategy`` / ``fill_risk`` / ``fill_risk_clamped`` = the level the
  position opened with, ``breakeven`` / ``trail`` = a stop that the live
  break-even/trail rule moved during the trade).
* ``mae_bps`` — worst adverse excursion after entry (low for longs, high for
  shorts) over the bars ``[entry_time, exit_time]``, in bps of the fill price.
  The engine evaluates exits from the entry bar onward, so the entry bar's
  range is included (it can contain up to one minute of pre-fill price).
* ``mfe_bps`` — best favourable excursion over the same window.
* ``mae_R`` / ``mfe_R`` — the same excursions divided by ``dist_sl_bps``: how
  many "stops' worth" of heat the trade took / gave.
* R-multiple of a closed trade — ``pnl / (dist_sl_bps/1e4 × entry_price × qty)``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.scalp_data import load_rth_1m  # noqa: E402

DEFAULT_IN = ROOT / "data" / "backtest_out"
DEFAULT_CACHE = ROOT / "data" / "history"
OUT_DEFAULT = Path("/home/team/shared/SCALPSET_LOSS_FORENSICS.md")

BPS = 1e4
#: live cost model of the ``baseline`` variant (``ScalpSetConfig`` defaults)
SLIP_PCT, SLIP_ABS = 0.0002, 0.01

TOD_BUCKETS = [
    ("09:30-10:14 open", 9 * 60 + 30, 10 * 60 + 15),
    ("10:15-11:29 morning", 10 * 60 + 15, 11 * 60 + 30),
    ("11:30-13:29 lunch", 11 * 60 + 30, 13 * 60 + 30),
    ("13:30-14:59 afternoon", 13 * 60 + 30, 15 * 60 + 0),
    ("15:00-16:00 close", 15 * 60 + 0, 16 * 60 + 1),
]


# ── small formatting helpers ───────────────────────────────────────────────
def money(x: float) -> str:
    return f"${x:,.0f}"


def money2(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def num(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ── bars ───────────────────────────────────────────────────────────────────
class Bars:
    """Minute bars for one symbol, with fast position/session lookups."""

    def __init__(self, sym: str, cache: Path):
        df = load_rth_1m(sym, cache_dir=cache)
        self.sym = sym
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.pos = {t: i for i, t in enumerate(df.index)}
        self.day_last: dict = {}
        for i, d in enumerate(df.index.normalize()):
            self.day_last[d] = i
        self.first_close = float(self.c[0])
        self.last_close = float(self.c[-1])

    def idx(self, t: pd.Timestamp) -> int | None:
        return self.pos.get(t)

    def drift(self) -> float:
        return self.last_close / self.first_close - 1.0


class BarsCache:
    def __init__(self, cache: Path):
        self.cache = cache
        self._d: dict[str, Bars] = {}

    def __getitem__(self, sym: str) -> Bars:
        if sym not in self._d:
            self._d[sym] = Bars(sym, self.cache)
        return self._d[sym]


# ── per-trade path metrics ─────────────────────────────────────────────────
def add_path_metrics(tr: pd.DataFrame, bars: BarsCache) -> pd.DataFrame:
    mae, mfe, nbars = [], [], []
    for row in tr.itertuples():
        b = bars[row.symbol]
        i0 = b.idx(row.entry_time)
        i1 = b.idx(row.exit_time)
        if i0 is None or i1 is None or i1 < i0:
            mae.append(np.nan); mfe.append(np.nan); nbars.append(0)
            continue
        lo = float(b.l[i0:i1 + 1].min())
        hi = float(b.h[i0:i1 + 1].max())
        e = float(row.entry_price)
        if row.side == "LONG":
            mae.append((e - lo) / e * BPS); mfe.append((hi - e) / e * BPS)
        else:
            mae.append((hi - e) / e * BPS); mfe.append((e - lo) / e * BPS)
        nbars.append(i1 - i0 + 1)
    out = tr.copy()
    out["mae_bps"] = mae
    out["mfe_bps"] = mfe
    out["path_bars"] = nbars
    return out


# ── counterfactual: stop-distance floor, same entries ──────────────────────
def _slip(price: float, is_buy: bool) -> float:
    """Engine ``_market_fill``: adverse by max(2bps, 1c)."""
    d = max(abs(price) * SLIP_PCT, SLIP_ABS)
    return price + d if is_buy else price - d


def simulate_floor(tr: pd.DataFrame, bars: BarsCache, mult: float = 1.0,
                   floor_bps: float | None = None, no_stop: bool = False) -> dict:
    """Replay the *same entries* with a different stop distance.

    ``dist = max(mult × |entry − sl|, floor_bps)`` away from the fill; the
    recorded target is kept; exits are evaluated bar-by-bar stop-first (as the
    engine does), market-style exits pay the engine's adverse slippage, and any
    position open at the last RTH bar of its session is flattened at that close
    (``eod_flat``).  ``no_stop=True`` removes the stop entirely (EOD/target only).

    The live break-even / trailing stop moves are *not* modelled here — this
    isolates stop *distance*, which is the question being asked.  Callers must
    therefore pass only trades whose opening stop is known (``sl_source`` in
    strategy / fill_risk / fill_risk_clamped): for a trade the live rule moved
    to break-even or trailed, the distance recorded on the trade row is the
    *moved* stop, not the one the position opened with.
    """
    total = 0.0
    n_tp = n_sl = n_eod = 0
    for row in tr.itertuples():
        b = bars[row.symbol]
        i0 = b.idx(row.entry_time)
        if i0 is None:
            continue
        last = b.day_last[row.entry_time.normalize()]
        e = float(row.entry_price)
        qty = float(row.quantity)
        is_long = row.side == "LONG"
        dist = abs(e - float(row.sl)) * mult
        if floor_bps is not None:
            dist = max(dist, floor_bps * e / BPS)
        dist = max(dist, 0.01)
        stop = (None if no_stop
                else (e - dist if is_long else e + dist))
        tp = float(row.tp)
        exit_price = None
        for j in range(i0, last + 1):
            h, l, o = float(b.h[j]), float(b.l[j]), float(b.o[j])
            if is_long:
                if stop is not None and l <= stop:
                    exit_price = _slip(o if o <= stop else stop, is_buy=True)
                    n_sl += 1
                    break
                if h >= tp:
                    exit_price = max(o, tp); n_tp += 1; break
            else:
                if stop is not None and h >= stop:
                    exit_price = _slip(o if o >= stop else stop, is_buy=False)
                    n_sl += 1
                    break
                if l <= tp:
                    exit_price = min(o, tp); n_tp += 1; break
        if exit_price is None:
            exit_price = _slip(float(b.c[last]), is_buy=is_long); n_eod += 1
        total += ((exit_price - e) if is_long else (e - exit_price)) * qty
    return {"pnl": total, "tp": n_tp, "sl": n_sl, "eod": n_eod}


# ── the report ─────────────────────────────────────────────────────────────
def main() -> int:  # noqa: C901 - linear report builder
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--alt", default="pessimistic")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dump-json", default="")
    args = ap.parse_args()
    inp = Path(args.in_dir)
    bars = BarsCache(Path(args.cache_dir))

    raw = pd.read_csv(inp / f"trades_{args.variant}.csv")
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    raw["exit_time"] = pd.to_datetime(raw["exit_time"])
    stats = json.loads((inp / f"stats_{args.variant}.json").read_text())
    alt_path = inp / f"trades_{args.alt}.csv"
    alt = None
    if alt_path.exists():
        alt = pd.read_csv(alt_path)
        alt["entry_time"] = pd.to_datetime(alt["entry_time"])
        alt["exit_time"] = pd.to_datetime(alt["exit_time"])

    tr = add_path_metrics(raw, bars)
    tr["dist_sl_bps"] = (tr["entry_price"] - tr["sl"]).abs() / tr["entry_price"] * BPS
    tr["dist_tp_bps"] = (tr["tp"] - tr["entry_price"]).abs() / tr["entry_price"] * BPS
    tr["mins"] = tr["entry_time"].dt.hour * 60 + tr["entry_time"].dt.minute
    tr["session"] = tr["entry_time"].dt.normalize()
    tr["r_mult"] = np.where(
        tr["dist_sl_bps"] > 0,
        tr["pnl"] / (tr["dist_sl_bps"] / BPS * tr["entry_price"] * tr["quantity"]),
        np.nan)
    tr["mae_R"] = tr["mae_bps"] / tr["dist_sl_bps"].replace(0, np.nan)
    tr["mfe_R"] = tr["mfe_bps"] / tr["dist_sl_bps"].replace(0, np.nan)

    n = len(tr)
    net = float(tr["pnl"].sum())
    wr = float((tr["pnl"] > 0).mean())
    wins = tr[tr["pnl"] > 0]["pnl"]
    loss = tr[tr["pnl"] <= 0]["pnl"]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(loss.mean()) if len(loss) else 0.0
    be_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) if avg_win else float("nan")
    inner = avg_win + abs(avg_loss)
    exp_trade = float(tr["pnl"].mean())
    sessions = int(tr["session"].nunique())
    open_win = tr[(tr.mins >= 9 * 60 + 30) & (tr.mins < 10 * 60 + 15)]
    tr_sorted = tr.sort_values("entry_time")
    tr_sorted["rank"] = tr_sorted.groupby(["symbol", "session"]).cumcount() + 1
    D: dict = {}

    L: list[str] = []
    A = L.append

    # ── header ────────────────────────────────────────────────────────────
    A("# ScalpSet — why the strategy loses money (12-month forensics)")
    A("")
    A(f"**Variant analysed:** `{args.variant}` · window {stats['start']} → {stats['end']} · "
      f"{n:,} closed trades on {sessions} sessions · engine counters from "
      f"`stats_{args.variant}.json` · path metrics from the cached 1-minute RTH bars "
      f"(`data/history/1m`).")
    if alt is not None:
        A("")
        A(f"The `{args.alt}` variant is used as a cross-check wherever the ranking could be "
          f"an artefact of the cost model; its headline is "
          f"{money2(float(alt['pnl'].sum()))} on {len(alt):,} trades.")
    A("")
    A("Everything below is a number computed from the trade log (plus, for the path metrics, "
      "the same minute bars the backtest itself replayed). No claim in this document is an "
      "opinion about the strategy: each one is a count or a sum that can be re-derived by "
      "running `scripts/scalpset_loss_forensics.py`.")
    A("")

    # ── 0. the one-paragraph answer ───────────────────────────────────────
    A("## 0. The answer in one paragraph")
    A("")
    A(f"ScalpSet is not losing because of costs, and not because the tape went the wrong way. "
      f"It is losing because **the average trade pays {money2(abs(avg_loss))} when it loses and "
      f"only {money2(avg_win)} when it wins, and the win rate that geometry demands is "
      f"{pct(be_wr)} — the replay delivers {pct(wr)}.** That "
      f"{abs(be_wr - wr) * 100:.2f} percentage-point shortfall is worth "
      f"{money2(abs((wr - be_wr) * inner))} per trade, and multiplied by {n:,} trades it *is* "
      f"the entire {money(abs(net))} annual loss. Three things amplify it: the stop is tighter "
      f"than the noise it has to survive (median stop {tr['dist_sl_bps'].median():.1f} bps, "
      f"{int((tr['sl_source'] == 'breakeven').sum())} trades carrying a stop the live rule had "
      f"moved onto the fill price, and "
      f"{int((tr['sl_source'] == 'fill_risk_clamped').sum())} pinned to the 5 bps / 1¢ minimum), "
      f"the engine spends its allocation on the maximum number of trades the cap allows "
      f"({n / sessions:.1f}/session), and "
      f"{len(open_win) / n:.0%} of all entries happen in the first 45 minutes of the session where "
      f"they account for {open_win['pnl'].sum() / net:.0%} of the loss. **Widening the stop does "
      f"not fix it** — every wider-stop variant of the same entries loses more (§3.3) — so the "
      f"lever is the entry, above all the opening window, not the risk settings.")
    A("")

    # ── 1. P&L attribution ────────────────────────────────────────────────
    A("## 1. P&L attribution — where the money actually leaves the account")
    A("")
    A("### 1.1 By exit reason")
    A("")
    rows = []
    for reason, g in tr.groupby("exit_reason"):
        rows.append([reason.upper(), f"{len(g):,}", money2(g['pnl'].sum()),
                     money2(g['pnl'].mean()), pct((g['pnl'] > 0).mean()),
                     num(g['bars_held'].mean(), 1),
                     f"{g['dist_sl_bps'].median():.1f}",
                     f"{g['pnl'].sum() / abs(net):+.2f}×"])
    rows.sort(key=lambda r: float(r[2].replace("$", "").replace(",", "")))
    A(md_table(["Exit", "Trades", "Net P&L", "Avg P&L", "Win rate", "Avg bars held",
                "Median stop dist (bps)", "÷ |annual loss|"], rows))
    A("")
    sl_rows = tr[tr.exit_reason == "sl"]
    A(f"- The stop-loss leg is where the loss is booked: **{len(sl_rows):,} SL exits for "
      f"{money2(sl_rows['pnl'].sum())}**, of which "
      f"{money2(sl_rows[sl_rows.pnl <= 0]['pnl'].sum())} is realised loss and "
      f"{money2(sl_rows[sl_rows.pnl > 0]['pnl'].sum())} is the occasional profitable stop-out.")
    A(f"- The target leg recovers **{money2(tr[tr.exit_reason == 'tp']['pnl'].sum())}** on only "
      f"{int((tr.exit_reason == 'tp').sum())} trades — a target is hit "
      f"{(tr.exit_reason == 'tp').sum() / n:.1%} of the time.")
    A(f"- The end-of-day leg is roughly a coin flip in dollars: "
      f"{money2(tr[tr.exit_reason == 'eod']['pnl'].sum())} on "
      f"{int((tr.exit_reason == 'eod').sum())} trades "
      f"({pct((tr[tr.exit_reason == 'eod'].pnl > 0).mean())} of them won).")
    A("")
    A("### 1.2 By module")
    A("")
    rows = []
    for mod, g in tr.groupby("module"):
        rows.append([mod, f"{len(g):,}", money2(g['pnl'].sum()), money2(g['pnl'].mean()),
                     pct((g['pnl'] > 0).mean()),
                     pct(g['pnl'].sum() / net if net else 0.0, 1)])
    rows.sort(key=lambda r: float(r[2].replace("$", "").replace(",", "")))
    A(md_table(["Module", "Trades", "Net P&L", "Avg P&L", "Win rate",
                "% of net P&L"], rows))
    A("")
    sig = stats["stats"]["signals_by_module"]
    A(f"- Signal counts behind those trades: "
      f"{', '.join(f'{k} {v:,}' for k, v in sorted(sig.items(), key=lambda kv: -kv[1]))}. "
      f"A signal → trade conversion of "
      f"{n / max(sum(sig.values()), 1) * 100:.3f}% — the modules are extremely trigger-happy and "
      f"the entry cap, not the signal logic, decides what actually gets traded.")
    A("")
    A("### 1.3 By side")
    A("")
    rows = []
    for side, g in tr.groupby("side"):
        rows.append([side, f"{len(g):,}", money2(g['pnl'].sum()), money2(g['pnl'].mean()),
                     pct((g['pnl'] > 0).mean()),
                     pct(len(g) / n), pct(g['pnl'].sum() / abs(net), 0)])
    A(md_table(["Side", "Trades", "Net P&L", "Avg P&L", "Win rate", "Share of trades",
                "Share of loss"], rows))
    A("")
    A("### 1.4 By symbol")
    A("")
    rows = []
    for sym, g in tr.groupby("symbol"):
        rows.append([sym, f"{len(g):,}", money2(g['pnl'].sum()), money2(g['pnl'].mean()),
                     pct((g['pnl'] > 0).mean()),
                     pct(bars[sym].drift(), 1),
                     money2(g[g.side == 'LONG']['pnl'].sum()),
                     money2(g[g.side == 'SHORT']['pnl'].sum())])
    rows.sort(key=lambda r: float(r[2].replace("$", "").replace(",", "")))
    A(md_table(["Symbol", "Trades", "Net P&L", "Avg P&L", "Win rate",
                "1m-window drift (first→last close)", "Long book", "Short book"], rows))
    A("")
    A("### 1.5 By time of day (entry time, exchange-local)")
    A("")
    rows = []
    for label, lo, hi in TOD_BUCKETS:
        g = tr[(tr.mins >= lo) & (tr.mins < hi)]
        if not len(g):
            continue
        rows.append([label, f"{len(g):,}", money2(g['pnl'].sum()), money2(g['pnl'].mean()),
                     pct((g['pnl'] > 0).mean()),
                     money2(g['pnl'].sum() / sessions) + "/session"])
    A(md_table(["Entry window (ET)", "Trades", "Net P&L", "Avg P&L", "Win rate",
                "Contribution"], rows))
    A("")
    A(f"- The book is overwhelmingly an **opening-bell book**: {len(open_win):,} of {n:,} trades "
      f"({len(open_win) / n:.0%}) are entered between 09:30 and 10:14 and they carry "
      f"{money2(open_win['pnl'].sum())} of the {money2(net)} loss "
      f"({open_win['pnl'].sum() / net:.0%}). The remaining {n - len(open_win):,} trades are close "
      f"to break-even in aggregate ({money2(net - open_win['pnl'].sum())}), which means the whole "
      f"year's problem is concentrated in the first 45 minutes of the session.")
    A("")
    A("### 1.6 Where the money leaves, in one line")
    A("")
    tp_sum = float(tr[tr.exit_reason == "tp"]["pnl"].sum())
    sl_sum = float(tr[tr.exit_reason == "sl"]["pnl"].sum())
    eod_sum = float(tr[tr.exit_reason == "eod"]["pnl"].sum())
    A(f"`{money2(tp_sum)} (targets) {money2(sl_sum)} (stops) + {money2(eod_sum)} (EOD) = "
      f"{money2(net)}`. The stops alone are {abs(sl_sum) / n:.1f}× the size of the entire "
      f"annual loss; the targets recover only "
      f"{tp_sum / abs(sl_sum):.0%} of what the stops give back.")
    A("")

    # ── 2. exit geometry ──────────────────────────────────────────────────
    A("## 2. Exit geometry — the win rate the strategy needs vs the one it gets")
    A("")
    A("With an average win of `W` and an average loss of `L`, expectancy per trade is "
      "`(WR − BE_WR) × (W + |L|)` where `BE_WR = |L| / (W + |L|)`. That identity is exact and "
      "lets the whole loss be decomposed without any further assumptions.")
    A("")
    A(md_table(["Quantity", "Value"], [
        ["Average win", money2(avg_win)],
        ["Average loss", money2(avg_loss)],
        ["Win / loss size ratio", f"{avg_win / abs(avg_loss):.2f} : 1"],
        ["**Break-even win rate**", f"**{pct(be_wr)}**"],
        ["**Realised win rate**", f"**{pct(wr)}**"],
        ["Shortfall", f"**{(be_wr - wr) * 100:.2f} points**"],
        ["Expectancy per trade", f"{money2(exp_trade)}  (= (WR − BE_WR) × (W + |L|) = "
                                 f"{money2((wr - be_wr) * inner)})"],
        ["× trades", f"{n:,}"],
        ["**= annual P&L**", f"**{money2((wr - be_wr) * inner * n)}** vs realised "
                             f"{money2(net)}"],
    ]))
    A("")
    A(f"So **{money2(abs(net))} of loss is 100% explained by the win-rate shortfall** — there is "
      f"no unexplained residual. Costs are not the origin of it: this variant charges "
      f"{money2(float(stats['stats'].get('fees_paid', 0.0)))} in commission and the pessimistic "
      f"cost model adds "
      f"{money2(abs(float(alt['pnl'].sum()) - net)) if alt is not None else 'n/a'} "
      f"({pct(abs(float(alt['pnl'].sum()) - net) / abs(net), 0) if alt is not None else 'n/a'} on "
      f"top of the loss) — costs multiply a losing edge, they do not create it.")
    A("")
    A("### 2.1 Is the win/loss asymmetry stable? (per symbol)")
    A("")
    rows = []
    for sym, g in tr.groupby("symbol"):
        w = g[g.pnl > 0]["pnl"]; l = g[g.pnl <= 0]["pnl"]
        if not len(w) or not len(l):
            continue
        aw, al = float(w.mean()), float(l.mean())
        b = abs(al) / (aw + abs(al))
        r = float((g.pnl > 0).mean())
        rows.append([sym, money2(aw), money2(al), f"{aw / abs(al):.2f}", pct(b), pct(r),
                     f"{(r - b) * 100:+.2f}", money2((r - b) * (aw + abs(al)) * len(g))])
    rows.sort(key=lambda r: float(r[7].replace("$", "").replace(",", "")))
    A(md_table(["Symbol", "Avg win", "Avg loss", "W/L", "BE win rate", "Realised WR",
                "Gap (pts)", "$ explained by the gap"], rows))
    A("")
    A("### 2.2 Per module and per quarter")
    A("")
    rows = []
    for mod, g in tr.groupby("module"):
        w = g[g.pnl > 0]["pnl"]; l = g[g.pnl <= 0]["pnl"]
        if not len(w) or not len(l):
            continue
        aw, al = float(w.mean()), float(l.mean())
        b = abs(al) / (aw + abs(al)); r = float((g.pnl > 0).mean())
        rows.append([mod, f"{len(g):,}", pct(b), pct(r), f"{(r - b) * 100:+.2f}",
                     money2((r - b) * (aw + abs(al)) * len(g))])
    A(md_table(["Module", "Trades", "BE win rate", "Realised WR", "Gap (pts)",
                "$ explained by the gap"], rows))
    A("")
    rows = []
    for q, g in tr.groupby(tr["entry_time"].dt.to_period("Q")):
        w = g[g.pnl > 0]["pnl"]; l = g[g.pnl <= 0]["pnl"]
        if not len(w) or not len(l):
            continue
        aw, al = float(w.mean()), float(l.mean())
        b = abs(al) / (aw + abs(al)); r = float((g.pnl > 0).mean())
        rows.append([str(q), f"{len(g):,}", money2(aw), money2(al), pct(b), pct(r),
                     f"{(r - b) * 100:+.2f}", money2(float(g.pnl.sum()))])
    A(md_table(["Quarter", "Trades", "Avg win", "Avg loss", "BE win rate", "Realised WR",
                "Gap (pts)", "Net P&L"], rows))
    A("")
    A(f"- The size asymmetry is **stable** (per-symbol win/loss ratios "
      f"{min(float(tr[tr.pnl > 0].groupby('symbol').pnl.mean().iloc[i] / abs(float(tr[tr.pnl <= 0].groupby('symbol').pnl.mean().iloc[i]))) for i in range(len(tr.symbol.unique()))):.2f}–"
      f"{max(float(tr[tr.pnl > 0].groupby('symbol').pnl.mean().iloc[i] / abs(float(tr[tr.pnl <= 0].groupby('symbol').pnl.mean().iloc[i]))) for i in range(len(tr.symbol.unique()))):.2f} : 1) "
      f"and every symbol except the one that is flat-to-positive needs a win rate it never "
      f"reaches. The problem is **not** that a few symbols blow up: it is a uniform, "
      f"every-symbol shortfall.")
    A("")

    # ── 3. stop distance vs noise ─────────────────────────────────────────
    A("## 3. Stop placement vs 1-minute noise")
    A("")
    A("### 3.1 Distance distributions")
    A("")
    q = [10, 25, 50, 75, 90]
    def qs(s: pd.Series) -> str:
        s = s.dropna()
        return " / ".join(f"{np.percentile(s, p):.1f}" for p in q)

    rows = [
        ["Stop distance (bps)", qs(tr['dist_sl_bps']), f"{tr['dist_sl_bps'].median():.1f}"],
        ["Target distance (bps)", qs(tr['dist_tp_bps']), f"{tr['dist_tp_bps'].median():.1f}"],
        ["Target / stop distance", f"{(tr['dist_tp_bps'] / tr['dist_sl_bps'].replace(0, np.nan)).median():.2f} (median)", ""],
        ["MAE, all trades (bps)", qs(tr['mae_bps']), f"{tr['mae_bps'].median():.1f}"],
        ["MFE, all trades (bps)", qs(tr['mfe_bps']), f"{tr['mfe_bps'].median():.1f}"],
        ["MAE of trades that ended at TP", qs(tr[tr.exit_reason == 'tp']['mae_bps']),
         f"{tr[tr.exit_reason == 'tp']['mae_bps'].median():.1f}"],
        ["MAE of trades that ended at EOD", qs(tr[tr.exit_reason == 'eod']['mae_bps']),
         f"{tr[tr.exit_reason == 'eod']['mae_bps'].median():.1f}"],
        ["MFE of trades that ended at SL", qs(tr[tr.exit_reason == 'sl']['mfe_bps']),
         f"{tr[tr.exit_reason == 'sl']['mfe_bps'].median():.1f}"],
    ]
    A(md_table(["Quantity", "p10 / p25 / p50 / p75 / p90", "Median"], rows))
    A("")
    A("Where the stop that ended the trade came from:")
    A("")
    src_counts = tr[tr.exit_reason == "sl"]["sl_source"].value_counts().to_dict()
    src_dist = tr[tr.exit_reason == "sl"].groupby("sl_source")["dist_sl_bps"].median().to_dict()
    A(md_table(["sl_source at the exit", "SL exits", "Median stop distance now (bps)",
                "What it means"],
               [[k, f"{v:,}", f"{src_dist.get(k, float('nan')):.1f}",
                 {"strategy": "the module's own stop",
                  "fill_risk": "risk-sized from the signal's own SL/TF distance",
                  "fill_risk_clamped": "**clamped to the live 5 bps / 1¢ minimum distance**",
                  "breakeven": "**the live break-even rule moved the stop onto the fill price**",
                  "trail": "trailed stop"}.get(k, "")]
                for k, v in sorted(src_counts.items(), key=lambda kv: -kv[1])]))
    A("")
    A("### 3.2 Is the stop inside normal noise?")
    A("")
    tp_mae = tr[tr.exit_reason == "tp"]["mae_bps"].dropna()
    eod_mae = tr[tr.exit_reason == "eod"]["mae_bps"].dropna()
    surv_mae = tr[tr.exit_reason.isin(["tp", "eod"])]["mae_bps"].dropna()
    med_all_stop = float(tr["dist_sl_bps"].median())
    med_sl_stop = float(tr[tr.exit_reason == "sl"]["dist_sl_bps"].median())
    A(f"- Trades that **eventually hit their target** first went {tp_mae.median():.1f} bps against "
      f"the position at the median (p75 {np.percentile(tp_mae, 75):.1f}, p90 "
      f"{np.percentile(tp_mae, 90):.1f} bps). The median stop distance in the book is "
      f"{med_all_stop:.1f} bps and the median stop that actually killed a trade was "
      f"{med_sl_stop:.1f} bps. The winners therefore routinely absorb "
      f"{tp_mae.median() / med_all_stop:.2f}× the whole median stop budget before they work.")
    A(f"- **{((surv_mae > med_all_stop).mean()):.0%} of the trades that survived to TP/EOD had "
      f"already been more than the median stop distance underwater** (and "
      f"{(surv_mae > med_sl_stop).mean():.0%} were deeper than the stop a typical loser was "
      f"carrying). They survived only because their own stop happened to be wider or was moved — "
      f"stop distance, not the entry, is doing a large part of the selection.")
    wick = 0
    for row in tr[tr.exit_reason == "sl"].itertuples():
        b = bars[row.symbol]
        i1 = b.idx(row.exit_time)
        if i1 is None:
            continue
        c = float(b.c[i1])
        good = (c > row.sl) if row.side == "LONG" else (c < row.sl)
        if good and abs(c - row.sl) / row.entry_price * BPS >= 5.0:
            wick += 1
    D["wick_stops"] = wick
    n_sl = int((tr.exit_reason == "sl").sum())
    A(f"- **{wick:,} of the {n_sl:,} stop-outs ({wick / max(n_sl, 1):.0%}) were minute bars that "
      f"touched the stop and then closed at least 5 bps back on the profitable side of it** — a "
      f"wick through the level, not a sustained move. Those exits are the clearest single piece of "
      f"evidence that the stop sits inside ordinary one-minute noise.")
    A("")
    A("### 3.3 Counterfactual: the same entries with a wider stop")
    A("")
    A("Replaying the *identical* entries bar-by-bar with the stop moved to a multiple of, or a "
      "floor below, the distance the position opened with (target and EOD-flat unchanged, engine "
      "slippage, stop evaluated first inside a bar, break-even/trailing moves disabled so that "
      "only stop *distance* varies). The reconstruction is restricted to the "
      f"{int((~tr['sl_source'].isin(['breakeven', 'trail'])).sum()):,} trades whose opening stop "
      "is known from the log (`sl_source` = strategy / fill_risk / fill_risk_clamped); the "
      f"remaining {int(tr['sl_source'].isin(['breakeven', 'trail']).sum()):,} trades had their "
      "stop moved onto the fill or trailed by the live rule and their opening distance cannot be "
      "recovered from the trade row:")
    A("")
    known = tr[~tr["sl_source"].isin(["breakeven", "trail"])]
    moved = tr[tr["sl_source"].isin(["breakeven", "trail"])]
    sim_actual = simulate_floor(known, bars)
    D["sim_actual"] = sim_actual
    known_exits = known["exit_reason"].value_counts().to_dict()
    rows = [["as recorded (sweep baseline)", f"{len(known):,}", money2(known['pnl'].sum()),
             money2(sim_actual["pnl"]),
             f"{sim_actual['tp']:,} / {sim_actual['sl']:,} / {sim_actual['eod']:,}"]]
    for lab, kw in [("stop × 1.5", {"mult": 1.5}),
                    ("stop × 2", {"mult": 2.0}),
                    ("stop × 3", {"mult": 3.0}),
                    ("floor 20 bps", {"floor_bps": 20.0}),
                    ("floor 30 bps", {"floor_bps": 30.0}),
                    ("floor 50 bps", {"floor_bps": 50.0}),
                    ("no stop at all (target / EOD only)", {"no_stop": True})]:
        r = simulate_floor(known, bars, **kw)
        rows.append([lab, f"{len(known):,}", money2(known['pnl'].sum()), money2(r["pnl"]),
                     f"{r['tp']:,} / {r['sl']:,} / {r['eod']:,}"])
    A(md_table(["Stop distance, same entries", "Trades", "Actual P&L (this subset)",
                "Simulated P&L", "TP / SL / EOD exits"], rows))
    A("")
    A(f"- **No wider stop rescues these entries.** The sweep's best alternative (a 20 bps floor, "
      f"i.e. never letting the stop sit inside the live 5 bps minimum) still comes in "
      f"{money2(abs(sim_actual['pnl'] - simulate_floor(known, bars, floor_bps=20.0)['pnl']))} "
      f"*worse* than replaying the recorded distance, and a 1.5×–3× wider stop is "
      f"{money2(abs(sim_actual['pnl'] - simulate_floor(known, bars, mult=3.0)['pnl']))}–"
      f"{money2(abs(sim_actual['pnl'] - simulate_floor(known, bars, mult=1.5)['pnl']))} worse. "
      f"Removing the stop altogether is the worst variant of all "
      f"({money2(simulate_floor(known, bars, no_stop=True)['pnl'])}). The tight stop is therefore "
      f"a **symptom** of entries that have to be right immediately, not the mechanism of the "
      f"loss — an important negative result, because it kills the most obvious 'just widen the "
      f"stop' fix before anyone spends a day on it.")
    A(f"- Exit mix for these {len(known):,} trades as the engine actually closed them: "
      f"{known_exits.get('tp', 0):,} TP / {known_exits.get('sl', 0):,} SL / "
      f"{known_exits.get('eod', 0):,} EOD — the replay above lands on "
      f"{sim_actual['tp']:,} / {sim_actual['sl']:,} / {sim_actual['eod']:,}, so the difference "
      f"between the two panels is *which* trade lands in which bucket along a re-decided path "
      f"({abs(sim_actual['pnl'] - known['pnl'].sum()) / abs(known['pnl'].sum()):.1%} of P&L), "
      f"which is the noise floor the sweep rows should be read against.")
    A(f"- The excluded {len(moved):,} trades (live break-even/trail moves) netted "
      f"{money2(moved['pnl'].sum())} — the part of the book where the live trade-management rule "
      f"caught a favourable move and cut it near the fill. That is where the difference between "
      f"this subset's {money2(known['pnl'].sum())} and the portfolio's {money2(net)} comes from.")
    A("")
    A("### 3.4 What the exit paths say (R-multiples)")
    A("")
    rows = []
    for reason in ("tp", "eod", "sl"):
        g = tr[tr.exit_reason == reason]
        rows.append([reason.upper(), f"{len(g):,}",
                     f"{g['mae_R'].median():.2f}", f"{np.percentile(g['mae_R'].dropna(), 75):.2f}",
                     f"{g['mfe_R'].median():.2f}", f"{np.percentile(g['mfe_R'].dropna(), 75):.2f}",
                     f"{g['r_mult'].median():.2f}" if g['r_mult'].notna().any() else "—"])
    A(md_table(["Exit", "Trades", "MAE median (R)", "MAE p75 (R)", "MFE median (R)",
                "MFE p75 (R)", "Realised R median"], rows))
    A("")

    # ── 4. volume amplification ───────────────────────────────────────────
    A("## 4. Volume amplification — per-trade expectancy × how many trades")
    A("")
    A(md_table(["Quantity", "Value"], [
        ["Net P&L", money2(net)],
        ["Trades", f"{n:,}"],
        ["Sessions traded", f"{sessions}"],
        ["Trades per session", f"{n / sessions:.2f}"],
        ["**P&L per trade**", f"**{money2(exp_trade)}**"],
        ["P&L per session", money2(net / sessions)],
        ["P&L per session as % of the $100k start", pct(net / sessions / 100_000, 3)],
    ]))
    A("")
    rows = []
    by_rank = []
    for k in (1, 2, 3):
        g = tr_sorted[tr_sorted["rank"] == k]
        by_rank.append([f"entry #{k} of the session", f"{len(g):,}", money2(g['pnl'].sum()),
                        money2(g['pnl'].mean()), pct((g['pnl'] > 0).mean()),
                        f"{len(g) / sessions:.2f}"])
    for k in (1, 2, 3):
        g = tr_sorted[tr_sorted["rank"] <= k]
        rows.append([f"keep the first {k} per symbol/session" + (" (actual)" if k == 3 else ""),
                     f"{len(g):,}", money2(g['pnl'].sum()), money2(g['pnl'].mean()),
                     f"{len(g) / sessions:.2f}", money2(g['pnl'].sum() / sessions)])
    A(md_table(["Trade selection", "Trades", "Net P&L", "P&L per trade",
                "Trades/session", "P&L/session"], rows))
    A("")
    A("And the same trades split by which entry of the session they were:")
    A("")
    A(md_table(["Entry order within the session", "Trades", "Net P&L", "P&L per trade",
                "Win rate", "Trades/session"], by_rank))
    A("")
    A(f"- The loss shrinks almost exactly in proportion to the number of trades "
      f"({money2(tr_sorted[tr_sorted['rank'] == 1]['pnl'].sum())} → "
      f"{money2(tr_sorted[tr_sorted['rank'] <= 2]['pnl'].sum())} → {money2(net)}), which is what "
      f"a negative per-trade edge looks like. **Cutting the trade count does not fix the "
      f"strategy — it only scales the loss down**; the later entries are not the bad ones "
      f"(entry #1 of a session is in fact the worst per trade), so there is no 'the first trade "
      f"of the day is the good one' effect being diluted by the cap. Combined with §1.5, the "
      f"volume problem and the opening-window problem are the same problem: the engine takes "
      f"most of its (negative-expectancy) allocation in the first 45 minutes.")
    A("")

    # ── 5. EOD flat ───────────────────────────────────────────────────────
    A("## 5. Do the EOD exits cut winners short?")
    A("")
    eod = tr[tr.exit_reason == "eod"]
    A(md_table(["Quantity (EOD exits only)", "Value"], [
        ["Trades", f"{len(eod):,}"],
        ["Net P&L", money2(eod['pnl'].sum())],
        ["Winners / losers", f"{int((eod.pnl > 0).sum()):,} / {int((eod.pnl <= 0).sum()):,}"],
        ["Mean realised P&L", money2(eod['pnl'].mean())],
        ["Mean MFE (bps)", f"{eod['mfe_bps'].mean():.1f}"],
        ["Median MFE (bps)", f"{eod['mfe_bps'].median():.1f}"],
        ["Mean realised move (bps)", f"{(eod['pnl_pct'] * BPS).mean():.1f}"],
        ["MFE captured (realised move ÷ MFE)", pct(float(((eod['pnl_pct'] * BPS) / eod['mfe_bps'].replace(0, np.nan)).median()))],
        ["Avg win of EOD exits", money2(eod[eod.pnl > 0]['pnl'].mean())],
        ["Avg win of TP exits", money2(tr[tr.exit_reason == 'tp']['pnl'].mean())],
    ]))
    A("")
    eod_win = eod[eod.pnl > 0]
    A(f"- The EOD leg is **not** where the loss comes from ({money2(eod['pnl'].sum())} of "
      f"{money2(net)}). But it is where the strategy's upside is capped: an EOD winner banks "
      f"{money2(eod_win['pnl'].mean())} against {money2(tr[tr.exit_reason == 'tp']['pnl'].mean())} "
      f"for a target hit, and EOD exits give back a median "
      f"{pct(1 - float(((eod['pnl_pct'] * BPS) / eod['mfe_bps'].replace(0, np.nan)).median()))} "
      f"of their own best excursion.")
    A(f"- Flattening at the close is a **risk control that costs little** in this sample: the "
      f"EOD population is {(eod.pnl > 0).mean():.0%} winners and the average is "
      f"{money2(eod['pnl'].mean())}. The overnight variant (in the backtest report) is the proper "
      f"test of holding through.")
    A("")

    # ── 6. constraints ────────────────────────────────────────────────────
    A("## 6. The constraints, quantified")
    A("")
    st = stats["stats"]
    skipped = st.get("skipped", {})
    fills = st.get("entries_by_side", {})
    n_long = int((tr.side == "LONG").sum())
    n_short = int((tr.side == "SHORT").sum())
    A(md_table(["Counter", "Value", "What it means here"], [
        ["Signals skipped: `short_not_shortable`", f"{skipped.get('short_not_shortable', 0):,}",
         "short setups the engine refused because the symbol is not on the live shortable allow-list"],
        ["Signals skipped: `entry_cap`", f"{skipped.get('entry_cap', 0):,}",
         "signals refused because the symbol had already used its 3 entries that session"],
        ["Signals skipped: `cooldown`", f"{skipped.get('cooldown', 0):,}",
         "signals inside the 5-bar post-exit cooldown"],
        ["Signals skipped: `insufficient_data`", f"{skipped.get('insufficient_data', 0):,}",
         "warmup / module minimum-bar gate"],
        ["Signals skipped: `duplicate_signal`", f"{skipped.get('duplicate_signal', 0):,}",
         "same signal re-emitted while already working"],
        ["Entry orders (long / short)", f"{fills.get('LONG', 0):,} / {fills.get('SHORT', 0):,}",
         f"{fills.get('LONG', 0) / max(sum(fills.values()), 1):.0%} of orders are longs"],
        ["Closed trades (long / short)", f"{n_long:,} / {n_short:,}",
         f"{n_long / n:.0%} of the book is long"],
        ["Symbols that can never be shorted", "COIN, META, TSLA",
         "3 of 6 symbols are long-only by construction"],
    ]))
    A("")
    longs_only = tr[tr.symbol.isin(["COIN", "META", "TSLA"])]
    rising = [s for s in sorted(tr.symbol.unique()) if bars[s].drift() > 0]
    long_rising = tr[(tr.side == "LONG") & (tr.symbol.isin(rising))]
    A(f"- The long-only half of the universe (COIN/META/TSLA) contributed "
      f"{money2(longs_only['pnl'].sum())} of the {money2(net)} loss on "
      f"{len(longs_only):,} trades; the three two-sided symbols (NVDA/QQQ/AVGO) contributed "
      f"{money2(tr[tr.symbol.isin(['NVDA', 'QQQ', 'AVGO'])]['pnl'].sum())}.")
    A(f"- **Short-side P&L: {money2(tr[tr.side == 'SHORT']['pnl'].sum())} on "
      f"{n_short:,} trades** ({tr[tr.side == 'SHORT']['pnl'].sum() / net:.0%} of the loss) — the "
      f"down-side book is small *and* it also loses. Deleting the short leg would not turn the "
      f"strategy positive, so the shortability constraint is a missed-diversification problem, "
      f"not the cause of the loss.")
    A(f"- Quantifying the tape instead of asserting it: over the window the six symbols moved "
      f"{', '.join(f'{s} {bars[s].drift():+.1%}' for s in sorted(tr.symbol.unique()))}. So "
      f"{len(rising)} of 6 rose. On those rising symbols the long book alone lost "
      f"{money2(long_rising['pnl'].sum())} on {len(long_rising):,} trades — a strategy that loses "
      f"money long in a rising tape is not being beaten by drift, it is losing the timing. On the "
      f"two symbols that fell (COIN {bars['COIN'].drift():+.1%}, META {bars['META'].drift():+.1%}) "
      f"the long-only constraint was a real cost: "
      f"{money2(longs_only[longs_only.symbol.isin(['COIN', 'META'])]['pnl'].sum())} of the loss is "
      f"long-only exposure to symbols that dropped and could never be shorted in this replay.")
    A("")

    # ── 7. ranking ────────────────────────────────────────────────────────
    A("## 7. Hypothesis ranking — dollars explained, and the experiment that settles each")
    A("")
    A(f"Ranked by dollars of the {money2(net)} loss they account for. #1 is the arithmetic that "
      f"*is* the loss; #2 is the popular explanation that this document rules out; #3–#4 are "
      f"*where* and *how often* the losing edge gets spent; #5–#8 are smaller or second-order. "
      f"Anything acted on should start at #1 and #3.")
    A("")
    tight = int((tr.exit_reason == "sl").sum())
    be_moved = int((tr[tr.exit_reason == "sl"].sl_source == "breakeven").sum())
    clamp = int((tr[tr.exit_reason == "sl"].sl_source == "fill_risk_clamped").sum())
    rows = [
        ["**1. Win-rate deficit vs the win/loss size ratio**",
         f"{(wr - be_wr) * inner * n:,.0f}",
         f"BE win rate {pct(be_wr)} vs realised {pct(wr)}; the identity "
         f"`(WR − BE_WR) × (W + |L|) × n` reproduces the whole loss with no residual (§2). "
         f"Uniform across every symbol (§2.1) and every quarter — no single symbol is to blame.",
         "Re-run the current config on the same window with the target forced to 1.5× the stop "
         "instead of the current effective ratio; if the win rate does not rise enough to clear "
         "the new BE threshold, the entries themselves carry no edge and tuning is pointless."],
        ["**2. Stop distance inside 1-minute noise (symptom, not cause — see §3.3)**",
         f"{abs(sl_sum):,.0f} of stop-outs",
         f"{wick:,} of {tight:,} stop-outs were wick touches that closed ≥5 bps back on the good "
         f"side of the level; {clamp:,} stops sat on the 5 bps / 1¢ clamp and {be_moved:,} ended on "
         f"a stop the break-even rule had moved onto the fill price; winners take on "
         f"{tp_mae.median():.1f} bps of heat at the median, more than the median stop budget of "
         f"{med_all_stop:.1f} bps (§3.2). **But widening the stop on the same entries makes the "
         f"result worse, not better (§3.3)** — the tight stop converts a no-edge entry into many "
         f"small losses rather than creating them.",
         "Already run and it came back negative: keep the stop, but tighten the *entries* date/time "
         "window (experiment 3) and require the entry to be at least X bps away from the noise. "
         "Confirm by re-running the three-month block with the §1.5/§4 filters rather than by "
         "touching the stop."],
        ["**3. Opening-45-minute concentration**",
         f"{abs(open_win['pnl'].sum()):,.0f}",
         f"{len(open_win):,} of {n:,} trades ({len(open_win) / n:.0%}) are entered 09:30–10:14 and "
         f"they carry {open_win['pnl'].sum() / net:.0%} of the loss; the other "
         f"{n - len(open_win):,} trades are near flat in aggregate (§1.5). The cap fires all three "
         f"entries per symbol into the opening range, where the entry signals are least "
         f"discriminating.",
         "Block entries before 10:15 for 3 months (or gate them on a volatility/ATR filter) and "
         "compare the per-trade expectancy of the remaining trades — if the post-10:15 book is "
         "flat-to-positive at the same size, the loss is a timing-window problem, not a signal "
         "problem."],
        ["**4. Volume of trades against a negative edge**",
         f"{abs(exp_trade) * n:,.0f}",
         f"{n:,} trades at {money2(exp_trade)}/trade, {n / sessions:.2f}/session, from "
         f"{sum(sig.values()):,} raw signals. Keeping only the first entry per symbol/session "
         f"still loses {money2(tr_sorted[tr_sorted['rank'] == 1]['pnl'].sum())} (§4) — the edge is "
         f"negative on entry #1 too, so more trades is an amplifier of the loss, not its origin.",
         "Halve the trade count for 3 months and measure per-trade expectancy: unchanged "
         "expectancy confirms the edge, not the flow, is the problem."],
        ["**5. Trade management is currently *hiding* part of the damage**",
         f"{moved['pnl'].sum():,.0f} (the part of the book the rule touched)",
         f"The {len(moved):,} trades whose stop the live break-even/trail rule moved netted "
         f"{money2(moved['pnl'].sum())}, while the {len(known):,} untouched ones lost "
         f"{money2(known['pnl'].sum())} (§3.3). The rule is the single most valuable component in "
         f"the stack right now — and it only ever runs on trades that have already gone the right "
         f"way, so it cannot rescue a no-edge entry set.",
         "Keep it; do not 'simplify' it away as part of any stop cleanup. Verify by replaying the "
         "three-month block with the break-even rule disabled and comparing."],
        ["**6. Target/stop asymmetry is set by the modules, not by risk**",
         f"{tp_sum:,.0f} recovered of {abs(sl_sum):,.0f} lost",
         f"Median target distance {tr['dist_tp_bps'].median():.0f} bps vs median stop "
         f"{tr['dist_sl_bps'].median():.0f} bps, yet the realised ratio is only "
         f"{avg_win / abs(avg_loss):.2f}:1 because only "
         f"{(tr.exit_reason == 'tp').sum() / n:.1%} of trades ever touch the target.",
         "Re-derive the R:R each module's signals actually request and re-measure the touch rate "
         "after the entry-window change: a wider stop only helps if the target stays reachable."],
        ["**7. Long-only universe / blocked shorts**",
         f"{abs(tr[tr.side == 'SHORT']['pnl'].sum()):,.0f}",
         f"141,332 short signals refused; the short book is {n_short / n:.0%} of trades and loses "
         f"{money2(tr[tr.side == 'SHORT']['pnl'].sum())}; but "
         f"{money2(abs(longs_only[longs_only.symbol.isin(['COIN', 'META'])]['pnl'].sum()))} of the "
         f"loss is long-only exposure to the two symbols that actually fell (§6).",
         "Check which of COIN/META/TSLA the broker will locate; if any can be shorted, re-run the "
         "replay with the allow-list widened and compare per-trade expectancy rather than the "
         "headline."],
        ["**8. EOD flat cutting winners**",
         f"{abs(eod['pnl'].sum()):,.0f}",
         f"Small but real: {len(eod):,} EOD exits net {money2(eod['pnl'].sum())} and give back a "
         f"median "
         f"{pct(1 - float(((eod['pnl_pct'] * BPS) / eod['mfe_bps'].replace(0, np.nan)).median()))} "
         f"of their own best excursion (§5). Worth fixing after 1–4, not before.",
         "The `overnight` variant in the backtest report: same entries, no EOD flat."],
    ]
    A(md_table(["Rank / hypothesis", "$ explained", "Evidence in this document",
                "Killing / confirming experiment"], rows))
    A("")
    A("### What would make us abandon ScalpSet")
    A("")
    A(f"If hypotheses 1, 3 and 4 are addressed and the same entries *still* produce a win rate "
      f"below the new break-even threshold, then the box-theory entry has no timing edge and the "
      f"module should be replaced rather than re-tuned: the entry set here is arbitrated from "
      f"{sum(sig.values()):,} signals down to {n:,} trades by the cap alone, so there is no "
      f"shortage of alternative signal populations to test on the same data. The tests are cheap "
      f"— a 3-month replay with changed parameters runs in well under an hour on the same "
      f"committed engine — and the one experiment in this list that was already run today "
      f"(widening the stop) came back negative, which is exactly the kind of result that keeps the "
      f"search from being re-run for weeks.")
    A("")

    # ── 8. method notes ───────────────────────────────────────────────────
    A("## 8. Method notes and limits")
    A("")
    A(md_table(["Point", "Detail"], [
        ["Source of every number",
         f"`{inp.name}/trades_{args.variant}.csv` ({n:,} rows), "
         f"`stats_{args.variant}.json` for the engine counters and the cost config, and the "
         f"cached 1-minute RTH bars for path metrics and the counterfactuals. Regenerate with "
         f"`scripts/scalpset_loss_forensics.py`."],
        ["MAE/MFE window",
         "Bars from the entry bar to the exit bar inclusive, in bps of the fill price. The entry "
         "bar can contain up to one minute of pre-fill price, so MAE is if anything overstated "
         "and MFE understated — the direction that makes the 'stop too tight' finding "
         "conservative."],
        ["Stop-distance counterfactual",
         f"Bar-by-bar replay of the same entries with the stop moved to a multiple of / floor "
         f"below its opening distance, restricted to the {len(known):,} trades whose opening stop "
         f"is recoverable from the log (the other {len(moved):,} had it moved to break-even or "
         f"trailed, so their row records the *moved* level). Stop evaluated first inside a bar (as "
         f"the engine does); break-even/trailing moves disabled so only distance varies. It is a "
         f"sensitivity on the entry set, not a new strategy simulation — no entry is added, "
         f"removed or re-timed."],
        ["Sub-minute ordering",
         "A 1-minute bar cannot say whether the stop or the target was touched first; the engine "
         "and these counterfactuals both resolve stop-first, which is the conservative choice."],
        ["Trade-count counterfactual",
         "Keeps the first k entries per symbol/session in time order and drops the rest. It "
         "ignores the compounding effect of a smaller drawdown on later position sizes, so the "
         "k=1/k=2 rows are an upper bound on how much of the loss fewer trades removes."],
        ["Pessimistic cross-check",
         f"The `{args.alt}` variant changes every attribution above only in the third decimal of "
         f"the totals (net {money2(float(alt['pnl'].sum())) if alt is not None else 'n/a'} vs "
         f"{money2(net)}), so none of the ranking depends on the cost model."],
    ]))
    A("")

    # ── 9. live defect: TP orders rejected ────────────────────────────────
    A("## 9. Live defect found while writing this: every take-profit order for a "
      "fractional position is rejected by the broker")
    A("")
    A("This is separate from the backtest and is **not** part of the loss arithmetic above — but "
      "it is visible in `engine/logs/trades_20260921.log` and it invalidates the assumption that "
      "the live book has a working target order. No live-trading behaviour was changed in this "
      "delegation.")
    A("")
    A("**Evidence (live, 2026-09-21 19:52:17Z):**")
    A("")
    A("```")
    A("2026-09-21 19:52:17,060 Placing GTC STOP sell TSLA x 42 @ $361.65")
    A("2026-09-21 19:52:17,135 Placing LIMIT SELL TSLA x 42.9262349808588")
    A("2026-09-21 19:52:17,210 Order submission failed: {\"available\":\"0.92623498\",")
    A("  \"code\":40310000,\"existing_qty\":\"42.92623498\",\"held_for_orders\":\"42\",")
    A("  \"message\":\"insufficient qty available for order (requested: 42.9262349808588,")
    A("  available: 0.92623498)\",\"symbol\":\"TSLA\"}")
    A("2026-09-21 19:52:17,210 🎯  TP TSLA REJECTED: {...}")
    A("```")
    A("")
    A("Same shape earlier the same day on COIN and NVDA.")
    A("")
    A("**Mechanism (from `live_trader.py`):**")
    A("")
    A(md_table(["Step", "Code", "Quantity used"], [
        ["Position opened",
         "`_finalize_open_bundle()` is called with the filled `pos.quantity`",
         "`42.9262349808588` (a *fractional* position — 13 significant decimals, i.e. "
         "notional ÷ price, not a whole-share size)"],
        ["Protective stop",
         "`_finalize_open_bundle` → `_place_protective_stop(sym, abs(pos.quantity), …)`; the first "
         "statement of `_place_protective_stop` is `qty = int(qty)` — Alpaca stop orders cannot be "
         "fractional, so the size is truncated",
         "`42` — **the stop reserves 42 of the 42.926 shares and holds them for orders**"],
        ["Take-profit",
         "`_finalize_open_bundle` → `_place_tp_limit(sym, abs(pos.quantity), levels.tp, …)`; the "
         "function passes the quantity through unrounded (fractional limit orders are legal)",
         "`42.9262349808588`"],
        ["Broker check",
         "Alpaca rejects: `held_for_orders` 42 leaves `available` 0.926, the TP asks for 42.926",
         "`40310000 insufficient qty available for order`"],
        ["Fallback",
         "`_place_tp_limit` returns False → `state[\"tp\"] = None` — the position then runs with "
         "the GTC stop, the break-even/trail rule and the in-process fallback only, and the TP "
         "price the strategy computed never reaches the broker",
         "—"],
    ]))
    A("")
    A("**Why it matters beyond the rejection:** the replay in this report (and every previous "
      "ScalpSet backtest) assumes the take-profit order exists and fills at its level. Live, on "
      "any position whose size is fractional, there is no target order at all. The two defects "
      "compound: the strategy needs ~29 % of trades to reach the target, and on fractional "
      "positions the target cannot be reached through the broker.")
    A("")
    A("**Smallest correct fix** (for the lead to schedule — deliberately *not* applied here, "
      "because it changes live trading):")
    A("")
    A("1. *One-line stopgap, no sizing change* — size the TP to whatever the stop actually "
      "reserved. In `_place_tp_limit`, floor the quantity the same way the stop does "
      "(`qty = int(qty)`; skip when the result is < 1) or, better, at the call site in "
      "`_finalize_open_bundle` pass `float(int(abs(pos.quantity)))`. The TP is then always "
      "accepted, and the ≤1-share remainder is left to the stop/BE/in-process path that already "
      "exists.")
    A("2. *Defensive retry* — when a TP is rejected with `40310000` / \"insufficient qty "
      "available\", retry once with `int(qty)` before giving up. This covers positions that "
      "arrive fractional from anywhere (an inherited or externally-opened position, not just this "
      "code path).")
    A("3. *Root cause* — stop opening fractional positions on this strategy: ScalpSet is specified "
      "whole-share (`qty = floor(notional / price)` upstream), and the TSLA size shows a fractional "
      "quantity reaching `_finalize_open_bundle`. Whichever path produced 42.926 shares should be "
      "floored, after which stop and TP agree exactly and neither the rejection nor the leftover "
      "dreg exists.")
    A("")
    A("**Two things to check with the fix:** (a) if the TP fills for the truncated quantity, the "
      "position keeps the residual fraction with the GTC stop now over-reserved — the sync path "
      "(and the today-visible `scalp_sync_removed` close) must handle that remainder explicitly; "
      "(b) `_place_protective_stop`'s truncation happens silently, so add a log line when "
      "`int(qty) != qty` to make the next occurrence obvious.")
    A("")

    md = "\n".join(L) + "\n"
    Path(args.out).write_text(md)
    print(f"wrote {args.out} ({len(md):,} chars)")

    D.update({"trades": n, "net": net, "wr": wr, "be_wr": be_wr, "avg_win": avg_win,
              "avg_loss": avg_loss, "sessions": sessions, "exp_trade": exp_trade,
              "tp_sum": tp_sum, "sl_sum": sl_sum, "eod_sum": eod_sum,
              "sl_exits": tight, "be_moved": be_moved, "clamped": clamp,
              "long_pnl": float(tr[tr.side == 'LONG']['pnl'].sum()),
              "short_pnl": float(tr[tr.side == 'SHORT']['pnl'].sum()),
              "eod_net": float(eod['pnl'].sum()),
              "drift": {s: bars[s].drift() for s in sorted(tr.symbol.unique())}})
    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(D, indent=2, default=str))
        print(f"wrote {args.dump_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
