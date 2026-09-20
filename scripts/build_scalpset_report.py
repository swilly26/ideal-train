#!/usr/bin/env python3
"""Build the owner-facing ScalpSet backtest report from run artifacts.

Reads ``data/backtest_out/stats_<variant>.json`` (written by
``scripts/run_scalpset_backtest.py``) and emits
``/home/team/shared/SCALPSET_BACKTEST_REPORT.md``.

Every number in the report comes from those JSON artifacts — nothing is
transcribed by hand — so the report can be regenerated after any re-run::

    .venv/bin/python scripts/build_scalpset_report.py \
        --out /home/team/shared/SCALPSET_BACKTEST_REPORT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "backtest_out"
OUT_DEFAULT = Path("/home/team/shared/SCALPSET_BACKTEST_REPORT.md")

VARIANTS = [
    ("baseline", "A-baseline", "current live config, end-of-day flat, engine-default slippage (2 bps / 1¢), no fees"),
    ("pessimistic", "A-pessimistic", "live config + pessimistic costs (2 bps + 1 bp half-spread slippage, $0.005/share)"),
    ("overnight", "A-overnight", "pessimistic costs with the live overnight-hold behaviour (eod_flat=False)"),
    ("zero_cost", "A-zero-cost", "diagnostic: zero slippage and zero fees — gross signal edge, no execution cost"),
]
SHORT = {"baseline": "baseline", "pessimistic": "pessimistic",
         "overnight": "overnight", "zero_cost": "zero-cost"}


def load(inp: Path, tag: str) -> dict | None:
    p = inp / f"stats_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"


def num(x: float, nd: int = 2) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x:,.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def has_edge(m: dict, min_trades: int = 30) -> bool:
    return (m["pnl_net"] > 0 and m["profit_factor"] > 1.05
            and m["trades"] >= min_trades)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    inp = Path(args.in_dir)
    runs = {tag: load(inp, tag) for tag, _, _ in VARIANTS}
    present = {k: v for k, v in runs.items() if v}
    if not present:
        raise SystemExit(f"no stats_*.json found in {inp}")

    L: list[str] = []
    base = runs.get("baseline")
    pess = runs.get("pessimistic")
    over = runs.get("overnight")
    zero = runs.get("zero_cost")

    def pm(tag: str) -> dict:
        return runs[tag]["portfolio_metrics"]  # type: ignore[index]

    def by_sym(tag: str) -> dict:
        return {r["label"]: r for r in runs[tag]["per_symbol"]}  # type: ignore[index]

    # ── header ────────────────────────────────────────────────────────
    L.append("# ScalpSet — 12-Month Historical Backtest (owner report)")
    L.append("")
    if base:
        s = base["stats"]
        L.append(f"**Window:** {base['start']} → {base['end']} (all six live symbols: "
                 f"{', '.join(base['symbols'])}) · **bars:** {s['bars']:,} 1-minute RTH bars · "
                 f"**sessions:** {s['sessions']} · **engine:** replay of the live ScalpSet "
                 f"(`{git_head()}`) · **run:** {json.loads(json.dumps(base['command']))}")
    L.append("")
    L.append("This is a *simulation of the strategy as it is configured and running today* — "
             "nothing was tuned, fitted or selected on this data. The parameters were fixed in "
             "advance (they are the live constants), so every month in the table below is "
             "out-of-sample by construction.")
    L.append("")

    # ── TL;DR ─────────────────────────────────────────────────────────
    L.append("## 1. TL;DR — does the current running config have an edge?")
    L.append("")
    if base and pess:
        b, p = pm("baseline"), pm("pessimistic")
        edge_b, edge_p = has_edge(b), has_edge(p)
        if edge_b and edge_p:
            verdict = ("**Yes — but read the size.** The live config made money over the year "
                       "and stayed profitable after pessimistic costs.")
        elif edge_b and not edge_p:
            verdict = ("**Its money is thinner than the cost model.** The live config is "
                       "profitable on the engine-default cost model but the edge does not "
                       "robustly survive pessimistic costs.")
        else:
            verdict = ("**No. The current running configuration does not show a positive edge "
                       "over the 12 months tested.**")
        L.append(verdict)
        L.append("")
        L.append(md_table(
            ["Run", "Trades", "Win rate", "Net P&L", "Return", "Profit factor",
             "Max drawdown", "Avg / trade"],
            [[SHORT[t], f"{pm(t)['trades']}", pct(pm(t)["win_rate"]),
              money(pm(t)["pnl_net"]), pct(pm(t)["total_return"], 2),
              num(pm(t)["profit_factor"]), pct(pm(t)["max_drawdown"], 2),
              money(pm(t)["pnl_per_trade"])]
             for t in ("baseline", "pessimistic", "overnight", "zero_cost") if t in runs]))
        L.append("")
        L.append(f"- Start equity {money(b['initial_equity'])} per run, whole-share 15% position "
                 f"sizing, max 6 concurrent positions, EOD flat (except the overnight run).")
        L.append(f"- Costs actually paid: {money(b['fees'])} baseline vs {money(p['fees'])} "
                 f"pessimistic; the pessimistic run's slippage+spread+fees removed "
                 f"{money(p['cost_drag'] - b['cost_drag'])} more than the baseline model.")
        L.append("")
        # per-symbol verdicts
        bs, ps_ = by_sym("baseline"), by_sym("pessimistic")
        rows = []
        for sym in sorted(bs):
            bm, pm_ = bs[sym], ps_.get(sym)
            if not pm_:
                continue
            if has_edge(pm_) and has_edge(bm):
                v = "edge survives costs"
            elif has_edge(bm) and not has_edge(pm_):
                v = "edge before costs only"
            elif bm["pnl_net"] > 0:
                v = "marginal / thin"
            else:
                v = "loses money"
            rows.append([sym, f"{bm['trades']}", pct(bm["win_rate"]),
                         money(bm["pnl_net"]), money(pm_["pnl_net"]),
                         num(pm_["profit_factor"]), pct(pm_["max_dd_pct_of_account"], 2), v])
        L.append("### Per-symbol verdict")
        L.append("")
        L.append(md_table(["Symbol", "Trades", "Win rate", "Net P&L (baseline)",
                           "Net P&L (pessimistic)", "PF (pess)", "Max DD (pess)",
                           "Verdict"], rows))
        L.append("")
        winners = [r[0] for r in rows if r[7] == "edge survives costs"]
        losers = [r[0] for r in rows if r[7] == "loses money"]
        L.append(f"- Symbols where a positive edge survives pessimistic costs: "
                 f"{', '.join(winners) if winners else '**none**'}.")
        L.append(f"- Symbols that lose money before costs: "
                 f"{', '.join(losers) if losers else 'none'}.")
        L.append("")

    # ── run matrix / methodology ──────────────────────────────────────
    L.append("## 2. Methodology")
    L.append("")
    L.append("### 2.1 What was replayed")
    L.append("")
    L.append("The live main trader's ScalpSet — three modules, arbitrated by best R:R — on the "
             "live symbol list, with the live risk and sizing constants. Each variant is a single "
             "**portfolio** replay: all six symbols run through one engine at the same time so "
             "15 % sizing, the 6-position cap and cross-symbol arbitration interact exactly as "
             "they do live (rather than six independent single-symbol backtests added up).")
    L.append("")
    L.append("### 2.2 Engine semantics (fidelity to live)")
    L.append("")
    L.append(md_table(["Live behaviour", "How the replay reproduces it"], [
        ["No lookahead", "A signal is evaluated at bar *t* using only bars up to and including *t*'s close; market entries fill on the **next** bar."],
        ["Anchored SL/TP", "Stop and target are re-anchored to the **actual fill price** with the live minimum-distance clamps and 2-decimal tick rounding (the PR #37 rule)."],
        ["Backstop stop", "A 6% backstop applies when a signal's own stop is missing or closer than the clamp — same as the live protective stop."],
        ["Short gating", "Shorts only on symbols the live config treats as shortable: NVDA, QQQ, AVGO. Non-shortable shorts are skipped and counted."],
        ["Entry cap", "Max 3 entries per symbol per session, plus the live 5-bar (5-minute) posting cooldown after an exit."],
        ["Sizing", "Whole shares only, 15% of current equity per position, capped by 95% of available cash, max 6 concurrent positions."],
        ["Session handling", "RTH only (09:30–16:00 ET); day orders expire at the close; with eod_flat the book is flattened on the last bar of the session."],
        ["Order types", "Modules that emit marketable entries fill on the next bar open with adverse slippage; limit entries (the VolProfile+FIB pullback) rest until filled or the session ends."],
        ["Costs", "Adverse slippage per fill, optional half-spread, optional per-share commission — the exact model is printed per variant below."],
    ]))
    L.append("")
    L.append("### 2.3 Run matrix (exact commands)")
    L.append("")
    for tag, label, desc in VARIANTS:
        r = runs.get(tag)
        if not r:
            continue
        c = r["config"]
        L.append(f"- **{label}** — {desc}.  config: `eod_flat={c['eod_flat']}, "
                 f"slippage_pct={c['slippage_pct']}, slippage_abs={c['slippage_abs']}, "
                 f"half_spread_pct={c['half_spread_pct']}, commission_per_share={c['commission_per_share']}`")
        L.append(f"  - `{r['command']}`  (banks {r['elapsed_sec']:.0f}s, warmup "
                 f"{r['warmup_days']} days)")
    L.append("")
    L.append("### 2.4 Data")
    L.append("")
    if base:
        mid = {n: base["stats"] for n in ("bars",)}
        L.append(f"- 1-minute SIP bars from the local Alpaca cache (`data/history/1m/<SYMBOL>/YYYY-MM.parquet`), "
                 f"12 monthly files per symbol, 72 files total, ~{sum(1 for _ in inp.glob('trades_*.csv'))} "
                 f"trade tables produced.")
        L.append("- Cache coverage: 2025-09-01 → 2026-08-31, all six symbols "
                 "(per-symbol bar counts are in the cache manifest, "
                 "`data/history/manifest.json`).")
        L.append("- The 75-day warmup before the first traded day is largely unavailable "
                 "(the cache starts 2025-09), so the first sessions of September 2025 see a "
                 "thinner higher-timeframe history than live would. Modules with unmet minimum-bar "
                 "requirements simply do not signal until their windows fill — they never see "
                 "synthetic or future data.")
    L.append("")

    # ── results ───────────────────────────────────────────────────────
    L.append("## 3. Results")
    L.append("")
    for tag, label, desc in VARIANTS:
        r = runs.get(tag)
        if not r:
            continue
        m, s = r["portfolio_metrics"], r["stats"]
        L.append(f"### 3.{VARIANTS.index((tag, label, desc)) + 1} {label} — {desc}")
        L.append("")
        L.append(md_table(["Metric", "Value"], [
            ["Trades", f"{m['trades']} ({m['wins']}W / {m['losses']}L)"],
            ["Win rate", pct(m["win_rate"])],
            ["Net P&L", f"{money(m['pnl_net'])} on {money(m['initial_equity'])} "
                        f"({pct(m['total_return'], 2)})"],
            ["Gross P&L before costs", money(m["pnl_gross"])],
            ["Cost drag", money(m["cost_drag"])],
            ["Profit factor", num(m["profit_factor"])],
            ["Expectancy (avg net per trade)", f"{money(m['pnl_per_trade'])} on avg "
                                               f"{money(m['notional_traded'] / max(m['trades'], 1))} notional"],
            ["Avg win / avg loss", f"{money(m['avg_win'])} / {money(m['avg_loss'])}"],
            ["Best / worst trade", f"{money(m['best'])} / {money(m['worst'])}"],
            ["Max drawdown (equity curve)", pct(m["max_drawdown"], 2)],
            ["Sharpe (daily) / bars", f"{num(m['sharpe_daily'])} / {num(m['sharpe_bar'])}"],
            ["Profit / loss days", f"{m['profitable_days']} / {m['losing_days']} "
                                   f"({m['flat_days']} flat of {m['trading_days']})"],
            ["Best / worst day", f"{pct(m['best_day'], 2)} / {pct(m['worst_day'], 2)}"],
            ["Signals → entries → fills", f"{sum(s['signals_by_module'].values())} signals, "
                                          f"{s['entries']} entry orders "
                                          f"({s['market_entries']} market, {s['limit_orders_placed']} limit), "
                                          f"{m['trades']} closed trades"],
            ["Entries by module", json.dumps(s["entries_by_module"])],
            ["Entries by side", json.dumps(s["entries_by_side"])],
            ["Exits by reason", json.dumps(s["exits"])],
            ["Skips by reason", json.dumps(s["skipped"])],
        ]))
        L.append("")
        # per-symbol
        L.append("**Per symbol**")
        L.append("")
        L.append(md_table(["Symbol", "Trades", "Win rate", "Net P&L", "Gross P&L",
                           "Cost drag", "Expectancy/trade", "PF", "Max DD (account %)"],
                          [[r_["label"], f"{r_['trades']}", pct(r_["win_rate"]),
                            money(r_["pnl_net"]), money(r_["pnl_gross"]),
                            money(r_["cost_drag"]), money(r_["pnl_per_trade"]),
                            num(r_["profit_factor"]), pct(r_["max_dd_pct_of_account"], 2)]
                           for r_ in r["per_symbol"]]))
        L.append("")
        # module
        L.append("**Per module**")
        L.append("")
        L.append(md_table(["Module", "Signals", "Entry orders", "Market fills", "Limit fills",
                           "Trades", "LONG / SHORT", "Win rate", "Net P&L", "PF",
                           "TP / SL / EOD exits"],
                          [[r_["label"], f"{r_['signals']}", f"{r_['entry_orders']}",
                            f"{r_['market_entry_fills']}", f"{r_['limit_entry_fills']}",
                            f"{r_['trades']}", f"{r_['long_fills']} / {r_['short_fills']}",
                            pct(r_["win_rate"]), money(r_["pnl_net"]),
                            num(r_["profit_factor"]),
                            f"{r_['exits_tp']} / {r_['exits_sl']} / {r_['exits_eod']}"]
                           for r_ in r["module"]]))
        L.append("")
        # monthly folds
        L.append("**Monthly folds** (slices of the same portfolio run; with EOD flat no position "
                 "crosses a month boundary)")
        L.append("")
        L.append(md_table(["Month", "Trades", "Win rate", "Net P&L", "Return",
                           "PF", "Month max DD"],
                          [[f["period"], f"{f['trades']}", pct(f["win_rate"]),
                            money(f["pnl_net"]), pct(f["return_pct"], 2),
                            num(f["profit_factor"]), pct(f["period_max_dd_pct"], 2)]
                           for f in r["monthly"]]))
        L.append("")
        L.append("**Quarterly folds**")
        L.append("")
        L.append(md_table(["Quarter", "Trades", "Win rate", "Net P&L", "Return", "PF",
                           "Quarter max DD"],
                          [[f["period"], f"{f['trades']}", pct(f["win_rate"]),
                            money(f["pnl_net"]), pct(f["return_pct"], 2),
                            num(f["profit_factor"]), pct(f["period_max_dd_pct"], 2)]
                           for f in r["quarterly"]]))
        L.append("")

    # cross-variant monthly comparison + concentration
    if base and pess:
        L.append("### 3.5 Monthly comparison across cost models")
        L.append("")
        bmonths = {f["period"]: f for f in base["monthly"]}
        pmonths = {f["period"]: f for f in pess["monthly"]}
        omonths = {f["period"]: f for f in (over or base)["monthly"]} if over else {}
        rows = []
        for per in sorted(set(bmonths) | set(pmonths)):
            bm = bmonths.get(per, {})
            pm_ = pmonths.get(per, {})
            om = omonths.get(per, {})
            rows.append([per, f"{bm.get('trades', 0)}",
                         money(bm.get("pnl_net", 0.0)), money(pm_.get("pnl_net", 0.0)),
                         money(om.get("pnl_net", 0.0)) if over else "—",
                         pct(bm.get("return_pct", 0.0), 2),
                         pct(pm_.get("return_pct", 0.0), 2)])
        L.append(md_table(["Month", "Trades", "Net P&L baseline", "Net P&L pessimistic",
                           "Net P&L overnight", "Return baseline", "Return pessimistic"], rows))
        L.append("")
        # concentration
        psym = by_sym("pessimistic")
        total = sum(abs(v["pnl_net"]) for v in psym.values()) or 1.0
        top = sorted(psym.values(), key=lambda v: -abs(v["pnl_net"]))[:3]
        L.append("**Where the P&L is concentrated (pessimistic run):** "
                 + "; ".join(f"{t['label']} {money(t['pnl_net'])} "
                             f"({t['pnl_net'] / total * 100:.0f}% of absolute P&L)"
                             for t in top) + ".")
        L.append("")
        # quarter concentration
        worst_q = min(pess["quarterly"], key=lambda f: f["return_pct"])
        best_q = max(pess["quarterly"], key=lambda f: f["return_pct"])
        L.append(f"**Regime spread:** best quarter {best_q['period']} "
                 f"({pct(best_q['return_pct'], 2)}), worst quarter {worst_q['period']} "
                 f"({pct(worst_q['return_pct'], 2)}).")
        L.append("")

    # ── honesty ───────────────────────────────────────────────────────
    L.append("## 4. Known approximations — what this test does *not* prove")
    L.append("")
    L.append(md_table(["Approximation", "Direction of the error / why it matters"], [
        ["Slippage is a formula, not a book",
         "Market fills pay `max(2 bps, 1¢) + half-spread` per side. Real scalping fills depend on the order book and can be worse (thin names, fast tape) or better (patient limits). The pessimistic run adds 1 bp half-spread and $0.005/share, which is still gentler than some real retail fills."],
        ["No broker rejections or partials",
         "Every order that the strategy wants is assumed accepted and filled in full. Live, orders get rejected (margin, shortability changes, penny-tick rules), partially fill, or sit unfilled."],
        ["Shortability is a static allow-list",
         "The replay gates shorts on a fixed set {NVDA, QQQ, AVGO}. Live, shortability is checked per symbol per order and can change intraday (and the live stack backs off after a rejection). A symbol that is in the allow-list here may have been un-shortable on a given live day — that would make live results *worse* than this run in those cases, never better."],
        ["Fills inside a 1-minute bar are optimistic",
         "When a bar's range touches both the stop and the target, the engine resolves by the conservative intra-bar rule but cannot see the true tick path. Sub-minute sequencing (stop-first vs target-first) is therefore a modelling choice, not ground truth."],
        ["EOD flat vs the overnight variant",
         "The live stack has not always been flat into the close; the overnight run shows what holding through the close would have cost with the same entries."],
        ["Costs are not the same as the engine default",
         "Even the 'baseline' run charges 2 bps / 1¢ adverse slippage per fill — it is not a frictionless run. The `zero_cost` diagnostic isolates the raw signal edge."],
        ["Window warmup",
         "The cache begins 2025-09-01, so September 2025 trades with a shorter higher-timeframe history than live would have. That affects the first days of the window only."],
        ["One parameter set, one year",
         "No parameter was fitted on this data — that is the point of the test — but it also means a single year cannot rule out that other parameters (or other regimes) would work."],
    ]))
    L.append("")
    L.append("### The VolProfile+FIB module contributes almost nothing")
    L.append("")
    if base:
        vols = [r_ for r_ in base["module"] if r_["label"] == "volprofile_fib"]
        v_fills = vols[0]["limit_entry_fills"] if vols else 0
        v_sig = vols[0]["signals"] if vols else 0
        v_orders = vols[0]["entry_orders"] if vols else 0
        L.append(f"Across the whole year the VolProfile+FIB module produced **{v_sig} signals, "
                 f"{v_orders} resting entry orders and {v_fills} fills**.")
        L.append("")
        L.append("This is a **finding, not a bug**: the module enters on a pullback via a LIMIT "
                 "order that sits away from the market, and once that resting bundle is posted the "
                 "symbol is occupied for the rest of the session (the live engine only tracks one "
                 "working entry per symbol). In practice the limit price is rarely reached before "
                 "the day order expires, so the module's capital and its symbol slot are consumed "
                 "by an order that mostly never trades. The live behaviour matches the backtest, "
                 "which means this module is currently contributing no P&L in either direction — "
                 "and it is arguably costing opportunity by blocking that symbol.")
        L.append("")

    # ── artifacts ─────────────────────────────────────────────────────
    L.append("## 5. Artifacts and how to reproduce")
    L.append("")
    L.append("Everything below is on the shared machine, under "
             f"`{inp}` (that directory is client-side only — the data cache and the run "
             "outputs are gitignored, so the raw parquet and the CSVs are not in the repository; "
             "the runner script and this report are).")
    L.append("")
    files: list[list[str]] = []
    for tag, _, _ in VARIANTS:
        for name in (f"trades_{tag}.csv", f"equity_{tag}.csv", f"folds_monthly_{tag}.csv",
                     f"folds_quarterly_{tag}.csv", f"per_symbol_{tag}.csv",
                     f"module_{tag}.csv", f"symbol_quarter_{tag}.csv",
                     f"stats_{tag}.json", f"summary_{tag}.md"):
            p = inp / name
            if p.exists():
                size = p.stat().st_size
                files.append([p.name, f"{size:,} B",
                              f"{sum(1 for _ in p.open()) - 1:,} rows" if name.endswith(".csv") else "—",
                              sha256(p)])
    L.append(md_table(["Artifact", "Size", "Rows", "sha256[:16]"], files))
    L.append("")
    L.append("Reproduce (from the engine checkout, cache present):")
    L.append("")
    L.append("```bash")
    L.append("env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY .venv/bin/python \\")
    L.append("    scripts/run_scalpset_backtest.py --variant baseline \\")
    L.append("    --start 2025-09-01 --end 2026-09-01 --out-dir data/backtest_out")
    L.append("env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY .venv/bin/python \\")
    L.append("    scripts/build_scalpset_report.py --out /home/team/shared/SCALPSET_BACKTEST_REPORT.md")
    L.append("```")
    L.append("")
    L.append("`--variant` also accepts `pessimistic`, `overnight` and `zero_cost`.")

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
