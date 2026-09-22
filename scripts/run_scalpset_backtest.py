#!/usr/bin/env python3
"""Full-window ScalpSet backtest runner + audit artifacts.

Runs the *portfolio* replay of the live ScalpSet configuration (all symbols in
one engine so 15 % sizing, the max-positions cap and best-R:R arbitration
behave exactly as live) over the cached 1m history, and writes the audit
artifacts the owner report is built from:

    <out-dir>/trades_<variant>.csv        one row per round trip
    <out-dir>/equity_<variant>.csv        portfolio mark-to-market per 1m bar
    <out-dir>/stats_<variant>.json        engine counters + config + metrics
    <out-dir>/per_symbol_<variant>.csv    P&L / win rate / PF / DD / expectancy
    <out-dir>/folds_monthly_<variant>.csv per-calendar-month slice metrics
    <out-dir>/folds_quarterly_<variant>.csv
    <out-dir>/module_<variant>.csv        per-module trades / fills / P&L
    <out-dir>/symbol_quarter_<variant>.csv symbol x quarter net P&L
    <out-dir>/summary_<variant>.md        human-readable dump of the above

Fold tables are *slices of the single portfolio run* (trades bucketed by exit
time, equity curve sliced by timestamp), not independent per-fold replays.
With ``eod_flat=True`` no position crosses a fold boundary, so a monthly slice
is the same trade set an independent monthly replay would produce; the only
difference is that position sizing compounds across folds exactly as it does
in the aggregated run.  Use ``--independent-folds`` to additionally verify one
fold set with true standalone replays.

Usage (from the engine repo root)::

    env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY .venv/bin/python \
        scripts/run_scalpset_backtest.py --variant baseline \
        --start 2025-09-01 --end 2026-09-01 --out-dir data/backtest_out

Variants: ``baseline`` (engine defaults: eod_flat, 2bps+1c slippage, no fees),
``pessimistic`` (``ScalpSetConfig.cost_variant()``: + 1bp half-spread,
$0.005/share), ``overnight`` (pessimistic costs, ``eod_flat=False``),
``zero_cost`` (no slippage / no fees — gross-edge diagnostic).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.scalpset_engine import (  # noqa: E402
    MODULE_ORDER,
    ScalpSetConfig,
    replay,
)

DEFAULT_SYMBOLS = ("NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO")


# ── variants ───────────────────────────────────────────────────────────
def build_config(variant: str) -> ScalpSetConfig:
    """Live params with the requested cost/overnight treatment."""
    if variant == "baseline":
        return ScalpSetConfig()
    if variant == "pessimistic":
        return ScalpSetConfig.cost_variant()
    if variant == "overnight":
        return ScalpSetConfig.cost_variant(eod_flat=False)
    if variant == "zero_cost":
        return replace(
            ScalpSetConfig(),
            slippage_pct=0.0,
            slippage_abs=0.0,
            half_spread_pct=0.0,
            commission_per_share=0.0,
            commission_pct=0.0,
        )
    raise SystemExit(f"unknown variant: {variant}")


# ── metric helpers ─────────────────────────────────────────────────────
def _profit_factor(pnl: pd.Series) -> float:
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _max_dd(values: pd.Series) -> float:
    """Max peak-to-trough drawdown of a value series (positive fraction)."""
    if len(values) < 2:
        return 0.0
    running = values.cummax()
    dd = (values - running) / running.replace(0, np.nan)
    return float(abs(dd.min())) if dd.notna().any() else 0.0


def trade_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    """Standard set of trade-level statistics for any trade subset."""
    n = len(trades)
    if n == 0:
        return {
            "label": label, "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "pnl_net": 0.0, "pnl_gross": 0.0, "fees": 0.0, "cost_drag": 0.0,
            "pnl_per_trade": 0.0, "profit_factor": 0.0, "gross_profit": 0.0,
            "gross_loss": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "best": 0.0, "worst": 0.0, "avg_bars_held": 0.0,
        }
    net = trades["pnl_after_costs"]
    gross = trades["pnl_gross"]
    wins = trades[trades["pnl_after_costs"] > 0]
    losses = trades[trades["pnl_after_costs"] <= 0]
    return {
        "label": label,
        "trades": int(n),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / n),
        "pnl_net": float(net.sum()),
        "pnl_gross": float(gross.sum()),
        "fees": float(trades["fees"].sum()),
        "cost_drag": float((gross - net).sum()),
        "pnl_per_trade": float(net.mean()),
        "profit_factor": _profit_factor(net),
        "gross_profit": float(net[net > 0].sum()),
        "gross_loss": float(abs(net[net < 0].sum())),
        "avg_win": float(net[net > 0].mean()) if len(wins) else 0.0,
        "avg_loss": float(net[net <= 0].mean()) if len(losses) else 0.0,
        "best": float(net.max()),
        "worst": float(net.min()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "notional_traded": float((trades["entry_price"] * trades["quantity"]).sum()),
    }


def per_symbol_table(trades: pd.DataFrame, equity: pd.Series,
                     initial_equity: float) -> pd.DataFrame:
    """Per-symbol P&L / win rate / PF / DD / expectancy.

    ``max_dd`` is the drawdown of that symbol's own cumulative net-P&L curve
    (starting at 0), scaled by the account's initial equity — the portfolio
    equity curve cannot be attributed to one symbol, so this is the honest
    per-symbol proxy and is labelled as such in the report.
    """
    rows = []
    for sym in sorted(set(trades["symbol"])) if len(trades) else []:
        t = trades[trades["symbol"] == sym].sort_values("exit_time")
        m = trade_metrics(t, label=sym)
        curve = t.set_index("exit_time")["pnl_after_costs"].cumsum()
        m["max_dd_usd"] = (_max_dd(curve + initial_equity) * initial_equity
                           if len(curve) > 1 else 0.0)
        m["max_dd_pct_of_account"] = m["max_dd_usd"] / initial_equity
        m["notional_traded"] = float((t["entry_price"] * t["quantity"]).sum())
        rows.append(m)
    cols = ["label", "trades", "wins", "losses", "win_rate", "pnl_net",
            "pnl_gross", "cost_drag", "pnl_per_trade", "profit_factor",
            "max_dd_usd", "max_dd_pct_of_account", "avg_win", "avg_loss",
            "avg_bars_held", "notional_traded"]
    return pd.DataFrame(rows, columns=cols)


def fold_table(trades: pd.DataFrame, equity: pd.Series, freq: str,
               initial_equity: float) -> pd.DataFrame:
    """Per-calendar-period slice metrics from the single portfolio run."""
    idx = pd.DatetimeIndex(equity.index)
    periods = idx.to_period(freq)
    exits = pd.PeriodIndex(pd.DatetimeIndex(trades["exit_time"]), freq=freq) \
        if len(trades) else pd.PeriodIndex([], freq=freq)
    rows = []
    prev_end = initial_equity
    for p in periods.unique():
        eq = equity[periods == p]
        t = trades[exits == p] if len(trades) else trades
        m = trade_metrics(t, label=str(p))
        m["period"] = str(p)
        m["session_bars"] = int(len(eq))
        m["equity_start"] = float(prev_end)
        m["equity_end"] = float(eq.iloc[-1])
        m["return_pct"] = float(eq.iloc[-1] / prev_end - 1.0) if prev_end else 0.0
        m["period_max_dd_pct"] = _max_dd(eq)
        prev_end = float(eq.iloc[-1])
        rows.append(m)
    cols = ["period", "trades", "wins", "losses", "win_rate", "pnl_net",
            "pnl_gross", "cost_drag", "pnl_per_trade", "profit_factor",
            "equity_start", "equity_end", "return_pct", "period_max_dd_pct",
            "avg_bars_held", "session_bars"]
    return pd.DataFrame(rows, columns=cols)


def module_table(trades: pd.DataFrame, stats: dict,
                 config: ScalpSetConfig) -> pd.DataFrame:
    """Per-module fills, entry orders, limit-order bookkeeping and P&L."""
    sig = stats.get("signals_by_module", {})
    ent = stats.get("entries_by_module", {})
    rows = []
    for mod in config.module_order:
        t = trades[trades["module"] == mod] if len(trades) else trades
        m = trade_metrics(t, label=mod)
        m["signals"] = int(sig.get(mod, 0))
        m["entry_orders"] = int(ent.get(mod, 0))
        m["limit_entry_fills"] = int((t["entry_type"] == "limit").sum()) if len(t) else 0
        m["market_entry_fills"] = int((t["entry_type"] == "market").sum()) if len(t) else 0
        m["long_fills"] = int((t["side"] == "LONG").sum()) if len(t) else 0
        m["short_fills"] = int((t["side"] == "SHORT").sum()) if len(t) else 0
        for reason in ("tp", "sl", "eod", "be", "trail"):
            m[f"exits_{reason}"] = int((t["exit_reason"] == reason).sum()) if len(t) else 0
        rows.append(m)
    cols = ["label", "signals", "entry_orders", "market_entry_fills",
            "limit_entry_fills", "trades", "long_fills", "short_fills",
            "win_rate", "pnl_net", "pnl_gross", "cost_drag", "pnl_per_trade",
            "profit_factor", "avg_bars_held", "exits_tp", "exits_sl",
            "exits_eod", "exits_be", "exits_trail"]
    return pd.DataFrame(rows, columns=cols)


def portfolio_metrics(equity: pd.Series, trades: pd.DataFrame,
                      initial_equity: float) -> dict:
    """Equity-curve level performance for one full run."""
    ret = equity.pct_change().dropna()
    day_eq = equity.groupby(pd.DatetimeIndex(equity.index).date).last()
    day_ret = day_eq.pct_change().dropna()
    def _sharpe(r: pd.Series, per_year: float) -> float:
        if len(r) < 2 or float(r.std()) == 0.0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(per_year))
    m = trade_metrics(trades, label="portfolio")
    m.update({
        "initial_equity": initial_equity,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_equity,
        "total_return": float(equity.iloc[-1] / initial_equity - 1.0) if len(equity) else 0.0,
        "max_drawdown": _max_dd(equity),
        "sharpe_bar": _sharpe(ret, 252 * 390),
        "sharpe_daily": _sharpe(day_ret, 252),
        "trading_days": int(len(day_eq)),
        "profitable_days": int((day_ret > 0).sum()),
        "losing_days": int((day_ret < 0).sum()),
        "flat_days": int((day_ret == 0).sum()),
        "worst_day": float(day_ret.min()) if len(day_ret) else 0.0,
        "best_day": float(day_ret.max()) if len(day_ret) else 0.0,
    })
    return m


# ── reporting ──────────────────────────────────────────────────────────
def _fmt(v, nd=2):
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:,.{nd}f}"
    return str(v)


def _md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".2f") -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    if df is None or len(df) == 0:
        return "_(no rows)_"
    cols = list(df.columns)
    header = ([""] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    idx_vals = list(df.index) if index else [None] * len(df)
    for pos, (_, row) in enumerate(df.iterrows()):
        cells = [str(idx_vals[pos])] if index else []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("inf" if v == float("inf")
                             else f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_summary(variant: str, cfg: ScalpSetConfig, stats: dict,
                     pm: dict, per_sym: pd.DataFrame, monthly: pd.DataFrame,
                     quarterly: pd.DataFrame, module: pd.DataFrame,
                     sym_q: pd.DataFrame, elapsed: float) -> str:
    L = [f"## Variant `{variant}`", ""]
    L.append(f"- config: eod_flat={cfg.eod_flat} slippage={cfg.slippage_pct:.5f}+${cfg.slippage_abs}"
             f" half_spread={cfg.half_spread_pct:.5f} commission/share=${cfg.commission_per_share}"
             f" position_size={cfg.position_size_pct} max_pos={cfg.max_positions}"
             f" shortable={sorted(cfg.shortable)}")
    L.append(f"- bars={stats['bars']} sessions={stats['sessions']} elapsed={elapsed:.1f}s")
    L.append(f"- entries={stats['entries']} (market={stats['market_entries']}, "
             f"limits placed={stats['limit_orders_placed']}, filled={stats['limit_orders_filled']}, "
             f"expired={stats['limit_orders_expired']})")
    L.append(f"- exits={stats['exits']}")
    L.append(f"- skipped={stats['skipped']}")
    L.append(f"- signals_by_module={stats['signals_by_module']}")
    L.append(f"- entries_by_module={stats['entries_by_module']} by_side={stats['entries_by_side']}")
    L.append(f"- fees_paid=${stats['fees_paid']:,.2f}")
    L.append("")
    L.append("### Portfolio")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    for k in ("trades", "win_rate", "pnl_net", "pnl_gross", "cost_drag", "fees",
              "pnl_per_trade", "profit_factor", "avg_win", "avg_loss",
              "initial_equity", "final_equity", "total_return", "max_drawdown",
              "sharpe_daily", "sharpe_bar", "trading_days", "profitable_days",
              "losing_days", "flat_days", "best_day", "worst_day"):
        L.append(f"| {k} | {_fmt(pm[k], 4 if 'return' in k or 'drawdown' in k or k.endswith('day') else 2)} |")
    L.append("")
    for name, df in (("Per symbol", per_sym), ("Per module", module),
                     ("Monthly folds", monthly), ("Quarterly folds", quarterly)):
        L.append(f"### {name}")
        L.append("")
        L.append(_md_table(df))
        L.append("")
    L.append("### Symbol x quarter net P&L")
    L.append("")
    L.append(_md_table(sym_q, index=True))
    L.append("")
    return "\n".join(L)


# ── main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--warmup-days", type=int, default=75)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "backtest_out"))
    ap.add_argument("--tag", default=None, help="output tag (default = variant)")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = build_config(args.variant)
    out = Path(args.out_dir)
    tag = args.tag or args.variant
    if not args.no_write:
        out.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] portfolio replay {symbols} {args.start}..{args.end} "
          f"(warmup {args.warmup_days}d)", flush=True)
    t0 = time.time()
    res = replay(symbols, start=args.start, end=args.end,
                 warmup_days=args.warmup_days, config=cfg)
    elapsed = time.time() - t0
    trades, equity, stats = res.trades, res.equity_curve, dict(res.stats)
    print(f"[{tag}] done in {elapsed:.1f}s bars={stats['bars']} "
          f"sessions={stats['sessions']} trades={len(trades)}", flush=True)

    pm = portfolio_metrics(equity, trades, cfg.initial_equity)
    per_sym = per_symbol_table(trades, equity, cfg.initial_equity)
    monthly = fold_table(trades, equity, "M", cfg.initial_equity)
    quarterly = fold_table(trades, equity, "Q", cfg.initial_equity)
    module = module_table(trades, stats, cfg)
    if len(trades):
        sq = (trades.assign(quarter=pd.PeriodIndex(pd.DatetimeIndex(trades["exit_time"]), freq="Q").astype(str))
              .pivot_table(index="symbol", columns="quarter", values="pnl_after_costs",
                           aggfunc="sum", fill_value=0.0))
        cnt = (trades.assign(quarter=pd.PeriodIndex(pd.DatetimeIndex(trades["exit_time"]), freq="Q").astype(str))
               .pivot_table(index="symbol", columns="quarter", values="pnl_after_costs",
                            aggfunc="count", fill_value=0))
        sq.columns = [f"{c}_pnl" for c in sq.columns]
        cnt.columns = [f"{c}_n" for c in cnt.columns]
        sym_q = sq.join(cnt)
        sym_q["YEAR_pnl"] = sq.sum(axis=1)
        sym_q["YEAR_n"] = cnt.sum(axis=1)
    else:
        sym_q = pd.DataFrame()

    md = markdown_summary(tag, cfg, stats, pm, per_sym, monthly, quarterly,
                          module, sym_q, elapsed)

    if args.no_write:
        print(md)
        return 0

    # artifacts first: a failure in the reporting text must not lose the run
    trades.to_csv(out / f"trades_{tag}.csv", index=False)
    equity.to_frame("equity").to_csv(out / f"equity_{tag}.csv")
    per_sym.to_csv(out / f"per_symbol_{tag}.csv", index=False)
    monthly.to_csv(out / f"folds_monthly_{tag}.csv", index=False)
    quarterly.to_csv(out / f"folds_quarterly_{tag}.csv", index=False)
    module.to_csv(out / f"module_{tag}.csv", index=False)
    sym_q.to_csv(out / f"symbol_quarter_{tag}.csv")
    (out / f"summary_{tag}.md").write_text(md)
    blob = {
        "variant": tag,
        "command": " ".join(sys.argv),
        "start": args.start, "end": args.end, "symbols": symbols,
        "warmup_days": args.warmup_days,
        "elapsed_sec": elapsed,
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v)
                   for k, v in cfg.__dict__.items()},
        "stats": {k: (dict(v) if isinstance(v, dict) else v)
                  for k, v in stats.items()},
        "portfolio_metrics": pm,
        "per_symbol": per_sym.to_dict(orient="records"),
        "module": module.to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "quarterly": quarterly.to_dict(orient="records"),
    }
    (out / f"stats_{tag}.json").write_text(json.dumps(blob, indent=2, default=str))
    print(md)
    print(f"[{tag}] artifacts written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
