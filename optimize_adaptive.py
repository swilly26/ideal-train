#!/usr/bin/env python3
"""
AlgoFlow autonomous parameter optimizer — CLI entry point.

Discovers robust, market-adaptive strategy parameters from **real** data:

* **Realized fills** — the deduplicated per-trade P&L CSVs the live traders
  logged (``per_trade_pnl_MAIN.csv`` / ``per_trade_pnl_TURBO.csv``), and
* **Historical price cache** — real intraday OHLCV fetched via the engine's
  Yahoo Finance provider for the exact trading window and cached to
  ``/home/team/shared/data/adaptive_ohlcv/`` (so it can be re-run offline).

The pipeline is walk-forward with an untouched holdout, S/A/B/C/F
degradation grading, and regime-adaptive (2×2 trend × volatility) output.

Results are written under ``configs/adaptive/``:

    report_<trader>.md / results_<trader>.json   — evidence & rankings
    <trader>_mean_reversion.json                 — best validated config
    regime/<trader>_<regime>.json                — per-regime configs
    APPLY.md                                     — how to adopt (do not wire
                                                  live without review)

Usage:
    python optimize_adaptive.py --trader main
    python optimize_adaptive.py --trader turbo
    python optimize_adaptive.py --trader all --no-fetch   # re-run offline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.data.cache import MarketDataCache
from src.data.yfinance_provider import YFinanceProvider
from src.optimization.adaptive import AdaptiveOptimizer, config_from_kwargs
from src.optimization.degradation import DEFAULT_THRESHOLDS
from src.optimization.realized import (
    load_realized_trades,
    realized_metrics,
    summarize_realized_window,
)
from src.optimization.walk_forward import split_is_oos_holdout
from src.strategies.base import StrategyConfig

logger = logging.getLogger("optimize_adaptive")

# Realized trade CSVs (the monthly analysis deliverables).
SHARED = Path("/home/team/shared")
MAIN_REALIZED = SHARED / "per_trade_pnl_MAIN.csv"
TURBO_REALIZED = SHARED / "per_trade_pnl_TURBO.csv"

# Intraday price snapshot (kept outside the repo so git stays clean).
OHLCV_DIR = SHARED / "data" / "adaptive_ohlcv"
OHLCV_MANIFEST = OHLCV_DIR / "manifest.json"

OUT_DIR = REPO / "configs" / "adaptive"

TIMEFRAME = "5min"
FETCH_START = datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc)
FETCH_END = datetime(2026, 8, 30, 0, 0, tzinfo=timezone.utc)

# Symbols that actually produced realized fills (union across traders).
ALL_SYMBOLS = [
    "NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO",
    "AAPL", "MSFT", "SPY", "SOXL", "TQQQ", "FNGU", "SPXL", "TZA", "TNA", "LABU",
]

# Per-trader search scope.
TRADER_SYMBOLS = {
    "main": ["NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO"],
    "turbo": ["SOXL", "TQQQ", "FNGU", "SPXL", "TZA", "TNA"],
}

# Grids: breadth not depth — a few hundred combos.
MAIN_GRID = {
    "entry_threshold": [0.4, 0.5, 0.6, 0.8],
    "stop_loss_pct": [0.02, 0.03, 0.05, 0.08],
    "take_profit_pct": [0.02, 0.04, 0.06, 0.10],
    "lookback": [10, 20, 30],
}
TURBO_GRID = {
    "entry_threshold": [0.4, 0.5, 0.7],
    "stop_loss_pct": [0.04, 0.06, 0.09],
    "take_profit_pct": [0.06, 0.08, 0.12],
    "lookback": [10, 20],
}


def load_ohlcv() -> dict[str, pd.DataFrame]:
    """Load the intraday OHLCV snapshot from disk.

    Returns dict symbol → OHLCV DataFrame.  Fails loudly if missing so the
    user runs with fetch enabled first.
    """
    if not OHLCV_DIR.exists():
        raise FileNotFoundError(
            f"No OHLCV snapshot at {OHLCV_DIR}. Run with fetch enabled "
            "(default) or check the manifest."
        )
    frames: dict[str, pd.DataFrame] = {}
    for sym in ALL_SYMBOLS:
        p = OHLCV_DIR / f"{sym}.parquet"
        if p.exists():
            frames[sym] = pd.read_parquet(p)
    if not frames:
        raise FileNotFoundError(f"No symbol parquet files under {OHLCV_DIR}")
    return frames


async def fetch_ohlcv() -> dict[str, pd.DataFrame]:
    """Fetch + cache intraday OHLCV for every traded symbol (real data)."""
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    provider = YFinanceProvider()
    frames: dict[str, pd.DataFrame] = {}
    manifest: dict = {
        "timeframe": TIMEFRAME,
        "start": FETCH_START.isoformat(),
        "end": FETCH_END.isoformat(),
        "symbols": {},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    for i, sym in enumerate(ALL_SYMBOLS, 1):
        out = OHLCV_DIR / f"{sym}.parquet"
        if out.exists():
            frames[sym] = pd.read_parquet(out)
            logger.info("[%d/%d] %s: using cached %d bars", i, len(ALL_SYMBOLS), sym, len(frames[sym]))
            manifest["symbols"][sym] = {"bars": int(len(frames[sym])), "cached": True}
            continue
        try:
            mdf = await provider.fetch_bars(sym, FETCH_START, FETCH_END, TIMEFRAME)
            df = mdf.df
            df.to_parquet(out)
            frames[sym] = df
            manifest["symbols"][sym] = {"bars": int(len(df)), "cached": False}
            logger.info("[%d/%d] %s: fetched %d bars", i, len(ALL_SYMBOLS), sym, len(df))
        except Exception as exc:  # keep going — some symbols may be delisted
            logger.warning("[%d/%d] %s: fetch failed: %s", i, len(ALL_SYMBOLS), sym, exc)
            manifest["symbols"][sym] = {"bars": 0, "error": str(exc)}
    OHLCV_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return frames


def build_split(frames: dict[str, pd.DataFrame], is_frac: float, oos_frac: float):
    """Chronological IS/OOS/holdout split over the union of price-bar days."""
    all_days = sorted(
        {
            ts.normalize()
            for df in frames.values()
            for ts in df.index
        }
    )
    if not all_days:
        raise ValueError("No price data — cannot build a split")
    # Reindex a synthetic day series to feed the splitter.
    daily = pd.DatetimeIndex(all_days)
    # split_is_oos_holdout works on any dated index — a one-bar-per-day frame.
    index = daily
    split = split_is_oos_holdout(index, is_frac=is_frac, oos_frac=oos_frac)
    return split


def run_trader(
    trader: str,
    realized_path: Path,
    frames: dict[str, pd.DataFrame],
    split,
    grid: dict[str, list[float]],
    base_config: StrategyConfig,
) -> dict:
    """Run the adaptive optimizer for one trader and return its report data."""
    realized = load_realized_trades(realized_path)
    realized_syms = set(realized["symbol"].str.upper())
    symbols = [s for s in TRADER_SYMBOLS[trader] if s in frames]

    from src.strategies.mean_reversion import MeanReversionStrategy

    optimizer = AdaptiveOptimizer(
        min_trades=8, top_k=6, grade_thresholds=DEFAULT_THRESHOLDS,
    )
    result = optimizer.optimize(
        MeanReversionStrategy,
        {s: frames[s] for s in symbols},
        split,
        grid,
        n_iter=400,
        base_config=base_config,
        name=f"{trader}_mean_reversion",
    )

    baseline_realized = summarize_realized_window(
        realized, split.is_days, split.oos_days, split.holdout_days,
    )

    return {
        "trader": trader,
        "strategy": "mean_reversion",
        "split": result.split.summary(),
        "split_days": {
            "is": result.split.is_days,
            "oos": result.split.oos_days,
            "holdout": result.split.holdout_days,
        },
        "baseline_realized": baseline_realized,
        "baseline_price_backtest": result.baseline,
        "rankings": result.rankings.to_dict(orient="records"),
        "best_config_kwargs": result.best_config_kwargs,
        "best_grade": result.best_grade,
        "regime_summary": (
            result.regime_summary.to_dict(orient="records")
            if not result.regime_summary.empty else []
        ),
        "regime_configs": result.regime_configs,
        "suggestions": result.suggestions,
        "algorithm": {
            "walk_forward": "IS/OOS/holdout over unique trading days",
            "grading": "S/A/B/C/F by absolute OOS PF + IS→OOS degradation",
            "min_oos_trades": DEFAULT_THRESHOLDS.min_oos_trades,
            "note": "Holdout window never used for search or selection.",
        },
    }


def write_reports(data: dict) -> Path:
    """Write markdown + JSON reports and the trader-consumable configs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trader = data["trader"]
    results_json = OUT_DIR / f"results_{trader}.json"
    results_json.write_text(json.dumps(data, indent=2, default=str))
    md = _render_markdown(data)
    md_path = OUT_DIR / f"report_{trader}.md"
    md_path.write_text(md)

    # Best config → trader-consumable JSON (persistence.save_config format).
    kwargs = data["best_config_kwargs"]
    best_json = OUT_DIR / f"{trader}_mean_reversion.json"
    if kwargs:
        cfg = config_from_kwargs(kwargs, "GENERIC")
        from src.optimization.persistence import save_config
        save_config(cfg, best_json)
    else:
        best_json.write_text(json.dumps({"grade": "F", "note": "no robust candidate"}))

    # Per-regime configs.
    regime_dir = OUT_DIR / "regime"
    regime_dir.mkdir(exist_ok=True)
    for regime, cfg_data in data["regime_configs"].items():
        (regime_dir / f"{trader}_{regime}.json").write_text(json.dumps(cfg_data, indent=2))
    return md_path


def _render_markdown(data: dict) -> str:
    trader = data["trader"].upper()
    lines: list[str] = []
    lines.append(f"# AlgoFlow Adaptive Parameter Optimization — {trader} (mean reversion)")
    lines.append("")
    split = data["split"]
    lines.append("## Data & split")
    lines.append("")
    lines.append(f"- Realized fills: {data['baseline_realized']['is']['days'][:1] if data['baseline_realized']['is']['days'] else 'n/a'} … ({split['is_days']} IS days, {split['oos_days']} OOS days, {split['holdout_days']} holdout days)")
    lines.append(f"- Split: IS={split['is']} positions, OOS={split['oos']}, holdout={split['holdout']} (unique trading days).")
    lines.append("")
    lines.append("## Baseline (current fixed parameters) — realized P&L by window")
    lines.append("")
    br = data["baseline_realized"]
    lines.append("| window | days | trades | PF | win_rate | net_pnl |")
    lines.append("|---|---|---|---|---|---|")
    for w in ("is", "oos", "holdout"):
        m = br[w]
        lines.append(
            f"| {w} | {len(m['days'])} | {m['n_trades']} | {m['profit_factor']:.2f} | "
            f"{m['win_rate']:.2f} | {m['net_pnl']:+.0f} |"
        )
    lines.append("")
    lines.append("## Search result (graded IS vs OOS, degradation)")
    lines.append("")
    lines.append("| grade | config | IS PF | IS n | OOS PF | OOS n | degrad. | holdout PF | holdout n |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in data["rankings"]:
        lines.append(
            f"| {r['grade']} | {r['config']} | {r['is_score']:.2f} | {r['is_n_trades']} | "
            f"{r['oos_score']:.2f} | {r['oos_n_trades']} | {r['degradation']:.0%} | "
            f"{r['holdout_score']:.2f} | {r['holdout_n_trades']} |"
        )
    lines.append("")
    lines.append(f"**Best graded:** `{data['best_config_kwargs']}` → grade **{data['best_grade']}**")
    lines.append("")
    lines.append("## Regime breakdown (winner, IS+OOS trades)")
    lines.append("")
    if data["regime_summary"]:
        lines.append("| regime | n | PF | win_rate | avg_pct | net_pnl |")
        lines.append("|---|---|---|---|---|---|")
        for r in data["regime_summary"]:
            lines.append(
                f"| {r['regime']} | {r['n_trades']} | {r['profit_factor']:.2f} | "
                f"{r['win_rate']:.2f} | {r['avg_pct']:.3f} | {r['net_pnl']:+.0f} |"
            )
    else:
        lines.append("_No regime-labeled trades._")
    lines.append("")
    lines.append("## Adaptive rules (experimental until fresh-window validation)")
    lines.append("")
    lines.append(f"```\n{data['suggestions']}\n```")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `optimize_adaptive.py`. Full numeric output: "
                 f"`results_{data['trader']}.json`*")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="AlgoFlow adaptive parameter optimizer")
    ap.add_argument("--trader", choices=["main", "turbo", "all"], default="all")
    ap.add_argument("--no-fetch", action="store_true", help="Use the on-disk OHLCV snapshot only")
    ap.add_argument("--is-frac", type=float, default=0.50)
    ap.add_argument("--oos-frac", type=float, default=0.25)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.no_fetch:
        frames = load_ohlcv()
    else:
        frames = asyncio.run(fetch_ohlcv())

    split = build_split(frames, args.is_frac, args.oos_frac)
    logger.info("Split: IS=%d days (…%s), OOS=%d (…%s), holdout=%d (…%s)",
                len(split.is_days), split.is_days[-1],
                len(split.oos_days), split.oos_days[-1],
                len(split.holdout_days), split.holdout_days[-1] if split.holdout_days else "—")

    traders = ["main", "turbo"] if args.trader == "all" else [args.trader]
    for trader in traders:
        realized_path = MAIN_REALIZED if trader == "main" else TURBO_REALIZED
        grid = MAIN_GRID if trader == "main" else TURBO_GRID
        base = StrategyConfig(
            entry_threshold=0.5,
            exit_threshold=0.1 if trader == "main" else 0.05,
            stop_loss_pct=0.03 if trader == "main" else 0.06,
            take_profit_pct=0.03 if trader == "main" else 0.08,
            max_position_pct=0.15 if trader == "main" else 0.50,
            extra={"lookback": 20, "std_dev_multiplier": 2.0},
        )
        logger.info("=== Running %s ===", trader)
        data = run_trader(trader, realized_path, frames, split, grid, base)
        path = write_reports(data)
        logger.info("Report written: %s", path)
    logger.info("Apply path doc: %s", OUT_DIR / "APPLY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())