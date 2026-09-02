"""Realized trade history — loading and scoring actual fills.

The optimizer's substrate is what actually happened: the per-trade P&L the
live traders logged.  This module loads the deduplicated per-trade CSVs
(``per_trade_pnl_MAIN.csv`` / ``per_trade_pnl_TURBO.csv`` produced by the
monthly analysis), normalizes them, and provides the profit-factor-based
objective used throughout the walk-forward search.

Profit factor (gross wins / gross losses) is preferred over plain total
return because it is far less sensitive to position sizing and to the
leveraged-ETF tail days that dominated the raw P&L — it measures the *edge
per trade*, which is what a parameter set should be optimising.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REALIZED_COLUMNS = [
    "trader", "date", "time", "symbol", "entry", "exit", "pnl", "pct",
    "reason", "strategy",
]

# Cap applied to an infinite profit factor (no losing trades) so scores stay
# comparable and finite.
PF_CAP = 3.0


def load_realized_trades(path: str | Path) -> pd.DataFrame:
    """Load a per-trade realized P&L CSV into a normalized DataFrame.

    Expected schema (matching ``per_trade_pnl_*.csv``):
    ``trader,date,time,symbol,entry,exit,pnl,pct,reason,strategy``

    Adds an ``exit_dt`` column (datetime) combining *date* and *time* (both
    treated as local market time) and coerces numeric columns.  Rows that
    cannot be parsed are dropped with a warning.

    Parameters
    ----------
    path : str | Path
        Path to the realized-trades CSV.

    Returns
    -------
    pd.DataFrame
        Normalized trades, sorted by ``exit_dt``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Realized trades file not found: {path}")

    df = pd.read_csv(path)
    missing = set(REALIZED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    for col in ("entry", "exit", "pnl", "pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dt = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce"
    )
    df["exit_dt"] = dt
    df = df.dropna(subset=["exit_dt", "pnl"]).copy()
    df = df.sort_values("exit_dt").reset_index(drop=True)
    logger.info(
        "Loaded %d realized trades from %s (window %s → %s)",
        len(df), path.name, df["exit_dt"].min(), df["exit_dt"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Trade metrics & objective
# ---------------------------------------------------------------------------

def profit_factor(trades: pd.DataFrame) -> float:
    """Gross winning P&L / gross losing P&L for *trades*.

    Returns ``inf`` when there are winners and no losers, and ``0.0`` when
    there are no trades.  Uses the ``pnl`` column.
    """
    if trades is None or trades.empty:
        return 0.0
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def realized_metrics(trades: pd.DataFrame) -> dict:
    """Standard per-trade metrics for a realized/backtested trade set.

    Returns
    -------
    dict with keys ``n_trades``, ``gross_win``, ``gross_loss``, ``profit_factor``,
    ``win_rate``, ``avg_pct`` (mean per-trade pct return), ``net_pnl``,
    ``expectancy_pct`` (mean per-trade % return, alias of ``avg_pct``).
    """
    if trades is None or trades.empty:
        return {
            "n_trades": 0, "gross_win": 0.0, "gross_loss": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "avg_pct": 0.0,
            "net_pnl": 0.0, "expectancy_pct": 0.0,
        }
    t = trades
    wins = t.loc[t["pnl"] > 0, "pnl"]
    losses = t.loc[t["pnl"] < 0, "pnl"]
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    pf = profit_factor(t)
    if not np.isfinite(pf):
        pf = float(PF_CAP)
    win_rate = float((t["pnl"] > 0).mean())
    pct_col = "pct" if "pct" in t.columns and t["pct"].notna().any() else "pnl_pct"
    if pct_col not in t.columns:
        t = t.assign(pct=0.0)
        pct_col = "pct"
    avg_pct = float(t[pct_col].astype(float).mean()) if len(t) else 0.0
    return {
        "n_trades": int(len(t)),
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(float(pf), 4),
        "win_rate": round(win_rate, 4),
        "avg_pct": round(avg_pct, 4),
        "net_pnl": round(float(t["pnl"].sum()), 2),
        "expectancy_pct": round(avg_pct, 4),
    }


def trade_objective(trades: pd.DataFrame, min_trades: int = 8) -> float:
    """Profit-factor based objective score for a trade set.

    The score **is** the profit factor when the trade set meets the minimum
    trade-count gate, and ``-inf`` otherwise.  Trade count is a hard gate,
    not a bonus: a parameter set that only fired twice is not evidence of
    edge, no matter how profitable those two trades were.  An infinite PF
    (no losers) is capped at :data:`PF_CAP` for comparability.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades with a ``pnl`` column.
    min_trades : int
        Minimum completed trades required for a finite score.

    Returns
    -------
    float
        The objective score (higher is better).
    """
    if trades is None or len(trades) < min_trades:
        return float("-inf")
    pf = profit_factor(trades)
    if not np.isfinite(pf):
        pf = float(PF_CAP)
    return float(pf)


def summarize_realized_window(
    trades: pd.DataFrame,
    is_days: list[str],
    oos_days: list[str],
    holdout_days: list[str],
) -> dict:
    """Summarize realized trades across the IS/OOS/holdout day windows.

    Used to report what the *already-deployed fixed parameters* actually did
    in each window — the baseline every candidate must beat out-of-sample.

    Returns
    -------
    dict with keys ``is``, ``oos``, ``holdout`` — each a :func:`realized_metrics`
    dict plus the day lists.
    """
    day_series = pd.Series(trades["exit_dt"]).dt.normalize().dt.date.astype(str)

    def _window_metrics(days: list[str]) -> dict:
        if not days:
            return {**realized_metrics(trades.iloc[0:0]), "days": days}
        mask = day_series.isin(set(days))
        subset = trades.loc[mask]
        m = realized_metrics(subset)
        m["days"] = days
        return m

    return {
        "is": _window_metrics(is_days),
        "oos": _window_metrics(oos_days),
        "holdout": _window_metrics(holdout_days),
    }