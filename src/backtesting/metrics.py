"""Performance metrics for backtest results.

Computes standard trading-performance statistics from an equity curve
and a trades DataFrame.  Shares implementations with
``src/optimization/objectives.py`` where possible so the optimiser uses
the same math as the backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization.objectives import max_drawdown as _max_dd_obj
from src.optimization.objectives import sharpe_ratio as _sharpe_obj
from src.optimization.objectives import sortino_ratio as _sortino_obj


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio computed from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time.
    risk_free_rate : float
        Annual risk-free rate (default 0 for intra-day).
    """
    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return _sharpe_obj(returns, risk_free_rate)


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio computed from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    return _sortino_obj(returns, risk_free_rate)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.15 = 15 %)."""
    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    # _max_dd_obj returns the negative of max dd (higher-is-better convention);
    # we want the raw positive fraction here.
    return float(abs(_max_dd_obj(returns)))


def total_return(equity_curve: pd.Series) -> float:
    """Total return as a fraction of initial equity (e.g. 0.05 = 5 %)."""
    if len(equity_curve) < 2:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades with positive PnL.  Returns 0.0 when there are no trades."""
    if trades.empty:
        return 0.0
    winners = (trades["pnl"] > 0).sum()
    return float(winners / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    """Ratio of gross gains to gross losses.  Returns ∞ (inf) when there are
    no losing trades and 0.0 when there are no trades at all."""
    if trades.empty:
        return 0.0
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def num_trades(trades: pd.DataFrame) -> int:
    """Total number of completed round-trip trades."""
    return len(trades)


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute a standard set of performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time (indexed by timestamp).
    trades : pd.DataFrame
        Completed trades with columns ``[entry_time, exit_time, symbol, side,
        entry_price, exit_price, quantity, pnl, pnl_pct, exit_reason]``.
    risk_free_rate : float
        Annualised risk-free rate (default 0 for intra-day backtests).

    Returns
    -------
    dict
        Keys: ``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown``,
        ``total_return``, ``win_rate``, ``profit_factor``, ``num_trades``.
    """
    return {
        "sharpe_ratio": sharpe_ratio(equity_curve, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity_curve, risk_free_rate),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return": total_return(equity_curve),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "num_trades": num_trades(trades),
    }
