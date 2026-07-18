"""Built-in objective functions for the optimisation layer.

All functions accept a ``pd.Series`` of period returns and return a
scalar where **higher is better**.  The optimiser maximises this value.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Period returns (e.g. daily).
    risk_free_rate : float
        Annual risk-free rate (default 0 for intra-day).
    """
    if returns.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / 252
    return float(excess / returns.std() * np.sqrt(252))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    excess = returns.mean() - risk_free_rate / 252
    downside = returns[returns < 0].std()
    if downside == 0 or pd.isna(downside):
        return 0.0
    return float(excess / downside * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a positive fraction.

    Returns the *negative* of max drawdown so that higher is better
    (consistent with Sharpe / Sortino for the optimiser).
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(-dd.min())
