"""Event-driven backtesting engine.

Replays historical market data through one or more strategies and
produces a ``BacktestResult`` with trade log, equity curve, and
performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.strategies.base import Signal, Strategy
from src.execution.broker import Order, OrderResult, OrderSide


@dataclass
class BacktestResult:
    """Aggregate results from a single backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value over time, indexed by timestamp.
    trades : pd.DataFrame
        Log of every filled trade.
    metrics : dict
        Performance metrics (Sharpe, max drawdown, win rate, etc.).
    """

    equity_curve: pd.Series
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtester.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value.
    commission : float
        Per-trade commission as a fraction of notional (e.g. 0.001).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """Run *strategy* over *data* and return the result.

        Parameters
        ----------
        strategy : Strategy
            A configured strategy instance.
        data : pd.DataFrame
            OHLCV bars indexed by timestamp.

        Returns
        -------
        BacktestResult
        """
        signals = strategy.generate_signals(data)
        result = self._simulate(signals, data)
        return result

    def _simulate(
        self,
        signals: list[Signal],
        data: pd.DataFrame,
    ) -> BacktestResult:
        """Stub simulation — replaces in a future iteration."""
        # For now, produce a flat equity curve so the module is functional
        index = data.index if isinstance(data, pd.DataFrame) else pd.DatetimeIndex([])
        equity = pd.Series(self.initial_capital, index=index, dtype=float)
        return BacktestResult(
            equity_curve=equity,
            trades=pd.DataFrame(),
            metrics={"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0},
        )
