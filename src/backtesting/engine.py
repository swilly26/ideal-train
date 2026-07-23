"""Event-driven backtesting engine.

Replays historical market data through one or more strategies and
produces a ``BacktestResult`` with trade log, equity curve, and
performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.execution.broker import Order, OrderResult, OrderSide
from src.backtesting.metrics import compute_metrics


@dataclass
class BacktestResult:
    """Aggregate results from a single backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value over time, indexed by timestamp.
    trades : pd.DataFrame
        Log of every filled trade with columns:
        ``[entry_time, exit_time, symbol, side, entry_price, exit_price,
          quantity, pnl, pnl_pct, exit_reason]``.
    metrics : dict
        Performance metrics (Sharpe, max drawdown, win rate, etc.).
    """

    equity_curve: pd.Series
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtester.

    Simulates a single-symbol, single-position-at-a-time strategy over
    historical OHLCV bars.  The AI optimisation layer calls ``run()``
    repeatedly with different ``StrategyConfig`` instances to discover
    profitable parameter sets.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value.
    commission : float
        Per-trade commission as a fraction of notional (e.g. 0.001 = 0.1 %).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            OHLCV bars indexed by timestamp.  Required columns:
            ``open``, ``high``, ``low``, ``close``.

        Returns
        -------
        BacktestResult
        """
        signals = strategy.generate_signals(data)
        equity, trades = self._simulate(signals, data, strategy.config)
        metrics = compute_metrics(equity, trades)
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------

    @staticmethod
    def _build_signal_map(
        signals: list[Signal],
        data_index: pd.DatetimeIndex,
    ) -> dict[pd.Timestamp, SignalType]:
        """Build a timestamp → SignalType lookup for fast bar-by-bar access.

        If multiple signals share a timestamp the *first* one wins
        (strategies should not emit overlapping signals).
        """
        sig_map: dict[pd.Timestamp, SignalType] = {}
        for sig in signals:
            if sig.timestamp not in sig_map:
                sig_map[sig.timestamp] = sig.signal_type
        return sig_map

    def _simulate(
        self,
        signals: list[Signal],
        data: pd.DataFrame,
        config: StrategyConfig,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Bar-by-bar event-driven simulation.

        Returns
        -------
        equity_curve : pd.Series
            Portfolio value at each bar close.
        trades : pd.DataFrame
            Record of every completed round-trip trade.
        """
        if data.empty:
            equity = pd.Series(
                [self.initial_capital],
                index=pd.DatetimeIndex([pd.Timestamp("2026-01-01")], name="timestamp"),
                dtype=float,
            )
            return equity, pd.DataFrame()

        sig_map = self._build_signal_map(signals, data.index)

        # State
        cash: float = self.initial_capital
        position: float = 0.0  # shares held (0 when flat)
        entry_price: float = 0.0
        entry_bar_idx: int = -1
        symbol: str = str(config.extra.get("symbol", "SYM"))

        equity_values: list[float] = [self.initial_capital]
        trade_records: list[dict] = []

        n_bars = len(data)
        for i in range(n_bars):
            bar = data.iloc[i]
            ts = data.index[i]
            open_p = float(bar["open"])
            high_p = float(bar["high"])
            low_p = float(bar["low"])
            close_p = float(bar["close"])

            signal_type = sig_map.get(ts)  # None, BUY, or SELL

            # ----------------------------------------------------------
            # 1. If we're in a position, check stop-loss / take-profit
            # ----------------------------------------------------------
            if position > 0:
                sl_price = entry_price * (1.0 - config.stop_loss_pct)
                tp_price = entry_price * (1.0 + config.take_profit_pct)

                stop_hit = low_p <= sl_price
                take_hit = high_p >= tp_price

                exit_price: float | None = None
                exit_reason: str = ""

                if stop_hit and take_hit:
                    # Both triggered — stop-loss takes priority (conservative).
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif stop_hit:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif take_hit:
                    exit_price = tp_price
                    exit_reason = "take_profit"

                if exit_price is not None:
                    pnl = position * (exit_price - entry_price)
                    notional = position * exit_price
                    commission_cost = notional * self.commission
                    pnl -= commission_cost
                    cash += position * exit_price - commission_cost

                    trade_records.append({
                        "entry_time": data.index[entry_bar_idx],
                        "exit_time": ts,
                        "symbol": symbol,
                        "side": "BUY",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": position,
                        "pnl": pnl,
                        "pnl_pct": (exit_price / entry_price - 1.0),
                        "exit_reason": exit_reason,
                    })

                    position = 0.0
                    entry_price = 0.0
                    entry_bar_idx = -1

            # ----------------------------------------------------------
            # 2. If flat, check for entry signals
            # ----------------------------------------------------------
            if position == 0 and signal_type == SignalType.BUY:
                equity = cash  # current equity (no position)
                max_position_value = equity * config.max_position_pct
                quantity = max_position_value / close_p
                if quantity > 0:
                    notional = quantity * close_p
                    commission_cost = notional * self.commission
                    cash -= notional + commission_cost
                    position = quantity
                    entry_price = close_p
                    entry_bar_idx = i

            # ----------------------------------------------------------
            # 3. If in a position and got a SELL signal, close it
            # ----------------------------------------------------------
            if position > 0 and signal_type == SignalType.SELL:
                pnl = position * (close_p - entry_price)
                notional = position * close_p
                commission_cost = notional * self.commission
                pnl -= commission_cost
                cash += notional - commission_cost

                trade_records.append({
                    "entry_time": data.index[entry_bar_idx],
                    "exit_time": ts,
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_price": entry_price,
                    "exit_price": close_p,
                    "quantity": position,
                    "pnl": pnl,
                    "pnl_pct": (close_p / entry_price - 1.0),
                    "exit_reason": "signal",
                })

                position = 0.0
                entry_price = 0.0
                entry_bar_idx = -1

            # ----------------------------------------------------------
            # 4. Mark-to-market equity at bar close
            # ----------------------------------------------------------
            if position > 0:
                equity = cash + position * close_p
            else:
                equity = cash

            equity_values.append(float(equity))

        # If we're still in a position at the end, mark it at final close
        # (position is already marked in the last equity value above).

        equity_curve = pd.Series(equity_values, index=data.index.insert(0, data.index[0] - pd.Timedelta("1min")), dtype=float)
        # Fix: align equity with data index properly — first value is initial capital
        # at a synthetic pre-first-bar timestamp. Re-index to data.index.
        equity_curve = pd.Series(
            equity_values,
            index=[data.index[0] - (data.index[1] - data.index[0])] + list(data.index),
            dtype=float,
        )

        trades_df = pd.DataFrame(
            trade_records,
            columns=[
                "entry_time", "exit_time", "symbol", "side",
                "entry_price", "exit_price", "quantity",
                "pnl", "pnl_pct", "exit_reason",
            ],
        )

        return equity_curve, trades_df
