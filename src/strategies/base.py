"""Base class and data structures for trading strategies.

Every strategy must extend ``Strategy`` and implement ``generate_signals``.
The AI optimisation layer reads and writes ``StrategyConfig`` to tune
strategy parameters dynamically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A single trading signal produced by a strategy."""

    symbol: str
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float = 1.0  # 0.0 – 1.0, used by the optimiser for weighting
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Tunable parameters exposed to the AI optimisation layer.

    The optimiser will mutate these fields (entry/exit thresholds,
    position sizing, stop-loss levels, etc.) and feed them back into
    the strategy for backtesting / live execution.
    """

    # Entry / exit thresholds
    entry_threshold: float = 0.5
    exit_threshold: float = 0.3

    # Position sizing
    max_position_pct: float = 0.10  # 10 % of portfolio per position
    min_position_pct: float = 0.01

    # Risk management
    stop_loss_pct: float = 0.02  # 2 % stop-loss
    take_profit_pct: float = 0.05  # 5 % take-profit

    # Strategy weight (if multiple strategies are combined)
    weight: float = 1.0

    # Arbitrary extra params a specific strategy may need
    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for every trading strategy.

    Subclasses only need to implement ``generate_signals``.  The
    ``config`` attribute is the AI-tunable interface — the optimiser
    mutates it between backtest runs to discover profitable settings.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Produce trading signals from a DataFrame of market data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV bars indexed by timestamp.

        Returns
        -------
        list[Signal]
            One signal per bar (or fewer, if the strategy is sparse).
        """
        ...
