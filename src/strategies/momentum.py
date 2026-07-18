"""Momentum trading strategy.

Compares current price to an N-period simple moving average and generates
signals when momentum crosses configurable thresholds.

- BUY when normalised momentum exceeds ``entry_threshold``.
- SELL when it drops below ``exit_threshold``.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.indicators import momentum as calc_momentum


class MomentumStrategy(Strategy):
    """Trend-following momentum strategy.

    Config parameters
    -----------------
    entry_threshold : float (default 0.5)
        Minimum normalised momentum to trigger a BUY.
    exit_threshold : float (default 0.3)
        Momentum level below which a SELL is emitted.
    extra.lookback : int (default 20)
        Number of bars for the rolling moving average.
    """

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        lookback: int = int(self.config.extra.get("lookback", 20))
        symbol: str = str(self.config.extra.get("symbol", "SYM"))

        close = data["close"]
        mom = calc_momentum(close, period=lookback)

        signals: list[Signal] = []
        for idx in range(len(data)):
            m = mom.iloc[idx]
            if pd.isna(m):
                continue
            ts = data.index[idx]

            if m > self.config.entry_threshold:
                confidence = min(1.0, abs(m) / (2.0 * self.config.entry_threshold))
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=ts,
                        signal_type=SignalType.BUY,
                        confidence=round(confidence, 6),
                        metadata={"momentum": round(m, 6), "lookback": lookback},
                    )
                )
            elif m < self.config.exit_threshold:
                confidence = min(1.0, abs(m) / (2.0 * abs(self.config.exit_threshold) + 1e-9))
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=ts,
                        signal_type=SignalType.SELL,
                        confidence=round(confidence, 6),
                        metadata={"momentum": round(m, 6), "lookback": lookback},
                    )
                )
        return signals
