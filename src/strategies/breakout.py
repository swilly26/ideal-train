"""Breakout trading strategy.

Uses rolling high / low channels to detect price breakouts beyond
recent trading ranges.

- BUY when price breaks above the rolling-high channel by at least
  ``entry_threshold``.
- SELL when price breaks below the rolling-low channel by at least
  ``entry_threshold``.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.indicators import breakout_distance


class BreakoutStrategy(Strategy):
    """Channel-breakout strategy.

    Config parameters
    -----------------
    entry_threshold : float (default 0.5)
        Minimum fractional distance beyond the rolling high/low to trigger
        a signal (e.g. 0.02 = 2 % beyond the channel).
    exit_threshold : float (default 0.3)
        Threshold for exiting a position (available for the AI tuner).
    extra.lookback : int (default 20)
        Number of bars for the rolling high/low window.
    """

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        lookback: int = int(self.config.extra.get("lookback", 20))
        symbol: str = str(self.config.extra.get("symbol", "SYM"))

        close = data["close"]
        high = data["high"]
        low = data["low"]

        upside, downside = breakout_distance(close, high, low, period=lookback)

        signals: list[Signal] = []
        for idx in range(len(data)):
            u = upside.iloc[idx]
            d = downside.iloc[idx]
            if pd.isna(u) or pd.isna(d):
                continue
            ts = data.index[idx]

            if u > self.config.entry_threshold:
                confidence = min(1.0, u / (2.0 * self.config.entry_threshold))
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=ts,
                        signal_type=SignalType.BUY,
                        confidence=round(confidence, 6),
                        metadata={
                            "upside_breakout": round(u, 6),
                            "lookback": lookback,
                        },
                    )
                )
            elif d > self.config.entry_threshold:
                confidence = min(1.0, d / (2.0 * self.config.entry_threshold))
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=ts,
                        signal_type=SignalType.SELL,
                        confidence=round(confidence, 6),
                        metadata={
                            "downside_breakout": round(d, 6),
                            "lookback": lookback,
                        },
                    )
                )
        return signals
