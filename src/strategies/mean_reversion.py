"""Mean-reversion trading strategy.

Uses rolling Z-score to detect overbought / oversold conditions and
generates reversal signals.

- BUY when Z-score drops below –entry_threshold (oversold).
- SELL when Z-score rises above +entry_threshold (overbought).
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.indicators import z_score


class MeanReversionStrategy(Strategy):
    """Z-score based mean-reversion strategy.

    Config parameters
    -----------------
    entry_threshold : float (default 0.5)
        Z-score magnitude that triggers an entry (absolute value).
    exit_threshold : float (default 0.3)
        Z-score magnitude at which an open position is exited.  Not used
        for pure signal generation — included so the AI tuner can adjust.
    extra.lookback : int (default 20)
        Rolling window for mean and standard-deviation calculation.
    extra.std_dev_multiplier : float (default 2.0)
        Multiplier for standard deviation in Bollinger-style bands.
        (Passed through to allow the optimiser to widen or tighten bands.)
    """

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        lookback: int = int(self.config.extra.get("lookback", 20))
        symbol: str = str(self.config.extra.get("symbol", "SYM"))
        # std_dev_multiplier is accepted but Z-score already incorporates std —
        # it is here so the AI can experiment with band-width variants.

        close = data["close"]
        z = z_score(close, period=lookback)
        threshold = abs(self.config.entry_threshold)

        signals: list[Signal] = []
        for idx in range(len(data)):
            zv = z.iloc[idx]
            if pd.isna(zv):
                continue
            ts = data.index[idx]

            if zv < -threshold:
                # Oversold → BUY (expect reversion up)
                confidence = min(1.0, abs(zv) / (2.0 * threshold))
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=ts,
                        signal_type=SignalType.BUY,
                        confidence=round(confidence, 6),
                        metadata={"z_score": round(zv, 6), "lookback": lookback},
                    )
                )
            elif zv > threshold:
                # Overbought → SELL (expect reversion down)
                confidence = min(1.0, abs(zv) / (2.0 * threshold))
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=ts,
                        signal_type=SignalType.SELL,
                        confidence=round(confidence, 6),
                        metadata={"z_score": round(zv, 6), "lookback": lookback},
                    )
                )
        return signals
