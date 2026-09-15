"""Shared types for the scalp strategy layer.

This module defines the *contract* between the strategy layer and the
integration/execution layer for the new main-trader scalping models:

* :class:`CandleFrame` — a typed container for one timeframe of OHLCV bars
  (numpy arrays + na\\[:meta\\]ive timestamps), so tests can feed hand-built
  mock candles and the future integration layer can build frames from Alpaca
  bars without any shared I/O.
* :class:`LiquidityMap` — the structural levels (PDH/PDL, previous-day
  Asia/London session highs/lows, today's rolling Asia/London levels) that
  the strategies use for sweep targets and profit targets.
* :class:`ScalpSignal` — the typed signal emitted by ``evaluate()``-style
  functions.  It carries the trade geometry (entry type / entry level, SL,
  TP, R:R) plus optional break-even / trail parameters so the execution
  layer can place a complete order bundle from one object.

Everything in this module is deterministic and side-effect-free (no I/O).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# Mapping of generic timeframe keys → Alpaca/yfinance bar intervals.
TIMEFRAME_KEYS = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")


class Direction(Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(Enum):
    """How the entry is executed.

    MARKET  — fill at the trigger candle's close (signal ``entry_price``).
    LIMIT   — resting order at ``entry_price``; fills when price trades
              through the level.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class CandleFrame:
    """One timeframe's worth of OHLCV bars.

    Fields are 1-D numpy arrays of equal length, index-aligned:
    ``ts`` holds ``datetime64`` timestamps (naive or tz-aware), the OHLCV
    arrays hold floats (volume may be int-like but stored as float for
    conveni ence).

    Only bars, never indicators — strategies derive everything from raw
    OHLCV so the inputs are fully mockable.
    """

    symbol: str
    timeframe: str
    ts: np.ndarray  # datetime64[ns]
    open: np.ndarray  # float64
    high: np.ndarray  # float64
    low: np.ndarray  # float64
    close: np.ndarray  # float64
    volume: np.ndarray  # float64

    def __post_init__(self) -> None:
        n = len(self.ts)
        for name in ("open", "high", "low", "close", "volume"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(
                    f"{self.symbol} {self.timeframe}: array '{name}' has shape "
                    f"{arr.shape}, expected {(n,)}"
                )
        if self.high.min() < self.low.min() or (self.high < self.low).any():
            raise ValueError(f"{self.symbol} {self.timeframe}: high < low detected")

    def __len__(self) -> int:
        return len(self.ts)

    @classmethod
    def from_dataframe(cls, symbol: str, timeframe: str, df: pd.DataFrame) -> "CandleFrame":
        """Build a frame from a DataFrame with columns open/high/low/close/volume.

        The DataFrame index is used as the timestamp array.  If the index is
        tz-aware it is kept as-is (the strategies document that intraday
        frames should be localized to the exchange timezone, so session
        windows line up with clock-time filters).
        """
        required = ("open", "high", "low", "close", "volume")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} (have {list(df.columns)})")
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            ts=np.asarray(df.index.to_numpy(), dtype="datetime64[ns]"),
            open=df["open"].to_numpy(dtype=float),
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            volume=df["volume"].to_numpy(dtype=float),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a plain OHLCV DataFrame indexed by ``ts`` (for harnesses)."""
        return pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            },
            index=pd.DatetimeIndex(self.ts, name="ts"),
        )

    def slice(self, start: int, end: int) -> "CandleFrame":
        """Return a shallow sub-frame [start, end)."""
        return CandleFrame(
            symbol=self.symbol,
            timeframe=self.timeframe,
            ts=self.ts[start:end],
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            volume=self.volume[start:end],
        )


@dataclass(frozen=True)
class SessionWindow:
    """A named clock-time window on a trading day, in exchange local time.

    Windows are defined on the *previous* trading day's bars so that "previous
    day's Asia high/low" has a clean, unambiguous meaning without any
    midnight-candle semantics.  Times are inclusive at ``start`` and exclusive
    at ``end``; either may wrap past midnight (e.g. Asia 20:00 → 24:00).
    """

    name: str
    start: str  # "HH:MM" exchange-local
    end: str  # "HH:MM" exchange-local (24:00 = end of that day)


@dataclass(frozen=True)
class LiquidityMap:
    """Structural liquidity levels known at the evaluation point.

    ``pdh``/``pdl`` come from the *previous* trading day's daily bar; the
    session levels come from the previous day's intraday bars in the
    configured windows.  ``session_high``/``session_low`` are today's
    in-progress (current-day RTH) extremes — useful as the "opposite
    liquidity" target inside the session.
    """

    pdh: Optional[float] = None
    pdl: Optional[float] = None
    asia_high: Optional[float] = None  # prev-day Asia window high
    asia_low: Optional[float] = None  # prev-day Asia window low
    london_high: Optional[float] = None  # prev-day London window high
    london_low: Optional[float] = None  # prev-day London window low
    session_high: Optional[float] = None  # today in-progress RTH high
    session_low: Optional[float] = None  # today in-progress RTH low
    prev_day_high: Optional[float] = None  # alias of prev-day full range high
    prev_day_low: Optional[float] = None  # alias of prev-day full range low

    def long_targets(self) -> list[float]:
        """Structural TP candidates for a LONG (liquidity above)."""
        out = [
            v
            for v in (
                self.pdh,
                self.london_high,
                self.asia_high,
                self.prev_day_high,
                self.session_high,
            )
            if v is not None
        ]
        return sorted(set(round(float(v), 6) for v in out))

    def short_targets(self) -> list[float]:
        """Structural TP candidates for a SHORT (liquidity below)."""
        out = [
            v
            for v in (
                self.pdl,
                self.london_low,
                self.asia_low,
                self.prev_day_low,
                self.session_low,
                self.pdl,
            )
            if v is not None
        ]
        return sorted(set(round(float(v), 6) for v in out), reverse=True)


@dataclass(frozen=True)
class ScalpSignal:
    """One typed trading signal from a scalp strategy.

    Trade geometry (all prices in the symbol's quote currency):

    * LONG  — ``entry_price`` is the market close (MARKET) or limit price
      (LIMIT); ``stop_loss`` < entry; ``take_profit`` > entry.
    * SHORT — inverse: ``stop_loss`` above entry, ``take_profit`` below.

    Break-even / trail rule (deterministic, documented here):

    * When price reaches ``breakeven_trigger_r`` x risk in profit, the stop
      is moved to ``entry_price`` ± ``breakeven_buffer`` (a small adverse
      buffer; 0 = exact break-even).
    * If ``trailing`` is True, the stop then trails at
      ``trail_distance_r`` x risk behind price once price has travelled at
      least ``trail_trigger_r`` x risk in profit (aggressive flush).
    """

    symbol: str
    timestamp: pd.Timestamp  # bar time at which the signal is emitted
    direction: Direction
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float  # |entry - SL|
    reward: float  # |TP - entry|
    rr: float  # reward / risk
    strategy: str  # module name, e.g. "ict_ifvg"
    breakeven_trigger_r: float = 1.0
    breakeven_buffer: float = 0.0
    trailing: bool = False
    trail_distance_r: float = 1.0
    trail_trigger_r: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.risk <= 0.0:
            raise ValueError(f"risk must be > 0, got {self.risk}")
        if self.direction == Direction.LONG:
            ok = self.stop_loss < self.entry_price and self.take_profit > self.entry_price
        else:
            ok = self.stop_loss > self.entry_price and self.take_profit < self.entry_price
        if not ok:
            raise ValueError(
                f"impossible geometry for {self.direction.value}: "
                f"entry={self.entry_price} SL={self.stop_loss} TP={self.take_profit}"
            )


@dataclass(frozen=True)
class ScalpContext:
    """Everything a module-level ``evaluate()`` needs.

    ``frames`` maps a timeframe key (``1m``/``5m``/``15m``/``30m``/``1h``/
    ``4h``/``1d``) to its :class:`CandleFrame`.  Intraday frames should be
    localized to the exchange timezone (America/New_York) so session windows
    line up with clock times.  ``liquidity`` carries prior-day structural
    levels; the module derives everything else.

    ``pair_frames`` optionally holds a correlated symbol's frames (e.g.
    ``{"1m": ...}``) used by the best-effort SMT divergence filter.  SMT is
    an approximation on US equities and is configurable (off by default in
    the strategy module is a documented default; here it can be enabled).
    """

    symbol: str
    frames: dict  # timeframe key -> CandleFrame
    liquidity: LiquidityMap = field(default_factory=LiquidityMap)
    tz: str = "America/New_York"
    pair_frames: dict = field(default_factory=dict)
    pair_symbol: str = "QQQ"

    def frame(self, key: str) -> CandleFrame:
        if key not in self.frames:
            raise KeyError(f"{self.symbol}: no '{key}' frame in context")
        return self.frames[key]