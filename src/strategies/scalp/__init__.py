"""Scalp strategy layer — the new main-trader strategy set.

Three modular, deterministic, side-effect-free strategy models plus shared
types and indicator primitives:

* :mod:`scalp_ict_ifvg`  — ICT HTF imbalance + 1m inversion-FVG model.
* :mod:`scalp_box_theory` — Daily Range Box Theory (PDH/PDL zones).
* :mod:`scalp_volprofile_fib` — Volume Profile + Fibonacci retracement on
  the opening sweep.

Each module exposes ``evaluate(ctx: ScalpContext, cfg: dict | None) ->
list[ScalpSignal]`` (empty list = no signal).  Frames are built with
:class:`CandleFrame` from raw OHLCV — the layer performs no I/O and makes no
broker calls; the integration layer (live trading) is a separate concern.
"""
from __future__ import annotations

from src.strategies.scalp.types import (
    CandleFrame,
    Direction,
    EntryType,
    LiquidityMap,
    ScalpContext,
    ScalpSignal,
    SessionWindow,
)
from src.strategies.scalp import indicators
from src.strategies.scalp import scalp_ict_ifvg
from src.strategies.scalp import scalp_box_theory
from src.strategies.scalp import scalp_volprofile_fib

__all__ = [
    "CandleFrame",
    "Direction",
    "EntryType",
    "LiquidityMap",
    "ScalpContext",
    "ScalpSignal",
    "SessionWindow",
    "indicators",
    "scalp_ict_ifvg",
    "scalp_box_theory",
    "scalp_volprofile_fib",
]