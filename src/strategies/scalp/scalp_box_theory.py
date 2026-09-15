"""Module 2 — Daily Range Box Theory scalp model (``scalp_box_theory.py``).

The "box" is the previous trading day's range (PDH/PDL) split into three
zones:

* **Zone A (top)** — price at or above ``pdh - zone_width * range``.
* **Zone B (bottom)** — price at or below ``pdl + zone_width * range``.
* **Zone C (middle)** — the do-nothing zone in between.

Rules (deterministic, documented):

* **SHORT trigger** — a 5m bar's *high* enters Zone A AND the bar closes 1
  red 5m candle below the previous candle's close (``close[i] < close[i-1]).
  Entry is a MARKET order at the trigger candle's close.  SL sits above the
  prior candle's high (``max(high[i], high[i-1])``) plus a buffer; TP at the
  opposite box boundary (PDL).
* **LONG trigger** — mirror: low enters Zone B, close is green
  (``close[i] > close[i-1]).  SL below ``min(low[i], low[i-1])`` minus a
  buffer; TP at PDH.
* **Zone C = do nothing** — no triggers inside the middle zone.
* **Cooldown** — one trade per boundary touch: a boundary does not re-fire
  until price closes back into Zone C.

BE / trail: after 1R in profit the stop moves to break-even; aggressive
trailing (1R trail distance) is exposed via the signal params.  R:R is never
below ``min_rr`` (the opposite boundary is far away, so the guard almost
never fires — it exists for pathological boxes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.strategies.scalp.indicators import ema  # noqa: F401  (kept import surface stable)
from src.strategies.scalp.types import (
    CandleFrame,
    Direction,
    EntryType,
    ScalpContext,
    ScalpSignal,
)

MODULE_NAME = "box_theory"

DEFAULT_CONFIG: dict = {
    "zone_width": 0.10,  # fraction of the daily range for zones A and B
    "sl_buffer": 0.01,  # absolute price buffer beyond structure
    "min_rr": 1.0,  # ground R:R floor
    "tp_fallback_r": 2.0,  # R-multiple fallback if TP is degenerate
    "breakeven_trigger_r": 1.0,
    "trailing": True,
    "trail_distance_r": 1.0,
}


@dataclass(frozen=True)
class Box:
    """Daily range box derived from the previous trading day.

    ``zone_top`` is the level at-or-above which price is in Zone A;
    ``zone_bottom`` is the level at-or-below which price is in Zone B.
    """

    pdh: float
    pdl: float
    range: float
    zone_top: float
    zone_bottom: float

    @classmethod
    def build(cls, pdh: float, pdl: float, zone_width: float) -> "Box":
        rng = pdh - pdl
        return cls(pdh=pdh, pdl=pdl, range=rng, zone_top=pdh - zone_width * rng, zone_bottom=pdl + zone_width * rng)


def classify_zone(price: float, box: Box) -> str:
    """Zone classification of a price: ``"A"`` (top), ``"B"`` (bottom), ``"C"`` (middle)."""
    if price >= box.zone_top:
        return "A"
    if price <= box.zone_bottom:
        return "B"
    return "C"


def _build_signal(
    f5m: CandleFrame,
    i: int, box: Box, direction: Direction, entry: float, sl: float, tp: float, cfg: dict,
) -> ScalpSignal:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return ScalpSignal(
        symbol=f5m.symbol,
        timestamp=pd.Timestamp(f5m.ts[i]),
        direction=direction,
        entry_type=EntryType.MARKET,
        entry_price=round(entry, 6),
        stop_loss=round(sl, 6),
        take_profit=round(tp, 6),
        risk=round(risk, 6),
        reward=round(reward, 6),
        rr=round(reward / risk, 4),
        strategy=MODULE_NAME,
        breakeven_trigger_r=float(cfg.get("breakeven_trigger_r", 1.0)),
        breakeven_buffer=0.0,
        trailing=bool(cfg.get("trailing", True)),
        trail_distance_r=float(cfg.get("trail_distance_r", 1.0)),
        trail_trigger_r=float(cfg.get("breakeven_trigger_r", 1.0)),
        metadata={
            "box": f"pdh={round(box.pdh, 4)} pdl={round(box.pdl, 4)}",
            "zone_bottom": round(box.zone_bottom, 4),
            "zone_top": round(box.zone_top, 4),
            "box_range": round(box.range, 4),
            "trigger_idx": int(i),
            "sl_buffer": float(cfg.get("sl_buffer", 0.01)),
        },
    )


def evaluate(ctx: ScalpContext, cfg: dict | None = None) -> list[ScalpSignal]:
    """All signals for one day's 5m frame.

    ``ctx.frames["5m"]`` holds the day's 5m bars (harness builds one day at a
    time); ``ctx.liquidity.pdh/pdl`` carry the PRIOR day's levels.
    Returns one signal per boundary touch that triggers (cooldown respected).
    Missing 5m frame or PDH/PDL → ``[]``.
    """
    cfg = cfg or DEFAULT_CONFIG
    f5m = ctx.frames.get("5m")
    if f5m is None or len(f5m) < 2:
        return []
    pdh, pdl = ctx.liquidity.pdh, ctx.liquidity.pdl
    if pdh is None or pdl is None or pdl >= pdh:
        return []
    box = Box.build(pdh, pdl, float(cfg.get("zone_width", 0.10)))
    buf = float(cfg.get("sl_buffer", 0.01))
    min_rr = float(cfg.get("min_rr", 1.0))
    top_fired = False
    bottom_fired = False
    out: list[ScalpSignal] = []

    for i in range(1, len(f5m)):
        hi, lo, cl, cl_prev = float(f5m.high[i]), float(f5m.low[i]), float(f5m.close[i]), float(f5m.close[i - 1])
        # re-arm when price closes back into Zone C
        in_c = box.zone_bottom < cl < box.zone_top
        if in_c:
            top_fired = False
            bottom_fired = False
        # ---- SHORT: high enters zone A, red close below prior close ----
        if not top_fired and hi >= box.zone_top and cl < cl_prev:
            entry = cl
            sl = max(hi, float(f5m.high[i - 1])) + buf
            tp = min(pdl, entry)  # opposite boundary (below entry)
            if sl > entry and tp < entry - 1e-9:
                if (entry - tp) / (sl - entry) >= min_rr:
                    out.append(_build_signal(f5m, i, box, Direction.SHORT, entry, sl, tp, cfg))
                    top_fired = True
                    continue
        # ---- LONG: low enters zone B, green close above prior close ----
        if not bottom_fired and lo <= box.zone_bottom and cl > cl_prev:
            entry = cl
            sl = min(lo, float(f5m.low[i - 1])) - buf
            tp = max(pdh, entry)  # opposite boundary (above entry)
            if sl < entry and tp > entry + 1e-9:
                if (tp - entry) / (entry - sl) >= min_rr:
                    out.append(_build_signal(f5m, i, box, Direction.LONG, entry, sl, tp, cfg))
                    bottom_fired = True
    return out