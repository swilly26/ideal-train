"""Module 1 — ICT HTF imbalance & IFVG scalp model (``scalp_ict_ifvg.py``).

The model combines higher-timeframe structure with a 1-minute inversion FVG
entry:

1. **HTF bias** — from 1H + 4H price-vs-EMA plus 1H swing structure
   (see :func:`htf_bias` for the documented definition).
2. **Active 30m FVG** — the most recent 30m fair value gap (3-candle
   imbalance: ``low[i] > high[i-2]`` for a bullish gap) within the last
   ``fvg_window_30m`` bars.
3. **15m tap** — a 15m bar spanning the active 30m gap zone (price has
   "tagged" the imbalance) within ``tap_window_15m`` bars.
4. **1m sweep + IFVG** — on the 1m frame price sweeps a session liquidity
   level (PDL / previous-day Asia / London low, or a REL) and then forms an
   *inversion* FVG within ``ifvg_window_1m`` bars of the sweep.  The entry is
   a LIMIT at the midpoint of that 1m IFVG.
5. **Filters** — SMT divergence against a correlated pair (best-effort on
   US equities; no-op when pair data is missing) and the AMD structure
   classifier (heuristic; recorded in the signal metadata).

SL sits below the local structure low (LONG) / above the local structure
high (SHORT); TP is the nearest opposite liquidity (REH / PDH / session
high for longs) with a configurable R-multiple fallback.  Break-even after
``breakeven_trigger_r`` x risk and an aggressive trail are exposed as
signal parameters — rule documented on
:class:`~src.strategies.scalp.types.ScalpSignal`.

Deterministic and side-effect-free (numpy in, list[ScalpSignal] out).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.strategies.scalp.indicators import (
    FVG,
    classify_amd,
    ema,
    equal_highs,
    equal_lows,
    fvg_gaps,
    swing_pivots,
    swing_structure,
    smt_divergence,
)
from src.strategies.scalp.types import (
    CandleFrame,
    Direction,
    EntryType,
    LiquidityMap,
    ScalpContext,
    ScalpSignal,
    SessionWindow,
)

MODULE_NAME = "ict_ifvg"

#: Default previous-day session windows (exchange-local ET clock times).
DEFAULT_SESSION_WINDOWS: tuple[SessionWindow, ...] = (
    SessionWindow("asia", "20:00", "24:00"),  # prev-day Asia kill zone
    SessionWindow("london", "02:00", "05:00"),  # London open
    SessionWindow("range", "09:30", "16:00"),  # prior RTH session
)

#: Full documented config surface (overridable per call; tests override it).
DEFAULT_CONFIG: dict = {
    "ema_1h_period": 20,
    "ema_4h_period": 10,
    "pivot_left": 2,
    "pivot_right": 2,
    "fvg_window_30m": 12,  # 30m bars — FVG must have formed within this window
    "tap_window_15m": 6,  # 15m bars — tag bar must be within this window
    "sweep_lookback_1m": 30,  # 1m bars — scope of the sweep scan
    "ifvg_window_1m": 8,  # 1m bars — IFVG must form within this many bars after the sweep
    "sl_buffer": 0.02,  # absolute buffer below structure (LONG) / above (SHORT)
    "tp_fallback_r": 2.0,  # R-multiple fallback TP when no structural liquidity
    "min_rr": 1.0,  # ground R:R floor beneath the TP
    "smt_enabled": True,  # best-effort SMT divergence filter (approximation on equities)
    "rel_tolerance_pct": 0.05,  # REL/REH clustering tolerance (fraction)
    "session_windows": DEFAULT_SESSION_WINDOWS,
    "trailing": True,  # aggressive flush after 1R
    "breakeven_trigger_r": 1.0,
    "trail_distance_r": 1.0,
}


def htf_bias(ctx: ScalpContext, cfg: dict | None = None) -> str:
    """HTF directional bias — ``"long"``, ``"short"`` or ``"neutral"``.

    Definition (documented, deterministic):

    * ``long``  — 1H close > EMA(ema_1h_period) AND 4H close >
      EMA(ema_4h_period) AND 1H swing structure is not ``"downtrend"``.
    * ``short`` — exact mirror.
    * ``neutral`` — anything else (mixed signals or missing frames).

    EMA period defaults: 20 on 1H, 10 on 4H; swing pivots use
    pivot_left/pivot_right (default 2/2).
    """
    cfg = cfg or DEFAULT_CONFIG
    p1h = int(cfg.get("ema_1h_period", 20))
    p4h = int(cfg.get("ema_4h_period", 10))
    f1h = ctx.frames.get("1h")
    f4h = ctx.frames.get("4h")
    if f1h is None or f4h is None:
        return "neutral"
    e1h = ema(f1h.close, p1h)
    e4h = ema(f4h.close, p4h)
    if len(f1h) < max(p1h, 3) or len(f4h) < max(p4h, 3):
        return "neutral"
    if np.isnan(e1h[-1]) or np.isnan(e4h[-1]):
        return "neutral"
    label1h, _, _ = swing_structure(
        f1h.high, f1h.low,
        left=int(cfg.get("pivot_left", 2)),
        right=int(cfg.get("pivot_right", 2)),
    )
    up = float(f1h.close[-1]) > float(e1h[-1]) and float(f4h.close[-1]) > float(e4h[-1]) and label1h != "downtrend"
    dn = float(f1h.close[-1]) < float(e1h[-1]) and float(f4h.close[-1]) < float(e4h[-1]) and label1h != "uptrend"
    if up and not dn:
        return "long"
    if dn and not up:
        return "short"
    return "neutral"


def _active_fvg(f30: CandleFrame, direction: str, cfg: dict) -> Optional[FVG]:
    """Most recent 30m FVG of *direction* within ``fvg_window_30m`` bars."""
    gaps = fvg_gaps(f30.high, f30.low)
    window = int(cfg.get("fvg_window_30m", 12))
    cands = [g for g in gaps if g.direction == direction and len(f30) - window < g.formation_idx <= len(f30) - 1]
    return max(cands, key=lambda g: g.formation_idx) if cands else None


def tap_index(f15: CandleFrame, fvg: FVG, cfg: dict) -> Optional[int]:
    """Index of the newest 15m bar tagging the *fvg* zone, or None.

    A bar tags the gap when it spans it (``low <= gap.top and
    high >= gap.bottom``).  Only bars within the last ``tap_window_15m``
    bars count.
    """
    window = int(cfg.get("tap_window_15m", 6))
    lo = max(0, len(f15) - window)
    for j in range(len(f15) - 1, lo - 1, -1):
        if float(f15.low[j]) <= fvg.top and float(f15.high[j]) >= fvg.bottom:
            return j
    return None


def _pair_extremes(
    ctx: ScalpContext, f1m: CandleFrame, sweep_idx: int, end_idx: int
) -> Optional[tuple[float, float, float, float]]:
    """(pair_now_lo, pair_now_hi, pair_prev_lo, pair_prev_hi) — or None.

    Windows are aligned by timestamp: *now* spans [ts(sweep_idx), ts(end_idx)]
    and *prev* is the equal-length span immediately before it.  Returns None
    when the pair frame is missing or does not overlap both windows.
    """
    pair = ctx.pair_frames.get("1m")
    if pair is None or len(pair) == 0:
        return None
    ts_sweep = f1m.ts[sweep_idx]
    ts_end = f1m.ts[min(end_idx, len(f1m) - 1)]
    span = ts_end - ts_sweep
    mask_now = (pair.ts >= ts_sweep) & (pair.ts <= ts_end)
    mask_prev = (pair.ts >= ts_sweep - span) & (pair.ts < ts_sweep)
    if not mask_now.any() or not mask_prev.any():
        return None
    return (
        float(pair.low[mask_now].min()),
        float(pair.high[mask_now].max()),
        float(pair.low[mask_prev].min()),
        float(pair.high[mask_prev].max()),
    )


def _nearest_above(targets: list[float], price: float) -> Optional[float]:
    above = [t for t in targets if t > price]
    return min(above) if above else None


def _nearest_below(targets: list[float], price: float) -> Optional[float]:
    below = [t for t in targets if t < price]
    return max(below) if below else None


def _build_signal(
    f1m: CandleFrame,
    g: FVG,
    entry: float,
    sl: float,
    tp: float,
    direction: Direction,
    liq: LiquidityMap,
    rels: list[float],
    rehs: list[float],
    bias: str,
    amd: str,
    cfg: dict,
    metadata_extra: dict,
) -> ScalpSignal:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    ts = f1m.ts[g.formation_idx]
    sig_ts = pd.Timestamp(ts)
    return ScalpSignal(
        symbol=f1m.symbol,
        timestamp=sig_ts,
        direction=direction,
        entry_type=EntryType.LIMIT,
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
            "bias": bias,
            "amd": amd,
            "rel": [round(x, 4) for x in rels],
            "reh": [round(x, 4) for x in rehs],
            "pdh": liq.pdh,
            "pdl": liq.pdl,
            "ifvg_idx": int(g.formation_idx),
            "ifvg_zone": (round(g.bottom, 4), round(g.top, 4)),
            **metadata_extra,
        },
    )


def _ifvg_long(
    f1m: CandleFrame,
    liq: LiquidityMap,
    sweep_level: float,
    rels: list[float],
    rehs: list[float],
    bias: str,
    amd: str,
    ctx: ScalpContext,
    cfg: dict,
) -> Optional[ScalpSignal]:
    gaps = fvg_gaps(f1m.high, f1m.low)
    lookback = int(cfg.get("sweep_lookback_1m", 30))
    win = int(cfg.get("ifvg_window_1m", 8))
    buf = float(cfg.get("sl_buffer", 0.02))
    smt_on = bool(cfg.get("smt_enabled", True))
    for g in [g for g in gaps if g.direction == "bullish"]:
        i = g.formation_idx
        s_lo = max(0, i - 2 - lookback)
        seg = f1m.low[s_lo : i - 1]
        if len(seg) == 0:
            continue
        s = s_lo + int(np.argmin(seg))
        if float(f1m.low[s]) >= sweep_level:  # must sweep the liquidity level
            continue
        if (i - 2) - s > win:  # IFVG too far after the sweep
            continue
        if smt_on:
            pair = _pair_extremes(ctx, f1m, s, i - 2)
            if pair is not None:
                now_lo, now_hi, prev_lo, prev_hi = pair
                if not smt_divergence(float(f1m.low[s]), prev_lo, now_lo, prev_lo, "long"):
                    continue  # pair also made a fresh low → no divergence
        entry = g.mid
        sl = float(np.min(f1m.low[i - 2 : i + 1])) - buf
        if sl >= entry:
            continue
        risk = entry - sl
        tp = _nearest_above(liq.long_targets() + rehs, entry)
        min_rr = float(cfg.get("min_rr", 1.0))
        if tp is None or (tp - entry) / risk < min_rr:
            tp = entry + risk * float(cfg.get("tp_fallback_r", 2.0))
        return _build_signal(
            f1m, g, entry, sl, tp, Direction.LONG, liq, rels, rehs, bias, amd,
            cfg,
            {"sweep_idx": int(s), "sweep_level": round(float(sweep_level), 4), "hit": "1m IFVG LIMIT"},
        )
    return None


def _ifvg_short(
    f1m: CandleFrame,
    liq: LiquidityMap,
    sweep_level: float,
    rels: list[float],
    rehs: list[float],
    bias: str,
    amd: str,
    ctx: ScalpContext,
    cfg: dict,
) -> Optional[ScalpSignal]:
    gaps = fvg_gaps(f1m.high, f1m.low)
    lookback = int(cfg.get("sweep_lookback_1m", 30))
    win = int(cfg.get("ifvg_window_1m", 8))
    buf = float(cfg.get("sl_buffer", 0.02))
    smt_on = bool(cfg.get("smt_enabled", True))
    for g in [g for g in gaps if g.direction == "bearish"]:
        i = g.formation_idx
        s_lo = max(0, i - 2 - lookback)
        seg = f1m.high[s_lo : i - 1]
        if len(seg) == 0:
            continue
        s = s_lo + int(np.argmax(seg))
        if float(f1m.high[s]) <= sweep_level:  # must sweep above the level
            continue
        if (i - 2) - s > win:
            continue
        if smt_on:
            pair = _pair_extremes(ctx, f1m, s, i - 2)
            if pair is not None:
                now_lo, now_hi, prev_lo, prev_hi = pair
                if not smt_divergence(float(f1m.high[s]), prev_hi, now_hi, prev_hi, "short"):
                    continue
        entry = g.mid
        sl = float(np.max(f1m.high[i - 2 : i + 1])) + buf
        if sl <= entry:
            continue
        risk = abs(entry - sl)
        tp = _nearest_below(liq.short_targets() + rels, entry)
        min_rr = float(cfg.get("min_rr", 1.0))
        if tp is None or (entry - tp) / risk < min_rr:
            tp = entry - risk * float(cfg.get("tp_fallback_r", 2.0))
        return _build_signal(
            f1m, g, entry, sl, tp, Direction.SHORT, liq, rels, rehs, bias, amd,
            cfg,
            {"sweep_idx": int(s), "sweep_level": round(float(sweep_level), 4), "hit": "1m IFVG LIMIT"},
        )
    return None


def evaluate(ctx: ScalpContext, cfg: dict | None = None) -> list[ScalpSignal]:
    """Emit the day's ICT IFVG signal (at most one — the latest setup).

    Requires frames ``1m/15m/30m/1h/4h`` in ``ctx.frames`` (the integration
    layer assembles them; missing frames → ``[]``).  ``1d`` is optional —
    PDH/PDL come from ``ctx.liquidity``.  SMT pair frame optional under
    ``ctx.pair_frames["1m"]`` (default pair symbol ``QQQ``).
    """
    cfg = cfg or DEFAULT_CONFIG
    for key in ("1m", "15m", "30m", "1h", "4h"):
        if key not in ctx.frames:
            return []
    f1m, f15, f30 = ctx.frames["1m"], ctx.frames["15m"], ctx.frames["30m"]
    if len(f1m) < 6 or len(f15) < 3 or len(f30) < 3:
        return []
    liq = ctx.liquidity
    bias = htf_bias(ctx, cfg)
    hi_piv, lo_piv = swing_pivots(
        f1m.high, f1m.low,
        left=int(cfg.get("pivot_left", 2)),
        right=int(cfg.get("pivot_right", 2)),
    )
    tol = float(cfg.get("rel_tolerance_pct", 0.05))
    rels = equal_lows(f1m.low, lo_piv, tol)
    rehs = equal_highs(f1m.high, hi_piv, tol)
    amd = classify_amd(f15.high, f15.low, int(cfg.get("pivot_left", 2)), int(cfg.get("pivot_right", 2)))

    if bias == "long":
        fvg30 = _active_fvg(f30, "bullish", cfg)
        if fvg30 is not None and tap_index(f15, fvg30, cfg) is not None:
            # sweep targets: prior-day session liquidity (PDL / Asia / London
            # lows) — REL is a *TP* magnet, not a sweep target (a sweep below
            # today's own REL would be self-referential)
            sweep_src = [liq.pdl, liq.asia_low, liq.london_low]
            sweep_level = min((v for v in sweep_src if v is not None), default=None)
            if sweep_level is not None:
                sig = _ifvg_long(f1m, liq, sweep_level, rels, rehs, bias, amd, ctx, cfg)
                if sig is not None:
                    return [sig]
    if bias == "short":
        fvg30 = _active_fvg(f30, "bearish", cfg)
        if fvg30 is not None and tap_index(f15, fvg30, cfg) is not None:
            sweep_src = [liq.pdh, liq.asia_high, liq.london_high]
            sweep_level = max((v for v in sweep_src if v is not None), default=None)
            if sweep_level is not None:
                sig = _ifvg_short(f1m, liq, sweep_level, rels, rehs, bias, amd, ctx, cfg)
                if sig is not None:
                    return [sig]
    return []