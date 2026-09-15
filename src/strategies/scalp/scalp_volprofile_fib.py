"""Module 3 — Volume Profile + Fibonacci liquidity-sweep retracement
(``scalp_volprofile_fib.py``).

The model trades the opening sweep-and-reclaim on 5m bars:

1. **Open sweep** — within the first ``open_sweep_bars_5m`` 5m bars of the
   session, price sweeps a prior-day liquidity level: a LONG setup sweeps
   below the lowest of (PDL, prev-day Asia low); a SHORT setup sweeps above
   the highest of (PDH, prev-day Asia/London high).
2. **Confirmation** — exactly 2 consecutive 5m candles opposite the sweep
   (2 green after a down-sweep; 2 red after an up-sweep).  The setup is
   INVALIDATED if either of those candles breaks the sweep extreme
   (a close back through the swept level — the flow is dead).
3. **Swing & profile** — the swing runs from the sweep extreme to the
   confirmed counter-extreme (``L = low[s]`` → ``H = max(high[s+1..s+2])``
   for longs).  A row-based volume profile (documented algorithm — rows zoom
   the swing band) is computed over the swing bars; VAH/VAL are the 70 %
   cumulative-volume prices.
4. **Validation** — SHORT only valid when ``VAH`` sits below the 0.50 AND
   0.58 retracement levels (spec rule, literal); LONG mirror: ``VAL`` above
   the 0.50 AND 0.58 levels.
5. **Entry** — LIMIT at the 0.58 retracement of the swing; SL beyond the
   swing extreme plus buffer; TP at the nearest opposite liquidity
   (opposite PDH/PDL, session level or REL/REH) — **not** a hardcoded point
   target; R-multiple fallback when no structural level exists.  Minimum
   1:1 R:R is enforced for consolidating markets.

Deterministic and side-effect-free.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.strategies.scalp.indicators import fib_levels, volume_profile
from src.strategies.scalp.types import (
    CandleFrame,
    Direction,
    EntryType,
    LiquidityMap,
    ScalpContext,
    ScalpSignal,
)

MODULE_NAME = "volprofile_fib"

DEFAULT_CONFIG: dict = {
    "open_sweep_bars_5m": 12,  # sweep must occur within the first N 5m bars (=1h)
    "confirm_bars": 2,  # exactly 2 consecutive opposite candles
    "fib_entry": "0.580",  # entry retracement ratio (spec: 0.58)
    "profile_rows": 24,  # volume-profile rows across the swing
    "vah_pct": 0.7,  # VAH/VAL cumulative-volume threshold
    "sl_buffer": 0.02,  # absolute buffer beyond the swing extreme
    "min_rr": 1.0,  # minimum 1:1 R:R rule
    "tp_fallback_r": 2.0,  # R-multiple fallback TP
    "breakeven_trigger_r": 1.0,
    "trailing": True,
    "trail_distance_r": 1.0,
}


def _long_sweep(liq: LiquidityMap) -> Optional[float]:
    """Lowest prior-day liquidity below which a LONG sweep is valid."""
    cands = [v for v in (liq.pdl, liq.asia_low, liq.london_low, liq.prev_day_low) if v is not None]
    return min(cands) if cands else None


def _short_sweep(liq: LiquidityMap) -> Optional[float]:
    """Highest prior-day liquidity above which a SHORT sweep is valid."""
    cands = [v for v in (liq.pdh, liq.asia_high, liq.london_high, liq.prev_day_high) if v is not None]
    return max(cands) if cands else None


def _swing_candidates(f5m: CandleFrame, sweep_level: float, direction: str, cfg: dict) -> list[dict]:
    """Find valid sweep→2-candle confirmation setups.

    Returns a list of dicts with keys ``s`` (sweep bar), ``L``, ``H``,
    ``levels`` (fib dict), ``profile`` (VolumeProfile) — one per valid setup
    in chronological order.
    """
    n = len(f5m)
    open_win = int(cfg.get("open_sweep_bars_5m", 12))
    confirm = int(cfg.get("confirm_bars", 2))
    out: list[dict] = []
    for s in range(1, n - confirm):
        if s > open_win and s != 1:
            pass  # open-window check below
        if s > open_win:
            continue  # sweep must be at the session open
        if direction == "LONG":
            if float(f5m.low[s]) >= sweep_level:
                continue  # no sweep below liquidity
            if s + confirm >= n:
                continue
            # exactly `confirm` consecutive green candles after the sweep
            ok = all(float(f5m.close[s + k]) > float(f5m.open[s + k]) for k in range(1, confirm + 1))
            if not ok:
                continue
            # invalidation: an intervening candle breaks the flow (close back
            # through the swept level)
            if any(float(f5m.close[s + k]) <= float(f5m.low[s]) for k in range(1, confirm + 1)):
                continue
            L = float(f5m.low[s])
            H = max(float(f5m.high[s + k]) for k in range(1, confirm + 1))
            if H <= L:
                continue
            levels = fib_levels(L, H, direction="up")
            bars_hi = f5m.high[s : s + confirm + 1]
            bars_lo = f5m.low[s : s + confirm + 1]
            bars_vol = f5m.volume[s : s + confirm + 1]
            prof = volume_profile(
                bars_hi, bars_lo, bars_vol, L, H,
                rows=int(cfg.get("profile_rows", 24)),
                vah_pct=float(cfg.get("vah_pct", 0.7)),
            )
            out.append({"s": s, "L": L, "H": H, "levels": levels, "profile": prof})
        else:  # SHORT
            if float(f5m.high[s]) <= sweep_level:
                continue
            if s + confirm >= n:
                continue
            ok = all(float(f5m.close[s + k]) < float(f5m.open[s + k]) for k in range(1, confirm + 1))
            if not ok:
                continue
            if any(float(f5m.close[s + k]) >= float(f5m.high[s]) for k in range(1, confirm + 1)):
                continue
            H = float(f5m.high[s])
            L = min(float(f5m.low[s + k]) for k in range(1, confirm + 1))
            if H <= L:
                continue
            levels = fib_levels(L, H, direction="down")
            bars_hi = f5m.high[s : s + confirm + 1]
            bars_lo = f5m.low[s : s + confirm + 1]
            bars_vol = f5m.volume[s : s + confirm + 1]
            prof = volume_profile(
                bars_hi, bars_lo, bars_vol, L, H,
                rows=int(cfg.get("profile_rows", 24)),
                vah_pct=float(cfg.get("vah_pct", 0.7)),
            )
            out.append({"s": s, "H": H, "L": L, "levels": levels, "profile": prof})
    return out


def evaluate(ctx: ScalpContext, cfg: dict | None = None) -> list[ScalpSignal]:
    """All valid Volume-Profile + Fib signals for one day's 5m frame.

    ``ctx.frames["5m"]`` holds the day's 5m bars; ``ctx.liquidity`` carries
    the prior-day levels.  Missing frame or no sweep level → ``[]``.
    """
    cfg = cfg or DEFAULT_CONFIG
    f5m = ctx.frames.get("5m")
    if f5m is None or len(f5m) < 5:
        return []
    liq = ctx.liquidity
    out: list[ScalpSignal] = []
    sl_buf = float(cfg.get("sl_buffer", 0.02))
    min_rr = float(cfg.get("min_rr", 1.0))
    entry_ratio = float(cfg.get("fib_entry", 0.58))

    # ---- LONG setups ----
    lvl = _long_sweep(liq)
    if lvl is not None:
        for cand in _swing_candidates(f5m, lvl, "LONG", cfg):
            lev = cand["levels"]  # fib levels of the up swing
            prof = cand["profile"]
            entry = lev[f"{entry_ratio:.3f}"]
            # VAL validation (mirror of the spec's short rule): VAL must sit
            # above the 0.50 AND 0.58 levels
            if not (prof.val > lev["0.500"] and prof.val > lev["0.580"]):
                continue
            sl = cand["L"] - sl_buf
            if sl >= entry:
                continue
            risk = entry - sl
            tp = _nearest_above(liq.long_targets(), entry)
            if tp is None or (tp - entry) / risk < min_rr:
                # no structural TP, or structural TP is tighter than 1:1 —
                # fall back to the R-multiple target (2R default, always >= 1R)
                tp = entry + risk * float(cfg.get("tp_fallback_r", 2.0))
            out.append(_make_signal(f5m, cand, Direction.LONG, entry, sl, tp, liq, cfg))

    # ---- SHORT setups ----
    lvl = _short_sweep(liq)
    if lvl is not None:
        for cand in _swing_candidates(f5m, lvl, "SHORT", cfg):
            lev = cand["levels"]
            prof = cand["profile"]
            entry = lev[f"{entry_ratio:.3f}"]
            # spec rule (literal): SHORT valid only when VAH sits below the
            # 0.50 AND 0.58 levels
            if not (prof.vah < lev["0.500"] and prof.vah < lev["0.580"]):
                continue
            sl = cand["H"] + sl_buf
            if sl <= entry:
                continue
            risk = abs(entry - sl)
            tp = _nearest_below(liq.short_targets(), entry)
            if tp is None or (entry - tp) / risk < min_rr:
                tp = entry - risk * float(cfg.get("tp_fallback_r", 2.0))
            out.append(_make_signal(f5m, cand, Direction.SHORT, entry, sl, tp, liq, cfg))
    return out


def _make_signal(
    f5m: CandleFrame, cand: dict, direction: Direction,
    entry: float, sl: float, tp: float, liq: LiquidityMap, cfg: dict,
) -> ScalpSignal:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    is_long = direction == Direction.LONG
    entry_key = f"{float(cfg.get('fib_entry', 0.58)):.3f}"
    return ScalpSignal(
        symbol=f5m.symbol,
        timestamp=pd.Timestamp(f5m.ts[cand["s"]]),
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
            "sweep_idx": int(cand["s"]),
            "swing_low": round(cand["L"], 4),
            "swing_high": round(cand["H"], 4),
            "fib_entry_level": f"{entry_key}",
            "fib_levels": {k: round(v, 4) for k, v in cand["levels"].items()},
            "vah": round(cand["profile"].vah, 4),
            "val": round(cand["profile"].val, 4),
            "poc": round(cand["profile"].poc, 4),
            "profile_rows": int(cfg.get("profile_rows", 24)),
            "row_size": round(cand["profile"].row_size, 6),
            "side": "val_above_0.50/0.58" if is_long else "vah_below_0.50/0.58",
        },
    )


def _nearest_above(targets: list[float], price: float) -> Optional[float]:
    above = [t for t in targets if t > price]
    return min(above) if above else None


def _nearest_below(targets: list[float], price: float) -> Optional[float]:
    below = [t for t in targets if t < price]
    return max(below) if below else None