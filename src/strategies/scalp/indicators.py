"""Pure deterministic primitives for the scalp strategy layer.

Everything here is side-effect-free: numpy in, numpy out.  Frames are
already validated by :class:`~src.strategies.scalp.types.CandleFrame`.

Conventions
-----------
* Index ``i`` is the *formation* bar: for a 3-candle FVG the three candles
  are ``i-2, i-1, i``.
* Session windows are applied to bar timestamps' local ``hour:minute``
  (frames must be localized to the exchange timezone).
* Prices respect strict inequalities (``<``, ``>``) rather than <= / >= so
  that "sweep" rules are deterministic on integer tick data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.strategies.scalp.types import CandleFrame, SessionWindow

EPS = 1e-9


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average of *values*.

    Seeded with the SMA of the first ``period`` samples (standard
    definition); the first ``period - 1`` outputs are NaN.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < period or period <= 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Swing pivots & equal-high/low clustering (REL / REH)
# ---------------------------------------------------------------------------
def swing_pivots(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate swing highs and swing lows.

    Index ``i`` is a swing high iff ``high[i]`` is strictly greater than all
    highs in ``[i-left, i+right]`` excluding itself.  Swing lows are
    symmetric on ``low``.  Returns ``(high_idxs, low_idxs)`` as int arrays.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    high_idxs: list[int] = []
    low_idxs: list[int] = []
    for i in range(left, n - right):
        seg_h = high[i - left : i + right + 1]
        seg_l = low[i - left : i + right + 1]
        if high[i] == np.max(seg_h) and np.count_nonzero(seg_h == high[i]) == 1:
            high_idxs.append(i)
        if low[i] == np.min(seg_l) and np.count_nonzero(seg_l == low[i]) == 1:
            low_idxs.append(i)
    return np.asarray(high_idxs, dtype=int), np.asarray(low_idxs, dtype=int)


def _cluster_levels(pivot_idxs: np.ndarray, pivot_prices: np.ndarray, tolerance_pct: float) -> list[float]:
    """Cluster neighbouring pivot prices within *tolerance_pct* of their mean.

    Returns one representative level per cluster (the cluster mean).
    Deterministic: pivots are processed left→right in price order.
    """
    if len(pivot_idxs) < 2:
        return [float(p) for p in pivot_prices]
    order = np.argsort(pivot_prices)
    idxs = pivot_idxs[order]
    prices = pivot_prices[order]
    clusters: list[list[float]] = [[float(prices[0])]]
    for i in range(1, len(prices)):
        if abs(prices[i] - float(np.mean(clusters[-1]))) <= tolerance_pct * float(np.mean(clusters[-1])):
            clusters[-1].append(float(prices[i]))
        else:
            clusters.append([float(prices[i])])
    return [float(np.mean(c)) for c in clusters]


def equal_lows(low: np.ndarray, low_pivot_idxs: np.ndarray, tolerance_pct: float = 0.05) -> list[float]:
    """Relative Equal Lows (REL): cluster levels from swing-low pivots.

    Two swing lows within ``tolerance_pct`` (percent of their mean) form one
    REL.  Returns one level per cluster (cluster mean price).
    """
    return _cluster_levels(low_pivot_idxs, np.asarray(low, dtype=float)[low_pivot_idxs].astype(float), tolerance_pct)


def equal_highs(high: np.ndarray, high_pivot_idxs: np.ndarray, tolerance_pct: float = 0.05) -> list[float]:
    """Relative Equal Highs (REH): cluster levels from swing-high pivots."""
    return _cluster_levels(high_pivot_idxs, np.asarray(high, dtype=float)[high_pivot_idxs].astype(float), tolerance_pct)


# ---------------------------------------------------------------------------
# Fair Value Gaps (3-candle imbalances)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FVG:
    """A detected fair value gap.

    ``formation_idx`` is the index of the third candle of the 3-candle
    pattern.  ``top``/``bottom`` bound the imbalance zone
    (``top`` >= ``bottom`` always).
    """

    formation_idx: int
    top: float
    bottom: float
    direction: str  # "bullish" | "bearish"

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


def fvg_gaps(high: np.ndarray, low: np.ndarray) -> list[FVG]:
    """Detect all 3-candle fair value gaps.

    For candles ``i-2, i-1, i``:

    * **bullish** FVG: ``low[i] > high[i-2]`` — the third candle gaps above
      the first; imbalance zone is ``[high[i-2], low[i]]`` (price later
      retraces *down into* this gap).
    * **bearish** FVG: ``high[i] < low[i-2]`` — third candle gaps below the
      first; imbalance zone is ``[high[i], low[i-2]]`` (price later retraces
      *up into* this gap).

    Returns gaps in chronological (formation) order.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    out: list[FVG] = []
    for i in range(2, len(high)):
        if low[i] > high[i - 2]:
            out.append(FVG(i, top=float(low[i]), bottom=float(high[i - 2]), direction="bullish"))
        elif high[i] < low[i - 2]:
            out.append(FVG(i, top=float(low[i - 2]), bottom=float(high[i]), direction="bearish"))
    return out


def last_fvg_in_window(
    gaps: list[FVG],
    direction: str,
    now_idx: int,
    window: int,
) -> Optional[FVG]:
    """Return the most recent FVG of *direction* formed at index <= now_idx
    within the last *window* bars (``now_idx - window < idx <= now_idx``).

    "Most recent" means the largest ``formation_idx``.  Returns None if none.
    """
    candidates = [g for g in gaps if g.direction == direction and now_idx - window < g.formation_idx <= now_idx]
    if not candidates:
        return None
    return max(candidates, key=lambda g: g.formation_idx)


# ---------------------------------------------------------------------------
# Session windows (prior-day clock-time high/low)
# ---------------------------------------------------------------------------
def _hms(ts: np.datetime64) -> tuple[int, int]:
    """Return (hour, minute) of a datetime64 in its own (local) frame."""
    dt = ts.astype("datetime64[m]").astype(object)  # minute resolution, naive
    return dt.hour, dt.minute


def _in_window(hour: int, minute: int, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    """True if (hour, minute) is in [start, end) — supporting midnight wrap."""
    now = hour * 60 + minute
    s = start_h * 60 + start_m
    e = end_h * 60 + end_m
    if s <= e:
        return s <= now < e
    # window wraps past midnight (e.g. 20:00 → 24:00 is handled as s<=e when
    # end_h == 24; a true wrap is e.g. 23:00 → 01:00)
    return now >= s or now < e


def session_extremes(frame: CandleFrame, window: SessionWindow) -> tuple[Optional[float], Optional[float]]:
    """High/low of *frame*'s bars whose local clock time falls in *window*.

    Returns ``(high, low)`` or ``(None, None)`` when no bars fall in the
    window.  Frames are expected to be exchange-localized, so this is simply
    a clock-time filter — no timezone math here.
    """
    start_h, start_m = (int(x) for x in window.start.split(":"))
    end_h, end_m = (int(x) for x in window.end.split(":"))
    highs: list[float] = []
    lows: list[float] = []
    for ts, hi, lo in zip(frame.ts, frame.high, frame.low):
        h, m = _hms(ts)
        if _in_window(h, m, start_h, start_m, end_h, end_m):
            highs.append(float(hi))
            lows.append(float(lo))
    if not highs:
        return None, None
    return float(max(highs)), float(min(lows))


# ---------------------------------------------------------------------------
# Volume profile (session-volume-profile over a swing)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeProfile:
    """Simple row-based volume profile over a price swing.

    Algorithm (documented): the swing price band ``[swing_low, swing_high]``
    is divided into ``rows`` equal-height bins ("rows zoomed per the swing").
    Each bar's volume is assigned to the row containing the bar's typical
    price (``(high+low)/2``).  VAH/VAL are the prices at which cumulative
    volume from the top (VAH) / bottom (VAL) of the profile reaches
    ``vah_pct`` (70 %) of total volume.  POC is the row with max volume.
    """

    vah: float
    val: float
    poc: float
    row_size: float
    rows: int
    total_volume: float
    prices: np.ndarray  # row centre prices (len == rows)

    def vah_below(self, level: float) -> bool:
        return self.vah < level

    def val_above(self, level: float) -> bool:
        return self.val > level


def volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    swing_low: float,
    swing_high: float,
    rows: int = 24,
    vah_pct: float = 0.7,
) -> VolumeProfile:
    """Volume profile over the bars in ``[swing_low, swing_high]``.

    *high*/``low``/``volume`` are the arrays of the *swing bars themselves*
    (the bars that make up the swing), and the band is zoomed to the swing.
    If total volume is 0, VAH/VAL/POC all equal the swing midpoint.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if swing_high <= swing_low or rows <= 0:
        raise ValueError(f"bad profile band: swing_low={swing_low} swing_high={swing_high} rows={rows}")
    row_size = (swing_high - swing_low) / rows
    counts = np.zeros(rows, dtype=float)
    for hi, lo, vol in zip(high, low, volume):
        tp = (hi + lo) / 2.0
        row = int((tp - swing_low) / row_size)
        row = min(max(row, 0), rows - 1)
        counts[row] += vol
    total = float(counts.sum())
    prices = swing_low + row_size * (np.arange(rows) + 0.5)
    if total <= 0:
        mid = (swing_high + swing_low) / 2.0
        return VolumeProfile(mid, mid, mid, row_size, rows, 0.0, prices)
    # VAH: scan rows top→bottom until cumulative >= vah_pct * total
    cum = 0.0
    vah_row = rows - 1
    for r in range(rows - 1, -1, -1):
        cum += counts[r]
        if cum >= vah_pct * total:
            vah_row = r
            break
        vah_row = r - 1
    cum = 0.0
    val_row = 0
    for r in range(rows):
        cum += counts[r]
        if cum >= vah_pct * total:
            val_row = r
            break
        val_row = r + 1
    poc_row = int(np.argmax(counts))
    vah = swing_low + row_size * (max(vah_row, 0) + 1)
    val = swing_low + row_size * (min(val_row, rows - 1))
    return VolumeProfile(vah, val, prices[poc_row], row_size, rows, total, prices)


# ---------------------------------------------------------------------------
# Fibonacci retracement levels
# ---------------------------------------------------------------------------
_FIB_RATIOS = (0.236, 0.382, 0.5, 0.58, 0.618)


def fib_levels(swing_low: float, swing_high: float, direction: str = "up") -> dict[str, float]:
    """Retracement levels across the swing ``[swing_low, swing_high]``.

    For an up swing (``direction="up"``) the levels give pull-back prices
    ``low + ratio*(high-low)``; for a down swing they give bounce prices
    ``high - ratio*(high-low)``.  Keys are the ratio strings
    (``"0.236"`` … ``"0.618"``); ``"0.58"`` is the spec's entry key.
    """
    rng = swing_high - swing_low
    out: dict[str, float] = {}
    for r in _FIB_RATIOS:
        if direction == "up":
            out[f"{r:.3f}"] = swing_low + r * rng
        else:
            out[f"{r:.3f}"] = swing_high - r * rng
    out["0.500"] = swing_low + 0.5 * rng if direction == "up" else swing_high - 0.5 * rng
    return out


# ---------------------------------------------------------------------------
# HTF structure / AMD classifier (documented heuristics)
# ---------------------------------------------------------------------------
def swing_structure(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> tuple[str, list[float], list[float]]:
    """Classify price structure from swing pivots.

    Compares the last two confirmed swing highs and the last two confirmed
    swing lows:

    * ``"uptrend"``  — both last swing high > previous swing high AND last
      swing low > previous swing low.
    * ``"downtrend"`` — both last swing high < previous swing high AND last
      swing low < previous swing low.
    * ``"ranging"``  — otherwise (or fewer than 2 pivots of a kind).

    Returns ``(label, swing_high_prices, swing_low_prices)``.
    """
    hi_idxs, lo_idxs = swing_pivots(high, low, left=left, right=right)
    highs = [float(high[i]) for i in hi_idxs]
    lows = [float(low[i]) for i in lo_idxs]
    label = "ranging"
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1] > highs[-2]
        hl = lows[-1] > lows[-2]
        lh = highs[-1] < highs[-2]
        ll = lows[-1] < lows[-2]
        if hh and hl:
            label = "uptrend"
        elif lh and ll:
            label = "downtrend"
    return label, highs, lows


def classify_amd(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> str:
    """Accumulation / Manipulation / Distribution classifier (heuristic).

    * ``"accumulation"``  — uptrend structure (HH/HL sequence).
    * ``"distribution"``  — downtrend structure (LH/LL sequence).
    * ``"manipulation"``  — the most recent confirmed swing low *breaks*
      below the previous swing low after an uptrend (a classic liquidity
      sweep), or the most recent swing high breaks above the previous swing
      high after a downtrend (a liquidation sweep).

    The classifier is deliberately simple and documented as a heuristic —
    it is not a substitute for tape-reading.
    """
    label, highs, lows = swing_structure(high, low, left=left, right=right)
    if label == "uptrend":
        return "accumulation"
    if label == "downtrend":
        return "distribution"
    # manipulation detection on break of prior swing extreme
    if len(lows) >= 2 and lows[-1] < lows[-2] and len(highs) >= 1:
        return "manipulation"
    if len(highs) >= 2 and highs[-1] > highs[-2] and len(lows) >= 1:
        return "manipulation"
    return label  # "ranging"


# ---------------------------------------------------------------------------
# SMT divergence (best-effort approximation, documented)
# ---------------------------------------------------------------------------
def smt_divergence(
    traded_now: float,
    traded_prev: float,
    pair_now: float,
    pair_prev: float,
    direction: str,
) -> bool:
    """SMT divergence check between the traded symbol and its correlated pair.

    Classic SMT (Smart-Money-Technique) divergence: the traded symbol makes a
    fresh extreme while the correlated pair refuses to make the same extreme,
    flagging a liquidity grab.

    * ``direction="long"`` — bullish divergence: ``traded_now < traded_prev``
      (new low) while ``pair_now >= pair_prev`` (pair holds).
    * ``direction="short"`` — bearish divergence: ``traded_now > traded_prev``
      (new high) while ``pair_now <= pair_prev`` (pair holds).

    Documented limitation: SMT is a forex-market concept where the "pair" is
    economically linked.  On US equities the correlation between e.g. QQQ and
    SPY is strong but not a fixed peg, so this filter is BEST-EFFORT — it is
    on by default for the ICT module but can be disabled via config, and it
    no-ops (passes) when pair data is unavailable.
    """
    if direction == "long":
        return traded_now < traded_prev and pair_now >= pair_prev
    return traded_now > traded_prev and pair_now <= pair_prev