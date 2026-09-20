"""Data layer for the ScalpSet historical replay (honest, no-lookahead).

The live main trader builds its strategy inputs from *yfinance* bars with a
fixed set of lookbacks (``SCALP_TF_SPEC`` in ``live_trader.py``): 3 days of
1m, 5 days of 5m, 10 days of 15m, 25 days of 30m, 60 days of 1h, a 4h frame
resampled from 1h, and 40 days of 1d.  This module reproduces exactly that
view of the market from the cached Alpaca 1m history in ``data/history/``:

* **RTH only.**  Yahoo's intraday frames are regular-hours only (no
  ``prepost``), so the frames here are built from 09:30-16:00 America/New_York
  1m bars only.  Extended-hours bars in the cache are dropped — they are never
  visible to the live strategies either.
* **Clock-aligned aggregation.**  5m/15m/30m bins are clock aligned (the RTH
  open 09:30 is a multiple of 5/15/30, so session alignment and clock
  alignment agree).  1h bins are anchored on the session open (09:30, 10:30,
  ...) which is what Yahoo's own 1h bars look like.  4h bins use wall-clock
  4-hour edges, matching ``df1h.resample("4h")`` in the live code.  1d bins are
  ET calendar sessions (RTH OHLC of the day).
* **In-progress bar included.**  Live fetches ``start=now - N days, end=now``,
  so the newest HTF bar in each frame is *partial* (the current bin aggregated
  up to the latest 1m close).  The replay rebuilds that partial bar at every
  1m step, so a module sees exactly the frame live would have seen.
* **Liquidity map.**  PDH/PDL come from the last-but-one daily bar (the live
  ``_build_liquidity`` rule); today's session high/low come from the 1m frame;
  the previous day's Asia/London windows are evaluated with the same
  ``session_extremes`` helper live uses (they are empty on RTH-only bars, which
  is what live gets from Yahoo too).

Everything here is deterministic, read-only and side-effect free apart from
reading the parquet cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.strategies.scalp.indicators import session_extremes
from src.strategies.scalp.types import (
    CandleFrame,
    LiquidityMap,
    ScalpContext,
    SessionWindow,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "history"

EXCHANGE_TZ = "America/New_York"
RTH_OPEN_MIN = 9 * 60 + 30  # 09:30 ET
RTH_CLOSE_MIN = 16 * 60  # 16:00 ET

#: Live lookbacks (calendar days) from ``SCALP_TF_SPEC`` in ``live_trader.py``.
LOOKBACK_DAYS = {
    "1m": 3,
    "5m": 5,
    "15m": 10,
    "30m": 25,
    "1h": 60,
    "4h": 60,
    "1d": 40,
}

#: Previous-day session windows used by the live liquidity map.
LIQUIDITY_WINDOWS = (
    SessionWindow("asia", "20:00", "24:00"),
    SessionWindow("london", "02:00", "05:00"),
)

#: Minutes-per-bin for the clock-aligned timeframes (1h is session-anchored,
#: 1d is one ET calendar day).
BIN_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "4h": 240}


def load_rth_1m(
    symbol: str,
    cache_dir: Path | str = DEFAULT_CACHE,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load every cached month for *symbol* and return RTH 1m bars.

    Returns a DataFrame with an ET-naive ``DatetimeIndex`` and columns
    ``open/high/low/close/volume`` sorted ascending, restricted to
    09:30 <= time < 16:00 exchange time.
    """
    cache = Path(cache_dir) / "1m" / symbol.upper()
    files = sorted(cache.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached 1m bars for {symbol} under {cache}")
    parts = [pd.read_parquet(p) for p in files]
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is None:  # pragma: no cover - cache is always tz-aware
        df.index = df.index.tz_localize("UTC")
    et = df.index.tz_convert(EXCHANGE_TZ)
    mod = et.hour * 60 + et.minute
    mask = (mod >= RTH_OPEN_MIN) & (mod < RTH_CLOSE_MIN)
    df = df.loc[mask, ["open", "high", "low", "close", "volume"]]
    idx = et[mask].tz_localize(None).tz_localize(None)
    df.index = pd.DatetimeIndex(idx, name="ts")
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index < pd.Timestamp(end)]
    return df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})


@dataclass
class _Binned:
    """Completed-bin aggregates for one timeframe, plus per-1m bin metadata."""

    key: str
    bin_id: np.ndarray  # per 1m bar: encoded bin identity
    start_idx: np.ndarray  # per 1m bar: index of the 1m bar that opened its bin
    n_completed: np.ndarray  # per 1m bar: number of COMPLETED bins before it
    ts: np.ndarray  # completed bin timestamps (datetime64[ns])
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    day: np.ndarray  # completed bin ET day ordinal (for lookback windows)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.ts)


class SymbolFrames:
    """Builds the exact ScalpContext frame set for one symbol at bar *i*."""

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        pair: Optional["SymbolFrames"] = None,
        pair_symbol: str = "QQQ",
    ) -> None:
        self.symbol = symbol.upper()
        self.ts = np.asarray(df.index.to_numpy(), dtype="datetime64[ns]")
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.v = df["volume"].to_numpy(dtype=float)
        self.n = len(df)
        self.pair = pair
        self.pair_symbol = pair_symbol
        # ET minute-of-day and day ordinal per bar (for binning + windows).
        et = pd.DatetimeIndex(self.ts)
        mod = et.hour.to_numpy() * 60 + et.minute.to_numpy()
        self.mod = mod.astype(np.int64)
        day = et.normalize()
        self.day = (day.view("int64") // (86_400 * 10**9)).astype(np.int64)
        self._bins: dict[str, _Binned] = {
            key: self._build_bins(key) for key in ("5m", "15m", "30m", "1h", "4h", "1d")
        }
        self._liq_cache: dict[int, tuple[Optional[float], Optional[float]]] = {}

    # ── binning ─────────────────────────────────────────────────────
    def _bin_keys(self, key: str) -> np.ndarray:
        """Per-1m-bar bin identity for timeframe *key*."""
        if key == "1d":
            return self.day
        if key == "1h":
            rel = self.mod - RTH_OPEN_MIN
            return self.day * 64 + (rel // 60)
        step = BIN_MINUTES[key]
        return self.day * 4096 + (self.mod // step)

    def _build_bins(self, key: str) -> _Binned:
        keys = self._bin_keys(key)
        if self.n == 0:
            # A symbol whose cache has no RTH bars in the requested window:
            # every frame is empty, no bins exist.  (No IndexError.)
            empty = np.array([], dtype="datetime64[ns]")
            return _Binned(key=key, bin_id=keys, start_idx=keys.copy(),
                           n_completed=keys.copy(), ts=empty, open=keys.copy(),
                           high=keys.copy(), low=keys.copy(), close=keys.copy(),
                           volume=keys.copy(), day=keys.copy())
        new = np.empty(self.n, dtype=bool)
        new[0] = True
        new[1:] = keys[1:] != keys[:-1]
        start_idx = np.maximum.accumulate(np.where(new, np.arange(self.n), 0))
        n_completed = np.cumsum(new) - 1  # completed bins strictly before this bar's bin
        starts = np.flatnonzero(new)
        ends = np.append(starts[1:], self.n)
        high = np.maximum.reduceat(self.h, starts)
        low = np.minimum.reduceat(self.l, starts)
        vol = np.add.reduceat(self.v, starts)
        return _Binned(
            key=key,
            bin_id=keys,
            start_idx=start_idx,
            n_completed=n_completed,
            ts=self.ts[starts],
            open=self.o[starts],
            high=high,
            low=low,
            close=self.c[ends - 1],
            volume=vol,
            day=self.day[starts],
        )

    # ── frame assembly ──────────────────────────────────────────────
    def _partial(self, key: str, i: int, b: _Binned):
        """Aggregate of the in-progress bin at bar *i* (live sees a partial bar)."""
        j = int(b.start_idx[i])
        return (
            self.ts[j],
            float(self.o[j]),
            float(self.h[j : i + 1].max()),
            float(self.l[j : i + 1].min()),
            float(self.c[i]),
            float(self.v[j : i + 1].sum()),
        )

    def frame_1m(self, i: int) -> CandleFrame:
        look = LOOKBACK_DAYS["1m"]
        lo_day = self.day[i] - look
        j0 = int(np.searchsorted(self.day, lo_day, side="left"))
        lo = min(j0, i)
        return CandleFrame(
            symbol=self.symbol,
            timeframe="1m",
            ts=self.ts[lo : i + 1],
            open=self.o[lo : i + 1],
            high=self.h[lo : i + 1],
            low=self.l[lo : i + 1],
            close=self.c[lo : i + 1],
            volume=self.v[lo : i + 1],
        )

    def frame(self, key: str, i: int) -> CandleFrame:
        """Completed bars in the live lookback window + the partial current bar."""
        b = self._bins[key]
        k = int(b.n_completed[i])
        lo_day = self.day[i] - LOOKBACK_DAYS[key]
        j0 = int(np.searchsorted(b.day, lo_day, side="left"))
        j0 = min(j0, max(k - 1, 0))
        ts_p, o_p, h_p, l_p, c_p, v_p = self._partial(key, i, b)
        ts = np.concatenate([b.ts[j0:k], np.array([ts_p], dtype=b.ts.dtype)])
        o = np.concatenate([b.open[j0:k], [o_p]])
        h = np.concatenate([b.high[j0:k], [h_p]])
        l = np.concatenate([b.low[j0:k], [l_p]])
        c = np.concatenate([b.close[j0:k], [c_p]])
        v = np.concatenate([b.volume[j0:k], [v_p]])
        return CandleFrame(
            symbol=self.symbol, timeframe=key, ts=ts, open=o, high=h, low=l, close=c, volume=v
        )

    def pair_1m(self, t: np.datetime64) -> Optional[CandleFrame]:
        """The correlated-pair 1m frame ending at (or just before) *t*."""
        if self.pair is None:
            return None
        j = int(np.searchsorted(self.pair.ts, t, side="right")) - 1
        if j < 0:
            return None
        return self.pair.frame_1m(j)

    def set_pair(self, pair: Optional["SymbolFrames"], pair_symbol: str = "QQQ") -> None:
        """Wire (or clear) the SMT correlated-pair frames for this symbol.

        Public API addition for the replay engine: symbols are usually built
        before their pair exists, so the pair is attached afterwards instead
        of at construction time.  Passing ``None`` disables the pair frame
        (the IFVG module then skips its best-effort SMT filter).
        """
        self.pair = pair
        self.pair_symbol = pair_symbol.upper()

    # ── liquidity map (live ``_build_liquidity`` semantics) ─────────
    def _prev_day_sessions(self, day: int) -> tuple[Optional[float], Optional[float]]:
        """(asia_high, asia_low, london_high, london_low) cache for *day*.

        Mirrors live: the previous ET day's 1m bars (from the 3-day frame) are
        scanned with ``session_extremes`` for each configured window.  Cached
        per day because the answer only changes once a session.
        """
        cached = self._liq_cache.get(day)
        if cached is not None:
            return cached
        out: list[Optional[float]] = [None, None, None, None]
        lo_day = day - LOOKBACK_DAYS["1m"]
        j0 = int(np.searchsorted(self.day, lo_day, side="left"))
        j1 = int(np.searchsorted(self.day, day, side="left"))
        if j1 > j0:
            prev = CandleFrame(
                symbol=self.symbol,
                timeframe="1m",
                ts=self.ts[j0:j1],
                open=self.o[j0:j1],
                high=self.h[j0:j1],
                low=self.l[j0:j1],
                close=self.c[j0:j1],
                volume=self.v[j0:j1],
            )
            for idx, w in enumerate(LIQUIDITY_WINDOWS):
                hi, lo = session_extremes(prev, w)
                out[idx * 2] = hi
                out[idx * 2 + 1] = lo
        res = (out[0], out[1], out[2], out[3])
        self._liq_cache[day] = res
        return res

    def liquidity(self, i: int, f1m: CandleFrame, f1d: CandleFrame) -> LiquidityMap:
        pdh = pdl = None
        if len(f1d) >= 2:
            pdh = float(f1d.high[-2])
            pdl = float(f1d.low[-2])
        liq = LiquidityMap(pdh=pdh, pdl=pdl, prev_day_high=pdh, prev_day_low=pdl)
        day = int(self.day[i])
        mask = self.day[self.day_i0(i) : i + 1] >= day  # today's bars inside the 1m window
        base = self.day_i0(i)
        if mask.any():
            hi = float(f1m.high[mask].max())
            lo = float(f1m.low[mask].min())
            liq = LiquidityMap(**{**liq.__dict__, "session_high": hi, "session_low": lo})
        a_hi, a_lo, l_hi, l_lo = self._prev_day_sessions(day)
        if a_hi is not None:
            liq = LiquidityMap(**{**liq.__dict__, "asia_high": a_hi, "asia_low": a_lo})
        if l_hi is not None:
            liq = LiquidityMap(**{**liq.__dict__, "london_high": l_hi, "london_low": l_lo})
        return liq

    def day_i0(self, i: int) -> int:
        """First 1m index of the 1m-frame window containing bar *i*."""
        lo_day = self.day[i] - LOOKBACK_DAYS["1m"]
        return min(int(np.searchsorted(self.day, lo_day, side="left")), i)

    # ── context ─────────────────────────────────────────────────────
    def context(self, i: int, timeframes: Iterable[str] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")) -> ScalpContext:
        frames: dict[str, CandleFrame] = {}
        for key in timeframes:
            frames[key] = self.frame_1m(i) if key == "1m" else self.frame(key, i)
        liq = self.liquidity(i, frames["1m"], frames["1d"]) if "1d" in frames and "1m" in frames else LiquidityMap()
        pair_frames: dict[str, CandleFrame] = {}
        if self.pair is not None:
            pf = self.pair_1m(self.ts[i])
            if pf is not None and len(pf) > 0:
                pair_frames["1m"] = pf
        return ScalpContext(
            symbol=self.symbol,
            frames=frames,
            liquidity=liq,
            tz=EXCHANGE_TZ,
            pair_frames=pair_frames,
            pair_symbol=self.pair_symbol,
        )


def load_symbol_frames(
    symbol: str,
    cache_dir: Path | str = DEFAULT_CACHE,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    return load_rth_1m(symbol, cache_dir=cache_dir, start=start, end=end)


def build_symbol_frames(
    symbol: str,
    cache_dir: Path | str = DEFAULT_CACHE,
    start: Optional[str] = None,
    end: Optional[str] = None,
    warmup_days: int = 75,
    pair: Optional[SymbolFrames] = None,
    pair_symbol: str = "QQQ",
) -> SymbolFrames:
    """Load cached bars and build the replay frame set for one symbol.

    ``start`` is the first TRADED timestamp; bars from
    ``start - warmup_days`` are loaded too so the live lookback windows (the
    1h frame alone reaches back 60 calendar days) are fully populated on the
    first traded bar — otherwise every module would sit behind its
    ``SCALP_MIN_BARS`` gate for the first days of the window.

    The engine restricts trading to its own ``trade_window``; this function
    only decides how much history exists.
    """
    load_start = None
    if start is not None:
        load_start = (pd.Timestamp(start) - pd.Timedelta(days=int(warmup_days))).strftime("%Y-%m-%d")
    df = load_rth_1m(symbol, cache_dir=cache_dir, start=load_start, end=end)
    return SymbolFrames(symbol.upper(), df, pair=pair, pair_symbol=pair_symbol)
