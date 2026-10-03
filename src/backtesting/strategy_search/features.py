"""Per-session features for the stage-1 strategy families.

Discipline this module exists to enforce (brief §4.2):

* **Indicators are computed per session.**  Round 1 of the turbo edge search
  found and fixed a modelling bug where indicators were computed across the
  overnight boundary, handing the first bars of a session a history the live
  trader never had.  Every rolling window here is applied inside a session
  group, so nothing leaks across the gap.  The only deliberate cross-session
  inputs are a *completed* prior session's high/low/close.
* **Everything a signal reads is a past value.**  A family may only build a
  signal from features at bar *t*; the engine fills it at bar *t+1*'s open.
  Features reindexed onto the global axis keep NaN where a symbol has no bar
  (never a forward-filled value that pretends a bar existed).
* Market-factor features (SPY) are shifted one bar when aligned onto a traded
  symbol, so a regime filter can never read the same bar's SPY close.

Prices are the cached RTH 1-minute bars, ET-naive, 09:30 <= t < 16:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from src.backtesting.scalp_data import load_rth_1m

#: The shared bar cache the turbo replay and the ScalpSet backtest both use.
DEFAULT_CACHE = Path("/home/team/shared/engine/data/history")

RTH_OPEN_MIN = 9 * 60 + 30
EOD_FLAT_MIN = 15 * 60 + 30

#: Feature names every symbol carries after :func:`build_features`.
FEATURE_NAMES = (
    "minute", "sess_min", "vwap", "ema_fast", "ema_slow", "slope", "z20",
    "atr", "atr_pct", "ret30", "rvol30", "sess_ret", "or30_hi", "or30_lo",
    "or60_hi", "or60_lo", "prev_hi", "prev_lo", "prev_close", "prev_ret",
    "sess_open", "atr_prev_abs", "atr_prev_pct",
)
#: The raw OHLCV columns :func:`build_features` passes through.
OHLCV_NAMES = ("open", "high", "low", "close", "volume")
#: **The declared feature set** (E4): a family may read these names and nothing
#: else.  Anything outside it raises rather than reading as an all-NaN column,
#: which used to turn a typo into a false "killed: too few trips".
DECLARED_FEATURES = frozenset(FEATURE_NAMES) | frozenset(OHLCV_NAMES)


def load_window(symbols: Iterable[str], start: str, end: str,
                cache_dir: Path | str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Cached RTH 1m bars for *symbols* over ``[start, end)``."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_rth_1m(sym, cache_dir=cache_dir, start=start, end=end)
        if len(df):
            out[sym.upper()] = df
    return out


def coverage(frames: Mapping[str, pd.DataFrame]) -> dict:
    """Sessions and bars per symbol, plus the months each symbol really has."""
    cov: dict = {}
    for sym, df in frames.items():
        days = pd.Index(df.index.normalize().unique())
        months = sorted({str(d)[:7] for d in days})
        cov[sym] = {"bars": int(len(df)), "sessions": int(len(days)),
                    "first": str(df.index[0]), "last": str(df.index[-1]),
                    "months": months}
    return cov


def _session_key(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.normalize(), index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All per-session features for one symbol's frame (index = ts)."""
    day = _session_key(df)
    o = df["open"]
    h = df["high"]
    low = df["low"]
    c = df["close"]
    v = df["volume"]
    minute = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index)
    sess_min = minute - RTH_OPEN_MIN                      # minutes since the open
    out = pd.DataFrame(index=df.index)
    out["minute"] = minute
    out["sess_min"] = sess_min
    for name in ("open", "high", "low", "close", "volume"):
        out[name] = df[name].to_numpy(dtype=float)

    tp = (h + low + c) / 3.0
    cum_vol = v.groupby(day).cumsum().replace(0.0, np.nan)
    out["vwap"] = (tp * v).groupby(day).cumsum() / cum_vol

    out["ema_fast"] = c.groupby(day).transform(
        lambda s: s.ewm(span=12, adjust=False).mean())
    out["ema_slow"] = c.groupby(day).transform(
        lambda s: s.ewm(span=36, adjust=False).mean())
    out["slope"] = out["ema_fast"].groupby(day).pct_change(5)

    def _z(s: pd.Series) -> pd.Series:
        m = s.rolling(20).mean()
        sd = s.rolling(20).std()
        return (s - m) / sd.replace(0.0, np.nan)

    out["z20"] = c.groupby(day).transform(_z)

    prev_close = c.groupby(day).shift(1)
    tr = pd.concat([h - low, (h - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    out["atr"] = tr.groupby(day).transform(lambda s: s.rolling(14).mean())
    out["atr_pct"] = out["atr"] / c

    out["ret30"] = c.groupby(day).pct_change(30)
    ret1 = c.groupby(day).pct_change()
    out["rvol30"] = ret1.groupby(day).transform(lambda s: s.rolling(30).std())

    sess_open = o.groupby(day).transform("first")
    out["sess_open"] = sess_open
    out["sess_ret"] = c / sess_open - 1.0

    for mins, hi_name, lo_name in ((30, "or30_hi", "or30_lo"), (60, "or60_hi", "or60_lo")):
        in_or = sess_min < mins
        out[hi_name] = h.where(in_or).groupby(day).transform("max")
        out[lo_name] = low.where(in_or).groupby(day).transform("min")

    sess = df.groupby(day).agg(hi=("high", "max"), lo=("low", "min"),
                               close=("close", "last"), open=("open", "first"))
    prev = sess.shift(1)
    out["prev_hi"] = day.map(prev["hi"]).to_numpy()
    out["prev_lo"] = day.map(prev["lo"]).to_numpy()
    out["prev_close"] = day.map(prev["close"]).to_numpy()
    out["prev_ret"] = day.map(prev["close"] / prev["open"] - 1.0).to_numpy()

    # ── the prior *completed* session's ATR(14) (R3-7: a causal normaliser) ──
    # ``atr`` above is the within-session ATR(14): it is NaN for the first 14
    # bars of every session, so it cannot normalise an opening signal (an
    # opening-gap or opening-range family would either skip the whole morning or
    # silently compare against a NaN — both are the round-1 failure class).
    # These two are the previous session's ATR **at its last bar**, i.e. a value
    # that is final at the prior close and never revised afterwards, which is
    # exactly the "causal, prior-session" normaliser every round-2 cell
    # declares.  ``atr_prev_pct`` divides by that session's close, so it is a
    # return-scale quantity the leverage of the symbol is already inside.
    sess_atr = out["atr"].groupby(day).last()
    sess_close_last = c.groupby(day).last()
    prev_atr = sess_atr.shift(1)
    prev_atr_close = sess_close_last.shift(1)
    out["atr_prev_abs"] = day.map(prev_atr).to_numpy()
    out["atr_prev_pct"] = (day.map(prev_atr) / day.map(prev_atr_close)).to_numpy()
    return out


@dataclass
class SymbolFeatures:
    symbol: str
    frame: pd.DataFrame
    feats: pd.DataFrame


class FeatureBook:
    """Features for every symbol, aligned onto the engine's global bar axis."""

    def __init__(self, frames: Mapping[str, pd.DataFrame],
                 cache_note: Optional[str] = None) -> None:
        self.frames = {s.upper(): f for s, f in frames.items()}
        self.features = {s: build_features(f) for s, f in self.frames.items()}
        self.cache_note = cache_note

    def align(self, axis: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        return {s: f.reindex(axis) for s, f in self.features.items()}

    def market_state(self, axis: pd.DatetimeIndex, symbol: str = "SPY") -> pd.DataFrame:
        """Market-factor features on *axis*, reindexed with ffill then **shifted**.

        The shift is the point: a filter asking "is SPY trending up?" at bar *t*
        must read SPY's value from bar *t−1* or earlier.  A bar with no SPY
        print keeps the previous known value, which is still a past value.
        """
        f = self.features[symbol].reindex(axis, method="ffill")
        return f.shift(1)
