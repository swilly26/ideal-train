"""Market-regime conditioning for adaptive parameters.

Fixed thresholds fail because the market is not stationary: a 3 % stop that
is generous on a quiet day gets eaten alive on a violent one.  This module
classifies every bar into a 2×2 market regime and lets the optimizer express
parameters as *functions of regime* instead of a single magic number.

Regimes (per bar, computed from the symbol's own OHLCV):

* **trend state** — ``up`` if close >= SMA(``ma_period``), else ``down``
  (mirrors the live traders' regime gate: price vs short MA).
* **volatility state** — ``volatile`` if ATR% (ATR/close) is above its
  rolling median over the last ``vol_lookback`` bars, else ``calm``.

The four combinations: ``calm_up``, ``calm_down``, ``volatile_up``,
``volatile_down``.

The output of the optimizer is then per-regime parameter sets plus a
volatility-scaling rule (stop/TP proportional to ATR%), giving the trader a
concrete runtime selection path.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = ["calm_up", "calm_down", "volatile_up", "volatile_down"]

TREND_UP = "up"
TREND_DOWN = "down"
VOL_CALM = "calm"
VOL_VOLATILE = "volatile"


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (simple mean of true range over *period* bars).

    True range = max(high - low, |high - prev_close|, |low - prev_close|).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR as a fraction of price (``atr / close``)."""
    return atr(high, low, close, period) / close


def classify_regime(
    df: pd.DataFrame,
    atr_period: int = 14,
    ma_period: int = 10,
    vol_lookback: int = 60,
) -> pd.Series:
    """Classify every bar of *df* into a 2×2 regime label.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV bars indexed by timestamp with ``high``, ``low``, ``close``.
    atr_period : int
        ATR window (bars).
    ma_period : int
        Trend SMA window (bars).
    vol_lookback : int
        Rolling window over which the ATR% median defines "calm" vs
        "volatile".

    Returns
    -------
    pd.Series of str
        One regime label per bar, aligned to ``df.index``.  Bars before the
        indicators are warm are labelled ``NaN``.
    """
    if df is None or df.empty:
        return pd.Series(dtype=object)

    close = df["close"]
    ap = atr_pct(df["high"], df["low"], close, period=atr_period)
    ma = close.rolling(window=ma_period).mean()

    # Volatility baseline: rolling median of ATR% over the lookback window
    # (NaN-safe: median of the window including the current bar).
    vol_baseline = ap.rolling(window=vol_lookback, min_periods=atr_period).median()
    vol_state = np.where(ap > vol_baseline, VOL_VOLATILE, VOL_CALM)

    trend_state = np.where(close >= ma, TREND_UP, TREND_DOWN)

    regime = np.where(
        vol_state == VOL_VOLATILE,
        np.where(trend_state == TREND_UP, "volatile_up", "volatile_down"),
        np.where(trend_state == TREND_UP, "calm_up", "calm_down"),
    )
    out = pd.Series(regime, index=df.index, dtype=object)
    out[ap.isna() | ma.isna() | vol_baseline.isna()] = np.nan
    return out


def join_trades_to_regime(
    trades: pd.DataFrame,
    price_by_symbol: dict[str, pd.DataFrame],
    regime_by_symbol: dict[str, pd.Series] | None = None,
    **regime_kwargs,
) -> pd.DataFrame:
    """Add a ``regime`` column to *trades* by matching each trade's entry
    time to the regime active on that symbol's bars.

    For each trade, the *last bar at or before* ``exit_dt`` (entry time in
    the realized logs) is located in the symbol's price frame; its regime
    label is attached.  Trades without a matching symbol/bar get ``None``.

    Parameters
    ----------
    trades : pd.DataFrame
        Realized/backtest trades with ``symbol`` and ``exit_dt`` columns
        (for backtest trades use ``entry_time`` — see note).
    price_by_symbol : dict[str, pd.DataFrame]
        OHLCV frames keyed by symbol (upper-case).
    regime_by_symbol : dict[str, pd.Series], optional
        Precomputed regime series per symbol.  When absent, they are
        computed on the fly with *regime_kwargs*.
    **regime_kwargs
        Passed to :func:`classify_regime`.

    Returns
    -------
    pd.DataFrame
        Copy of *trades* with the added ``regime`` column.
    """
    if trades is None or trades.empty:
        return trades.copy()

    if regime_by_symbol is None:
        regime_by_symbol = {
            sym: classify_regime(df, **regime_kwargs)
            for sym, df in price_by_symbol.items()
        }

    out = trades.copy()
    regimes: list[str | None] = []
    for _, row in out.iterrows():
        sym = str(row.get("symbol", "")).upper()
        ts_key = row.get("exit_dt")
        if ts_key is None or pd.isna(ts_key) or sym not in price_by_symbol:
            regimes.append(None)
            continue
        regime_series = regime_by_symbol[sym]
        price_idx = price_by_symbol[sym].index
        stamped = pd.to_datetime(pd.Series(price_idx), errors="coerce")
        # Normalize tz so we can compare against the (usually naive) trade
        # timestamps regardless of whether the price index is tz-aware.
        if getattr(stamped.dt, "tz", None) is not None:
            stamped = stamped.dt.tz_localize(None)
        ts_cmp = pd.Timestamp(ts_key)
        if ts_cmp.tz is not None:
            ts_cmp = ts_cmp.tz_localize(None)
        # Last bar at or before the trade time.
        le = stamped[stamped <= ts_cmp]
        if le.empty:
            regimes.append(None)
            continue
        bar_ts = le.iloc[-1]
        regimes.append(regime_series.loc[bar_ts] if not pd.isna(regime_series.loc[bar_ts]) else None)

    out["regime"] = regimes
    return out


def regime_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-regime performance (n, profit factor, win rate, avg pct, net pnl).

    Returns
    -------
    pd.DataFrame indexed by regime with the columns above, rows for regimes
    that never occurred omitted.
    """
    from src.optimization.realized import realized_metrics

    if trades is None or trades.empty or "regime" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for regime, group in trades.groupby("regime"):
        if regime is None or (isinstance(regime, float) and np.isnan(regime)):
            continue
        m = realized_metrics(group)
        rows.append(
            {
                "regime": regime,
                "n_trades": m["n_trades"],
                "profit_factor": m["profit_factor"],
                "win_rate": m["win_rate"],
                "avg_pct": m["avg_pct"],
                "net_pnl": m["net_pnl"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["regime", "n_trades", "profit_factor", "win_rate", "avg_pct", "net_pnl"])
    return pd.DataFrame(rows).set_index("regime").sort_values("n_trades", ascending=False)


# ---------------------------------------------------------------------------
# Volatility-scaled stop / take-profit
# ---------------------------------------------------------------------------

def scale_stop_tp(
    base_stop_pct: float,
    base_tp_pct: float,
    atr_pct_value: float,
    ref_atr_pct: float,
    min_stop_pct: float = 0.01,
    max_stop_pct: float = 0.12,
    min_tp_pct: float = 0.015,
    max_tp_pct: float = 0.25,
    max_scale: float = 3.0,
) -> tuple[float, float]:
    """Scale stop/TP by the current ATR% relative to a reference ATR%.

    The idea: a stop should be *multiples of the instrument's own noise*, not
    a fixed percentage.  When the market is twice as volatile, the stop and
    take-profit are scaled up so the same *statistical* move is captured and
    ordinary noise doesn't trigger the stop.

    Parameters
    ----------
    base_stop_pct : float
        Reference stop fraction at the reference volatility (e.g. 0.03).
    base_tp_pct : float
        Reference take-profit fraction (e.g. 0.04).
    atr_pct_value : float
        Current ATR% (ATR/close, e.g. 0.02 for 2 % per bar).
    ref_atr_pct : float
        Reference ATR% the base stop/TP were designed for (e.g. the
        full-window median ATR%).
    min_stop_pct / max_stop_pct : float
        Clamp bounds for the scaled stop.
    min_tp_pct / max_tp_pct : float
        Clamp bounds for the scaled take-profit.
    max_scale : float
        Absolute cap on the scaling factor (protects against a single wild
        ATR spike producing absurd stops).

    Returns
    -------
    (stop_pct, tp_pct) : tuple[float, float]
    """
    if ref_atr_pct <= 0 or not np.isfinite(ref_atr_pct):
        return float(base_stop_pct), float(base_tp_pct)
    scale = float(atr_pct_value / ref_atr_pct)
    scale = min(max(scale, 1.0 / max_scale), max_scale)
    stop = float(np.clip(base_stop_pct * scale, min_stop_pct, max_stop_pct))
    tp = float(np.clip(base_tp_pct * scale, min_tp_pct, max_tp_pct))
    return stop, tp