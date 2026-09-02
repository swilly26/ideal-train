"""Tests for regime classification, trade joining, and ATR scaling."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.regime import (
    REGIMES,
    atr,
    atr_pct,
    classify_regime,
    join_trades_to_regime,
    regime_summary,
    scale_stop_tp,
)


def _mean_reverting_ohlcv(n_bars: int = 400, seed: int = 7, slope: float = 0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars)
    close = 100.0 + slope * t + 4.0 * np.sin(t / 12.0) + rng.normal(0, 0.4, n_bars)
    idx = pd.date_range("2026-07-01 09:30", periods=n_bars, freq="5min")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": np.full(n_bars, 1000),
        },
        index=idx,
    )
    return df


def test_atr_positive_and_bounded():
    df = _mean_reverting_ohlcv()
    a = atr(df["high"], df["low"], df["close"], period=14)
    assert a.notna().sum() > 0
    assert (a.dropna() > 0).all()
    ap = atr_pct(df["high"], df["low"], df["close"], period=14)
    assert (ap.dropna() > 0).all()
    assert (ap.dropna() < 0.05).all()  # sane magnitude


def test_classify_regime_labels():
    df = _mean_reverting_ohlcv()
    regime = classify_regime(df)
    assert len(regime) == len(df)
    known = set(REGIMES)
    values = set(regime.dropna().unique())
    assert values <= known
    assert len(values) > 0


def test_classify_regime_uptrend_is_up():
    # A sustained strong uptrend: close vs MA10 is positive nearly every bar.
    n = 400
    rng = np.random.default_rng(3)
    t = np.arange(n)
    close = 100.0 + 0.2 * t + 0.5 * np.sin(t / 12.0) + rng.normal(0, 0.15, n)
    idx = pd.date_range("2026-07-01 09:30", periods=n, freq="5min")
    df = pd.DataFrame({"open": close, "high": close + 0.4, "low": close - 0.4,
                       "close": close, "volume": 1000}, index=idx)
    regime = classify_regime(df, ma_period=10)
    up_share = regime.dropna().str.endswith("_up").mean()
    assert up_share > 0.8


def test_join_trades_to_regime_attaches_labels():
    df = _mean_reverting_ohlcv()
    trades = pd.DataFrame(
        {
            "symbol": ["SYM", "SYM"],
            "exit_dt": [df.index[100], df.index[300]],
        }
    )
    joined = join_trades_to_regime(trades, {"SYM": df})
    assert "regime" in joined.columns
    assert joined.iloc[0]["regime"] in REGIMES
    assert joined.iloc[1]["regime"] in REGIMES


def test_join_trades_missing_symbol_gets_none():
    df = _mean_reverting_ohlcv()
    trades = pd.DataFrame({"symbol": ["OTHER"], "exit_dt": [df.index[5]]})
    joined = join_trades_to_regime(trades, {"SYM": df})
    assert joined.iloc[0]["regime"] is None


def test_regime_summary_groups():
    df = _mean_reverting_ohlcv()
    trades = pd.DataFrame(
        {
            "symbol": ["SYM"] * 12,
            "exit_dt": list(df.index[50:62]),
            "pnl": [10.0, 5.0, -3.0, 7.0, 4.0, -2.0, 6.0, -1.0, 8.0, 3.0, -4.0, 2.0],
        }
    )
    joined = join_trades_to_regime(trades, {"SYM": df})
    summary = regime_summary(joined)
    assert not summary.empty
    assert summary.index.name == "regime"
    assert summary["n_trades"].sum() == 12
    assert (summary["profit_factor"] >= 0).all()


def test_scale_stop_tp():
    stop, tp = scale_stop_tp(0.03, 0.04, atr_pct_value=0.010, ref_atr_pct=0.010)
    assert stop == pytest.approx(0.03)
    assert tp == pytest.approx(0.04)

    # 2× volatility → stop and TP roughly double (within clamps).
    stop2, tp2 = scale_stop_tp(0.03, 0.04, atr_pct_value=0.020, ref_atr_pct=0.010)
    assert stop2 == pytest.approx(0.06)
    assert tp2 == pytest.approx(0.08)

    # Clamped at the bounds.
    stop_c, _ = scale_stop_tp(0.03, 0.04, atr_pct_value=0.5, ref_atr_pct=0.010)
    assert stop_c <= 0.12

    # Bad reference degenerates to base values.
    stop_bad, tp_bad = scale_stop_tp(0.03, 0.04, atr_pct_value=0.01, ref_atr_pct=0.0)
    assert stop_bad == pytest.approx(0.03)
    assert tp_bad == pytest.approx(0.04)