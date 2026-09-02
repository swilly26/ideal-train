"""Tests for the adaptive walk-forward optimizer end-to-end pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.adaptive import (
    AdaptiveOptimizer,
    AdaptiveResult,
    _slice_by_days,
    build_grid_combinations,
    config_from_kwargs,
)
from src.optimization.walk_forward import split_is_oos_holdout
from src.strategies.base import StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy


def _market_index(days: int = 22, bars_per_day: int = 78) -> pd.DatetimeIndex:
    """Synthetic intraday index: *days* unique trading days × *bars_per_day*
    5-min bars (09:30–16:00, the standard US market session)."""
    parts = []
    for d in range(days):
        day = pd.Timestamp("2026-07-01 09:30") + pd.Timedelta(days=d)
        parts.append(day + pd.timedelta_range(0, periods=bars_per_day, freq="5min"))
    return pd.DatetimeIndex(np.concatenate(parts))


def _mean_reverting_ohlcv(n_bars: int, seed: int, symbol: str):
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars)
    close = 100.0 + 4.0 * np.sin(t / 12.0) + rng.normal(0, 0.4, n_bars)
    idx = _market_index(days=1, bars_per_day=n_bars)[:n_bars] if n_bars <= 78 \
        else _market_index(days=int(np.ceil(n_bars / 78)), bars_per_day=78)[:n_bars]
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": np.full(n_bars, 1000),
        },
        index=idx,
    )


def _price_by_symbol(days: int = 22, bars_per_day: int = 78):
    """Three symbols over *days* unique trading days."""
    total = days * bars_per_day
    idx = _market_index(days=days, bars_per_day=bars_per_day)
    out = {}
    for i, sym in enumerate(["AAA", "BBB", "CCC"]):
        rng = np.random.default_rng(100 + i)
        t = np.arange(total)
        close = 100.0 + 4.0 * np.sin(t / 12.0) + rng.normal(0, 0.4, total)
        out[sym] = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.6,
                "low": close - 0.6,
                "close": close,
                "volume": np.full(total, 1000),
            },
            index=idx,
        )
    return out


def _split_for(frames):
    index = pd.DatetimeIndex(
        sorted({ts.normalize() for df in frames.values() for ts in df.index})
    )
    return split_is_oos_holdout(index, is_frac=0.5, oos_frac=0.25)


def test_build_grid_combinations():
    grid = {"entry_threshold": [0.4, 0.5], "stop_loss_pct": [0.02, 0.04], "lookback": [10, 20]}
    combos = build_grid_combinations(grid)
    assert len(combos) == 8
    assert {"entry_threshold": 0.5, "stop_loss_pct": 0.04, "lookback": 20} in combos


def test_build_grid_combinations_n_iter_truncates():
    grid = {"entry_threshold": [0.4, 0.5, 0.6], "stop_loss_pct": [0.02, 0.04, 0.06]}
    combos = build_grid_combinations(grid, n_iter=4)
    assert len(combos) == 4


def test_build_grid_combinations_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown grid keys"):
        build_grid_combinations({"entry_threshol": [0.5]})  # typo


def test_build_grid_combinations_empty():
    assert build_grid_combinations({}) == [{}]


def test_config_from_kwargs_preserves_base_and_extra():
    base = StrategyConfig(
        entry_threshold=0.5, exit_threshold=0.1, stop_loss_pct=0.03,
        take_profit_pct=0.03, max_position_pct=0.15,
        extra={"lookback": 20, "std_dev_multiplier": 2.0},
    )
    cfg = config_from_kwargs(
        {"entry_threshold": 0.8, "lookback": 10}, "NVDA", base=base
    )
    assert cfg.entry_threshold == pytest.approx(0.8)
    assert cfg.stop_loss_pct == pytest.approx(0.03)  # carried from base
    assert cfg.exit_threshold == pytest.approx(0.1)
    assert cfg.extra["lookback"] == 10
    assert cfg.extra["symbol"] == "NVDA"
    assert cfg.extra["std_dev_multiplier"] == 2.0


def test_config_from_kwargs_default_base():
    cfg = config_from_kwargs({}, "QQQ")
    assert cfg.extra["symbol"] == "QQQ"
    assert cfg.entry_threshold == pytest.approx(0.5)


def test_split_is_built_on_days_applied_to_bar_frames():
    """The split is defined on unique trading days; slicing a bar-level frame
    by its day masks must yield ALL bars of the window's days (regression:
    misapplying day positions to bar frames silently shrank windows to a
    handful of bars and starved every candidate of trades).  Also covers
    tz-aware indexes (real market data is America/New_York)."""
    frames = _price_by_symbol(days=6, bars_per_day=78)
    # Make indexes tz-aware to mimic real yfinance data.
    for sym, df in frames.items():
        frames[sym] = df.tz_localize("America/New_York")
    split = _split_for(frames)
    df0 = frames["AAA"]
    for win, days_attr in (("is", "is_days"), ("oos", "oos_days"), ("holdout", "holdout_days")):
        days = getattr(split, days_attr)
        window = _slice_by_days(df0, days)
        assert len(window) == len(days) * 78, (win, len(window))
        assert window.index.normalize().nunique() == len(days)


def test_adaptive_optimizer_end_to_end():
    frames = _price_by_symbol(days=22, bars_per_day=78)
    split = _split_for(frames)
    assert len(split.is_days) > 0 and len(split.oos_days) > 0 and len(split.holdout_days) > 0

    base = StrategyConfig(
        entry_threshold=0.5,
        exit_threshold=0.1,
        stop_loss_pct=0.03,
        take_profit_pct=0.04,
        max_position_pct=0.15,
        extra={"lookback": 20, "std_dev_multiplier": 2.0},
    )
    opt = AdaptiveOptimizer(min_trades=6, top_k=4)
    result = opt.optimize(
        MeanReversionStrategy,
        frames,
        split,
        {
            "entry_threshold": [0.4, 0.6],
            "stop_loss_pct": [0.02, 0.05],
            "take_profit_pct": [0.03, 0.06],
            "lookback": [15, 25],
        },
        n_iter=32,
        base_config=base,
        name="test_mr",
    )
    assert isinstance(result, AdaptiveResult)
    assert not result.all_scores.empty
    # Ranked table contains the baseline row.
    assert "BASELINE (current fixed)" in result.rankings["config"].tolist()
    for col in ("grade", "is_score", "oos_score", "degradation", "holdout_score"):
        assert col in result.rankings.columns
    # Grades are from the allowed alphabet.
    assert set(result.rankings["grade"].unique()) <= {"S", "A", "B", "C", "F"}
    # IS scores are never NaN.
    assert result.all_scores["is_score"].notna().all()
    # Regime output either produced configs or honestly reported nothing.
    assert isinstance(result.regime_configs, dict)
    assert isinstance(result.suggestions, dict)


def test_adaptive_optimizer_poison_grid_never_crashes():
    """Unknown grid keys must fail loudly before any backtest runs."""
    frames = _price_by_symbol(days=10, bars_per_day=20)
    split = _split_for(frames)
    opt = AdaptiveOptimizer(min_trades=4)
    with pytest.raises(ValueError, match="Unknown grid keys"):
        opt.optimize(
            MeanReversionStrategy,
            frames,
            split,
            {"profit_magic": [1.0]},
            n_iter=8,
        )


def test_adaptive_optimizer_empty_data_is_honest():
    """No data → every candidate reports -inf and the verdict is F, not a
    phantom 'best config'."""
    split = _split_for(_price_by_symbol(days=6, bars_per_day=10))
    opt = AdaptiveOptimizer(min_trades=4, top_k=3)
    result = opt.optimize(
        MeanReversionStrategy,
        {},
        split,
        {"entry_threshold": [0.4, 0.5]},
        n_iter=6,
    )
    assert (result.all_scores["is_score"] == float("-inf")).all()
    assert result.best_config_kwargs == {}
    assert result.best_grade == "F"