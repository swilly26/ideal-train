"""Autonomous walk-forward parameter optimizer with regime-adaptive output.

This is the focused deliverable for replacing fixed strategy thresholds with
market-adaptive tuning that generalizes.  The pipeline:

1. **Substrate** — per-symbol OHLCV price history (the available historical
   price cache) plus the realized per-trade P&L of the live traders.
2. **Walk-forward split** — chronological IS / OOS / untouched-holdout
   windows over unique trading days (``src.optimization.walk_forward``).
3. **Systematic search on IS only** — a few hundred parameter combos are
   backtested on the in-sample window; no OOS bar is ever seen during search.
4. **OOS validation + degradation grading** — the top-K IS candidates are
   re-scored on the OOS window; each is graded S/A/B/C/F by *both* its
   absolute OOS profit factor and its IS→OOS degradation.  A set that is
   great in-sample but collapses out-of-sample is graded F (discarded).
5. **Untouched holdout** — the newest window is scored only for the selected
   candidate and reported (never used for selection).
6. **Regime-adaptive output** — per-regime parameter sets (2×2
   trend×volatility) plus an ATR-scaled stop/TP rule, written to paths the
   traders can consume.

Everything new is off by default in live traders; this module only *produces*
configs and evidence — it does not change live behavior.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.optimization.degradation import GradeThresholds, grade_config, grade_table
from src.optimization.regime import (
    REGIMES,
    atr_pct,
    classify_regime,
    join_trades_to_regime,
    regime_summary,
    scale_stop_tp,
)
from src.optimization.realized import realized_metrics, trade_objective
from src.optimization.walk_forward import Split
from src.strategies.base import Strategy, StrategyConfig

logger = logging.getLogger(__name__)

# StrategyConfig fields the grid may tune (plus ``lookback`` which is an
# ``extra`` key in the current strategy implementations).
FIELD_KEYS = {
    "entry_threshold", "exit_threshold", "stop_loss_pct",
    "take_profit_pct", "max_position_pct",
}
EXTRA_KEYS = {"lookback", "std_dev_multiplier"}


def build_grid_combinations(
    param_grid: dict[str, list[float]],
    n_iter: int = 300,
) -> list[dict[str, float]]:
    """Expand *param_grid* into a list of candidate kwargs dicts.

    Keeps breadth manageable: combinations beyond *n_iter* are truncated in
    a stable (first-listed-values) order.  Unknown keys raise ``ValueError``
    so a typo'd grid can never silently do nothing.
    """
    known = FIELD_KEYS | EXTRA_KEYS
    unknown = set(param_grid) - known
    if unknown:
        raise ValueError(
            f"Unknown grid keys: {sorted(unknown)}. Supported: {sorted(known)}"
        )
    if not param_grid:
        return [{}]
    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos[:n_iter]]


def config_from_kwargs(
    kwargs: dict[str, float],
    symbol: str,
    base: StrategyConfig | None = None,
) -> StrategyConfig:
    """Build a ``StrategyConfig`` for a single-symbol backtest from grid
    kwargs, preserving base values for every key not in *kwargs*."""
    cfg = base or StrategyConfig()
    field_kwargs = {k: v for k, v in kwargs.items() if k in FIELD_KEYS}
    extra = dict(cfg.extra)
    for k, v in kwargs.items():
        if k in EXTRA_KEYS:
            extra[k] = v
    extra["symbol"] = symbol
    new_cfg = StrategyConfig(**field_kwargs, extra=extra)
    # Carry over any fields not overridden by the grid.
    for f in FIELD_KEYS:
        if f not in field_kwargs:
            setattr(new_cfg, f, getattr(cfg, f))
    return new_cfg


@dataclass
class AdaptiveResult:
    """Full output of an :class:`AdaptiveOptimizer` run.

    Attributes
    ----------
    strategy_name : str
        Registry name of the strategy searched.
    split : Split
        The chronological IS/OOS/holdout split used.
    all_scores : pd.DataFrame
        Full IS-search table (config kwargs → IS score → IS trade count).
    rankings : pd.DataFrame
        Top-K candidates + baseline, each graded (IS score, OOS score,
        degradation, grade, holdout score).
    best_config_kwargs : dict
        Winning config kwargs (empty if nothing passed).
    best_grade : str
        Grade of the winner (``"F"`` if nothing robust found).
    baseline : dict
        Metrics of the currently-deployed fixed config on IS/OOS/holdout.
    regime_summary : pd.DataFrame
        Per-regime performance of the winning config's trades.
    regime_configs : dict[str, dict]
        Per-regime ``StrategyConfig`` kwargs + ATR-scaled stops.
    suggestions : dict
        Human-readable adaptive rules and apply notes.
    """

    strategy_name: str
    split: Split
    all_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_config_kwargs: dict = field(default_factory=dict)
    best_grade: str = "F"
    baseline: dict = field(default_factory=dict)
    regime_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_configs: dict[str, dict] = field(default_factory=dict)
    suggestions: dict = field(default_factory=dict)


def _slice_by_days(df: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """Return the rows of *df* whose (normalized) trading day is in *days*.

    Timezone-safe: the frame index may be tz-aware (real market data is
    ``America/New_York``) while the split days are naive ISO date strings.
    """
    if df is None or df.empty or not days:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    ts_days = {pd.Timestamp(d).normalize() for d in days}
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    mask = idx.normalize().isin(ts_days)
    return df[mask]


class AdaptiveOptimizer:
    """Walk-forward optimizer with degradation grading and regime output.

    Parameters
    ----------
    engine : BacktestEngine, optional
        Reused backtest engine (default: zero-commission, $100k).
    objective_fn : callable, optional
        Scores a pooled trade DataFrame (default
        :func:`src.optimization.realized.trade_objective`).
    min_trades : int
        Minimum pooled trades for a finite score (default 8).
    top_k : int
        How many IS-ranked candidates to carry into OOS validation (default 5).
    grade_thresholds : GradeThresholds, optional
        Grading cut-offs (default ``DEFAULT_THRESHOLDS``).
    """

    def __init__(
        self,
        engine: BacktestEngine | None = None,
        objective_fn: Callable[[pd.DataFrame, int], float] | None = None,
        *,
        min_trades: int = 8,
        top_k: int = 5,
        grade_thresholds: GradeThresholds | None = None,
    ) -> None:
        self._engine = engine or BacktestEngine()
        self.objective_fn = objective_fn or trade_objective
        self.min_trades = min_trades
        self.top_k = max(1, top_k)
        self.grade_thresholds = grade_thresholds or GradeThresholds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy_cls: type[Strategy],
        price_by_symbol: dict[str, pd.DataFrame],
        split: Split,
        param_grid: dict[str, list[float]],
        *,
        n_iter: int = 300,
        base_config: StrategyConfig | None = None,
        name: str = "strategy",
    ) -> AdaptiveResult:
        """Run the full walk-forward + grading + regime pipeline.

        Parameters
        ----------
        strategy_cls : type[Strategy]
            Strategy class to optimize (instantiated per candidate).
        price_by_symbol : dict[str, pd.DataFrame]
            OHLCV frames for every symbol, indexed by timestamp.
        split : Split
            IS/OOS/holdout partition (see :func:`walk_forward.split_is_oos_holdout`).
        param_grid : dict[str, list[float]]
            Grid keys: ``entry_threshold``, ``exit_threshold``,
            ``stop_loss_pct``, ``take_profit_pct``, ``max_position_pct``,
            ``lookback``.
        n_iter : int
            Max candidate evaluations (breadth, not depth — a few hundred is
            the intended scale).
        base_config : StrategyConfig, optional
            Config to start every candidate from (defaults to
            ``StrategyConfig()``).  The optimizer also scores this exact
            config as the **baseline** for comparison with the live defaults.
        name : str
            Label for logging/reports.

        Returns
        -------
        AdaptiveResult
        """
        combos = build_grid_combinations(param_grid, n_iter=n_iter)
        logger.info(
            "%s: walk-forward search over %d combos on %d symbols "
            "(IS days=%d, OOS days=%d, holdout days=%d)",
            name, len(combos), len(price_by_symbol),
            len(split.is_days), len(split.oos_days), len(split.holdout_days),
        )

        # Precompute per-symbol window slices once.  The split is defined on
        # unique trading days, so we slice each symbol's bar frame by the
        # day lists (never by raw bar positions — symbols have different
        # bars-per-day and the split's index is day-level).
        slices = {
            sym: {
                win: _slice_by_days(df, getattr(split, win + "_days"))
                for win in ("is", "oos", "holdout")
            }
            for sym, df in price_by_symbol.items()
        }

        # ---- 1. IS search -------------------------------------------
        is_rows: list[dict] = []
        for kwargs in combos:
            pooled = self._pool_trades(
                strategy_cls, kwargs, slices, "is", base_config=base_config
            )
            score = self.objective_fn(pooled, self.min_trades)
            is_rows.append(
                {
                    "config": _label(kwargs),
                    "kwargs": kwargs,
                    "is_score": score,
                    "is_n_trades": len(pooled),
                }
            )
        all_scores = pd.DataFrame(is_rows).sort_values("is_score", ascending=False).reset_index(drop=True)

        # ---- 2. Baseline: the current fixed config ------------------
        base_kwargs: dict[str, float] = {}
        baseline = self._score_candidate(
            strategy_cls, base_kwargs, slices, base_config=base_config
        )

        # ---- 3. Top-K → OOS validation + grading --------------------
        finite = all_scores[all_scores["is_score"] > float("-inf")]
        candidates = finite.head(self.top_k) if not finite.empty else all_scores.head(0)

        ranked_rows: list[dict] = []
        for _, row in candidates.iterrows():
            eval_row = self._score_candidate(
                strategy_cls, row["kwargs"], slices, base_config=base_config
            )
            grade = grade_config(
                row["is_score"], eval_row["oos_score"],
                eval_row["oos_n_trades"], self.grade_thresholds,
            )
            ranked_rows.append(
                {
                    "config": row["config"],
                    "kwargs": row["kwargs"],
                    "is_score": row["is_score"],
                    "is_n_trades": row["is_n_trades"],
                    "oos_score": eval_row["oos_score"],
                    "oos_n_trades": eval_row["oos_n_trades"],
                    "degradation": grade.degradation,
                    "grade": grade.letter,
                    "passed": grade.passed,
                    "reason": grade.reason,
                    "holdout_score": eval_row["holdout_score"],
                    "holdout_n_trades": eval_row["holdout_n_trades"],
                }
            )
        # Baseline always appears for comparison.
        base_grade = grade_config(
            baseline["is_score"], baseline["oos_score"],
            baseline["oos_n_trades"], self.grade_thresholds,
        )
        ranked_rows.append(
            {
                "config": "BASELINE (current fixed)",
                "kwargs": {},
                "is_score": baseline["is_score"],
                "is_n_trades": baseline["is_n_trades"],
                "oos_score": baseline["oos_score"],
                "oos_n_trades": baseline["oos_n_trades"],
                "degradation": base_grade.degradation,
                "grade": base_grade.letter,
                "passed": base_grade.passed,
                "reason": base_grade.reason,
                "holdout_score": baseline["holdout_score"],
                "holdout_n_trades": baseline["holdout_n_trades"],
            }
        )
        rankings = grade_table(
            ranked_rows, thresholds=self.grade_thresholds,
        )
        label_to_kwargs = {r["config"]: r["kwargs"] for r in ranked_rows}

        # ---- 4. Select best graded candidate -------------------------
        best_kwargs: dict = {}
        best_letter = "F"
        non_baseline = rankings[rankings["config"] != "BASELINE (current fixed)"]
        if not non_baseline.empty:
            winners = non_baseline[non_baseline["grade"].isin(("S", "A"))]
            if winners.empty:
                # Nothing passed: report the least-bad (best OOS PF) without
                # recommending adoption — grade stays as computed.
                sel = non_baseline.sort_values("oos_score", ascending=False).iloc[0]
            else:
                sel = winners.sort_values(
                    ["grade", "oos_score"],
                    key=lambda s: s.map({"S": 0, "A": 1}) if s.name == "grade" else s,
                ).iloc[0]
            best_kwargs = label_to_kwargs.get(sel["config"], {})
            best_letter = sel["grade"]

        # ---- 5. Regime-adaptive output for the winner ----------------
        regime_summary_df = pd.DataFrame()
        regime_configs: dict[str, dict] = {}
        suggestions: dict = {}
        if best_kwargs:
            regime_summary_df, regime_configs, suggestions = self._regime_output(
                strategy_cls, best_kwargs, price_by_symbol, slices,
                base_config=base_config,
            )

        # ---- 6. Holdout metrics for the winner ----------------------
        winner_holdout_score = float("nan")
        winner_holdout_n = 0
        if best_kwargs:
            eval_row = next(
                (r for r in ranked_rows if r["config"] == _label(best_kwargs)),
                None,
            )
            if eval_row is not None:
                winner_holdout_score = eval_row["holdout_score"]
                winner_holdout_n = eval_row["holdout_n_trades"]

        logger.info(
            "%s: best grade=%s kwargs=%s  (holdout score=%.3f)",
            name, best_letter, best_kwargs, winner_holdout_score,
        )

        return AdaptiveResult(
            strategy_name=name,
            split=split,
            all_scores=all_scores,
            rankings=rankings,
            best_config_kwargs=best_kwargs,
            best_grade=best_letter,
            baseline=baseline,
            regime_summary=regime_summary_df,
            regime_configs=regime_configs,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    def _pool_trades(
        self,
        strategy_cls: type[Strategy],
        kwargs: dict[str, float],
        slices: dict[str, dict[str, pd.DataFrame]],
        window: str,
        base_config: StrategyConfig | None,
    ) -> pd.DataFrame:
        """Backtest *kwargs* per symbol on *window* and pool all trades."""
        frames: list[pd.DataFrame] = []
        for sym, win_slices in slices.items():
            df = win_slices[window]
            if df is None or df.empty:
                continue
            cfg = config_from_kwargs(kwargs, sym, base=base_config)
            result = self._engine.run(strategy_cls(cfg), df)
            if result.trades is not None and not result.trades.empty:
                t = result.trades.copy()
                t["symbol"] = sym
                frames.append(t)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _score_candidate(
        self,
        strategy_cls: type[Strategy],
        kwargs: dict[str, float],
        slices: dict[str, dict[str, pd.DataFrame]],
        base_config: StrategyConfig | None,
    ) -> dict:
        """Score a candidate on IS, OOS and holdout windows."""
        pooled = self._pool_trades(strategy_cls, kwargs, slices, "is", base_config)
        is_score = self.objective_fn(pooled, self.min_trades)
        is_n = len(pooled)

        oos = self._pool_trades(strategy_cls, kwargs, slices, "oos", base_config)
        oos_score = self.objective_fn(oos, self.min_trades)
        oos_n = len(oos)

        hold = self._pool_trades(strategy_cls, kwargs, slices, "holdout", base_config)
        holdout_score = self.objective_fn(hold, self.min_trades)
        holdout_n = len(hold)

        return {
            "is_score": is_score,
            "is_n_trades": is_n,
            "oos_score": oos_score,
            "oos_n_trades": oos_n,
            "holdout_score": holdout_score,
            "holdout_n_trades": holdout_n,
            "holdout_metrics": realized_metrics(hold),
        }

    def _regime_output(
        self,
        strategy_cls: type[Strategy],
        kwargs: dict[str, float],
        price_by_symbol: dict[str, pd.DataFrame],
        slices: dict[str, dict[str, pd.DataFrame]],
        base_config: StrategyConfig | None,
    ) -> tuple[pd.DataFrame, dict[str, dict], dict]:
        """Run the winner on IS+OOS, classify trades by regime, and produce
        per-regime configs (ATR-scaled stops) plus an adaptive-rule summary.

        Holdout trades are intentionally **not** used here — regime rules are
        derived from the same evidence windows and must be validated on fresh
        data before adoption (stated in the suggestions).
        """
        # Winner trades pooled from IS+OOS (regime conditioning + rule fitting
        # must not touch the holdout).
        is_trades = self._pool_trades(strategy_cls, kwargs, slices, "is", base_config)
        oos_trades = self._pool_trades(strategy_cls, kwargs, slices, "oos", base_config)
        frames = [t for t in (is_trades, oos_trades) if t is not None and not t.empty]
        if not frames:
            return pd.DataFrame(), {}, {"error": "winner produced no trades on IS+OOS"}
        trades = pd.concat(frames, ignore_index=True)

        full_regimes: dict[str, pd.Series] = {
            sym: classify_regime(df) for sym, df in price_by_symbol.items()
        }
        trades = join_trades_to_regime(trades, price_by_symbol, full_regimes)
        summary = regime_summary(trades)

        # Reference ATR%: full-window median across symbols (used as the
        # baseline volatility the fixed stop/TP were designed for).
        atr_pcts: list[float] = []
        for df in price_by_symbol.values():
            s = atr_pct(df["high"], df["low"], df["close"])
            atr_pcts.append(float(s.median()))
        ref_atr = float(np.nanmedian(atr_pcts)) if atr_pcts else 0.0

        cfg = config_from_kwargs(kwargs, "REFSYM", base=base_config)
        base_stop = float(cfg.stop_loss_pct)
        base_tp = float(cfg.take_profit_pct)
        global_pf = realized_metrics(trades)["profit_factor"]

        regime_configs: dict[str, dict] = {}
        for regime in REGIMES:
            sub = trades.loc[trades["regime"] == regime]
            if sub.empty:
                continue
            m = realized_metrics(sub)
            reg_atr = float(np.nanmedian(atr_pcts)) if atr_pcts else ref_atr
            stop, tp = scale_stop_tp(base_stop, base_tp, reg_atr, ref_atr)
            regime_configs[regime] = {
                **kwargs,
                "stop_loss_pct": round(stop, 4),
                "take_profit_pct": round(tp, 4),
                "n_trades": m["n_trades"],
                "profit_factor": m["profit_factor"],
                "note": (
                    f"{m['n_trades']} trades in IS+OOS, PF {m['profit_factor']:.2f} "
                    f"(global PF {global_pf:.2f}); stop/TP ATR-scaled to reg ATR% "
                    f"{reg_atr:.3f} vs ref {ref_atr:.3f}"
                ),
            }

        suggestions = {
            "ref_atr_pct": round(ref_atr, 5),
            "base_stop_pct": base_stop,
            "base_tp_pct": base_tp,
            "rule": (
                "Scale stop/TP by ATR%: stop = clip({stop} * atr_pct/{ref:.3f}, "
                "1%, 12%), tp = clip({tp} * atr_pct/{ref2:.3f}, 1.5%, 25%). "
                "Regime configs are experimental — validate on a fresh window "
                "before runtime adoption."
            ).format(stop=base_stop, ref=ref_atr, tp=base_tp, ref2=ref_atr),
            "holdout_not_used": True,
        }
        return summary, regime_configs, suggestions


def _label(kwargs: dict[str, float]) -> str:
    """Short stable label for a candidate kwargs dict."""
    if not kwargs:
        return "default_fixed"
    return ",".join(f"{k}={v:g}" for k, v in sorted(kwargs.items()))