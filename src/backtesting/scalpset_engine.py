"""ScalpSet historical replay engine — faithful LIVE semantics, no lookahead.

This is the historical backtester for the main trader's ScalpSet (the three
live modules in :mod:`src.strategies.scalp`: ICT IFVG / Box Theory /
VolProfile+FIB).  It is a *separate* module from the old generic harness in
``src/backtesting/engine.py`` (single-DataFrame signal-map simulation, no
fills / shorts / level anchoring) — that one is untouched.

Why a new engine
----------------
The live main trader (``live_trader.py``, PR #37 + the post-go-live anchoring
fix) is a tick loop over 1-minute bars: build the frame set for a symbol,
evaluate the three modules, arbitrate ONE signal (``best_rr``), gate it
(cooldown / per-session entry cap / max positions / shortability), then place
a MARKET or LIMIT order and manage the position with SL/TP levels that are
**re-anchored to the actual fill**.  A backtest is only informative if it
reproduces those mechanics, so this module replays exactly that loop over
cached 1m bars.

Replay semantics (each rule mirrors a live code path)
-----------------------------------------------------
* **Bar-by-bar 1m loop, RTH only.**  Bars come from
  :mod:`src.backtesting.scalp_data` (same RTH filter, same clock-aligned
  5m/15m/30m/1h/4h/1d aggregation, same live lookback windows, same
  partial "in-progress" bar at the right edge that live sees when it fetches
  ``start=now-Nd, end=now``).  Out-of-RTH bars are dropped.
* **No lookahead.**  A signal at bar *i* is decided from data with
  timestamp <= close(*i*) only, and every subsequent fill is derived from
  bars *> i*: MARKET orders fill at the NEXT bar's open (never the signal
  bar's close), LIMIT orders fill when a LATER bar's range trades through
  the limit (fill at the limit; if the bar gapped past it, at the open).
  Stop / take-profit exits use a later bar's range against the level.
* **Costs.**  Adverse slippage ``max(price*slippage_pct, slippage_abs)`` plus
  ``half_spread_pct`` on market-style fills (market entries, stop exits,
  EOD exits).  Limit fills (limit entries and take-profit exits) fill AT the
  limit — a limit order cannot fill through its own price — with a gap
  improvement when the bar opened beyond it.  Commission is configurable
  (per-share + notional bps; default 0).  ``ScalpSetConfig.cost_variant()``
  is the pessimistic ~2bps + half-spread setting for sensitivity runs.
* **SL/TP anchored to the ACTUAL fill** (``anchor_scalp_levels`` in
  ``live_trader.py``; identical rule re-implemented here as
  :func:`anchor_levels` and pinned by a parity test against the live
  function): keep the strategy level when it is still on the correct side of
  the fill and at least ``min_dist = max(fill*0.05%, 1ct)`` away; otherwise
  re-price it from the fill using the signal's own risk/reward distance and
  clamp to ``min_dist``; a degenerate stop falls back to the −6% backstop
  (long: ``fill*0.94``, short: ``fill*1.06``) and a degenerate take-profit is
  dropped entirely (BE/trail + stops take over).
* **Break-even + trail** (live ``_scalp_risk_pass``): once a trade is
  ``breakeven_trigger_r`` R in profit the stop moves to the fill
  (± ``breakeven_buffer``); a trailing signal then trails
  ``trail_distance_r`` R behind price, only ever in the profitable
  direction and only when the move exceeds 0.1% (the live replace guard).
* **Gating.**  5-bar (real-time minutes) cooldown per symbol after a close;
  ``SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION = 3`` entries submitted per
  symbol per trading day (pending LIMIT placements count, like live);
  ``MAX_POSITIONS = 6`` concurrent positions; one active setup per symbol
  (a live position OR a resting LIMIT bundle blocks new signals);
  duplicate-signal dedupe via the live ``_signal_key``.
* **Sizing.**  15% of equity per position, floored to whole shares (brief:
  whole-share only), capped by 95% of available cash (the live BP cap).
* **Shorts.**  A SHORT signal is only tradable for symbols on the
  shortability allow-list (default ``{NVDA, QQQ, AVGO}`` — the live
  observations); everything else counts as an untaken signal in
  ``stats["skipped"]["short_not_shortable"]`` exactly like live's
  ``_short_capability_allows`` skip.
* **EOD.**  ``eod_flat=True`` (default) closes anything still open at the
  session's last RTH bar (reason ``eod``) and lets resting LIMIT entries die
  at the close, mirroring the live day-limit TPs.  Live main positions *may*
  actually hold overnight behind GTC stops; set ``eod_flat=False`` to replay
  the overnight-hold variant.

Everything is deterministic and offline: no network, no broker, no env
credentials.  The strategy modules are consumed exactly as live imports them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.metrics import compute_metrics
from src.backtesting.scalp_data import (
    DEFAULT_CACHE,
    EXCHANGE_TZ,
    RTH_CLOSE_MIN,
    RTH_OPEN_MIN,
    SymbolFrames,
    build_symbol_frames,
)
from src.strategies.scalp import scalp_box_theory, scalp_ict_ifvg, scalp_volprofile_fib
from src.strategies.scalp.types import Direction, EntryType, ScalpContext, ScalpSignal

#: Module arbitration order / tie-break priority (live ``SCALP_MODULE_ORDER``).
MODULE_ORDER: tuple[str, ...] = ("ict_ifvg", "box_theory", "volprofile_fib")

#: Minimum bars per timeframe before a module may emit signals (live
#: ``SCALP_MIN_BARS``) — the fail-safe data gate, reproduced verbatim.
MIN_BARS: dict[str, dict[str, int]] = {
    "ict_ifvg": {"1m": 35, "15m": 10, "30m": 15, "1h": 25, "4h": 12, "1d": 2},
    "box_theory": {"5m": 3, "1d": 2},
    "volprofile_fib": {"5m": 16, "1d": 2},
}

#: Symbols whose SHORT signals are executable (live observations: these are
#: the shortable ones; COIN/META/TSLA shorts are rejected by the broker).
DEFAULT_SHORTABLE: frozenset[str] = frozenset({"NVDA", "QQQ", "AVGO"})

ModuleFn = Callable[[ScalpContext], list[ScalpSignal]]


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class ScalpSetConfig:
    """Live ScalpSet parameters + replay cost model.

    Every live-side default is the value the running main trader uses; the
    cost defaults are the "no-cost" baseline (0 commission, 2bps slippage)
    so a first pass measures the strategy rather than the fee schedule.
    """

    # ── universe / portfolio ──
    symbols: tuple[str, ...] = ("NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO")
    initial_equity: float = 100_000.0
    position_size_pct: float = 0.15          # live POSITION_SIZE_PCT
    bp_usage_pct: float = 0.95               # live MAIN_BP_USAGE_PCT
    max_positions: int = 6                   # live MAX_POSITIONS
    whole_shares: bool = True                # brief: whole-share only

    # ── gating (live constants) ──
    cooldown_bars: int = 5                   # live SCALP_COOLDOWN_BARS (minutes)
    max_entries_per_symbol_per_session: int = 3
    arbitration: str = "best_rr"             # live SCALP_ARBITRATION
    module_order: tuple[str, ...] = MODULE_ORDER
    enabled_modules: tuple[str, ...] = MODULE_ORDER
    shortable: frozenset[str] = field(default_factory=lambda: DEFAULT_SHORTABLE)
    require_shortable: bool = True
    min_bars: Mapping[str, Mapping[str, int]] = field(
        default_factory=lambda: MIN_BARS)

    # ── anchoring / risk (live PROTECTIVE_STOP_PCT + SCALP_STOP_MIN_DISTANCE_*) ──
    backstop_pct: float = 0.06
    min_dist_pct: float = 0.0005
    min_dist_abs: float = 0.01
    be_trail: bool = True
    trail_min_replace_pct: float = 0.001

    # ── costs ──
    slippage_pct: float = 0.0002             # 2bps adverse on market fills
    slippage_abs: float = 0.01               # ... or 1 cent, whichever is worse
    half_spread_pct: float = 0.0             # extra adverse bps per side
    commission_per_share: float = 0.0
    commission_pct: float = 0.0              # of traded notional, per side

    # ── session handling ──
    eod_flat: bool = True
    tick_rounding: bool = True               # live _normalize_order_price

    @classmethod
    def cost_variant(cls, **overrides) -> "ScalpSetConfig":
        """Pessimistic cost variant (~2bps slippage + half-spread + fees)."""
        base = dict(slippage_pct=0.0002, slippage_abs=0.01, half_spread_pct=0.0001,
                    commission_per_share=0.005, commission_pct=0.0)
        base.update(overrides)
        return cls(**base)


# ── Anchored SL/TP (live rule, re-implemented) ─────────────────────────


def normalize_price(price: float, enabled: bool = True) -> float:
    """Live ``_normalize_order_price``: 2dp at/above $1, else 4dp."""
    if not enabled:
        return float(price)
    try:
        p = float(price)
    except (TypeError, ValueError):
        return float(price)
    if not math.isfinite(p):
        return p
    return round(p, 2 if abs(p) >= 1.0 else 4)


@dataclass(frozen=True)
class AnchoredLevels:
    """Result of :func:`anchor_levels` (mirrors live ``AnchoredLevels``)."""

    sl: Optional[float]
    tp: Optional[float]
    sl_source: str
    tp_source: str
    min_distance: float = 0.0


def anchor_levels(
    *,
    is_short: bool,
    fill: float,
    entry_ref: Optional[float] = None,
    sl_ref: Optional[float] = None,
    tp_ref: Optional[float] = None,
    min_dist_pct: float = 0.0005,
    min_dist_abs: float = 0.01,
    backstop_pct: float = 0.06,
    rounding: bool = True,
) -> AnchoredLevels:
    """Anchor a ScalpSet signal's SL/TP to the ACTUAL fill price.

    Byte-for-byte the rule of ``anchor_scalp_levels`` in ``live_trader.py``
    (see that function's docstring for the live COIN 2026-09-16 evidence).
    Kept as a separate implementation because importing ``live_trader`` pulls
    in the broker/data stack and configures logging; ``tests/
    test_scalpset_engine.py`` asserts the two agree on a table of cases.
    """
    min_dist = max(float(fill) * abs(min_dist_pct), abs(min_dist_abs))
    sign = 1.0 if is_short else -1.0

    # ── stop-loss ──
    sl: Optional[float]
    sl_source: str
    if sl_ref is not None and (
        sl_ref >= fill + min_dist if is_short else sl_ref <= fill - min_dist
    ):
        sl, sl_source = normalize_price(sl_ref, rounding), "strategy"
    else:
        risk = abs(entry_ref - sl_ref) if (entry_ref is not None and sl_ref is not None) else None
        if risk is not None and risk > 0:
            raw = fill + sign * risk
            clamped = max(raw, fill + min_dist) if is_short else min(raw, fill - min_dist)
            level = normalize_price(clamped, rounding)
            if level > 0 and ((level > fill) if is_short else (level < fill)):
                sl = level
                sl_source = "fill_risk" if abs(clamped - raw) <= 1e-9 else "fill_risk_clamped"
            else:
                sl, sl_source = None, "backstop"
        else:
            sl, sl_source = None, ("backstop" if sl_ref is not None else "no_strategy_sl")
    if sl is None:
        level = normalize_price(fill * (1 + backstop_pct) if is_short else fill * (1 - backstop_pct),
                                rounding)
        if level > 0 and ((level > fill) if is_short else (level < fill)):
            sl, sl_source = level, ("backstop" if sl_ref is not None else "no_strategy_sl")
        else:  # pragma: no cover - only reachable for non-positive prices
            sl, sl_source = None, "unusable"

    # ── take-profit ──
    tp: Optional[float]
    tp_source: str
    if tp_ref is not None and (
        tp_ref <= fill - min_dist if is_short else tp_ref >= fill + min_dist
    ):
        tp, tp_source = normalize_price(tp_ref, rounding), "strategy"
    else:
        reward = abs(tp_ref - entry_ref) if (tp_ref is not None and entry_ref is not None) else None
        tp, tp_source = None, "none"
        if reward is not None and reward > 0:
            raw = fill - sign * reward
            clamped = min(raw, fill - min_dist) if is_short else max(raw, fill + min_dist)
            level = normalize_price(clamped, rounding)
            if level > 0 and ((level < fill) if is_short else (level > fill)):
                tp = level
                tp_source = "fill_reward" if abs(clamped - raw) <= 1e-9 else "fill_reward_clamped"
    return AnchoredLevels(sl=sl, tp=tp, sl_source=sl_source, tp_source=tp_source,
                          min_distance=min_dist)


# ── Replay result ──────────────────────────────────────────────────────

TRADE_COLUMNS: tuple[str, ...] = (
    "symbol", "module", "side", "entry_type", "entry_time", "exit_time",
    "entry_price", "exit_price", "quantity", "sl", "tp", "sl_source",
    "tp_source", "exit_reason", "pnl_gross", "fees", "pnl", "pnl_after_costs",
    "pnl_pct", "bars_held",
)


@dataclass
class ReplayResult:
    """Trades + equity curve + execution stats of one replay."""

    trades: pd.DataFrame
    equity_curve: pd.Series
    stats: dict
    config: ScalpSetConfig

    def metrics(self, risk_free_rate: float = 0.0) -> dict:
        """Standard performance metrics (``src/backtesting/metrics.py``)."""
        return compute_metrics(self.equity_curve, self.trades, risk_free_rate)

    def summary(self) -> str:  # pragma: no cover - convenience for CLIs
        m = self.metrics()
        return (
            f"trades={len(self.trades)} "
            f"pnl=${self.trades['pnl_after_costs'].sum():,.2f} "
            f"total_return={m['total_return']:.2%} "
            f"win_rate={m['win_rate']:.1%} "
            f"max_dd={m['max_drawdown']:.2%} "
            f"sharpe={m['sharpe_ratio']:.2f} "
            f"final_equity=${self.stats['final_equity']:,.2f}"
        )


# ── Internal order / position records ──────────────────────────────────


@dataclass
class _PendingMarket:
    signal: ScalpSignal
    qty: float
    placed_time: pd.Timestamp


@dataclass
class _PendingLimit:
    signal: ScalpSignal
    qty: float
    limit_price: float        # == signal.entry_price (live day-limit entry)
    placed_time: pd.Timestamp
    session_day: object


@dataclass
class _Position:
    symbol: str
    module: str
    direction: Direction
    qty: float                # always positive; sign comes from direction
    entry_price: float
    entry_time: pd.Timestamp
    entry_index: int
    session_day: object
    sl: Optional[float]
    tp: Optional[float]
    sl_source: str
    tp_source: str
    be_trigger_r: float
    be_buffer: float
    trailing: bool
    trail_r: float
    trail_trigger_r: float
    entry_fees: float
    entry_type: str = "MARKET"
    be_done: bool = False

    @property
    def is_short(self) -> bool:
        return self.direction == Direction.SHORT


# ── Engine ─────────────────────────────────────────────────────────────


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30 <= t < 16:00 exchange-local bars (live sees no more)."""
    if df.empty:
        return df
    et = pd.DatetimeIndex(df.index)
    if et.tz is not None:
        et = et.tz_convert(EXCHANGE_TZ).tz_localize(None)
    mod = et.hour.to_numpy() * 60 + et.minute.to_numpy()
    keep = (mod >= RTH_OPEN_MIN) & (mod < RTH_CLOSE_MIN)
    out = df.loc[keep]
    out.index = pd.DatetimeIndex(et[keep], name=df.index.name or "ts")
    return out.sort_index()


def frames_from_bars(
    bars: Mapping[str, pd.DataFrame],
    *,
    pair_symbol: str = "QQQ",
    rth_only: bool = True,
) -> dict[str, SymbolFrames]:
    """Build :class:`SymbolFrames` from plain ET-naive OHLCV frames.

    Test/utility entry point (the cache-reading equivalent is
    :func:`src.backtesting.scalp_data.build_symbol_frames`).  The symbol in
    ``bars`` matching ``pair_symbol`` is wired as the SMT pair for the others.
    """
    prepared = {s.upper(): (filter_rth(df) if rth_only else df) for s, df in bars.items()}
    out = {s: SymbolFrames(s, df) for s, df in prepared.items()}
    pair = out.get(pair_symbol.upper())
    if pair is not None:
        for s, sf in out.items():
            if s != pair_symbol.upper():
                sf.set_pair(pair, pair_symbol.upper())
    return out


class ScalpSetEngine:
    """Bar-by-bar ScalpSet replay over cached/custom 1m bars.

    Parameters
    ----------
    frames:
        ``{symbol: SymbolFrames}``.  The frames hold the FULL loaded history
        (warmup included) so the live lookback windows are populated; the
        replay only trades inside ``trade_window``.
    config:
        :class:`ScalpSetConfig` (live defaults).
    modules:
        Optional ``{name: callable}`` override — used by the hermetic tests to
        script signals without needing 35+ bars of synthetic history.  Names
        must appear in ``config.module_order``.
    trade_window:
        ``(start, end)`` timestamps (inclusive / exclusive) restricting which
        bars are traded.  ``None`` = everything loaded.
    """

    def __init__(
        self,
        frames: Mapping[str, SymbolFrames],
        config: Optional[ScalpSetConfig] = None,
        *,
        modules: Optional[Mapping[str, ModuleFn]] = None,
        trade_window: Optional[tuple[object, object]] = None,
        pair_symbol: str = "QQQ",
    ) -> None:
        self.config = config or ScalpSetConfig()
        self.frames = {s.upper(): f for s, f in frames.items()}
        self.modules: dict[str, ModuleFn] = dict(
            modules if modules is not None else self._default_modules())
        self.pair_symbol = pair_symbol.upper()
        self.trade_window = (
            (pd.Timestamp(trade_window[0]), pd.Timestamp(trade_window[1]))
            if trade_window else None
        )
        for sym in self.config.symbols:
            if sym.upper() in self.frames and sym.upper() != self.pair_symbol:
                self.frames[sym.upper()].set_pair(
                    self.frames.get(self.pair_symbol), self.pair_symbol)

        # ── run state ──
        self._cash = float(self.config.initial_equity)
        self._positions: dict[str, _Position] = {}
        self._pending_market: dict[str, _PendingMarket] = {}
        self._pending_limit: dict[str, _PendingLimit] = {}
        self._last_signal_key: dict[str, str] = {}
        self._cooldown_until: dict[str, pd.Timestamp] = {}
        self._entries_today: dict[tuple[str, object], int] = {}
        self._last_close: dict[str, float] = {}
        self._trades: list[dict] = []
        self._stats: dict = {}

    # ── wiring ──────────────────────────────────────────────────────
    @staticmethod
    def _default_modules() -> dict[str, ModuleFn]:
        return {
            "ict_ifvg": scalp_ict_ifvg.evaluate,
            "box_theory": scalp_box_theory.evaluate,
            "volprofile_fib": scalp_volprofile_fib.evaluate,
        }

    # ── public API ──────────────────────────────────────────────────
    def run(self) -> ReplayResult:
        """Replay every bar in the trade window and return the result."""
        symbols = [s.upper() for s in self.config.symbols if s.upper() in self.frames]
        symbols = [s for s in symbols if len(self.frames[s].ts) > 0]
        if not symbols:
            raise ValueError("no configured symbol has a frame to replay")

        ts_axis = np.unique(np.concatenate([self.frames[s].ts for s in symbols]))
        if self.trade_window is not None:
            lo, hi = self.trade_window
            ts_axis = ts_axis[(ts_axis >= np.datetime64(lo)) & (ts_axis < np.datetime64(hi))]

        local: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sym in symbols:
            own = self.frames[sym].ts
            idx = np.clip(np.searchsorted(own, ts_axis), 0, len(own) - 1)
            local[sym] = (idx, own[idx] == ts_axis)

        equity_rows: list[tuple[pd.Timestamp, float]] = []
        stats = self._new_stats(symbols, len(ts_axis))
        self._stats = stats
        last_day: object = None

        for t64 in ts_axis:
            t = pd.Timestamp(t64)
            day = t.date()
            new_session = last_day is None or day != last_day
            if new_session:
                stats["sessions"] += 1
                self._expire_session_orders(t, stats)
            last_day = day

            bar_symbols = [s for s in symbols if self._has_bar(local, s, t64, ts_axis)]

            # 1) working orders + open positions, symbol by symbol
            for sym in bar_symbols:
                i = int(local[sym][0][self._bar_pos(ts_axis, t64)])
                self._process_bar(sym, i, t)

            # 2) new signals at this bar's close (data <= close only)
            for sym in bar_symbols:
                i = int(local[sym][0][self._bar_pos(ts_axis, t64)])
                self._evaluate_and_place(sym, i, t)

            # 3) session close: flatten (optional) + kill the day's orders
            if self._is_session_end(ts_axis, t64):
                if self.config.eod_flat:
                    for sym in bar_symbols:
                        i = int(local[sym][0][self._bar_pos(ts_axis, t64)])
                        self._close_at_session_end(sym, i, t)
                self._expire_session_orders(t, stats)

            equity_rows.append((t, self._equity()))   # marks updated in _process_bar

        trades = pd.DataFrame(self._trades, columns=list(TRADE_COLUMNS))
        equity = pd.Series([v for _, v in equity_rows],
                           index=pd.DatetimeIndex([k for k, _ in equity_rows], name="ts"),
                           name="equity")
        stats["final_equity"] = float(equity.iloc[-1]) if len(equity) else float(self.config.initial_equity)
        stats["open_positions_at_end"] = len(self._positions)
        stats["unfilled_limits_at_end"] = len(self._pending_limit)
        return ReplayResult(trades=trades, equity_curve=equity, stats=stats, config=self.config)

    # ── helpers: bar bookkeeping ────────────────────────────────────
    @staticmethod
    def _new_stats(symbols: list[str], bars: int) -> dict:
        """Fresh stats container (also the shape the report phase reads)."""
        return {
            "symbols": list(symbols),
            "bars": int(bars),
            "sessions": 0,
            "signals_by_module": {},
            "signals_considered": 0,
            "entries": 0,
            "entries_by_module": {},
            "entries_by_side": {"LONG": 0, "SHORT": 0},
            "market_entries": 0,
            "limit_orders_placed": 0,
            "limit_orders_filled": 0,
            "limit_orders_expired": 0,
            "exits": {},
            "skipped": {},
            "fees_paid": 0.0,
            "realized_pnl": 0.0,
        }

    @staticmethod
    def _bar_pos(ts_axis: np.ndarray, t64: np.datetime64) -> int:
        return int(np.searchsorted(ts_axis, t64))

    def _has_bar(self, local, sym: str, t64, ts_axis) -> bool:
        return bool(local[sym][1][self._bar_pos(ts_axis, t64)])

    def _is_session_end(self, ts_axis: np.ndarray, t64: np.datetime64) -> bool:
        """True when *t64* is the last RTH bar of its exchange day (15:59 ET)."""
        t = pd.Timestamp(t64)
        if t.hour * 60 + t.minute >= RTH_CLOSE_MIN - 1:
            return True
        p = self._bar_pos(ts_axis, t64)
        if p + 1 >= len(ts_axis):
            return True
        return pd.Timestamp(ts_axis[p + 1]).date() != t.date()

    def _equity(self) -> float:
        eq = self._cash
        for sym, pos in self._positions.items():
            mark = self._last_close.get(sym, pos.entry_price)
            signed = -pos.qty if pos.is_short else pos.qty
            eq += signed * mark
        return float(eq)

    # ── helpers: costs ──────────────────────────────────────────────
    def _slip(self, price: float) -> float:
        cfg = self.config
        slip = max(abs(price) * cfg.slippage_pct, cfg.slippage_abs)
        slip += abs(price) * cfg.half_spread_pct
        return slip

    def _market_fill(self, price: float, is_buy: bool) -> float:
        """Adverse price for a MARKET-style fill."""
        slip = self._slip(price)
        raw = price + slip if is_buy else price - slip
        return normalize_price(raw, self.config.tick_rounding)

    def _fees(self, qty: float, price: float) -> float:
        cfg = self.config
        return float(qty * cfg.commission_per_share + abs(qty * price) * cfg.commission_pct)

    # ── helpers: sizing ─────────────────────────────────────────────
    def _size(self, price: float) -> float:
        """Whole-share qty: 15% of equity, capped by 95% of available cash."""
        cfg = self.config
        if price <= 0:
            return 0.0
        equity = self._equity()
        notional = min(equity * cfg.position_size_pct, max(self._cash, 0.0) * cfg.bp_usage_pct)
        qty = notional / price
        if cfg.whole_shares:
            qty = math.floor(qty)
        return float(qty)

    # ── per-bar processing ──────────────────────────────────────────
    def _process_bar(self, sym: str, i: int, t: pd.Timestamp) -> None:
        sf = self.frames[sym]
        o, h, l, c = float(sf.o[i]), float(sf.h[i]), float(sf.l[i]), float(sf.c[i])
        self._last_close[sym] = c

        # (a) working MARKET order from the previous bar → fills at this open
        pm = self._pending_market.pop(sym, None)
        if pm is not None:
            self._open_position(pm.signal, pm.qty, o, t, i, gap_fill=True)

        # (b) resting LIMIT entry → fills when this bar trades through it
        pl = self._pending_limit.get(sym)
        if pl is not None:
            fill = self._limit_fill(pl, o, h, l)
            if fill is not None:
                self._pending_limit.pop(sym, None)
                self._stats["limit_orders_filled"] += 1
                self._open_position(pl.signal, pl.qty, fill, t, i, gap_fill=False)

        # (c) exits against this bar's range (position may be brand new)
        pos = self._positions.get(sym)
        if pos is None:
            return
        exit_reason = None
        exit_price = None
        sl, tp = pos.sl, pos.tp
        if pos.is_short:
            sl_hit = sl is not None and h >= sl
            tp_hit = tp is not None and l <= tp
        else:
            sl_hit = sl is not None and l <= sl
            tp_hit = tp is not None and h >= tp
        if sl_hit:                      # ambiguous bar (both levels inside) → stop first
            exit_reason = "sl"
            gap = o >= sl if pos.is_short else o <= sl
            base = o if gap else float(sl)
            exit_price = self._market_fill(base, is_buy=pos.is_short)
        elif tp_hit:
            exit_reason = "tp"
            base = max(o, float(tp)) if not pos.is_short else min(o, float(tp))
            exit_price = normalize_price(base, self.config.tick_rounding)
        if exit_reason is not None:
            self._close_position(sym, exit_price, t, i, exit_reason)
            return

        # (d) break-even / trail on the close (live _scalp_risk_pass)
        if self.config.be_trail:
            self._update_be_trail(pos, c)

    def _limit_fill(self, pl: _PendingLimit, o: float, h: float, l: float) -> Optional[float]:
        """Limit fill from a later bar: at the limit, or the open if gapped past."""
        limit = pl.limit_price
        if pl.signal.direction == Direction.LONG:
            if o <= limit:
                return normalize_price(o, self.config.tick_rounding)
            if l <= limit:
                return normalize_price(limit, self.config.tick_rounding)
        else:
            if o >= limit:
                return normalize_price(o, self.config.tick_rounding)
            if h >= limit:
                return normalize_price(limit, self.config.tick_rounding)
        return None

    def _update_be_trail(self, pos: _Position, price: float) -> None:
        """Move the stop to break-even, then trail it (live risk-pass rules)."""
        if pos.sl is None:
            return
        risk = abs(pos.entry_price - pos.sl)
        if risk <= 0:
            return
        profit_r = ((pos.entry_price - price) / risk) if pos.is_short else ((price - pos.entry_price) / risk)
        if not pos.be_done and pos.be_trigger_r > 0 and profit_r >= pos.be_trigger_r:
            be = (pos.entry_price - pos.be_buffer) if pos.is_short else (pos.entry_price + pos.be_buffer)
            better = (be < pos.sl) if pos.is_short else (be > pos.sl)
            if better:
                # live replaces the stop with the break-even level (buffer 0
                # means the stop sits exactly on the fill price)
                pos.sl = normalize_price(be, self.config.tick_rounding)
                pos.sl_source = "breakeven"
                pos.be_done = True
                return
        if pos.trailing and pos.trail_r > 0 and profit_r >= pos.trail_trigger_r:
            new_sl = (price + pos.trail_r * risk) if pos.is_short else (price - pos.trail_r * risk)
            better = (new_sl < pos.sl) if pos.is_short else (new_sl > pos.sl)
            big_enough = abs(new_sl - pos.sl) / max(abs(pos.sl), 1e-9) >= self.config.trail_min_replace_pct
            if better and big_enough and new_sl > 0:
                pos.sl = normalize_price(new_sl, self.config.tick_rounding)
                pos.sl_source = "trail"

    # ── order placement ─────────────────────────────────────────────
    def _context(self, sym: str, i: int) -> ScalpContext:
        return self.frames[sym].context(i)

    def _data_ready(self, ctx: ScalpContext) -> bool:
        """Live ``_scalp_data_ready``: every enabled module has its min bars."""
        for key in self.config.enabled_modules:
            for tf_key, need in self.config.min_bars.get(key, {}).items():
                frame = ctx.frames.get(tf_key)
                if frame is None or len(frame) < need:
                    return False
        return True

    def _arbitrate(self, signals: list[ScalpSignal]) -> Optional[ScalpSignal]:
        if not signals:
            return None
        if len(signals) == 1:
            return signals[0]
        order = self.config.module_order
        if self.config.arbitration == "first":
            return min(signals, key=lambda s: order.index(s.strategy) if s.strategy in order else 99)
        return min(signals, key=lambda s: (
            -s.rr, order.index(s.strategy) if s.strategy in order else 99))

    @staticmethod
    def _signal_key(s: ScalpSignal) -> str:
        return (f"{s.strategy}|{s.direction.value}|{s.entry_type.value}|"
                f"{s.entry_price:.4f}|{s.stop_loss:.4f}|{s.take_profit:.4f}")

    def _bump(self, bucket: str, key: str, n: int = 1) -> None:
        d = self._stats.setdefault(bucket, {})
        d[key] = d.get(key, 0) + n

    def _evaluate_and_place(self, sym: str, i: int, t: pd.Timestamp) -> None:
        stats = self._stats
        # one active setup per symbol (live: position OR resting bundle)
        if sym in self._positions or sym in self._pending_limit or sym in self._pending_market:
            return
        if t < self._cooldown_until.get(sym, pd.Timestamp.min):
            self._bump("skipped", "cooldown")
            return
        if len(self._positions) >= self.config.max_positions:
            self._bump("skipped", "max_positions")
            return
        cap = self.config.max_entries_per_symbol_per_session
        if cap and cap > 0 and self._entries_for(sym, t) >= cap:
            self._bump("skipped", "entry_cap")
            return

        ctx = self._context(sym, i)
        if not self._data_ready(ctx):
            self._bump("skipped", "insufficient_data")
            return
        signals: list[ScalpSignal] = []
        for key in self.config.enabled_modules:
            fn = self.modules.get(key)
            if fn is None:
                continue
            try:
                out = fn(ctx)
            except Exception:            # pragma: no cover - live logs+continues
                out = []
            if out:
                self._bump("signals_by_module", key, len(out))
                signals.extend(out)
        chosen = self._arbitrate(signals)
        if chosen is None:
            return
        if chosen.symbol.upper() != sym:
            # defensive: a module must not emit another symbol from this ctx
            self._bump("skipped", "symbol_mismatch")
            return
        stats["signals_considered"] += 1
        if self._last_signal_key.get(sym) == self._signal_key(chosen):
            self._bump("skipped", "duplicate_signal")
            return
        if chosen.direction == Direction.SHORT and self.config.require_shortable \
                and sym not in {s.upper() for s in self.config.shortable}:
            self._bump("skipped", "short_not_shortable")
            return

        size_ref = (float(chosen.entry_price) if chosen.entry_type == EntryType.LIMIT
                    else float(self.frames[sym].c[i]))
        qty = self._size(size_ref)
        if qty < 1:
            self._bump("skipped", "qty_lt_1")
            return

        self._last_signal_key[sym] = self._signal_key(chosen)
        self._count_entry(sym, t)
        stats["entries"] += 1
        self._bump("entries_by_module", chosen.strategy)
        self._bump("entries_by_side", chosen.direction.value)
        if chosen.entry_type == EntryType.LIMIT:
            self._pending_limit[sym] = _PendingLimit(
                signal=chosen, qty=qty, limit_price=float(chosen.entry_price),
                placed_time=t, session_day=t.date())
            stats["limit_orders_placed"] += 1
        else:
            self._pending_market[sym] = _PendingMarket(signal=chosen, qty=qty, placed_time=t)
            stats["market_entries"] += 1

    def _open_position(
        self, signal: ScalpSignal, qty: float, ref_price: float, t: pd.Timestamp,
        i: int, gap_fill: bool,
    ) -> None:
        """Record a fill at *ref_price* and anchor SL/TP to it (live rule)."""
        is_short = signal.direction == Direction.SHORT
        fill = (self._market_fill(ref_price, is_buy=not is_short) if gap_fill
                else normalize_price(ref_price, self.config.tick_rounding))
        if fill <= 0:
            return
        levels = anchor_levels(
            is_short=is_short, fill=fill, entry_ref=float(signal.entry_price),
            sl_ref=float(signal.stop_loss), tp_ref=float(signal.take_profit),
            min_dist_pct=self.config.min_dist_pct, min_dist_abs=self.config.min_dist_abs,
            backstop_pct=self.config.backstop_pct, rounding=self.config.tick_rounding)
        fees = self._fees(qty, fill)
        if is_short:                                  # short sale adds proceeds to cash
            self._cash += qty * fill - fees
        else:
            self._cash -= qty * fill + fees
        self._positions[signal.symbol.upper()] = _Position(
            symbol=signal.symbol.upper(), module=signal.strategy, direction=signal.direction,
            qty=qty, entry_price=fill, entry_time=t, entry_index=i,
            session_day=t.date(), sl=levels.sl, tp=levels.tp,
            sl_source=levels.sl_source, tp_source=levels.tp_source,
            be_trigger_r=float(signal.breakeven_trigger_r), be_buffer=float(signal.breakeven_buffer),
            trailing=bool(signal.trailing), trail_r=float(signal.trail_distance_r),
            trail_trigger_r=float(signal.trail_trigger_r), entry_fees=fees,
            entry_type=signal.entry_type.value)
        self._stats["fees_paid"] = round(self._stats["fees_paid"] + fees, 6)

    def _close_position(self, sym: str, price: float, t: pd.Timestamp, i: int,
                        reason: str) -> None:
        pos = self._positions.pop(sym, None)
        if pos is None:
            return
        fees = self._fees(pos.qty, price)
        if pos.is_short:
            self._cash -= pos.qty * price + fees
            gross = (pos.entry_price - price) * pos.qty
        else:
            self._cash += pos.qty * price - fees
            gross = (price - pos.entry_price) * pos.qty
        total_fees = pos.entry_fees + fees
        net = gross - total_fees
        notional = abs(pos.entry_price * pos.qty) or 1.0
        self._trades.append({
            "symbol": sym, "module": pos.module, "side": pos.direction.value,
            "entry_type": pos.entry_type,
            "entry_time": pos.entry_time, "exit_time": t,
            "entry_price": pos.entry_price, "exit_price": price, "quantity": pos.qty,
            "sl": pos.sl, "tp": pos.tp, "sl_source": pos.sl_source,
            "tp_source": pos.tp_source, "exit_reason": reason,
            "pnl_gross": gross, "fees": total_fees,
            "pnl": net, "pnl_after_costs": net, "pnl_pct": net / notional,
            "bars_held": int(i - pos.entry_index),
        })
        self._stats["fees_paid"] = round(self._stats["fees_paid"] + fees, 6)
        self._stats["realized_pnl"] = round(self._stats["realized_pnl"] + net, 6)
        self._bump("exits", reason)
        self._cooldown_until[sym] = t + pd.Timedelta(minutes=self.config.cooldown_bars)
        self._last_signal_key.pop(sym, None)
        self._pending_limit.pop(sym, None)
        self._pending_market.pop(sym, None)

    def _close_at_session_end(self, sym: str, i: int, t: pd.Timestamp) -> None:
        pos = self._positions.get(sym)
        if pos is None:
            return
        close = float(self.frames[sym].c[i])
        self._close_position(sym, self._market_fill(close, is_buy=pos.is_short), t, i, "eod")

    def _expire_session_orders(self, t: pd.Timestamp, stats: dict) -> None:
        """Day orders die at the close; a new session starts with a clean book."""
        for sym in list(self._pending_limit):
            self._pending_limit.pop(sym, None)
            stats["limit_orders_expired"] += 1
        self._pending_market.clear()

    # ── entry accounting (live per-session cap semantics) ───────────
    def _entries_for(self, sym: str, t: pd.Timestamp) -> int:
        return int(self._entries_today.get((sym, t.date()), 0))

    def _count_entry(self, sym: str, t: pd.Timestamp) -> None:
        key = (sym, t.date())
        self._entries_today[key] = self._entries_today.get(key, 0) + 1


# ── Convenience entry points ───────────────────────────────────────────


def replay(
    symbols: Sequence[str] | str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: Path | str = DEFAULT_CACHE,
    warmup_days: int = 75,
    config: Optional[ScalpSetConfig] = None,
    pair_symbol: str = "QQQ",
    modules: Optional[Mapping[str, ModuleFn]] = None,
    trade_window: Optional[tuple[object, object]] = None,
) -> ReplayResult:
    """Load the cached 1m history and replay the ScalpSet over a window.

    ``warmup_days`` of bars *before* ``start`` are loaded so the live lookback
    windows (up to 60 days for the 1h frame) are populated on the first traded
    day; only ``[start, end)`` is traded.
    """
    cfg = config or ScalpSetConfig()
    if isinstance(symbols, str):
        symbols = [symbols]
    syms = [s.upper() for s in symbols]
    load_start = None
    if start is not None:
        load_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    frames: dict[str, SymbolFrames] = {}
    aux = [pair_symbol.upper()] if pair_symbol and pair_symbol.upper() not in syms else []
    for sym in syms + aux:
        frames[sym] = build_symbol_frames(sym, cache_dir=cache_dir, start=load_start, end=end)
    patch = {}
    if tuple(syms) != tuple(cfg.symbols):
        patch["symbols"] = tuple(syms)
    engine_cfg = replace(cfg, **patch) if patch else cfg
    engine = ScalpSetEngine(frames, engine_cfg, modules=modules,
                            trade_window=trade_window or (start, end),
                            pair_symbol=pair_symbol)
    return engine.run()


def run_bars(
    bars: Mapping[str, pd.DataFrame],
    config: Optional[ScalpSetConfig] = None,
    *,
    modules: Optional[Mapping[str, ModuleFn]] = None,
    pair_symbol: str = "QQQ",
    trade_window: Optional[tuple[object, object]] = None,
) -> ReplayResult:
    """Replay hand-built frames (hermetic / test entry point)."""
    cfg = config or ScalpSetConfig()
    frames = frames_from_bars(bars, pair_symbol=pair_symbol)
    patch = {}
    if tuple(s.upper() for s in bars) != tuple(cfg.symbols):
        patch["symbols"] = tuple(s.upper() for s in bars)
    engine_cfg = replace(cfg, **patch) if patch else cfg
    engine = ScalpSetEngine(frames, engine_cfg, modules=modules,
                            trade_window=trade_window, pair_symbol=pair_symbol)
    return engine.run()


__all__ = [
    "AnchoredLevels",
    "DEFAULT_SHORTABLE",
    "MIN_BARS",
    "MODULE_ORDER",
    "ReplayResult",
    "ScalpSetConfig",
    "ScalpSetEngine",
    "anchor_levels",
    "filter_rth",
    "frames_from_bars",
    "normalize_price",
    "replay",
    "run_bars",
]
