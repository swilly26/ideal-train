"""Turbo-trader replay engine (frozen ``classic`` profile and the current logic).

Why this module exists
---------------------
The live turbo trader's only real track record (2026-07-30 → 2026-09-23,
+$22,496 over 233 broker-verified round trips) has been attributed to market
beta rather than selection: 99.8 % of the early window's P&L is explained by
the pooled move of the same four leveraged ETFs over the same holding
windows, and the single largest winner was a position a *dead process* was
holding when the next startup sold it into the opening gap
(``/home/team/shared/TURBO_FORENSICS_20260924.md``).  Nobody had ever replayed
the classic *logic* itself with the end-of-day flatten ON, which is the number
that decides whether the frozen profile is worth running.

This engine replays the turbo rule set bar-by-bar over cached 1-minute bars.
It is not the live trader: it re-implements its decision rules, and the
divergences are listed below so a reader can judge the fidelity claim.

What it models (``TurboConfig.classic()`` — PR #41, commit 72b9d8d)
-------------------------------------------------------------------
* Pool: the base four 3x ETFs only (SOXL, TQQQ, FNGU, SPXL); no VIOLENCE tier.
* No regime gate, long-only (no momentum shorts, no mean-reversion shorts).
* Legacy momentum SELL confidence (``min(1, dist_pct * 10)``), which in
  practice almost never reaches the 0.4 activation threshold — so longs exit
  on the 30-minute time exit / 6 % stop / 8 % target / EOD flatten.
* Entry signals: mean-reversion (z-score of the 20-bar close, |z| > 0.5) and
  momentum (10-bar MA + RSI(14): BUY when close > MA and RSI > 40; SELL on
  the MA break), confidence >= 0.4, best-confidence wins.
* Sizing: 50 % of equity per position, capped at 95 % of available cash,
  fractional longs allowed, entries smaller than 1 share are skipped.
* MAX_POSITIONS 2, and at most one new entry per minute (live
  ``CHECK_INTERVAL = 60``).
* Exits: broker-anchored 6 % stop / 8 % target measured from the *fill*,
  checked intrabar and gap-aware; hard 30-minute time exit; mandatory EOD
  flatten 30 minutes before the close (15:30 ET).
* The live loop fetches a 60-minute window and evaluates *every* bar in it,
  so a signal that fired up to an hour ago can still trigger an entry now.
  This engine reproduces that quirk (sliding 60-bar window, earliest bar wins
  a confidence tie, mean-reversion ahead of momentum).
* The live loop skips a symbol whose fetched window has fewer than 25 bars,
  so with the 60-minute lookback no entry can happen before ~09:54 ET and a
  window never spans the overnight boundary.  Both are modelled.

Known divergences from the live trader (all deliberate)
------------------------------------------------------
1. **No deaths, no restarts, no unfilled orders.**  The replay never carries a
   position overnight (EOD flatten always runs), never inherits a position
   from a crashed process, and assumes every order fills.  The live record is
   dominated by exactly those events.
2. **Equity is this book's own** ($100k convention), not the shared paper
   account's equity, and it compounds trade by trade.
3. **Indicators are computed per session**, exactly as the live 60-minute window
   does (its rolling history never crosses the overnight boundary), so the first
   bars of each session carry the same NaN history live saw.  Within the body of
   a session the two are identical.
4. **Liquidity-sweep signals are not modelled.**  They need precomputed daily
   levels and only ever fire on structure breaks; the live window's entry mix
   is measured against the replay in the fidelity report.
5. **Exits fill at the current bar's close (±slippage)** for signal exits and
   time exits, and intrabar (gap-aware) for stop/target.  Live submits a market
   order at the tick and takes whatever the next print is.
6. The EOD flatten is pinned to 15:30 ET.  Live derives it from a 20:00 UTC
   constant, which is 16:00 ET in summer (correct) and 15:00 ET in winter.
7. The 30-minute time exit fires at exactly entry + 30 bars; live measures
   from the fill timestamp a few seconds after the bar, so it can fire one
   minute later.

Costs mirror ``ScalpSetConfig`` (the ScalpSet replay's model): ``baseline`` is
2 bps or 1 cent of adverse slippage per fill, whichever is larger; ``zero_cost``
turns every cost off as a gross-edge diagnostic; ``pessimistic`` adds a 1 bp
half-spread and $0.005/share.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from src.backtesting.scalp_data import DEFAULT_CACHE, load_rth_1m

# ── live constants the replay is pinned to (turbo_trader.py, PR #41) ────
RTH_OPEN_MIN = 9 * 60 + 30          # 09:30 ET
RTH_CLOSE_MIN = 16 * 60             # 16:00 ET
EOD_FLAT_MIN = 15 * 60 + 30         # 15:30 ET — MANDATORY_CLOSE_MINUTES = 30
BASE4 = ("SOXL", "TQQQ", "FNGU", "SPXL")

TRADE_COLUMNS = [
    "symbol", "side", "qty", "entry_time", "entry_price", "exit_time",
    "exit_price", "hold_minutes", "exit_reason", "entry_conf",
    "entry_strategy", "pnl_gross", "fees", "pnl_after_costs", "ret_pct",
]


@dataclass
class TurboConfig:
    """Live turbo parameters + the replay cost model (see module docstring)."""

    # ── universe / portfolio ──
    symbols: tuple[str, ...] = BASE4
    initial_equity: float = 100_000.0
    position_size_pct: float = 0.50          # live POSITION_SIZE_PCT
    bp_usage_pct: float = 0.95               # live BUYING_POWER_USAGE_PCT (#29)
    max_positions: int = 2                   # live MAX_POSITIONS
    min_entry_qty: float = 1.0               # live `if qty < 1: return`

    # ── risk (base tier; ``_risk_params_for``) ──
    stop_loss_pct: float = 0.06
    take_profit_pct: float = 0.08
    max_hold_minutes: int = 30               # live MAX_HOLD_MINUTES
    max_hold_extended_minutes: int = 60      # violence tier only → inert here
    trend_extension: bool = False
    eod_flat: bool = True
    eod_flat_minute: int = EOD_FLAT_MIN

    # ── signals ──
    confidence_threshold: float = 0.4
    ma_period: int = 10                      # MOMENTUM_CONFIG["ma_period"]
    rsi_period: int = 14
    rsi_threshold: float = 40.0              # MOMENTUM_CONFIG["rsi_threshold"]
    legacy_sell_confidence: bool = True      # classic profile
    mr_lookback: int = 20                    # STRATEGY_CONFIG extra["lookback"]
    mr_entry_threshold: float = 0.5
    regime_gate: bool = False                # classic: off
    allow_shorts: bool = False               # classic: long-only
    mr_short: bool = False
    lookback_bars: int = 60                  # live 60-minute fetch
    min_bars: int = 25                       # live `len(data) < 25 → skip`
    one_entry_per_bar: bool = True           # live max 1 entry per tick

    # ── costs (mirror ScalpSetConfig) ──
    slippage_pct: float = 0.0002
    slippage_abs: float = 0.01
    half_spread_pct: float = 0.0
    commission_per_share: float = 0.0
    commission_pct: float = 0.0

    # ── bookkeeping ──
    verbose: bool = False

    # ── constructors ───────────────────────────────────────────────────
    @classmethod
    def classic(cls, **overrides) -> "TurboConfig":
        """The frozen pre-2026-08-11 profile (``TURBO_PROFILE=classic``)."""
        return cls(**overrides)

    @classmethod
    def current_base4(cls, **overrides) -> "TurboConfig":
        """Current (post-2026-08-11) logic restricted to the base four.

        This is NOT the live current profile — the VIOLENCE tier's seven extra
        symbols are not in the cached universe — so the comparison isolates the
        *policy* changes (regime gate, shorts, MR-short, modern SELL
        confidence), not the pool change.  Use it as a directional comparison
        only, and label it as such wherever it is reported.
        """
        base = dict(
            legacy_sell_confidence=False,
            regime_gate=True,
            allow_shorts=True,
            mr_short=True,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def pessimistic(cls, **overrides) -> "TurboConfig":
        base = dict(slippage_pct=0.0002, slippage_abs=0.01, half_spread_pct=0.0001,
                    commission_per_share=0.005)
        base.update(overrides)
        return cls(**base)

    def zero_cost(self) -> "TurboConfig":
        return replace(self, slippage_pct=0.0, slippage_abs=0.0, half_spread_pct=0.0,
                       commission_per_share=0.0, commission_pct=0.0)


# ── per-bar signal candidates ──────────────────────────────────────────


def _per_session(close: pd.Series, sess_start: np.ndarray, fn) -> list:
    """Run *fn* on each session's bars separately and concatenate the results.

    The live loop fetches a 60-minute window, so an indicator's rolling history
    never crosses the overnight boundary: at 09:30 ET the momentum MA has no
    history at all.  Computing indicators on the continuous series would hand
    the first bars of each session indicator values the live trader never had —
    and because the highest-confidence signal in the window is often a tie, that
    changes which bar *wins* the window at the open.  Sessions are therefore
    evaluated independently.
    """
    bounds = list(np.nonzero(sess_start)[0]) + [len(close)]
    out: list = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.extend(fn(close.iloc[a:b]))
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Same RSI as the live ``_compute_rsi`` (plain rolling means)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _z_score(close: pd.Series, period: int = 20) -> pd.Series:
    """Same z-score as ``src.strategies.indicators.z_score``."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return (close - middle) / std


@dataclass
class _Candidate:
    """Per-bar best signal: direction (+1 BUY / -1 SELL), confidence, strategy."""

    conf: float = -1.0
    direction: int = 0
    strategy: str = ""


def _momentum_candidate(close: pd.Series, cfg: TurboConfig) -> list[_Candidate]:
    """Momentum signals, one candidate per bar (live ``_generate_momentum_signals``)."""
    ma = close.rolling(window=cfg.ma_period).mean()
    rsi = _rsi(close, cfg.rsi_period)
    vals = close.to_numpy(dtype=float)
    ma_v = ma.to_numpy(dtype=float)
    rsi_v = rsi.to_numpy(dtype=float)
    out: list[_Candidate] = []
    for i in range(len(vals)):
        if i == 0 or math.isnan(ma_v[i]) or math.isnan(rsi_v[i]):
            out.append(_Candidate())
            continue
        c, m, r = vals[i], ma_v[i], rsi_v[i]
        if c > m and r > cfg.rsi_threshold:
            ma_score = min(1.0, ((c - m) / (abs(m) + 1e-9)) * 25)
            rsi_score = max(0.0, (r - cfg.rsi_threshold) / 40.0)
            conf = round(0.3 + 0.7 * (ma_score + rsi_score) / 2, 4)
            out.append(_Candidate(min(1.0, max(0.0, conf)), 1, "momentum"))
            continue
        prev_ma = ma_v[i - 1]
        prev_c = vals[i - 1]
        if math.isnan(prev_ma):
            out.append(_Candidate())
            continue
        if (c < m and prev_c >= prev_ma) or (c < m and r < 60):
            dist_pct = abs(c - m) / (abs(m) + 1e-9)
            if cfg.legacy_sell_confidence:
                conf = min(1.0, dist_pct * 10)          # classic (PR #41)
            else:
                dist_score = min(1.0, dist_pct * 40)
                rsi_score = max(0.0, min(1.0, (60.0 - r) / 40.0))
                cross_bonus = 0.15 if (c < m and prev_c >= prev_ma) else 0.0
                conf = min(1.0, 0.3 + 0.7 * (0.5 * dist_score + 0.3 * rsi_score)
                           + cross_bonus)
            out.append(_Candidate(round(conf, 6), -1, "momentum"))
            continue
        out.append(_Candidate())
    return out


def _mr_candidate(close: pd.Series, cfg: TurboConfig) -> list[_Candidate]:
    """Mean-reversion signals, one candidate per bar (live MeanReversionStrategy)."""
    z = _z_score(close, cfg.mr_lookback).to_numpy(dtype=float)
    thr = abs(cfg.mr_entry_threshold)
    out: list[_Candidate] = []
    for zv in z:
        if math.isnan(zv):
            out.append(_Candidate())
            continue
        if zv < -thr:
            out.append(_Candidate(round(min(1.0, abs(zv) / (2.0 * thr)), 6), 1,
                                  "mean_reversion"))
        elif zv > thr:
            out.append(_Candidate(round(min(1.0, abs(zv) / (2.0 * thr)), 6), -1,
                                  "mean_reversion"))
        else:
            out.append(_Candidate())
    return out


def _window_best(values: np.ndarray, session_start: np.ndarray, window: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window argmax over *values* within each session.

    Returns ``(best_value, best_index)`` per bar where the window is the last
    ``window`` bars of the same session (the live fetch cannot see across the
    overnight boundary and never exceeds 60 minutes).  Ties keep the EARLIEST
    index, matching the live loop's strict ``>`` comparison.
    """
    n = len(values)
    best_v = np.full(n, -np.inf)
    best_i = np.full(n, -1, dtype=np.int64)
    dq: deque[int] = deque()
    for i in range(n):
        if session_start[i]:
            dq.clear()
        while dq and values[dq[-1]] < values[i]:
            dq.pop()
        dq.append(i)
        lo = i - window + 1
        if session_start[i]:
            lo = i                       # first bar of a session: window = {i}
        while dq and dq[0] < lo:
            dq.popleft()
        # the deque is monotonic with equal values retained, so its front is the
        # earliest bar attaining the window maximum.
        j = dq[0]
        best_v[i] = values[j]
        best_i[i] = j
    return best_v, best_i


# ── positions / result ─────────────────────────────────────────────────


@dataclass
class _Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_index: int
    stop: float
    target: float
    entry_conf: float
    entry_strategy: str


@dataclass
class TurboReplayResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    stats: dict
    config: TurboConfig
    daily_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    per_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    folds_monthly: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        s = self.stats
        return (f"trades={s['round_trips']} net=${s['pnl_after_costs']:,.0f} "
                f"({s['total_return']:+.1%}) zero-cost=${s['pnl_gross']:,.0f} "
                f"PF={s['profit_factor']:.2f} win={s['win_rate']:.1%} "
                f"maxDD={s['max_drawdown']:.1%}")


# ── engine ─────────────────────────────────────────────────────────────


class TurboReplay:
    """Bar-by-bar replay of the turbo rule set over cached 1m bars."""

    def __init__(self, frames: Mapping[str, pd.DataFrame], config: TurboConfig,
                 trade_window: Optional[tuple[str, str]] = None) -> None:
        self.config = config
        self.frames: dict[str, pd.DataFrame] = {
            s.upper(): f.sort_index() for s, f in frames.items() if len(f)
        }
        if trade_window is not None:
            lo, hi = pd.Timestamp(trade_window[0]), pd.Timestamp(trade_window[1])
            hi_excl = hi if hi.time() == pd.Timestamp(hi.date()).time() else hi
            self.frames = {s: f[(f.index >= lo) & (f.index < hi_excl + pd.Timedelta(days=1))]
                           for s, f in self.frames.items()}
        self._prep()

    # ── preparation ────────────────────────────────────────────────────
    def _prep(self) -> None:
        cfg = self.config
        self.o: dict[str, np.ndarray] = {}
        self.h: dict[str, np.ndarray] = {}
        self.l: dict[str, np.ndarray] = {}
        self.c: dict[str, np.ndarray] = {}
        self.ts: dict[str, np.ndarray] = {}
        self.day: dict[str, np.ndarray] = {}
        self.after_open: dict[str, np.ndarray] = {}
        self.sess_start: dict[str, np.ndarray] = {}
        self.best_dir: dict[str, np.ndarray] = {}
        self.best_conf: dict[str, np.ndarray] = {}
        self.best_idx: dict[str, np.ndarray] = {}
        self.best_strat: dict[str, np.ndarray] = {}
        self.best_bar: dict[str, np.ndarray] = {}
        for sym, df in self.frames.items():
            close = df["close"]
            self.o[sym] = df["open"].to_numpy(dtype=float)
            self.h[sym] = df["high"].to_numpy(dtype=float)
            self.l[sym] = df["low"].to_numpy(dtype=float)
            self.c[sym] = close.to_numpy(dtype=float)
            ts = df.index.to_numpy()
            self.ts[sym] = ts
            days = np.array([t.date() for t in df.index])
            self.day[sym] = days
            sess_start = np.empty(len(df), dtype=bool)
            sess_start[0] = True
            sess_start[1:] = days[1:] != days[:-1]
            self.sess_start[sym] = sess_start
            minute = np.array([t.hour * 60 + t.minute for t in df.index])
            self.after_open[sym] = minute

            mom = _per_session(close, sess_start, lambda c: _momentum_candidate(c, cfg))
            mr = _per_session(close, sess_start, lambda c: _mr_candidate(c, cfg))
            mom_v = np.array([x.conf for x in mom], dtype=float)
            mr_v = np.array([x.conf for x in mr], dtype=float)
            mom_bv, mom_bi = _window_best(mom_v, sess_start, cfg.lookback_bars)
            mr_bv, mr_bi = _window_best(mr_v, sess_start, cfg.lookback_bars)

            n = len(df)
            direction = np.zeros(n, dtype=np.int64)
            conf = np.full(n, -np.inf)
            strat = np.empty(n, dtype=object)
            strat[:] = ""
            idx = np.full(n, -1, dtype=np.int64)
            # mean-reversion signals are listed first in the live loop, so an
            # MR candidate wins a confidence tie against momentum.
            use_mr = mr_bv >= mom_bv
            mr_ok = mr_bv > -np.inf
            mom_ok = mom_bv > -np.inf
            pick_mr = use_mr & mr_ok
            pick_mom = (~pick_mr) & mom_ok
            for mask, bv, bi, src, name in ((pick_mr, mr_bv, mr_bi, mr, "mean_reversion"),
                                            (pick_mom, mom_bv, mom_bi, mom, "momentum")):
                ii = np.nonzero(mask)[0]
                for k in ii:
                    j = int(bi[k])
                    if j < 0:
                        continue
                    conf[k] = bv[k]
                    direction[k] = src[j].direction
                    strat[k] = name
                    idx[k] = j
            self.best_dir[sym] = direction
            self.best_conf[sym] = conf
            self.best_idx[sym] = idx
            self.best_strat[sym] = strat
            # the trailing window must hold >= min_bars bars (live `len(data) < 25`)
            win_len = np.empty(n, dtype=np.int64)
            run = 0
            for i in range(n):
                run = 1 if sess_start[i] else run + 1
                win_len[i] = min(run, cfg.lookback_bars)
            self.best_bar[sym] = win_len

    # ── helpers ────────────────────────────────────────────────────────
    def _slip(self, price: float) -> float:
        cfg = self.config
        return max(abs(price) * cfg.slippage_pct, cfg.slippage_abs) \
            + abs(price) * cfg.half_spread_pct

    def _market_fill(self, price: float, is_buy: bool) -> float:
        slip = self._slip(price)
        return price + slip if is_buy else price - slip

    def _fees(self, qty: float, price: float) -> float:
        cfg = self.config
        return float(qty * cfg.commission_per_share
                     + abs(qty * price) * cfg.commission_pct)

    def _equity(self) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            eq += pos.qty * self.last_close.get(sym, pos.entry_price)
        return float(eq)

    def _size(self, price: float) -> float:
        cfg = self.config
        if price <= 0:
            return 0.0
        equity = self._equity()
        notional = min(equity * cfg.position_size_pct, max(self.cash, 0.0) * cfg.bp_usage_pct)
        qty = notional / price
        return 0.0 if qty < cfg.min_entry_qty else qty

    # ── main loop ──────────────────────────────────────────────────────
    def run(self) -> TurboReplayResult:
        cfg = self.config
        symbols = [s.upper() for s in cfg.symbols if s.upper() in self.frames]
        if not symbols:
            raise ValueError("no configured symbol has a frame to replay")

        axis = np.unique(np.concatenate([self.ts[s] for s in symbols]))
        local: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sym in symbols:
            own = self.ts[sym]
            pos = np.searchsorted(own, axis).clip(0, len(own) - 1)
            local[sym] = (pos, own[pos] == axis)

        self.cash = float(cfg.initial_equity)
        self.positions: dict[str, _Position] = {}
        self.last_close: dict[str, float] = {}
        self.trades: list[dict] = []
        equity_rows: list[tuple[pd.Timestamp, float]] = []
        self.stats = {
            "symbols": list(symbols),
            "bars": int(len(axis)),
            "sessions": 0,
            "signals_seen": 0,
            "entries": 0,
            "entries_by_symbol": {},
            "skipped": {"min_bars": 0, "max_positions": 0, "cash": 0, "threshold": 0},
            "exits": {},
            "fees_paid": 0.0,
            "eod_flats": 0,
        }
        last_day = None
        session_done = False
        for t64 in axis:
            t = pd.Timestamp(t64)
            day = t.date()
            if last_day is None or day != last_day:
                self.stats["sessions"] += 1
                last_day = day
                session_done = False
            session_done = self._process_bar(t, local, axis, session_done, symbols)
            equity_rows.append((t, self._equity()))

        if self.positions:  # should not happen with eod_flat; never carry silently
            for sym in list(self.positions):
                t = pd.Timestamp(axis[-1])
                self._close(sym, float(self.last_close.get(sym, self.positions[sym].entry_price)),
                            t, self.positions[sym].entry_index, "forced_end")
        trades = pd.DataFrame(self.trades, columns=TRADE_COLUMNS)
        eq = pd.Series([v for _, v in equity_rows],
                       index=pd.DatetimeIndex([k for k, _ in equity_rows], name="ts"),
                       name="equity")
        result = TurboReplayResult(trades=trades, equity_curve=eq, stats=self.stats,
                                   config=cfg)
        self._finalise(result)
        return result

    def _process_bar(self, t: pd.Timestamp, local, axis, session_done: bool,
                     symbols: list[str]) -> bool:
        cfg = self.config
        barmin = t.hour * 60 + t.minute
        bar_symbols = []
        for sym in symbols:
            pos_arr, has = local[sym]
            i = int(np.searchsorted(axis, np.datetime64(t)))
            if i >= len(has) or not has[i]:
                continue
            j = int(pos_arr[i])
            bar_symbols.append((sym, j))

        # 1) intrabar stop / target on every open position
        for sym, j in bar_symbols:
            self.last_close[sym] = float(self.c[sym][j])
            pos = self.positions.get(sym)
            if pos is None:
                continue
            o, h, l = float(self.o[sym][j]), float(self.h[sym][j]), float(self.l[sym][j])
            hit_stop = l <= pos.stop
            hit_target = h >= pos.target
            if hit_stop:
                base = o if o <= pos.stop else pos.stop      # gap-aware
                self._close(sym, self._market_fill(base, is_buy=False), t, j, "stop")
            elif hit_target:
                base = max(o, pos.target)
                self._close(sym, self._market_fill(base, is_buy=False), t, j, "target")

        # 2) time exit (>= MAX_HOLD_MINUTES) and the mandatory EOD flatten
        eod_now = cfg.eod_flat and barmin >= cfg.eod_flat_minute
        if eod_now and not session_done:
            for sym in list(self.positions):
                if sym in self.last_close:
                    j = int(np.searchsorted(self.ts[sym], np.datetime64(t)).clip(0, len(self.ts[sym]) - 1))
                    self._close(sym, self._market_fill(self.last_close[sym], is_buy=False),
                                t, j, "eod")
                    self.stats["eod_flats"] += 1
            session_done = True
        elif not session_done:
            for sym, j in bar_symbols:
                pos = self.positions.get(sym)
                if pos is None:
                    continue
                held = (t - pos.entry_time).total_seconds() / 60.0
                if held >= cfg.max_hold_minutes:
                    self._close(sym, self._market_fill(float(self.c[sym][j]), is_buy=False),
                                t, j, "time")

        if session_done:
            return session_done

        # 3) signal pass — one entry per bar, symbols in live order
        entered = False
        for sym, j in bar_symbols:
            if entered and cfg.one_entry_per_bar:
                break
            conf = float(self.best_conf[sym][j])
            if conf < cfg.confidence_threshold:
                self.stats["skipped"]["threshold"] += 1
                continue
            if self.best_bar[sym][j] < cfg.min_bars:
                self.stats["skipped"]["min_bars"] += 1
                continue
            direction = int(self.best_dir[sym][j])
            strategy = str(self.best_strat[sym][j])
            self.stats["signals_seen"] += 1
            price = float(self.c[sym][j])
            if direction > 0:                                    # BUY
                if sym in self.positions:
                    continue
                if len(self.positions) >= cfg.max_positions:
                    self.stats["skipped"]["max_positions"] += 1
                    continue
                qty = self._size(price)
                if qty <= 0:
                    self.stats["skipped"]["cash"] += 1
                    continue
                fill = self._market_fill(price, is_buy=True)
                fee = self._fees(qty, fill)
                self.cash -= qty * fill + fee
                self.positions[sym] = _Position(
                    symbol=sym, qty=qty, entry_price=fill, entry_time=t, entry_index=j,
                    stop=fill * (1 - cfg.stop_loss_pct),
                    target=fill * (1 + cfg.take_profit_pct),
                    entry_conf=conf, entry_strategy=strategy)
                self.stats["entries"] += 1
                self.stats["entries_by_symbol"][sym] = \
                    self.stats["entries_by_symbol"].get(sym, 0) + 1
                self.stats["fees_paid"] += fee
                entered = True
            elif direction < 0:                                  # SELL
                pos = self.positions.get(sym)
                if pos is not None:
                    self._close(sym, self._market_fill(price, is_buy=False), t, j, "signal")
                    entered = False
        return session_done

    def _close(self, sym: str, price: float, t: pd.Timestamp, j: int, reason: str) -> None:
        pos = self.positions.pop(sym)
        fee = self._fees(pos.qty, price)
        self.cash += pos.qty * price - fee
        self.stats["fees_paid"] += fee
        self.stats["exits"][reason] = self.stats["exits"].get(reason, 0) + 1
        gross = (price - pos.entry_price) * pos.qty
        net = gross - fee - self._fees(pos.qty, pos.entry_price)
        self.trades.append({
            "symbol": sym, "side": "long", "qty": pos.qty,
            "entry_time": pos.entry_time, "entry_price": pos.entry_price,
            "exit_time": t, "exit_price": price,
            "hold_minutes": (t - pos.entry_time).total_seconds() / 60.0,
            "exit_reason": reason, "entry_conf": pos.entry_conf,
            "entry_strategy": pos.entry_strategy,
            "pnl_gross": gross, "fees": fee + self._fees(pos.qty, pos.entry_price),
            "pnl_after_costs": net,
            "ret_pct": (price / pos.entry_price - 1.0) if pos.entry_price else 0.0,
        })

    # ── result assembly ────────────────────────────────────────────────
    def _finalise(self, result: TurboReplayResult) -> None:
        cfg = self.config
        trades = result.trades
        equity = result.equity_curve
        stats = result.stats
        final_eq = float(equity.iloc[-1]) if len(equity) else cfg.initial_equity
        stats["final_equity"] = final_eq
        stats["initial_equity"] = float(cfg.initial_equity)
        stats["total_return"] = final_eq / cfg.initial_equity - 1.0
        stats["round_trips"] = int(len(trades))
        if len(trades):
            net = trades["pnl_after_costs"]
            gross = trades["pnl_gross"]
            wins = net[net > 0]
            losses = net[net <= 0]
            stats["pnl_after_costs"] = float(net.sum())
            stats["pnl_gross"] = float(gross.sum())
            stats["win_rate"] = float(len(wins)) / len(trades)
            stats["avg_win"] = float(wins.mean()) if len(wins) else 0.0
            stats["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
            gains = float(wins.sum())
            big_loss = abs(float(losses.sum()))
            stats["profit_factor"] = (gains / big_loss) if big_loss else float("inf")
            stats["break_even_win_rate"] = (
                abs(stats["avg_loss"]) / (stats["avg_win"] + abs(stats["avg_loss"]))
                if (stats["avg_win"] + abs(stats["avg_loss"])) > 0 else 0.0)
            stats["expectancy"] = float(net.mean())
            stats["best"] = float(net.max())
            stats["worst"] = float(net.min())
            stats["avg_hold_minutes"] = float(trades["hold_minutes"].mean())
            stats["median_hold_minutes"] = float(trades["hold_minutes"].median())
            stats["trades_per_session"] = (float(len(trades)) / stats["sessions"]
                                           if stats["sessions"] else 0.0)
        else:
            for k in ("pnl_after_costs", "pnl_gross", "win_rate", "avg_win", "avg_loss",
                      "profit_factor", "break_even_win_rate", "expectancy", "best",
                      "worst", "avg_hold_minutes", "median_hold_minutes",
                      "trades_per_session"):
                stats[k] = 0.0

        # drawdown / sharpe from the equity curve
        if len(equity) > 1:
            running = equity.cummax()
            dd = (equity - running) / running.replace(0, np.nan)
            stats["max_drawdown"] = float(abs(dd.min())) if dd.notna().any() else 0.0
            daily = equity.resample("1D").last().dropna()
            stats["daily_sharpe"] = _sharpe(daily.pct_change().dropna())
            daily_sessions = daily[equity.resample("1D").count().reindex(daily.index) > 0]
        else:
            stats["max_drawdown"] = 0.0
            stats["daily_sharpe"] = 0.0
            daily = pd.Series(dtype=float)
        session_last = equity.groupby(equity.index.date).last()
        result.daily_equity = session_last
        stats["session_sharpe"] = _sharpe(session_last.pct_change().dropna())
        stats["session_return"] = (session_last.iloc[-1] / cfg.initial_equity - 1.0
                                   if len(session_last) else 0.0)

        if len(trades):
            per = trades.groupby("symbol").agg(
                trades=("pnl_after_costs", "size"),
                pnl=("pnl_after_costs", "sum"),
                win_rate=("pnl_after_costs", lambda s: float((s > 0).mean())),
                gross=("pnl_gross", "sum"),
                best=("pnl_after_costs", "max"), worst=("pnl_after_costs", "min"),
            ).sort_values("pnl", ascending=False)
        else:
            per = pd.DataFrame()
        result.per_symbol = per

        if len(trades):
            t = trades.copy()
            t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M").astype(str)
            rows = []
            for month, grp in t.groupby("month"):
                net = grp["pnl_after_costs"]
                gains = float(net[net > 0].sum())
                blo = abs(float(net[net <= 0].sum()))
                rows.append({
                    "month": month, "trades": len(grp), "pnl": float(net.sum()),
                    "win_rate": float((net > 0).mean()),
                    "profit_factor": (gains / blo) if blo else float("inf"),
                    "return_pct": float(net.sum()) / cfg.initial_equity,
                })
            folds = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
        else:
            folds = pd.DataFrame()
        result.folds_monthly = folds
        stats["folds_positive"] = int((folds["pnl"] > 0).sum()) if len(folds) else 0
        stats["folds_total"] = int(len(folds))


def _sharpe(returns: pd.Series, periods: int = 252) -> float:
    if returns is None or len(returns) < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods))


# ── convenience entry points ───────────────────────────────────────────


def load_frames(symbols: Iterable[str], cache_dir=DEFAULT_CACHE,
                start: Optional[str] = None, end: Optional[str] = None
                ) -> dict[str, pd.DataFrame]:
    """Load cached RTH 1m bars for *symbols* (reuses the ScalpSet loader)."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frames = load_rth_1m(sym, cache_dir=cache_dir, start=start, end=end)
        if len(frames):
            out[sym.upper()] = frames
    return out


def replay(frames: Mapping[str, pd.DataFrame], config: TurboConfig,
           trade_window: Optional[tuple[str, str]] = None) -> TurboReplayResult:
    return TurboReplay(frames, config, trade_window=trade_window).run()
