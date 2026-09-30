"""Shared event loop for the stage-1 strategy search.

Every family in the search is *declarative*: it hands this engine

* a set of **instruments** (a single symbol, or a pair of legs opened together),
* a per-bar **entry direction** array (+1 long / −1 short / 0 none) evaluated at
  a bar's **close**,
* a per-bar **exit-now** array, also evaluated at a bar's close,

and the engine owns everything that the failures of the old rule sets were made
of — when an entry is allowed, how many, how big, and what it costs.

Fill conventions (no lookahead, stated once and pinned by a test):

* a signal evaluated on bar *t* is filled at bar *t+1*'s **open**, ± slippage;
  an entry is never filled across the overnight boundary (the fill bar must be
  in the same session as the signal bar),
* stops / targets / trailing stops are intrabar and **gap-aware**: if the bar
  opens beyond the trigger the fill is the bar's open, otherwise the trigger
  price; if a bar touches both the stop and the target the **stop** is taken
  (conservative).  A trailing stop ratchets on the bar's extreme, never on the
  close alone,
* time exits fill at the close of the bar that completes the hold,
* every position is flattened at the first bar at or after ``eod_flat_min``
  (15:30 ET by default, matching the live mandate) at that bar's close, and no
  new entry may be signalled after it,
* indicators are computed **per session** by the family modules (see
  ``features.py``); this engine never looks across the overnight boundary
  except through arrays a family deliberately builds from a completed prior
  session.

Sizing: ``fixed_notional`` puts a fixed $ notional on each new trade (reported
at $50k), ``equity_fraction`` reproduces the live engine's 50 %-of-equity,
95 %-of-cash convention.  Both are reported for every config, so a config
cannot look good merely by making fewer, bigger trades.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.replay_costs import CostModel

EOD_FLAT_MIN = 15 * 60 + 30      # 15:30 ET — the live mandatory flatten
MIN_ENTRY_QTY = 1.0


# ── instruments ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Leg:
    """One side of an instrument.  ``weight`` is relative notional."""

    symbol: str
    side: int          # +1 long, −1 short
    weight: float = 1.0


@dataclass(frozen=True)
class Instrument:
    """Something the engine can hold: one symbol, or a pair of legs at once."""

    key: str
    legs: tuple[Leg, ...]
    entry_dir: np.ndarray      # per global-axis bar, evaluated at the close
    exit_now: np.ndarray       # int8 per bar: 0 none, 1 exit long, −1 exit short, 2 either
    valid: np.ndarray          # True where every leg has a bar to trade
    kind: str = "single"
    atr_pct: Optional[np.ndarray] = None   # ATR/close per bar, for ATR-scaled stops


@dataclass(frozen=True)
class SearchConfig:
    """Every knob that decides what a config trades, and nothing else."""

    family: str
    name: str
    sizing: str = "fixed_notional"        # 'fixed_notional' | 'equity_fraction'
    initial_equity: float = 100_000.0
    position_size_pct: float = 0.50
    bp_usage_pct: float = 0.95
    notional_usd: float = 50_000.0
    max_positions: int = 2
    entry_start_min: int = 10 * 60        # minutes since midnight, ET
    entry_end_min: int = 14 * 60
    max_entries_per_session: int = 1
    min_minutes_between_entries: int = 0
    stop_pct: Optional[float] = None
    target_pct: Optional[float] = None
    trail_pct: Optional[float] = None
    time_exit_minutes: Optional[int] = None
    allow_short: bool = False
    eod_flat_min: int = EOD_FLAT_MIN
    extras: Mapping[str, object] = field(default_factory=dict)

    def params(self) -> dict:
        d = asdict(self)
        extras = d.pop("extras") or {}
        d.update(extras)
        return d


@dataclass
class SearchResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    stats: dict
    config: SearchConfig

    def summary(self) -> str:
        s = self.stats
        return (f"trips={s['round_trips']} net=${s['pnl_after_costs']:,.0f} "
                f"({s['total_return']:+.1%}) netbps={s['net_bps_per_trip']:+.2f} "
                f"PF={s['profit_factor']:.2f} win={s['win_rate']:.1%} "
                f"maxDD={s['max_drawdown']:.1%} folds={s['folds_positive']}/"
                f"{s['folds_total']}")


# ── market arrays ──────────────────────────────────────────────────────


class Market:
    """Aligned arrays on one global bar axis (union of every symbol's bars)."""

    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        if not frames:
            raise ValueError("Market needs at least one symbol frame")
        self.symbols = tuple(sorted(f.upper() for f in frames))
        frames = {s.upper(): f.sort_index() for s, f in frames.items() if len(f)}
        axis = np.unique(np.concatenate([f.index.to_numpy() for f in frames.values()]))
        self.axis = pd.DatetimeIndex(axis, name="ts")
        self.day = np.array([t.date() for t in self.axis])
        self.minute = np.array([t.hour * 60 + t.minute for t in self.axis])
        self.day_change = np.empty(len(self.axis), dtype=bool)
        if len(self.axis):
            self.day_change[0] = True
            self.day_change[1:] = self.day[1:] != self.day[:-1]
        self.o: dict[str, np.ndarray] = {}
        self.h: dict[str, np.ndarray] = {}
        self.l: dict[str, np.ndarray] = {}
        self.c: dict[str, np.ndarray] = {}
        self.present: dict[str, np.ndarray] = {}
        for sym, df in frames.items():
            r = df.reindex(self.axis)
            self.o[sym] = r["open"].to_numpy(dtype=float)
            self.h[sym] = r["high"].to_numpy(dtype=float)
            self.l[sym] = r["low"].to_numpy(dtype=float)
            self.c[sym] = r["close"].to_numpy(dtype=float)
            self.present[sym] = r["close"].notna().to_numpy()

    def n(self) -> int:
        return len(self.axis)

    def valid(self, legs: Sequence[Leg]) -> np.ndarray:
        ok = np.ones(self.n(), dtype=bool)
        for leg in legs:
            ok &= self.present[leg.symbol]
        return ok

    def session_index(self) -> np.ndarray:
        """Index of the first bar of each bar's session (for per-session holds)."""
        out = np.zeros(self.n(), dtype=np.int64)
        start = 0
        for k in range(self.n()):
            if self.day_change[k]:
                start = k
            out[k] = start
        return out


# ── engine ─────────────────────────────────────────────────────────────


@dataclass
class _OpenLeg:
    symbol: str
    side: int
    qty: float
    entry_fill: float
    entry_signal: float
    entry_slip: float
    entry_fee: float


@dataclass
class _OpenPos:
    key: str
    legs: list
    notional: float            # gross entry notional at the fill prices
    entry_time: pd.Timestamp
    entry_index: int
    side: int                  # +1 long, −1 short (single-leg reference side)
    ref_symbol: str            # symbol whose bars drive stop/target/trail
    ref_entry: float           # reference fill price at entry
    stop: Optional[float]
    target: Optional[float]
    extreme: float             # running high (long) / low (short) of the reference
    trail_pct: Optional[float] = None


class SearchReplay:
    """Bar-by-bar replay of one declarative config."""

    def __init__(self, market: Market, instruments: Mapping[str, Instrument],
                 config: SearchConfig, costs: CostModel) -> None:
        self.m = market
        self.instruments = dict(instruments)
        self.config = config
        self.costs = costs
        self.sess_start = market.session_index()

    # ── bookkeeping helpers ────────────────────────────────────────────
    def _open_book(self) -> None:
        self.cash = float(self.config.initial_equity)
        self.positions: dict[str, _OpenPos] = {}
        self.last_close: dict[str, float] = {}
        self.entries_by_session: dict[str, int] = {}
        self.last_entry_minute: dict[str, int] = {}
        self.session_locked: dict[str, int] = {}
        self.trades: list[dict] = []
        self.equity: list[float] = []
        self.stats: dict = {
            "entries": 0, "signals": 0, "exits": {},
            "skipped": {"max_positions": 0, "cash": 0, "min_qty": 0,
                        "session_cap": 0, "min_gap": 0, "late": 0,
                        "no_next_bar": 0, "gate": 0, "already_held": 0},
        }

    def _equity(self) -> float:
        eq = self.cash
        for pos in self.positions.values():
            for leg in pos.legs:
                px = self.last_close.get(leg.symbol)
                if px is not None and not math.isnan(px):
                    eq += leg.side * leg.qty * px
        return float(eq)

    def _leg_fills(self, inst: Instrument, k: int, is_entry: bool
                   ) -> dict[str, float]:
        """Fill price of each leg's symbol at bar *k* for an entry or an exit."""
        fills: dict[str, float] = {}
        for leg in inst.legs:
            raw = float(self.m.o[leg.symbol][k])
            if math.isnan(raw):
                return {}
            buy = (leg.side > 0) if is_entry else (leg.side < 0)
            fills[leg.symbol] = self.costs.fill_price(raw, is_buy=buy)
        return fills

    def _leg_target_notional(self) -> float:
        cfg = self.config
        if cfg.sizing == "fixed_notional":
            return float(cfg.notional_usd)
        return min(self._equity() * cfg.position_size_pct,
                   max(self.cash, 0.0) * cfg.bp_usage_pct)

    def _effective_pcts(self, inst: Instrument, sig_k: int) -> tuple:
        """Stop / target / trail percentages for an entry signalled at *sig_k*.

        A config may quote them as plain percentages, or — for the volatility
        families — as multiples of the ATR that was known on the signal bar.
        """
        cfg = self.config
        atr_mult = float(cfg.extras.get("stop_atr_mult") or 0.0) if cfg.extras else 0.0
        tgt_mult = float(cfg.extras.get("target_atr_mult") or 0.0) if cfg.extras else 0.0
        trl_mult = float(cfg.extras.get("trail_atr_mult") or 0.0) if cfg.extras else 0.0
        atr_pct = None
        if inst.atr_pct is not None and 0 <= sig_k < len(inst.atr_pct):
            v = float(inst.atr_pct[sig_k])
            atr_pct = None if math.isnan(v) else v
        stop = cfg.stop_pct
        target = cfg.target_pct
        trail = cfg.trail_pct
        if atr_pct is not None:
            if atr_mult:
                stop = atr_mult * atr_pct
            if tgt_mult:
                target = tgt_mult * atr_pct
            if trl_mult:
                trail = trl_mult * atr_pct
        elif atr_mult or tgt_mult or trl_mult:
            return None, None, None      # no ATR yet: the config cannot be priced
        return stop, target, trail

    def _try_open(self, inst: Instrument, k: int, sig_k: int) -> None:
        cfg = self.config
        fills = self._leg_fills(inst, k, is_entry=True)
        if not fills:
            return
        stop_pct, target_pct, trail_pct = self._effective_pcts(inst, sig_k)
        if (cfg.stop_pct is not None or cfg.extras.get("stop_atr_mult")) and stop_pct is None:
            self.stats["skipped"]["min_qty"] += 1
            return
        total_w = sum(abs(leg.weight) for leg in inst.legs) or 1.0
        notional = self._leg_target_notional()
        if notional <= 0:
            self.stats["skipped"]["cash"] += 1
            return
        open_legs: list[_OpenLeg] = []
        outlay = 0.0
        for leg in inst.legs:
            fill = fills[leg.symbol]
            leg_notional = notional * abs(leg.weight) / total_w
            qty = leg_notional / fill
            if qty < MIN_ENTRY_QTY:
                self.stats["skipped"]["min_qty"] += 1
                return
            raw = float(self.m.o[leg.symbol][k])
            slip = self.costs.slip_per_share(raw)
            fee = self.costs.fees(qty, fill)
            outlay += leg.side * (qty * fill) + fee
            open_legs.append(_OpenLeg(symbol=leg.symbol, side=leg.side, qty=qty,
                                      entry_fill=fill, entry_signal=raw,
                                      entry_slip=slip, entry_fee=fee))
        if outlay > self.cash:
            self.stats["skipped"]["cash"] += 1
            return
        for ol in open_legs:
            self.cash -= ol.side * ol.qty * ol.entry_fill + ol.entry_fee
        primary = open_legs[0]
        single = inst.kind == "single"
        ref_fill = fills[inst.legs[0].symbol]
        stop = target = None
        if single and stop_pct is not None:
            stop = (ref_fill * (1 - stop_pct) if primary.side > 0
                    else ref_fill * (1 + stop_pct))
        if single and target_pct is not None:
            target = (ref_fill * (1 + target_pct) if primary.side > 0
                      else ref_fill * (1 - target_pct))
        self.positions[inst.key] = _OpenPos(
            key=inst.key, legs=open_legs,
            notional=sum(ol.qty * ol.entry_fill for ol in open_legs),
            entry_time=self.m.axis[k], entry_index=k,
            side=primary.side if single else 0,
            ref_symbol=primary.symbol, ref_entry=ref_fill,
            stop=stop, target=target, extreme=ref_fill,
            trail_pct=trail_pct if single else None)
        self.stats["entries"] += 1
        self.entries_by_session[inst.key] = self.entries_by_session.get(inst.key, 0) + 1
        self.last_entry_minute[inst.key] = int(self.m.minute[k])

    def _close(self, key: str, k: int, fills: dict, reason: str,
               at_time: Optional[pd.Timestamp] = None) -> None:
        pos = self.positions.pop(key)
        gross = 0.0
        fees = 0.0
        drag = 0.0
        for leg in pos.legs:
            fill = fills[leg.symbol]
            raw = float(self.m.o[leg.symbol][k])
            exit_slip = self.costs.slip_per_share(raw)
            exit_fee = self.costs.fees(leg.qty, fill)
            self.cash += leg.side * leg.qty * fill - exit_fee
            gross += leg.side * (fill - leg.entry_fill) * leg.qty
            fees += leg.entry_fee + exit_fee
            drag += leg.qty * (leg.entry_slip + exit_slip) + leg.entry_fee + exit_fee
        net = gross - fees
        t = at_time if at_time is not None else self.m.axis[k]
        self.stats["exits"][reason] = self.stats["exits"].get(reason, 0) + 1
        self.trades.append({
            "instrument": key,
            "symbols": ",".join(f"{l.symbol}{'+' if l.side > 0 else '-'}" for l in pos.legs),
            "legs": len(pos.legs),
            "notional": pos.notional,
            "entry_time": pos.entry_time, "exit_time": t,
            "entry_price": pos.legs[0].entry_fill,
            "exit_price": fills[pos.legs[0].symbol],
            "hold_minutes": (t - pos.entry_time).total_seconds() / 60.0,
            "exit_reason": reason,
            "pnl_gross": gross, "fees": fees, "cost_drag": drag,
            "pnl_after_costs": net,
            "ret_pct": net / pos.notional if pos.notional else 0.0,
            "net_bps": (net / pos.notional * 1e4) if pos.notional else 0.0,
        })

    # ── the loop ───────────────────────────────────────────────────────
    def run(self) -> SearchResult:
        cfg = self.config
        self._open_book()
        n = self.m.n()
        pending_entry: dict[str, int] = {}
        pending_exit: dict[str, str] = {}
        keys = list(self.instruments)
        for k in range(n):
            minute = int(self.m.minute[k])
            new_session = bool(self.m.day_change[k])
            for sym, px in ((s, self.m.c[s][k]) for s in self.m.symbols):
                if not math.isnan(px):
                    self.last_close[sym] = px
            if new_session:
                self.entries_by_session.clear()
                self.last_entry_minute.clear()
                # a position never survives the session: EOD flatten already ran
            for key in keys:
                inst = self.instruments[key]
                if not bool(inst.valid[k]):
                    continue
                prev = k - 1
                same_session = prev >= 0 and not new_session
                # 1) fill an exit signalled on the previous bar
                if key in pending_exit and key in self.positions:
                    reason = pending_exit.pop(key)
                    fills = self._leg_fills(inst, k, is_entry=False)
                    if fills:
                        self._close(key, k, fills, reason)
                    else:
                        pending_exit[key] = reason
                # 2) fill an entry signalled on the previous bar
                if key in pending_entry:
                    sig_k = pending_entry.pop(key)
                    if same_session and key not in self.positions:
                        self._open_or_count(inst, k, sig_k)
                # 3) intrabar management (stops / targets / trailing)
                if key in self.positions:
                    self._manage(inst, k)
                # 4) mandatory end-of-day flatten
                if minute >= cfg.eod_flat_min and key in self.positions:
                    fills = {l.symbol: self.costs.fill_price(float(self.m.c[l.symbol][k]),
                                                             is_buy=l.side < 0)
                             for l in inst.legs}
                    self._close(key, k, fills, "eod", at_time=self.m.axis[k])
                if minute >= cfg.eod_flat_min:
                    continue
                # 5) signal pass on this bar's close -> fill on the next bar
                if key in self.positions:
                    want = int(inst.exit_now[k])
                    pos = self.positions.get(key)
                    if pos is not None and (want == 2 or want == pos.side):
                        pending_exit[key] = "signal"
                    if cfg.time_exit_minutes is not None and key in self.positions:
                        pos = self.positions.get(key)
                        if pos is not None and (k - pos.entry_index) >= cfg.time_exit_minutes:
                            raw = float(self.m.c[inst.legs[0].symbol][k])
                            if all(not math.isnan(float(self.m.c[l.symbol][k]))
                                   for l in inst.legs):
                                fills = {l.symbol: self.costs.fill_price(raw, is_buy=l.side < 0)
                                         for l in inst.legs}
                                self._close(key, k, fills, "time")
                    continue
                if key in pending_entry:
                    continue
                direction = int(inst.entry_dir[k])
                if direction == 0:
                    continue
                if direction < 0 and not cfg.allow_short:
                    continue
                self.stats["signals"] += 1
                if not (cfg.entry_start_min <= minute <= cfg.entry_end_min):
                    self.stats["skipped"]["gate"] += 1
                    continue
                if len(self.positions) >= cfg.max_positions:
                    self.stats["skipped"]["max_positions"] += 1
                    continue
                if self.entries_by_session.get(key, 0) >= cfg.max_entries_per_session:
                    self.stats["skipped"]["session_cap"] += 1
                    continue
                last = self.last_entry_minute.get(key)
                if last is not None and minute - last < cfg.min_minutes_between_entries:
                    self.stats["skipped"]["min_gap"] += 1
                    continue
                if minute > cfg.entry_end_min:
                    self.stats["skipped"]["late"] += 1
                    continue
                pending_entry[key] = k
            self.equity.append(self._equity())
        result = SearchResult(trades=self._trades_df(), equity_curve=self._equity_series(),
                              stats=self._stats(), config=cfg)
        return result

    def _open_or_count(self, inst: Instrument, k: int, sig_k: int) -> None:
        """Open unless a gate refuses; counts mirror the signal-path skips."""
        cfg = self.config
        if len(self.positions) >= cfg.max_positions:
            self.stats["skipped"]["max_positions"] += 1
            return
        self._try_open(inst, k, sig_k)

    def _manage(self, inst: Instrument, k: int) -> None:
        """Intrabar stop / target / trailing-stop for a single-leg position."""
        cfg = self.config
        pos = self.positions.get(inst.key)
        if pos is None or pos.side == 0:
            return
        sym = pos.ref_symbol
        o = float(self.m.o[sym][k])
        h = float(self.m.h[sym][k])
        low = float(self.m.l[sym][k])
        if math.isnan(o) or math.isnan(h) or math.isnan(low):
            return
        if pos.trail_pct is not None:
            if pos.side > 0:
                pos.extreme = max(pos.extreme, h)
                trail = pos.extreme * (1 - pos.trail_pct)
                pos.stop = max(pos.stop, trail) if pos.stop is not None else trail
            else:
                pos.extreme = min(pos.extreme, low)
                trail = pos.extreme * (1 + pos.trail_pct)
                pos.stop = min(pos.stop, trail) if pos.stop is not None else trail
        stop, target = pos.stop, pos.target
        hit_stop = stop is not None and (low <= stop if pos.side > 0 else h >= stop)
        hit_target = target is not None and (h >= target if pos.side > 0
                                             else low <= target)
        if hit_stop:
            price = min(o, stop) if pos.side > 0 else max(o, stop)
            fills = {l_.symbol: self.costs.fill_price(price, is_buy=l_.side < 0)
                     for l_ in pos.legs}
            self._close(inst.key, k, fills,
                        "trail" if pos.trail_pct is not None else "stop")
        elif hit_target:
            price = max(o, target) if pos.side > 0 else min(o, target)
            fills = {l_.symbol: self.costs.fill_price(price, is_buy=l_.side < 0)
                     for l_ in pos.legs}
            self._close(inst.key, k, fills, "target")

    # ── result assembly ────────────────────────────────────────────────
    def _trades_df(self) -> pd.DataFrame:
        cols = ["instrument", "symbols", "legs", "notional", "entry_time",
                "exit_time", "entry_price", "exit_price", "hold_minutes",
                "exit_reason", "pnl_gross", "fees",
                "cost_drag", "pnl_after_costs", "ret_pct", "net_bps"]
        return pd.DataFrame(self.trades, columns=cols)

    def _equity_series(self) -> pd.Series:
        return pd.Series(self.equity, index=self.m.axis, name="equity")

    def _stats(self) -> dict:
        cfg = self.config
        s = dict(self.stats)
        trades = self._trades_df()
        eq = self._equity_series()
        s["round_trips"] = int(len(trades))
        s["initial_equity"] = float(cfg.initial_equity)
        final = float(eq.iloc[-1]) if len(eq) else cfg.initial_equity
        s["final_equity"] = final
        s["total_return"] = final / cfg.initial_equity - 1.0
        if len(eq):
            peak = eq.cummax()
            s["max_drawdown"] = float(((eq / peak) - 1.0).min())
        else:
            s["max_drawdown"] = 0.0
        if len(trades):
            net = trades["pnl_after_costs"]
            gross = trades["pnl_gross"]
            wins = net[net > 0]
            losses = net[net <= 0]
            s["pnl_after_costs"] = float(net.sum())
            s["pnl_gross_same_fills"] = float(gross.sum())
            s["pnl_zero_cost_same_fills"] = float((gross + trades["cost_drag"]).sum())
            s["cost_drag_total"] = float(trades["cost_drag"].sum())
            s["cost_drag_per_trip"] = float(trades["cost_drag"].mean())
            s["fees_paid"] = float(trades["fees"].sum())
            s["win_rate"] = float(len(wins)) / len(trades)
            s["break_even_win_rate"] = self._break_even(trades)
            gross_win = float(gross[gross > 0].sum())
            gross_loss = float(-gross[gross <= 0].sum())
            s["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf")
            s["expectancy"] = float(net.mean())
            bps = trades["net_bps"].to_numpy(dtype=float)
            s["net_bps_per_trip"] = float(np.mean(bps))
            s["net_bps_per_trip_sd"] = float(np.std(bps, ddof=1)) if len(bps) > 1 else 0.0
            s["net_bps_t_stat"] = (s["net_bps_per_trip"] / s["net_bps_per_trip_sd"]
                                   * math.sqrt(len(bps))) if s["net_bps_per_trip_sd"] > 0 else 0.0
            s["notional_per_trip"] = float(trades["notional"].mean())
            s["avg_hold_minutes"] = float(trades["hold_minutes"].mean())
            s["median_hold_minutes"] = float(trades["hold_minutes"].median())
        else:
            for k in ("pnl_after_costs", "pnl_gross_same_fills",
                      "pnl_zero_cost_same_fills", "cost_drag_total", "fees_paid"):
                s[k] = 0.0
            s.update({"cost_drag_per_trip": 0.0, "win_rate": 0.0,
                      "break_even_win_rate": 0.0, "profit_factor": 0.0,
                      "expectancy": 0.0, "net_bps_per_trip": 0.0,
                      "net_bps_per_trip_sd": 0.0, "net_bps_t_stat": 0.0,
                      "notional_per_trip": 0.0, "avg_hold_minutes": 0.0,
                      "median_hold_minutes": 0.0})
        n_sessions = len(np.unique(self.m.day)) if len(self.m.day) else 0
        s["sessions"] = int(n_sessions)
        s["trips_per_session"] = (len(trades) / n_sessions) if n_sessions else 0.0
        return s

    @staticmethod
    def _break_even(trades: pd.DataFrame) -> float:
        """Win rate at which the *average* win pays for the *average* loss."""
        net = trades["pnl_after_costs"]
        wins = net[net > 0]
        losses = net[net <= 0]
        if len(wins) == 0 or len(losses) == 0:
            return float("nan")
        avg_w, avg_l = float(wins.mean()), float(-losses.mean())
        return avg_l / (avg_w + avg_l) if (avg_w + avg_l) > 0 else float("nan")


def monthly_folds(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month net / gross P&L and trip count."""
    if not len(trades):
        return pd.DataFrame(columns=["month", "round_trips", "pnl_after_costs",
                                     "pnl_gross_same_fills", "net_bps_per_trip"])
    t = trades.copy()
    if "pnl_gross" not in t.columns:
        t["pnl_gross"] = t.get("pnl_gross_same_fills", 0.0)
    t["month"] = pd.to_datetime(t["exit_time"]).dt.strftime("%Y-%m")
    g = t.groupby("month", sort=True)
    out = pd.DataFrame({
        "round_trips": g.size(),
        "pnl_after_costs": g["pnl_after_costs"].sum(),
        "pnl_gross_same_fills": g["pnl_gross"].sum(),
        "net_bps_per_trip": g["net_bps"].mean(),
    }).reset_index()
    return out


def per_symbol(trades: pd.DataFrame) -> pd.DataFrame:
    if not len(trades):
        return pd.DataFrame(columns=["instrument", "round_trips", "pnl_after_costs",
                                     "net_bps_per_trip"])
    g = trades.groupby("instrument", sort=True)
    return pd.DataFrame({
        "round_trips": g.size(),
        "pnl_after_costs": g["pnl_after_costs"].sum(),
        "net_bps_per_trip": g["net_bps"].mean(),
        "hold_minutes": g["hold_minutes"].mean(),
    }).reset_index()


def run_search(market: Market, instruments: Mapping[str, Instrument],
               config: SearchConfig, costs: CostModel) -> SearchResult:
    return SearchReplay(market, instruments, config, costs).run()
