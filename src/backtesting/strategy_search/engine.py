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
  (15:30 ET by default, matching the live mandate) **at which every leg of
  every open position has a bar**, at that bar's close, and no new entry may be
  signalled after ``eod_flat_min``.  The flatten minute is therefore one minute
  per session, shared by the whole book (L11.1); on a normal session the first
  such bar *is* 15:30, so the rule is a no-op there.  A leg with no bar never
  gets a fabricated price — a carried-forward close is not a fill (L11.2) — so
  the flatten waits for a real bar and a session that ends with a position still
  open is **fatal**, never a position carried overnight.  Each trip records
  whether its flatten was delayed and by how many minutes (L11.6),
* indicators are computed **per session** by the family modules (see
  ``features.py``); this engine never looks across the overnight boundary
  except through arrays a family deliberately builds from a completed prior
  session.

**Leg-fill symmetry (stage 2 §1).**  Both legs of a pair are priced by the same
two functions and nothing else: ``_leg_fills`` opens (buying a leg whose
effective side is long, selling one whose effective side is short) and
``_exit_fills`` closes (buying back a short leg, selling a long one).  Every
exit path — a signalled exit, the mandatory EOD flatten, a time exit, a stop
and a target — goes through ``_exit_fills``, and each charged the same
``max(slippage_pct × price, slippage_abs)`` adverse slippage on the same bar
under the same t+1 convention.  Both legs therefore pay the toll twice per
round trip, and the mirror direction (+1 ↔ −1) pays exactly the same toll on
the same tape.  ``tests/test_strategy_search_engine.py`` pins this with a
synthetic flat-pair fixture that must lose two full tolls and never book a
profit.

Sizing: ``fixed_notional`` puts a fixed $ notional on each new trade (reported
at $50k), ``equity_fraction`` reproduces the live engine's 50 %-of-equity,
95 %-of-cash convention.  Both are reported for every config, so a config
cannot look good merely by making fewer, bigger trades.

**Cost columns, stated once (A2, round-2 gate audit).**  Every trade carries
four cost columns, and they are not interchangeable:

* ``slip_drag`` — slippage only: ``Σ qty·(entry_slip + exit_slip)``, where each
  slip is ``CostModel.slip_per_share`` of the **price that fill was made at**
  (the signal bar's open for an entry, and for an exit the bar open, the bar
  close, or the stop/target level the exit actually filled at — A1);
* ``fees`` — commissions only, charged on both fills;
* ``cost_drag`` — ``slip_drag + fees``, the whole toll (kept because every
  existing artefact prints it);
* ``pnl_gross`` — P&L at the **fills actually charged**, i.e. after slippage and
  before fees: ``pnl_after_costs = pnl_gross − fees``;
* ``pnl_gross_precost`` — P&L at the **reference prices** of those same fills,
  ``pnl_gross + slip_drag``.  This is the zero-cost net at the same fills, so it
  is what ``pnl_zero_cost_same_fills`` sums, and
  ``pnl_after_costs = pnl_gross_precost − slip_drag − fees``.

The old zero-cost column was ``pnl_gross + cost_drag``, which added the
commissions a *second* time — invisible at baseline (fees are 0) and
optimistically wrong at every fee-charging level.  ``reconcile_zero_cost`` stays
the independent guard: it compares this reconstruction against a real replay
with every cost switched off.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.replay_costs import CostModel

EOD_FLAT_MIN = 15 * 60 + 30      # 15:30 ET — the live mandatory flatten
#: The only declared EOD-flatten values (P2): the pinned 15:30 and the 15:25
#: neighbour cell.  Any other value is refused by the builder — an edge that
#: only exists in the last five minutes is an artefact of where the flatten is.
EOD_PINNED_MIN = EOD_FLAT_MIN
EOD_FLAT_NEIGHBOURS = (15 * 60 + 25, 15 * 60 + 30)
MIN_ENTRY_QTY = 1.0
MAX_HOLD_MINUTES = 390.0         # one RTH session; a longer hold is a bug
#: **The declared gross-leverage limit** (R3-7 E6): the *gross* notional of one
#: basket, ``Σ|qty·fill|``, may never exceed ``initial_equity`` — i.e. the book
#: runs at 100 % leverage at most.  The check used to be the **net** outlay,
#: which a long+short basket passes at ~$0 of outlay while opening two full
#: notionals of gross exposure; a $100k account could then carry an unbounded
#: gross book, and the two-leg families (F, and any future rank rotation) are
#: exactly the shape that does it.
MAX_GROSS_LEVERAGE = 1.0
#: Skip counters every run carries.  A silent drop is not allowed: each one of
#: these is incremented on a counted signal, and every run asserts
#: ``entries + Σ skipped == signals`` before it returns.
SKIP_KEYS = ("max_positions", "cash", "gross_leverage", "min_qty", "session_cap",
             "min_gap", "late", "gate", "allow_short", "no_fill", "no_atr",
             "stale_entry", "already_held", "survived_session")


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
        """The config as a plain dict, with ``extras`` a **nested** key (R3-7 E2).

        ``extras`` used to be merged *over* the real fields, so an ``extras``
        entry could **shadow** a real field: two configs that differ only in an
        extras entry produced the same ``params()`` — and therefore hashed alike,
        and were recorded alike.  That is the channel the stage-2 defect came
        through (a knob the engine read that the artefact never recorded), so it
        must stay distinguishable in the config's own identity.  Callers that
        want the declared surface read ``params()["extras"]`` explicitly.
        """
        return asdict(self)


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
    direction: int = 1         # the signal direction this position was opened with
    trail_pct: Optional[float] = None
    #: Value of ``exit_now`` that closes this position.  A single-leg position
    #: exits when the family signals *its own* side; a pair's ``side`` is 0 by
    #: construction, so it must be compared against the signal direction, never
    #: against 0 (which would exit a pair on a *no-signal* bar and hold it
    #: through a signalled exit).
    exit_side: int = 0


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
            "identity_max_abs_residual": 0.0,
            "skipped": {k: 0 for k in SKIP_KEYS},
        }
        #: L11.6 — the mandatory flatten may fire *later* than ``eod_flat_min``
        #: when a leg of an open position has no bar on the flatten minute
        #: (L11.1).  Both counters ride in every run record: a delayed flatten
        #: that cannot be seen in the record is an undeclared degree of freedom.
        self.stats["eod_flatten_delayed_trips"] = 0
        self.stats["eod_flatten_max_delay_min"] = 0
        #: Axis index of the first bar at/after ``eod_flat_min`` in the current
        #: session at which the flatten could **not** be priced; ``None`` when
        #: it was priced, or when no position was open.  It is what separates
        #: "the flatten minute never arrived" (fail-closed, L11.1) from "the
        #: flatten never ran" (a bug, the pre-existing guard).
        self.eod_wait_k: Optional[int] = None
        #: True once the current session has a bar at/after ``eod_flat_min`` at
        #: all.  A session whose tape stops before the flatten minute can never
        #: price the flatten either, so both this and ``eod_wait_k`` feed the
        #: same fail-closed decision.
        self.eod_saw_flat_bar = False

    def _equity(self) -> float:
        eq = self.cash
        for pos in self.positions.values():
            for leg in pos.legs:
                px = self.last_close.get(leg.symbol)
                if px is not None and not math.isnan(px):
                    eq += leg.side * leg.qty * px
        return float(eq)

    def _leg_fills(self, inst: Instrument, k: int, is_entry: bool,
                   direction: int = 1) -> dict[tuple[int, str], float]:
        """Fill price of each leg at bar *k* for an entry or an exit.

        Keyed by ``(leg_index, symbol)``, **never by symbol alone**: two legs of
        one instrument may be the same symbol (a same-symbol pair), and a
        symbol-keyed dict silently collapses them into one fill — the cash then
        cancels to exactly zero toll for that symbol while ``cost_drag`` still
        books both legs' theoretical toll, fabricating a profit equal to the
        cancelled toll.  The index is what keeps the two legs distinct.

        *direction* mirrors the declared legs: a −1 signal on ``Leg(A,+1)``
        enters **short** A, so its fill must be the sell price.  Getting this
        wrong books the slippage of a short trade as a profit.
        """
        fills: dict[tuple[int, str], float] = {}
        for i, leg in enumerate(inst.legs):
            raw = float(self.m.o[leg.symbol][k])
            if math.isnan(raw):
                return {}
            side = int(leg.side) * int(direction)
            buy = (side > 0) if is_entry else (side < 0)
            fills[(i, leg.symbol)] = self.costs.fill_price(raw, is_buy=buy)
        return fills

    def _exit_fills(self, pos: _OpenPos, k: int, price=None
                    ) -> tuple[dict[tuple[int, str], float],
                               dict[tuple[int, str], float]]:
        """Fill prices that **close** *pos* at bar *k* — every exit path's only door.

        ``pos.legs`` carry the **actual** sides (the entry side is already
        mirrored by the signal direction), so a short leg is *bought back* and a
        long leg *sold*.  That is the mirror of ``_leg_fills`` on the entry
        side; getting it wrong prices a short's exit as a sell, which books the
        slippage of the exit as a profit and turns a flat pair trip into a
        cost-free (or positive) one.

        *price* is ``None`` for each leg's own bar **open** (the t+1 fill
        convention), a ``float`` for one shared level (a stop or target), or a
        mapping of symbol → price (the per-leg **close**, used by the mandatory
        EOD flatten and the time exit).  The result is keyed by
        ``(leg_index, symbol)`` for the same reason as :meth:`_leg_fills`.

        Returns ``(fills, refs)``: ``refs[(i, symbol)]`` is the **price that
        leg's fill was made at** (the bar open, the shared level, or the close),
        which is the price its slippage must be charged on.  Charging it on the
        exit bar's *open* instead — whatever the fill was — books a cost the
        fills never charged, and for a pair it does not cancel; that is the A1
        defect in the round-2 gate audit, and it is why the engine's own
        zero-cost reconstruction disagreed with a real zero-cost replay on any
        non-flat tape priced above the 1¢ slippage floor.  A leg with no price
        at *k* returns ``({}, {})``: the caller reads that as "cannot be priced
        here", never as a fill.
        """
        fills: dict[tuple[int, str], float] = {}
        refs: dict[tuple[int, str], float] = {}
        for i, leg in enumerate(pos.legs):
            if isinstance(price, MappingABC):
                raw = float(price[leg.symbol])
            elif price is not None:
                raw = float(price)
            else:
                raw = float(self.m.o[leg.symbol][k])
            if math.isnan(raw):
                return {}, {}
            fills[(i, leg.symbol)] = self.costs.fill_price(raw, is_buy=leg.side < 0)
            refs[(i, leg.symbol)] = raw
        return fills, refs

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

    def _try_open(self, inst: Instrument, k: int, sig_k: int, direction: int) -> None:
        """Open the instrument with *direction* (+1 as declared, −1 mirrored).

        A short entry is not a separate instrument: the legs' declared sides are
        multiplied by the signal direction, so a −1 signal on
        ``Leg(A,+1) + Leg(B,−1)`` opens short A / long B.
        """
        cfg = self.config
        fills = self._leg_fills(inst, k, is_entry=True, direction=direction)
        if not fills:
            self.stats["skipped"]["no_fill"] += 1
            return
        stop_pct, target_pct, trail_pct = self._effective_pcts(inst, sig_k)
        if (cfg.stop_pct is not None or cfg.extras.get("stop_atr_mult")) and stop_pct is None:
            # the config asked for an ATR-scaled stop but the ATR of the signal
            # bar is not known yet: the entry cannot be priced, and it is *this*
            # that is skipped — not a minimum-quantity refusal.
            self.stats["skipped"]["no_atr"] += 1
            return
        total_w = sum(abs(leg.weight) for leg in inst.legs) or 1.0
        notional = self._leg_target_notional()
        if notional <= 0:
            self.stats["skipped"]["cash"] += 1
            return
        open_legs: list[_OpenLeg] = []
        outlay = 0.0
        gross = 0.0
        for i, leg in enumerate(inst.legs):
            fill = fills[(i, leg.symbol)]
            side = int(leg.side) * int(direction)
            leg_notional = notional * abs(leg.weight) / total_w
            qty = leg_notional / fill
            if qty < MIN_ENTRY_QTY:
                self.stats["skipped"]["min_qty"] += 1
                return
            raw = float(self.m.o[leg.symbol][k])
            slip = self.costs.slip_per_share(raw)
            fee = self.costs.fees(qty, fill)
            outlay += side * (qty * fill) + fee
            gross += abs(qty * fill)
            open_legs.append(_OpenLeg(symbol=leg.symbol, side=side, qty=qty,
                                      entry_fill=fill, entry_signal=raw,
                                      entry_slip=slip, entry_fee=fee))
        # **Gross** leverage first (R3-7 E6): a basket whose two sides offset
        # passes a net-cash check at ~$0 of outlay while carrying two full
        # notionals of exposure.  The declared limit is 100 % of
        # ``initial_equity``, reported per basket and in the run stats.
        if gross > MAX_GROSS_LEVERAGE * float(cfg.initial_equity):
            self.stats["skipped"]["gross_leverage"] += 1
            return
        if outlay > self.cash:
            self.stats["skipped"]["cash"] += 1
            return
        for ol in open_legs:
            self.cash -= ol.side * ol.qty * ol.entry_fill + ol.entry_fee
        primary = open_legs[0]
        single = inst.kind == "single"
        ref_fill = fills[(0, inst.legs[0].symbol)]
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
            stop=stop, target=target, extreme=ref_fill, direction=int(direction),
            trail_pct=trail_pct if single else None,
            exit_side=primary.side if single else int(direction))
        self.stats["entries"] += 1
        self.entries_by_session[inst.key] = self.entries_by_session.get(inst.key, 0) + 1
        self.last_entry_minute[inst.key] = int(self.m.minute[k])

    def _close(self, key: str, k: int, fills: dict, refs: dict, reason: str,
               at_time: Optional[pd.Timestamp] = None) -> None:
        """Book the closing trade.  *refs* is **required** (A1).

        ``refs[(i, symbol)]`` is the price that leg's exit fill was made at, as
        returned by :meth:`_exit_fills`.  It is a positional argument with no
        default on purpose: re-deriving it from bar *k*'s open is exactly the
        defect that priced four of the five exit paths' slippage off a price the
        fill was never made at.
        """
        pos = self.positions.pop(key)
        unpriced = sorted(i for i in fills if i not in refs)
        if unpriced:
            raise RuntimeError(
                f"{key}: _close got fills for {unpriced} with no reference "
                f"price — the price each exit fill was made at is not optional")
        gross = 0.0
        slip_drag = 0.0
        fees = 0.0
        gross_notional = 0.0
        net_outlay = 0.0
        detail: list[str] = []
        for i, leg in enumerate(pos.legs):
            fill = float(fills[(i, leg.symbol)])
            raw = float(refs[(i, leg.symbol)])
            exit_slip = self.costs.slip_per_share(raw)
            exit_fee = self.costs.fees(leg.qty, fill)
            self.cash += leg.side * leg.qty * fill - exit_fee
            gross += leg.side * (fill - leg.entry_fill) * leg.qty
            slip_drag += leg.qty * (leg.entry_slip + exit_slip)
            fees += leg.entry_fee + exit_fee
            gross_notional += abs(leg.qty * leg.entry_fill)
            net_outlay += leg.side * (leg.qty * leg.entry_fill) + leg.entry_fee
            detail.append(f"{i}|{leg.symbol}|{leg.side:+d}|{leg.qty:.12f}|"
                          f"{leg.entry_fill:.12f}|{fill:.12f}")
        # ``cost_drag`` stays the whole toll (slippage + fees); ``slip_drag`` is
        # the slippage half on its own so a reader can reconstruct the zero-cost
        # column without adding the commissions twice (A2).
        drag = slip_drag + fees
        #: P&L at the **reference prices** of the same fills: no slippage, no
        #: fees, same entry and exit bars.  This is the zero-cost net at the same
        #: fills, so it is what ``pnl_zero_cost_same_fills`` sums.
        precost = gross + slip_drag
        net = gross - fees
        # Per-trip accounting identity, recomputed leg by leg from the recorded
        # fills: pnl_after_costs == Σ leg.side·(exit_fill − entry_fill)·qty − fees.
        identity = sum(leg.side * (float(fills[(i, leg.symbol)]) - leg.entry_fill)
                       * leg.qty for i, leg in enumerate(pos.legs)) - fees
        residual = float(net - identity)
        self.stats["identity_max_abs_residual"] = max(
            float(self.stats.get("identity_max_abs_residual", 0.0)), abs(residual))
        if abs(residual) > 1e-6:
            raise RuntimeError(
                f"per-trip accounting identity violated on {key}: net={net!r} "
                f"vs leg-by-leg={identity!r} (residual {residual!r})")
        t = at_time if at_time is not None else self.m.axis[k]
        hold = (t - pos.entry_time).total_seconds() / 60.0
        if pd.Timestamp(t).date() != pd.Timestamp(pos.entry_time).date():
            raise RuntimeError(
                f"{key} was held across the session boundary: "
                f"{pos.entry_time} -> {t}")
        if not hold < MAX_HOLD_MINUTES:
            raise RuntimeError(
                f"{key} hold {hold:.1f} min >= {MAX_HOLD_MINUTES} min "
                f"({pos.entry_time} -> {t})")
        self.stats["exits"][reason] = self.stats["exits"].get(reason, 0) + 1
        if reason == "eod":
            # L11.6 — the accounting for a flatten that fired later than the
            # declared minute.  The delay is measured from the config's **own**
            # ``eod_flat_min``; on the pinned cell that minute is 15:30, so
            # "delayed" is exactly "delayed past 15:30" there.
            delay = int(t.hour) * 60 + int(t.minute) - int(self.config.eod_flat_min)
            if delay > 0:
                self.stats["eod_flatten_delayed_trips"] += 1
                self.stats["eod_flatten_max_delay_min"] = max(
                    int(self.stats["eod_flatten_max_delay_min"]), delay)
        self.trades.append({
            "instrument": key,
            "symbols": ",".join(f"{l.symbol}{'+' if l.side > 0 else '-'}" for l in pos.legs),
            "legs": len(pos.legs),
            "notional": pos.notional,
            "gross_notional": gross_notional,
            "net_outlay": net_outlay,
            "entry_time": pos.entry_time, "exit_time": t,
            "entry_price": pos.legs[0].entry_fill,
            "exit_price": fills[(0, pos.legs[0].symbol)],
            "hold_minutes": hold,
            "exit_reason": reason,
            "pnl_gross": gross, "slip_drag": slip_drag, "fees": fees,
            "cost_drag": drag, "pnl_gross_precost": precost,
            "pnl_after_costs": net,
            "identity_check": identity,
            "identity_residual": residual,
            "leg_detail": ";".join(detail),
            "ret_pct": net / pos.notional if pos.notional else 0.0,
            "net_bps": (net / pos.notional * 1e4) if pos.notional else 0.0,
        })

    # ── the loop ───────────────────────────────────────────────────────
    def _all_open_priceable(self, k: int) -> bool:
        """True when every leg of **every open position** has a bar at bar *k*.

        L11.1's flatten minute is a property of the whole book: it is the first
        bar at or after ``eod_flat_min`` at which this holds, so the session has
        one flatten minute shared by every position — the same shape the fixed
        15:30 minute had.  ``Instrument.valid`` is the AND over the instrument's
        legs (``Market.valid``), so a leg with no bar takes its whole position
        out of the flatten minute, which is exactly the tape blocker's
        condition.
        """
        return all(bool(self.instruments[key].valid[k]) for key in self.positions)

    def _eod_unpriced(self, ts) -> RuntimeError:
        """The fail-closed error for a flatten minute that never arrived (L11.1).

        *ts* is the bar the flatten was last due at — the session's own last
        bar, because that is the point at which "no priceable bar at or after
        ``eod_flat_min``" became knowable.  The text is the tape blocker's own
        message (``ROUND2_TAPE_BLOCKER_PIN.md`` §6): the positions that could
        not be flattened, the bar, and the refusal to carry them.
        """
        keys = ", ".join(sorted(self.positions))
        return RuntimeError(
            f"{keys}: mandatory EOD flatten at {ts} could not be priced (a leg has "
            f"no bar): refusing to carry the position")

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
                if self.positions:
                    # A position that survived the boundary means the EOD
                    # flatten never ran.  Either the flatten minute never
                    # arrived (L11.1: no bar at/after ``eod_flat_min`` existed
                    # at which every leg of every open position could be
                    # priced, and the session has now ended) — which is the
                    # fail-closed case and raises the tape blocker's own
                    # message — or the flatten did not run at all, which is a
                    # bug and keeps the older guard.
                    if self.eod_wait_k is not None or not self.eod_saw_flat_bar:
                        raise self._eod_unpriced(self.m.axis[k - 1])
                    self.stats["skipped"]["survived_session"] += len(self.positions)
                    raise RuntimeError(
                        f"{len(self.positions)} position(s) survived the session "
                        f"boundary into {self.m.axis[k]}: the mandatory EOD "
                        f"flatten did not run")
                self.entries_by_session.clear()
                self.last_entry_minute.clear()
                self.eod_wait_k = None
                self.eod_saw_flat_bar = False
            # L11.1 — the flatten's priceability is a property of the **whole
            # book** at this bar, not of one instrument: the flatten fires on
            # the first bar at/after ``eod_flat_min`` at which every leg of
            # every *open* position has a bar, so the session has one flatten
            # minute, exactly as it did when the minute was pinned at 15:30.
            eod_ready = False
            if minute >= cfg.eod_flat_min:
                self.eod_saw_flat_bar = True
                if self.positions:
                    eod_ready = self._all_open_priceable(k)
                    if not eod_ready and self.eod_wait_k is None:
                        self.eod_wait_k = k
            for key in keys:
                inst = self.instruments[key]
                valid_here = bool(inst.valid[k])
                if not valid_here:
                    # L11.1 — a bar this instrument cannot be priced on is no
                    # longer fatal by itself.  At/after ``eod_flat_min`` the
                    # flatten (step 4) waits for the first bar the whole book
                    # can be priced on and no fabricated price is ever used
                    # (L11.2); the run is fatal only when that bar never
                    # arrives, which is checked at the session boundary and
                    # after the last bar.  Before ``eod_flat_min`` this is the
                    # ordinary "no bar here, nothing to do" path it always was.
                    continue
                prev = k - 1
                same_session = prev >= 0 and not new_session
                # 1) fill an exit signalled on the previous bar
                if key in pending_exit and key in self.positions:
                    reason = pending_exit.pop(key)
                    pos = self.positions[key]
                    fills, refs = self._exit_fills(pos, k)
                    if fills:
                        self._close(key, k, fills, refs, reason)
                    else:
                        pending_exit[key] = reason
                # 2) fill an entry signalled on the previous bar — or discard it
                if key in pending_entry:
                    sig_k, entry_dir = pending_entry.pop(key)
                    if minute >= cfg.eod_flat_min:
                        self.stats["skipped"]["stale_entry"] += 1
                    elif k != sig_k + 1 or not same_session:
                        # "discard, not delay": a pending entry whose fill bar is
                        # not the immediately-next bar for this instrument is
                        # dropped.  Filling it at a *later* bar's open hands the
                        # config a free option, systematically flattering any
                        # gap/reversal family.
                        self.stats["skipped"]["stale_entry"] += 1
                    elif key in self.positions:
                        self.stats["skipped"]["already_held"] += 1
                    else:
                        self._open_or_count(inst, k, sig_k, entry_dir)
                # 3) intrabar management (stops / targets / trailing)
                if key in self.positions:
                    self._manage(inst, k)
                # 4) mandatory end-of-day flatten (L11.1): it fires at the first
                # bar at or after ``eod_flat_min`` at which every leg of every
                # open position has a bar — one flatten minute for the session,
                # shared by the whole book.  A leg with no bar is never priced
                # by a carried-forward close (L11.2); the position simply waits
                # for the next bar that can price it.
                if minute >= cfg.eod_flat_min and key in self.positions and eod_ready:
                    pos = self.positions[key]
                    fills, refs = self._exit_fills(
                        pos, k,
                        price={l.symbol: self.m.c[l.symbol][k] for l in pos.legs})
                    if not fills:
                        raise RuntimeError(
                            f"{key}: mandatory EOD flatten at {self.m.axis[k]} "
                            f"could not be priced: refusing to carry the position")
                    self._close(key, k, fills, refs, "eod", at_time=self.m.axis[k])
                if minute >= cfg.eod_flat_min:
                    continue
                # 5) signal pass on this bar's close -> fill on the next bar
                if key in self.positions:
                    want = int(inst.exit_now[k])
                    pos = self.positions.get(key)
                    if pos is not None and (want == 2 or want == pos.exit_side):
                        pending_exit[key] = "signal"
                    if cfg.time_exit_minutes is not None and key in self.positions:
                        pos = self.positions.get(key)
                        if pos is not None and (
                                self.m.axis[k] - pos.entry_time
                                >= pd.Timedelta(minutes=int(cfg.time_exit_minutes))):
                            closes = {l.symbol: float(self.m.c[l.symbol][k])
                                      for l in pos.legs}
                            if all(not math.isnan(v) for v in closes.values()):
                                fills, refs = self._exit_fills(pos, k, price=closes)
                                self._close(key, k, fills, refs, "time")
                    continue
                if key in pending_entry:
                    continue
                direction = int(inst.entry_dir[k])
                if direction == 0:
                    continue
                self.stats["signals"] += 1
                if direction < 0 and not cfg.allow_short:
                    self.stats["skipped"]["allow_short"] += 1
                    continue
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
                pending_entry[key] = (k, direction)
            self.equity.append(self._equity())
        if self.positions:
            if self.eod_wait_k is not None or not self.eod_saw_flat_bar:
                # L11.1: the session ended without a bar at/after
                # ``eod_flat_min`` at which every leg of every open position
                # could be priced.  Fail closed, with the tape blocker's message.
                raise self._eod_unpriced(self.m.axis[-1] if n else "none")
            raise RuntimeError(
                f"run ended with {len(self.positions)} open position(s) "
                f"(last bar {self.m.axis[-1] if n else 'none'}): the EOD flatten "
                f"did not run")
        self.stats["skipped"]["stale_entry"] += len(pending_entry)
        total_skipped = sum(self.stats["skipped"].values())
        if self.stats["entries"] + total_skipped != self.stats["signals"]:
            raise RuntimeError(
                f"signal accounting broken: entries={self.stats['entries']} + "
                f"skipped={total_skipped} != signals={self.stats['signals']} "
                f"({self.stats['skipped']})")
        result = SearchResult(trades=self._trades_df(), equity_curve=self._equity_series(),
                              stats=self._stats(), config=cfg)
        return result

    def _open_or_count(self, inst: Instrument, k: int, sig_k: int,
                       direction: int) -> None:
        """Open unless a gate refuses; counts mirror the signal-path skips."""
        cfg = self.config
        if len(self.positions) >= cfg.max_positions:
            self.stats["skipped"]["max_positions"] += 1
            return
        self._try_open(inst, k, sig_k, direction)

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
            fills, refs = self._exit_fills(pos, k, price=price)
            self._close(inst.key, k, fills, refs,
                        "trail" if pos.trail_pct is not None else "stop")
        elif hit_target:
            price = max(o, target) if pos.side > 0 else min(o, target)
            fills, refs = self._exit_fills(pos, k, price=price)
            self._close(inst.key, k, fills, refs, "target")

    # ── result assembly ────────────────────────────────────────────────
    def _trades_df(self) -> pd.DataFrame:
        cols = ["instrument", "symbols", "legs", "notional", "gross_notional",
                "net_outlay", "entry_time",
                "exit_time", "entry_price", "exit_price", "hold_minutes",
                "exit_reason", "pnl_gross", "slip_drag", "fees",
                "cost_drag", "pnl_gross_precost", "pnl_after_costs",
                "identity_check",
                "identity_residual", "leg_detail", "ret_pct", "net_bps"]
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
        s["skipped_total"] = int(sum(s["skipped"].values()))
        #: ``max_drawdown`` marks stale closes: the equity series carries the
        #: last known close of a symbol that has no bar at a timestamp, so a
        #: drawdown that happens entirely while a symbol is not printing is
        #: understated.  Quote it with this caveat (P2/E6).
        s["max_drawdown_note"] = ("equity marks stale closes (last known close) "
                                  "for a symbol with no bar at that timestamp")
        #: The declared leverage limit, so every artefact records the bound the
        #: basket check actually enforced (R3-7 E6).
        s["max_gross_leverage"] = float(MAX_GROSS_LEVERAGE)
        s["gross_limit_usd"] = float(MAX_GROSS_LEVERAGE) * float(cfg.initial_equity)
        if len(trades):
            s["gross_notional_per_trip"] = float(trades["gross_notional"].mean())
            s["net_outlay_per_trip"] = float(trades["net_outlay"].mean())
        else:
            s["gross_notional_per_trip"] = 0.0
            s["net_outlay_per_trip"] = 0.0
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
            #: The zero-cost column = P&L at the **reference prices** of the same
            #: fills = ``pnl_gross + slip_drag`` (A2).  It used to be
            #: ``pnl_gross + cost_drag``, which added the commissions a second
            #: time: invisible at baseline (fees 0), optimistic at every
            #: fee-charging cost level.  ``slip_drag`` is slippage only.
            s["pnl_zero_cost_same_fills"] = float(
                (gross + trades["slip_drag"]).sum())
            s["slip_drag_total"] = float(trades["slip_drag"].sum())
            s["slip_drag_per_trip"] = float(trades["slip_drag"].mean())
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
                      "pnl_zero_cost_same_fills", "cost_drag_total", "fees_paid",
                      "slip_drag_total", "slip_drag_per_trip"):
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


def replay_legs(trades: pd.DataFrame) -> list[list[tuple[int, float, float, float]]]:
    """``(side, qty, entry_fill, exit_fill)`` per leg per trip, from ``leg_detail``.

    The one place a fill is re-readable from the record, used by the
    reconciliation below.  A zero-cost replay's fills **are** its raw prices
    (``fill_price(raw) == raw`` when every cost is 0), which is what makes an
    independent cross-check of the baseline's own arithmetic possible.  The
    numbers in ``leg_detail`` are printed to 12 decimals so this cross-check can
    hold a 1e-6 tolerance over hundreds of legs; at 6 decimals the reading error
    alone was of the same order as the tolerance.
    """
    out: list[list[tuple[int, float, float, float]]] = []
    for detail in trades["leg_detail"]:
        legs: list[tuple[int, float, float, float]] = []
        for chunk in str(detail).split(";"):
            _i, _sym, side, qty, ef, xf = chunk.split("|")
            legs.append((int(side), float(qty), float(ef), float(xf)))
        out.append(legs)
    return out


def reconcile_zero_cost(baseline: SearchResult, zero: SearchResult,
                        tol: float = 1e-6) -> dict:
    """Reconcile the baseline's own zero-cost reconstruction against a real replay.

    The same config is replayed with every cost switched off.  Its fills **are**
    the raw prices, so the baseline's reconstruction can be checked against them
    without trusting the baseline's own arithmetic:

        Σ(pnl_gross + slip_drag) == Σ side·(raw_exit − raw_entry)·qty_baseline

    The left side is what the record publishes (and what
    ``pnl_zero_cost_same_fills`` sums); the right side is the same quantity
    rebuilt from **another replay's** raw prices at the baseline's **own**
    quantities.  The two must agree exactly, because ``pnl_gross`` is computed
    from *slipped* fills — it carries ``−Σ qty·slip`` — and ``slip_drag`` adds
    exactly those slips back, so the slips cancel term by term.  They only
    cancel if every slip was charged on the price its fill was actually made at:
    that is the A1 property, and charging the exit slip on the exit bar's *open*
    is what this guard caught on real data.

    **The totals of the two runs are deliberately not compared directly.**  The
    sizing modes are fill-dependent (``qty = notional / fill``), so the zero-cost
    replay trades a slightly *different* quantity from the baseline and its net
    legitimately differs by the whole gross edge times that difference.  That
    difference is reported as ``sizing_term`` and explained here rather than
    hidden: the guard's job is the arithmetic of the *recorded* column, and the
    screen reads a real zero-cost run's own numbers in any case.

    ``pnl_gross + cost_drag − fees`` is the same left-hand quantity
    (``cost_drag`` is ``slip_drag + fees``), spelled that way on purpose: adding
    the whole of ``cost_drag`` back — the old ``pnl_zero_cost_same_fills`` —
    added the commissions a second time (A2).

    Known edge (pre-existing, not introduced here): if a config declares a stop
    or target, the trigger level is anchored on each run's own entry *fill*, so
    the two replays can trade different trips.  ``same_trips`` reports that and
    the runner refuses rather than reconciling incomparable runs.
    """
    b = baseline.trades
    z = zero.trades
    out: dict = {
        "trips_baseline": int(len(b)), "trips_zero_cost": int(len(z)),
        "same_trips": bool(len(b) == len(z)),
        "same_fill_timestamps": False,
        "reconstructed": 0.0,
        "raw_pnl_baseline_qty": 0.0,
        "zero_cost_net": 0.0,
        "sizing_term": 0.0,
        "sizing_term_note": ("zero_cost_net − raw_pnl_baseline_qty: the sizing "
                             "mode sizes off the fill price, so the zero-cost "
                             "replay trades a different quantity"),
        "reconstruction_form": ("pnl_gross + slip_drag == "
                                "Σ side·(raw_exit − raw_entry)·qty_baseline, "
                                "raw prices read from the zero-cost replay"),
        "difference": 0.0, "within_tolerance": False, "mismatches": [],
    }
    if len(b) != len(z):
        out["mismatches"].append(
            f"trip count differs: baseline {len(b)} vs zero-cost {len(z)}")
        return out
    ts_ok = True
    for i in range(len(b)):
        for col in ("entry_time", "exit_time", "instrument"):
            if str(b.iloc[i][col]) != str(z.iloc[i][col]):
                ts_ok = False
                out["mismatches"].append(
                    f"trip {i}: {col} differs ({b.iloc[i][col]} vs {z.iloc[i][col]})")
    out["same_fill_timestamps"] = bool(ts_ok)
    recon = float((b["pnl_gross"] + b["slip_drag"]).sum())
    out["reconstructed"] = recon
    out["zero_cost_net"] = float(z["pnl_after_costs"].sum())
    legs_b, legs_z = replay_legs(b), replay_legs(z)
    raw_at_baseline_qty = 0.0
    same_qty = True
    for i, (lb, lz) in enumerate(zip(legs_b, legs_z)):
        if len(lb) != len(lz):
            out["mismatches"].append(
                f"trip {i}: {len(lb)} leg(s) baseline vs {len(lz)} zero-cost")
            same_qty = False
            continue
        for (side_b, qty_b, _ef_b, _xf_b), (side_z, qty_z, ef_z, xf_z) in zip(lb, lz):
            if side_b != side_z:
                out["mismatches"].append(
                    f"trip {i}: leg sides differ ({side_b} vs {side_z})")
                same_qty = False
                continue
            raw_at_baseline_qty += side_z * (xf_z - ef_z) * qty_b
    out["raw_pnl_baseline_qty"] = float(raw_at_baseline_qty)
    if not same_qty:
        out["within_tolerance"] = False
        return out
    out["sizing_term"] = float(out["zero_cost_net"] - raw_at_baseline_qty)
    out["difference"] = float(recon - raw_at_baseline_qty)
    out["within_tolerance"] = abs(out["difference"]) <= tol
    if not out["within_tolerance"]:
        out["mismatches"].append(
            f"the baseline's own reconstruction {recon!r} != the raw-price P&L at "
            f"the baseline's own quantities {raw_at_baseline_qty!r} rebuilt from "
            f"the zero-cost replay's raw prices (difference "
            f"{out['difference']!r})")
    return out
