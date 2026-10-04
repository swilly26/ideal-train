"""The four stage-1 strategy families, as declared grids.

Each family is **materially different** in what it believes, not a rescaling of
the old rules (brief §3):

* **A — selective intraday momentum/trend.**  Trade only with the session's own
  trend (price vs VWAP, fast/slow EMA slope), optionally gated on the market
  factor's trend, with a trailing or time-based exit.  Selection is the point:
  the first 30 minutes of the session were negative in every round-1 variant, so
  no family may enter there.
* **B — mean reversion with a regime filter.**  Only in a declared low-trend,
  high-noise regime, reverting a per-session z-score back to the mean.
* **C — volatility / breakout.**  Opening-range breakout with volatility-scaled
  (ATR) stops and targets, no entries after a declared cutoff.
* **D — market-neutral / relative strength.**  Pairs among the four leveraged
  ETFs and beta-hedged legs against SPY, flat by the close.

Every config declares, explicitly: an entry-time gate, a maximum entries per
session and/or a minimum gap between entries, an exit rule, and a per-trade
sizing mode.  The grids below were written before any result was seen.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.strategy_search.engine import (
    EOD_FLAT_NEIGHBOURS,
    EOD_PINNED_MIN,
    Instrument,
    Leg,
    Market,
    SearchConfig,
)
from src.backtesting.strategy_search.features import DECLARED_FEATURES

RTH_OPEN_MIN = 9 * 60 + 30
ETF4 = ("SOXL", "TQQQ", "FNGU", "SPXL")

#: Every ``SearchConfig`` field a spec's ``params`` may set (besides the
#: identity fields ``family``/``name`` and ``extras``).
PARAM_FIELDS = ("sizing", "initial_equity", "position_size_pct", "bp_usage_pct",
                "notional_usd", "max_positions", "entry_start_min",
                "entry_end_min", "max_entries_per_session",
                "min_minutes_between_entries", "stop_pct", "target_pct",
                "trail_pct", "time_exit_minutes", "allow_short", "eod_flat_min")
#: Behaviour-changing params with **no default**: a spec must declare every one
#: of them.  The E2 failure was a knob the engine reads (``z_window``) being
#: absent from the recorded artefact because the declaration could omit it and
#: the builder silently merged a whitelist.
REQUIRED_PARAMS = ("sizing", "initial_equity", "position_size_pct", "bp_usage_pct",
                   "notional_usd", "max_positions", "entry_start_min",
                   "entry_end_min", "max_entries_per_session",
                   "min_minutes_between_entries", "allow_short", "eod_flat_min")
#: The only declared ``extras`` keys (ATR-scaled exits).
EXTRA_KEYS = ("stop_atr_mult", "target_atr_mult", "trail_atr_mult")

#: Signal keys a family may read, and those it must be given.
SIGNAL_KEYS: dict[str, tuple[str, ...]] = {
    "A": ("entry_mode", "regime", "exit_mode"),
    "B": ("z_entry", "z_exit", "regime", "vol_floor", "max_abs_spy_ret"),
    "C": ("or_minutes", "allow_short"),
    "D": ("kind", "leg_long", "leg_short", "z_entry", "z_exit", "beta",
          "z_window", "at_minute", "threshold"),
    # round 2 — one knob per declared axis and nothing else (R3-1)
    "E1": ("gap_threshold",),
    "E2": ("gap_threshold",),
    "G": ("breach_atr",),
    "H": ("dev_threshold", "rvol_min"),
}
SIGNAL_REQUIRED: dict[str, tuple[str, ...]] = {
    "A": ("entry_mode", "regime", "exit_mode"),
    "B": ("z_entry", "regime"),
    "C": ("or_minutes", "allow_short"),
    "D": ("kind",),
    "E1": ("gap_threshold",),
    "E2": ("gap_threshold",),
    "G": ("breach_atr",),
    "H": ("dev_threshold", "rvol_min"),
}
#: The declared value sets of the discrete signal knobs.  For the round-2
#: families these are the **declared axis levels**: a cell off the grid raises
#: instead of quietly becoming a 37th hypothesis.
SIGNAL_CHOICES: dict[str, dict[str, tuple]] = {
    "A": {"entry_mode": ("vwap_ema", "prev_break"),
          "regime": ("spy_trend", "none"),
          "exit_mode": ("vwap", "none")},
    "B": {"regime": ("lowtrend", "none")},
    "C": {"or_minutes": (30, 60)},
    "D": {"kind": ("ratio_z", "rs30", "hedge_rs")},
    "E1": {"gap_threshold": (0.5, 1.0, 1.5)},
    "E2": {"gap_threshold": (0.5, 1.0, 1.5)},
    "G": {"breach_atr": (0.25, 0.5, 1.0)},
    "H": {"dev_threshold": (0.5, 1.0, 1.5),
          "rvol_min": (0.0002, 0.0005, 0.0010)},
}


class SpecError(ValueError):
    """A declared config that the engine cannot honestly execute."""


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(obj: object) -> str:
    """Stable short hash of a canonical JSON document."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:16]


def config_hash(family: str, cfg: SearchConfig, signal: Mapping,
                symbols: Sequence[str] = ()) -> str:
    """The config's identity, over the **real channel** (R3-7 E2).

    Hash input = {family, the ``SearchConfig`` fields with ``extras`` as a
    **nested** key, the **resolved signal dict the family actually used**, the
    universe}.  Three things are deliberate:

    * ``extras`` is nested, **never merged over the real fields**: merged, an
      extras entry shadows a real field and two genuinely different configs
      hash — and are recorded — alike;
    * the **resolved** signal dict is in, not the declaration: hashing
      ``params()`` alone does not cover the channel that produced the stage-2
      defect (a ``z_window`` the engine read and the artefact never recorded);
    * ``name`` is **out**: the name is generated from this hash, so including it
      would be circular.
    """
    fields_ = dataclasses.asdict(cfg)
    fields_.pop("name", None)
    fields_["extras"] = dict(cfg.extras)
    payload = {"family": family, "config": fields_, "signal": dict(signal),
               "symbols": list(symbols)}
    return canonical_hash(payload)


def resolved_hash(family: str, resolved: Mapping) -> str:
    """``config_hash`` of a *resolved spec* — the value ``cfg.name`` carries."""
    cfg = SearchConfig(family=family, name="",
                       extras=dict(resolved["extras"]),
                       **dict(resolved["params"]))
    return config_hash(family, cfg, resolved["signal"],
                       resolved.get("symbols", ()))


def resolved_name(family: str, resolved: Mapping[str, object]) -> str:
    """The config's name, generated from its *resolved* parameters."""
    return f"{family.lower()}-{resolved_hash(family, resolved)}"


def resolve_params(family: str, spec: Mapping) -> dict:
    """Validate a spec's ``params`` and return them with nothing left implicit."""
    given = dict(spec.get("params", {}))
    unknown = sorted(set(given) - set(PARAM_FIELDS))
    if unknown:
        raise SpecError(
            f"family {family} spec {spec.get('name')!r}: unknown param key(s) "
            f"{unknown}; declared keys are {list(PARAM_FIELDS)}")
    missing = sorted(k for k in REQUIRED_PARAMS if k not in given)
    if missing:
        raise SpecError(
            f"family {family} spec {spec.get('name')!r}: required param(s) "
            f"{missing} are missing — every behaviour-changing field must be "
            f"declared, with no default")
    sizings = ("fixed_notional", "equity_fraction")
    if given["sizing"] not in sizings:
        raise SpecError(f"sizing {given['sizing']!r} not in {sizings}")
    if int(given["eod_flat_min"]) not in EOD_FLAT_NEIGHBOURS:
        raise SpecError(
            f"eod_flat_min={given['eod_flat_min']} is not a declared value "
            f"{list(EOD_FLAT_NEIGHBOURS)} (pinned at {EOD_PINNED_MIN})")
    return given


def resolve_extras(family: str, spec: Mapping) -> dict:
    given = dict(spec.get("extras", {}))
    unknown = sorted(set(given) - set(EXTRA_KEYS))
    if unknown:
        raise SpecError(
            f"family {family} spec {spec.get('name')!r}: unknown extras key(s) "
            f"{unknown}; declared keys are {list(EXTRA_KEYS)}")
    return given


def resolve_signal(family: str, spec: Mapping) -> dict:
    """Validate a spec's ``signal`` and return it with every default resolved.

    Every knob the family builder reads is present in the returned dict, so the
    artefact can record what actually ran instead of the declaration.  An
    unknown key, a missing required key or an undeclared value is an error, not
    a silent default.
    """
    given = dict(spec.get("signal", {}))
    if family not in SIGNAL_KEYS:
        raise SpecError(f"unknown family {family!r}")
    unknown = sorted(set(given) - set(SIGNAL_KEYS[family]))
    if unknown:
        raise SpecError(
            f"family {family} spec {spec.get('name')!r}: unknown signal key(s) "
            f"{unknown}; declared keys are {list(SIGNAL_KEYS[family])}")
    missing = sorted(k for k in SIGNAL_REQUIRED[family] if k not in given)
    if missing:
        raise SpecError(
            f"family {family} spec {spec.get('name')!r}: required signal key(s) "
            f"{missing} are missing (no default)")
    out = dict(given)
    for key, choices in SIGNAL_CHOICES.get(family, {}).items():
        if key in out and out[key] not in choices:
            raise SpecError(
                f"family {family} spec {spec.get('name')!r}: {key}="
                f"{out[key]!r} is not one of {list(choices)}")
    if family == "B":
        out.setdefault("z_exit", 0.0)
        if out["regime"] == "lowtrend":
            for k in ("vol_floor", "max_abs_spy_ret"):
                if k not in out:
                    raise SpecError(
                        f"family B spec {spec.get('name')!r}: regime "
                        f"'lowtrend' needs {k!r} declared (no default)")
    elif family == "D":
        kind = out["kind"]
        out.setdefault("beta", 1.0)
        if kind in ("ratio_z", "hedge_rs"):
            for k in ("leg_long", "leg_short", "z_entry"):
                if k not in out:
                    raise SpecError(
                        f"family D spec {spec.get('name')!r}: kind {kind!r} "
                        f"needs {k!r} declared (no default)")
            out.setdefault("z_exit", 0.0)
            out.setdefault("z_window", 30)
        else:                                             # rs30
            for k in ("leg_long", "leg_short", "at_minute", "threshold"):
                if k not in out:
                    raise SpecError(
                        f"family D spec {spec.get('name')!r}: kind {kind!r} "
                        f"needs {k!r} declared (no default)")
    return out


def resolve_spec(family: str, spec: Mapping) -> dict:
    """The whole resolved spec — what the run actually executes.

    It carries **no declaration name**: the hash and the generated config name
    are functions of behaviour only, so two cells that declare the same
    behaviour resolve to the same config, and two cells that declare different
    behaviour can never share one.
    """
    return {
        "params": resolve_params(family, spec),
        "signal": resolve_signal(family, spec),
        "extras": resolve_extras(family, spec),
        "symbols": list(spec.get("symbols", ())),
    }


def _check_universe(family: str, instruments: Mapping[str, Instrument],
                    market: Market) -> None:
    """``2k <= |U| - 1`` for rank-rotation instruments, and ``kind`` coherence.

    The rule the lead declared for round 2 (E1).  A universe too small for its
    instruments is a universe in which the legs cannot be spread honestly over
    distinct symbols, so the builder refuses rather than trades a degenerate
    book.  It is scoped to **k >= 2**: with family F struck it would otherwise
    fire on a single-symbol family (R3-7).

    ``kind == "single"`` **iff** exactly one leg — a single-leg instrument
    marked ``pair`` (or the reverse) would silently change which exit paths run
    (``_manage`` returns immediately when ``side == 0``), so the two must agree.
    """
    n_u = len(market.symbols)
    for key, inst in instruments.items():
        if (inst.kind == "single") != (len(inst.legs) == 1):
            raise SpecError(
                f"family {family}: instrument {key!r} declares kind="
                f"{inst.kind!r} with {len(inst.legs)} leg(s); 'single' means "
                f"exactly one leg and anything else means more than one")
        k = len(inst.legs)
        if k >= 2 and 2 * k > n_u - 1:
            raise SpecError(
                f"family {family}: instrument {key!r} has {k} legs but the "
                f"universe U has only {n_u} symbols; the declared rule "
                f"2k <= |U| - 1 requires at least {2 * k + 1}")


# ── helpers ────────────────────────────────────────────────────────────


def _col(feats: Mapping[str, pd.DataFrame], sym: str, name: str,
         axis: pd.DatetimeIndex) -> np.ndarray:
    """One declared feature column, or raise.

    Returning all-NaN for an unknown name turned a typo into "killed: too few
    trips" — a false negative that looks like a result (E4).  A family may only
    read the declared feature set, and a symbol whose features are missing is
    an error, not an empty column.
    """
    if name not in DECLARED_FEATURES:
        raise SpecError(
            f"feature {name!r} is not in the declared feature set "
            f"({len(DECLARED_FEATURES)} names, see features.FEATURE_NAMES)")
    if sym not in feats:
        raise SpecError(f"no features for symbol {sym!r}: the family asked for "
                        f"{name!r}")
    f = feats[sym]
    if name not in f.columns:
        raise SpecError(f"symbol {sym!r} has no feature column {name!r}")
    return f[name].to_numpy(dtype=float)


def _shift1(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    out[1:] = a[:-1]
    return out


def _market(feats: Mapping[str, pd.DataFrame], axis: pd.DatetimeIndex) -> pd.DataFrame:
    """SPY features on *axis*, forward-filled and shifted one bar (see features.py)."""
    f = feats.get("SPY")
    if f is None:
        return pd.DataFrame(index=axis)
    return f.reindex(axis, method="ffill").shift(1)


def _dir_array(longs: np.ndarray, shorts: np.ndarray) -> np.ndarray:
    d = np.zeros(len(longs), dtype=np.int8)
    d[longs] = 1
    d[shorts] = -1
    return d


def _exit_array(exit_long: np.ndarray, exit_short: np.ndarray) -> np.ndarray:
    e = np.zeros(len(exit_long), dtype=np.int8)
    e[exit_long] = 1
    e[exit_short] = -1
    both = exit_long & exit_short
    e[both] = 2
    return e


# ── family A — selective intraday momentum / trend ─────────────────────


def _family_a(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
              ) -> dict[str, Instrument]:
    p = spec["signal"]
    axis = market.axis
    spy = _market(feats, axis)
    out: dict[str, Instrument] = {}
    for sym in spec["symbols"]:
        c = _close(feats, sym, axis)
        vwap = _col(feats, sym, "vwap", axis)
        ef = _col(feats, sym, "ema_fast", axis)
        es = _col(feats, sym, "ema_slow", axis)
        slope = _col(feats, sym, "slope", axis)
        prev_hi = _col(feats, sym, "prev_hi", axis)
        prev_lo = _col(feats, sym, "prev_lo", axis)
        base_long = (c > vwap) & (ef > es) & (slope > 0)
        base_short = (c < vwap) & (ef < es) & (slope < 0)
        if p["entry_mode"] == "prev_break":
            base_long = c > prev_hi
            base_short = c < prev_lo
        if p["regime"] == "spy_trend":
            up = _col_feat(spy, "sess_ret") > 0
            up &= _col_feat(spy, "ema_fast") > _col_feat(spy, "ema_slow")
            down = ~up
            base_long = base_long & up
            base_short = base_short & down
        shorts = base_short if spec["params"]["allow_short"] else np.zeros(len(c), bool)
        entry = _dir_array(base_long, shorts)
        if p["exit_mode"] == "vwap":
            exit_long = c < vwap
            exit_short = c > vwap
        else:
            exit_long = np.zeros(len(c), bool)
            exit_short = np.zeros(len(c), bool)
        out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),), entry_dir=entry,
                              exit_now=_exit_array(exit_long, exit_short),
                              valid=market.valid((Leg(sym, 1, 1.0),)))
    return out


def _close(feats: Mapping[str, pd.DataFrame], sym: str, axis: pd.DatetimeIndex) -> np.ndarray:
    f = feats.get(sym)
    if f is None or "close" not in f.columns:
        raise SpecError(f"no close series for symbol {sym!r}")
    return f["close"].to_numpy(dtype=float)


def _col_feat(df: pd.DataFrame, name: str) -> np.ndarray:
    """One declared feature column of an *aligned* frame, or raise (E4)."""
    if name not in DECLARED_FEATURES:
        raise SpecError(f"feature {name!r} is not in the declared feature set")
    if name not in df.columns:
        raise SpecError(f"aligned frame has no feature column {name!r}")
    return df[name].to_numpy(dtype=float)


# ── family C — volatility / breakout ───────────────────────────────────


def _family_c(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
              ) -> dict[str, Instrument]:
    p = spec["signal"]
    axis = market.axis
    or_min = int(p["or_minutes"])
    hi_name, lo_name = f"or{or_min}_hi", f"or{or_min}_lo"
    out: dict[str, Instrument] = {}
    for sym in spec["symbols"]:
        c = _close(feats, sym, axis)
        hi = _col(feats, sym, hi_name, axis)
        lo = _col(feats, sym, lo_name, axis)
        cp = _shift1(c)
        hp = _shift1(hi)
        lp = _shift1(lo)
        ready = _col(feats, sym, "sess_min", axis) >= or_min
        cross_up = (c > hi) & (cp <= hp) & ready
        cross_dn = (c < lo) & (cp >= lp) & ready
        if not p["allow_short"]:
            cross_dn = np.zeros(len(c), bool)
        entry = _dir_array(cross_up, cross_dn)
        out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),), entry_dir=entry,
                              exit_now=np.zeros(len(c), dtype=np.int8),
                              valid=market.valid((Leg(sym, 1, 1.0),)),
                              atr_pct=_col(feats, sym, "atr_pct", axis))
    return out


# ── family B — mean reversion with a regime filter ─────────────────────


def _family_b(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
              ) -> dict[str, Instrument]:
    p = spec["signal"]
    axis = market.axis
    spy = _market(feats, axis)
    z_entry = float(p["z_entry"])
    z_exit = float(p["z_exit"])
    out: dict[str, Instrument] = {}
    for sym in spec["symbols"]:
        c = _close(feats, sym, axis)
        z = _col(feats, sym, "z20", axis)
        longs = z <= -z_entry
        shorts = z >= z_entry
        if p["regime"] == "lowtrend":
            vol = _col_feat(spy, "rvol30")
            quiet = np.abs(_col_feat(spy, "sess_ret")) <= float(p["max_abs_spy_ret"])
            noisy = vol >= float(p["vol_floor"])
            ok = quiet & noisy
            longs = longs & ok
            shorts = shorts & ok
        if not spec["params"]["allow_short"]:
            shorts = np.zeros(len(c), bool)
        exit_long = z >= z_exit
        exit_short = z <= -z_exit
        out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),),
                              entry_dir=_dir_array(longs, shorts),
                              exit_now=_exit_array(exit_long, exit_short),
                              valid=market.valid((Leg(sym, 1, 1.0),)))
    return out


# ── family D — market-neutral / relative strength ──────────────────────


def _family_d(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
              ) -> dict[str, Instrument]:
    p = spec["signal"]
    axis = market.axis
    kind = p["kind"]
    out: dict[str, Instrument] = {}
    if kind in ("ratio_z", "rs30", "hedge_rs"):
        a_sym, b_sym = p["leg_long"], p["leg_short"]
        beta = float(p["beta"])
        ca = _close(feats, a_sym, axis)
        cb = _close(feats, b_sym, axis)
        valid = market.valid((Leg(a_sym, 1, 1.0), Leg(b_sym, -1, 1.0)))
        legs = ((Leg(a_sym, 1, 1.0), Leg(b_sym, -1, beta)) if beta != 1.0
                else (Leg(a_sym, 1, 1.0), Leg(b_sym, -1, 1.0)))
        if kind in ("ratio_z", "hedge_rs"):
            spread = np.log(ca) - beta * np.log(cb)
            s = pd.Series(spread, index=axis)
            day = pd.Series(axis.normalize(), index=axis)
            zw = int(p["z_window"])     # bars of the rolling z, per session
            mean = s.groupby(day).transform(lambda x: x.rolling(zw).mean())
            sd = s.groupby(day).transform(lambda x: x.rolling(zw).std())
            z = ((s - mean) / sd.replace(0.0, np.nan)).to_numpy(dtype=float)
            z_entry = float(p["z_entry"])
            longs = z <= -z_entry          # leg A cheap vs leg B
            shorts = z >= z_entry
            exit_long = z >= float(p["z_exit"])
            exit_short = z <= -float(p["z_exit"])
            entry = _dir_array(longs, shorts)
            exit_now = _exit_array(exit_long, exit_short)
        else:                              # rs30: relative strength at one minute
            at = int(p["at_minute"])
            minute = np.array([t.hour * 60 + t.minute for t in axis])
            ra = _col(feats, a_sym, "ret30", axis)
            rb = _col(feats, b_sym, "ret30", axis)
            at_bar = minute == at
            strength = ra - beta * rb
            longs = at_bar & (strength >= float(p["threshold"]))
            shorts = at_bar & (strength <= -float(p["threshold"]))
            entry = _dir_array(longs, shorts)
            exit_now = np.full(len(axis), 2, dtype=np.int8)
        key = f"{a_sym}/{b_sym}"
        out[key] = Instrument(key=key, legs=legs, entry_dir=entry,
                              exit_now=exit_now, valid=valid, kind="pair")
    else:
        raise ValueError(f"unknown family D kind: {kind}")
    return out


# ── round-2 families: E1/E2 gap behaviour, G opening-range reversal, H VWAP ──
#
# Every round-2 cell declares exactly two 3-level axes and nothing else
# (R3-1: a full 3x3 factorial, so the neighbour rule is mechanical — each cell
# has exactly two neighbours per axis, whichever cell survives).  Everything
# else is a family-wide literal pinned here and asserted in ``_check_round2_cell``:
# the universe, the caps, the entry window, ``eod_flat_min = 15:30``,
# ``allow_short = True``, and one signal per session (the first bar of the
# session inside the window that satisfies the declared condition).
#
# Signal arithmetic, all of it causal (prior-session or expanding only):
#
#   gap_atr = (sess_open / prev_close - 1) / atr_prev_pct
#   dev     = (close - vwap) / (atr_prev_pct * close)
#   breach  = high >= or30_hi + b * atr_prev_abs   (down: low <= or30_lo - b*atr)
#
# ``prev_close`` is the prior session's 15:59 cached print while the strategy is
# flat from 15:30, and ``atr_prev_*`` is the prior **completed** session's ATR(14)
# at its own last bar — a value that is final at the prior close and never
# revised.  The session's own ``atr_pct`` is NaN for its first 14 bars and cannot
# normalise an opening signal; that is exactly the round-1 failure class.

#: The pinned round-2 universe (R3-6.6).  FNGU is **excluded** with the reason
#: recorded: it has no bars before 2025-02, so it starts mid-W2 and would change
#: both the trip count and the 12-month stability guard.
R2_UNIVERSE = ("SOXL", "TQQQ", "SPXL", "SPY")
#: The pinned caps (R3-6.4, **re-pinned by R3-11.1**): the 150-trip power floor
#: must be reachable by finding an edge, never by loosening a cap and trading
#: more.  ``max_positions`` is the size of the universe and ``notional_usd`` is
#: equity/4, so the cap can never bind, no symbol is crowded out, and total gross
#: exposure is unchanged at 100 % of the 100k equity (4 x 25_000).  The re-pin
#: exists because ``R2_UNIVERSE`` is also the instrument insertion order and
#: ``gap`` is session-constant: at ``max_positions = 2`` family E's SOXL and TQQQ
#: always took both slots and SPXL/SPY could never trade, which made every E
#: number a two-symbol, two-correlated-3x-leveraged-ETF number.
R2_CAPS = {"initial_equity": 100_000.0, "notional_usd": 25_000.0,
           "max_positions": 4, "max_entries_per_session": 1,
           "min_minutes_between_entries": 0}
#: The pinned entry windows, one literal per family (R3-7 E3).
#: ``_base_params`` sets 10:00, which would put **every** gap signal from 09:31
#: to 10:00 into ``skipped["gate"]`` and make family E trade zero trips — then
#: read as "killed: too few trips".  Asserted per cell, not trusted.
R2_ENTRY_WINDOWS = {
    "E1": (9 * 60 + 31, 10 * 60 + 30),     # E-continuation: 09:31 - 10:30
    "E2": (9 * 60 + 31, 10 * 60 + 30),     # E-fade:        09:31 - 10:30
    "G": (10 * 60, 14 * 60 + 30),          # G:             10:00 - 14:30
    "H": (10 * 60, 14 * 60 + 30),          # H:             10:00 - 14:30
}
#: G's opening-range length — a **family literal**, and the reason G's window
#: starts at 10:00: ``or30_hi``/``or30_lo`` are only final then.
G_OR_MINUTES = 30
#: H's mean-reversion exit (ATR units) and time stop — family literals.  Only
#: the deviation threshold and the regime gate are axes.
H_DEV_EXIT_ATR = 0.25
H_TIME_STOP_MINUTES = 60


def _window_mask(market: Market, start_min: int, end_min: int) -> np.ndarray:
    return (market.minute >= int(start_min)) & (market.minute <= int(end_min))


def _first_per_session(market: Market, mask: np.ndarray) -> np.ndarray:
    """True on the **first** bar of each session where *mask* holds (round 2).

    Every round-2 cell declares "the first bar ... that satisfies the
    condition".  Leaving the mask on every qualifying bar would let the engine
    re-signal after a skipped entry — a different rule than the one declared,
    and one the census would then count differently from the trips.
    """
    n = market.n()
    out = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(market.day_change)
    if not len(starts):
        return out
    m = np.asarray(mask, dtype=bool)
    for s, e in zip(starts, np.append(starts[1:], n)):
        seg = m[s:e]
        if seg.any():
            out[s + int(np.argmax(seg))] = True
    return out


def _gap_atr(feats: Mapping[str, pd.DataFrame], sym: str,
             axis: pd.DatetimeIndex) -> np.ndarray:
    """``(sess_open / prev_close − 1) / atr_prev_pct`` — the gap, in ATR units.

    The gap is measured close-to-open across the overnight boundary; the
    **trade** is open→close and never captures the gap move itself (P1 says so
    wherever family E's result is quoted).
    """
    so = _col(feats, sym, "sess_open", axis)
    pc = _col(feats, sym, "prev_close", axis)
    atrp = _col(feats, sym, "atr_prev_pct", axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = (so / pc - 1.0) / atrp
    return np.asarray(gap, dtype=float)


def _gap_family(family: str, spec: dict, market: Market,
                feats: Mapping[str, pd.DataFrame], sign_mode: int
                ) -> dict[str, Instrument]:
    """E1 (``sign_mode=+1``, continuation) and E2 (``sign_mode=-1``, fade).

    Entry: the first bar in ``[09:31, 10:30]`` of a session whose
    ``|gap_atr| >= gap_threshold``, in the direction ``sign_mode·sign(gap_atr)``.
    The engine fills it at the next bar's open and drops it, counted
    ``stale_entry``, unless that bar is the immediately-next bar of the same
    session.
    """
    p = spec["signal"]
    axis = market.axis
    g = float(p["gap_threshold"])
    win = _window_mask(market, *R2_ENTRY_WINDOWS[family])
    n = market.n()
    out: dict[str, Instrument] = {}
    for sym in spec["symbols"]:
        gap = _gap_atr(feats, sym, axis)
        eligible = win & np.isfinite(gap) & (np.abs(gap) >= g)
        first = _first_per_session(market, eligible)
        entry = np.zeros(n, dtype=np.int8)
        idx = np.flatnonzero(first)
        if len(idx):
            entry[idx] = (np.sign(gap[idx]).astype(int) * int(sign_mode))
        out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),), entry_dir=entry,
                              exit_now=np.zeros(n, dtype=np.int8),
                              valid=market.valid((Leg(sym, 1, 1.0),)))
    return out


def _family_e1(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
               ) -> dict[str, Instrument]:
    """Family E1 — **E-continuation** (gap continues in the gap's direction)."""
    return _gap_family("E1", spec, market, feats, +1)


def _family_e2(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
               ) -> dict[str, Instrument]:
    """Family E2 — **E-fade** (gap reverses; the sign-opposite hypothesis)."""
    return _gap_family("E2", spec, market, feats, -1)


def _family_g(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
              ) -> dict[str, Instrument]:
    """Family G — opening-range reversal: enter **counter** to the breach.

    A breach up is ``high >= or30_hi + b·atr_prev_abs`` (down is symmetric), and
    the entry is the opposite side of it.  G is the sign-opposite of family C on
    the same bars, so it is **not independent evidence** from C's kill — say so
    wherever G's result is quoted (D1).
    """
    p = spec["signal"]
    axis = market.axis
    b = float(p["breach_atr"])
    win = _window_mask(market, *R2_ENTRY_WINDOWS["G"])
    hi_name, lo_name = f"or{G_OR_MINUTES}_hi", f"or{G_OR_MINUTES}_lo"
    n = market.n()
    out: dict[str, Instrument] = {}
    for sym in spec["symbols"]:
        high = _col(feats, sym, "high", axis)
        low = _col(feats, sym, "low", axis)
        hi = _col(feats, sym, hi_name, axis)
        lo = _col(feats, sym, lo_name, axis)
        atr = _col(feats, sym, "atr_prev_abs", axis)
        ready = _col(feats, sym, "sess_min", axis) >= G_OR_MINUTES
        breach_up = high >= hi + b * atr
        breach_dn = low <= lo - b * atr
        first = _first_per_session(market, (breach_up | breach_dn) & ready & win)
        entry = np.zeros(n, dtype=np.int8)
        idx = np.flatnonzero(first)
        if len(idx):
            # counter to the breach: through the upper band -> short, lower -> long.
            # A bar that breaches both bands (a bar whose range exceeds both) is
            # not a tradeable distinction, so the short side is taken, by rule.
            entry[idx] = np.where(breach_up[idx], -1, 1).astype(np.int8)
        out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),), entry_dir=entry,
                              exit_now=np.zeros(n, dtype=np.int8),
                              valid=market.valid((Leg(sym, 1, 1.0),)))
    return out


def _family_h(spec: dict, market: Market, feats: Mapping[str, pd.DataFrame]
              ) -> dict[str, Instrument]:
    """Family H — session-VWAP reversion behind a realised-volatility gate.

    ``dev = (close − vwap) / (atr_prev_pct · close)``; enter **counter** to the
    deviation when ``|dev| >= d`` **and** the causal ``rvol30 >= theta``.  Exit
    when the deviation's sign has flipped or it is back inside
    ``0.25·atr_prev`` of VWAP, else the 60-minute time stop, else the EOD
    flatten.
    """
    p = spec["signal"]
    axis = market.axis
    d = float(p["dev_threshold"])
    theta = float(p["rvol_min"])
    win = _window_mask(market, *R2_ENTRY_WINDOWS["H"])
    n = market.n()
    out: dict[str, Instrument] = {}
    for sym in spec["symbols"]:
        c = _close(feats, sym, axis)
        vwap = _col(feats, sym, "vwap", axis)
        atrp = _col(feats, sym, "atr_prev_pct", axis)
        rvol = _col(feats, sym, "rvol30", axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = (c - vwap) / (atrp * c)
        regime = rvol >= theta
        longs = (dev <= -d) & regime & win          # below VWAP -> long
        shorts = (dev >= d) & regime & win          # above VWAP -> short
        first = _first_per_session(market, longs | shorts)
        entry = np.zeros(n, dtype=np.int8)
        idx = np.flatnonzero(first)
        if len(idx):
            entry[idx] = np.where(longs[idx], 1, -1).astype(np.int8)
        # A long exits when dev is back inside the band or has flipped:
        # ``dev >= -H_DEV_EXIT_ATR`` covers both (the flip at dev >= 0 and the
        # |dev| <= 0.25 band at dev >= -0.25).  A short is the mirror.
        exit_long = dev >= -H_DEV_EXIT_ATR
        exit_short = dev <= H_DEV_EXIT_ATR
        out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),), entry_dir=entry,
                              exit_now=_exit_array(exit_long, exit_short),
                              valid=market.valid((Leg(sym, 1, 1.0),)))
    return out


BUILDERS = {"A": _family_a, "C": _family_c, "B": _family_b, "D": _family_d,
            "E1": _family_e1, "E2": _family_e2, "G": _family_g, "H": _family_h}


def _check_multileg_exits(family: str, cfg: SearchConfig,
                          instruments: Mapping[str, Instrument]) -> None:
    """A multi-leg config may not declare a stop / target / trail (E5).

    Stops and targets are built only for ``single`` instruments and ``_manage``
    returns immediately when a pair's ``side`` is 0, so a multi-leg config could
    *declare* ``stop_pct``, *record* it in the artefact, and run with no stop at
    all.  Until the engine wires multi-leg management (with its own test), the
    builder refuses the combination outright.
    """
    atr_mult = any(float(cfg.extras.get(k) or 0.0)
                   for k in ("stop_atr_mult", "target_atr_mult", "trail_atr_mult"))
    declares = (cfg.stop_pct is not None or cfg.target_pct is not None
                or cfg.trail_pct is not None or atr_mult)
    if not declares:
        return
    bad = sorted(k for k, inst in instruments.items() if len(inst.legs) > 1)
    if bad:
        raise SpecError(
            f"family {family}: instrument(s) {bad} have more than one leg but "
            f"the config declares a stop/target/trail "
            f"(stop_pct={cfg.stop_pct}, target_pct={cfg.target_pct}, "
            f"trail_pct={cfg.trail_pct}, extras={dict(cfg.extras)}): the engine "
            f"would record a stop it never runs")


def _check_zero_hold_guard(family: str, cfg: SearchConfig, resolved: Mapping) -> None:
    """``entry_end_min <= eod_flat_min - 2``, asserted for every family (R3-6.5).

    A signal at 15:29 fills and can be flattened on its own bar's close: a
    two-toll trip that inflates the trip count without ever holding a position.
    The guard is the mechanical form of "no zero-hold trips" and applies to the
    whole search, not only the round-2 cells.
    """
    if int(cfg.entry_end_min) > int(cfg.eod_flat_min) - 2:
        raise SpecError(
            f"family {family} spec {resolved.get('symbols')!r}: entry_end_min="
            f"{int(cfg.entry_end_min)} > eod_flat_min={int(cfg.eod_flat_min)} - 2: "
            f"a signal in the last two minutes is a zero-hold, two-toll trip")


def _check_round2_cell(family: str, cfg: SearchConfig, instruments: Mapping,
                       resolved: Mapping) -> None:
    """The pinned literals every round-2 cell must declare (R3-1 / R3-6).

    A mis-set flag does not error anywhere in the engine — it silently deletes
    half the family (``allow_short`` drops every short signal), trades the wrong
    window (entry window), or puts the whole family in a gate skip.  So each
    pinned value is asserted here, per cell, rather than trusted.
    """
    start, end = R2_ENTRY_WINDOWS[family]
    if cfg.allow_short is not True:
        raise SpecError(
            f"family {family}: allow_short={cfg.allow_short!r} — every round-2 "
            f"family declares True; the flag silently deletes the short half "
            f"of the grid (E-signals on a gap down, G breaches up, H deviations "
            f"above VWAP)")
    if (int(cfg.entry_start_min), int(cfg.entry_end_min)) != (start, end):
        raise SpecError(
            f"family {family}: entry window "
            f"({int(cfg.entry_start_min)}, {int(cfg.entry_end_min)}) is not the "
            f"pinned literal ({start}, {end})")
    if tuple(resolved.get("symbols", ())) != R2_UNIVERSE:
        raise SpecError(
            f"family {family}: universe {list(resolved.get('symbols', ()))!r} is "
            f"not the pinned {list(R2_UNIVERSE)!r} (FNGU is excluded on purpose: "
            f"no bars before 2025-02)")
    if int(cfg.max_positions) != R2_CAPS["max_positions"]:
        raise SpecError(f"family {family}: max_positions={cfg.max_positions} is "
                        f"not the pinned {R2_CAPS['max_positions']}")
    if int(cfg.max_entries_per_session) != R2_CAPS["max_entries_per_session"]:
        raise SpecError(
            f"family {family}: max_entries_per_session="
            f"{cfg.max_entries_per_session} is not the pinned "
            f"{R2_CAPS['max_entries_per_session']}")
    if int(cfg.min_minutes_between_entries) != R2_CAPS["min_minutes_between_entries"]:
        raise SpecError(f"family {family}: min_minutes_between_entries="
                        f"{cfg.min_minutes_between_entries} is not pinned at "
                        f"{R2_CAPS['min_minutes_between_entries']}")
    if float(cfg.initial_equity) != R2_CAPS["initial_equity"]:
        raise SpecError(f"family {family}: initial_equity={cfg.initial_equity} is "
                        f"not the pinned {R2_CAPS['initial_equity']}")
    if float(cfg.notional_usd) != R2_CAPS["notional_usd"]:
        raise SpecError(f"family {family}: notional_usd={cfg.notional_usd} is not "
                        f"the pinned {R2_CAPS['notional_usd']}")
    if int(cfg.eod_flat_min) != EOD_PINNED_MIN:
        raise SpecError(f"family {family}: eod_flat_min={cfg.eod_flat_min} is not "
                        f"pinned at {EOD_PINNED_MIN} for every round-2 cell")
    _check_multileg_exits(family, cfg, instruments)


def build(family: str, spec: dict, market: Market,
          feats: Mapping[str, pd.DataFrame]
          ) -> tuple[SearchConfig, dict[str, Instrument], dict]:
    """Build the resolved config and its instruments.

    Returns ``(cfg, instruments, resolved)``.  ``resolved`` is what the run
    actually executes — every param, signal knob and extra with its value
    filled in — and ``cfg.name`` is generated from it, so two different
    resolved configs can never share a name and the artefact cannot record a
    declaration that the engine did not run (E2).
    """
    if family not in BUILDERS:
        raise SpecError(f"unknown family {family!r}")
    resolved = resolve_spec(family, spec)
    cfg = SearchConfig(family=family, name=spec["name"],
                       extras=dict(resolved["extras"]),
                       **resolved["params"])
    rspec = dict(resolved)
    rspec["symbols"] = list(spec.get("symbols", ()))
    resolved["symbols"] = list(rspec["symbols"])
    instruments = BUILDERS[family](rspec, market, feats)
    _check_universe(family, instruments, market)
    _check_multileg_exits(family, cfg, instruments)
    _check_zero_hold_guard(family, cfg, resolved)
    if family in R2_ENTRY_WINDOWS:
        _check_round2_cell(family, cfg, instruments, resolved)
    cfg = dataclasses.replace(cfg, name=resolved_name(family, resolved))
    return cfg, instruments, resolved


# ── the pre-declared grids ─────────────────────────────────────────────
# Written before any result was seen.  Coarse steps, 8 configs per family.

def _base_params(**kw) -> dict:
    p = dict(sizing="fixed_notional", notional_usd=50_000.0,
             initial_equity=100_000.0, position_size_pct=0.50, bp_usage_pct=0.95,
             max_positions=2, entry_start_min=10 * 60, entry_end_min=14 * 60,
             max_entries_per_session=1, min_minutes_between_entries=0,
             stop_pct=None, target_pct=None, trail_pct=None,
             time_exit_minutes=None, allow_short=False,
             eod_flat_min=EOD_PINNED_MIN)
    p.update(kw)
    return p


GRID_A = [
    dict(name="a_trend_trail50_reg",
         params=_base_params(trail_pct=0.005, stop_pct=0.010, entry_end_min=13 * 60,
                             time_exit_minutes=None),
         extras=dict(stop_atr_mult=0.0),
         signal=dict(entry_mode="vwap_ema", regime="spy_trend", exit_mode="none")),
    dict(name="a_trend_trail100_reg",
         params=_base_params(trail_pct=0.010, stop_pct=0.010, entry_end_min=13 * 60),
         signal=dict(entry_mode="vwap_ema", regime="spy_trend", exit_mode="none")),
    dict(name="a_trend_eod_reg",
         params=_base_params(stop_pct=0.015, entry_end_min=13 * 60, time_exit_minutes=None),
         signal=dict(entry_mode="vwap_ema", regime="spy_trend", exit_mode="none")),
    dict(name="a_trend_trail50_noreg",
         params=_base_params(trail_pct=0.005, stop_pct=0.010, entry_end_min=13 * 60),
         signal=dict(entry_mode="vwap_ema", regime="none", exit_mode="none")),
    dict(name="a_trend_both_trail50_reg",
         params=_base_params(trail_pct=0.005, stop_pct=0.010, entry_end_min=13 * 60,
                             allow_short=True, max_positions=4),
         signal=dict(entry_mode="vwap_ema", regime="spy_trend", exit_mode="none")),
    dict(name="a_trend_late_trail50_reg",
         params=_base_params(trail_pct=0.005, stop_pct=0.010, entry_start_min=10 * 60 + 30,
                             entry_end_min=14 * 60 + 30),
         signal=dict(entry_mode="vwap_ema", regime="spy_trend", exit_mode="none")),
    dict(name="a_prevbreak_trail50_reg",
         params=_base_params(trail_pct=0.005, stop_pct=0.010, entry_end_min=14 * 60),
         signal=dict(entry_mode="prev_break", regime="spy_trend", exit_mode="none")),
    dict(name="a_trend_cap2_reg",
         params=_base_params(trail_pct=0.005, stop_pct=0.010, entry_end_min=13 * 60 + 30,
                             max_entries_per_session=2, min_minutes_between_entries=60),
         signal=dict(entry_mode="vwap_ema", regime="spy_trend", exit_mode="none")),
]

GRID_C = [
    dict(name="c_or30_long_atr1_0_eod",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=12 * 60,
                             max_entries_per_session=1),
         extras=dict(stop_atr_mult=1.0, target_atr_mult=0.0, trail_atr_mult=0.0),
         signal=dict(or_minutes=30, allow_short=False)),
    dict(name="c_or30_long_atr1_2",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=12 * 60),
         extras=dict(stop_atr_mult=1.0, target_atr_mult=2.0, trail_atr_mult=0.0),
         signal=dict(or_minutes=30, allow_short=False)),
    dict(name="c_or30_both_atr1_2",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=12 * 60,
                             allow_short=True, max_positions=4),
         extras=dict(stop_atr_mult=1.0, target_atr_mult=2.0, trail_atr_mult=0.0),
         signal=dict(or_minutes=30, allow_short=True)),
    dict(name="c_or30_both_atr1_2_late",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60,
                             allow_short=True, max_positions=4),
         extras=dict(stop_atr_mult=1.0, target_atr_mult=2.0, trail_atr_mult=0.0),
         signal=dict(or_minutes=30, allow_short=True)),
    dict(name="c_or60_both_atr1_2",
         params=_base_params(entry_start_min=11 * 60, entry_end_min=13 * 60,
                             allow_short=True, max_positions=4),
         extras=dict(stop_atr_mult=1.0, target_atr_mult=2.0, trail_atr_mult=0.0),
         signal=dict(or_minutes=60, allow_short=True)),
    dict(name="c_or30_both_trail15atr",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60,
                             allow_short=True, max_positions=4),
         extras=dict(stop_atr_mult=1.5, target_atr_mult=0.0, trail_atr_mult=1.5),
         signal=dict(or_minutes=30, allow_short=True)),
    dict(name="c_or30_both_atr05_1_cap2",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60,
                             allow_short=True, max_positions=4,
                             max_entries_per_session=2, min_minutes_between_entries=90),
         extras=dict(stop_atr_mult=0.5, target_atr_mult=1.0, trail_atr_mult=0.0),
         signal=dict(or_minutes=30, allow_short=True)),
    dict(name="c_or60_both_trail15atr",
         params=_base_params(entry_start_min=11 * 60, entry_end_min=14 * 60,
                             allow_short=True, max_positions=4),
         extras=dict(stop_atr_mult=1.5, target_atr_mult=0.0, trail_atr_mult=1.5),
         signal=dict(or_minutes=60, allow_short=True)),
]

GRID_D = [
    dict(name="d_pair_z15_soxl_tqqq",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             allow_short=True, max_positions=4, notional_usd=50_000.0),
         signal=dict(kind="ratio_z", leg_long="SOXL", leg_short="TQQQ",
                     z_entry=1.5, z_exit=0.0, beta=1.0)),
    dict(name="d_pair_z20_soxl_tqqq",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             allow_short=True, max_positions=4),
         signal=dict(kind="ratio_z", leg_long="SOXL", leg_short="TQQQ",
                     z_entry=2.0, z_exit=0.0, beta=1.0)),
    dict(name="d_pair_z15_spxl_tqqq",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             allow_short=True, max_positions=4),
         signal=dict(kind="ratio_z", leg_long="SPXL", leg_short="TQQQ",
                     z_entry=1.5, z_exit=0.0, beta=1.0)),
    dict(name="d_pair_z15_spxl_soxl",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             allow_short=True, max_positions=4),
         signal=dict(kind="ratio_z", leg_long="SPXL", leg_short="SOXL",
                     z_entry=1.5, z_exit=0.0, beta=1.0)),
    dict(name="d_pair_z15_fngu_tqqq",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             allow_short=True, max_positions=4),
         signal=dict(kind="ratio_z", leg_long="FNGU", leg_short="TQQQ",
                     z_entry=1.5, z_exit=0.0, beta=1.0)),
    dict(name="d_rs30_soxl_tqqq",
         params=_base_params(entry_start_min=10 * 60 + 30, entry_end_min=10 * 60 + 30,
                             allow_short=True, max_positions=4),
         signal=dict(kind="rs30", leg_long="SOXL", leg_short="TQQQ",
                     at_minute=10 * 60 + 30, threshold=0.005, beta=1.0)),
    dict(name="d_hedge_soxl_spy3",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60,
                             allow_short=True, max_positions=3, notional_usd=50_000.0),
         signal=dict(kind="hedge_rs", leg_long="SOXL", leg_short="SPY",
                     z_entry=1.5, z_exit=0.0, beta=3.0)),
    dict(name="d_pair_z15_soxl_tqqq_cap2",
         params=_base_params(entry_start_min=10 * 60 + 30, entry_end_min=15 * 60,
                             allow_short=True, max_positions=4,
                             max_entries_per_session=2, min_minutes_between_entries=120),
         signal=dict(kind="ratio_z", leg_long="SOXL", leg_short="TQQQ",
                     z_entry=1.5, z_exit=0.0, beta=1.0)),
]

GRID_B = [
    dict(name="b_mr_z10_reg30",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.010, time_exit_minutes=30),
         signal=dict(z_entry=1.0, z_exit=0.0, regime="lowtrend",
                     vol_floor=0.0005, max_abs_spy_ret=0.0020)),
    dict(name="b_mr_z15_reg30",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.010, time_exit_minutes=30),
         signal=dict(z_entry=1.5, z_exit=0.0, regime="lowtrend",
                     vol_floor=0.0005, max_abs_spy_ret=0.0020)),
    dict(name="b_mr_z20_reg30",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.010, time_exit_minutes=30),
         signal=dict(z_entry=2.0, z_exit=0.0, regime="lowtrend",
                     vol_floor=0.0005, max_abs_spy_ret=0.0020)),
    dict(name="b_mr_z15_reg60",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.015, time_exit_minutes=60),
         signal=dict(z_entry=1.5, z_exit=0.0, regime="lowtrend",
                     vol_floor=0.0005, max_abs_spy_ret=0.0020)),
    dict(name="b_mr_z15_reg_eod",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=13 * 60,
                             stop_pct=0.015, time_exit_minutes=None),
         signal=dict(z_entry=1.5, z_exit=0.0, regime="lowtrend",
                     vol_floor=0.0005, max_abs_spy_ret=0.0020)),
    dict(name="b_mr_z15_noreg",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.010, time_exit_minutes=30),
         signal=dict(z_entry=1.5, z_exit=0.0, regime="none")),
    dict(name="b_mr_z15_reg_short",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.010, time_exit_minutes=30, allow_short=True,
                             max_positions=4),
         signal=dict(z_entry=1.5, z_exit=0.0, regime="lowtrend",
                     vol_floor=0.0005, max_abs_spy_ret=0.0020)),
    dict(name="b_mr_z20_reg_tight",
         params=_base_params(entry_start_min=10 * 60, entry_end_min=14 * 60 + 30,
                             stop_pct=0.010, time_exit_minutes=30),
         signal=dict(z_entry=2.0, z_exit=0.5, regime="lowtrend",
                     vol_floor=0.0008, max_abs_spy_ret=0.0015)),
]

GRIDS = {"A": GRID_A, "C": GRID_C, "D": GRID_D, "B": GRID_B}


# ── the round-2 grids: four 3x3 full factorials, 9 cells each (R3-1) ────
#
# Written before any run was seen.  Every cell declares two — and only two —
# 3-level axes, so each cell has exactly two neighbours per axis whatever wins,
# and the neighbour rule is mechanical rather than a judgement.  N = 36 cells
# (4 groups x 9) counts every one of them, neighbours included; adding a cell
# increments N and re-declares the Bonferroni bar before any verdict is read.
#
# Family ids E1 / E2 are **E-continuation** and **E-fade**: the brief declares
# them as separate groups with the same two axes, and the sign of the hypothesis
# is the family, not a knob (a knob would make the pair look like one axis).

def _r2_params(family: str, **kw) -> dict:
    """The pinned caps and the family's entry window, with nothing implicit."""
    start, end = R2_ENTRY_WINDOWS[family]
    p = dict(sizing="fixed_notional",
             initial_equity=R2_CAPS["initial_equity"],
             position_size_pct=0.50, bp_usage_pct=0.95,
             notional_usd=R2_CAPS["notional_usd"],
             max_positions=R2_CAPS["max_positions"],
             entry_start_min=start, entry_end_min=end,
             max_entries_per_session=R2_CAPS["max_entries_per_session"],
             min_minutes_between_entries=R2_CAPS["min_minutes_between_entries"],
             stop_pct=None, target_pct=None, trail_pct=None,
             time_exit_minutes=None, allow_short=True,
             eod_flat_min=EOD_PINNED_MIN)
    p.update(kw)
    return p


def _r2_cell(family: str, name: str, signal: dict, **kw) -> dict:
    return dict(name=name, symbols=list(R2_UNIVERSE),
                params=_r2_params(family, **kw), extras={}, signal=signal)


def _num(x: float) -> str:
    """``0.5`` / ``1`` / ``1.5`` — a stable filename-friendly level label."""
    return f"{float(x):g}"


E_GAP_LEVELS = (0.5, 1.0, 1.5)
#: Axis 2 of both E groups: EOD flatten, a 60-minute stop, a 120-minute stop.
E_EXIT_LEVELS: tuple[tuple[str, Optional[int]], ...] = (
    ("eod", None), ("t60", 60), ("t120", 120))


def _gap_grid(family: str, tag: str) -> list[dict]:
    return [_r2_cell(family, f"{tag}_g{_num(g)}_{label}",
                     dict(gap_threshold=g), time_exit_minutes=tstop)
            for g in E_GAP_LEVELS for label, tstop in E_EXIT_LEVELS]


GRID_E1 = _gap_grid("E1", "e1_cont")
GRID_E2 = _gap_grid("E2", "e2_fade")

G_BREACH_LEVELS = (0.25, 0.5, 1.0)
G_STOP_LEVELS = (15, 30, 60)
GRID_G = [_r2_cell("G", f"g_orrev_b{_num(b)}_t{t}", dict(breach_atr=b),
                   time_exit_minutes=t)
          for b in G_BREACH_LEVELS for t in G_STOP_LEVELS]

H_DEV_LEVELS = (0.5, 1.0, 1.5)
H_RVOL_LEVELS = (0.0002, 0.0005, 0.0010)
GRID_H = [_r2_cell("H", f"h_vwap_d{_num(d)}_th{int(round(th * 1e4))}",
                   dict(dev_threshold=d, rvol_min=th),
                   time_exit_minutes=H_TIME_STOP_MINUTES)
          for d in H_DEV_LEVELS for th in H_RVOL_LEVELS]

GRIDS.update({"E1": GRID_E1, "E2": GRID_E2, "G": GRID_G, "H": GRID_H})

#: The round-2 families, in the order they are reported (E first: the two with
#: the largest documented effects, per the brief).
ROUND2_FAMILIES = ("E1", "E2", "G", "H")
ROUND2_GRID_SIZES = {f: len(GRIDS[f]) for f in ROUND2_FAMILIES}
