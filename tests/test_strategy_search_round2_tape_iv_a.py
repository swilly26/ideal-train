"""L11.1 / L11.2 / L11.6 — the mandatory flatten on a tape with an intra-day hole.

Source: ``/home/team/shared/ROUND2_TAPE_POLICY_RULING.md`` (binding on the
round-2 implementation), grounded on
``/home/team/shared/ROUND2_TAPE_BLOCKER_PIN.md``.

The pinned round-2 universe could not be run **at all** on the pre-change
engine: six sessions across the two windows die at
``SPXL: mandatory EOD flatten at … could not be priced (a leg has no bar)``,
because a symbol's own tape has a genuine hole across the flatten minute
(15:30).  The lead ruled (L11.1, adopted as option (iv-a)) that the mandatory
flatten fills at the **first session bar at or after ``eod_flat_min`` at which
every leg of every open position has a bar** — one flatten minute per session,
shared by the whole book — that **no stale carried-forward close may ever price
a fill** (L11.2, rejected permanently), that the **session itself is the bound**
(no invented "+≤30 min" literal), and that a session ending with a position
still open stays **fatal** with the tape blocker's own message.  L11.6 requires
every run to carry the delayed-flatten accounting.

Every test in this file fails on the pre-change commit ``e72dc2a`` (the
recorded pre-change run is ``docs/ROUND2_TAPE_IVA_PREFIX_PYTEST.txt``), and
each one is written **by value**: the assertion is the price the flatten filled
at and the minute it filled at, never merely "the run did not raise".

* (a) a leg missing the 15:30 bar but present at 15:34 — completes, and the
  flatten fill **is** that 15:34 bar's price;
* (b) the same shape with the leg absent from 15:30 to the session end — still
  fatal, fail-closed message unchanged;
* (c) a session with no hole — the flatten minute is still 15:30, so the rule is
  a no-op where the tape is complete;
* (d) the delayed fill is **not** a stale close: replacing the last known close
  before the hole does not move the asserted flatten price;
* (e) the flatten minute is a property of the whole book — a position whose own
  legs are priceable at 15:30 waits for the position that is not;
* the L11.6 accounting, by value, on every case above.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import families  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Instrument,
    Leg,
    Market,
    replay_legs,
    run_search,
)
from tests.test_strategy_search_engine import (  # noqa: E402
    OPEN_MIN,
    simple_cfg,
)

SESSION_BARS = 390                       # 09:30 .. 15:59
FLAT_INDEX = 15 * 60 + 30 - OPEN_MIN     # bar 360 — the 15:30 flatten minute
FLAT_TS = pd.Timestamp("2025-01-02") + pd.Timedelta(minutes=15 * 60 + 30)
DELAY_INDEX = FLAT_INDEX + 4             # bar 364 — 15:34
DELAY_TS = pd.Timestamp("2025-01-02") + pd.Timedelta(minutes=15 * 60 + 34)
PX = 100.0                               # above $50, so slippage is 2 bps not the 1¢ floor
HOLED_PX = 137.0                         # the price the holed symbol prints again at 15:34
STALE_PX = 999.0                         # a price only a carried-forward close could use
FAIL_CLOSED = (r"mandatory EOD flatten at .* could not be priced "
               r"\(a leg has no bar\): refusing to carry the position")


# ── fixtures ───────────────────────────────────────────────────────────


def frame(prices: dict, day: str = "2025-01-02") -> pd.DataFrame:
    """One bar per minute offset in *prices*: ``{minute_index: close}``.

    ``minute_index`` is minutes after the 09:30 open, so 360 is 15:30 and 364 is
    15:34.  Open == close == the given price: the flatten fills at the **close**,
    so a fixture that wants to distinguish bar *k* from bar *k+1* only has to
    give them different prices.
    """
    idx, rows = [], []
    for m in sorted(prices):
        c = float(prices[m])
        idx.append(pd.Timestamp(day) + pd.Timedelta(minutes=OPEN_MIN + m))
        rows.append((c, c + 0.05, c - 0.05, c, 1000.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex(idx, name="ts"))


def full_tape(price: float = PX, day: str = "2025-01-02") -> pd.DataFrame:
    """A complete session: the control the rule must be a no-op on."""
    return frame({m: price for m in range(SESSION_BARS)}, day=day)


def holed_tape(price: float = PX, live: float = HOLED_PX,
               hole: tuple = (FLAT_INDEX, DELAY_INDEX - 1),
               last_bar: int = SESSION_BARS - 1,
               day: str = "2025-01-02") -> pd.DataFrame:
    """A session with no bars inside *hole* and a different price after it.

    Default hole is 15:30 .. 15:33 with 15:34 printing ``live`` — the tape
    blocker's own shape, four minutes late.  ``last_bar`` shortens the session
    so a variant can be written that never prints again (test (b)).
    """
    lo, hi = hole
    prices = {m: price for m in range(0, min(lo, last_bar + 1))}
    prices.update({m: live for m in range(hi + 1, last_bar + 1)})
    return frame(prices, day=day)


def one_trip(market: Market, legs, key: str, sig_index: int = 5,
             direction: int = 1) -> dict:
    """One position that opens on bar ``sig_index`` + 1 and is held to the flat."""
    entry = np.zeros(market.n(), dtype=np.int8)
    entry[sig_index] = direction
    inst = Instrument(key=key, legs=legs, entry_dir=entry,
                      exit_now=np.zeros(market.n(), dtype=np.int8),
                      valid=market.valid(legs),
                      kind="single" if len(legs) == 1 else "pair")
    return {key: inst}


def holed_market(hole: tuple = (FLAT_INDEX, DELAY_INDEX - 1), price: float = PX,
                 live: float = HOLED_PX, last_bar: int = SESSION_BARS - 1,
                 companion: bool = True) -> Market:
    """SOXL with a tape hole, plus a companion symbol that is whole.

    The companion is what makes the fixture the tape blocker's own shape: the
    global axis carries the flatten-minute bar (from the companion) while the
    *leg* has none, which is precisely "a leg has no bar" — and it is why the
    pinned universe was blocked at all.  A single-symbol holed tape is a
    different (and much weaker) fixture: the axis simply has no flatten-minute
    bar, so the pre-change engine flattened at the next bar it did have.
    """
    frames = {"SOXL": holed_tape(price=price, live=live, hole=hole,
                                 last_bar=last_bar)}
    if companion:
        frames["SPY"] = full_tape()
    return Market(frames)


def trip_of(res) -> pd.Series:
    assert len(res.trades) == 1, "the fixture signals exactly one trip"
    return res.trades.iloc[0]


# ── (a) a hole at the flatten minute delays the flatten, by value ──────


def test_a_hole_at_the_flatten_minute_fills_at_the_next_priceable_bar():
    """15:34 is present, so the run completes and fills there — not at 15:30.

    The zero-cost replay asserts the **raw price of that bar**: with every cost
    switched off a fill *is* the price it was made at, so ``exit_price == 137``
    is a statement about the tape, not about the cost model.
    """
    market = holed_market()
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    cfg = simple_cfg(max_positions=1)

    zero = trip_of(run_search(market, insts, cfg, CostModel.zero_cost()))
    assert zero["exit_reason"] == "eod", "it is the mandatory flatten that moved"
    assert zero["exit_time"] == DELAY_TS, "the flatten minute is 15:34, not 15:30"
    assert zero["exit_price"] == pytest.approx(HOLED_PX), \
        "the flatten fill is the 15:34 bar's own price"

    costs = CostModel.baseline()
    tr = trip_of(run_search(market, insts, cfg, costs))
    assert tr["exit_time"] == DELAY_TS
    assert tr["exit_price"] == pytest.approx(
        costs.fill_price(HOLED_PX, is_buy=False)), \
        "the exit fill is the 15:34 close, slipped adversely like any sell"


def test_a_delayed_flatten_is_counted_with_its_delay_in_minutes():
    """L11.6: the delay is in the record, by value (4 minutes, one trip)."""
    market = holed_market()
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    res = run_search(market, insts, simple_cfg(max_positions=1),
                     CostModel.baseline())
    assert res.stats["eod_flatten_delayed_trips"] == 1
    assert res.stats["eod_flatten_max_delay_min"] == 4
    assert res.stats["exits"] == {"eod": 1}


def test_the_delay_is_measured_from_the_config_s_own_flatten_minute():
    """The 15:25 neighbour cell's flatten is due at 15:25, not 15:30 (P2).

    Its own hole sits on its own flatten minute (15:25 .. 15:28), and the flatten
    then fires at 15:29: a four-minute delay, counted against the declared
    minute.  L11.6's "delayed past 15:30" is therefore recorded as "delayed past
    the config's ``eod_flat_min``", which is the same statement on the pinned
    cell and the honest one on the neighbour.
    """
    market = holed_market(hole=(355, 358))
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    cfg = simple_cfg(max_positions=1, eod_flat_min=15 * 60 + 25)
    res = run_search(market, insts, cfg, CostModel.baseline())
    tr = trip_of(res)
    assert tr["exit_time"] == pd.Timestamp("2025-01-02 15:29")
    assert res.stats["eod_flatten_delayed_trips"] == 1
    assert res.stats["eod_flatten_max_delay_min"] == 4, "15:25 → 15:29"


# ── (b) a session that can never price the flatten is still fatal ──────


def test_a_session_that_never_prices_the_flatten_is_still_fatal():
    """The failure the tape blocker found keeps its fail-closed message.

    This is the pre-change failure mode itself: the holed symbol never prints at
    or after 15:30, so no flatten minute exists inside the session.  The run must
    raise, must name the instrument, and must keep the message the pin recorded —
    only the bar it names moves (the session's own last bar, which is the point
    at which the absence became knowable).
    """
    market = Market({"SOXL": holed_tape(live=PX, hole=(FLAT_INDEX, SESSION_BARS),
                                        last_bar=FLAT_INDEX - 1)})
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    with pytest.raises(RuntimeError, match=FAIL_CLOSED) as exc:
        run_search(market, insts, simple_cfg(max_positions=1),
                   CostModel.baseline())
    assert str(exc.value).startswith("SOXL: mandatory EOD flatten at ")
    assert "15:29:00" in str(exc.value), \
        "it names the session's last bar, where the tape stopped printing"


def test_a_surviving_position_is_fatal_at_the_session_boundary_too():
    """The same shape one session earlier in the tape: the boundary check fires.

    A position must never be carried into the next session — and the message it
    dies with must be the fail-closed one, not the generic "flatten did not run"
    bug guard.
    """
    empty_then_full = {"SOXL": pd.concat([
        holed_tape(live=PX, hole=(FLAT_INDEX, SESSION_BARS),
                   last_bar=FLAT_INDEX - 1),
        full_tape(day="2025-01-03"),
    ])}
    market = Market(empty_then_full)
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    with pytest.raises(RuntimeError, match=FAIL_CLOSED) as exc:
        run_search(market, insts, simple_cfg(max_positions=1),
                   CostModel.baseline())
    msg = str(exc.value)
    assert msg.startswith("SOXL: mandatory EOD flatten at ")
    assert "2025-01-02 15:29:00" in msg, "the bar the flatten was last due at"
    assert "position(s) survived" not in msg, \
        "the fail-closed case must not read as the did-not-run bug"


def test_a_positions_own_hole_is_not_fatal_when_the_tape_repairs_itself():
    """The pair case of (a): both legs priced, both on their own bars."""
    market = Market({"SOXL": holed_tape(), "SPY": full_tape()})
    legs = (Leg("SOXL", 1, 1.0), Leg("SPY", -1, 1.0))
    insts = one_trip(market, legs, "SOXL/SPY")
    cfg = simple_cfg(max_positions=4, notional_usd=25_000.0, allow_short=True)
    res = run_search(market, insts, cfg, CostModel.zero_cost())
    tr = trip_of(res)
    assert tr["exit_time"] == DELAY_TS
    assert res.stats["eod_flatten_delayed_trips"] == 1
    assert res.stats["eod_flatten_max_delay_min"] == 4
    detail = replay_legs(res.trades)[0]
    d_soxl, d_spy = detail[0], detail[1]
    assert d_soxl[3] == pytest.approx(HOLED_PX), "SOXL's leg fills at its 15:34 bar"
    assert d_spy[3] == pytest.approx(PX), "SPY's leg fills at its own 15:34 close"


# ── (c) the rule is a no-op on a complete session ──────────────────────


def test_a_session_with_no_hole_still_flattens_at_1530():
    """The 494 unaffected sessions must behave exactly as they did before."""
    market = Market({"SOXL": full_tape()})
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    cfg = simple_cfg(max_positions=1)
    zero = trip_of(run_search(market, insts, cfg, CostModel.zero_cost()))
    assert zero["exit_reason"] == "eod"
    assert zero["exit_time"] == FLAT_TS, "the flatten minute is still 15:30"
    assert zero["exit_price"] == pytest.approx(PX)
    res = run_search(market, insts, cfg, CostModel.baseline())
    assert res.stats["eod_flatten_delayed_trips"] == 0
    assert res.stats["eod_flatten_max_delay_min"] == 0
    assert res.stats["exits"] == {"eod": 1}


# ── (d) never a stale carried-forward close (L11.2) ────────────────────


def test_the_delayed_fill_is_not_a_stale_carried_forward_close():
    """Replace the last close before the hole; the flatten price must not move.

    This is the (iv-b) failure mode made visible: if the engine carried the last
    known close across the hole, the flatten would fill near ``STALE_PX`` and the
    two runs below would disagree.  They must agree, and both must be the 15:34
    bar's own price.
    """
    legs = (Leg("SOXL", 1, 1.0),)
    market_100 = holed_market(price=PX)
    market_999 = holed_market(price=STALE_PX)
    cfg = simple_cfg(max_positions=1)
    costs = CostModel.baseline()
    a = trip_of(run_search(market_100, one_trip(market_100, legs, "SOXL"), cfg,
                           costs))
    b = trip_of(run_search(market_999, one_trip(market_999, legs, "SOXL"), cfg,
                           costs))
    assert a["exit_time"] == DELAY_TS and b["exit_time"] == DELAY_TS
    assert a["exit_price"] == pytest.approx(
        costs.fill_price(HOLED_PX, is_buy=False))
    assert b["exit_price"] == pytest.approx(a["exit_price"], rel=1e-12), \
        "the pre-hole close must not reach the fill"
    assert abs(b["exit_price"] - STALE_PX) > 1.0, \
        "a stale carried-forward close is what (iv-b) would have filled at"
    zero = trip_of(run_search(market_999, one_trip(market_999, legs, "SOXL"),
                              cfg, CostModel.zero_cost()))
    assert zero["exit_price"] == pytest.approx(HOLED_PX)


def test_the_run_record_carries_the_delayed_flatten_accounting(tmp_path,
                                                              monkeypatch):
    """L11.6 at the record level: the counters must reach the artefact.

    ``Runner.run`` is the only writer of a round-2 run record, so the check runs
    the real record path against this file's holed tape — the family build is
    replaced with the fixture's config, everything downstream (the replay, the
    folds, the record, the filename, the write) is the runner's own code.
    """
    import dataclasses

    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import run_strategy_search as mod

    market = holed_market()
    insts = one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL")
    resolved = {"params": {"sizing": "fixed_notional",
                           "eod_flat_min": 15 * 60 + 30},
                "signal": {"fixture": True}, "extras": {}}
    cfg = dataclasses.replace(simple_cfg(max_positions=1),
                              name=families.resolved_name("A", resolved))
    monkeypatch.setattr(mod.families, "build",
                        lambda *a, **k: (cfg, insts, resolved))
    runner = mod.Runner(cache_dir=Path("/nonexistent-cache"),
                        out_dir=Path(tmp_path), verbose=False, reconcile=False)
    runner._window = lambda window: {"frames": {}, "market": market,
                                     "aligned": None, "cov": {},
                                     "dates": ["2025-01-02", "2025-01-31"]}
    spec = {"name": "iva_cell", "params": {}, "signal": {}, "extras": {},
            "symbols": ["SOXL"]}
    record = runner.run("A", spec, "W1", "baseline", "fixed_notional")
    assert record["exits"] == {"eod": 1}
    assert record["eod_flatten_delayed_trips"] == 1, \
        "the record must carry the delayed-flatten count, by value"
    assert record["eod_flatten_max_delay_min"] == 4, \
        "and the largest delay in minutes"
    on_disk = json.loads((Path(tmp_path) / next(
        Path(tmp_path).iterdir()).name).read_text())
    assert on_disk["eod_flatten_delayed_trips"] == 1
    assert on_disk["eod_flatten_max_delay_min"] == 4


# ── (e) one flatten minute for the whole book ──────────────────────────


def test_the_flatten_minute_is_shared_by_every_open_position():
    """A priceable position waits for the position that is not (L11.1).

    "Every leg of **every open position**" is the ruling's wording, so the
    session keeps one flatten minute: SPXL-style hole in one symbol delays the
    whole book, exactly as the fixed 15:30 minute flattened the whole book
    before.  Both trips are therefore delayed, by the same four minutes.
    """
    market = Market({"SOXL": holed_tape(), "SPY": full_tape()})
    insts = {}
    insts.update(one_trip(market, (Leg("SOXL", 1, 1.0),), "SOXL"))
    insts.update(one_trip(market, (Leg("SPY", 1, 1.0),), "SPY"))
    cfg = simple_cfg(max_positions=4, notional_usd=25_000.0)
    res = run_search(market, insts, cfg, CostModel.zero_cost())
    assert len(res.trades) == 2, "both positions must be open through 15:30"
    assert set(res.trades["exit_time"]) == {DELAY_TS}, \
        "one flatten minute per session: the book moves together"
    assert res.stats["exits"] == {"eod": 2}
    assert res.stats["eod_flatten_delayed_trips"] == 2
    assert res.stats["eod_flatten_max_delay_min"] == 4
    by_instrument = dict(zip(res.trades["instrument"], res.trades["exit_price"]))
    assert by_instrument["SPY"] == pytest.approx(PX)
    assert by_instrument["SOXL"] == pytest.approx(HOLED_PX)
