"""Engine + feature tests for the stage-1 strategy search.

These pin the four properties the brief calls non-negotiable (§4), each with a
test that fails if the property is lost:

* indicators are computed **per session** and never across the overnight gap,
* a signal read on bar *t* is filled on bar *t+1*, so a signal that is only
  profitable with a same-bar fill actually loses,
* the entry-time gate is enforced (the first 30 minutes are known-negative),
* every position is flattened at the session's EOD time, never carried overnight,

plus the mechanical screen's kill rule (no W2 / no zero-cost run once a family
has no W1 survivor).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import families, screen  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Instrument,
    Leg,
    Market,
    SearchConfig,
    run_search,
)
from src.backtesting.strategy_search.features import (  # noqa: E402
    FeatureBook,
    build_features,
)

OPEN_MIN = 9 * 60 + 30


def mk_frame(sessions: dict, volume: float = 1000.0) -> pd.DataFrame:
    """Build an RTH 1m frame from ``{date: [(o, h, l, c), ...]}``."""
    idx, rows = [], []
    for day, bars in sessions.items():
        for i, (o, h, l, c) in enumerate(bars):
            idx.append(pd.Timestamp(day) + pd.Timedelta(minutes=OPEN_MIN + i))
            rows.append((o, h, l, c, volume))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex(idx, name="ts"))


def flat_session(day: str, price: float, bars: int = 21) -> list:
    return [(price, price + 0.02, price - 0.02, price)] * bars


def simple_cfg(**kw) -> SearchConfig:
    base = dict(family="A", name="test", sizing="fixed_notional",
                notional_usd=50_000.0, initial_equity=100_000.0,
                max_positions=2, entry_start_min=OPEN_MIN,
                entry_end_min=15 * 60 + 29, max_entries_per_session=1,
                eod_flat_min=15 * 60 + 30)
    base.update(kw)
    return SearchConfig(**base)


# ── per-session indicators (no overnight leak) ─────────────────────────


def test_indicators_do_not_carry_across_the_overnight_gap():
    s1 = [(100.0 + i, 100.6 + i, 99.4 + i, 100.5 + i) for i in range(25)]
    s2 = [(200.0 + i, 200.6 + i, 199.4 + i, 200.5 + i) for i in range(25)]
    df = mk_frame({"2025-01-02": s1, "2025-01-03": s2})
    f = build_features(df)
    first2 = df.index.normalize() == pd.Timestamp("2025-01-03")
    i = int(np.argmax(first2))
    # VWAP of the first bar of a session is that bar's own typical price
    tp = (200.6 + 199.4 + 200.5) / 3.0
    assert f["vwap"].iloc[i] == pytest.approx(tp)
    # a rolling window has no history at the session's first bar
    assert np.isnan(f["z20"].iloc[i])
    assert pd.isna(f["atr"].iloc[i])
    # the EMA restarts: it is that bar's own close, not yesterday's level
    assert f["ema_fast"].iloc[i] == pytest.approx(df["close"].iloc[i])
    # the only deliberate cross-session input is a *completed* prior session
    assert f["prev_hi"].iloc[i] == pytest.approx(max(h for _, h, _, _ in s1))
    assert f["prev_lo"].iloc[i] == pytest.approx(min(l for _, _, l, _ in s1))
    assert np.isnan(f["prev_hi"].iloc[0])


def test_opening_range_levels_are_the_first_n_bars_of_that_session_only():
    bars = [(100.0, 100.0 + i * 0.1, 99.9, 100.0) for i in range(70)]
    df = mk_frame({"2025-01-02": bars})
    f = build_features(df)
    assert f["or30_hi"].iloc[10] == pytest.approx(100.0 + 29 * 0.1)
    assert f["or60_hi"].iloc[10] == pytest.approx(100.0 + 59 * 0.1)
    assert f["or30_lo"].iloc[40] == pytest.approx(99.9)


# ── no lookahead: fill on t+1 ──────────────────────────────────────────


def _one_symbol_market(day_bars: dict) -> tuple[Market, np.ndarray]:
    m = Market({"SOXL": mk_frame(day_bars)})
    return m, m.axis


def test_a_signal_that_only_wins_with_a_same_bar_fill_loses_at_t_plus_1():
    """Bar 16 closes at 110; bar 17 opens at 111 and stays there.

    Filling at the signal bar's close would book +$1/share.  Filling where the
    engine is allowed to fill — the next bar's open — books nothing and pays
    the costs, so the trip must be negative.
    """
    bars = [(100.0, 100.05, 99.95, 100.0)] * 16
    bars.append((100.0, 110.0, 99.9, 110.0))          # the signal bar
    bars += [(111.0, 111.05, 110.95, 111.0)] * 4      # gap up, then flat
    market, axis = _one_symbol_market({"2025-01-02": bars})
    entry = np.zeros(len(axis), dtype=np.int8)
    entry[16] = 1                                     # signal on bar 16's close
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(len(axis), dtype=np.int8),
                      valid=market.valid((Leg("SOXL", 1, 1.0),)))
    cfg = simple_cfg(eod_flat_min=OPEN_MIN + 20)      # flatten at the last bar
    res = run_search(market, {"SOXL": inst}, cfg, CostModel.baseline())
    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    assert trade["entry_time"] == axis[17], "entry must fill on the bar AFTER the signal"
    assert trade["entry_price"] == pytest.approx(
        CostModel.baseline().fill_price(111.0, is_buy=True))
    assert trade["pnl_after_costs"] < 0, "the move was already gone at t+1"
    # and the same-bar convention would have shown a profit — i.e. the test bites
    naive = (111.0 - 110.0) * trade["notional"] / 111.0
    assert naive - trade["cost_drag"] > 0


def test_a_signal_is_never_filled_across_the_overnight_gap():
    bars = [(100.0, 100.05, 99.95, 100.0)] * 21
    market, axis = _one_symbol_market({"2025-01-02": bars, "2025-01-03": bars})
    entry = np.zeros(len(axis), dtype=np.int8)
    entry[20] = 1                      # last bar of session 1 signals
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(len(axis), dtype=np.int8),
                      valid=market.valid((Leg("SOXL", 1, 1.0),)))
    res = run_search(market, {"SOXL": inst}, simple_cfg(), CostModel.baseline())
    assert len(res.trades) == 0


# ── the entry-time gate ────────────────────────────────────────────────


def test_no_entry_is_filled_before_the_declared_gate():
    """The first 30 minutes were negative in every round-1 variant."""
    bars = [(100.0 + 0.05 * i, 100.1 + 0.05 * i, 99.9 + 0.05 * i, 100.0 + 0.05 * i)
            for i in range(390)]
    market, axis = _one_symbol_market({"2025-01-02": bars})
    entry = np.ones(len(axis), dtype=np.int8)          # signals on every close
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(len(axis), dtype=np.int8),
                      valid=market.valid((Leg("SOXL", 1, 1.0),)))
    cfg = simple_cfg(entry_start_min=OPEN_MIN + 30, max_positions=1)
    res = run_search(market, {"SOXL": inst}, cfg, CostModel.baseline())
    assert len(res.trades) >= 1
    first_entry = min(pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute
                      for t in res.trades["entry_time"])
    assert first_entry >= OPEN_MIN + 31, "an entry slipped into the first 30 minutes"


# ── EOD flatten ────────────────────────────────────────────────────────


def test_every_position_is_flattened_at_the_session_close():
    bars = [(100.0 + 0.01 * i, 100.2 + 0.01 * i, 99.8 + 0.01 * i, 100.0 + 0.01 * i)
            for i in range(390)]
    market, axis = _one_symbol_market({"2025-01-02": bars, "2025-01-03": bars})
    entry = np.ones(len(axis), dtype=np.int8)
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(len(axis), dtype=np.int8),
                      valid=market.valid((Leg("SOXL", 1, 1.0),)))
    cfg = simple_cfg(max_positions=1, entry_start_min=OPEN_MIN + 30)
    res = run_search(market, {"SOXL": inst}, cfg, CostModel.baseline())
    assert len(res.trades) == 2, "one position per session, never carried over"
    minutes = [pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute
               for t in res.trades["exit_time"]]
    assert all(m == 15 * 60 + 30 for m in minutes)
    assert res.trades["hold_minutes"].max() < 390


# ── sizing ─────────────────────────────────────────────────────────────


def test_fixed_notional_sizing_puts_fifty_thousand_on_the_trade():
    bars = [(100.0, 100.2, 99.8, 100.0)] * 390
    market, axis = _one_symbol_market({"2025-01-02": bars})
    entry = np.zeros(len(axis), dtype=np.int8)
    entry[60] = 1
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(len(axis), dtype=np.int8),
                      valid=market.valid((Leg("SOXL", 1, 1.0),)))
    res = run_search(market, {"SOXL": inst}, simple_cfg(), CostModel.baseline())
    assert res.trades["notional"].iloc[0] == pytest.approx(50_000.0, rel=1e-6)


def test_a_pair_pays_costs_on_both_legs_and_is_one_round_trip():
    bars = [(100.0, 100.2, 99.8, 100.0)] * 390
    market, _ = _one_symbol_market({"2025-01-02": bars})
    m2 = Market({"SOXL": mk_frame({"2025-01-02": bars}),
                 "TQQQ": mk_frame({"2025-01-02": bars})})
    entry = np.zeros(m2.n(), dtype=np.int8)
    entry[60] = 1
    legs = (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0))
    inst = Instrument(key="SOXL/TQQQ", legs=legs, entry_dir=entry,
                      exit_now=np.full(m2.n(), 2, dtype=np.int8),
                      valid=m2.valid(legs), kind="pair")
    res = run_search(m2, {"SOXL/TQQQ": inst}, simple_cfg(max_positions=2),
                     CostModel.baseline())
    assert len(res.trades) == 1, "a pair is one round trip"
    trade = res.trades.iloc[0]
    assert trade["legs"] == 2
    assert trade["notional"] == pytest.approx(50_000.0, rel=1e-6)
    assert trade["pnl_after_costs"] < 0
    _ = market


def test_a_negative_signal_actually_shorts():
    """A −1 signal must mirror the declared legs, not re-open them long."""
    falling = [(100.0 - 0.01 * i, 100.2 - 0.01 * i, 99.8 - 0.01 * i, 100.0 - 0.01 * i)
               for i in range(390)]
    market, axis = _one_symbol_market({"2025-01-02": falling})
    entry = np.zeros(len(axis), dtype=np.int8)
    entry[60] = -1
    inst = Instrument(key="SOXL", legs=(Leg("SOXL", 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(len(axis), dtype=np.int8),
                      valid=market.valid((Leg("SOXL", 1, 1.0),)))
    res = run_search(market, {"SOXL": inst},
                     simple_cfg(allow_short=True), CostModel.baseline())
    assert len(res.trades) == 1
    trade = res.trades.iloc[0]
    assert trade["symbols"] == "SOXL-"
    assert trade["entry_price"] > trade["exit_price"], "a short sells high, buys low"
    assert trade["pnl_after_costs"] > 0, "a short in a falling tape must make money"


def test_a_negative_signal_mirrors_both_legs_of_a_pair():
    bars = [(100.0, 100.2, 99.8, 100.0)] * 390
    m2 = Market({"SOXL": mk_frame({"2025-01-02": bars}),
                 "TQQQ": mk_frame({"2025-01-02": bars})})
    entry = np.zeros(m2.n(), dtype=np.int8)
    entry[60] = -1
    legs = (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0))
    inst = Instrument(key="SOXL/TQQQ", legs=legs, entry_dir=entry,
                      exit_now=np.full(m2.n(), 2, dtype=np.int8),
                      valid=m2.valid(legs), kind="pair")
    res = run_search(m2, {"SOXL/TQQQ": inst},
                     simple_cfg(max_positions=2, allow_short=True),
                     CostModel.baseline())
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["symbols"] == "SOXL-,TQQQ+"
    assert res.trades.iloc[0]["pnl_after_costs"] < 0, "flat prices, costs only"


# ── §1 leg-fill symmetry: a flat pair must lose two full tolls ─────────
#
# The stage-2 brief's decisive test.  Two legs, identical price paths, so the
# true spread P&L is exactly zero and the only thing a round trip can book is
# the cost of four fills (entry and exit, both legs).  Both entries were
# signalled and, on the mirror, the *side* of each leg is flipped; a harness
# that flips the side but not the fill prices a short's exit as a sell and
# books that slippage as profit — which is how a flat tape turns into edge.

FLAT_PX = 100.0
FILLS_PER_TRIP = 4          # two legs × (entry + exit)


def _flat_pair(kind: str = "pair", n: int = 390, px: float = FLAT_PX):
    bars = flat_session("2025-01-02", px, n)
    m = Market({"SOXL": mk_frame({"2025-01-02": bars}),
                "TQQQ": mk_frame({"2025-01-02": bars})})
    return m, bars


def _pair_trip(direction: int, mode: str, n: int = 390,
               time_exit: int | None = None, eod: int = 15 * 60 + 30,
               costs: CostModel | None = None):
    """One pair trip on a flat tape; *mode* selects the exit path.

    ``mode="signal"`` emits the family's "either side may exit" value (2), so
    the trip closes on the bar after the signal.  ``mode="hold"`` emits the
    value that belongs to the *other* side, which no correct engine treats as
    an exit, so the trip is carried to the mandatory EOD flatten.  ``mode="no
    signal"`` emits 0 — the family saying "hold", which must also never exit.
    """
    m, _ = _flat_pair()
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[5] = direction
    value = {"signal": 2, "hold": -direction, "no_signal": 0}[mode]
    legs = (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0))
    inst = Instrument(key="SOXL/TQQQ", legs=legs, entry_dir=entry,
                      exit_now=np.full(m.n(), value, dtype=np.int8),
                      valid=m.valid(legs), kind="pair")
    cfg = simple_cfg(allow_short=True, max_positions=4, eod_flat_min=eod,
                     time_exit_minutes=time_exit)
    return run_search(m, {"SOXL/TQQQ": inst}, cfg,
                      costs if costs is not None else CostModel.baseline())


def _expected_toll(qty_leg: float, px: float = FLAT_PX,
                   costs: CostModel | None = None) -> float:
    """Two full tolls: four fills (entry + exit on each of two legs)."""
    c = costs if costs is not None else CostModel.baseline()
    return FILLS_PER_TRIP * qty_leg * c.slip_per_share(px)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("mode,exit_reason,hold", [
    ("signal", "signal", 1.0),   # a signalled exit, filled at t+1's open
    ("hold", "eod", None),       # held to the mandatory flatten, also t+1
])
def test_a_flat_pair_loses_two_full_tolls_and_never_books_a_profit(
        direction, mode, exit_reason, hold):
    res = _pair_trip(direction, mode)
    assert len(res.trades) == 1, "the fixture signals exactly one trip"
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == exit_reason
    if hold is not None:
        assert tr["hold_minutes"] == pytest.approx(hold)
    qty_leg = tr["notional"] / 2.0 / FLAT_PX          # $25k per leg at $100
    toll = _expected_toll(qty_leg)
    # The fills already carry the toll, so the *same-fill* gross is −toll and
    # adding the drag back reconstructs the strategy's own P&L: on identical
    # price paths that must be exactly zero, never a profit.
    assert tr["pnl_gross"] + tr["cost_drag"] == pytest.approx(0.0, abs=1e-6), \
        "identical price paths: the strategy's own spread P&L is exactly zero"
    assert tr["pnl_gross"] == pytest.approx(-toll, rel=1e-6)
    assert tr["cost_drag"] == pytest.approx(toll, rel=1e-6)
    assert tr["pnl_after_costs"] == pytest.approx(-toll, rel=1e-6), \
        "a flat pair must lose exactly one adverse fill per leg per side"
    assert tr["pnl_after_costs"] < 0, "never a positive P&L on a flat tape"


@pytest.mark.parametrize("mode", ["signal", "hold"])
def test_the_mirror_direction_pays_exactly_the_same_toll(mode):
    """Short SOXL / long TQQQ must cost what long SOXL / short TQQQ costs."""
    longs = _pair_trip(1, mode).trades.iloc[0]
    mirror = _pair_trip(-1, mode).trades.iloc[0]
    assert longs["exit_reason"] == mirror["exit_reason"]
    assert mirror["symbols"] == "SOXL-,TQQQ+"
    assert mirror["cost_drag"] == pytest.approx(longs["cost_drag"], rel=1e-12)
    assert mirror["pnl_after_costs"] == pytest.approx(longs["pnl_after_costs"],
                                                      rel=1e-12)
    assert mirror["pnl_after_costs"] < 0


@pytest.mark.parametrize("direction", [1, -1])
def test_every_exit_path_prices_both_legs_the_same_way(direction):
    """Entry, signalled exit, time exit and the EOD flatten, leg by leg.

    The fills are asserted against the cost model directly, so this fails if
    any exit path ever stops mirroring a leg's side.
    """
    c = CostModel.baseline()
    slip = c.slip_per_share(FLAT_PX)

    # entry (t+1 open) and signalled exit (t+1 open)
    tr = _pair_trip(direction, "signal").trades.iloc[0]
    soxl_long = direction > 0
    want_entry = FLAT_PX + slip if soxl_long else FLAT_PX - slip
    assert tr["entry_price"] == pytest.approx(want_entry)
    assert tr["exit_price"] == pytest.approx(
        FLAT_PX - slip if soxl_long else FLAT_PX + slip)

    # the mandatory EOD flatten, on the same tape
    eod = _pair_trip(direction, "hold").trades.iloc[0]
    qty_leg = eod["notional"] / 2.0 / FLAT_PX
    assert eod["exit_reason"] == "eod"
    assert eod["pnl_after_costs"] == pytest.approx(
        -FILLS_PER_TRIP * qty_leg * slip, rel=1e-6)

    # the time exit, which fills at the completing bar's close
    t = _pair_trip(direction, "hold", time_exit=3).trades.iloc[0]
    assert t["exit_reason"] == "time"
    qty_leg = t["notional"] / 2.0 / FLAT_PX
    assert t["pnl_after_costs"] == pytest.approx(
        -FILLS_PER_TRIP * qty_leg * slip, rel=1e-6)


@pytest.mark.parametrize("direction", [1, -1])
def test_a_pair_is_not_exited_on_a_bar_the_family_says_hold(direction):
    """``exit_now == 0`` means hold — it must never trigger an exit.

    A pair's reference side is 0 (it holds two legs), so comparing the exit
    signal against ``pos.side`` used to make *no signal* an exit and a
    *signalled* exit a hold, silently converting "exit at z = 0 or EOD" into
    "always hold to EOD".
    """
    res = _pair_trip(direction, "no_signal")
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "eod", \
        "exit_now == 0 must hold the position, not close it on the next bar"
    assert tr["hold_minutes"] > 1.0


# ── family C is a real breakout family, gated ──────────────────────────

def test_family_c_only_trades_breakouts_after_the_opening_range():
    or_bars = [(100.0, 100.6, 99.4, 100.0)] * 30
    move_bars = [(100.0 + 0.4 * i, 100.4 + 0.4 * i, 99.9 + 0.4 * i, 100.3 + 0.4 * i)
                 for i in range(1, 360)]
    df = mk_frame({"2025-01-02": or_bars + move_bars})
    market = Market({"SOXL": df, "SPY": df})
    book = FeatureBook({"SOXL": df, "SPY": df})
    aligned = book.align(market.axis)
    spec = dict(families.GRID_C[1])
    spec["symbols"] = ["SOXL"]
    cfg, instruments, _resolved = families.build("C", spec, market, aligned)
    res = run_search(market, instruments, cfg, CostModel.baseline())
    assert len(res.trades) >= 1, "a post-opening-range breakout must trade"
    minutes = [pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute
               for t in res.trades["entry_time"]]
    assert min(minutes) >= 10 * 60


# ── the screen is code, not judgement ──────────────────────────────────


def test_a_family_with_no_w1_survivor_is_killed_without_touching_w2():
    calls = []

    def fake_run(spec, window, cost, sizing):
        calls.append((spec["name"], window, cost, sizing))
        return {"family": "X", "config": spec["name"], "window": window,
                "sizing": sizing, "trips": 10, "net_pnl": -100.0, "net_pct": -0.001,
                "net_bps_per_trip": -4.0, "net_bps_t_stat": -1.0,
                "cost_drag_per_trip": 20.0, "notional_per_trip": 50_000.0,
                "win_rate": 0.4, "break_even_win_rate": 0.5,
                "folds_positive": 2, "folds_total": 12, "avg_hold_minutes": 30.0,
                "trips_per_session": 0.4, "folds": []}

    specs = [{"name": "x1", "params": {}, "signal": {}},
             {"name": "x2", "params": {}, "signal": {}}]
    verdict = screen.screen_family("X", specs, fake_run, log=lambda *_: None)
    assert verdict["verdict"] == "KILLED_ON_W1"
    assert {c[1] for c in calls} == {"W1"}, "a killed family must not run W2"
    assert {c[2] for c in calls} == {"baseline"}, "nor zero-cost"
    assert verdict["kill_reason"]


def test_a_config_that_clears_w1_and_fails_w2_is_not_a_survivor():
    def fake_run(spec, window, cost, sizing):
        scale = 1.0 if window == "W1" else -3.0
        return {"family": "X", "config": spec["name"], "window": window,
                "sizing": sizing, "trips": 200,
                "net_pnl": 1000 * scale, "net_pct": 0.01 * scale,
                "net_bps_per_trip": 5.0 * scale, "net_bps_t_stat": 1.0,
                "cost_drag_per_trip": 20.0, "notional_per_trip": 50_000.0,
                "win_rate": 0.5, "break_even_win_rate": 0.45,
                "folds_positive": 8, "folds_total": 12, "avg_hold_minutes": 30.0,
                "trips_per_session": 0.8,
                "folds": [{"month": f"2025-{m:02d}", "net_bps_per_trip": 1.0}
                          for m in range(1, 13)]}

    verdict = screen.screen_family("X", [{"name": "x1", "params": {}, "signal": {}}],
                                   fake_run, log=lambda *_: None)
    assert verdict["verdict"] == "NO_W2_SURVIVOR"
    assert verdict["ranked"] == []


def test_a_full_survivor_is_ranked_and_reports_both_windows():
    def fake_run(spec, window, cost, sizing):
        mult = 1.0 if cost == "baseline" else 1.5
        sign = -1.0 if cost == "zero_cost" and window == "W2" else 1.0
        return {"family": "X", "config": spec["name"], "window": window,
                "sizing": sizing, "trips": 200,
                "net_pnl": 2000 * mult, "net_pct": 0.02 * mult,
                "net_bps_per_trip": 6.0 * mult * sign, "net_bps_t_stat": 2.5,
                "cost_drag_per_trip": 20.0, "notional_per_trip": 50_000.0,
                "win_rate": 0.55, "break_even_win_rate": 0.5,
                "folds_positive": 9, "folds_total": 12, "avg_hold_minutes": 40.0,
                "trips_per_session": 0.8,
                "folds": [{"month": f"2025-{m:02d}", "round_trips": 20,
                           "net_bps_per_trip": 1.0}
                          for m in range(1, 13)]}

    verdict = screen.screen_family("X", [{"name": "x1", "params": {}, "signal": {}}],
                                   fake_run, log=lambda *_: None)
    assert verdict["verdict"] == "SURVIVORS"
    top = verdict["ranked"][0]
    assert top["config"] == "x1"
    assert top["net_pct_w1"] == pytest.approx(0.02)
    assert top["net_pct_w2"] == pytest.approx(0.02)
    assert top["folds_positive_min"] == 9
