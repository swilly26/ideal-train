"""Measurement-knob contract tests for the turbo classic replay (2026-09-24).

Round 1 of the search for an edge varies ONE behaviour of the frozen classic
rule set at a time (see ``/home/team/shared/TURBO_REPLAY_ITERATION_01.md``), so
the knobs added to ``TurboConfig`` have to be provably inert by default: a run
of ``TurboConfig.classic()`` must be identical to a run whose knobs are
explicitly set to the live values, otherwise the 12-month baseline those
experiments are compared against would no longer describe what the code does.

The fixture is synthetic and seeded (no market cache needed), and the pinned
round-trip counts below are the fingerprint of the live behaviour on it: if a
default changes, or a knob stops being read, one of these tests fails.
"""
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from src.backtesting.turbo_engine import TurboConfig, replay

# ── fixture ────────────────────────────────────────────────────────────────
N_SESSIONS = 3
LIVE_KNOBS = dict(window_tie="earliest", mr_conf_mode="capped",
                  max_signal_age_bars=0, max_entries_per_session=0)


def make_frames(n_sessions: int = N_SESSIONS, seed: int = 7) -> dict:
    """Deterministic 1m RTH frames for the base four (3 sessions x 390 bars).

    A seeded random walk with an hour-on/hour-off drift, scheduled so every
    knob has something to bite on: |z| saturates (confidence ties), some
    sessions carry an entry whose winning signal bar is old, and entries are
    plentiful enough for a per-session cap to bind.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n_sessions)
    frames = {}
    for k, sym in enumerate(("SOXL", "TQQQ", "FNGU", "SPXL")):
        idx, closes = [], []
        price = 100.0 + 10 * k
        for d in days:
            for m in range(390):
                idx.append(pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30 + m))
                drift = 0.0002 if (m // 60) % 2 == 0 else -0.0004
                price *= 1.0 + rng.normal(0, 0.004) + drift
                closes.append(round(price, 4))
        frames[sym] = pd.DataFrame(
            {"open": closes, "high": [c * 1.0006 for c in closes],
             "low": [c * 0.9994 for c in closes], "close": closes,
             "volume": 1e6},
            index=pd.DatetimeIndex(idx),
        )
    return frames


@pytest.fixture(scope="module")
def frames() -> dict:
    return make_frames()


@pytest.fixture(scope="module")
def baseline(frames):
    return replay(frames, TurboConfig.classic())


# ── the defaults ARE the live classic behaviour ────────────────────────────
def test_knobs_default_to_the_live_values():
    cfg = TurboConfig.classic()
    assert LIVE_KNOBS == dict(window_tie=cfg.window_tie,
                              mr_conf_mode=cfg.mr_conf_mode,
                              max_signal_age_bars=cfg.max_signal_age_bars,
                              max_entries_per_session=cfg.max_entries_per_session)
    # ``classic()`` adds no knob override, so the dataclass defaults are live
    for f in fields(TurboConfig):
        if f.name in LIVE_KNOBS:
            assert getattr(cfg, f.name) == f.default


def test_classic_defaults_reproduce_the_baseline_run(frames, baseline):
    """The published 12-month baseline was produced with these defaults.

    A run whose knobs are set *explicitly* to the live values must be the same
    run, trade for trade -- that is what makes "one knob at a time" meaningful.
    The round-trip count is pinned as the fingerprint of the live path on this
    fixture; the real-data reproduction command is in the module docstring of
    ``scripts/run_turbo_backtest.py``.
    """
    explicit = replay(frames, replace(TurboConfig.classic(), **LIVE_KNOBS))
    pd.testing.assert_frame_equal(baseline.trades, explicit.trades)
    assert baseline.stats["round_trips"] == explicit.stats["round_trips"] == 89
    assert baseline.stats["skipped"]["stale_signal"] == 0
    assert baseline.stats["skipped"]["session_cap"] == 0


# ── every knob actually changes something ─────────────────────────────────
def test_window_tie_latest_changes_the_entry(frames, baseline):
    """The live loop resolves a confidence tie toward the OLDEST bar; the knob
    flips it to the most recent one, which changes which entry fires."""
    r = replay(frames, replace(TurboConfig.classic(), window_tie="latest"))
    assert not r.trades.equals(baseline.trades)
    assert r.stats["round_trips"] == 99


def test_mr_conf_mode_linear_changes_selection(frames, baseline):
    """``capped`` saturates at |z| = 1.0 (ties everywhere); ``linear`` ranks by
    the real |z|, so a different bar -- and a different direction -- wins."""
    r = replay(frames, replace(TurboConfig.classic(), mr_conf_mode="linear"))
    assert not r.trades.equals(baseline.trades)
    assert r.stats["round_trips"] == 78
    assert r.stats["round_trips"] != baseline.stats["round_trips"]


def test_max_signal_age_bars_cuts_stale_entries(frames, baseline):
    """A stale signal (winning bar up to an hour old) is refused, not filled."""
    r = replay(frames, replace(TurboConfig.classic(), max_signal_age_bars=5))
    assert not r.trades.equals(baseline.trades)
    assert r.stats["skipped"]["stale_signal"] > 0
    assert r.stats["round_trips"] < baseline.stats["round_trips"]


def test_max_entries_per_session_caps_each_session(frames, baseline):
    cap = replay(frames, replace(TurboConfig.classic(), max_entries_per_session=2))
    assert cap.stats["round_trips"] == 2 * N_SESSIONS       # 2 entries x 3 sessions
    assert cap.stats["skipped"]["session_cap"] > 0
    assert cap.stats["round_trips"] < baseline.stats["round_trips"]
    five = replay(frames, replace(TurboConfig.classic(), max_entries_per_session=5))
    assert five.stats["round_trips"] == 15
    assert cap.stats["round_trips"] < five.stats["round_trips"]


def test_knobs_reject_impossible_values():
    with pytest.raises(ValueError):
        TurboConfig(window_tie="oldest")
    with pytest.raises(ValueError):
        TurboConfig(mr_conf_mode="quadratic")
    with pytest.raises(ValueError):
        TurboConfig(max_signal_age_bars=-1)
    with pytest.raises(ValueError):
        TurboConfig(max_entries_per_session=-1)


# ── the reported cost lines cannot imply a cost model that was not run ────
def test_cost_drag_is_per_trade_and_agrees_with_the_totals(frames, baseline):
    t = baseline.trades
    assert "cost_drag" in t.columns
    # drag = adverse slippage embedded in both fills + commissions
    assert (t["cost_drag"] >= t["fees"] - 1e-9).all()
    assert t["cost_drag"].sum() == pytest.approx(
        baseline.stats["cost_drag_same_fills"], rel=1e-9)
    assert baseline.stats["cost_drag_same_fills"] > 0


def test_pnl_gross_is_before_commissions_not_before_costs(frames):
    """``pnl_gross`` moves with the slippage embedded in the fill prices, so it
    is never a zero-cost figure; only ``--variant zero_cost`` is."""
    baseline = replay(frames, TurboConfig.classic())
    t = baseline.trades
    assert t["pnl_gross"].to_numpy() == pytest.approx(
        (t["pnl_after_costs"] + t["fees"]).to_numpy(), rel=1e-9, abs=1e-6)
    zero = replay(frames, TurboConfig.classic().zero_cost())
    assert zero.stats["cost_drag_same_fills"] == 0.0
    z = zero.trades
    assert z["pnl_gross"].to_numpy() == pytest.approx(
        z["pnl_after_costs"].to_numpy(), rel=1e-9, abs=1e-6)
    # the two cost models do not produce the same trades/prices
    assert not zero.trades.equals(baseline.trades)


def test_inert_flags_are_reported_for_the_current_config():
    """``current_base4()`` sets three flags this engine never reads; a run that
    reported them as "the current policy" would claim a comparison it did not
    make, so the config names them."""
    assert TurboConfig.classic().inert_flags() == ()
    assert TurboConfig.current_base4().inert_flags() == (
        "regime_gate", "allow_shorts", "mr_short")
