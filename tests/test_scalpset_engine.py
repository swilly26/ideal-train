"""Hermetic tests for the ScalpSet historical replay engine.

Everything here is in-memory: hand-built 1m OHLCV frames + scripted signal
modules, so the fill / anchoring / gating rules are pinned without needing
35+ bars of synthetic history for the real modules to fire.  The real
modules are exercised separately (tests/test_scalp_*.py) and end-to-end by
the NVDA smoke run.

Covered live semantics (see ``src/backtesting/scalpset_engine.py``):
no-lookahead fill timing, anchored SL/TP vs fill (+ parity with the live
``anchor_scalp_levels``), degenerate stop → backstop, TP dropping, LIMIT
fills, stop/TP/EOD exits, cooldown, per-session entry cap, short gating,
whole-share sizing, RTH filtering, max positions, arbitration, metrics.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.backtesting import scalpset_engine as E
from src.strategies.scalp.types import Direction, EntryType, ScalpSignal

START = "2026-08-03 09:30"          # Monday, ET-naive like the cached bars
WARM = "2026-08-03 09:20"           # pre-market bar for the RTH test


# ── helpers ────────────────────────────────────────────────────────────


def frame(rows, start: str = START) -> pd.DataFrame:
    """Build an OHLCV frame from ``(open, high, low, close)`` rows."""
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx).assign(volume=1000.0)


def flat(prices, spread: float = 0.05) -> list[tuple[float, float, float, float]]:
    return [(p, p + spread, p - spread, p) for p in prices]


def upto(i: int, start: str = START) -> pd.Timestamp:
    return pd.Timestamp(start) + pd.Timedelta(minutes=i)


def signal(sym="NVDA", ts=None, direction=Direction.LONG, entry_type=EntryType.MARKET,
           entry=100.0, sl=99.0, tp=105.0, strategy="scripted", rr=None, **kw) -> ScalpSignal:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return ScalpSignal(
        symbol=sym, timestamp=ts if ts is not None else upto(0), direction=direction,
        entry_type=entry_type, entry_price=entry, stop_loss=sl, take_profit=tp,
        risk=risk, reward=reward, rr=rr if rr is not None else round(reward / risk, 4),
        strategy=strategy, **kw)


class Scripted:
    """A fake module: emits pre-built signals when the bar time matches."""

    def __init__(self, name="scripted", at=None, builder=None):
        self.name = name
        self.at = at or {}
        self.builder = builder          # optional fn(ts) -> list[ScalpSignal]
        self.calls: list[pd.Timestamp] = []
        self.seen_last_ts: list[pd.Timestamp] = []
        self.future_seen = False

    def __call__(self, ctx):
        last = pd.Timestamp(ctx.frames["1m"].ts[-1])
        self.calls.append(last)
        self.seen_last_ts.append(last)
        if any(len(ctx.frames[k].ts) and pd.Timestamp(ctx.frames[k].ts[-1]) > last
               for k in ctx.frames if k != "1m"):
            self.future_seen = True
        out = list(self.at.get(last, ()))
        if self.builder is not None:
            out = self.builder(last)
        return out


def cfg(**over):
    """Engine config for a single scripted symbol."""
    base = dict(symbols=("NVDA",), enabled_modules=("scripted",),
                module_order=("scripted",), min_bars={},
                shortable=frozenset({"NVDA"}), initial_equity=100_000.0)
    base.update(over)
    return replace(E.ScalpSetConfig(), **base)


def run(bars, module, config=None, pair_symbol="QQQ", **kw):
    return E.run_bars(bars, config or cfg(), modules={"scripted": module},
                      pair_symbol=pair_symbol, **kw)


# ── RTH filter ─────────────────────────────────────────────────────────


def test_filter_rth_drops_extended_hours_naive_and_utc():
    naive = frame(flat([100, 101, 102]), start="2026-08-03 08:00")
    assert E.filter_rth(naive).empty                       # pre-market only

    utc = frame(flat([100, 101, 102]), start=START)        # 09:30 ET
    utc.index = utc.index.tz_localize("America/New_York").tz_convert("UTC")   # 13:30Z
    out = E.filter_rth(utc)
    assert list(out.index) == [pd.Timestamp("2026-08-03 09:30"),
                               pd.Timestamp("2026-08-03 09:31"),
                               pd.Timestamp("2026-08-03 09:32")]

    closing = frame(flat([100]), start="2026-08-03 16:00")
    assert E.filter_rth(closing).empty                     # 16:00 is exclusive


def test_engine_only_trades_rth_bars():
    bars = frame(flat([100, 101, 102, 103, 104, 105]), start="2026-08-03 09:25")
    sf = E.frames_from_bars({"NVDA": bars})["NVDA"]
    assert [pd.Timestamp(t) for t in sf.ts] == [pd.Timestamp(START)]   # 09:30 only
    res = run({"NVDA": bars}, Scripted())
    assert len(res.equity_curve) == 1
    assert res.equity_curve.index[0] == pd.Timestamp(START)

    premarket = frame(flat([100, 101]), start=WARM)                    # no RTH bars at all
    assert len(E.frames_from_bars({"NVDA": premarket})["NVDA"].ts) == 0


# ── fill timing / no lookahead ─────────────────────────────────────────


def test_market_entry_fills_at_next_bar_open_not_signal_close():
    rows = [(100.0, 100.5, 99.5, 100.0),      # 0: signal bar close = 100
            (101.0, 101.5, 100.5, 101.0),     # 1: fill at this OPEN
            (101.0, 106.0, 100.9, 105.5),     # 2: TP (105) inside range
            (105.0, 105.5, 104.5, 105.0)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=105.0)]})
    res = run({"NVDA": frame(rows)}, mod)

    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    expected_fill = E.normalize_price(101.0 + max(101.0 * 0.0002, 0.01))
    assert t["entry_time"] == upto(1)
    assert t["entry_price"] == pytest.approx(expected_fill)
    assert t["exit_reason"] == "tp"
    assert t["exit_time"] == upto(2)
    assert t["exit_price"] == pytest.approx(105.0)
    # sized on the SIGNAL bar close (live uses the price it fetched), whole shares
    assert t["quantity"] == 150.0            # floor(15_000 / 100)


def test_module_never_sees_data_beyond_the_current_bar():
    rows = flat([100 + 0.1 * i for i in range(12)])
    quiet = Scripted()                      # no signals → evaluated every bar
    res = run({"NVDA": frame(rows)}, quiet)
    assert not quiet.future_seen
    assert quiet.calls == [upto(i) for i in range(12)]
    assert quiet.seen_last_ts[-1] == upto(11)   # 1m frame ends exactly on each bar
    assert len(res.trades) == 0

    # ...and with an open position the module is not polled at all (live: the
    # per-symbol loop skips symbols that already hold a position)
    armed = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    run({"NVDA": frame(rows)}, armed)
    assert armed.calls == [upto(i) for i in range(1)]      # bar 0 only
    assert not armed.future_seen


def test_market_entry_never_uses_the_signal_bar_range():
    # A stop that would trigger on the signal bar's own low must not fire
    # before the fill — the position does not exist yet.
    rows = [(100.0, 100.2, 98.0, 100.0),
            (100.0, 100.5, 99.8, 100.2),
            (100.2, 100.4, 99.9, 100.1),
            (100.1, 100.3, 100.0, 100.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.5, tp=110.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "eod"      # never stopped out


# ── anchoring (post PR #37) ────────────────────────────────────────────


def test_anchor_levels_keeps_valid_strategy_levels():
    lvl = E.anchor_levels(is_short=False, fill=100.0, entry_ref=100.0, sl_ref=99.0, tp_ref=105.0)
    assert (lvl.sl, lvl.sl_source) == (99.0, "strategy")
    assert (lvl.tp, lvl.tp_source) == (105.0, "strategy")


def test_anchor_levels_reanchors_wrong_side_sl_to_fill_risk():
    """The live COIN case: fill far below the signal reference."""
    lvl = E.anchor_levels(is_short=False, fill=90.0, entry_ref=100.0, sl_ref=99.0, tp_ref=105.0)
    assert lvl.sl is not None and lvl.sl < 90.0
    assert lvl.sl_source == "fill_risk"
    assert lvl.sl == pytest.approx(89.0)                    # 90 - signal risk (1.0)
    assert (lvl.tp, lvl.tp_source) == (105.0, "strategy")


def test_anchor_levels_clamps_tiny_risk_to_min_distance():
    lvl = E.anchor_levels(is_short=False, fill=100.0, entry_ref=100.0, sl_ref=99.999, tp_ref=100.001)
    assert lvl.min_distance == pytest.approx(0.05)          # max(0.05%, 1ct)
    assert lvl.sl_source == "fill_risk_clamped"
    assert lvl.sl == pytest.approx(99.95)
    assert lvl.tp_source == "fill_reward_clamped"
    assert lvl.tp == pytest.approx(100.05)


def test_anchor_levels_backstop_without_derivable_sl_and_dropped_tp():
    long_no_sl = E.anchor_levels(is_short=False, fill=100.0, entry_ref=None, sl_ref=None, tp_ref=None)
    assert (long_no_sl.sl, long_no_sl.sl_source) == (94.0, "no_strategy_sl")
    assert (long_no_sl.tp, long_no_sl.tp_source) == (None, "none")

    short_degenerate = E.anchor_levels(is_short=True, fill=100.0, entry_ref=None, sl_ref=99.0,
                                       tp_ref=105.0)
    assert (short_degenerate.sl, short_degenerate.sl_source) == (106.0, "backstop")  # +6%
    assert short_degenerate.tp is None                      # no reward leg derivable


def test_anchor_levels_short_side_reward_leg_and_backstop_sign():
    lvl = E.anchor_levels(is_short=True, fill=100.0, entry_ref=100.0, sl_ref=101.0, tp_ref=95.0)
    assert (lvl.sl, lvl.sl_source) == (101.0, "strategy")
    assert (lvl.tp, lvl.tp_source) == (95.0, "strategy")
    re_anchored = E.anchor_levels(is_short=True, fill=110.0, entry_ref=100.0, sl_ref=101.0,
                                  tp_ref=95.0)
    assert re_anchored.sl_source == "fill_risk" and re_anchored.sl == pytest.approx(111.0)
    # the TP is still on the correct side of the new fill → kept verbatim
    assert re_anchored.tp_source == "strategy"
    assert re_anchored.tp == pytest.approx(95.0)


def test_anchor_levels_short_side_rewards_are_repriced_when_needed():
    lvl = E.anchor_levels(is_short=True, fill=94.0, entry_ref=100.0, sl_ref=101.0, tp_ref=95.0)
    assert lvl.sl_source == "strategy" and lvl.sl == pytest.approx(101.0)
    assert lvl.tp_source == "fill_reward" and lvl.tp == pytest.approx(89.0)   # 94 - reward(5)
    stop = E.anchor_levels(is_short=True, fill=104.0, entry_ref=100.0, sl_ref=101.0, tp_ref=95.0)
    assert stop.sl_source == "fill_risk" and stop.sl == pytest.approx(105.0)  # 104 + risk(1)


def test_engine_reanchors_stale_signal_sl_below_the_fill():
    """A gapped-down fill must not leave the stale stop (above the market) in use."""
    rows = [(100.0, 100.5, 99.5, 100.0),      # signal bar (close 100), stale SL 99
            (90.0, 92.0, 89.0, 91.0),         # gap-down open → fill ~90.02
            (91.0, 91.5, 90.5, 91.0),
            (91.0, 91.5, 90.8, 91.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=105.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["entry_price"] < 91.0
    assert t["sl_source"] == "fill_risk"
    assert t["sl"] == pytest.approx(t["entry_price"] - 1.0)   # signal risk, from the fill
    assert t["sl"] < t["entry_price"]                        # never wrong-side
    # the anchored stop is what fills — not the stale 99.0, and not an
    # instant wrong-side "breach" (the live COIN 2026-09-16 failure mode)
    assert t["exit_reason"] == "sl"
    assert t["exit_price"] < 90.0
    assert t["pnl_after_costs"] == pytest.approx(-1.0 * t["quantity"], rel=0.05)


def test_engine_never_leaves_a_tp_on_the_wrong_side_of_the_fill():
    """A gapped-up fill re-prices the reward leg above the fill."""
    rows = [(100.0, 100.2, 99.8, 100.0),      # signal bar: entry_ref 100, TP 100.2
            (102.0, 102.5, 101.9, 102.2),     # gap-up fill ≈ 102.02
            (102.2, 103.0, 102.0, 102.9),     # high crosses the re-priced TP
            (102.9, 103.0, 102.8, 102.9)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.9, tp=100.2)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["tp"] > t["entry_price"]
    assert t["tp_source"] == "fill_reward"          # 102.02 + signal reward 0.2
    assert t["tp"] == pytest.approx(102.22)
    assert t["exit_reason"] == "tp"
    assert t["exit_price"] == pytest.approx(t["tp"])


def test_anchor_levels_parity_with_live_implementation():
    """The replay's anchoring must equal live ``anchor_scalp_levels``."""
    live_trader = pytest.importorskip("live_trader")
    cases = [
        dict(is_short=False, fill=100.0, entry_ref=100.0, sl_ref=99.0, tp_ref=105.0),
        dict(is_short=False, fill=90.0, entry_ref=100.0, sl_ref=99.0, tp_ref=105.0),   # COIN case
        dict(is_short=True, fill=110.0, entry_ref=100.0, sl_ref=101.0, tp_ref=95.0),
        dict(is_short=True, fill=100.0, entry_ref=100.0, sl_ref=101.0, tp_ref=95.0),
        dict(is_short=False, fill=100.0, entry_ref=100.0, sl_ref=99.999, tp_ref=100.001),
        dict(is_short=False, fill=100.0, entry_ref=100.0, sl_ref=None, tp_ref=105.0),
        dict(is_short=True, fill=100.0, entry_ref=100.0, sl_ref=100.001, tp_ref=99.999),
        dict(is_short=False, fill=100.0, entry_ref=None, sl_ref=99.0, tp_ref=None),
    ]
    for case in cases:
        mine = E.anchor_levels(**case, min_dist_pct=0.0005, min_dist_abs=0.01, backstop_pct=0.06)
        live = live_trader.anchor_scalp_levels(**case, min_distance_pct=0.0005,
                                               min_distance_abs=0.01, backstop_pct=0.06)
        assert (mine.sl, mine.tp, mine.sl_source, mine.tp_source) == (
            live.sl, live.tp, live.sl_source, live.tp_source), case
        assert mine.min_distance == pytest.approx(live.min_distance)


# ── LIMIT entries ──────────────────────────────────────────────────────


def test_limit_entry_fills_when_a_later_bar_trades_through():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.5, 99.8),      # trades to 99.5 → limit 99.6 fills
            (99.8, 100.4, 99.7, 100.2),
            (100.2, 100.6, 100.0, 100.4)]
    mod = Scripted(at={upto(0): [signal(entry_type=EntryType.LIMIT, entry=99.6, sl=98.6,
                                   tp=102.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(99.6)           # at the limit, no slippage
    assert t["entry_time"] == upto(1)
    assert res.stats["limit_orders_placed"] == 1
    assert res.stats["limit_orders_filled"] == 1


def test_limit_entry_gapped_past_fills_at_open():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (98.0, 98.5, 97.5, 98.2),        # opens below the 99.6 buy limit
            (98.2, 98.6, 98.0, 98.4),
            (98.4, 98.8, 98.2, 98.6)]
    mod = Scripted(at={upto(0): [signal(entry_type=EntryType.LIMIT, entry=99.6, sl=98.6,
                                   tp=102.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    assert res.trades.iloc[0]["entry_price"] == pytest.approx(98.0)


def test_unfilled_limit_expires_at_session_end():
    rows = flat([100.0] * 5)
    mod = Scripted(at={upto(0): [signal(entry_type=EntryType.LIMIT, entry=95.0, sl=94.0,
                                   tp=100.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    assert res.trades.empty
    assert res.stats["limit_orders_expired"] == 1
    assert res.stats["unfilled_limits_at_end"] == 0


# ── exits ──────────────────────────────────────────────────────────────


def test_stop_exit_uses_bar_range_with_adverse_slippage():
    rows = [(100.0, 100.2, 99.9, 100.0),
            (100.0, 100.3, 99.8, 100.1),
            (100.1, 100.2, 98.4, 98.6)]      # low trades through the anchored stop
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "sl"
    assert t["exit_time"] == upto(2)
    assert t["exit_price"] == pytest.approx(E.normalize_price(99.0 - max(99.0 * 0.0002, 0.01)))
    assert t["pnl_after_costs"] < 0


def test_stop_gap_through_level_fills_at_open():
    rows = [(100.0, 100.2, 99.9, 100.0),
            (100.0, 100.3, 99.8, 100.1),
            (96.0, 96.5, 95.0, 96.0)]        # gaps far below the 99 stop
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(
        E.normalize_price(96.0 - max(96.0 * 0.0002, 0.01)))


def test_short_side_stop_and_tp_ranges():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.1, 99.9, 100.0),
            (100.2, 101.5, 100.0, 101.2)]    # high crosses the 101 short stop
    mod = Scripted(at={upto(0): [signal(direction=Direction.SHORT, entry=100.0, sl=101.0,
                                        tp=95.0, strategy="scripted")]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["side"] == "SHORT" and t["exit_reason"] == "sl"
    assert t["exit_price"] > 101.0           # adverse slippage on the cover


def test_tp_exit_fills_at_the_limit_level():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.8, 100.0),
            (100.1, 106.0, 100.0, 105.0)]    # high crosses the 104.5 TP
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.5)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "tp"
    assert t["exit_price"] == pytest.approx(104.5)
    assert t["quantity"] == 150.0


def test_ambiguous_bar_both_levels_inside_resolves_to_stop_first():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 105.0, 98.0, 104.0)]     # range covers SL and TP
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    assert res.trades.iloc[0]["exit_reason"] == "sl"


def test_eod_flat_uses_final_bar_close_with_slippage():
    rows = [(100.0, 100.2, 99.9, 100.0),
            (100.0, 100.4, 99.8, 100.3),
            (100.3, 100.6, 100.1, 100.5),
            (100.5, 100.9, 100.4, 100.8)]     # last bar of the window
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "eod"
    assert t["exit_time"] == upto(3)
    assert t["exit_price"] == pytest.approx(E.normalize_price(100.8 - max(100.8 * 0.0002, 0.01)))
    assert res.stats["exits"] == {"eod": 1}
    assert res.stats["open_positions_at_end"] == 0


def test_eod_flat_disabled_holds_overnight_across_the_window_edge():
    rows = flat([100.0, 100.2, 100.4])
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    res = run({"NVDA": frame(rows)}, mod, config=cfg(eod_flat=False))
    assert res.trades.empty
    assert res.stats["open_positions_at_end"] == 1


def test_break_even_stop_stays_at_the_fill_live_quirk():
    """After BE the live risk pass has no risk left → no trailing (faithful).

    ``_scalp_risk_pass`` recomputes ``risk = abs(entry - state["sl"])`` on
    every tick and ``_replace_stop`` overwrites ``state["sl"]`` with the new
    level, so once the stop sits on the fill price the R maths degenerates
    and live never trails afterwards.  The replay reproduces that instead of
    inventing better-than-live behaviour.
    """
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 102.5, 100.0, 102.2),    # +1R → stop to the fill price
            (102.2, 102.4, 101.0, 101.2),    # ...and it stays there
            (101.2, 101.3, 100.0, 100.2)]    # back to the fill → stopped
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0, trailing=True,
                                    trail_distance_r=1.0, trail_trigger_r=1.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "sl"
    assert t["sl_source"] == "breakeven"
    assert t["sl"] == pytest.approx(t["entry_price"])
    assert t["exit_price"] == pytest.approx(
        E.normalize_price(t["entry_price"] - max(t["entry_price"] * 0.0002, 0.01)))
    assert t["pnl_after_costs"] == pytest.approx(-0.02 * t["quantity"], rel=0.15)


def test_trailing_stop_locks_in_profit_when_break_even_is_out_of_the_way():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 102.5, 100.0, 102.2),    # +2R → trail puts the stop at ~101.2
            (102.2, 102.4, 101.0, 101.2),    # dips through the trailed stop
            (101.2, 101.3, 101.1, 101.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0, trailing=True,
                                    trail_distance_r=1.0, trail_trigger_r=1.0,
                                    breakeven_trigger_r=99.0)]})   # BE never fires
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "sl"
    assert t["sl_source"] == "trail"
    assert t["sl"] > t["entry_price"]                       # already locked in
    assert t["exit_price"] > t["entry_price"]               # profitable exit
    assert t["pnl_after_costs"] > 0


def test_break_even_and_trail_can_be_disabled():
    rows = [(100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 102.5, 100.0, 102.2),
            (102.2, 102.4, 101.0, 101.2),
            (101.2, 101.3, 101.1, 101.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0, trailing=True)]})
    res = run({"NVDA": frame(rows)}, mod, config=cfg(be_trail=False))
    t = res.trades.iloc[0]
    assert t["sl_source"] == "strategy"          # the stop was never moved
    assert t["exit_reason"] == "eod"


# ── gating ─────────────────────────────────────────────────────────────


def episode_bars(k: int) -> list[tuple[float, float, float, float]]:
    """One scalp episode: signal bar, fill bar, stop-out bar, 4 cooldown bars."""
    base = 100.0 + k
    return (flat([base]) + flat([base]) + [(base, base + 0.1, base - 3.0, base - 2.9)]
            + flat([base - 2.9] * 4))


def test_entry_cap_three_per_symbol_per_session():
    rows: list = []
    signals = {}
    for k in range(6):                      # 6 episodes → 3 legal entries
        signals[upto(len(rows))] = [signal(entry=100.0 + k, sl=99.0 + k, tp=105.0 + k)]
        rows += episode_bars(k)
    res = run({"NVDA": frame(rows)}, Scripted(at=signals))
    # the cap is checked BEFORE the modules run (live tick-loop order), so it
    # counts refused bars once exhausted — what matters is 3 entries, not more
    assert res.stats["entries"] == 3
    assert res.stats["exits"]["sl"] == 3
    assert res.stats["skipped"]["entry_cap"] >= 1
    assert res.stats["signals_by_module"]["scripted"] == 3


def test_cooldown_blocks_entries_for_five_minutes_after_a_close():
    rows = flat([100.0] * 12)
    rows = [(100.0, 100.2, 99.9, 100.0),      # 0 signal
            (100.0, 100.4, 99.9, 100.2),      # 1 entry fills at 100.02
            (100.1, 100.2, 98.5, 98.6)] + flat([98.0] * 9)   # 2 stopped out
    signals = {}
    for k in (0, 2, 3, 4, 5):
        # distinct keys (varying TP) so dedupe cannot interfere
        signals[upto(k)] = [signal(entry=100.0, sl=99.0, tp=105.0 + k, rr=2.0 + k)]
    res = run({"NVDA": frame(rows)}, Scripted(at=signals))
    # bar 0 → entry fills at bar 1, stopped out at bar 2 → cooldown until bar 7
    assert res.stats["exits"]["sl"] == 1
    assert res.stats["skipped"]["cooldown"] == 5      # bars 2,3,4,5,6 refused
    assert res.stats["entries"] == 1


def test_cooldown_expires_after_five_minutes():
    rows = [(100.0, 100.2, 99.9, 100.0),
            (100.0, 100.4, 99.9, 100.2),
            (100.1, 100.2, 98.5, 98.6)] + flat([98.0] * 9)
    signals = {upto(0): [signal(entry=100.0, sl=99.0, tp=120.0)],
               upto(7): [signal(entry=110.0, sl=109.0, tp=120.0)]}   # 5 bars after the exit
    res = run({"NVDA": frame(rows)}, Scripted(at=signals))
    assert res.stats["entries"] == 2                     # the bar-7 signal got through
    assert res.stats["skipped"]["cooldown"] == 5         # bars 2..6 were refused


def test_short_gating_blocks_non_shortable_symbols_and_counts_them():
    rows = flat([100.0] * 6)
    short = signal(sym="META", direction=Direction.SHORT, entry=100.0, sl=101.0, tp=95.0)
    mod = Scripted(at={upto(0): [short]})
    res = run({"META": frame(rows)}, mod,
              config=replace(cfg(), symbols=("META",), shortable=frozenset({"NVDA"})))
    assert res.trades.empty
    assert res.stats["skipped"]["short_not_shortable"] == 1
    assert res.stats["entries"] == 0


def test_short_gating_allows_shortable_symbols():
    rows = [(100.0, 100.1, 99.9, 100.0), (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 95.5, 96.0), (96.0, 96.2, 95.0, 95.5)]
    mod = Scripted(at={upto(0): [signal(direction=Direction.SHORT, entry=100.0, sl=101.0,
                                        tp=95.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["side"] == "SHORT"
    assert t["quantity"] == 150.0            # whole shares (floor)
    assert t["exit_reason"] in ("tp", "eod")


def test_shortability_set_is_configurable():
    rows = flat([100.0] * 6)
    mod = Scripted(at={upto(0): [signal(sym="META", direction=Direction.SHORT, entry=100.0,
                                        sl=101.0, tp=95.0)]})
    res = run({"META": frame(rows)}, mod,
              config=replace(cfg(), symbols=("META",), shortable=frozenset({"META"})))
    assert res.stats["entries"] == 1
    assert res.stats["skipped"].get("short_not_shortable", 0) == 0


def test_max_positions_caps_concurrent_exposure():
    rows = flat([100.0] * 6)
    mod = Scripted(at={upto(0): [signal(sym="NVDA", entry=100.0, sl=99.0, tp=110.0)]})
    bars = {"NVDA": frame(rows), "QQQ": frame(rows)}
    res = E.run_bars(bars, replace(cfg(), symbols=("NVDA", "QQQ"), max_positions=1),
                     modules={"scripted": mod}, pair_symbol="ZZZ")
    assert res.stats["entries"] == 1                       # only one slot existed
    assert res.stats["skipped"]["max_positions"] >= 1      # the other symbol refused


def test_signal_for_another_symbol_is_rejected():
    rows = flat([100.0] * 5)
    mod = Scripted(at={upto(0): [signal(sym="TSLA", entry=100.0, sl=99.0, tp=105.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    assert res.trades.empty
    assert res.stats["skipped"]["symbol_mismatch"] == 1


def test_whole_share_sizing_floors_and_skips_sub_share_orders():
    rows = flat([100.0] * 6)
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    res = run({"NVDA": frame(rows)}, mod, config=cfg(initial_equity=1_234.0))
    assert res.trades.iloc[0]["quantity"] == 1.0             # floor(185.1 / 100)

    res2 = run({"NVDA": frame(rows)}, mod, config=cfg(initial_equity=500.0))
    assert res2.trades.empty                                 # floor(75/100) == 0
    assert res2.stats["skipped"]["qty_lt_1"] == 1


def test_duplicate_signal_key_is_skipped():
    rows = flat([100.0] * 4)
    dup = signal(entry=100.0, sl=99.0, tp=105.0)
    engine = E.ScalpSetEngine(E.frames_from_bars({"NVDA": frame(rows)}), cfg(),
                              modules={"scripted": Scripted(at={upto(0): [dup]})})
    engine._stats = engine._new_stats(["NVDA"], 4)
    engine._last_signal_key["NVDA"] = engine._signal_key(dup)
    engine._evaluate_and_place("NVDA", 0, upto(0))
    assert engine._stats["skipped"]["duplicate_signal"] == 1
    assert "NVDA" not in engine._pending_market


def test_arbitration_prefers_highest_rr_then_module_order():
    engine = E.ScalpSetEngine({}, cfg(enabled_modules=("ict_ifvg", "box_theory"),
                                      module_order=("ict_ifvg", "box_theory"), min_bars={}))
    low_rr = signal(entry=100.0, sl=99.0, tp=102.0, strategy="ict_ifvg")     # 2R
    high_rr = signal(entry=100.0, sl=99.0, tp=104.0, strategy="box_theory")  # 4R
    assert engine._arbitrate([low_rr, high_rr]) is high_rr
    tie_other = signal(entry=100.0, sl=99.0, tp=102.0, strategy="box_theory")
    assert engine._arbitrate([tie_other, low_rr]) is low_rr                  # IFVG wins the tie
    assert engine._arbitrate([]) is None
    assert engine._arbitrate([low_rr]) is low_rr


def test_insufficient_data_gate_blocks_signals_until_min_bars_met():
    rows = flat([100.0] * 6)
    mod = Scripted(at={upto(0): [signal()]})
    config = replace(cfg(), min_bars={"scripted": {"5m": 60}})   # never satisfied
    res = run({"NVDA": frame(rows)}, mod, config=config)
    assert res.trades.empty
    assert res.stats["skipped"]["insufficient_data"] >= 1


def test_config_symbols_absent_from_frames_are_ignored():
    rows = flat([100.0] * 4)
    config = replace(E.ScalpSetConfig(), symbols=("NVDA", "META"), enabled_modules=("scripted",),
                     module_order=("scripted",), min_bars={})
    res = E.run_bars({"NVDA": frame(rows)}, config, modules={"scripted": Scripted()},
                     pair_symbol="ZZZ")
    assert res.stats["symbols"] == ["NVDA"]


def test_engine_rejects_an_empty_universe():
    with pytest.raises(ValueError):
        E.ScalpSetEngine({}, cfg()).run()


# ── outputs / metrics / costs ──────────────────────────────────────────


def test_trades_and_equity_curve_feed_compute_metrics():
    rows = [(100.0, 100.1, 99.9, 100.0), (100.0, 100.2, 99.8, 100.1),
            (100.1, 106.0, 100.0, 105.0), (105.0, 105.2, 104.8, 105.0),
            (105.0, 105.5, 104.9, 105.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.5)]})
    res = run({"NVDA": frame(rows)}, mod)

    assert list(res.trades.columns) == list(E.TRADE_COLUMNS)
    m = res.metrics()
    for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "total_return",
                "win_rate", "profit_factor", "num_trades"):
        assert key in m
    assert m["num_trades"] == 1
    assert m["win_rate"] == 1.0
    assert res.stats["final_equity"] == pytest.approx(
        res.config.initial_equity + res.trades["pnl_after_costs"].sum())


def test_costs_reduce_pnl_and_pnl_column_is_net():
    rows = [(100.0, 100.1, 99.9, 100.0), (100.0, 100.2, 99.8, 100.1),
            (100.1, 106.0, 100.0, 105.0), (105.0, 105.2, 104.8, 105.0),
            (105.0, 105.5, 104.9, 105.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.5)]})
    free = run({"NVDA": frame(rows)}, mod)
    costly = run({"NVDA": frame(rows)}, mod, config=E.ScalpSetConfig.cost_variant(
        symbols=("NVDA",), enabled_modules=("scripted",), module_order=("scripted",),
        min_bars={}))
    t = costly.trades.iloc[0]
    assert t["fees"] > 0
    assert t["pnl"] == t["pnl_after_costs"] == pytest.approx(t["pnl_gross"] - t["fees"])
    assert costly.trades["pnl_after_costs"].sum() < free.trades["pnl_after_costs"].sum()
    assert costly.stats["fees_paid"] == pytest.approx(t["fees"])


def test_slippage_is_adverse_on_both_legs():
    rows = [(100.0, 100.1, 99.9, 100.0), (100.0, 100.2, 99.8, 100.1),
            (100.1, 106.0, 100.0, 105.0), (105.0, 105.2, 104.8, 105.0),
            (105.0, 105.5, 104.9, 105.2)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=104.5)]})
    res = run({"NVDA": frame(rows)}, mod)
    t = res.trades.iloc[0]
    assert t["entry_price"] > 100.0
    # TP is a limit → fills exactly at the level (no slippage through it)
    assert t["exit_price"] == pytest.approx(104.5)
    assert t["pnl_gross"] == pytest.approx((104.5 - t["entry_price"]) * t["quantity"])


def test_equity_curve_is_marked_to_market_during_the_hold():
    rows = [(100.0, 100.1, 99.9, 100.0), (100.0, 100.2, 99.8, 100.1),
            (100.1, 103.0, 100.0, 102.5), (102.5, 103.5, 102.4, 103.0),
            (103.0, 103.2, 102.9, 103.1)]
    mod = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    res = run({"NVDA": frame(rows)}, mod)
    qty = res.trades.iloc[0]["quantity"]
    entry = res.trades.iloc[0]["entry_price"]
    assert res.equity_curve.loc[upto(2)] == pytest.approx(100_000 + (102.5 - entry) * qty)
    assert res.equity_curve.iloc[-1] == pytest.approx(res.stats["final_equity"])


def test_run_bars_requires_no_network_and_is_deterministic():
    rows = flat([100.0 + 0.05 * i for i in range(20)])
    mod_a = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    mod_b = Scripted(at={upto(0): [signal(entry=100.0, sl=99.0, tp=110.0)]})
    res_a = run({"NVDA": frame(rows)}, mod_a)
    res_b = run({"NVDA": frame(rows)}, mod_b)
    pd.testing.assert_frame_equal(res_a.trades, res_b.trades)
    pd.testing.assert_series_equal(res_a.equity_curve, res_b.equity_curve)
