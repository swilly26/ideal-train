"""Unit tests for the Volume Profile + Fibonacci model (scalp_volprofile_fib.py).

All expected values are hand-computed from the module's documented arithmetic.
"""
import pytest

from src.strategies.scalp.scalp_volprofile_fib import evaluate
from src.strategies.scalp.types import Direction, EntryType, LiquidityMap, ScalpContext


def _liq_long() -> LiquidityMap:
    # _long_sweep = min(pdl=100.0, asia_low=100.4, london_low=100.6) = 100.0
    return LiquidityMap(pdh=103.0, pdl=100.0, asia_low=100.4, london_low=100.6, asia_high=104.0, london_high=104.2)


def _liq_short() -> LiquidityMap:
    # _short_sweep = max(pdh=110.0, asia_high=109.8, london_high=109.5) = 110.0
    return LiquidityMap(pdh=110.0, pdl=100.0, asia_high=109.8, london_high=109.5,
                        london_low=109.0, asia_low=108.5)


def _day(make_frame, o, h, l, c, v):
    import pandas as pd

    ts = pd.date_range("2026-09-10 09:30", periods=len(o), freq="5min")
    ts = [str(t) for t in ts]
    return make_frame("TEST", "5m", ts, o, h, l, c, v)


class TestLongSetups:
    def test_long_sweep_confirm_exact_geometry(self, make_frame):
        # bar1 sweeps pdl 100.0 (low 99.8); bars 2-3 green; swing L=99.8 H=101.2
        f5 = _day(make_frame,
                  [101.2, 101.3, 100.0, 100.4, 100.8, 101.0, 101.1],
                  [101.6, 101.5, 100.6, 101.2, 101.4, 101.6, 101.5],
                  [101.0, 99.8, 99.9, 100.3, 100.6, 100.7, 100.8],
                  [101.4, 100.2, 100.5, 101.1, 101.2, 101.3, 101.0],
                  [500, 800, 1000, 1200, 600, 500, 400])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_long())
        sigs = evaluate(ctx)
        assert len(sigs) == 1, f"expected 1 signal, got {sigs}"
        s = sigs[0]
        assert s.direction == Direction.LONG
        assert s.entry_type == EntryType.LIMIT
        assert s.entry_price == pytest.approx(99.8 + 0.58 * 1.4)  # 100.612 — 0.58 of the swing
        assert s.stop_loss == pytest.approx(99.8 - 0.02)  # below swing low + buffer
        assert s.take_profit == pytest.approx(103.0)  # nearest opposite liquidity above
        assert s.metadata["swing_low"] == pytest.approx(99.8)
        assert s.metadata["swing_high"] == pytest.approx(101.2)
        assert s.metadata["vah"] == pytest.approx(100.2667, abs=1e-3)
        assert s.metadata["val"] == pytest.approx(100.7333, abs=1e-3)
        assert s.metadata["side"] == "val_above_0.50/0.58"

    def test_long_invalidated_when_intervening_candle_breaks_flow(self, make_frame):
        # bar2's close 99.85 breaks back through the sweep low 99.9 -> invalid
        f5 = _day(make_frame,
                  [101.2, 101.3, 99.8],
                  [101.6, 101.5, 100.1],
                  [101.0, 99.9, 99.75],
                  [101.4, 100.2, 99.85],  # green (99.85 > 99.8) but breaks 99.9
                  [500, 800, 1000])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_long())
        assert evaluate(ctx) == []

    def test_long_requires_exactly_two_consecutive_green(self, make_frame):
        # bar2 green, bar3 red -> no confirmation
        f5 = _day(make_frame,
                  [101.2, 101.3, 100.0, 100.6],
                  [101.6, 101.5, 100.6, 101.0],
                  [101.0, 99.8, 99.9, 100.2],
                  [101.4, 100.2, 100.5, 100.3],  # bar3 red close (100.3 < 100.6 open)
                  [500, 800, 1000, 1200])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_long())
        assert evaluate(ctx) == []

    def test_long_validation_requires_val_above_levels(self, make_frame):
        # overwhelming volume in the LOWER rows -> VAL falls below the 0.58 level
        f5 = _day(make_frame,
                  [101.2, 101.3, 100.0, 100.4],
                  [101.6, 101.5, 100.6, 101.2],
                  [101.0, 99.8, 99.9, 100.3],
                  [101.4, 100.2, 100.5, 101.1],
                  [100, 100, 3000, 100])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_long())
        assert evaluate(ctx) == []

    def test_long_sweep_must_be_at_open(self, make_frame):
        # no sweep before bar 13 (lows stay above pdl 100); the sweep at bar 13 is
        # beyond open_sweep_bars_5m=12 and also has no room for confirmations
        o = [100.0] * 14
        h = [100.6] * 14
        l = [100.05] * 13 + [99.4]
        c = [100.2] * 14
        v = [500] * 14
        f5 = _day(make_frame, o, h, l, c, v)
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_long())
        assert evaluate(ctx) == []


class TestShortSetups:
    def test_short_setup_exact_geometry(self, make_frame):
        # bar1 sweeps pdh 110.0 (high 111.2); bars 2-3 red; swing H=111.2 L=109.9
        f5 = _day(make_frame,
                  [110.2, 110.3, 111.0, 110.4, 110.0, 110.1],
                  [110.6, 111.2, 110.6, 110.5, 110.2, 110.3],
                  [110.1, 110.5, 110.1, 109.9, 109.8, 109.9],
                  [110.4, 111.0, 110.3, 110.0, 109.9, 110.0],
                  [500, 800, 1200, 1000, 600, 500])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_short())
        sigs = evaluate(ctx)
        assert len(sigs) == 1, f"expected 1 signal, got {sigs}"
        s = sigs[0]
        assert s.direction == Direction.SHORT
        assert s.entry_type == EntryType.LIMIT
        assert s.entry_price == pytest.approx(111.2 - 0.58 * 1.3)  # 110.446
        assert s.stop_loss == pytest.approx(111.2 + 0.02)  # above swing high + buffer
        assert s.take_profit == pytest.approx(109.0)  # nearest opposite liquidity below
        assert s.metadata["side"] == "vah_below_0.50/0.58"
        assert s.metadata["vah"] == pytest.approx(110.225, abs=1e-3)

    def test_short_invalidated_when_intervening_candle_breaks_flow(self, make_frame):
        # bar2 red but its close 111.3 closes back above the sweep high 111.2
        f5 = _day(make_frame,
                  [110.2, 110.3, 111.4],
                  [110.6, 111.2, 111.5],
                  [110.1, 110.5, 110.2],
                  [110.4, 111.0, 111.3],
                  [500, 800, 1200])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_short())
        assert evaluate(ctx) == []

    def test_short_validation_requires_vah_below_levels(self, make_frame):
        # volume concentrated in the TOP rows -> VAH sits above the 0.58 level
        f5 = _day(make_frame,
                  [110.2, 110.3, 111.0, 110.4],
                  [110.6, 111.2, 110.6, 110.5],
                  [110.1, 110.5, 110.1, 109.9],
                  [110.4, 111.0, 110.3, 110.0],
                  [3000, 100, 100, 100])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=_liq_short())
        assert evaluate(ctx) == []

    def test_short_tp_falls_back_when_structural_tp_tighter_than_1r(self, make_frame):
        liq = LiquidityMap(pdh=110.0, pdl=100.0, asia_high=109.8, london_high=109.5,
                           london_low=110.3, asia_low=110.35)  # TP candidate 110.35 -> rr < 1
        f5 = _day(make_frame,
                  [110.2, 110.3, 111.0, 110.4, 110.0],
                  [110.6, 111.2, 110.6, 110.5, 110.2],
                  [110.1, 110.5, 110.1, 109.9, 109.8],
                  [110.4, 111.0, 110.3, 110.0, 109.9],
                  [500, 800, 1200, 1000, 600])
        ctx = ScalpContext(symbol="TEST", frames={"5m": f5}, liquidity=liq)
        sigs = evaluate(ctx)
        assert len(sigs) == 1
        s = sigs[0]
        entry = 111.2 - 0.58 * 1.3
        risk = (111.2 + 0.02) - entry
        assert s.take_profit == pytest.approx(entry - risk * 2.0)  # 2R fallback
        assert s.metadata["vah"] < entry  # validation still passed


class TestNoData:
    def test_missing_5m_frame_returns_empty(self):
        ctx = ScalpContext(symbol="TEST", frames={}, liquidity=_liq_long())
        assert evaluate(ctx) == []

    def test_no_sweep_level_returns_empty(self, make_frame):
        f5 = _day(make_frame, [1.0] * 6, [2.0] * 6, [0.5] * 6, [1.2] * 6, [10.0] * 6)
        liq = LiquidityMap(pdh=None, pdl=None)
        assert evaluate(ScalpContext(symbol="T", frames={"5m": f5}, liquidity=liq)) == []