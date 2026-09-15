"""Unit tests for the Daily Range Box Theory model (scalp_box_theory.py).

Every geometry test computes the exact expected SL/TP from the module's own
documented arithmetic — no randomness anywhere.
"""
import pytest

from src.strategies.scalp.scalp_box_theory import Box, classify_zone, evaluate
from src.strategies.scalp.types import Direction, EntryType, LiquidityMap, ScalpContext

PDH, PDL = 110.0, 100.0
ZONE = 0.10  # zone_width -> zone_top=109.0, zone_bottom=101.0
BUF = 0.01  # sl_buffer


def _ctx(frame, pdh=PDH, pdl=PDL, **kw):
    return ScalpContext(
        symbol="TEST", frames={"5m": frame},
        liquidity=LiquidityMap(pdh=pdh, pdl=pdl, **kw),
    )


class TestBoxBuilding:
    def test_box_zones_and_classification(self):
        box = Box.build(PDH, PDL, ZONE)
        assert box.zone_top == pytest.approx(109.0)
        assert box.zone_bottom == pytest.approx(101.0)
        assert box.range == pytest.approx(10.0)
        assert classify_zone(109.0, box) == "A"  # threshold inclusive
        assert classify_zone(109.5, box) == "A"
        assert classify_zone(108.99, box) == "C"
        assert classify_zone(105.0, box) == "C"
        assert classify_zone(101.0, box) == "B"
        assert classify_zone(99.5, box) == "B"

    def test_custom_zone_width(self):
        box = Box.build(PDH, PDL, 0.05)
        assert box.zone_top == pytest.approx(109.5)
        assert box.zone_bottom == pytest.approx(100.5)


class TestShortSide:
    def test_box_short_confirmation_signal_exact_geometry(self, make_frame):
        ts = ["2026-09-10 09:30", "2026-09-10 09:35", "2026-09-10 09:40"]
        f5 = make_frame(
            "TEST", "5m", ts,
            [107.0, 108.0, 109.3],  # opens
            [107.8, 109.3, 109.6],  # highs: bar2 high 109.6 >= 109.0 zone A
            [106.8, 107.9, 108.6],  # lows
            [107.2, 108.9, 108.7],  # closes: bar2 red (108.7 < 108.9)
            [1000, 1000, 1000],
        )
        sigs = evaluate(_ctx(f5))
        assert len(sigs) == 1
        s = sigs[0]
        assert s.direction == Direction.SHORT
        assert s.entry_type == EntryType.MARKET
        assert s.entry_price == pytest.approx(108.7)
        assert s.stop_loss == pytest.approx(max(109.6, 109.3) + BUF)  # above prior/trigger high + buf
        assert s.take_profit == pytest.approx(PDL)  # opposite box boundary
        assert s.risk == pytest.approx(109.61 - 108.7)
        assert s.rr == pytest.approx((108.7 - PDL) / (109.61 - 108.7), abs=5e-5)  # module rounds rr to 4 dp

    def test_box_short_requires_red_confirmation_close(self, make_frame):
        f5 = make_frame(
            "TEST", "5m",
            ["2026-09-10 09:30", "2026-09-10 09:35", "2026-09-10 09:40"],
            [107.0, 107.2, 109.2],
            [107.8, 107.9, 109.5],  # high in zone A
            [106.8, 106.9, 109.0],
            [107.2, 107.8, 109.3],  # close 109.3 > 107.8 -> green close -> no short
            [1000, 1000, 1000],
        )
        assert evaluate(_ctx(f5)) == []

    def test_box_short_requires_zone_a_touch(self, make_frame):
        f5 = make_frame(
            "TEST", "5m",
            ["2026-09-10 09:30", "2026-09-10 09:35", "2026-09-10 09:40"],
            [108.4, 108.6, 108.9],
            [108.8, 108.9, 108.95],  # never >= 109.0
            [108.3, 108.5, 108.7],
            [108.55, 108.9, 108.8],  # red close but no zone A touch
            [1000, 1000, 1000],
        )
        assert evaluate(_ctx(f5)) == []

    def test_box_short_cooldown_blocks_repeat_touches_without_zone_c_close(self, make_frame):
        # bar2 fires. bars 3-4 touch zone A with red closes but close stays in
        # zone A (109.3 / 109.0 >= 109.0) -> boundary locked -> exactly 1 signal.
        f5 = make_frame(
            "TEST", "5m",
            ["2026-09-10 09:30", "2026-09-10 09:35", "2026-09-10 09:40",
             "2026-09-10 09:45", "2026-09-10 09:50", "2026-09-10 09:55"],
            [107.0, 108.0, 109.3, 109.4, 109.3, 106.0],
            [107.8, 109.3, 109.6, 109.6, 109.5, 106.2],
            [106.8, 107.9, 108.6, 109.2, 109.0, 105.6],
            [107.2, 108.9, 108.7, 109.3, 109.0, 105.9],
            [1000] * 6,
        )
        sigs = evaluate(_ctx(f5))
        assert len(sigs) == 1
        assert sigs[0].entry_price == pytest.approx(108.7)

    def test_box_short_rearms_after_zone_c_close(self, make_frame):
        # bar2 fires (red close in A). bar3 touches A but closes 109.2 in A -> locked
        # (no C close -> no re-arm). bar4 closes 105.8 in C -> re-arms. bar5 touches A
        # (high 109.4) with red close 105.9 < 105.9 — green; bar5 does NOT fire.
        # To get a fire after re-arm we need a red close after the C close: bar5
        # high 109.4, close 105.6 < close[4]=105.8 -> fires. 2 signals total... but
        # bar3 close 109.2 in A keeps lock until bar4 C close. bar5 then fires.
        f5 = make_frame(
            "TEST", "5m",
            ["2026-09-10 09:30", "2026-09-10 09:35", "2026-09-10 09:40",
             "2026-09-10 09:45", "2026-09-10 09:50", "2026-09-10 09:55"],
            [107.0, 108.0, 109.3, 109.4, 106.0, 109.0],
            [107.8, 109.3, 109.6, 109.7, 106.2, 109.4],
            [106.8, 107.9, 108.6, 109.2, 105.5, 105.3],
            [107.2, 108.9, 108.7, 109.2, 105.8, 105.6],
            [1000] * 6,
        )
        sigs = evaluate(_ctx(f5))
        assert len(sigs) == 2, f"expected 2 signals, got {len(sigs)}"
        # bar2 then bar5 (high 109.4 >= 109.0, close 105.6 < 105.8 after C close 105.8)
        assert sigs[1].entry_price == pytest.approx(105.6)
        assert sigs[1].stop_loss == pytest.approx(max(109.4, 106.2) + BUF)
        assert sigs[1].take_profit == pytest.approx(PDL)
        assert sigs[1].direction == Direction.SHORT


class TestLongSide:
    def test_box_long_confirmation_signal_exact_geometry(self, make_frame):
        f5 = make_frame(
            "TEST", "5m",
            ["2026-09-10 09:30", "2026-09-10 09:35"],
            [102.5, 102.0],
            [103.0, 102.6],
            [102.1, 100.4],  # low 100.4 <= 101.0 zone B
            [101.8, 102.2],  # green close (102.2 > 101.8)
            [1000, 1000],
        )
        sigs = evaluate(_ctx(f5))
        assert len(sigs) == 1
        s = sigs[0]
        assert s.direction == Direction.LONG
        assert s.entry_type == EntryType.MARKET
        assert s.entry_price == pytest.approx(102.2)
        assert s.stop_loss == pytest.approx(min(100.4, 102.1) - BUF)  # below prior/trigger low - buf
        assert s.take_profit == pytest.approx(PDH)  # opposite box boundary
        assert s.risk == pytest.approx(102.2 - (100.4 - BUF))
        assert s.rr == pytest.approx((PDH - 102.2) / (102.2 - (100.4 - BUF)), abs=5e-5)
        assert s.metadata["trigger_idx"] == 1


class TestNoSignalCases:
    def test_box_do_nothing_in_zone_c(self, make_frame):
        ts = [f"2026-09-10 09:{m:02d}" for m in range(30, 60)]
        f5 = make_frame("TEST", "5m", ts,
                       [105.0] * 30, [105.5] * 30, [104.5] * 30,
                       [105.2 if i % 2 else 104.9 for i in range(30)][:30], [1000] * 30)
        assert evaluate(_ctx(f5)) == []

    def test_box_missing_pdh_pdl_returns_empty(self, make_frame):
        f5 = make_frame("TEST", "5m", ["2026-09-10 09:30"],
                        [1.0], [2.5], [0.5], [1.2], [1000.0])
        assert evaluate(_ctx(f5, pdh=None, pdl=None)) == []

    def test_box_zero_range_returns_empty(self, make_frame):
        f5 = make_frame("TEST", "5m", ["2026-09-10 09:30", "2026-09-10 09:35"],
                        [1.0, 1.0], [2.0, 2.0], [0.5, 0.5], [1.2, 0.8], [10.0, 10.0])
        assert evaluate(_ctx(f5, pdh=100.0, pdl=100.0)) == []

    def test_box_red_close_required_even_in_zone_b_for_long(self, make_frame):
        f5 = make_frame(
            "TEST", "5m",
            ["2026-09-10 09:30", "2026-09-10 09:35"],
            [102.0, 101.2], [102.6, 101.5], [99.8, 100.2],
            [101.6, 100.9],  # red close (100.9 < 101.6) despite zone B low -> no long
            [1000, 1000],
        )
        assert evaluate(_ctx(f5)) == []