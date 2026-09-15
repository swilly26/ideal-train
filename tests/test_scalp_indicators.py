"""Unit tests for the scalp indicator primitives (src/strategies/scalp/indicators.py)."""
import numpy as np
import pytest

from src.strategies.scalp.indicators import (
    classify_amd,
    ema,
    equal_highs,
    equal_lows,
    fib_levels,
    fvg_gaps,
    last_fvg_in_window,
    session_extremes,
    smt_divergence,
    swing_pivots,
    swing_structure,
    volume_profile,
)
from src.strategies.scalp.types import SessionWindow


class TestEMA:
    def test_ema_constant_series_equals_constant(self):
        out = ema(np.full(10, 5.0), 3)
        assert out[2] == pytest.approx(5.0)
        assert out[-1] == pytest.approx(5.0)

    def test_ema_linear_ramp_exact_seed_and_last(self):
        # seed = SMA(first 3) = 2.0; alpha = 0.5
        out = ema(np.arange(1.0, 11.0), 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert out[2] == pytest.approx(2.0)
        assert out[-1] == pytest.approx(9.0)  # mirrored recursion: ema[i]=0.5*x[i]+0.5*ema[i-1]

    def test_ema_insufficient_data_all_nan(self):
        assert np.isnan(ema(np.array([1.0, 2.0]), 5)).all()


class TestSwingPivots:
    def test_swing_highs_and_lows_indices(self):
        high = np.array([10.0, 11.0, 12.0, 11.5, 10.5, 11.0, 12.5, 11.6])
        low = np.array([9.5, 9.8, 9.6, 9.7, 9.4, 9.5, 9.9, 9.3])
        hi_idxs, lo_idxs = swing_pivots(high, low, left=1, right=1)
        assert hi_idxs.tolist() == [2, 6]
        assert lo_idxs.tolist() == [2, 4]

    def test_swing_pivot_requires_strict_peak_not_tie(self):
        high = np.array([10.0, 12.0, 12.0, 10.0, 11.0])
        # tie at index 1/2: neither is a strict local max with left=1,right=1
        hi_idxs, _ = swing_pivots(high, high, left=1, right=1)
        assert hi_idxs.tolist() == []

    def test_swing_pivots_no_pivots_on_tiny_frame(self):
        hi_idxs, lo_idxs = swing_pivots(np.array([1.0, 2.0]), np.array([0.5, 1.5]))
        assert len(hi_idxs) == 0 and len(lo_idxs) == 0


class TestEqualLowsHighs:
    def test_equal_lows_clusters_within_tolerance(self):
        lows = np.array([10.0, 10.02, 10.05, 12.0, 12.03, 12.05])
        pivots = np.array([0, 2, 3, 5])  # swing-low indices into `lows`
        levels = equal_lows(lows, pivots, tolerance_pct=0.01)
        assert len(levels) == 2
        # clusters are formed from the PIVOT prices only (10.0,10.05 | 12.0,12.05)
        assert levels[0] == pytest.approx(10.025)
        assert levels[1] == pytest.approx(12.025)

    def test_equal_lows_no_cluster_when_apart(self):
        lows = np.array([10.0, 12.0])
        levels = equal_lows(lows, np.array([0, 1]), tolerance_pct=0.01)
        assert levels == [10.0, 12.0]

    def test_equal_highs_single_pivot_returns_level(self):
        highs = np.array([11.0, 11.2])
        levels = equal_highs(highs, np.array([0]), tolerance_pct=0.05)
        assert levels == [11.0]


class TestFVG:
    def test_bullish_and_bearish_fvg_exact_zones(self):
        # candle1(idx0): h=10, l=9.5 | candle2(idx1): bridge | candle3(idx2): h=11.2 l=10.5
        # -> bullish gap at formation_idx 2, zone (10.0, 10.5).
        # idx3/idx4 continue up; idx5 drops hard -> bearish gap vs idx3 at
        # formation_idx 5, zone (9.4, 10.1); idx6 recovers (no further gap).
        high = np.array([10.0, 11.0, 11.2, 10.6, 10.8, 9.4, 10.5])
        low = np.array([9.5, 10.0, 10.5, 10.1, 10.5, 9.0, 10.2])
        gaps = fvg_gaps(high, low)
        bull = [g for g in gaps if g.direction == "bullish"]
        bear = [g for g in gaps if g.direction == "bearish"]
        assert len(bull) == 1 and len(bear) == 1
        b = bull[0]
        assert b.formation_idx == 2
        assert b.bottom == pytest.approx(10.0) and b.top == pytest.approx(10.5)
        assert b.mid == pytest.approx(10.25)
        s = bear[0]
        assert s.formation_idx == 5
        assert s.bottom == pytest.approx(9.4) and s.top == pytest.approx(10.1)

    def test_no_fvg_without_imbalance(self):
        high = np.array([10.0, 10.2, 10.1, 10.4, 10.3])
        low = np.array([9.5, 9.6, 9.55, 9.7, 9.6])
        assert fvg_gaps(high, low) == []

    def test_last_fvg_in_window_picks_most_recent(self):
        # bullish gaps at formation_idx 3 (low3 10.6 > high1 10.5) and
        # formation_idx 5 (low5 10.9 > high3 10.8); no bearish gaps
        high = np.array([10.0, 10.5, 10.6, 10.8, 10.9, 11.2, 11.0])
        low = np.array([9.0, 9.5, 9.7, 10.6, 10.2, 10.9, 10.3])
        gaps = fvg_gaps(high, low)
        last = last_fvg_in_window(gaps, "bullish", now_idx=len(high) - 1, window=10)
        assert last is not None and last.formation_idx == 5
        assert last_fvg_in_window(gaps, "bearish", now_idx=len(high) - 1, window=10) is None
        assert last_fvg_in_window([], "bullish", now_idx=5, window=3) is None
        # outside the window -> None
        assert last_fvg_in_window(gaps, "bullish", now_idx=4, window=1) is None


class TestSessionExtremes:
    def test_session_extremes_filters_by_local_clock(self, make_frame):
        ts = ["2026-09-10 19:30", "2026-09-10 20:05", "2026-09-10 21:00", "2026-09-10 23:30"]
        f = make_frame("T", "5m", ts, [1, 2, 3, 4], [2, 3, 4, 5], [0.5, 1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5], [10] * 4)
        hi, lo = session_extremes(f, SessionWindow("asia", "20:00", "24:00"))
        assert hi == pytest.approx(5.0)  # bars at 20:05/21:00/23:30 → max high 5
        assert lo == pytest.approx(1.5)
        hi2, lo2 = session_extremes(f, SessionWindow("london", "02:00", "05:00"))
        assert hi2 is None and lo2 is None

    def test_session_window_midnight_wrap(self, make_frame):
        ts = ["2026-09-10 22:00", "2026-09-10 00:30"]
        f = make_frame("T", "5m", ts, [1, 2], [2, 4], [0.5, 2.0], [1.5, 3.0], [10] * 2)
        hi, lo = session_extremes(f, SessionWindow("asia", "23:00", "01:00"))
        assert hi == pytest.approx(4.0) and lo == pytest.approx(2.0)


class TestVolumeProfile:
    def test_vah_val_poc_exact_rows(self):
        high = np.array([105.0, 109.0, 108.0])
        low = np.array([101.0, 106.0, 106.5])
        vol = np.array([100.0, 400.0, 50.0])
        prof = volume_profile(high, low, vol, swing_low=100.0, swing_high=110.0, rows=10, vah_pct=0.7)
        assert prof.row_size == pytest.approx(1.0)
        assert prof.total_volume == pytest.approx(550.0)
        assert prof.vah == pytest.approx(108.0)
        assert prof.val == pytest.approx(107.0)
        assert prof.poc == pytest.approx(107.5)
        assert prof.vah_below(108.5) and not prof.vah_below(107.5)
        assert prof.val_above(106.5) and not prof.val_above(107.5)

    def test_volume_profile_zero_volume_returns_mid(self):
        prof = volume_profile(np.array([5.0]), np.array([5.0]), np.array([0.0]), 0.0, 10.0, rows=4)
        assert prof.vah == pytest.approx(5.0) and prof.val == pytest.approx(5.0)

    def test_volume_profile_rejects_degenerate_band(self):
        with pytest.raises(ValueError):
            volume_profile(np.array([5.0]), np.array([5.0]), np.array([1.0]), 5.0, 5.0, rows=4)


class TestFibLevels:
    def test_fib_levels_up_swing_exact(self):
        lev = fib_levels(100.0, 110.0, direction="up")
        assert lev["0.236"] == pytest.approx(102.36)
        assert lev["0.382"] == pytest.approx(103.82)
        assert lev["0.500"] == pytest.approx(105.0)
        assert lev["0.580"] == pytest.approx(105.8)
        assert lev["0.618"] == pytest.approx(106.18)

    def test_fib_levels_down_swing_mirror(self):
        lev = fib_levels(100.0, 110.0, direction="down")
        assert lev["0.580"] == pytest.approx(110.0 - 0.58 * 10.0)
        assert "0.380" not in lev  # no typo keys
        assert lev["0.382"] == pytest.approx(110.0 - 0.382 * 10.0)

    def test_fib_levels_keys_format(self):
        lev = fib_levels(0.0, 100.0, direction="up")
        assert set(lev.keys()) >= {"0.236", "0.382", "0.500", "0.580", "0.618"}


class TestStructureAndAMD:
    def test_swing_structure_uptrend(self):
        high = np.arange(10.0, 16.0)
        low = np.arange(9.0, 15.0)
        label, _, _ = swing_structure(high.astype(float), low.astype(float))
        assert label in ("uptrend", "ranging")  # rising series → pivots only at edges

    def test_amd_accumulation_pattern(self):
        # HH/HL sequence
        high = np.array([10, 11, 10.5, 12, 11.5, 13, 12.5, 14.0], float)
        low = np.array([9, 9.8, 9.5, 10.5, 10.0, 11.0, 10.8, 12.0], float)
        assert classify_amd(high, low, left=1, right=1) == "accumulation"

    def test_amd_distribution_pattern(self):
        high = np.array([14, 13, 13.5, 12, 12.5, 11, 11.5, 10.0], float)
        low = np.array([13, 12.2, 12.6, 11.5, 12.0, 10.5, 11.0, 9.5], float)
        assert classify_amd(high, low, left=1, right=1) == "distribution"

    def test_amd_manipulation_break_of_prior_low(self):
        # uptrend, then a final swing low BREAKS the prior swing low → manipulation
        high = np.array([10, 12, 11, 13, 12, 11.5, 11.2], float)
        low = np.array([9, 10.5, 10, 11.5, 11.2, 9.6, 9.4], float)
        # last swing low 9.4 breaks prior swing low 9.6
        assert classify_amd(high, low, left=1, right=1) == "manipulation"


class TestSMT:
    def test_bullish_smt_divergence_detected(self):
        assert smt_divergence(traded_now=100.0, traded_prev=101.0, pair_now=101.5, pair_prev=101.0, direction="long")
        assert not smt_divergence(traded_now=100.0, traded_prev=101.0, pair_now=100.5, pair_prev=101.0, direction="long")

    def test_bearish_smt_divergence_detected(self):
        assert smt_divergence(traded_now=110.0, traded_prev=109.0, pair_now=108.5, pair_prev=109.0, direction="short")
        assert not smt_divergence(traded_now=110.0, traded_prev=109.0, pair_now=109.5, pair_prev=109.0, direction="short")