"""Unit tests for the ICT HTF imbalance + IFVG model (scalp_ict_ifvg.py).

The positive-geometry test builds a full multi-timeframe day by hand so the
expected signal's entry / SL / TP are exact closed-form numbers.
"""
import numpy as np
import pandas as pd
import pytest

from src.strategies.scalp.scalp_ict_ifvg import DEFAULT_CONFIG, evaluate, htf_bias
from src.strategies.scalp.types import CandleFrame, Direction, EntryType, LiquidityMap, ScalpContext

DAY = "2026-09-10"

# session_high is deliberately below the 1m frame's own swing highs so the
# structural TP (128.9) gives < 1R and the deterministic 2R fallback kicks in
# — keeps the exact-geometry assertion free of REL/REH clustering noise.
_LIQ = LiquidityMap(
    pdh=130.0, pdl=128.4,
    asia_high=131.0, london_high=130.8,
    asia_low=128.3, london_low=128.6,
    session_high=128.9, session_low=127.9,
)


def _rising_htf(n, start, tf):
    """Deterministically rising HTF frame."""
    o = [start + i for i in range(n)]
    h = [start + i + 0.8 for i in range(n)]
    l = [start + i - 0.5 for i in range(n)]
    c = [start + i + 0.3 for i in range(n)]
    ts = pd.date_range("2026-09-03 00:00", periods=n, freq=tf)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * n}, index=ts)


def _make1m() -> pd.DataFrame:
    """275 1m bars: decline -> sweep (127.9) -> bullish IFVG (128.4..128.7) -> rise."""
    ts = pd.date_range(f"{DAY} 09:30", periods=275, freq="1min")
    o, h, l, c = [], [], [], []
    for i in range(275):
        if i <= 29:  # slow decline 129.8 -> 128.9 (no sweep yet)
            open_ = 129.8 - 0.03 * i
        elif i == 30:  # SWEEP bar: low 127.9 < pdl 128.4
            open_ = 128.9
        elif i == 31:  # IFVG candle 1
            open_ = 128.1
        elif i == 32:  # bridge candle
            open_ = 128.3
        elif i == 33:  # IFVG candle 3: low 128.7 > high[31] 128.4 -> gap (128.4,128.7)
            open_ = 128.6
        elif i == 34:  # carefully overlap: low 128.62 <= high[32] 128.7 -> NO new gap
            open_ = 128.65
        elif i <= 60:  # slow drift up
            open_ = 128.68 + 0.008 * (i - 34)
        else:  # steady rise (lows keep <= highs[i-2] -> no further gaps)
            open_ = 128.85 + 0.0022 * (i - 60)
        if i == 30:
            o.append(open_); h.append(129.0); l.append(127.9); c.append(128.1)
        elif i == 31:
            o.append(open_); h.append(128.4); l.append(128.0); c.append(128.3)
        elif i == 32:
            o.append(open_); h.append(128.7); l.append(128.2); c.append(128.6)
        elif i == 33:
            o.append(open_); h.append(129.0); l.append(128.7); c.append(128.9)
        elif i == 34:
            o.append(open_); h.append(128.73); l.append(128.62); c.append(128.68)
        elif i <= 29:
            o.append(open_); h.append(open_ + 0.1); l.append(open_ - 0.1); c.append(open_ - 0.05)
        else:
            o.append(open_); h.append(open_ + 0.08); l.append(open_ - 0.06); c.append(open_ + 0.03)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * 275}, index=ts)


def _make15m(include_tap: bool = True) -> pd.DataFrame:
    """27 15m bars; bar idx 24 taps the 30m gap (130.5,130.6) when include_tap."""
    ts = pd.date_range(f"{DAY} 09:30", periods=27, freq="15min")
    o, h, l, c = [], [], [], []
    for i in range(27):
        if i < 24:
            base = 129.7 + 0.05 * i
            o.append(base); h.append(base + 0.2); l.append(base - 0.15); c.append(base + 0.05)
        elif i == 24:  # the tap: low 130.45 <= 130.6 AND high 130.9 >= 130.5
            o.append(130.85); h.append(130.9); l.append(130.45); c.append(130.6)
        else:
            base = 130.6 + 0.08 * (i - 24)
            o.append(base); h.append(base + 0.1); l.append(base - 0.08); c.append(base + 0.04)
    if not include_tap:  # bar 24 becomes a normal continuation bar (no gap span)
        o[24], h[24], l[24], c[24] = 130.9, 131.0, 130.7, 130.95
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * 27}, index=ts)


def _make30m(with_fvg: bool = True) -> pd.DataFrame:
    """14 30m bars; bullish FVG at formation idx 5 (zone 130.5..130.6)."""
    ts = pd.date_range(f"{DAY} 09:30", periods=14, freq="30min")
    mat = [
        (129.6, 129.9, 129.4, 129.7), (129.7, 130.0, 129.5, 129.9), (129.9, 130.2, 129.6, 130.1),
        (130.1, 130.5, 130.0, 130.4), (130.4, 130.7, 130.2, 130.6), (130.6, 130.9, 130.6, 130.8),
    ]
    # extension bars overlap so they never create extra FVGs
    while len(mat) < 14:
        prev = mat[-1]
        mat.append((prev[3], prev[3] + 0.2, prev[3] - 0.2, prev[3] + 0.05))
    if not with_fvg:
        mat[5] = (130.6, 130.9, 130.3, 130.8)  # low 130.3 <= high[3] 130.5 -> no gap
    return pd.DataFrame(
        {"open": [m[0] for m in mat], "high": [m[1] for m in mat],
         "low": [m[2] for m in mat], "close": [m[3] for m in mat], "volume": [1000.0] * len(mat)},
        index=ts,
    )


def _full_frames(with_fvg=True, with_tap=True, fourh_trend="up"):
    f1h = CandleFrame.from_dataframe("TEST", "1h", _rising_htf(60, 100.0, "1h"))
    if fourh_trend == "up":
        f4h = CandleFrame.from_dataframe("TEST", "4h", _rising_htf(30, 100.0, "4h"))
    else:
        f4h = CandleFrame.from_dataframe("TEST", "4h", _rising_htf(30, 300.0, "4h").iloc[::-1].copy())
    f30 = CandleFrame.from_dataframe("TEST", "30m", _make30m(with_fvg=with_fvg))
    f15 = CandleFrame.from_dataframe("TEST", "15m", _make15m(include_tap=with_tap))
    f1m = CandleFrame.from_dataframe("TEST", "1m", _make1m())
    ctx = ScalpContext(
        symbol="TEST",
        frames={"1h": f1h, "4h": f4h, "30m": f30, "15m": f15, "1m": f1m},
        liquidity=_LIQ,
    )
    return ctx


class TestBias:
    def test_htf_bias_long_on_rising_frames(self):
        assert htf_bias(_full_frames()) == "long"

    def test_htf_bias_neutral_when_4h_falls(self):
        assert htf_bias(_full_frames(fourh_trend="down")) == "neutral"

    def test_htf_bias_neutral_when_frames_missing(self):
        ctx = ScalpContext(symbol="T", frames={})
        assert htf_bias(ctx) == "neutral"


class TestIFVGSignal:
    def test_long_ifvg_signal_exact_geometry(self):
        ctx = _full_frames()
        sigs = evaluate(ctx)
        assert len(sigs) == 1, f"expected 1 signal, got {sigs}"
        s = sigs[0]
        assert s.direction == Direction.LONG
        assert s.entry_type == EntryType.LIMIT
        assert s.entry_price == pytest.approx(128.55)  # IFVG midpoint (128.4+128.7)/2
        assert s.stop_loss == pytest.approx(127.98)  # min(low31..33)=128.0 minus 0.02 buffer
        # nearest structural TP (session_high 128.9) gives < 1R -> deterministic
        # 2R fallback: entry + 2 * risk
        assert s.risk == pytest.approx(0.57)
        assert s.take_profit == pytest.approx(128.55 + 2 * 0.57)  # 129.69
        assert s.reward == pytest.approx(2 * 0.57)
        assert s.rr == pytest.approx(2.0, abs=1e-4)
        assert s.metadata["bias"] == "long"
        assert s.metadata["ifvg_zone"][0] == pytest.approx(128.4)
        assert s.metadata["ifvg_zone"][1] == pytest.approx(128.7)
        assert s.metadata["sweep_level"] == pytest.approx(128.3)
        assert s.metadata["sweep_idx"] == 30
        assert s.metadata["ifvg_idx"] == 33

    def test_no_signal_when_htf_bias_neutral(self):
        assert evaluate(_full_frames(fourh_trend="down")) == []

    def test_no_signal_without_active_30m_fvg(self):
        assert evaluate(_full_frames(with_fvg=False)) == []

    def test_no_signal_without_15m_tap(self):
        assert evaluate(_full_frames(with_tap=False)) == []

    def test_no_signal_without_sweep(self):
        ctx = _full_frames()
        liq = LiquidityMap(pdh=130.0, pdl=127.0, asia_low=126.9, london_low=127.1, session_high=128.9)
        ctx = ScalpContext(symbol="TEST", frames=ctx.frames, liquidity=liq)
        assert evaluate(ctx) == []

    def test_smt_pair_also_makes_new_low_blocks_signal(self, make_frame):
        pair = make_frame(
            "QQQ", "1m", ["2026-09-10 09:59", "2026-09-10 10:00"],
            [128.5, 127.5], [128.6, 127.7], [128.4, 127.5], [128.5, 127.6], [1000, 1000],
        )
        ctx = _full_frames()
        ctx = ScalpContext(
            symbol=ctx.symbol, frames=ctx.frames, liquidity=ctx.liquidity,
            pair_frames={"1m": pair},
        )
        assert evaluate(ctx) == []

    def test_smt_pair_holding_low_allows_signal(self, make_frame):
        pair = make_frame(
            "QQQ", "1m", ["2026-09-10 09:59", "2026-09-10 10:00"],
            [128.5, 128.7], [128.6, 128.8], [128.4, 128.6], [128.5, 128.7], [1000, 1000],
        )
        ctx = _full_frames()
        ctx = ScalpContext(
            symbol=ctx.symbol, frames=ctx.frames, liquidity=ctx.liquidity,
            pair_frames={"1m": pair},
        )
        sigs = evaluate(ctx)
        assert len(sigs) == 1

    def test_smt_can_be_disabled_via_config(self, make_frame):
        pair = make_frame(
            "QQQ", "1m", ["2026-09-10 09:59", "2026-09-10 10:00"],
            [128.5, 127.5], [128.6, 127.7], [128.4, 127.5], [128.5, 127.6], [1000, 1000],
        )
        ctx = _full_frames()
        ctx = ScalpContext(
            symbol=ctx.symbol, frames=ctx.frames, liquidity=ctx.liquidity,
            pair_frames={"1m": pair},
        )
        cfg = dict(DEFAULT_CONFIG)
        cfg["smt_enabled"] = False
        assert len(evaluate(ctx, cfg)) == 1

    def test_missing_required_frames_returns_empty(self):
        assert evaluate(ScalpContext(symbol="T", frames={})) == []