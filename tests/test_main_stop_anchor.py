"""Post-go-live stop-anchoring fix (2026-09-16) — hermetic regression tests.

Live evidence (``logs/trades_20260916.log``, main trader booted from 7c2a583):

* COIN Box-Theory MARKET entry — signal ``LONG entry=167.96 SL=167.01
  TP=184.14`` — filled at **162.58**, i.e. 5+ points below the signal
  reference.  The GTC stop was submitted at the strategy SL *above* the
  fill, Alpaca rejected it 4x with ``42210000 "stop price must be less than
  current price"``, the position ran with NO broker stop, and the
  in-process SL then instantly treated the stale level as breached and
  closed the trade (``scalp_sl_inproc``).  After the 5-bar cooldown the same
  setup re-entered and repeated the whole loop.
* Strategy levels such as ``211.160004`` were also rejected as sub-penny
  (``42210000`` sub-penny increment), silently leaving positions without a
  take-profit order.

These tests pin the fix:

(a) long signal whose strategy SL sits above the fill -> an ANCHORED stop
    below the fill is what gets submitted to the broker;
(b) degenerate geometry -> -6% backstop + loud log;
(c) ``42210000`` invalid-level rejection -> ONE log, NO retries (while
    other 422 classes keep retrying);
(d) broker stop absent -> the in-process SL is evaluated from the ANCHORED
    stop (a price below the stale signal SL does not close the position),
    plus the loud per-minute "no broker stop" warning;
(e) the per-session entry cap blocks the 4th entry for a symbol.

The fakes are the hermetic ones from ``tests/test_live_scalp_integration.py``
(in-memory broker, deterministic bar feed) — nothing here touches the
network or a real account.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

import live_trader
from src.execution.broker import OrderType
from src.strategies.scalp.types import Direction, EntryType

from tests.test_live_scalp_integration import (
    FakeBroker,
    FakeProvider,
    _frame_df,
    _make_scalp_trader,
    _sig,
)

STOP_LESS = ('{"code":42210000,"market_price":"162.6",'
             '"message":"stop price must be less than current price",'
             '"stop_price":"167.01"}')
STOP_GREATER = ('{"code":42210000,"market_price":"162.6",'
                '"message":"stop price must be greater than current price",'
                '"stop_price":"161.0"}')
WASH_TRADE = ('{"code":40310000,"message":"potential wash trade detected. '
              'use complex orders","reject_reason":"opposite side '
              'market/stop order exists"}')


class StopRejectingBroker(FakeBroker):
    """FakeBroker whose stop submissions fail with *error*."""

    def __init__(self, *args, error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = error
        self.stop_attempts = []

    async def place_stop_order(self, symbol, qty, stop_price, client_id=None, side="SELL"):
        self.stop_attempts.append((symbol, qty, stop_price, side))
        if self.error is not None:
            raise Exception(self.error)
        return await super().place_stop_order(
            symbol, qty, stop_price, client_id=client_id, side=side)


def _one_min_frames(close: float):
    """Minimal 1-minute frame the risk pass reads (last close == *close*)."""
    return {"1min": _frame_df([close] * 6, freq="1min")}


# ---------------------------------------------------------------------------
# (a/b) anchored levels — pure function
# ---------------------------------------------------------------------------
class TestAnchorLevels:
    def test_long_stale_sl_above_fill_is_reanchored(self):
        """The live COIN case: SL 167.01 with a 162.58 fill."""
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=162.58, entry_ref=167.96,
            sl_ref=167.01, tp_ref=184.14,
        )
        # risk = 167.96 - 167.01 = 0.95 -> 162.58 - 0.95 = 161.63 (< fill)
        assert levels.sl == 161.63
        assert levels.sl < 162.58
        assert levels.sl_source == "fill_risk"
        assert levels.sl_reanchored is True
        # the TP was still valid vs the fill and is kept verbatim
        assert levels.tp == 184.14
        assert levels.tp_source == "strategy"

    def test_short_stale_levels_are_reanchored(self):
        levels = live_trader.anchor_scalp_levels(
            is_short=True, fill=102.0, entry_ref=100.0,
            sl_ref=99.0, tp_ref=104.0,
        )
        # short: SL must be ABOVE the fill, TP BELOW it
        assert levels.sl == 103.0 and levels.sl > 102.0
        assert levels.sl_source == "fill_risk"
        assert levels.tp == 98.0 and levels.tp < 102.0
        assert levels.tp_source == "fill_reward"

    def test_valid_strategy_geometry_is_kept_verbatim(self):
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=100.0, entry_ref=100.0,
            sl_ref=98.0, tp_ref=106.0,
        )
        assert (levels.sl, levels.sl_source) == (98.0, "strategy")
        assert (levels.tp, levels.tp_source) == (106.0, "strategy")
        assert levels.sl_reanchored is False and levels.tp_reanchored is False

    def test_min_distance_floor_clamps_a_too_close_stop(self):
        """A tiny strategy risk is widened to the min-distance floor."""
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=100.0, entry_ref=100.0,
            sl_ref=99.99, tp_ref=100.01,
            min_distance_pct=0.0005, min_distance_abs=0.01,
        )
        assert levels.min_distance == 0.05          # 0.05% of 100.0
        assert levels.sl == 99.95 and levels.sl < 100.0
        assert levels.sl_source == "fill_risk_clamped"
        assert levels.tp == 100.05 and levels.tp > 100.0
        assert levels.tp_source == "fill_reward_clamped"

    def test_degenerate_geometry_falls_back_to_backstop(self):
        """risk (100 - 99 = 1.0) exceeds the 0.50 fill -> stop would be < 0."""
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=0.5, entry_ref=100.0,
            sl_ref=99.0, tp_ref=106.0,
        )
        assert levels.sl_is_backstop is True
        assert levels.sl_source == "backstop"
        assert levels.sl == round(0.5 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)
        assert levels.sl < 0.5
        assert levels.sl_reanchored is False

    def test_missing_strategy_sl_uses_backstop_from_fill(self):
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=50.0, entry_ref=50.0, sl_ref=None, tp_ref=None,
        )
        assert levels.sl_source == "no_strategy_sl"
        assert levels.sl == round(50.0 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)
        assert levels.tp is None and levels.tp_source == "none"

    def test_tp_dropped_when_no_reward_can_be_derived(self):
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=100.0, entry_ref=100.0,
            sl_ref=98.0, tp_ref=100.0,   # TP == entry -> zero reward
        )
        assert levels.sl == 98.0
        assert levels.tp is None and levels.tp_source == "none"

    def test_unanchorable_fill_keeps_strategy_levels(self):
        levels = live_trader.anchor_scalp_levels(
            is_short=False, fill=0.0, entry_ref=100.0, sl_ref=98.0, tp_ref=106.0,
        )
        assert levels.sl_source == "unanchored" and levels.sl == 98.0
        assert levels.tp_source == "unanchored" and levels.tp == 106.0

    def test_levels_are_tick_normalised(self):
        assert live_trader._normalize_order_price(211.160004) == 211.16
        assert live_trader._normalize_order_price(106.999) == 107.0
        assert live_trader._normalize_order_price(0.123456) == 0.1235

    def test_anchor_kill_switch_is_env_backed(self):
        assert isinstance(live_trader.SCALP_ANCHOR_LEVELS, bool)


class TestInvalidStopLevelClassification:
    def test_less_than_current_price_is_invalid_level(self):
        assert live_trader._is_invalid_stop_level_error(Exception(STOP_LESS)) is True

    def test_greater_than_current_price_is_invalid_level(self):
        assert live_trader._is_invalid_stop_level_error(Exception(STOP_GREATER)) is True

    def test_other_422_classes_stay_retryable(self):
        assert live_trader._is_invalid_stop_level_error(Exception(WASH_TRADE)) is False
        sub_penny = ('{"code":42210000,"message":"invalid limit_price 211.160004. '
                     'sub-penny increment does not fulfill minimum pricing criteria"}')
        assert live_trader._is_invalid_stop_level_error(Exception(sub_penny)) is False


# ---------------------------------------------------------------------------
# (a) end-to-end: the anchored stop is what reaches the broker
# ---------------------------------------------------------------------------
class TestAnchoredStopReachesBroker:
    @pytest.mark.asyncio
    async def test_long_signal_with_stale_sl_submits_stop_below_fill(self, caplog):
        # broker fill (162.58) is what the entry ACTUALLY filled at
        broker = FakeBroker(fill_price=162.58)
        trader = _make_scalp_trader(broker)
        sig = _sig("COIN", Direction.LONG, 167.96, 167.01, 184.14, "box_theory")
        with caplog.at_level("WARNING", logger="live_trader"):
            ok = await trader._scalp_enter(sig, 167.96)
        assert ok is True
        # position recorded at the FILL, not the signal reference
        pos = trader.pm.get_positions()["COIN"]
        assert pos.entry_price == 162.58
        # exactly one stop submission, at the anchored level BELOW the fill
        assert len(broker.stop_requests) == 1
        sym, qty, stop_price, _cid, side = broker.stop_requests[0]
        assert (sym, side) == ("COIN", "SELL")
        assert stop_price == 161.63
        assert stop_price < pos.entry_price
        # the stale signal SL (167.01) was never submitted
        assert stop_price != 167.01
        # in-process state mirrors the anchored level (requirement 3)
        state = trader._scalp_positions["COIN"]
        assert state["sl"] == 161.63 and state["stop_placed"] is True
        assert state["sl_source"] == "fill_risk"
        # the keep-leg TP survives (valid vs the fill) and is tick-normalised
        tps = [o for o in broker.orders if o.order_type == OrderType.LIMIT]
        assert len(tps) == 1 and float(tps[0].limit_price) == 184.14
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "stale vs fill" in text and "re-anchored" in text

    @pytest.mark.asyncio
    async def test_sub_penny_tp_is_tick_normalised_before_submission(self):
        broker = FakeBroker(fill_price=100.0)
        trader = _make_scalp_trader(broker)
        sig = _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.160004, "box_theory")
        assert await trader._scalp_enter(sig, 100.0) is True
        tps = [o for o in broker.orders if o.order_type == OrderType.LIMIT]
        assert len(tps) == 1
        assert float(tps[0].limit_price) == 106.16   # not 106.160004
        assert trader._scalp_positions["NVDA"]["tp"] == 106.16

    @pytest.mark.asyncio
    async def test_fill_from_order_result_wins_over_position_avg(self):
        """OrderResult.filled_avg_price is the first source of truth."""
        broker = FakeBroker(fill_price=100.0)
        trader = _make_scalp_trader(broker)
        result = MagicMock(filled_avg_price=95.0)
        assert await trader._resolve_fill_price("NVDA", result, 100.0) == 95.0

    @pytest.mark.asyncio
    async def test_fill_falls_back_to_position_then_signal_price(self, caplog):
        broker = FakeBroker(fill_price=100.0)
        trader = _make_scalp_trader(broker)
        broker.positions = {"NVDA": {"symbol": "NVDA", "qty": 5.0,
                                     "avg_entry_price": 101.25}}
        assert await trader._resolve_fill_price("NVDA", MagicMock(), 100.0) == 101.25
        broker.positions = {}
        with caplog.at_level("WARNING", logger="live_trader"):
            assert await trader._resolve_fill_price("NVDA", MagicMock(), 100.0) == 100.0
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "broker fill price unknown" in text


# ---------------------------------------------------------------------------
# (b) degenerate geometry -> backstop + loud log
# ---------------------------------------------------------------------------
class TestDegenerateGeometryBackstop:
    @pytest.mark.asyncio
    async def test_backstop_used_and_logged_loudly(self, caplog):
        # strategy risk (100 - 99 = 1.0) is LARGER than the fill (0.5), so
        # re-pricing the stop from the fill degenerates below zero — the only
        # remaining option is the -6% backstop, loudly.
        broker = FakeBroker(fill_price=0.5)
        trader = _make_scalp_trader(broker)
        sig = _sig("COIN", Direction.LONG, 100.0, 99.0, 106.0, "box_theory")
        with caplog.at_level("ERROR", logger="live_trader"):
            assert await trader._scalp_enter(sig, 100.0) is True
        pos = trader.pm.get_positions()["COIN"]
        assert pos.entry_price == 0.5
        expected = round(0.5 * (1 - live_trader.PROTECTIVE_STOP_PCT), 2)
        assert expected == 0.47
        assert broker.stop_requests[0][2] == expected
        state = trader._scalp_positions["COIN"]
        assert state["sl"] == expected and state["sl_source"] == "backstop"
        assert state["sl"] == pos.stop_loss_price
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "strategy SL invalid vs fill — using backstop" in text


# ---------------------------------------------------------------------------
# (c) invalid-level 42210000 -> single log, no retries
# ---------------------------------------------------------------------------
class TestInvalidLevelFastFail:
    @pytest.mark.asyncio
    async def test_invalid_level_logs_once_and_does_not_retry(self, caplog):
        broker = StopRejectingBroker(error=STOP_LESS)
        trader = _make_scalp_trader(broker)
        with caplog.at_level("ERROR", logger="live_trader"):
            ok = await trader._place_protective_stop(
                "COIN", 100, 162.58, is_short=False, stop_price=167.01,
                initial_delay=0.01,
            )
        assert ok is False
        assert len(broker.stop_attempts) == 1          # ONE attempt only
        msgs = [r.getMessage() for r in caplog.records]
        invalid = [m for m in msgs if "INVALID-LEVEL" in m]
        assert len(invalid) == 1
        assert "NOT retrying" in invalid[0]
        assert not any("retrying in" in m for m in msgs)
        assert not any("FAILED after" in m for m in msgs)

    @pytest.mark.asyncio
    async def test_greater_than_current_price_also_fast_fails(self):
        broker = StopRejectingBroker(error=STOP_GREATER)
        trader = _make_scalp_trader(broker)
        ok = await trader._place_protective_stop(
            "COIN", 100, 162.58, is_short=True, stop_price=161.0,
            initial_delay=0.01,
        )
        assert ok is False and len(broker.stop_attempts) == 1

    @pytest.mark.asyncio
    async def test_other_rejections_still_retry_with_backoff(self):
        broker = StopRejectingBroker(error=WASH_TRADE)
        trader = _make_scalp_trader(broker)
        ok = await trader._place_protective_stop(
            "NVDA", 10, 100.0, is_short=False, stop_price=98.0,
            max_attempts=4, initial_delay=0.0,
        )
        assert ok is False
        assert len(broker.stop_attempts) == 4          # unchanged behaviour


# ---------------------------------------------------------------------------
# (d) in-process SL is evaluated from the ANCHORED stop
# ---------------------------------------------------------------------------
class TestInProcessStopUsesAnchoredLevel:
    def _trader_with_position(self, price):
        broker = FakeBroker(fill_price=price)
        trader = _make_scalp_trader(broker, FakeProvider(_one_min_frames(price)))
        trader.pm.open_position("COIN", 100.0, 162.58)
        trader._scalp_positions["COIN"] = {
            "entry": 162.58, "sl": 161.63, "tp": None,
            "direction": "LONG", "stop_placed": False,
            "be_done": False, "be_trigger_r": 1.0, "be_buffer": 0.0,
            "trailing": True, "trail_r": 1.0, "trail_trigger_r": 1.0,
            "stop_order_id": None, "tp_placed": False, "no_stop_warn_ts": None,
        }
        return trader

    @pytest.mark.asyncio
    async def test_price_between_anchored_and_stale_sl_does_not_close(self, caplog):
        """162.00 < stale signal SL 167.01 but > anchored SL 161.63."""
        trader = self._trader_with_position(162.0)
        trader._scalp_close_position = AsyncMock(return_value=True)
        with caplog.at_level("WARNING", logger="live_trader"):
            await trader._scalp_risk_pass()
        trader._scalp_close_position.assert_not_awaited()
        assert trader.pm.has_position("COIN")
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "NO BROKER STOP" in text      # loud, never silent
        assert "$161.63" in text

    @pytest.mark.asyncio
    async def test_price_below_anchored_sl_closes(self, caplog):
        trader = self._trader_with_position(161.0)
        trader._scalp_close_position = AsyncMock(return_value=True)
        with caplog.at_level("WARNING", logger="live_trader"):
            await trader._scalp_risk_pass()
        trader._scalp_close_position.assert_awaited_once()
        args = trader._scalp_close_position.await_args.args
        assert args[0] == "COIN" and args[2] == "scalp_sl_inproc"
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "in-process SL $161.63 breached" in text

    @pytest.mark.asyncio
    async def test_no_broker_stop_warning_is_rate_limited_to_a_minute(self, caplog):
        trader = self._trader_with_position(162.0)
        trader._scalp_close_position = AsyncMock(return_value=True)
        with caplog.at_level("WARNING", logger="live_trader"):
            await trader._scalp_risk_pass()
            await trader._scalp_risk_pass()     # same tick cadence → suppressed
        warns = [r.getMessage() for r in caplog.records if "NO BROKER STOP" in r.getMessage()]
        assert len(warns) == 1
        state = trader._scalp_positions["COIN"]
        assert state["no_stop_warn_ts"] is not None
        # ...and it fires again once the window has elapsed
        state["no_stop_warn_ts"] -= (live_trader.SCALP_NO_STOP_WARN_SECONDS + 1.0)
        with caplog.at_level("WARNING", logger="live_trader"):
            await trader._scalp_risk_pass()
        warns = [r.getMessage() for r in caplog.records if "NO BROKER STOP" in r.getMessage()]
        assert len(warns) == 2


# ---------------------------------------------------------------------------
# (e) per-session re-entry churn cap
# ---------------------------------------------------------------------------
class TestEntryChurnCap:
    def _reset(self, trader):
        if trader.pm.has_position("NVDA"):
            trader.pm.close_position("NVDA", exit_price=100.0, exit_reason="test")
        trader._scalp_cleanup_state("NVDA")
        trader._cooldown_until.pop("NVDA", None)

    @pytest.mark.asyncio
    async def test_cap_blocks_the_fourth_entry(self, caplog):
        assert live_trader.SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION == 3
        broker = FakeBroker(fill_price=100.0)
        trader = _make_scalp_trader(broker)
        sig = _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.0, "box_theory")
        results = []
        with caplog.at_level("INFO", logger="live_trader"):
            for _ in range(4):
                trader._last_signal_key.pop("NVDA", None)
                results.append(await trader._scalp_enter(sig, 100.0))
                self._reset(trader)
        assert results == [True, True, True, False]
        # three entry orders reached the broker, the 4th never did
        entries = [o for o in broker.orders if o.order_type == OrderType.MARKET]
        assert len(entries) == 3
        assert len(broker.orders) == 6      # 3 entries + 3 day-limit TPs
        assert trader._scalp_entries["NVDA"] == 3
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "session entry cap reached" in text
        assert "entry 3/3 this session (cap reached" in text

    @pytest.mark.asyncio
    async def test_cap_is_per_symbol_and_per_day(self, caplog):
        broker = FakeBroker(fill_price=100.0)
        trader = _make_scalp_trader(broker)
        trader._record_scalp_entry("NVDA")
        trader._record_scalp_entry("NVDA")
        trader._record_scalp_entry("NVDA")
        assert trader._scalp_entry_cap_reached("NVDA") is True
        # another symbol keeps its own budget
        assert trader._scalp_entry_cap_reached("COIN") is False
        # a new day clears the counter
        trader._scalp_entries_day = "1999-01-01"
        assert trader._scalp_entry_cap_reached("NVDA") is False
        assert trader._scalp_entries == {}

    @pytest.mark.asyncio
    async def test_tick_loop_skips_capped_symbol(self, caplog):
        broker = FakeBroker(fill_price=100.0)
        trader = _make_scalp_trader(broker)
        for _ in range(3):
            trader._record_scalp_entry("NVDA")
        with caplog.at_level("INFO", logger="live_trader"):
            await trader._tick_scalp(1)
        assert broker.orders == []
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "session entry cap reached" in text

    def test_cap_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(live_trader, "SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION", 0)
        trader = _make_scalp_trader()
        for _ in range(10):
            trader._record_scalp_entry("NVDA")
        assert trader._scalp_entry_cap_reached("NVDA") is False
