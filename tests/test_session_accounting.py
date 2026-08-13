"""Tests for the broker-accounting fixes:
- ``get_account()`` returns the unavailable sentinel (equity None) on failure,
  never a fabricated 0.0 — killing the spurious -100% P&L print.
- ``shutdown()`` prints UNKNOWN P&L when the end-equity fetch fails.
- ``is_market_open()`` returns None (indeterminate) on timeout and the run
  loop retries instead of breaking into shutdown/liquidation.
- ``start_equity`` is persisted to a JSON state file and re-loaded on
  same-day restarts.
"""
import asyncio
import json
import logging

import pytest

import turbo_trader
from src.execution.alpaca_broker import ACCOUNT_UNAVAILABLE, account_equity
from src.execution.session_state import (
    load_start_equity,
    ny_today,
    save_start_equity,
)
from src.execution.position_manager import PositionManager


# ── account_equity helper ────────────────────────────────────────────────
class TestAccountEquityHelper:
    def test_success_returns_float(self):
        assert account_equity({"equity": 123.45, "available": True}) == 123.45

    def test_sentinel_returns_none(self):
        assert account_equity(dict(ACCOUNT_UNAVAILABLE)) is None

    def test_missing_equity_returns_none(self):
        assert account_equity({"buying_power": 1.0}) is None

    def test_non_numeric_equity_returns_none(self):
        assert account_equity({"equity": "boom", "available": True}) is None

    def test_non_dict_returns_none(self):
        assert account_equity(None) is None


# ── session_state persistence ────────────────────────────────────────────
class TestSessionState:
    def test_save_then_load_roundtrip(self, tmp_path):
        f = tmp_path / "session_state.json"
        assert save_start_equity(117113.60, state_file=f) is True
        assert load_start_equity(state_file=f) == pytest.approx(117113.60)

    def test_load_ignores_other_day(self, tmp_path):
        f = tmp_path / "session_state.json"
        f.write_text(json.dumps({"trading_date": "1999-01-01", "start_equity": 42.0}))
        assert load_start_equity(state_file=f) is None  # stale day → unknown

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_start_equity(state_file=tmp_path / "nope.json") is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        f = tmp_path / "session_state.json"
        f.write_text("{not json")
        assert load_start_equity(state_file=f) is None

    def test_ny_today_format(self):
        assert len(ny_today()) == 10  # YYYY-MM-DD


# ── shutdown P&L: UNKNOWN when end equity fetch fails ────────────────────
class TestShutdownPnlUnknown:
    def _make_trader(self, broker):
        trader = object.__new__(turbo_trader.TurboTrader)
        trader.broker = broker
        trader.pm = PositionManager(turbo_trader.STRATEGY_CONFIG)
        trader.start_equity = 100000.0
        return trader

    @pytest.mark.asyncio
    async def test_shutdown_prints_unknown_not_minus100(self, caplog):
        """A timed-out account fetch (sentinel) must print UNKNOWN P&L —
        never a fabricated -100% — and must skip the loss banner."""
        class FakeBroker:
            async def get_positions(self):
                return []
            async def get_account(self):
                return dict(ACCOUNT_UNAVAILABLE)
            async def close(self):
                pass
        trader = self._make_trader(FakeBroker())
        with caplog.at_level(logging.INFO, logger="turbo_trader"):
            await trader.shutdown(close_broker=False)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "P&L:    UNKNOWN (account fetch timed out)" in text
        assert "TURBO LOSS" not in text
        assert "TURBO PROFIT" not in text
        assert "-100" not in text
        # baseline reset so the next session re-captures
        assert trader.start_equity == 0.0

    @pytest.mark.asyncio
    async def test_shutdown_prints_real_pnl_when_equity_known(self, caplog):
        """A healthy fetch prints the normal P&L banner (regression guard)."""
        class FakeBroker:
            async def get_positions(self):
                return []
            async def get_account(self):
                return {"equity": 101000.0, "available": True}
            async def close(self):
                pass
        trader = self._make_trader(FakeBroker())
        with caplog.at_level(logging.INFO, logger="turbo_trader"):
            await trader.shutdown(close_broker=False)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "P&L:    $+1,000.00  (+1.00%)" in text
        assert "TURBO PROFIT" in text


# ── run loop: retry on None, break only on confirmed False ───────────────
class TestRunLoopRetriesOnNone:
    @pytest.mark.asyncio
    async def test_loop_retries_on_none_and_shuts_down_on_false(self, monkeypatch):
        """market_open: None (timeout) → loop retries; True → trades;
        False (confirmed) → EOD shutdown.  The None must NOT trigger
        shutdown/liquidation."""
        calls = {"ticks": 0, "shutdowns": 0, "clock_calls": 0}

        class FakeClock:
            async def is_market_open(self):
                results = [None, None, True, True, False]
                idx = min(calls["clock_calls"], len(results) - 1)
                calls["clock_calls"] += 1
                return results[idx]

        class FakeBroker(FakeClock):
            async def startup_health_check(self):
                pass
            async def cancel_orders_by_client_id_prefix(self, prefix):
                return 0
            async def get_positions(self):
                return []
            async def get_account(self):
                return {"equity": 1000.0, "available": True}
            async def close(self):
                pass

        trader = object.__new__(turbo_trader.TurboTrader)
        trader.broker = FakeBroker()
        trader.pm = PositionManager(turbo_trader.STRATEGY_CONFIG)
        trader.start_equity = 0.0
        trader.levels_cache = {}
        trader._entry_times = {}
        trader.day_trades = []
        trader._ls_symbols = set()
        trader._extended_holds = set()
        trader.liquidity_sweep = type("LS", (), {})()
        trader.liquidity_sweep.levels = {}

        async def noop_calculate_levels(provider):
            pass
        trader.liquidity_sweep.calculate_levels = noop_calculate_levels
        trader.provider = None

        async def fake_wait_for_market_open():
            pass
        trader.wait_for_market_open = fake_wait_for_market_open
        trader._ensure_protective_stops = lambda: asyncio.sleep(0)
        trader._post_startup_cleanup = lambda: asyncio.sleep(0)
        trader._sync_positions_from_broker = lambda: asyncio.sleep(0)
        trader._is_near_close = lambda: False

        async def fake_safe_tick(tick_num):
            calls["ticks"] += 1
        trader._safe_tick = fake_safe_tick

        async def fake_shutdown(close_broker=True):
            calls["shutdowns"] += 1
        trader.shutdown = fake_shutdown

        monkeypatch.setattr(turbo_trader, "CHECK_INTERVAL", 0.001)
        await trader.run()

        assert calls["clock_calls"] >= 5
        assert calls["ticks"] == 2          # traded after the clock recovered
        assert calls["shutdowns"] == 1      # exactly one EOD shutdown (confirmed False)


# ── begin_session baseline persistence ────────────────────────────────────
class TestBeginSessionBaseline:
    @pytest.mark.asyncio
    async def test_reloads_persisted_baseline_same_day(self, tmp_path, monkeypatch):
        """A same-day restart re-loads the persisted baseline instead of
        re-capturing mid-day equity."""
        import src.execution.session_state as ss
        state_file = tmp_path / "session_state.json"
        save_start_equity(50000.0, state_file=state_file)
        monkeypatch.setattr(ss, "_DEFAULT_STATE_FILE", state_file)

        trader = object.__new__(turbo_trader.TurboTrader)
        trader.start_equity = 0.0
        trader.broker = None  # must NOT be touched when baseline reloads
        assert await trader._begin_session(assumed=False) is True
        assert trader.start_equity == 50000.0
