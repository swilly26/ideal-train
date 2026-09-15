"""Regression tests for the main-trader market-gate hardening (PR #35).

2026-09-14 incident that motivated this: a whole-container freeze hid a 9-hour
trader stall.  The pre-open gate waited silently on an indeterminate broker
clock, a healthy-but-quiet session produced zero intraday log lines, and the
watchdog's staleness check could not tell "alive but silent" from "dead"
(2026-09-14 Monday: two unattributed restarts, an empty supervise.log, zero
watchdog liveness evidence; the missed Monday session was discovered by chance
almost at the end of the trading day).

This suite locks in the new behavior for the MAIN trader::

1. ``wait_for_market_open``
   (a) ``is_market_open`` RAISES -> loud per-attempt retry on the SHORT
      backoff (MARKET_GATE_RETRY_SECONDS); it MUST NOT silently long-sleep;
      once the clock recovers it trades via the CONFIRMED-open path.
   (b) ``is_market_open`` returns False (CONFIRMED-CLOSED) -> defers forever
      with the configured heartbeat; it NEVER fail-opens while confirmed
      closed (a confirmed close is the ONE valid reason to defer, even 30 min
      past the fail-open buffer).
   (b2) local ET past 09:30 + buffer AND status NOT confirmed-closed
      (None/exception/UNKNOWN) -> FAIL-OPEN on local time and proceeds to the
      tick loop via ``_begin_session(assumed=True)``.
   (c) while RTH per the local DST-aware schedule says open but the gate is
      still waiting, a loud overdue line fires every
      MARKET_GATE_OVERDUE_LOG_SECONDS (with N minutes past and the fail-open
      countdown).  The overdue line is gate-scoped: it stops once the gate
      returns and the trader is in the tick loop.
2. intraday market check inside ``run()`` (d): UNKNOWN at loop top -> retries
   on MARKET_LOOP_UNKNOWN_RETRY_SECONDS and is NEVER treated as a confirmed
   close (no spurious mid-session liquidation); a CONFIRMED False mid-day ends
   the session exactly as before (EOD break + shutdown(close_broker=False)) —
   main may hold overnight with broker stops; that contract is unchanged.
   While in the tick loop during RTH the intraday liveness heartbeat
   ("Tick loop alive") fires — and the gate's overdue line must NOT.

Hermetic: the broker clock is fully mocked (raise/None/True/False queues plus
stand-in stubs for every startup call site); no network, no real dirs/pids
from the live stack, and no test-time timer longer than ~0.05 s.
"""
import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

import live_trader
from src.execution.position_manager import PositionManager

# ---------------------------------------------------------------------------
# Hermetic clock drivers
# ---------------------------------------------------------------------------


def _freeze_datetime(monkeypatch, et_iso: str) -> datetime:
    """Freeze ``live_trader.datetime.now`` at a fixed America/New_York instant."""
    fixed = datetime.fromisoformat(et_iso)  # tz-aware, ET

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr(live_trader, "datetime", FakeDT)
    return fixed


class FakeClockBroker:
    """Broker whose clock is an explicit queue.  Queue items:

    - ``True`` / ``False``  -> confirmed open / confirmed closed
    - ``None``             -> indeterminate clock answer (timeout sentinel)
    - an ``Exception``      -> the clock call raises (network error)
    """

    def __init__(self, clock_results):
        self._clock_results = list(clock_results)
        self.call_count = 0

    async def is_market_open(self):
        if self._clock_results:
            self._last_answer = self._clock_results.pop(0)
        # Once the queue is exhausted the clock keeps answering with its LAST
        # answer (a stable real-world clock): a test that runs longer than its
        # queue must NOT see an AssertionError — that would be an artificial
        # "UNKNOWN" and could mask a real never-fail-open regression.
        result = self._last_answer
        self.call_count += 1
        if isinstance(result, Exception):
            raise result
        return result  # True / False / None, as queued

    async def close(self):
        pass


def _make_gate_trader(broker=None):
    trader = object.__new__(live_trader.LiveTrader)
    trader.broker = broker
    trader.start_equity = 0.0
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader.day_trades = []
    trader._begin_session = AsyncMock(return_value=True)
    return trader


def _fast_gate_knobs(monkeypatch, *, retry=0.02, heartbeat=0.05,
                    fail_open_buffer=1.0, overdue=3600.0):
    """Gate env knobs -> test-fast values (real defaults: 20/240/600/60 s)."""
    monkeypatch.setattr(live_trader, "MARKET_GATE_RETRY_SECONDS", retry)
    monkeypatch.setattr(live_trader, "MARKET_GATE_HEARTBEAT_SECONDS", heartbeat)
    monkeypatch.setattr(live_trader, "MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS", fail_open_buffer)
    monkeypatch.setattr(live_trader, "MARKET_GATE_OVERDUE_LOG_SECONDS", overdue)


def _stub_seconds_until_open(monkeypatch, value=0.0):
    """The DST math is tested elsewhere (test_market_open_wait.py); here we
    test the CLOCK handling only, so the gate is always 'at/past open' and
    goes straight to the broker-clock branch."""
    monkeypatch.setattr(live_trader.LiveTrader, "_seconds_until_open",
                        staticmethod(lambda: value))


_RTH_MORNING = "2026-09-16T10:00:00-04:00"  # Wed 10:00 ET == inside RTH,
# 30 min past 09:30, 20 min after the 10:00+buffer line when buffer <= 600.


# ---------------------------------------------------------------------------
# (a) clock raises -> loud retry, no silent long sleep -> trades once recovered
# ---------------------------------------------------------------------------


class TestClockRaisesRetriesLoudly:
    @pytest.mark.asyncio
    async def test_raises_then_recovers_proceeds_confirmed(self, monkeypatch, caplog):
        """Two raises then True: log every attempt loudly, sleep only the
        SHORT backoff, and begin the session CONFIRMED (assumed=False)."""
        _freeze_datetime(monkeypatch, _RTH_MORNING)
        _fast_gate_knobs(monkeypatch, retry=0.02, heartbeat=240.0,
                         fail_open_buffer=1e7, overdue=3600.0)
        _stub_seconds_until_open(monkeypatch, value=0.0)
        broker = FakeClockBroker([RuntimeError("clock endpoint down"),
                                  RuntimeError("clock endpoint down"),
                                  True])
        trader = _make_gate_trader(broker)
        with caplog.at_level(logging.WARNING, logger="live_trader"):
            # 5s wall cap: if the code wrongly slept the 240s heartbeat
            # interval (the old silent behavior) this fails loudly on timeout.
            await asyncio.wait_for(trader.wait_for_market_open(), timeout=5.0)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Market clock check FAILED (pre-open — attempt 1)" in text
        assert "Market clock check FAILED (pre-open — attempt 2)" in text
        assert "Market clock UNKNOWN (attempt 2, last error: RuntimeError:" in text
        assert broker.call_count == 3
        trader._begin_session.assert_awaited_once_with(assumed=False)

    @pytest.mark.asyncio
    async def test_none_value_is_unknown_and_retried_not_closed(self, monkeypatch, caplog):
        """Broker returning None (timeout sentinel) must behave exactly like a
        raise: UNKNOWN, loud retry, and once it recovers -> confirmed entry.
        (None must never be mistaken for a confirmed close; regression guard.)"""
        _freeze_datetime(monkeypatch, _RTH_MORNING)
        _fast_gate_knobs(monkeypatch, retry=0.02, heartbeat=240.0,
                        fail_open_buffer=1e7, overdue=3600.0)
        _stub_seconds_until_open(monkeypatch, value=0.0)
        broker = FakeClockBroker([None, None, None, True])
        trader = _make_gate_trader(broker)
        with caplog.at_level(logging.WARNING, logger="live_trader"):
            await asyncio.wait_for(trader.wait_for_market_open(), timeout=5.0)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Market clock UNKNOWN (attempt 1, last error: None)" in text
        assert "Market clock UNKNOWN (attempt 3, last error: None)" in text
        assert "NOT a confirmed close" in text
        assert broker.call_count == 4
        trader._begin_session.assert_awaited_once_with(assumed=False)


# ---------------------------------------------------------------------------
# (b) confirmed-closed defers with heartbeat; never fail-opens while closed
# ---------------------------------------------------------------------------


class TestConfirmedClosedDefers:
    @pytest.mark.asyncio
    async def test_confirmed_closed_past_buffer_still_defers_until_true(self, monkeypatch, caplog):
        """10:00 ET = 30 min past the 09:30 instants; a CONFIRMED-CLOSED clock
        must STILL defer (heartbeat heartbeats, repeated checks).  The buffer
        is a fail-open escape only for NON-confirmed statuses."""
        _freeze_datetime(monkeypatch, _RTH_MORNING)
        _fast_gate_knobs(monkeypatch, retry=0.02, heartbeat=0.05,
                        fail_open_buffer=1.0, overdue=3600.0)
        _stub_seconds_until_open(monkeypatch, value=0.0)
        broker = FakeClockBroker([False] * 8)
        trader = _make_gate_trader(broker)
        begin = trader._begin_session
        with caplog.at_level(logging.INFO, logger="live_trader"):
            with pytest.raises(asyncio.TimeoutError):
                # 0.5s of test time ~= 6 clock checks; gate must STILL be
                # waiting when the cap expires (only a later True/UNKNOWN could
                # exit — False alone never exits).
                await asyncio.wait_for(trader.wait_for_market_open(), timeout=0.5)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert broker.call_count >= 3       # kept re-asking the broker clock
        assert begin.await_count == 0       # never started a session
        assert "FAIL-OPEN" not in text      # confirmed-closed never fail-opens
        assert "Still waiting for market open, next check in" in text  # heartbeat


# ---------------------------------------------------------------------------
# (b2) UNKNOWN clock + local ET past buffer -> fail-open on local time
# ---------------------------------------------------------------------------


class TestFailOpenOnUnknownPastBuffer:
    @pytest.mark.asyncio
    async def test_unknown_past_buffer_fails_open_with_assumed_session(self, monkeypatch, caplog):
        """Clock keeps raising; local ET is past 09:30+buffer; last status is
        UNKNOWN (not confirmed-closed) -> FAIL-OPEN on local time and proceed
        to the tick loop via _begin_session(assumed=True)."""
        _freeze_datetime(monkeypatch, _RTH_MORNING)
        _fast_gate_knobs(monkeypatch, retry=0.02, heartbeat=240.0,
                        fail_open_buffer=1.0, overdue=3600.0)
        _stub_seconds_until_open(monkeypatch, value=0.0)
        broker = FakeClockBroker([RuntimeError("api down"), True])
        trader = _make_gate_trader(broker)
        with caplog.at_level(logging.WARNING, logger="live_trader"):
            await asyncio.wait_for(trader.wait_for_market_open(), timeout=5.0)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "FAIL-OPEN at 10:00:00 ET" in text
        assert "last clock status was UNKNOWN" in text
        trader._begin_session.assert_awaited_once_with(assumed=True)


# ---------------------------------------------------------------------------
# (c) loud overdue line while RTH says open but the gate still waits
# ---------------------------------------------------------------------------


class TestLoudOverdueWhileRthOpen:
    @pytest.mark.asyncio
    async def test_overdue_loud_line_fires_with_minutes_and_countdown(self, monkeypatch, caplog):
        """10:00 ET (RTH, 30 min past open), clock confirmed-closed: the
        'STILL WAITING FOR OPEN N MINUTES…' loud line fires with the FAIL-OPEN
        countdown; once the clock recovers the gate returns (session starts) —
        the overdue line is gate-scoped and cannot fire inside the tick loop."""
        _freeze_datetime(monkeypatch, _RTH_MORNING)
        _fast_gate_knobs(monkeypatch, retry=0.02, heartbeat=240.0,
                       fail_open_buffer=600.0, overdue=0.02)
        _stub_seconds_until_open(monkeypatch, value=0.0)
        broker = FakeClockBroker([False, False, False, False, False, True])
        trader = _make_gate_trader(broker)
        with caplog.at_level(logging.WARNING, logger="live_trader"):
            await asyncio.wait_for(trader.wait_for_market_open(), timeout=5.0)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "STILL WAITING FOR OPEN 30 MINUTES PAST 09:30 ET" in text
        assert "(market_open=CONFIRMED-CLOSED, attempt=" in text
        assert "FAIL-OPEN in 0s" in text  # at 10:00 the 600s buffer has lapsed
        trader._begin_session.assert_awaited_once_with(assumed=False)


# ---------------------------------------------------------------------------
# (d) intraday (run loop): UNKNOWN retried, never treated as closed; confirmed
#     False ends the session exactly as before; tick heartbeat fires
# ---------------------------------------------------------------------------


class TestIntradayMarketCheck:
    @pytest.mark.asyncio
    async def test_unknown_retried_never_closed_confirmed_false_ends_session(self, monkeypatch, caplog):
        """run() tick loop with clock sequence UNKNOWN,UNKNOWN,OPEN,OPEN,CLOSED:
        - UNKNOWN at loop top -> retry on MARKET_LOOP_UNKNOWN_RETRY_SECONDS,
          no tick, NEVER treated as a confirmed close (no EOD liquidation).
        - OPEN -> ticks run (2 ticks).
        - CONFIRMED False -> the standard EOD break: shutdown(close_broker=False)
          (main may hold overnight with broker stops — unchanged contract).
        - intraday liveness heartbeat "Tick loop alive" fires during RTH, and
          the gate's overdue line is absent inside the tick loop."""
        class IntradayBroker(FakeClockBroker):
            def __init__(self, results):
                super().__init__(results)

            async def startup_health_check(self):
                pass

            async def cancel_orders_by_client_id_prefix(self, prefix):
                return 0

            async def get_open_orders(self):
                return []

            async def get_positions(self):
                return []

            async def get_account(self):
                return {"equity": 1000.0, "available": True}

            async def close(self):
                pass

        broker = IntradayBroker([None, None, True, True, False])
        trader = _make_gate_trader(broker)
        ticks = []
        shutdowns = []

        async def fake_safe_tick(tick_num):
            ticks.append(tick_num)

        trader._safe_tick = fake_safe_tick

        async def fake_shutdown(close_broker=True):
            shutdowns.append(close_broker)

        trader.shutdown = fake_shutdown

        async def noop():
            pass

        trader._sync_positions_from_broker = noop
        trader._post_startup_cleanup = noop
        trader._ensure_protective_stops = noop

        waits = {"n": 0}

        async def fake_wait():
            waits["n"] += 1
            if waits["n"] >= 2:
                raise KeyboardInterrupt  # end the outer session loop

        trader.wait_for_market_open = fake_wait

        monkeypatch.setattr(live_trader, "CHECK_INTERVAL", 0.001)
        monkeypatch.setattr(live_trader, "MARKET_LOOP_UNKNOWN_RETRY_SECONDS", 0.001)
        monkeypatch.setattr(live_trader, "MARKET_TICK_HEARTBEAT_SECONDS", 0.001)

        with caplog.at_level(logging.INFO, logger="live_trader"):
            with pytest.raises(KeyboardInterrupt):
                await asyncio.wait_for(trader.run(), timeout=10.0)

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Market clock UNKNOWN intraday (attempt 1" in text, text
        assert "never treating unknown as closed" in text
        assert ticks == [3, 4]                        # no tick while UNKNOWN
        assert shutdowns == [False]                   # session ended by CONFIRMED close
        assert "Market closed — completing session and waiting for next open" in text
        assert "Tick loop alive — tick" in text       # intraday liveness heartbeat
        assert "STILL WAITING FOR OPEN" not in text    # overdue line is gate-only