#!/usr/bin/env python3
"""
AlgoFlow Live Paper Trading Runner
==================================
Runs the AI-optimized mean reversion strategy on Alpaca paper trading.
Waits for market open, trades throughout the day, shuts down at market close.

Start: python3 live_trader.py
Logs: /home/team/shared/engine/logs/trades_YYYYMMDD.log
"""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── Project imports ────────────────────────────────────────────────
import src.strategies  # registers strategies
from src.data.yfinance_provider import YFinanceProvider
from src.execution.alpaca_broker import AlpacaBroker, account_equity, is_order_alive
from src.execution.broker import Order, OrderSide, OrderType
import time
from src.execution.position_manager import PositionManager
from src.execution.session_state import load_start_equity, save_start_equity
from src.execution.verified_close import close_position_verified
from src.strategies.base import SignalType, StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.indicators import sma

# ── Configuration ──────────────────────────────────────────────────
SYMBOLS = ["NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO"]
CHECK_INTERVAL = 60  # seconds between polls
# ── Market-gate hardening (2026-09-15, PR #35) ────────────────────
# The gate must never wait silently for hours (observed 2026-09-14: a
# whole-container freeze hid the traders' stall for 9h because a
# confirmed-closed clock answer is the ONLY valid reason to defer —
# anything else must retry loudly and FAIL OPEN once local ET passes
# 09:30 + buffer).  Every knob below is env-overridable for drills/tests.
MARKET_GATE_RETRY_SECONDS = float(          # unknown/error backoff    (15-30s)
    os.environ.get("MARKET_GATE_RETRY_SECONDS", "20"))
MARKET_GATE_HEARTBEAT_SECONDS = float(      # confirmed-closed heartbeat
    os.environ.get("MARKET_GATE_HEARTBEAT_SECONDS", "240"))
MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS = float(  # local-ET fail-open buffer
    os.environ.get("MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS", "600"))
MARKET_GATE_OVERDUE_LOG_SECONDS = float(    # loud overdue-open cadence
    os.environ.get("MARKET_GATE_OVERDUE_LOG_SECONDS", "60"))
MARKET_LOOP_UNKNOWN_RETRY_SECONDS = float(  # intraday unknown-clock retry
    os.environ.get("MARKET_LOOP_UNKNOWN_RETRY_SECONDS", "20"))
MARKET_TICK_HEARTBEAT_SECONDS = float(      # intraday tick-loop liveness
    os.environ.get("MARKET_TICK_HEARTBEAT_SECONDS", "240"))
# NOTE: Analysis shows low-confidence trades (0.4-0.6 bucket) average +$33.46 —
# actually MORE profitable than high-confidence trades. More signals = more opportunities.
CONFIDENCE_THRESHOLD = 0.3
MAX_POSITIONS = 6
POSITION_SIZE_PCT = 0.15  # 15% of equity per position

# Market close in UTC (4 PM ET = 20:00 UTC)
MARKET_CLOSE_UTC_HOUR = 20
MARKET_CLOSE_UTC_MINUTE = 0
MANDATORY_CLOSE_MINUTES = 15  # Liquidate all positions 15 min before close
MAX_HOLD_MINUTES = 30          # Max time to hold a position — recycle capital

STRATEGY_CONFIG = StrategyConfig(
    entry_threshold=0.5,       # z-score to enter (lower = more trades)
    exit_threshold=0.1,        # z-score to exit
    stop_loss_pct=0.03,        # 3% stop-loss
    take_profit_pct=0.03,      # 3% take-profit
    max_position_pct=POSITION_SIZE_PCT,
    extra={"lookback": 20, "std_dev_multiplier": 2.0},
)

# ── Regime gate (Recommendation: stop bleeding into falling markets) ──
# The main trader is pure mean reversion — it buys oversold dips, which is
# exactly the wrong thing to do in a confirmed downtrend (falling knives).
# This gate mirrors the turbo trader's `_regime_gate_allows_long`: it SKIPS
# the mean-reversion LONG when price is below the 10-bar MA AND RSI(14) < 40
# (a confirmed downtrend).  Flippable for paper-trading A/B comparison.
ENABLE_REGIME_GATE = True   # False → restore unconditional dip-buying
REGIME_MA_PERIOD = 10       # short MA that defines the trend reference
REGIME_RSI_PERIOD = 14      # RSI period used by the weakness filter
REGIME_RSI_THRESHOLD = 40.0 # RSI below this = oversold/weak → block the long

# ── Broker-side protective stops ────────────────────────────────────
# Every held AND every newly-opened main position also gets a GTC
# stop-loss order at the broker (Alpaca holds it even if every process on
# this machine dies — the whole point: positions must be protected with
# zero processes running).  The in-process 3% risk stop fires first while
# the process is alive; the broker stop is the hard backstop at
# entry − PROTECTIVE_STOP_PCT (6% default, mirroring the turbo trader's
# protective-stop band for the mean-reversion profile).  Configurable via
# MAIN_PROTECTIVE_STOP_PCT (e.g. 0.03 = 3%).
PROTECTIVE_STOP_PCT = float(os.environ.get("MAIN_PROTECTIVE_STOP_PCT", "0.06"))
# Stop-placement retry: submitting the stop while the entry order is still
# open makes Alpaca reject it as a "potential wash trade" (opposite-side
# market/stop order exists), which previously left positions running with
# no broker-side protection.  Same policy as the turbo trader.
STOP_PLACEMENT_MAX_ATTEMPTS = 4
STOP_PLACEMENT_INITIAL_DELAY = 2.0

# ── Logging ────────────────────────────────────────────────────────
log_dir = Path("/home/team/shared/engine/logs")
log_dir.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y%m%d")
log_file = log_dir / f"trades_{today}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("live_trader")


# ── Trading logic ──────────────────────────────────────────────────

def _compute_rsi(close: "pd.Series", period: int = 14) -> "pd.Series":
    """Compute RSI (Relative Strength Index) over *period* bars.

    Mirrors the turbo trader's RSI so both engines use identical regime
    math.  RSI < REGIME_RSI_THRESHOLD means the instrument is weak/oversold.
    """
    import pandas as pd
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _regime_gate_allows_long(
    data: "pd.DataFrame",
    ma_period: int = 10,
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
) -> "tuple[bool, str]":
    """Return ``(allow, reason)`` for a mean-reversion LONG on *data*.

    A long is BLOCKED only in a confirmed downtrend — price below the
    *ma_period*-bar MA AND RSI(*rsi_period*) below *rsi_threshold*.  This is
    the #1 bleeding source on down days: the main trader unconditionally
    buys mean-reversion dips, catching falling knives (e.g. -$351 COIN,
    -$116 AVGO on 2026-08-26).  When either condition is healthy (price
    above the MA, or RSI recovering), the long is allowed — this is a
    filter, not a trend follower.  ``reason`` is human-readable for audit
    logging (empty string when allowed).
    """
    close = data["close"]
    if len(close) < max(ma_period, rsi_period) + 1:
        return True, "insufficient data"
    ma = sma(close, period=ma_period)
    rsi = _compute_rsi(close, period=rsi_period)
    cur = float(close.iloc[-1])
    cur_ma = float(ma.iloc[-1])
    cur_rsi = float(rsi.iloc[-1])
    if pd.isna(cur_ma) or pd.isna(cur_rsi):
        return True, "insufficient indicator history"
    below_ma = cur < cur_ma
    weak_rsi = cur_rsi < rsi_threshold
    if below_ma and weak_rsi:
        return False, (
            f"downtrend: price {cur:.2f} < MA{ma_period} {cur_ma:.2f} "
            f"AND RSI {cur_rsi:.1f} < {rsi_threshold}"
        )
    return True, ""


def _main_stop_client_id(symbol: str) -> str:
    """Generate a unique idempotency key for main protective stop orders.

    Format: ``algoflow_MAIN_{SYMBOL}_STOP_{timestamp_ns}`` — the
    ``algoflow_MAIN_`` prefix keeps stale stops inside the boot-time
    stale-order cancellation window (``cancel_orders_by_client_id_prefix``
    in ``run()``), while the ``_STOP_`` infix distinguishes them from
    entry/exit fill orders.
    """
    return f"algoflow_MAIN_{symbol.upper()}_STOP_{time.monotonic_ns()}"


class LiveTrader:
    def __init__(self):
        self.broker = AlpacaBroker()
        self.provider = YFinanceProvider()
        self.strategy = MeanReversionStrategy(config=STRATEGY_CONFIG)
        self.pm = PositionManager(STRATEGY_CONFIG)
        self._entry_times: dict[str, datetime] = {}  # when each position was opened
        self.day_trades: list[dict] = []
        self.start_equity = 0.0

    # ── Session baseline (start equity) ──────────────────────────────
    def _restore_start_equity(self) -> bool:
        """Reload today's persisted start baseline (stable across restarts)."""
        equity = load_start_equity()
        if equity is not None and equity > 0:
            self.start_equity = equity
            return True
        return False

    def _set_start_equity(self, equity: float) -> None:
        """Record the day baseline and persist it for watchdog restarts."""
        self.start_equity = float(equity)
        if not save_start_equity(self.start_equity):
            logger.warning("Could not persist session start equity to %s",
                           "logs/session_state.json")

    async def _begin_session(self, assumed: bool) -> bool:
        """Establish the day's P&L baseline; return True when ready to trade."""
        if self.start_equity is None or self.start_equity <= 0:
            if self._restore_start_equity():
                logger.info("✅ %s — resuming session (baseline reloaded)",
                            "Market is OPEN" if not assumed else "Market assumed OPEN")
                return True
            account = await self.broker.get_account()
            equity = account_equity(account)
            if equity is None or equity <= 0:
                logger.warning("Account equity unavailable — cannot establish session baseline")
                return False
            self._set_start_equity(equity)
        logger.info("✅ %s — starting trading",
                    "Market is OPEN" if not assumed else "Market assumed OPEN")
        logger.info("   Starting equity: ${:,.2f}".format(self.start_equity))
        return True

    def _market_gate_status_name(self, market_open: bool | None) -> str:
        """Human-readable status for gate logs: OPEN / CONFIRMED-CLOSED /
        UNKNOWN.  ``None`` is UNKNOWN — an indeterminate clock that must
        NEVER be treated as a confirmed close."""
        if market_open is True:
            return "OPEN"
        if market_open is False:
            return "CONFIRMED-CLOSED"
        return "UNKNOWN"

    async def _market_clock(self, context: str) -> tuple[bool | None, int]:
        """One broker clock check with per-attempt logging.

        ``context`` names the call site (``pre-open`` / ``intraday``) so
        attempt counters and last errors are kept separately.  Returns
        ``(status, attempt_number)``; status ``None`` means indeterminate
        (timeout/exception) and is logged loudly with the last error —
        never a silent retry.
        """
        if not hasattr(self, "_clock_attempts"):
            self._clock_attempts: dict[str, int] = {}
            self._clock_last_error: dict[str, str | None] = {}
        self._clock_attempts[context] = self._clock_attempts.get(context, 0) + 1
        attempt = self._clock_attempts[context]
        try:
            status = await self.broker.is_market_open()
        except Exception as exc:  # network errors, SDK failures — indeterminate
            self._clock_last_error[context] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Market clock check FAILED (%s — attempt %d): %s — treating as UNKNOWN, will retry",
                context, attempt, self._clock_last_error[context],
            )
            return None, attempt
        self._clock_last_error[context] = None
        return status, attempt

    async def wait_for_market_open(self):
        """Wait for 9:30 ET using local DST-aware time as the primary gate.

        Hardened 2026-09-15 (PR #35): ONLY a CONFIRMED-CLOSED Alpaca clock
        defers.  ``None``/exception is indeterminate and is retried on a
        SHORT backoff with one log line per attempt (attempt N, last
        error) — a silent multi-hour wait is impossible.  Once local ET is
        >= 09:30 + buffer and the last status was NOT confirmed-closed,
        the gate FAILS OPEN on local time.  While RTH per the local DST-
        aware schedule is open but we are still waiting, a loud overdue
        log fires every 60s so a stuck gate is glaring.
        """
        from zoneinfo import ZoneInfo
        from src.watchdog.market_status import in_rth_schedule
        logger.info("Waiting for market to open (9:30 AM ET)...")
        last_heartbeat = time.monotonic()
        last_overdue = time.monotonic()
        retry_seconds = MARKET_GATE_RETRY_SECONDS
        heartbeat_interval = MARKET_GATE_HEARTBEAT_SECONDS
        fail_open_buffer = timedelta(seconds=MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS)
        overdue_cadence = MARKET_GATE_OVERDUE_LOG_SECONDS
        last_status: bool | None = None  # None = unknown (NOT confirmed-closed)
        while True:
            seconds_until = self._seconds_until_open()
            now_et = datetime.now(ZoneInfo("America/New_York"))
            before_open = now_et.weekday() < 5 and (now_et.hour, now_et.minute) < (9, 30)
            if seconds_until > 0 and (before_open or now_et.weekday() >= 5 or now_et.hour >= 16):
                # Sleep until shortly before the DST-aware opening instant;
                # cap it so heartbeat messages remain useful.
                sleep_for = min(max(seconds_until - 60.0, 1.0), heartbeat_interval)
                await asyncio.sleep(sleep_for)
                # Heartbeat: pre-open / weekend waits produce no trading
                # output for hours, so emit a low-frequency marker that
                # keeps the watchdog's staleness check truthful.
                if time.monotonic() - last_heartbeat >= heartbeat_interval:
                    mins_left = max(int(self._seconds_until_open() // 60), 0)
                    logger.info(
                        "heartbeat: waiting for market open, ~%dm remaining",
                        mins_left,
                    )
                    last_heartbeat = time.monotonic()
                continue
            # ── At/past the calculated open: ask the Alpaca clock. ──
            status, attempt = await self._market_clock("pre-open")
            if status is None:
                # Indeterminate clock — loud per-attempt retry, never silent.
                logger.warning(
                    "Market clock UNKNOWN (attempt %d, last error: %s) — "
                    "retrying in %.0fs — NOT a confirmed close",
                    attempt, self._clock_last_error.get("pre-open"), retry_seconds,
                )
            last_status = status
            status_name = self._market_gate_status_name(status)
            # ── Loud overdue log while RTH says open but we still wait. ──
            # This is the missing observability that let the 2026-09-14
            # stall rot for 9 hours: a stuck gate is now impossible to
            # miss.  Fires for ANY status (closed/unknown) — every 60s.
            if in_rth_schedule(now_et):
                mins_past = max(
                    (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30), 0
                )
                if time.monotonic() - last_overdue >= overdue_cadence:
                    open_instant = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                    fail_in = max(
                        int((open_instant + fail_open_buffer - now_et).total_seconds()), 0
                    )
                    logger.warning(
                        "⚠️ STILL WAITING FOR OPEN %d MINUTES PAST 09:30 ET "
                        "(market_open=%s, attempt=%d) — FAIL-OPEN in %ds",
                        mins_past, status_name, attempt, fail_in,
                    )
                    last_overdue = time.monotonic()
            # ── Confirmed open → begin session. ──
            if status is True:
                if await self._begin_session(assumed=False):
                    return
            # ── FAIL-OPEN: past 09:30+buffer and last status was NOT
            #    confirmed-closed → proceed on local time. ──
            elif status is not False:
                open_instant = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                if now_et >= open_instant + fail_open_buffer:
                    logger.warning(
                        "⚠️ FAIL-OPEN at %s ET: local time past 09:30+%ds buffer and "
                        "last clock status was %s (NOT confirmed-closed, attempt %d) "
                        "— proceeding on local time",
                        now_et.strftime("%H:%M:%S"),
                        int(MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS),
                        status_name, attempt,
                    )
                    if await self._begin_session(assumed=True):
                        return
            # status False → confirmed closed → keep waiting (heartbeat).
            if time.monotonic() - last_heartbeat >= heartbeat_interval:
                logger.info(
                    "Still waiting for market open, next check in %.0fs (clock confirmation)",
                    retry_seconds,
                )
                last_heartbeat = time.monotonic()
            await asyncio.sleep(retry_seconds)
    @staticmethod
    def _seconds_until_open() -> float:
        """Seconds until next 9:30 AM ET market open (DST-aware).

        Market open is 9:30 AM in the America/New_York timezone, which is
        13:30 UTC in summer (EDT, UTC-4) but 14:30 UTC in winter (EST,
        UTC-5).  The old hardcoded 13:30 UTC target was only correct during
        DST.  Computing the target in the NY zone keeps it right year-round.
        """
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        now = datetime.now(ny)
        target = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.weekday() >= 5:  # weekend
            days_until_mon = (7 - now.weekday()) % 7
            target += timedelta(days=days_until_mon)
        elif now > target:
            target += timedelta(days=1)
            if target.weekday() >= 5:
                days_until_mon = (7 - target.weekday()) % 7
                target += timedelta(days=days_until_mon)
        return (target - now).total_seconds()

    async def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("AlgoFlow Live Trader — STARTING")
        logger.info(f"Symbols: {SYMBOLS}")
        logger.info(f"Strategy: MeanReversion | Confidence ≥ {CONFIDENCE_THRESHOLD}")
        logger.info(f"Max positions: {MAX_POSITIONS} | Size: {POSITION_SIZE_PCT*100:.0f}% equity")
        logger.info("=" * 60)

        try:
            await self.broker.startup_health_check()
        except Exception as exc:
            logger.critical("FATAL: broker authentication/account health check failed; refusing to trade: %s", exc)
            return

        # ── Layer 4: Cancel stale orders from prior sessions ─────────
        logger.info("Cancelling any stale orders from prior sessions…")
        cancelled = await self.broker.cancel_orders_by_client_id_prefix("algoflow_MAIN_")
        logger.info(f"Cancelled {cancelled} stale order(s)")
        remaining = await self.broker.get_open_orders()
        stale_cancel_unconfirmed = any(
            str(getattr(o, "client_order_id", "")).startswith("algoflow_MAIN_")
            for o in remaining
        )
        if stale_cancel_unconfirmed:
            # A leftover order the broker pins as "pending cancel" (e.g. a
            # cancel was in flight when a previous host died) answers every
            # cancel attempt with 42210000 and can stay visible for days.
            # That must NOT kill the boot: the order is already being
            # cancelled at the broker, so we skip only the post-startup
            # position-cleanup step (its intent was to liquidate leftovers
            # whose orders we expected to be gone) and continue the normal
            # boot — sync positions, re-place protective stops, wait for
            # market. Mirror of the turbo trader's tolerant handling.
            logger.warning(
                "Stale order cancellation was not confirmed — deferring post-startup "
                "position cleanup; continuing boot (positions will be synced and "
                "protective stops re-placed)"
            )

        # ── Layer 2: Sync positions from Alpaca at startup ───────────
        logger.info("STEP 1/3: Syncing positions from broker…")
        await self._sync_positions_from_broker()
        logger.info("STEP 1/3: Position sync complete — %d open positions tracked",
                     self.pm.get_open_count())

        # ── Layer 2b: Log inherited position state ────────────────────
        for sym in self.pm.get_open_symbols():
            pos = self.pm.get_positions().get(sym)
            if pos:
                logger.info("  Inherited: %s x %s @ $%.2f", pos.quantity, sym, pos.entry_price)

        # ── Post-startup stale position cleanup ──────────────────────
        if stale_cancel_unconfirmed:
            logger.warning(
                "Post-startup stale position cleanup DEFERRED (stale order "
                "cancellation unconfirmed)"
            )
        else:
            await self._post_startup_cleanup()

        # ── Ensure every held position has a broker-side GTC stop ────
        # Placed right here — immediately after sync — so positions are
        # protected at the broker even if every process on this box dies
        # minutes later (the exact failure mode this trader must survive).
        logger.info("STEP 1/3: Ensuring protective stops on held positions…")
        await self._ensure_protective_stops()

        # Keep the process alive between sessions.  Each iteration waits for
        # the next market open, trades one session, reports its summary, then
        # resets local state before waiting for the following trading day.
        while True:
            await self.wait_for_market_open()
            logger.info("STEP 2/3: Entering main tick loop…")
            tick = 0
            self._tick_loop_last_heartbeat = time.monotonic()
            try:
                while True:
                    tick += 1
                    # ── EOD mandatory liquidation check ──────────────────
                    try:
                        if self._is_near_close():
                            logger.info(f"⏰ Within {MANDATORY_CLOSE_MINUTES} min of close — "
                                        "triggering mandatory EOD liquidation")
                            await self._eod_liquidate()
                            break
                    except Exception:
                        logger.exception("EOD check/liquidate failed — continuing")
                    # ── Intraday market check (same hardening as the gate) ──
                    # None/exception = indeterminate clock: NEVER treated as
                    # a confirmed close.  Retry loudly with per-attempt
                    # logging on a short backoff — a transient API outage
                    # cannot trigger a false mid-session liquidation, and a
                    # stuck clock cannot be silent.
                    try:
                        market_open, attempt = await self._market_clock("intraday")
                    except Exception:
                        logger.exception("Market-open check failed — treating as unknown, will retry")
                        market_open, attempt = None, self._clock_attempts.get("intraday", 0)
                    if market_open is None:
                        logger.warning(
                            "Market clock UNKNOWN intraday (attempt %d, last error: %s) "
                            "— retrying in %.0fs — never treating unknown as closed",
                            attempt, self._clock_last_error.get("intraday"),
                            MARKET_LOOP_UNKNOWN_RETRY_SECONDS,
                        )
                        await asyncio.sleep(MARKET_LOOP_UNKNOWN_RETRY_SECONDS)
                        continue
                    if not market_open:
                        logger.info("⏹️  Market closed — completing session and waiting for next open")
                        break
                    await self._safe_tick(tick)
                    # Intraday liveness heartbeat: a healthy session can be
                    # INFO-silent for hours (no signals → no order logs),
                    # which made watchdog staleness useless intraday
                    # (2026-09-14: a 9h freeze produced zero lines during
                    # RTH).  One line per cadence proves the loop is alive.
                    if time.monotonic() - self._tick_loop_last_heartbeat >= MARKET_TICK_HEARTBEAT_SECONDS:
                        logger.info(
                            "❤️ Tick loop alive — tick %d, market OPEN (clock confirmed)",
                            tick,
                        )
                        self._tick_loop_last_heartbeat = time.monotonic()
                    logger.debug("Tick %d: complete — sleeping %ds", tick, CHECK_INTERVAL)
                    await asyncio.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                await self.shutdown()
                return
            except Exception as e:
                logger.exception("FATAL: Unhandled exception in main loop — %s", e)

            # Preserve the session summary while keeping the broker connection
            # alive for the next session. Positions should already be flat from
            # EOD liquidation; shutdown remains a safety net if not.
            await self.shutdown(close_broker=False)
            self.pm.reset()
            self._entry_times.clear()
            self.day_trades.clear()
            logger.info("Session state reset — waiting for next market open")

    async def _safe_tick(self, tick_num: int):
        """Wrapper around _tick that catches all exceptions so one bad tick
        never kills the trader.  Logs the full traceback and continues."""
        try:
            await self._tick(tick_num)
        except Exception:
            logger.exception("Tick %d crashed — continuing to next tick", tick_num)
            await asyncio.sleep(CHECK_INTERVAL)

    async def _tick(self, tick_num: int):
        """One polling cycle: fetch → signals → execute."""
        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=30)

        logger.debug("Tick %d: evaluating %d symbols (%d positions open)",
                     tick_num, len(SYMBOLS), self.pm.get_open_count())

        for symbol in SYMBOLS:
            # Skip if at max positions and don't hold this one
            if self.pm.get_open_count() >= MAX_POSITIONS and not self.pm.has_position(symbol):
                continue

            try:
                logger.debug("Tick %d: fetching %s 1m bars…", tick_num, symbol)
                mdf = await self.provider.fetch_bars(
                    symbol, start=lookback, end=now, timeframe="1min"
                )
                data = mdf.df
                if data.empty:
                    continue

                signals = self.strategy.generate_signals(data)
                if not signals:
                    continue

                latest = signals[-1]
                if latest.confidence < CONFIDENCE_THRESHOLD:
                    continue

                current_price = float(data["close"].iloc[-1])

                if latest.signal_type == SignalType.BUY:
                    # ── Regime gate: skip the mean-reversion LONG in a
                    #    confirmed downtrend (price < MA10 AND RSI < 40) so we
                    #    stop catching falling knives.  This is the #1 bleeding
                    #    source on down days. ──
                    if ENABLE_REGIME_GATE:
                        allow, reason = _regime_gate_allows_long(
                            data,
                            ma_period=REGIME_MA_PERIOD,
                            rsi_period=REGIME_RSI_PERIOD,
                            rsi_threshold=REGIME_RSI_THRESHOLD,
                        )
                        if not allow:
                            logger.info(
                                "🚫 REGIME GATE %s: skipping mean-reversion LONG — %s",
                                symbol, reason,
                            )
                            continue
                    await self._handle_buy(symbol, current_price, latest.confidence)
                elif latest.signal_type == SignalType.SELL:
                    await self._handle_sell(symbol, current_price, latest.confidence)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Check stop-loss / take-profit
        await self._check_risk_stops()

    async def _handle_buy(self, symbol: str, price: float, confidence: float):
        if self.pm.has_position(symbol):
            return
        if self.pm.get_open_count() >= MAX_POSITIONS:
            return

        account = await self.broker.get_account()
        equity = account_equity(account)
        if equity is None:
            logger.warning(f"⚠️  {symbol}: account equity unavailable — skipping entry")
            return
        if not self.pm.can_open(symbol, equity):
            return

        value = equity * POSITION_SIZE_PCT
        qty = value / price if price > 0 else 0
        if qty < 1:
            return

        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET,
                      client_id=f"algoflow_MAIN_{symbol.upper()}_BUY_{time.monotonic_ns()}")
        result = await self.broker.place_order(order)

        if is_order_alive(result.status):
            self.pm.open_position(symbol, qty, price)
            self._entry_times[symbol.upper()] = datetime.now(timezone.utc)  # time-based exit
            logger.info(f"📈 BUY  {symbol}: {qty:.1f} shares @ ${price:.2f} = ${value:,.2f} | "
                        f"conf={confidence:.2f} | order={result.order_id[:8]}")
            # ── Place GTC protective stop at broker ─────────────────
            # The stop-placement retry machinery handles the case where the
            # entry BUY is still open (wash-trade reject) by waiting and
            # retrying; the position is never left broker-unprotected.
            await self._place_protective_stop(symbol, qty, price)
        else:
            logger.warning(f"❌ BUY {symbol} REJECTED: {result.status}")

    async def _handle_sell(self, symbol: str, price: float, confidence: float):
        if not self.pm.has_position(symbol):
            return

        pos = self.pm.get_positions().get(symbol.upper())
        if pos is None:
            return
        abs_qty = abs(pos.quantity)
        is_short = pos.quantity < 0

        # ── Cancel the GTC protective stop before closing ────────────
        # Otherwise the stop stays open (GTC) after the position closes and
        # could later trigger as an accidental short.  If the cancellation
        # is not confirmed, defer the close — the position remains tracked
        # AND protected, the safest state.
        if not await self._cancel_protective_stops(symbol):
            logger.warning(
                "SELL %s deferred: protective-order cancellation was not confirmed",
                symbol,
            )
            return

        outcome = await close_position_verified(
            self.pm,
            self.broker,
            symbol,
            exit_reason="signal",
            client_id=f"algoflow_MAIN_{symbol.upper()}_SELL_{time.monotonic_ns()}",
        )
        if outcome.status == "filled" and outcome.trade is not None:
            self._entry_times.pop(symbol.upper(), None)
            trade = outcome.trade
            logger.info(
                f"📉 SELL {symbol}: {abs(trade.quantity):.1f} shares @ ${outcome.fill_price:.2f} | "
                f"P&L: ${trade.pnl:,.2f} | conf={confidence:.2f} | order={outcome.order_id[:8]}"
            )
        elif outcome.status == "pending":
            # The close order is live but the broker has not confirmed a fill.
            # We keep the position tracked and book NO P&L; the next position
            # sync reconciles the fill from broker fill history.
            logger.info(
                f"⏳ SELL {symbol}: close pending (fill not confirmed) — no P&L booked | "
                f"conf={confidence:.2f}"
            )
        elif outcome.status == "rejected":
            error = (outcome.message or "").lower()
            if "cannot be sold short" in error:
                self.pm.discard_position(symbol, reason="broker says position is not held")
                self._entry_times.pop(symbol.upper(), None)
                logger.warning("SELL %s rejected as phantom position; removed from tracking", symbol)
            else:
                # The protective stop was cancelled above — restore it so the
                # position doesn't run naked because of a rejected exit.
                logger.warning(
                    "SELL %s rejected (%s); keeping position tracked — restoring protective stop",
                    symbol, outcome.message,
                )
                await self._place_protective_stop(symbol, abs_qty, pos.entry_price, is_short=is_short)

    async def _sync_positions_from_broker(self):
        """Reconcile local PositionManager with Alpaca's actual positions.

        - New positions on Alpaca that we don't know about → add to PM.
        - Positions in PM that Alpaca doesn't have → remove from PM (stale).
        """
        logger.info("Syncing positions from broker…")
        try:
            broker_positions = await self.broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to fetch broker positions for sync: {e}")
            return

        broker_symbols = {p["symbol"].upper(): p for p in broker_positions}
        pm_symbols = set(self.pm.get_open_symbols())

        added = 0
        removed = 0

        # Add positions the broker knows about but we don't
        for sym, pos_data in broker_symbols.items():
            if sym not in pm_symbols:
                self.pm.open_position(
                    symbol=sym,
                    quantity=pos_data["qty"],
                    entry_price=pos_data["avg_entry_price"],
                )
                logger.info(
                    "  + Added %s: %s shares @ $%.2f (sync)",
                    sym, pos_data["qty"], pos_data["avg_entry_price"],
                )
                added += 1

        # Remove positions we track but broker doesn't have.  A tracked
        # position that vanishes from the broker (e.g. a pending cleanup
        # MARKET SELL that filled in the background, or an external sale) is
        # booked at the broker's true last fill price.  If no fill price can
        # be determined, the position is DROPPED without booking P&L — never
        # fabricate a mark-based loss.
        for sym in pm_symbols - set(broker_symbols):
            fill_price = await self.broker.get_last_fill_price(sym)
            if fill_price is not None:
                self.pm.close_position(sym, exit_price=fill_price, exit_reason="sync_removed")
                logger.info(
                    "  - Removed %s (not on broker) — P&L booked at broker fill $%.2f",
                    sym, fill_price,
                )
            else:
                self.pm.discard_position(sym, reason="sync_removed_no_fill")
                logger.info(
                    "  - Removed stale %s (not on broker) — no fill price available; "
                    "dropped WITHOUT booking P&L",
                    sym,
                )
            removed += 1

        logger.info(
            "Synced %d positions from broker (%d added, %d removed)",
            len(broker_symbols), added, removed,
        )

    # ── Post-startup stale position cleanup ────────────────────────────

    async def _post_startup_cleanup(self):
        """Liquidate stale positions if trader starts while market is closed.

        If the market is closed (e.g. after a crash/restart/sandbox cycle
        that missed the regular EOD window) and there are open positions
        at the broker, liquidate them immediately so nothing hangs overnight.
        """
        try:
            try:
                market_open = await self.broker.is_market_open()
            except Exception as exc:
                # Indeterminate clock (timeout/error) — do NOT liquidate on
                # an unknown clock.  Only a CONFIRMED close may trigger
                # cleanup, or we could sell into a live tape.
                logger.warning(
                    "🧹 Post-close cleanup skipped: market clock UNKNOWN (%s) — "
                    "only a confirmed close may liquidate", exc,
                )
                return
            if market_open is not False:
                if market_open is None:
                    logger.info(
                        "🧹 Post-close cleanup skipped: market clock UNKNOWN — "
                        "only a confirmed close may liquidate",
                    )
                return  # Market is open (True) or unknown (None) — no cleanup

            positions = await self.broker.get_positions()
            if not positions:
                return

            logger.info(
                "🧹 Post-close cleanup: liquidating %d stale position(s) from previous session",
                len(positions),
            )
            liquidated = 0
            for p in positions:
                sym = p.get("symbol")
                qty = float(p.get("qty", 0))
                if qty > 0:
                    # Cancel the GTC protective stop before liquidating so it
                    # can't outlive the position.
                    if not await self._cancel_protective_stops(sym):
                        logger.warning("🧹 Cleanup SELL %s deferred: cancellation not confirmed", sym)
                        continue
                    logger.info("🧹 Cleanup SELL %s: %s shares", sym, qty)
                    outcome = await close_position_verified(
                        self.pm, self.broker, sym,
                        exit_reason="post_close_cleanup",
                    )
                    if outcome.status == "filled":
                        liquidated += 1
                    elif outcome.status == "pending":
                        logger.info(
                            "🧹 Cleanup SELL %s: close pending (fill not confirmed) — "
                            "no P&L booked, will reconcile at next sync",
                            sym,
                        )
                    else:
                        logger.warning(
                            "🧹 Cleanup SELL %s rejected (%s); restoring protective stop and keeping position",
                            sym, outcome.message,
                        )
                        await self._place_protective_stop(
                            sym, qty, float(p.get("avg_entry_price", 0)) or 0,
                        )
            logger.info(
                "🧹 Post-close cleanup complete — %d position(s) liquidated",
                liquidated,
            )
        except Exception:
            logger.exception("🧹 Post-close cleanup failed — continuing startup")

    # ── EOD mandatory liquidation ─────────────────────────────────────

    @staticmethod
    def _is_near_close() -> bool:
        """Return True if we are within MANDATORY_CLOSE_MINUTES of market close.

        Market close is 20:00 UTC (4 PM ET).  We trigger mandatory liquidation
        MANDATORY_CLOSE_MINUTES before that.
        """
        now = datetime.now(timezone.utc)
        close_time = now.replace(
            hour=MARKET_CLOSE_UTC_HOUR,
            minute=MARKET_CLOSE_UTC_MINUTE,
            second=0,
            microsecond=0,
        )
        seconds_until_close = (close_time - now).total_seconds()
        return 0 <= seconds_until_close <= (MANDATORY_CLOSE_MINUTES * 60)

    async def _eod_liquidate(self):
        """Liquidate ALL broker positions for mandatory end-of-day close."""
        positions = await self.broker.get_positions()
        count = len(positions)
        if count == 0:
            logger.info("⏰ Mandatory EOD liquidation — no positions to close")
            return

        liquidated = 0
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                # ── Cancel the GTC protective stop before liquidating ──
                # If the cancellation is not confirmed, defer: the position
                # stays tracked AND protected overnight — safer than a naked
                # position or an orphaned stop.
                if not await self._cancel_protective_stops(sym):
                    logger.warning("⏰ EOD SELL %s deferred: cancellation not confirmed", sym)
                    continue
                logger.info(f"⏰ EOD closing {sym}: {qty} shares...")
                outcome = await close_position_verified(
                    self.pm, self.broker, sym, exit_reason="eod",
                )
                if outcome.status == "filled":
                    liquidated += 1
                elif outcome.status == "pending":
                    logger.info(
                        "⏰ EOD SELL %s: close pending (fill not confirmed) — no P&L booked",
                        sym,
                    )
                else:
                    logger.warning(
                        "⏰ EOD SELL %s rejected (%s); restoring protective stop and keeping position",
                        sym, outcome.message,
                    )
                    await self._place_protective_stop(
                        sym, qty, float(p.get("avg_entry_price", 0)) or 0,
                    )

        logger.info(f"⏰ Mandatory EOD liquidation — {liquidated} positions closed.")

    # ── Broker-level protective stops ────────────────────────────────

    async def _place_protective_stop(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        max_attempts: int = STOP_PLACEMENT_MAX_ATTEMPTS,
        initial_delay: float = STOP_PLACEMENT_INITIAL_DELAY,
        is_short: bool = False,
    ) -> bool:
        """Place a GTC protective stop-loss order at the broker.

        This order survives process death and sandbox cycling — Alpaca holds
        it until triggered or cancelled.  For LONG positions the stop is a
        SELL at ``entry_price * (1 - PROTECTIVE_STOP_PCT)`` (6% below entry);
        for SHORT positions it is a BUY at ``entry_price * (1 + pct)`` (6%
        ABOVE entry — a short loses money when price rises).  GTC orders
        require whole shares, so *qty* is floored to an integer (the
        fractional remainder is still covered by in-process risk checks).

        Retries with backoff: submitting the stop while the entry order is
        still open makes Alpaca reject it as a "potential wash trade"
        (``opposite side market/stop order exists``).  Each attempt
        re-checks the symbol's open orders so the stop is never submitted
        while an opposite-side order is live.
        """
        sym = symbol.upper()
        qty = int(qty)
        if qty <= 0:
            logger.warning("🛡️  STOP %s: position too small for protective stop (qty < 1 share)", sym)
            return False

        stop_loss_pct = PROTECTIVE_STOP_PCT
        stop_price = round(
            entry_price * (1 + stop_loss_pct) if is_short
            else entry_price * (1 - stop_loss_pct),
            2,
        )
        stop_side = "BUY" if is_short else "SELL"
        opposite_open = "BUY" if is_short else "SELL"

        for attempt in range(1, max_attempts + 1):
            # ── Re-check the symbol's open orders before each attempt ──
            # Another process may have placed a stop order during the
            # entry-to-stop window, and an open opposite-side order would
            # make the stop bounce off Alpaca's wash-trade filter.
            try:
                existing = await self.broker.get_open_orders(symbol=sym)
                # For a long, an existing SELL order means the stop is already
                # there.  For a short, an existing BUY order plays that role
                # (the short's entry is a SELL, so it can't be confused).
                if any(str(getattr(o, "side", "")).upper().endswith(stop_side) for o in existing):
                    logger.info("🛡️  STOP %s: existing %s order found; not submitting duplicate",
                                sym, stop_side)
                    return True
                if any(str(getattr(o, "side", "")).upper().endswith(opposite_open) for o in existing):
                    if attempt < max_attempts:
                        logger.info(
                            "🛡️  STOP %s: entry %s still open — waiting %.0fs before retry (%d/%d)",
                            sym, opposite_open, initial_delay, attempt, max_attempts,
                        )
                        await asyncio.sleep(initial_delay)
                        continue
            except Exception as exc:
                logger.warning(
                    "🛡️  STOP %s: cannot verify open orders (attempt %d/%d): %s",
                    sym, attempt, max_attempts, exc,
                )

            client_id = _main_stop_client_id(sym)
            try:
                await self.broker.place_stop_order(
                    symbol=sym,
                    qty=qty,
                    stop_price=stop_price,
                    client_id=client_id,
                    side=stop_side,
                )
                direction = "+" if is_short else "-"
                logger.info(
                    "🛡️  STOP %s: GTC %s stop-loss at $%.2f (entry=%.2f, %s%.0f%%)",
                    sym, stop_side, stop_price, entry_price, direction,
                    stop_loss_pct * 100,
                )
                return True
            except Exception as exc:
                if attempt < max_attempts:
                    delay = initial_delay * attempt
                    logger.warning(
                        "🛡️  STOP %s: placement rejected (attempt %d/%d) — %s; retrying in %.0fs",
                        sym, attempt, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "🛡️  STOP %s: FAILED after %d attempts — %s. Position has NO "
                        "broker-level stop; in-process risk checks still active.",
                        sym, max_attempts, exc,
                    )
        return False

    async def _cancel_protective_stops(self, symbol: str) -> bool:
        """Cancel all open orders for *symbol* (protective stops).

        Called before closing a position so the GTC stop doesn't remain
        open after the position is gone (an orphaned SELL stop could later
        trigger as an accidental short).  Returns True only when every
        previously-open order for the symbol is confirmed gone.
        """
        sym = symbol.upper()
        try:
            open_orders = await self.broker.get_open_orders(symbol=sym)
        except Exception as exc:
            logger.warning("Failed to fetch open orders for %s: %s", sym, exc)
            return False

        cancelled = 0
        for o in open_orders:
            try:
                if await self.broker.cancel_order_and_wait(str(o.id)):
                    cancelled += 1
                    logger.debug("  Cancelled order %s for %s", str(o.id)[:8], sym)
                else:
                    logger.warning("  Cancellation not confirmed for order %s", str(o.id)[:8])
            except Exception as exc:
                logger.warning("  Failed to cancel order %s: %s", str(o.id)[:8], exc)

        if cancelled:
            logger.info("🗑️  Cancelled %d protective order(s) for %s", cancelled, sym)
        remaining = await self.broker.get_open_orders()
        remaining_ids = {str(getattr(order, "id", "")) for order in remaining}
        return not any(str(o.id) in remaining_ids for o in open_orders)

    async def _ensure_protective_stops(self):
        """Ensure every inherited main position has a GTC protective stop.

        Called right after ``_sync_positions_from_broker()`` at startup, so
        positions that survived a process-group kill / sandbox cycle are
        protected within seconds of boot.  For each tracked main symbol we
        check whether a stop order already exists at Alpaca; if not, we
        place a fresh one (entry − PROTECTIVE_STOP_PCT).  Only symbols in
        the main trader's ``SYMBOLS`` set are touched: the account is shared
        with the turbo trader, and we must never place our own stops on its
        leveraged-ETF positions.
        """
        main_set = {s.upper() for s in SYMBOLS}
        if not self.pm.get_open_symbols():
            logger.info("🛡️  No inherited positions — skipping protective stop check")
            return

        # Fetch all open orders once so we can check stop coverage
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            logger.warning("Cannot verify protective stops — order fetch failed: %s", exc)
            return

        # Build a set of symbols that already have an open stop order
        # (SELL stop for longs, BUY stop for shorts — either means covered).
        covered_symbols: set[str] = set()
        for o in open_orders:
            o_sym = str(o.symbol).upper()
            o_side = str(o.side).upper()
            if o_sym in main_set and o_side in ("SELL", "BUY"):
                covered_symbols.add(o_sym)

        for sym in list(self.pm.get_open_symbols()):
            if sym not in main_set:
                continue
            pos = self.pm.get_positions().get(sym)
            if pos is None:
                continue

            if sym in covered_symbols:
                logger.info("🛡️  %s: existing stop order found — covered", sym)
                continue

            is_short = pos.quantity < 0
            logger.warning(
                "🛡️  %s: NO protective stop found for inherited position "
                "(%s shares @ $%.2f%s) — placing one now",
                sym, abs(pos.quantity), pos.entry_price, " [short]" if is_short else "",
            )
            await self._place_protective_stop(
                sym, int(abs(pos.quantity)), pos.entry_price, is_short=is_short,
            )

    async def _check_risk_stops(self):
        """Check stop-loss / take-profit for open positions."""
        for symbol in list(self.pm.get_open_symbols()):
            if not self.pm.has_position(symbol):
                continue
            try:
                mdf = await self.provider.fetch_bars(
                    symbol,
                    start=datetime.now(timezone.utc) - timedelta(minutes=5),
                    end=datetime.now(timezone.utc),
                    timeframe="1min",
                )
                if mdf.df.empty:
                    continue
                price = float(mdf.df["close"].iloc[-1])
                pos = self.pm.get_positions().get(symbol.upper())
                if pos is None:
                    continue
                entry = pos.entry_price
                if entry == 0:
                    continue

                change_pct = (price - entry) / entry

                # ── Time-based exit: recycle capital after MAX_HOLD_MINUTES ──
                entry_time = self._entry_times.get(symbol.upper())
                held_minutes = 0.0
                if entry_time is not None:
                    held_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60

                if held_minutes >= MAX_HOLD_MINUTES:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.info(f"⏰ TIME-EXIT {symbol}: held {held_minutes:.0f}min, P&L={change_pct*100:+.1f}%")
                    continue

                if change_pct <= -STRATEGY_CONFIG.stop_loss_pct:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.warning(f"🛑 STOP-LOSS {symbol}: -{abs(change_pct)*100:.1f}%")
                elif change_pct >= STRATEGY_CONFIG.take_profit_pct:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.info(f"🎯 TAKE-PROFIT {symbol}: +{change_pct*100:.1f}%")
            except Exception as e:
                logger.error(f"Risk check error {symbol}: {e}")

    async def shutdown(self, close_broker: bool = True):
        """Close positions and report P&L; optionally retain broker for next session."""
        logger.info("Shutting down..." )

        # Close all positions
        positions = await self.broker.get_positions()
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                logger.info(f"Closing {sym}: {qty} shares...")
                # Cancel the GTC protective stop so it can't outlive the
                # position (best-effort on this path — shutdown is a last
                # resort; positions should already be flat from EOD).
                try:
                    await self._cancel_protective_stops(sym)
                except Exception:
                    logger.exception("Failed to cancel protective stop for %s during shutdown", sym)
                order = Order(symbol=sym, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
                await self.broker.place_order(order)

        # Final account state — NEVER fabricate an end equity.  A timed-out
        # fetch returns the unavailable sentinel (equity None); we then print
        # "UNKNOWN" instead of computing a bogus -100% from 0.0.
        account = await self.broker.get_account()
        end_equity = account_equity(account)
        start_equity = self.start_equity
        if start_equity is None or start_equity <= 0:
            persisted = load_start_equity()
            if persisted is not None:
                start_equity = persisted
        logger.info("=" * 60)
        logger.info(f"SESSION COMPLETE")
        if start_equity is None or start_equity <= 0:
            logger.info("Start: UNKNOWN (no baseline captured)")
        else:
            logger.info(f"Start: ${start_equity:,.2f}")
        if end_equity is None:
            logger.info("End:   UNKNOWN (account fetch failed)")
            logger.info("P&L:   UNKNOWN (account fetch timed out)")
        elif start_equity is None or start_equity <= 0:
            logger.info(f"End:   ${end_equity:,.2f}")
            logger.info("P&L:   UNKNOWN (no session start baseline)")
        else:
            pnl = end_equity - start_equity
            pnl_pct = (pnl / start_equity * 100)
            logger.info(f"End:   ${end_equity:,.2f}")
            logger.info(f"P&L:   ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Log:   {log_file}")
        logger.info("=" * 60)
        # Reset the day baseline so the next session re-captures it.  The
        # persisted state file is date-stamped, so it is ignored on a new day
        # and re-used on a same-day restart (stable day P&L).
        self.start_equity = 0.0
        if close_broker:
            await self.broker.close()


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.run())
