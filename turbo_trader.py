#!/usr/bin/env python3
"""
AlgoFlow TURBO Live Paper Trading Runner
========================================
Aggressive trading mode for leveraged ETFs (3x) targeting 20-100%+ returns
per trade on small (~$100) tester accounts.

Runs dual strategy (mean reversion + momentum), wider stops, mandatory
EOD liquidation, and separate logging.  Coexists with ``live_trader.py``
on the same Alpaca paper account.

Start: python3 turbo_trader.py
Logs:  /home/team/shared/engine/logs/turbo_YYYYMMDD.log
"""
import asyncio
import logging
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
from src.execution.position_manager import PositionManager
from src.execution.session_state import load_start_equity, ny_today, save_start_equity
from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.liquidity_sweep import LiquiditySweepStrategy
from src.strategies.indicators import sma, z_score

# ── Configuration ──────────────────────────────────────────────────
TURBO_SYMBOLS = ["SOXL", "TQQQ", "FNGU", "SPXL"]  # 3x leveraged ETFs (LABU dropped — data errors)
# ── High-volatility ("VIOLENCE") tier ─────────────────────────────────────
# More aggressive instruments for the owner's "embrace volatility" direction:
# bigger per-trade moves mean bigger payouts even at a 60% win rate.  These
# run ALONGSIDE the base turbo symbols — the effective pool is
# TURBO_SYMBOLS + VIOLENCE_SYMBOLS (up to 9 symbols).
VIOLENCE_SYMBOLS = [
    "TNA",   # 3x Russell 2000 bull — small-caps are more volatile than large-cap ETFs
    "TZA",   # 3x Russell 2000 bear
    "LABU",  # 3x biotech bull — biotech is extremely volatile
    "LABD",  # 3x biotech bear
    "UVXY",  # 1.5x VIX short-term futures — tracks volatility itself, massive swings
    "NVDL",  # 2x NVDA bull — single-stock leverage, amplifies NVDA's big moves
    "TSLR",  # 2x TSLA bull — single-stock leverage
]
ENABLE_VIOLENCE_TIER = True    # False → pool falls back to the original 4 turbo symbols
# Violence-tier risk profile: tight stops get eaten alive on these names, so
# the stop widens to 9% and take-profit to 13%; position size drops to 40%
# (vs 50%) to survive the drawdowns.  Same max-2-positions rule.
VIOLENCE_STOP_LOSS_PCT = 0.09     # 8-10% band — tight stops get eaten alive here
VIOLENCE_TAKE_PROFIT_PCT = 0.13   # 12-15% band
VIOLENCE_POSITION_SIZE_PCT = 0.40 # 35-40% band — smaller size survives drawdowns
MAX_HOLD_EXTENDED_MINUTES = 60    # trend-extension cap: profitable 30-min hold → 60 min
CHECK_INTERVAL = 60
CONFIDENCE_THRESHOLD = 0.4   # Lower threshold = more entries
MAX_POSITIONS = 2            # 50% per position × 2 = 100% deployed — no margin needed
POSITION_SIZE_PCT = 0.50     # 50% per position — max conviction sizing
MANDATORY_CLOSE_MINUTES = 30  # Liquidate all positions 30 min before market close
MAX_HOLD_MINUTES = 30          # Max time to hold a position — recycle capital
def _effective_symbols() -> list[str]:
    """Effective turbo symbol pool: base 4 + violence tier when enabled."""
    if ENABLE_VIOLENCE_TIER:
        return list(TURBO_SYMBOLS) + list(VIOLENCE_SYMBOLS)
    return list(TURBO_SYMBOLS)
SYMBOLS = _effective_symbols()
_VIOLENCE_SET = {s.upper() for s in VIOLENCE_SYMBOLS}
def _is_violence_symbol(symbol: str) -> bool:
    """True if *symbol* belongs to the high-volatility ("VIOLENCE") tier."""
    return symbol.upper() in _VIOLENCE_SET
def _risk_params_for(symbol: str) -> tuple[float, float]:
    """(stop_loss_pct, take_profit_pct) for *symbol* — tier-aware routing."""
    if _is_violence_symbol(symbol):
        return VIOLENCE_STOP_LOSS_PCT, VIOLENCE_TAKE_PROFIT_PCT
    return STRATEGY_CONFIG.stop_loss_pct, STRATEGY_CONFIG.take_profit_pct
def _position_size_pct_for(symbol: str) -> float:
    """Position size (fraction of equity) for *symbol* — tier-aware routing."""
    return VIOLENCE_POSITION_SIZE_PCT if _is_violence_symbol(symbol) else POSITION_SIZE_PCT


STRATEGY_CONFIG = StrategyConfig(
    entry_threshold=0.5,
    exit_threshold=0.05,        # Tighter exit = recycle capital faster
    stop_loss_pct=0.06,         # Wider stop-loss (6%) for ETF volatility
    take_profit_pct=0.08,       # Wider take-profit (8%)
    max_position_pct=POSITION_SIZE_PCT,
    extra={"lookback": 20, "std_dev_multiplier": 2.0},
)
# UVXY mean-reversion config: VIX products have massive decay and revert
# faster than equities, so a 20-bar Z-score lookback lags too much — use a
# shorter 10-bar lookback so reversion entries fire while the move is fresh.
UVXY_MR_CONFIG = StrategyConfig(
    entry_threshold=STRATEGY_CONFIG.entry_threshold,
    exit_threshold=STRATEGY_CONFIG.exit_threshold,
    stop_loss_pct=VIOLENCE_STOP_LOSS_PCT,
    take_profit_pct=VIOLENCE_TAKE_PROFIT_PCT,
    max_position_pct=VIOLENCE_POSITION_SIZE_PCT,
    extra={"lookback": 10, "std_dev_multiplier": 2.0},
)


# Momentum strategy config (separate thresholds for the dual-strategy approach)
MOMENTUM_CONFIG = {
    "ma_period": 10,            # Shorter MA for faster crossover signals
    "trend_periods": 5,         # Periods for trend strength (statistical significance)
    "rsi_period": 14,
    "rsi_threshold": 40,        # RSI > 40 = momentum not oversold (lowered from 50)
}
# ── Feature flags (Recommendation #1 from the weekly trade analysis) ──
# Each can be flipped independently for paper-trading A/B comparison.
ENABLE_REGIME_GATE = True   # Skip mean-reversion LONGs in a confirmed downtrend
                            # (price below 10-bar MA AND RSI(14) < 40).  The turbo
                            # trader went 0-for-8 this week buying 3x ETFs into
                            # falling markets; the gate stops that.
ENABLE_SHORT_SELLING = True # Open SHORT positions when momentum SELL fires in a
                            # downtrend (price < MA, RSI < 40, high confidence),
                            # with inverted stop-loss (above entry) / take-profit
                            # (below entry).  Flattened with the EOD liquidation.
ENABLE_MEAN_REVERSION_SHORT = True  # When the regime gate blocks a mean-reversion
                            # LONG in a confirmed downtrend, OPEN A SHORT on the
                            # same down-tape instead of only doing nothing.  A
                            # gated MR-BUY means the instrument is oversold AND
                            # still below the MA — a falling knife; shorting it
                            # monetizes the downside the gate is refusing to buy.
                            # Shares the momentum short machinery (_handle_short_sell
                            # → inverted stop above entry), the "never double up"
                            # guard, and the max-1-entry-per-tick rule.

# Market close in UTC (4 PM ET = 20:00 UTC standard, 20:00 UTC year-round
# for simplicity — Alpaca clock is the final authority for is_market_open)
MARKET_CLOSE_UTC_HOUR = 20
MARKET_CLOSE_UTC_MINUTE = 0

# ── Logging ────────────────────────────────────────────────────────
log_dir = Path("/home/team/shared/engine/logs")
log_dir.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y%m%d")
log_file = log_dir / f"turbo_{today}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("turbo_trader")


# ── Inline helpers ─────────────────────────────────────────────────

def _compute_rsi(close: "pd.Series", period: int = 14) -> "pd.Series":
    """Compute RSI (Relative Strength Index) over *period* bars."""
    import pandas as pd
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _generate_momentum_signals(
    data: "pd.DataFrame",
    symbol: str,
    ma_period: int = 10,
    trend_periods: int = 5,
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
) -> "list[Signal]":
    """Generate momentum signals for leveraged ETFs.

    - BUY: price above MA AND RSI > threshold (momentum confirmed, not oversold).
    - SELL: price crosses below MA (was above, now below), OR price < MA AND RSI < 60.
    """
    close = data["close"]
    ma = sma(close, period=ma_period)
    rsi = _compute_rsi(close, period=rsi_period)

    signals: list[Signal] = []
    for idx in range(1, len(data)):
        cur_close = close.iloc[idx]
        cur_ma = ma.iloc[idx]
        cur_rsi = rsi.iloc[idx]

        if pd.isna(cur_ma) or pd.isna(cur_rsi):
            continue

        prev_close = close.iloc[idx - 1]
        prev_ma = ma.iloc[idx - 1]
        ts = data.index[idx]

        # ── BUY signal: price > MA AND RSI > 40 (simplified — no trend double-count) ──
        if cur_close > cur_ma and cur_rsi > rsi_threshold:
            ma_distance = (cur_close - cur_ma) / (abs(cur_ma) + 1e-9)
            # Confidence: sigmoid-like blend of MA distance and RSI strength
            # Produces gradients between 0.3 and 1.0 instead of all 1.0
            ma_score = min(1.0, ma_distance * 25)       # ma_distance typically 0.005-0.04
            rsi_score = max(0.0, (cur_rsi - 40) / 40)   # 0 at RSI=40, 1 at RSI=80
            confidence = round(0.3 + 0.7 * (ma_score + rsi_score) / 2, 4)
            confidence = min(1.0, max(0.0, confidence))
            signal = Signal(
                symbol=symbol,
                timestamp=ts,
                signal_type=SignalType.BUY,
                confidence=round(confidence, 6),
                metadata={
                    "strategy": "momentum",
                    "ma_distance": round(ma_distance, 6),
                    "rsi": round(cur_rsi, 2),
                },
            )
            signals.append(signal)
            logger.debug(
                "🔍 MOMENTUM %s: BUY conf=%.4f (price=%.2f, MA=%.2f, RSI=%.1f)",
                symbol, confidence, cur_close, cur_ma, cur_rsi,
            )

        # ── SELL signal: MA cross-under OR price < MA and weak RSI ──
        elif (cur_close < cur_ma and prev_close >= prev_ma) or \
             (cur_close < cur_ma and cur_rsi < 60):
            # Cross below MA — momentum broken, or price below MA with weakening RSI.
            # Confidence bug (fixed): the old formula ``dist_pct * 10`` produced
            # 0.05–0.4 for realistic 3x ETF moves (1–3% from the MA), so the SELL
            # signal could essentially never cross the 0.4 activation threshold.
            # Rescaled below so a 1% move scores ~0.4 and 2–3% moves reach
            # 0.5–1.0, blended with RSI weakness and a fresh-cross bonus.
            dist_pct = abs(cur_close - cur_ma) / (abs(cur_ma) + 1e-9)
            dist_score = min(1.0, dist_pct * 40)            # 1% away → 0.4, 2.5%+ → 1.0
            rsi_score = max(0.0, min(1.0, (60.0 - cur_rsi) / 40.0))  # 0 @ RSI=60, 1 @ RSI<=20
            cross_bonus = 0.15 if (cur_close < cur_ma and prev_close >= prev_ma) else 0.0
            confidence = min(1.0, 0.3 + 0.7 * (0.5 * dist_score + 0.3 * rsi_score) + cross_bonus)
            signal = Signal(
                symbol=symbol,
                timestamp=ts,
                signal_type=SignalType.SELL,
                confidence=round(confidence, 6),
                metadata={
                    "strategy": "momentum",
                    "cross_below_ma": bool(cur_close < cur_ma and prev_close >= prev_ma),
                    "rsi": round(cur_rsi, 2),
                },
            )
            signals.append(signal)
            logger.debug(
                "🔍 MOMENTUM %s: SELL conf=%.4f (price=%.2f, MA=%.2f, RSI=%.1f)",
                symbol, confidence, cur_close, cur_ma, cur_rsi,
            )

    return signals


# ── Regime gate ────────────────────────────────────────────────────
def _regime_gate_allows_long(
    data: "pd.DataFrame",
    ma_period: int = 10,
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
) -> "tuple[bool, str]":
    """Return ``(allow, reason)`` for a mean-reversion LONG on *data*.

    A long is BLOCKED only in a confirmed downtrend — price below the
    10-bar MA AND RSI(14) below *rsi_threshold*.  The turbo trader's
    0-for-8 week was caused by buying 3x leveraged ETFs into exactly this
    regime.  When either condition is healthy (price above the MA, or RSI
    recovering), the long is allowed — this is a filter, not a trend
    follower.

    ``reason`` is a human-readable string for audit logging when blocked
    (empty string when allowed).
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


def _trend_extension_qualifies(data: "pd.DataFrame", pos) -> bool:
    """Return True if a violence-tier position held >= MAX_HOLD_MINUTES is
    profitably trending in the right direction and deserves the extended
    60-min hold (instead of the hard 30-min exit).

    Qualification: the position must be in profit (price beyond entry in the
    direction of the trade) AND the latest bar must be moving further in the
    trade's direction (last close vs first close of the fetched window).
    - long:  price > entry and price rising
    - short: price < entry and price falling
    VIX/leveraged names trend hard when they move; cutting a winner at 30 min
    is how the turbo trader capped its upside.  Base-tier symbols never get
    the extension — they keep the plain 30-min max hold.
    """
    close = data["close"]
    if len(close) < 2:
        return False
    price = float(close.iloc[-1])
    entry = float(getattr(pos, "entry_price", 0) or 0)
    qty = float(getattr(pos, "quantity", 0) or 0)
    if entry <= 0:
        return False
    if qty < 0:  # short — profitable when price fell; needs continued downside
        if price >= entry:
            return False
        return float(close.iloc[-1]) < float(close.iloc[0])
    # long — profitable when price rose; needs continued upside
    if price <= entry:
        return False
    return float(close.iloc[-1]) > float(close.iloc[0])

# ── Turbo-specific idempotency key generator ────────────────────────

def _turbo_client_id(symbol: str, side: OrderSide) -> str:
    """Generate a unique idempotency key for turbo orders.

    Format: ``algoflow_TURBO_{SYMBOL}_{SIDE}_{timestamp_ns}``
    """
    return f"algoflow_TURBO_{symbol.upper()}_{side.value}_{time.monotonic_ns()}"


def _turbo_stop_client_id(symbol: str) -> str:
    """Generate a unique idempotency key for turbo protective stop orders.

    Format: ``algoflow_TURBO_{SYMBOL}_STOP_{timestamp_ns}``
    """
    return f"algoflow_TURBO_{symbol.upper()}_STOP_{time.monotonic_ns()}"


# ── TurboTrader ────────────────────────────────────────────────────

class TurboTrader:
    def __init__(self):
        self.broker = AlpacaBroker()
        self.provider = YFinanceProvider()
        self.mean_reversion = MeanReversionStrategy(config=STRATEGY_CONFIG)
        # UVXY gets its own mean-reversion instance with a shorter 10-bar
        # lookback — VIX products decay and revert faster than equities.
        self.uvxy_mean_reversion = MeanReversionStrategy(config=UVXY_MR_CONFIG)
        self.liquidity_sweep = LiquiditySweepStrategy(symbols=SYMBOLS)
        self.levels_cache: dict[str, dict[str, float | None]] = {}
        self._ls_symbols: set[str] = set()  # symbols entered via liquidity sweep
        self._entry_times: dict[str, datetime] = {}  # when each position was opened
        self._extended_holds: set[str] = set()  # violence-tier holds granted the 60-min trend extension
        self.pm = PositionManager(STRATEGY_CONFIG)
        self.day_trades: list[dict] = []
        self.start_equity = 0.0

    # ── Market open wait ─────────────────────────────────────────────

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
        """Establish the day's P&L baseline; return True when ready to trade.

        Prefers the persisted baseline for today (so a watchdog restart mid-day
        keeps the day's P&L anchored to the original market-open equity), and
        falls back to fetching the account.  Never starts trading with an
        unknown baseline.
        """
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
        logger.info("✅ %s — starting TURBO trading",
                    "Market is OPEN" if not assumed else "Market assumed OPEN")
        logger.info("   Starting equity: ${:,.2f}".format(self.start_equity))
        return True

    async def wait_for_market_open(self):
        """Wait for 9:30 ET using local DST-aware time as the primary gate.
        Alpaca's clock is only a confirmation: a timeout (``None``) cannot
        strand the trader after the session has opened (with a five-minute
        grace period) and can never be mistaken for a confirmed close.
        """
        from zoneinfo import ZoneInfo
        logger.info("🚀 Waiting for market to open (9:30 AM ET)...")
        last_heartbeat = time.monotonic()
        # ~4 min cadence keeps quiet waits below the watchdog's staleness
        # threshold (WATCHDOG_STALE_SECONDS, default 300s) so the log
        # freshness signal is truthful even without the closed-market
        # exemption (watchdog.sh / src/watchdog/policy.py).
        heartbeat_interval = 240.0
        check_interval = 30.0
        while True:
            seconds_until = self._seconds_until_open()
            now_et = datetime.now(ZoneInfo("America/New_York"))
            before_open = now_et.weekday() < 5 and (now_et.hour, now_et.minute) < (9, 30)
            if seconds_until > 0 and (before_open or now_et.weekday() >= 5 or now_et.hour >= 16):
                await asyncio.sleep(min(max(seconds_until - 60.0, 1.0), heartbeat_interval))
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
            try:
                market_open = await self.broker.is_market_open()
            except Exception as e:
                logger.warning("Market check failed: %s", e)
                market_open = None
            if market_open is True:
                if await self._begin_session(assumed=False):
                    return
            elif market_open is None:
                # Clock unavailable — indeterminate, NOT a confirmed close.
                if now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 35):
                    logger.warning("Market clock unavailable after grace period — proceeding on local time")
                    if await self._begin_session(assumed=True):
                        return
                else:
                    logger.warning("Market clock unavailable during grace period — retrying")
            # market_open is False → confirmed closed → keep waiting.
            if time.monotonic() - last_heartbeat >= heartbeat_interval:
                logger.info("Still waiting for market open, next check in %.0fs (clock confirmation)", check_interval)
                last_heartbeat = time.monotonic()
            await asyncio.sleep(check_interval)
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

    # ── EOD check ────────────────────────────────────────────────────

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

    # ── Main loop ────────────────────────────────────────────────────

    async def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("🚀 AlgoFlow TURBO Live Trader — STARTING")
        logger.info(f"Symbols: {SYMBOLS}")
        if ENABLE_VIOLENCE_TIER:
            logger.info(
                f"🔥 VIOLENCE tier ENABLED: {len(VIOLENCE_SYMBOLS)} high-vol symbols "
                f"(stops {VIOLENCE_STOP_LOSS_PCT:.0%}/TP {VIOLENCE_TAKE_PROFIT_PCT:.0%}, "
                f"size {VIOLENCE_POSITION_SIZE_PCT:.0%}, hold ≤ {MAX_HOLD_EXTENDED_MINUTES}min w/ trend ext)"
            )
        logger.info(f"Strategy: MeanReversion + Momentum + LiquiditySweep | Confidence ≥ {CONFIDENCE_THRESHOLD}")
        logger.info(f"Max positions: {MAX_POSITIONS} | Base size: {POSITION_SIZE_PCT*100:.0f}% equity")
        logger.info(f"Base stop-loss: {STRATEGY_CONFIG.stop_loss_pct*100:.0f}% | "
                    f"take-profit: {STRATEGY_CONFIG.take_profit_pct*100:.0f}%")
        logger.info(f"⚠️  Mandatory EOD liquidation {MANDATORY_CLOSE_MINUTES} min before close")
        logger.info("=" * 60)

        try:
            await self.broker.startup_health_check()
        except Exception as exc:
            logger.critical("FATAL: broker authentication/account health check failed; refusing to trade: %s", exc)
            return

        # ── Cancel stale orders from prior sessions ──────────────────
        # NOTE: cancel_all_orders() cancels EVERY open order on the account
        # (both turbo's AND the main trader's). This is a known side effect of
        # sharing an Alpaca paper account. We log a warning so operators are aware.
        # This also cancels any stale protective stop orders from prior sessions,
        # which will be re-created below during position sync.
        logger.info("Cancelling any stale orders from prior sessions…")
        cancelled = await self.broker.cancel_orders_by_client_id_prefix("algoflow_TURBO_")

        # ── Sync positions from Alpaca at startup ────────────────────
        logger.info("STEP 1/4: Syncing positions from broker…")
        await self._sync_positions_from_broker()
        logger.info("STEP 1/4: Position sync complete — %d open positions tracked",
                     self.pm.get_open_count())

        # ── Log inherited position state ────────────────────────────
        for sym in self.pm.get_open_symbols():
            pos = self.pm.get_positions().get(sym)
            if pos:
                logger.info("  Inherited: %s x %s @ $%.2f", pos.quantity, sym, pos.entry_price)

        # ── Post-startup stale position cleanup ──────────────────────
        await self._post_startup_cleanup()

        # Keep the process alive between sessions.  Each iteration waits for
        # the next market open, prepares session data, trades, and reports a
        # summary before resetting state for the following trading day.
        while True:
            await self.wait_for_market_open()

            # ── Calculate liquidity sweep levels ────────────────────────
            logger.info("STEP 2/4: Calculating liquidity sweep levels…")
            await self.liquidity_sweep.calculate_levels(self.provider)
            self.levels_cache = self.liquidity_sweep.levels
            logger.info("STEP 2/4: Levels calculated for %d symbols", len(self.levels_cache))

            # ── Ensure inherited positions have protective stops ────────
            logger.info("STEP 3/4: Checking protective stops for inherited positions…")
            await self._ensure_protective_stops()
            logger.info("STEP 3/4: Protective stop check complete")

            logger.info("STEP 4/4: Entering main tick loop…")
            tick = 0
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

                    try:
                        market_open = await self.broker.is_market_open()
                    except Exception:
                        logger.exception("Market-open check failed — treating as unknown, will retry")
                        market_open = None
                    if market_open is None:
                        # Indeterminate clock (timeout/outage) — never treat as
                        # a confirmed close.  Retry instead of shutting down,
                        # so a transient API outage cannot trigger a false
                        # mid-session liquidation.
                        logger.warning("Market clock UNKNOWN — retrying instead of shutting down")
                        await asyncio.sleep(CHECK_INTERVAL)
                        continue
                    if not market_open:
                        logger.info("⏹️  Market closed — completing session and waiting for next open")
                        break

                    await self._safe_tick(tick)
                    logger.debug("Tick %d: complete — sleeping %ds", tick, CHECK_INTERVAL)
                    await asyncio.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                await self.shutdown()
                return
            except Exception as e:
                logger.exception("FATAL: Unhandled exception in main loop — %s", e)

            await self.shutdown(close_broker=False)
            self.pm.reset()
            self._entry_times.clear()
            self.day_trades.clear()
            self._ls_symbols.clear()
            self._extended_holds.clear()
            self.levels_cache.clear()
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
        """One polling cycle: fetch → triple-strategy signals → execute."""

        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=60)  # deeper lookback for MA/RSI

        # Track which symbols have active liquidity sweep positions
        # (prune any that have been closed)
        self._ls_symbols &= set(self.pm.get_open_symbols())

        logger.debug("Tick %d: evaluating %d symbols (%d positions open)",
                     tick_num, len(SYMBOLS), self.pm.get_open_count())

        entered_this_tick = False  # max 1 new entry per tick cycle

        for symbol in SYMBOLS:
            if entered_this_tick:
                break  # one entry per tick — prevents simultaneous all-ins
            # Skip if at max positions and don't hold this one
            if self.pm.get_open_count() >= MAX_POSITIONS and not self.pm.has_position(symbol):
                continue

            try:
                logger.debug("Tick %d: fetching %s 1m bars…", tick_num, symbol)
                mdf = await self.provider.fetch_bars(
                    symbol, start=lookback, end=now, timeframe="1min"
                )
                data = mdf.df
                if data.empty or len(data) < 25:
                    continue

                # ── Generate signals from all three strategies ───────
                # UVXY reverts faster than equities (massive decay) → route
                # it to the short-lookback (10-bar) mean-reversion instance.
                mr_strategy = (
                    self.uvxy_mean_reversion if symbol.upper() == "UVXY"
                    else self.mean_reversion
                )
                mr_signals = mr_strategy.generate_signals(data)
                mom_signals = _generate_momentum_signals(
                    data,
                    symbol=symbol,
                    ma_period=MOMENTUM_CONFIG["ma_period"],
                    trend_periods=MOMENTUM_CONFIG["trend_periods"],
                    rsi_period=MOMENTUM_CONFIG["rsi_period"],
                    rsi_threshold=MOMENTUM_CONFIG["rsi_threshold"],
                )
                ls_signals: list[Signal] = []
                symbol_levels = self.levels_cache.get(symbol.upper(), {})
                if symbol_levels:
                    # Inject the symbol name into levels so generate_signals can read it
                    symbol_levels["_symbol"] = symbol
                    ls_signals = self.liquidity_sweep.generate_signals(data, symbol_levels)

                # ── Safety: filter LS signals ────────────────────────
                filtered_ls: list[Signal] = []
                for sig in ls_signals:
                    sym_up = sig.symbol.upper()
                    # Max 1 liquidity sweep position at a time
                    if sig.signal_type == SignalType.BUY and len(self._ls_symbols) >= 1:
                        continue
                    # Skip if LS signal conflicts with existing positions
                    if sig.signal_type == SignalType.BUY:
                        if self.pm.has_position(sym_up):
                            continue
                    elif sig.signal_type == SignalType.SELL:
                        if not self.pm.has_position(sym_up):
                            continue
                    filtered_ls.append(sig)

                # Pick the highest-confidence signal that meets threshold
                best_signal: Signal | None = None
                for sig in (mr_signals + mom_signals + filtered_ls):
                    if sig.confidence < CONFIDENCE_THRESHOLD:
                        continue
                    if best_signal is None or sig.confidence > best_signal.confidence:
                        best_signal = sig

                if best_signal is None:
                    continue

                current_price = float(data["close"].iloc[-1])
                strategy_name = best_signal.metadata.get("strategy", "mean_reversion")
                pos = self.pm.get_positions().get(symbol.upper())

                if best_signal.signal_type == SignalType.BUY:
                    if pos is not None and pos.quantity < 0:
                        # Momentum flipped bullish while we are SHORT — buy to
                        # cover the short (downtrend broken).
                        logger.info(
                            "📗 COVER %s: momentum BUY while short — buying to cover",
                            symbol,
                        )
                        await self._handle_sell(symbol, current_price, best_signal.confidence, strategy_name)
                        self._ls_symbols.discard(symbol.upper())
                    else:
                        # ── Regime gate: in a confirmed downtrend (price <
                        #    MA10 AND RSI < 40) NO LONG may enter — neither a
                        #    mean-reversion dip-buy (falling knife) nor a
                        #    momentum long fighting the trend.  When the gate
                        #    blocks a mean-reversion LONG, we OPEN A SHORT on
                        #    the same down-tape instead of only doing nothing. ──
                        if ENABLE_REGIME_GATE and \
                                strategy_name in ("mean_reversion", "momentum"):
                            allow, reason = _regime_gate_allows_long(
                                data,
                                ma_period=MOMENTUM_CONFIG["ma_period"],
                                rsi_period=MOMENTUM_CONFIG["rsi_period"],
                                rsi_threshold=MOMENTUM_CONFIG["rsi_threshold"],
                            )
                            if not allow:
                                if strategy_name == "mean_reversion" and \
                                        ENABLE_SHORT_SELLING and \
                                        ENABLE_MEAN_REVERSION_SHORT:
                                    # Falling knife: confirmed downtrend + MR
                                    # oversold dip.  Monetize the downside the
                                    # gate is refusing to buy.
                                    logger.info(
                                        "🔻 MR-SHORT %s: gate blocked the LONG — "
                                        "opening SHORT instead (%s)",
                                        symbol, reason,
                                    )
                                    await self._handle_short_sell(
                                        symbol, current_price,
                                        best_signal.confidence, strategy_name,
                                    )
                                    entered_this_tick = True  # max 1 new entry per tick
                                else:
                                    logger.info(
                                        "🚫 REGIME GATE %s: skipping %s LONG — %s",
                                        symbol, strategy_name, reason,
                                    )
                                continue
                        await self._handle_buy(symbol, current_price, best_signal.confidence, strategy_name)
                        entered_this_tick = True  # max 1 entry per tick
                        # Track liquidity sweep entries for the max-1-LS-position guard
                        if strategy_name == "liquidity_sweep":
                            self._ls_symbols.add(symbol.upper())
                elif best_signal.signal_type == SignalType.SELL:
                    if pos is not None and pos.quantity < 0:
                        # Already short — never double up on the downside
                        logger.debug(
                            "Momentum SELL %s: already short — skipping (no doubling)",
                            symbol,
                        )
                        continue
                    if pos is not None:
                        await self._handle_sell(symbol, current_price, best_signal.confidence, strategy_name)
                        # If we sold an LS position, remove from tracking
                        self._ls_symbols.discard(symbol.upper())
                    elif ENABLE_SHORT_SELLING and strategy_name == "momentum" and \
                            float(best_signal.metadata.get("rsi", 100)) < MOMENTUM_CONFIG["rsi_threshold"]:
                        # No position + momentum SELL in a confirmed downtrend
                        # (price < MA, RSI < 40) → open a SHORT instead of
                        # doing nothing.
                        await self._handle_short_sell(symbol, current_price, best_signal.confidence, strategy_name)
                        entered_this_tick = True  # max 1 new entry per tick
                    else:
                        logger.debug(
                            "SELL %s: no long to close (short selling %s)",
                            symbol,
                            "disabled" if not ENABLE_SHORT_SELLING
                            else "requires momentum RSI < %s" % MOMENTUM_CONFIG["rsi_threshold"],
                        )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Check stop-loss / take-profit
        await self._check_risk_stops()

    async def _handle_buy(self, symbol: str, price: float, confidence: float, strategy: str = ""):
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

        value = equity * _position_size_pct_for(symbol)  # tier-aware: 50% base / 40% violence
        qty = value / price if price > 0 else 0
        if qty < 1:
            return

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=qty,
            order_type=OrderType.MARKET,
            client_id=_turbo_client_id(symbol, OrderSide.BUY),
        )
        result = await self.broker.place_order(order)

        if is_order_alive(result.status):
            # ── Wait for the entry to fill before attaching the stop ──
            # A SELL stop submitted while the entry BUY is still open is
            # rejected by Alpaca as a "potential wash trade" (opposite side
            # market/stop order exists).  Poll the entry until it settles so
            # the stop goes in on a filled position with the real fill price.
            fill_price = result.filled_avg_price if result.filled_avg_price else price
            filled_qty = qty
            order_id = result.order_id
            if order_id:
                filled = await self.broker.wait_for_order_fill(order_id, timeout=8.0)
                if filled is False:
                    # Entry order died (rejected/canceled) after submission —
                    # do NOT track a position that doesn't exist.
                    logger.warning(f"❌ BUY {symbol} entry order died after submission — not opening position")
                    return
                if filled is not None:
                    fq = float(getattr(filled, "qty", 0) or 0)
                    fp = float(getattr(filled, "filled_avg_price", 0) or 0)
                    if fq > 0:
                        filled_qty = fq
                    if fp > 0:
                        fill_price = fp
                else:
                    logger.warning(
                        f"⏳ BUY {symbol}: fill not confirmed within 8s — opening position "
                        f"defensively (in-process risk checks still active)"
                    )

            self.pm.open_position(symbol, filled_qty, fill_price)
            self._entry_times[symbol.upper()] = datetime.now(timezone.utc)  # time-based exit
            strat_tag = f" [{strategy}]" if strategy else ""
            logger.info(
                f"🚀 BUY  {symbol}: {filled_qty:.1f} shares @ ${fill_price:.2f} = ${value:,.2f} | "
                f"conf={confidence:.2f}{strat_tag} | order={result.order_id[:8]}"
            )
            # ── Place GTC protective stop at broker ─────────────────
            await self._place_protective_stop(symbol, filled_qty, fill_price)
        else:
            logger.warning(f"❌ BUY {symbol} REJECTED: {result.status}")

    async def _handle_short_sell(self, symbol: str, price: float, confidence: float, strategy: str = ""):
        """Open a SHORT position when momentum SELL fires in a downtrend.

        Submits a SELL market order (Alpaca opens a short when flat), records
        the position with a NEGATIVE quantity in the PositionManager, and
        attaches a GTC BUY stop ABOVE entry (the stop-loss side for shorts).
        Sizing mirrors ``_handle_buy``: ``POSITION_SIZE_PCT`` of equity.
        """
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
        value = equity * _position_size_pct_for(symbol)  # tier-aware: 50% base / 40% violence
        qty = value / price if price > 0 else 0
        if qty < 1:
            return
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,  # short sell — opens a short when flat
            quantity=qty,
            order_type=OrderType.MARKET,
            client_id=_turbo_client_id(symbol, OrderSide.SELL),
        )
        result = await self.broker.place_order(order)
        if is_order_alive(result.status):
            # ── Wait for the entry to fill before attaching the stop ──
            # A BUY stop submitted while the entry SELL is still open is
            # rejected by Alpaca's wash-trade filter; poll the entry until it
            # settles so the stop goes in on a filled position.
            fill_price = result.filled_avg_price if result.filled_avg_price else price
            filled_qty = qty
            order_id = result.order_id
            if order_id:
                filled = await self.broker.wait_for_order_fill(order_id, timeout=8.0)
                if filled is False:
                    # Entry order died (rejected/canceled) after submission —
                    # do NOT track a position that doesn't exist.
                    logger.warning(f"❌ SHORT {symbol} entry order died after submission — not opening position")
                    return
                if filled is not None:
                    fq = float(getattr(filled, "qty", 0) or 0)
                    fp = float(getattr(filled, "filled_avg_price", 0) or 0)
                    if fq > 0:
                        filled_qty = fq
                    if fp > 0:
                        fill_price = fp
                else:
                    logger.warning(
                        f"⏳ SHORT {symbol}: fill not confirmed within 8s — opening position "
                        f"defensively (in-process risk checks still active)"
                    )

            # Negative quantity = short position (PnL math stays correct).
            self.pm.open_position(symbol, -filled_qty, fill_price)
            self._entry_times[symbol.upper()] = datetime.now(timezone.utc)  # time-based exit
            strat_tag = f" [{strategy}]" if strategy else ""
            logger.info(
                f"🔻 SHORT {symbol}: {filled_qty:.1f} shares @ ${fill_price:.2f} = ${value:,.2f} | "
                f"conf={confidence:.2f}{strat_tag} | order={result.order_id[:8]}"
            )
            # ── Place GTC protective BUY stop ABOVE entry ────────────
            await self._place_protective_stop(symbol, filled_qty, fill_price, is_short=True)
        else:
            logger.warning(f"❌ SHORT {symbol} REJECTED: {result.status}")

    async def _handle_sell(self, symbol: str, price: float, confidence: float, strategy: str = ""):
        if not self.pm.has_position(symbol):
            return

        pos = self.pm.get_positions().get(symbol.upper())
        if pos is None:
            return
        qty = pos.quantity
        entry = pos.entry_price
        is_short = qty < 0
        abs_qty = abs(qty)
        # Closing a long = SELL; closing a short = BUY (buy to cover)
        side = OrderSide.BUY if is_short else OrderSide.SELL
        action = "COVER" if is_short else "SELL"

        # ── Cancel protective stop before selling ───────────────────
        if not await self._cancel_protective_stops(symbol):
            logger.warning("%s %s deferred: protective-order cancellation was not confirmed",
                           action, symbol)
            return

        order = Order(
            symbol=symbol,
            side=side,
            quantity=abs_qty,
            order_type=OrderType.MARKET,
            client_id=_turbo_client_id(symbol, side),
        )
        result = await self.broker.place_order(order)

        if not is_order_alive(result.status):
            error = (getattr(result, "error_message", None) or "").lower()
            if not is_short and "cannot be sold short" in error:
                self.pm.discard_position(symbol, reason="broker says position is not held")
                self._entry_times.pop(symbol.upper(), None)
                logger.warning("SELL %s rejected as phantom position; removed from tracking", symbol)
            else:
                # The protective stop was cancelled above — restore it so the
                # position doesn't run naked because of a rejected exit.
                logger.warning(
                    "%s %s rejected (%s); keeping position tracked — restoring protective stop",
                    action, symbol, result.status,
                )
                await self._place_protective_stop(symbol, abs_qty, entry, is_short=is_short)
            return

        # Direction-aware P&L: a short profits when price falls.
        if is_short:
            pnl = (entry - price) * abs_qty
            pnl_pct = ((entry / price) - 1.0) * 100 if price else 0
        else:
            pnl = (price - entry) * qty
            pnl_pct = ((price / entry) - 1.0) * 100 if entry else 0
        self.pm.close_position(symbol, price)
        self._entry_times.pop(symbol.upper(), None)  # clean up time tracker

        strat_tag = f" [{strategy}]" if strategy else ""
        log_icon = "📗" if is_short else "📉"
        logger.info(
            f"{log_icon} {action} {symbol}: {abs_qty:.1f} shares @ ${price:.2f} | "
            f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%) | conf={confidence:.2f}{strat_tag} | "
            f"order={result.order_id[:8]}"
        )

    async def _sync_positions_from_broker(self):
        """Reconcile local PositionManager with Alpaca's actual positions."""
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

        turbo_set = {s.upper() for s in SYMBOLS}

        for sym, pos_data in broker_symbols.items():
            if sym not in turbo_set:
                continue  # Ignore non-turbo symbols (e.g. main trader's positions)
            if sym not in pm_symbols:
                entry_price = pos_data["avg_entry_price"]
                current_price = float(pos_data.get("current_price", entry_price))
                qty = pos_data["qty"]  # negative for short positions
                is_short = qty < 0
                abs_qty = abs(qty)

                # ── Cleanup: liquidate inherited underwater positions (>2% loss) ──
                # A short is underwater when price RISES above entry.
                if current_price > 0 and entry_price > 0:
                    if is_short:
                        pnl_pct = (entry_price - current_price) / entry_price
                    else:
                        pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct < -0.02:
                        side = OrderSide.BUY if is_short else OrderSide.SELL
                        action = "COVER" if is_short else "SELL"
                        logger.info(
                            "🧹 Cleanup %s %s: inherited at $%.2f, now $%.2f = %.1f%%",
                            action, sym, entry_price, current_price, pnl_pct * 100,
                        )
                        order = Order(
                            symbol=sym,
                            side=side,
                            quantity=abs_qty,
                            order_type=OrderType.MARKET,
                            client_id=_turbo_client_id(sym, side),
                        )
                        result = await self.broker.place_order(order)
                        if is_order_alive(result.status):
                            continue  # Don't add to PM — the exit was accepted
                        logger.warning(
                            "🧹 Cleanup %s %s rejected (%s); tracking position",
                            action, sym, result.status,
                        )

                self.pm.open_position(
                    symbol=sym,
                    quantity=qty,  # negative qty = short, P&L math stays correct
                    entry_price=entry_price,
                )
                logger.info(
                    "  + Added %s: %s shares @ $%.2f (sync)%s",
                    sym, abs_qty, entry_price, " [SHORT]" if is_short else "",
                )
                added += 1

        for sym in pm_symbols - set(broker_symbols):
            if sym not in turbo_set:
                continue  # Never remove non-turbo positions from PM
            self.pm.close_position(sym, exit_price=0, exit_reason="sync_removed")
            logger.info("  - Removed stale %s (not on broker)", sym)
            removed += 1

        logger.info(
            "Synced %d positions from broker (%d added, %d removed)",
            len(broker_symbols), added, removed,
        )

    # ── Post-startup stale position cleanup ────────────────────────────

    async def _post_startup_cleanup(self):
        """Liquidate stale turbo positions if trader starts while market is closed.

        If the market is closed (e.g. after a crash/restart/sandbox cycle
        that missed the regular EOD window) and there are open turbo positions
        at the broker, liquidate them immediately so nothing hangs overnight.
        """
        try:
            market_status = await self.broker.is_market_open()
            if market_status is True:
                return  # Market is open — normal trading, no cleanup needed
            if market_status is None:
                # Indeterminate clock (outage) — never liquidate on a guess.
                logger.warning("Market status UNKNOWN at startup — skipping stale-position cleanup")
                return
            positions = await self.broker.get_positions()
            # Filter to turbo symbols only — never touch main trader's positions
            turbo_set = {s.upper() for s in SYMBOLS}
            positions = [p for p in positions if p.get("symbol", "").upper() in turbo_set]
            if not positions:
                return

            logger.info(
                "🧹 Post-close cleanup: liquidating %d stale turbo position(s) from previous session",
                len(positions),
            )
            for p in positions:
                sym = p.get("symbol")
                qty = float(p.get("qty", 0))
                if qty == 0:
                    continue
                is_short = qty < 0
                abs_qty = abs(qty)
                side = OrderSide.BUY if is_short else OrderSide.SELL
                action = "COVER" if is_short else "SELL"
                # Cancel protective stop before liquidating
                if not await self._cancel_protective_stops(sym):
                    logger.warning("🧹 Cleanup %s %s deferred: cancellation not confirmed",
                                   action, sym)
                    continue
                logger.info("🧹 Cleanup %s %s: %s shares%s",
                            action, sym, abs_qty, " [short]" if is_short else "")
                order = Order(
                    symbol=sym,
                    side=side,
                    quantity=abs_qty,
                    order_type=OrderType.MARKET,
                    client_id=_turbo_client_id(sym, side),
                )
                result = await self.broker.place_order(order)
                if not is_order_alive(result.status):
                    # Stop was cancelled above — restore it so the stale
                    # position isn't left naked if the cleanup exit fails.
                    logger.warning(
                        "🧹 Cleanup %s %s rejected (%s); restoring protective stop and keeping position",
                        action, sym, result.status,
                    )
                    await self._place_protective_stop(
                        sym, abs_qty, float(p.get("avg_entry_price", 0)) or 0,
                        is_short=is_short,
                    )
                    continue
                self.pm.close_position(
                    sym,
                    exit_price=float(p.get("current_price", 0)),
                    exit_reason="post_close_cleanup",
                )
            logger.info(
                "🧹 Post-close cleanup complete — %d turbo position(s) liquidated",
                len(positions),
            )
        except Exception:
            logger.exception("🧹 Post-close cleanup failed — continuing startup")

    # ── Broker-level protective stops ────────────────────────────────

    async def _place_protective_stop(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        max_attempts: int = 4,
        initial_delay: float = 2.0,
        is_short: bool = False,
    ) -> bool:
        """Place a GTC protective stop-loss order at the broker.

        This order survives process death and sandbox cycling — Alpaca holds
        it until triggered or cancelled.  For LONG positions the stop is a
        SELL at ``entry_price * (1 - stop_loss_pct)`` (6% below entry); for
        SHORT positions it is a BUY at ``entry_price * (1 + stop_loss_pct)``
        (6% ABOVE entry — a short loses money when price rises).

        Retries with backoff: submitting the stop while the entry order is
        still open makes Alpaca reject it as a "potential wash trade"
        (``opposite side market/stop order exists``), which previously left
        positions running without any broker-side protection for the rest of
        the session.  Each attempt re-checks the symbol's open orders so the
        stop is never submitted while an opposite-side order is live.
        """
        sym = symbol.upper()
        qty = int(qty)
        if qty <= 0:
            logger.warning("🛡️  STOP %s: position too small for protective stop (qty < 1 share)", sym)
            return False

        # Tier-aware stop distance: 6% for base turbo symbols, 9% for the
        # violence tier (tight stops get eaten alive on high-vol names).
        stop_loss_pct = _risk_params_for(symbol)[0]
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

            client_id = _turbo_stop_client_id(sym)
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
        """Cancel all open orders for *symbol* — stop-loss and take-profit.

        Called before selling a position so the GTC stop doesn't trigger
        after the position is closed.
        """
        sym = symbol.upper()
        try:
            open_orders = await self.broker.get_open_orders(symbol=sym)
        except Exception as exc:
            logger.warning("Failed to fetch open orders for %s: %s", sym, exc)
            return

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
        """Ensure every inherited turbo position has a GTC protective stop.

        Called after ``_sync_positions_from_broker()`` at startup.  For
        positions that survived a sandbox cycle, we check whether a stop
        order already exists at Alpaca.  If not, we place a fresh one.
        """
        turbo_set = {s.upper() for s in SYMBOLS}
        if not self.pm.get_open_symbols():
            logger.info("🛡️  No inherited turbo positions — skipping protective stop check")
            return

        # Fetch all open orders once so we can check stop coverage
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            logger.warning("Cannot verify protective stops — order fetch failed: %s", exc)
            return

        # Build a set of symbols that already have an open stop order
        # (SELL stop for longs, BUY stop for shorts — either means covered)
        covered_symbols: set[str] = set()
        for o in open_orders:
            o_sym = str(o.symbol).upper()
            o_side = str(o.side).upper()
            if o_sym in turbo_set and o_side in ("SELL", "BUY"):
                covered_symbols.add(o_sym)

        for sym in list(self.pm.get_open_symbols()):
            if sym not in turbo_set:
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
        """Check stop-loss / take-profit for open positions.

        Risk parameters are tier-aware: base turbo symbols use the 6%/8%
        stops from ``STRATEGY_CONFIG``; violence-tier symbols use the wider
        9%/13% band.  Time-based exit: base symbols hard-exit after
        ``MAX_HOLD_MINUTES`` (30); violence-tier symbols may extend a
        profitably-trending position to ``MAX_HOLD_EXTENDED_MINUTES`` (60).
        """
        # Prune extension markers for positions that were closed elsewhere
        self._extended_holds &= {s.upper() for s in self.pm.get_open_symbols()}
        for symbol in list(self.pm.get_open_symbols()):
            if symbol.upper() not in {s.upper() for s in SYMBOLS}:
                continue  # Safety: never touch non-turbo symbols
            if not self.pm.has_position(symbol):
                continue
            try:
                # 10-min window: enough bars for the trend-extension check
                mdf = await self.provider.fetch_bars(
                    symbol,
                    start=datetime.now(timezone.utc) - timedelta(minutes=10),
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
                is_short = pos.quantity < 0
                stop_loss_pct, take_profit_pct = _risk_params_for(symbol)
                # ── Time-based exit: recycle capital after MAX_HOLD_MINUTES ──
                entry_time = self._entry_times.get(symbol.upper())
                held_minutes = 0.0
                if entry_time is not None:
                    held_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
                extended = symbol.upper() in self._extended_holds
                max_hold = MAX_HOLD_EXTENDED_MINUTES if extended else MAX_HOLD_MINUTES
                if held_minutes >= max_hold:
                    # ── Trend extension (violence tier only): at the 30-min
                    #    mark, a position that is profitable AND still moving
                    #    in the trade's direction gets 60 min instead of an
                    #    automatic exit — cutting a trending winner at 30 min
                    #    caps exactly the upside this tier exists for.
                    if (not extended and _is_violence_symbol(symbol)
                            and _trend_extension_qualifies(mdf.df, pos)):
                        self._extended_holds.add(symbol.upper())
                        logger.info(
                            f"⏳ TREND-EXTENSION {symbol}: held {held_minutes:.0f}min, "
                            f"P&L={change_pct*100:+.1f}%, still trending — "
                            f"extending hold to {MAX_HOLD_EXTENDED_MINUTES}min"
                        )
                        continue
                    await self._handle_sell(symbol, price, 1.0, "time_exit")
                    self._extended_holds.discard(symbol.upper())
                    logger.info(f"⏰ TIME-EXIT {symbol}: held {held_minutes:.0f}min, P&L={change_pct*100:+.1f}%")
                    continue
                if is_short:
                    # Inverted risk math: a short loses when price RISES above
                    # entry (stop-loss) and profits when price falls below
                    # entry by the take-profit distance.
                    if change_pct >= stop_loss_pct:
                        await self._handle_sell(symbol, price, 1.0, "risk_stop")
                        logger.warning(f"🛑 STOP-LOSS (SHORT) {symbol}: {change_pct*100:+.1f}% above entry")
                    elif change_pct <= -take_profit_pct:
                        await self._handle_sell(symbol, price, 1.0, "risk_stop")
                        logger.info(f"🎯 TAKE-PROFIT (SHORT) {symbol}: {change_pct*100:+.1f}% below entry")
                else:
                    if change_pct <= -stop_loss_pct:
                        await self._handle_sell(symbol, price, 1.0, "risk_stop")
                        logger.warning(f"🛑 STOP-LOSS {symbol}: -{abs(change_pct)*100:.1f}%")
                    elif change_pct >= take_profit_pct:
                        await self._handle_sell(symbol, price, 1.0, "risk_stop")
                        logger.info(f"🎯 TAKE-PROFIT {symbol}: +{change_pct*100:.1f}%")
            except Exception as e:
                logger.error(f"Risk check error {symbol}: {e}")
    async def _eod_liquidate(self):
        """Sell all open TURBO positions for mandatory end-of-day liquidation.
        Longs are sold; shorts are bought to cover.  Either way every turbo
        position is flat before the close."""
        positions = await self.broker.get_positions()
        # Filter to turbo symbols only — never liquidate main trader's positions
        positions = [p for p in positions if p.get("symbol", "").upper() in {s.upper() for s in SYMBOLS}]
        count = len(positions)
        if count == 0:
            logger.info("⏰ Mandatory EOD liquidation — no positions to close")
            return

        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty == 0:
                continue
            is_short = qty < 0
            abs_qty = abs(qty)
            side = OrderSide.BUY if is_short else OrderSide.SELL
            action = "COVER" if is_short else "SELL"
            # ── Cancel protective stop before liquidating ─────────
            if not await self._cancel_protective_stops(sym):
                logger.warning("⏰ EOD %s %s deferred: cancellation not confirmed", action, sym)
                continue
            logger.info(f"⏰ EOD closing {sym}: {abs_qty} shares ({'short' if is_short else 'long'})...")
            order = Order(
                symbol=sym,
                side=side,
                quantity=abs_qty,
                order_type=OrderType.MARKET,
                client_id=_turbo_client_id(sym, side),
            )
            result = await self.broker.place_order(order)
            if not is_order_alive(result.status):
                # Stop was cancelled above — restore it so the position
                # doesn't sit naked overnight if the EOD exit fails.
                logger.warning(
                    "⏰ EOD %s %s rejected (%s); restoring protective stop and keeping position",
                    action, sym, result.status,
                )
                await self._place_protective_stop(
                    sym, abs_qty, float(p.get("avg_entry_price", 0)) or 0,
                    is_short=is_short,
                )
                continue
            self.pm.close_position(sym, exit_price=float(p.get("current_price", 0)), exit_reason="eod")

        logger.info(f"⏰ Mandatory EOD liquidation — {count} positions closed.")

    async def shutdown(self, close_broker: bool = True):
        """Liquidate all TURBO positions and report P&L; optionally retain broker."""
        logger.info("🚀 TURBO SHUTDOWN — Liquidating all turbo positions")

        # Close all TURBO positions only (never touch main trader's symbols)
        positions = await self.broker.get_positions()
        positions = [p for p in positions if p.get("symbol", "").upper() in {s.upper() for s in SYMBOLS}]
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty == 0:
                continue
            is_short = qty < 0
            abs_qty = abs(qty)
            side = OrderSide.BUY if is_short else OrderSide.SELL
            action = "COVER" if is_short else "SELL"
            # ── Cancel protective stop before liquidating ─────────
            if not await self._cancel_protective_stops(sym):
                logger.warning("Shutdown %s %s deferred: cancellation not confirmed", action, sym)
                continue
            logger.info(f"Closing {sym}: {abs_qty} shares ({'short' if is_short else 'long'})...")
            order = Order(
                symbol=sym,
                side=side,
                quantity=abs_qty,
                order_type=OrderType.MARKET,
                client_id=_turbo_client_id(sym, side),
            )
            result = await self.broker.place_order(order)
            if not is_order_alive(result.status):
                # Stop was cancelled above — restore it so the position
                # doesn't run naked if the shutdown exit fails.
                logger.warning(
                    "Shutdown %s %s rejected (%s); restoring protective stop",
                    action, sym, result.status,
                )
                await self._place_protective_stop(
                    sym, abs_qty, float(p.get("avg_entry_price", 0)) or 0,
                    is_short=is_short,
                )

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
        logger.info("🚀 TURBO SESSION COMPLETE")
        if start_equity is None or start_equity <= 0:
            logger.info("Start:  UNKNOWN (no baseline captured)")
        else:
            logger.info(f"Start:  ${start_equity:,.2f}")
        if end_equity is None:
            logger.info("End:    UNKNOWN (account fetch failed)")
            logger.info("P&L:    UNKNOWN (account fetch timed out)")
        elif start_equity is None or start_equity <= 0:
            logger.info(f"End:    ${end_equity:,.2f}")
            logger.info("P&L:    UNKNOWN (no session start baseline)")
        else:
            pnl = end_equity - start_equity
            pnl_pct = (pnl / start_equity * 100)
            logger.info(f"End:    ${end_equity:,.2f}")
            logger.info(f"P&L:    ${pnl:+,.2f}  ({pnl_pct:+.2f}%)")
            if pnl_pct > 0:
                logger.info("🔥 TURBO PROFIT — account growing!")
            elif pnl_pct < 0:
                logger.info("💥 TURBO LOSS — aggressive mode took a hit")
        logger.info(f"Log:    {log_file}")
        logger.info("=" * 60)
        # Reset the day baseline so the next session re-captures it.  The
        # persisted state file is date-stamped, so it is ignored on a new day
        # and re-used on a same-day restart (stable day P&L).
        self.start_equity = 0.0
        if close_broker:
            await self.broker.close()


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = TurboTrader()
    asyncio.run(trader.run())
