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
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Project imports ────────────────────────────────────────────────
import src.strategies  # registers strategies
from src.data.yfinance_provider import YFinanceProvider
from src.execution.alpaca_broker import AlpacaBroker, is_order_alive
from src.execution.broker import Order, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.strategies.base import SignalType, StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy

# ── Configuration ──────────────────────────────────────────────────
SYMBOLS = ["NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO"]
CHECK_INTERVAL = 60  # seconds between polls
# NOTE: Analysis shows low-confidence trades (0.4-0.6 bucket) average +$33.46 —
# actually MORE profitable than high-confidence trades. More signals = more opportunities.
CONFIDENCE_THRESHOLD = 0.3
MAX_POSITIONS = 6
POSITION_SIZE_PCT = 0.15  # 15% of equity per position

# Market close in UTC (4 PM ET = 20:00 UTC)
MARKET_CLOSE_UTC_HOUR = 20
MARKET_CLOSE_UTC_MINUTE = 0
MANDATORY_CLOSE_MINUTES = 15  # Liquidate all positions 15 min before close

STRATEGY_CONFIG = StrategyConfig(
    entry_threshold=0.5,       # z-score to enter (lower = more trades)
    exit_threshold=0.1,        # z-score to exit
    stop_loss_pct=0.03,        # 3% stop-loss
    take_profit_pct=0.03,      # 3% take-profit
    max_position_pct=POSITION_SIZE_PCT,
    extra={"lookback": 20, "std_dev_multiplier": 2.0},
)

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

class LiveTrader:
    def __init__(self):
        self.broker = AlpacaBroker()
        self.provider = YFinanceProvider()
        self.strategy = MeanReversionStrategy(config=STRATEGY_CONFIG)
        self.pm = PositionManager(STRATEGY_CONFIG)
        self.day_trades: list[dict] = []
        self.start_equity = 0.0

    async def wait_for_market_open(self):
        """Block until the market opens."""
        logger.info("Waiting for market to open (9:30 AM ET)...")
        while True:
            try:
                if await self.broker.is_market_open():
                    logger.info("✅ Market is OPEN — starting trading")
                    account = await self.broker.get_account()
                    self.start_equity = float(account.get("equity", 100_000))
                    logger.info(f"   Starting equity: ${self.start_equity:,.2f}")
                    return
            except Exception as e:
                logger.warning(f"Market check failed: {e}")
            
            # Sleep 30 seconds between checks
            remaining = self._seconds_until_open()
            sleep_time = min(30, max(5, remaining // 2))
            logger.debug(f"   Next check in {sleep_time}s")
            await asyncio.sleep(sleep_time)

    @staticmethod
    def _seconds_until_open() -> float:
        """Seconds until next 9:30 AM ET market open."""
        now = datetime.now(timezone.utc)
        # Market opens at 13:30 UTC (9:30 ET) / 14:30 UTC during DST
        # Simple: target 13:30 UTC Mon-Fri
        target = now.replace(hour=13, minute=30, second=0, microsecond=0)
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

        # ── Layer 4: Cancel stale orders from prior sessions ─────────
        logger.info("Cancelling any stale orders from prior sessions…")
        cancelled = await self.broker.cancel_all_orders()
        logger.info(f"Cancelled {cancelled} stale order(s)")

        # Wait for open
        await self.wait_for_market_open()

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

        logger.info("STEP 2/3: Entering main tick loop…")
        tick = 0
        try:
            while True:
                tick += 1

                # ── EOD mandatory liquidation check ──────────────────
                if self._is_near_close():
                    logger.info(f"⏰ Within {MANDATORY_CLOSE_MINUTES} min of close — "
                                "triggering mandatory EOD liquidation")
                    await self._eod_liquidate()
                    break

                logger.debug("Tick %d: checking market open…", tick)
                market_open = await self.broker.is_market_open()
                if not market_open:
                    logger.info("⏹️  Market closed — shutting down")
                    break

                logger.debug("Tick %d: running strategy evaluation…", tick)
                await self._tick(tick)
                logger.debug("Tick %d: complete — sleeping %ds", tick, CHECK_INTERVAL)
                await asyncio.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await self.shutdown()

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
        equity = float(account.get("equity", 100_000))
        if not self.pm.can_open(symbol, equity):
            return

        value = equity * POSITION_SIZE_PCT
        qty = value / price if price > 0 else 0
        if qty < 1:
            return

        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET)
        result = await self.broker.place_order(order)

        if is_order_alive(result.status):
            self.pm.open_position(symbol, qty, price)
            logger.info(f"📈 BUY  {symbol}: {qty:.1f} shares @ ${price:.2f} = ${value:,.2f} | "
                        f"conf={confidence:.2f} | order={result.order_id[:8]}")
        else:
            logger.warning(f"❌ BUY {symbol} REJECTED: {result.status}")

    async def _handle_sell(self, symbol: str, price: float, confidence: float):
        if not self.pm.has_position(symbol):
            return

        pos = self.pm.get_positions().get(symbol.upper())
        if pos is None:
            return
        qty = pos.quantity
        entry = pos.entry_price
        order = Order(symbol=symbol, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
        result = await self.broker.place_order(order)

        pnl = (price - entry) * qty if entry else 0
        self.pm.close_position(symbol, price)

        logger.info(f"📉 SELL {symbol}: {qty:.1f} shares @ ${price:.2f} | "
                    f"P&L: ${pnl:,.2f} | conf={confidence:.2f} | order={result.order_id[:8]}")

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

        # Remove positions we track but broker doesn't have
        for sym in pm_symbols - set(broker_symbols):
            self.pm.close_position(sym, exit_price=0, exit_reason="sync_removed")
            logger.info("  - Removed stale %s (not on broker)", sym)
            removed += 1

        logger.info(
            "Synced %d positions from broker (%d added, %d removed)",
            len(broker_symbols), added, removed,
        )

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

        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                logger.info(f"⏰ EOD closing {sym}: {qty} shares...")
                order = Order(symbol=sym, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
                await self.broker.place_order(order)
                self.pm.close_position(sym, exit_price=float(p.get("current_price", 0)), exit_reason="eod")

        logger.info(f"⏰ Mandatory EOD liquidation — {count} positions closed.")

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

                if change_pct <= -STRATEGY_CONFIG.stop_loss_pct:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.warning(f"🛑 STOP-LOSS {symbol}: -{abs(change_pct)*100:.1f}%")
                elif change_pct >= STRATEGY_CONFIG.take_profit_pct:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.info(f"🎯 TAKE-PROFIT {symbol}: +{change_pct*100:.1f}%")
            except Exception as e:
                logger.error(f"Risk check error {symbol}: {e}")

    async def shutdown(self):
        """Graceful shutdown: close positions, report P&L."""
        logger.info("Shutting down...")

        # Close all positions
        positions = await self.broker.get_positions()
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                logger.info(f"Closing {sym}: {qty} shares...")
                order = Order(symbol=sym, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
                await self.broker.place_order(order)

        # Final account state
        account = await self.broker.get_account()
        end_equity = float(account.get("equity", 0))
        pnl = end_equity - self.start_equity
        pnl_pct = (pnl / self.start_equity * 100) if self.start_equity > 0 else 0

        logger.info("=" * 60)
        logger.info(f"SESSION COMPLETE")
        logger.info(f"Start: ${self.start_equity:,.2f}")
        logger.info(f"End:   ${end_equity:,.2f}")
        logger.info(f"P&L:   ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Log:   {log_file}")
        logger.info("=" * 60)

        await self.broker.close()


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.run())
