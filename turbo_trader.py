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
from src.execution.alpaca_broker import AlpacaBroker, is_order_alive, _is_duplicate_client_order_id_error
from src.execution.broker import Order, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.liquidity_sweep import LiquiditySweepStrategy
from src.strategies.indicators import sma, z_score

# ── Configuration ──────────────────────────────────────────────────
SYMBOLS = ["SOXL", "TQQQ", "FNGU", "SPXL"]  # 3x leveraged ETFs (LABU dropped — data errors)
CHECK_INTERVAL = 60
CONFIDENCE_THRESHOLD = 0.4   # Lower threshold = more entries
MAX_POSITIONS = 3            # Stay focused — fewer, bigger bets
POSITION_SIZE_PCT = 0.25     # 25% per position (only 3 max = 75% deployed)
MANDATORY_CLOSE_MINUTES = 30  # Liquidate all positions 30 min before market close

STRATEGY_CONFIG = StrategyConfig(
    entry_threshold=0.5,
    exit_threshold=0.05,        # Tighter exit = recycle capital faster
    stop_loss_pct=0.06,         # Wider stop-loss (6%) for ETF volatility
    take_profit_pct=0.08,       # Wider take-profit (8%)
    max_position_pct=POSITION_SIZE_PCT,
    extra={"lookback": 20, "std_dev_multiplier": 2.0},
)

# Momentum strategy config (separate thresholds for the dual-strategy approach)
MOMENTUM_CONFIG = {
    "ma_period": 10,            # Shorter MA for faster crossover signals
    "trend_periods": 5,         # Periods for trend strength (statistical significance)
    "rsi_period": 14,
    "rsi_threshold": 40,        # RSI > 40 = momentum not oversold (lowered from 50)
}

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
            logger.info(
                "🔍 MOMENTUM %s: BUY conf=%.4f (price=%.2f, MA=%.2f, RSI=%.1f)",
                symbol, confidence, cur_close, cur_ma, cur_rsi,
            )

        # ── SELL signal: MA cross-under OR price < MA and weak RSI ──
        elif (cur_close < cur_ma and prev_close >= prev_ma) or \
             (cur_close < cur_ma and cur_rsi < 60):
            # Cross below MA — momentum broken, or price below MA with weakening RSI
            confidence = min(1.0, abs(cur_close - cur_ma) / (abs(cur_ma) + 1e-9) * 10)
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
            logger.info(
                "🔍 MOMENTUM %s: SELL conf=%.4f (price=%.2f, MA=%.2f, RSI=%.1f)",
                symbol, confidence, cur_close, cur_ma, cur_rsi,
            )

    return signals


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
        self.liquidity_sweep = LiquiditySweepStrategy(symbols=SYMBOLS)
        self.levels_cache: dict[str, dict[str, float | None]] = {}
        self._ls_symbols: set[str] = set()  # symbols entered via liquidity sweep
        self.pm = PositionManager(STRATEGY_CONFIG)
        self.day_trades: list[dict] = []
        self.start_equity = 0.0

    # ── Market open wait ─────────────────────────────────────────────

    async def wait_for_market_open(self):
        """Block until the market opens."""
        logger.info("🚀 Waiting for market to open (9:30 AM ET)...")
        while True:
            try:
                if await self.broker.is_market_open():
                    logger.info("✅ Market is OPEN — starting TURBO trading")
                    account = await self.broker.get_account()
                    self.start_equity = float(account.get("equity", 100_000))
                    logger.info(f"   Starting equity: ${self.start_equity:,.2f}")
                    return
            except Exception as e:
                logger.warning(f"Market check failed: {e}")

            remaining = self._seconds_until_open()
            sleep_time = min(30, max(5, remaining // 2))
            logger.debug(f"   Next check in {sleep_time}s")
            await asyncio.sleep(sleep_time)

    @staticmethod
    def _seconds_until_open() -> float:
        """Seconds until next 9:30 AM ET market open (13:30 UTC)."""
        now = datetime.now(timezone.utc)
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
        logger.info(f"Symbols: {SYMBOLS} (3x Leveraged ETFs)")
        logger.info(f"Strategy: MeanReversion + Momentum + LiquiditySweep | Confidence ≥ {CONFIDENCE_THRESHOLD}")
        logger.info(f"Max positions: {MAX_POSITIONS} | Size: {POSITION_SIZE_PCT*100:.0f}% equity")
        logger.info(f"Stop-loss: {STRATEGY_CONFIG.stop_loss_pct*100:.0f}% | "
                    f"Take-profit: {STRATEGY_CONFIG.take_profit_pct*100:.0f}%")
        logger.info(f"⚠️  Mandatory EOD liquidation {MANDATORY_CLOSE_MINUTES} min before close")
        logger.info("=" * 60)

        # ── Cancel stale orders from prior sessions ──────────────────
        # NOTE: cancel_all_orders() cancels EVERY open order on the account
        # (both turbo's AND the main trader's). This is a known side effect of
        # sharing an Alpaca paper account. We log a warning so operators are aware.
        # This also cancels any stale protective stop orders from prior sessions,
        # which will be re-created below during position sync.
        logger.info("Cancelling any stale orders from prior sessions…")
        cancelled = await self.broker.cancel_all_orders()
        logger.warning(
            "Cancelled %d open order(s) — this includes ALL orders on the shared "
            "account (turbo + main trader). Main trader may need to re-submit orders.",
            cancelled,
        )

        # Wait for open
        await self.wait_for_market_open()

        # ── Calculate liquidity sweep levels ────────────────────────
        logger.info("STEP 1/4: Calculating liquidity sweep levels…")
        await self.liquidity_sweep.calculate_levels(self.provider)
        self.levels_cache = self.liquidity_sweep.levels
        logger.info("STEP 1/4: Levels calculated for %d symbols", len(self.levels_cache))

        # ── Sync positions from Alpaca at startup ────────────────────
        logger.info("STEP 2/4: Syncing positions from broker…")
        await self._sync_positions_from_broker()
        logger.info("STEP 2/4: Position sync complete — %d open positions tracked",
                     self.pm.get_open_count())

        # ── Log inherited position state ────────────────────────────
        for sym in self.pm.get_open_symbols():
            pos = self.pm.get_positions().get(sym)
            if pos:
                logger.info("  Inherited: %s x %s @ $%.2f", pos.quantity, sym, pos.entry_price)

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
        """One polling cycle: fetch → triple-strategy signals → execute."""

        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=60)  # deeper lookback for MA/RSI

        # Track which symbols have active liquidity sweep positions
        # (prune any that have been closed)
        self._ls_symbols &= set(self.pm.get_open_symbols())

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
                if data.empty or len(data) < 25:
                    continue

                # ── Generate signals from all three strategies ───────
                mr_signals = self.mean_reversion.generate_signals(data)
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

                if best_signal.signal_type == SignalType.BUY:
                    await self._handle_buy(symbol, current_price, best_signal.confidence, strategy_name)
                    # Track liquidity sweep entries for the max-1-LS-position guard
                    if strategy_name == "liquidity_sweep":
                        self._ls_symbols.add(symbol.upper())
                elif best_signal.signal_type == SignalType.SELL:
                    await self._handle_sell(symbol, current_price, best_signal.confidence, strategy_name)
                    # If we sold an LS position, remove from tracking
                    self._ls_symbols.discard(symbol.upper())

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
        equity = float(account.get("equity", 100_000))
        if not self.pm.can_open(symbol, equity):
            return

        value = equity * POSITION_SIZE_PCT
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
            # Use the fill price if available, otherwise fall back to the signal price
            fill_price = result.filled_avg_price if result.filled_avg_price else price
            self.pm.open_position(symbol, qty, fill_price)
            strat_tag = f" [{strategy}]" if strategy else ""
            logger.info(
                f"🚀 BUY  {symbol}: {qty:.1f} shares @ ${fill_price:.2f} = ${value:,.2f} | "
                f"conf={confidence:.2f}{strat_tag} | order={result.order_id[:8]}"
            )
            # ── Place GTC protective stop at broker ─────────────────
            await self._place_protective_stop(symbol, qty, fill_price)
        else:
            logger.warning(f"❌ BUY {symbol} REJECTED: {result.status}")

    async def _handle_sell(self, symbol: str, price: float, confidence: float, strategy: str = ""):
        if not self.pm.has_position(symbol):
            return

        pos = self.pm.get_positions().get(symbol.upper())
        if pos is None:
            return
        qty = pos.quantity
        entry = pos.entry_price

        # ── Cancel protective stop before selling ───────────────────
        await self._cancel_protective_stops(symbol)

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=qty,
            order_type=OrderType.MARKET,
            client_id=_turbo_client_id(symbol, OrderSide.SELL),
        )
        result = await self.broker.place_order(order)

        pnl = (price - entry) * qty if entry else 0
        pnl_pct = ((price / entry) - 1.0) * 100 if entry else 0
        self.pm.close_position(symbol, price)

        strat_tag = f" [{strategy}]" if strategy else ""
        logger.info(
            f"📉 SELL {symbol}: {qty:.1f} shares @ ${price:.2f} | "
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
                qty = pos_data["qty"]

                # ── Cleanup: sell inherited underwater positions (>2% loss) ──
                if current_price > 0 and entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct < -0.02:
                        logger.info(
                            "🧹 Cleanup SELL %s: inherited at $%.2f, now $%.2f = %.1f%%",
                            sym, entry_price, current_price, pnl_pct * 100,
                        )
                        order = Order(
                            symbol=sym,
                            side=OrderSide.SELL,
                            quantity=qty,
                            order_type=OrderType.MARKET,
                            client_id=_turbo_client_id(sym, OrderSide.SELL),
                        )
                        await self.broker.place_order(order)
                        continue  # Don't add to PM — we just sold it

                self.pm.open_position(
                    symbol=sym,
                    quantity=qty,
                    entry_price=entry_price,
                )
                logger.info(
                    "  + Added %s: %s shares @ $%.2f (sync)",
                    sym, qty, entry_price,
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

    # ── Broker-level protective stops ────────────────────────────────

    async def _place_protective_stop(self, symbol: str, qty: float, entry_price: float):
        """Place a GTC stop-loss sell order at the broker.

        This order survives process death and sandbox cycling — Alpaca holds
        it until triggered or cancelled.  The stop price is set at
        ``entry_price * (1 - stop_loss_pct)`` (currently 6% below entry).
        """
        await asyncio.sleep(2.0)
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce

        # GTC orders require whole shares; fractional orders must be DAY.
        qty = int(qty)
        if qty == 0:
            logger.warning("🛡️  STOP %s: position too small for protective stop (qty < 1 share)", symbol.upper())
            return

        stop_price = round(entry_price * (1 - STRATEGY_CONFIG.stop_loss_pct), 2)
        sym = symbol.upper()
        client_id = _turbo_stop_client_id(sym)

        req = StopOrderRequest(
            symbol=sym,
            qty=qty,
            side=AlpacaSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price,
            client_order_id=client_id,
        )

        try:
            result = await asyncio.to_thread(
                self.broker._trading_client.submit_order, req
            )
            logger.info(
                "🛡️  STOP %s: GTC stop-loss at $%.2f (entry=%.2f, -%.0f%%) — order=%s",
                sym, stop_price, entry_price, STRATEGY_CONFIG.stop_loss_pct * 100,
                str(result.id)[:8],
            )
        except Exception as exc:
            if _is_duplicate_client_order_id_error(exc):
                logger.info("🛡️  STOP %s: already placed (duplicate client_order_id)", sym)
            else:
                logger.error("🛡️  STOP %s: failed to place — %s", sym, exc)

    async def _cancel_protective_stops(self, symbol: str):
        """Cancel all open orders for *symbol* — stop-loss and take-profit.

        Called before selling a position so the GTC stop doesn't trigger
        after the position is closed.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        sym = symbol.upper()
        try:
            flt = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[sym],
            )
            open_orders = await asyncio.to_thread(
                lambda: self.broker._trading_client.get_orders(filter=flt),
            )
        except Exception as exc:
            logger.warning("Failed to fetch open orders for %s: %s", sym, exc)
            return

        cancelled = 0
        for o in open_orders:
            try:
                await asyncio.to_thread(
                    self.broker._trading_client.cancel_order_by_id, str(o.id)
                )
                cancelled += 1
                logger.debug("  Cancelled order %s for %s", str(o.id)[:8], sym)
            except Exception as exc:
                logger.warning("  Failed to cancel order %s: %s", str(o.id)[:8], exc)

        if cancelled:
            logger.info("🗑️  Cancelled %d protective order(s) for %s", cancelled, sym)

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
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        try:
            flt = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = await asyncio.to_thread(
                lambda: self.broker._trading_client.get_orders(filter=flt),
            )
        except Exception as exc:
            logger.warning("Cannot verify protective stops — order fetch failed: %s", exc)
            return

        # Build a set of symbols that already have an open sell order
        covered_symbols: set[str] = set()
        for o in open_orders:
            o_sym = str(o.symbol).upper()
            o_side = str(o.side).upper()
            if o_sym in turbo_set and o_side == "SELL":
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

            logger.warning(
                "🛡️  %s: NO protective stop found for inherited position "
                "(%s shares @ $%.2f) — placing one now",
                sym, pos.quantity, pos.entry_price,
            )
            await self._place_protective_stop(sym, int(pos.quantity), pos.entry_price)

    async def _check_risk_stops(self):
        """Check stop-loss / take-profit for open positions."""
        for symbol in list(self.pm.get_open_symbols()):
            if symbol.upper() not in {s.upper() for s in SYMBOLS}:
                continue  # Safety: never touch non-turbo symbols
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
                    await self._handle_sell(symbol, price, 1.0, "risk_stop")
                    logger.warning(f"🛑 STOP-LOSS {symbol}: -{abs(change_pct)*100:.1f}%")
                elif change_pct >= STRATEGY_CONFIG.take_profit_pct:
                    await self._handle_sell(symbol, price, 1.0, "risk_stop")
                    logger.info(f"🎯 TAKE-PROFIT {symbol}: +{change_pct*100:.1f}%")
            except Exception as e:
                logger.error(f"Risk check error {symbol}: {e}")

    async def _eod_liquidate(self):
        """Sell all open TURBO positions for mandatory end-of-day liquidation."""
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
            if qty > 0:
                # ── Cancel protective stop before liquidating ─────────
                await self._cancel_protective_stops(sym)
                logger.info(f"⏰ EOD closing {sym}: {qty} shares...")
                order = Order(
                    symbol=sym,
                    side=OrderSide.SELL,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    client_id=_turbo_client_id(sym, OrderSide.SELL),
                )
                await self.broker.place_order(order)
                self.pm.close_position(sym, exit_price=float(p.get("current_price", 0)), exit_reason="eod")

        logger.info(f"⏰ Mandatory EOD liquidation — {count} positions closed.")

    async def shutdown(self):
        """🚀 TURBO SHUTDOWN — Liquidate all TURBO positions and report final P&L."""
        logger.info("🚀 TURBO SHUTDOWN — Liquidating all turbo positions")

        # Close all TURBO positions only (never touch main trader's symbols)
        positions = await self.broker.get_positions()
        positions = [p for p in positions if p.get("symbol", "").upper() in {s.upper() for s in SYMBOLS}]
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                # ── Cancel protective stop before liquidating ─────────
                await self._cancel_protective_stops(sym)
                logger.info(f"Closing {sym}: {qty} shares...")
                order = Order(
                    symbol=sym,
                    side=OrderSide.SELL,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    client_id=_turbo_client_id(sym, OrderSide.SELL),
                )
                await self.broker.place_order(order)

        # Final account state
        account = await self.broker.get_account()
        end_equity = float(account.get("equity", 0))
        pnl = end_equity - self.start_equity
        pnl_pct = (pnl / self.start_equity * 100) if self.start_equity > 0 else 0

        logger.info("=" * 60)
        logger.info("🚀 TURBO SESSION COMPLETE")
        logger.info(f"Start:  ${self.start_equity:,.2f}")
        logger.info(f"End:    ${end_equity:,.2f}")
        logger.info(f"P&L:    ${pnl:+,.2f}  ({pnl_pct:+.2f}%)")
        if pnl_pct > 0:
            logger.info("🔥 TURBO PROFIT — account growing!")
        elif pnl_pct < 0:
            logger.info("💥 TURBO LOSS — aggressive mode took a hit")
        logger.info(f"Log:    {log_file}")
        logger.info("=" * 60)

        await self.broker.close()


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = TurboTrader()
    asyncio.run(trader.run())
