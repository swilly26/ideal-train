"""Live (paper) trading runner — the main orchestration loop.

Wires together a data provider, strategy, broker, and position manager
into a continuous intraday trading loop.  Polls market data at a
configurable interval, generates signals, filters by confidence, and
executes orders via the broker.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Type

import pandas as pd

from src.execution.broker import Broker, Order, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.execution.alpaca_broker import is_order_alive
from src.strategies.base import SignalType, Strategy, StrategyConfig
from src.data.provider import DataProvider

logger = logging.getLogger(__name__)


class LiveTradingRunner:
    """Orchestrates the full data → strategy → execution loop.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to trade (e.g. ``["AAPL", "MSFT"]``).
    strategy_cls : type[Strategy]
        Strategy class to use for signal generation.
    broker : Broker
        A broker implementation (e.g. ``AlpacaBroker``).
    data_provider : DataProvider
        Market data provider for polling quotes.
    config : StrategyConfig | None
        Strategy configuration (AI-optimised or default).
    check_interval_seconds : float
        Polling interval in seconds (default 60).
    confidence_threshold : float
        Minimum signal confidence to act on (default 0.5).
    max_positions : int
        Maximum number of concurrent open positions (default 5).
    close_on_shutdown : bool
        If ``True``, close all positions on SIGINT/SIGTERM.
    cancel_orders_on_shutdown : bool
        If ``True``, best-effort cancel all open orders on shutdown.
    """

    def __init__(
        self,
        symbols: list[str],
        strategy_cls: Type[Strategy],
        broker: Broker,
        data_provider: DataProvider,
        config: StrategyConfig | None = None,
        check_interval_seconds: float = 60.0,
        confidence_threshold: float = 0.5,
        max_positions: int = 5,
        close_on_shutdown: bool = False,
        cancel_orders_on_shutdown: bool = True,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._strategy_cls = strategy_cls
        self._broker = broker
        self._data_provider = data_provider
        self._config = config or StrategyConfig()
        self._check_interval = check_interval_seconds
        self._confidence_threshold = confidence_threshold
        self._max_positions = max_positions
        self._close_on_shutdown = close_on_shutdown
        self._cancel_on_shutdown = cancel_orders_on_shutdown

        self._position_manager = PositionManager(self._config)
        self._running = False
        self._strategy: Strategy | None = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the live trading loop.

        Runs until ``stop()`` is called or a shutdown signal is received.
        """
        self._running = True
        self._strategy = self._strategy_cls(config=self._config)

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                # Windows / some environments don't support add_signal_handler
                pass

        logger.info(
            "Live runner started: symbols=%s strategy=%s interval=%.1fs",
            self._symbols,
            self._strategy_cls.__name__,
            self._check_interval,
        )

        try:
            while self._running:
                try:
                    await self._tick()
                except Exception as exc:
                    logger.error("Error in tick: %s", exc, exc_info=True)

                await asyncio.sleep(self._check_interval)
        finally:
            await self._shutdown()

    def stop(self) -> None:
        """Gracefully signal the main loop to stop."""
        logger.info("Stop requested — finishing current tick…")
        self._running = False

    def _handle_shutdown(self) -> None:
        """Signal handler callback — schedules stop."""
        logger.info("Received shutdown signal")
        self._running = False

    # ------------------------------------------------------------------
    # Tick logic
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """Execute one polling cycle: fetch data, generate signals, execute."""
        # 1. Check if market is open (best-effort — only if broker supports it)
        market_open = True
        if hasattr(self._broker, "is_market_open"):
            market_open = await self._broker.is_market_open()
        if not market_open:
            logger.debug("Market is closed — skipping tick")
            return

        # 2. Fetch latest bars for each symbol
        now = datetime.now(timezone.utc)
        bars_lookback = now - timedelta(minutes=30)  # 30-min lookback

        for symbol in self._symbols:
            # Skip if we can't open more positions and don't hold this one
            if (
                self._position_manager.get_open_count() >= self._max_positions
                and not self._position_manager.has_position(symbol)
            ):
                continue

            data = await self._fetch_data(symbol, bars_lookback, now)
            if data is None or data.empty:
                continue

            # 3. Generate signals
            signals = self._strategy.generate_signals(data)
            if not signals:
                continue

            # Only act on the latest signal
            latest = signals[-1]
            logger.debug(
                "Signal: %s %s confidence=%.2f",
                latest.symbol,
                latest.signal_type.value,
                latest.confidence,
            )

            if latest.confidence < self._confidence_threshold:
                logger.debug("Signal below confidence threshold — ignored")
                continue

            # 4. Act on signal
            await self._execute_signal(latest, data)

        # 5. Check stop-loss / take-profit for open positions
        await self._check_risk_stops(data)

        # 6. Log P&L summary
        self._log_pnl()

    async def _fetch_data(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        """Fetch recent bars and return a DataFrame."""
        try:
            result = await self._data_provider.fetch_bars(
                symbol, start, end, timeframe="1min"
            )
            return result.df
        except Exception as exc:
            logger.error("Failed to fetch data for %s: %s", symbol, exc)
            return None

    async def _execute_signal(
        self, signal, data: pd.DataFrame
    ) -> None:
        """Convert a signal into a broker order and update the position manager."""
        symbol = signal.symbol.upper()
        latest_bar = data.iloc[-1]
        current_price = float(latest_bar["close"])

        if signal.signal_type == SignalType.BUY:
            # Don't buy if already in position
            if self._position_manager.has_position(symbol):
                return

            # Don't buy if at max positions
            if self._position_manager.get_open_count() >= self._max_positions:
                return

            # Check equity
            acct = await self._broker.get_account()
            equity = float(acct.get("equity", 0))
            if not self._position_manager.can_open(symbol, equity):
                return

            max_position_value = equity * self._config.max_position_pct
            quantity = max_position_value / current_price if current_price > 0 else 0
            if quantity <= 0:
                return

            # Calculate stop-loss / take-profit prices
            sl_price = current_price * (1.0 - self._config.stop_loss_pct)
            tp_price = current_price * (1.0 + self._config.take_profit_pct)

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            result = await self._broker.place_order(order)
            logger.info(
                "BUY %s x %.4f @ %.2f — %s (id=%s)",
                symbol,
                quantity,
                current_price,
                result.status,
                result.order_id,
            )

            if is_order_alive(result.status):
                filled_qty = result.filled_quantity if result.filled_quantity > 0 else quantity
                self._position_manager.open_position(
                    symbol=symbol,
                    quantity=filled_qty,
                    entry_price=result.filled_avg_price or current_price,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                )

        elif signal.signal_type == SignalType.SELL:
            if not self._position_manager.has_position(symbol):
                return

            pos = self._position_manager.get_positions().get(symbol)
            if pos is None:
                return

            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
            )

            result = await self._broker.place_order(order)
            logger.info(
                "SELL %s x %.4f — %s (id=%s)",
                symbol,
                pos.quantity,
                result.status,
                result.order_id,
            )

            if not is_order_alive(result.status):
                logger.warning("SELL %s rejected (%s); keeping position tracked", symbol, result.status)
                return

            exit_price = result.filled_avg_price or current_price
            self._position_manager.close_position(
                symbol=symbol,
                exit_price=exit_price,
                exit_reason="signal",
            )

    async def _check_risk_stops(self, data: pd.DataFrame | None = None) -> None:
        """Check open positions for stop-loss / take-profit triggers.

        Fetches fresh bars **per symbol**: the tick-level ``data`` argument
        only covers the last-polled symbol, so using it to price other
        symbols' stops could trigger erroneous stop/take-profit exits on
        the wrong symbol.  Kept as an optional argument for callers that
        already hold the right data.
        """
        positions = self._position_manager.get_positions()
        if not positions:
            return

        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=5)

        for symbol, pos in list(positions.items()):
            try:
                mdf = await self._data_provider.fetch_bars(
                    symbol, start=lookback, end=now, timeframe="1min"
                )
                if mdf.df.empty:
                    continue
                bar = mdf.df.iloc[-1]
            except Exception as exc:
                logger.error("Failed to fetch risk data for %s: %s", symbol, exc)
                continue

            high = float(bar["high"])
            low = float(bar["low"])

            hit_stop = (
                pos.stop_loss_price is not None
                and low <= pos.stop_loss_price
            )
            hit_take = (
                pos.take_profit_price is not None
                and high >= pos.take_profit_price
            )

            if not hit_stop and not hit_take:
                continue

            exit_price: float
            reason: str
            if hit_stop and hit_take:
                exit_price = pos.stop_loss_price  # stop-loss takes priority
                reason = "stop_loss"
            elif hit_stop:
                exit_price = pos.stop_loss_price
                reason = "stop_loss"
            else:
                exit_price = pos.take_profit_price
                reason = "take_profit"

            logger.info("Risk stop triggered: %s %s @ %.2f", symbol, reason, exit_price)

            # Place market sell order
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
            )
            result = await self._broker.place_order(order)
            if not is_order_alive(result.status):
                logger.warning("Risk SELL %s rejected (%s); keeping position tracked", symbol, result.status)
                continue

            self._position_manager.close_position(
                symbol=symbol,
                exit_price=exit_price,
                exit_reason=reason,
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Graceful shutdown: optionally close positions or cancel orders."""
        logger.info("Shutting down live runner…")

        if self._close_on_shutdown:
            positions = self._position_manager.get_positions()
            for symbol, pos in positions.items():
                try:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        order_type=OrderType.MARKET,
                    )
                    result = await self._broker.place_order(order)
                    if not is_order_alive(result.status):
                        logger.warning("Shutdown SELL %s rejected (%s); keeping position tracked", symbol, result.status)
                        continue
                    exit_price = result.filled_avg_price or (pos.current_price or 0)
                    self._position_manager.close_position(
                        symbol=symbol,
                        exit_price=exit_price,
                        exit_reason="shutdown",
                    )
                except Exception as exc:
                    logger.error("Failed to close %s on shutdown: %s", symbol, exc)

        try:
            await self._data_provider.close()
        except Exception:
            pass

        try:
            await self._broker.close()
        except Exception:
            pass

        self._log_pnl()
        logger.info("Live runner shutdown complete.")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_pnl(self) -> None:
        """Log a one-line P&L summary."""
        realized = self._position_manager.get_realized_pnl()
        unrealized = self._position_manager.get_unrealized_pnl()
        total = realized + unrealized
        positions = self._position_manager.get_open_count()
        logger.info(
            "P&L: realized=%.2f  unrealized=%.2f  total=%.2f  positions=%d",
            realized,
            unrealized,
            total,
            positions,
        )

    # Make position manager accessible for testing / monitoring
    @property
    def position_manager(self) -> PositionManager:
        return self._position_manager
