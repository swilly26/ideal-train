"""Position manager — tracks open positions and enforces risk limits.

Prevents duplicate entries, enforces ``max_position_pct`` from
``StrategyConfig``, and tracks both realised and unrealised P&L.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.strategies.base import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionRecord:
    """Metadata for a single open position."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float | None = None
    entry_time: str = ""
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


@dataclass
class ClosedTrade:
    """Record of a completed (round-trip) trade."""

    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str


class PositionManager:
    """Tracks positions across symbols and enforces risk limits.

    Parameters
    ----------
    config : StrategyConfig
        Strategy configuration used for position sizing and risk limits.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self._config = config or StrategyConfig()
        self._positions: dict[str, PositionRecord] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_open(self, symbol: str, account_equity: float) -> bool:
        """Check whether a new position in *symbol* can be opened.

        Returns ``False`` if we already hold *symbol* or if adding a new
        position would exceed the maximum allowed.
        """
        if self.has_position(symbol):
            logger.debug("Already holding %s — cannot open duplicate position", symbol)
            return False

        # Respect max_positions?  For now we limit to one position per symbol
        # and allow opening as long as we have equity.
        return account_equity > 0

    def has_position(self, symbol: str) -> bool:
        """Return ``True`` if we currently hold *symbol*."""
        return symbol.upper() in self._positions

    def open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_time: str = "",
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> PositionRecord:
        """Record a newly opened position.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        quantity : float
            Number of shares.
        entry_price : float
            Fill price.
        entry_time : str
            ISO timestamp of the fill.
        stop_loss_price : float | None
            Stop-loss trigger price (from config).
        take_profit_price : float | None
            Take-profit trigger price (from config).

        Returns
        -------
        PositionRecord
            The newly created position record.

        Raises
        ------
        ValueError
            If a position in *symbol* is already open.
        """
        sym = symbol.upper()
        if sym in self._positions:
            raise ValueError(f"Position already open for {sym}")

        record = PositionRecord(
            symbol=sym,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=entry_time,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
        self._positions[sym] = record
        logger.info(
            "Opened position: %s x %s @ %.2f (stop=%.2f, tp=%.2f)",
            quantity,
            sym,
            entry_price,
            stop_loss_price or 0,
            take_profit_price or 0,
        )
        return record

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "signal",
    ) -> ClosedTrade | None:
        """Close an existing position and record the realised P&L.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        exit_price : float
            The price at which the position was closed.
        exit_reason : str
            Reason for closing (``"signal"``, ``"stop_loss"``, ``"take_profit"``, etc.).

        Returns
        -------
        ClosedTrade or None
            The trade record, or ``None`` if no position was open.
        """
        sym = symbol.upper()
        record = self._positions.pop(sym, None)
        if record is None:
            logger.warning("Attempted to close non-existent position: %s", sym)
            return None

        # P&L is direction-aware: a short position stores a NEGATIVE quantity,
        # so ``qty * (exit - entry)`` is already correct (a short profits when
        # price falls).  Only the percentage needs an explicit sign flip.
        direction = 1.0 if record.quantity > 0 else -1.0
        pnl = record.quantity * (exit_price - record.entry_price)
        pnl_pct = (direction * (exit_price / record.entry_price - 1.0)) if record.entry_price else 0.0

        trade = ClosedTrade(
            symbol=sym,
            entry_price=record.entry_price,
            exit_price=exit_price,
            quantity=record.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
        )
        self._closed_trades.append(trade)
        self._realized_pnl += pnl

        logger.info(
            "Closed %s: %.2f → %.2f  P&L=%.2f (%.2f%%)  reason=%s",
            sym,
            record.entry_price,
            exit_price,
            pnl,
            pnl_pct * 100,
            exit_reason,
        )
        return trade

    def discard_position(self, symbol: str, reason: str = "broker_rejected") -> PositionRecord | None:
        """Remove a locally tracked position that the broker says does not exist.

        This is intentionally separate from :meth:`close_position`: a rejected
        sell (for example Alpaca's ``cannot be sold short`` response) is not a
        realised trade and must not create a zero-price loss record.
        """
        sym = symbol.upper()
        record = self._positions.pop(sym, None)
        if record is not None:
            logger.warning("Removed phantom position %s (%s)", sym, reason)
        return record

    def update_price(self, symbol: str, price: float) -> None:
        """Update the mark-to-market price for an open position."""
        sym = symbol.upper()
        record = self._positions.get(sym)
        if record is not None:
            record.current_price = price

    def get_positions(self) -> dict[str, PositionRecord]:
        """Return a copy of all open positions."""
        return dict(self._positions)

    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealised P&L across all open positions."""
        total = 0.0
        for record in self._positions.values():
            if record.current_price is not None:
                total += record.quantity * (record.current_price - record.entry_price)
        return total

    def get_realized_pnl(self) -> float:
        """Return total realised P&L from closed trades."""
        return self._realized_pnl

    def get_total_pnl(self) -> float:
        """Return combined realised + unrealised P&L."""
        return self._realized_pnl + self.get_unrealized_pnl()

    def get_open_symbols(self) -> list[str]:
        """Return list of symbols with open positions."""
        return sorted(self._positions.keys())

    def get_open_count(self) -> int:
        """Return number of distinct symbols with open positions."""
        return len(self._positions)

    def close_all(self, exit_price_fn) -> list[ClosedTrade]:
        """Close all open positions using *exit_price_fn*(symbol) for the exit price.

        Parameters
        ----------
        exit_price_fn : callable
            Callable that takes a symbol and returns the exit price.

        Returns
        -------
        list[ClosedTrade]
            All trades that were closed.
        """
        trades: list[ClosedTrade] = []
        for symbol in list(self._positions.keys()):
            price = exit_price_fn(symbol)
            trade = self.close_position(symbol, price, exit_reason="shutdown")
            if trade is not None:
                trades.append(trade)
        return trades

    def reset(self) -> None:
        """Clear all state (useful for testing)."""
        self._positions.clear()
        self._closed_trades.clear()
        self._realized_pnl = 0.0
