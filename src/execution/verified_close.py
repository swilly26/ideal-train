"""Verified position close — never book realised P&L from the order
submission response alone.

Background (incidents 2026-09-08 and 2026-09-09): the cleanup path placed
MARKET SELLs, saw ``is_order_alive(new)`` and immediately logged
"Closed ... P&L=..." / "cleanup complete — N liquidated" — but the orders
stayed NEW/ACCEPTED pre-open and never filled until the next open.  The
engine booked ~-$939 across main+turbo while the broker realised ~-$4,842
(SOXL gapped through its -6% stop).  The books were wrong by ~$3.9k and
"liquidated" claims were false.

This module centralises the correct flow for every close path:

1. Place the market close order.
2. If the broker already reports it FILLED → book realised P&L at the
   broker's average fill price (never the last mark).
3. If the broker reports it pending (NEW / ACCEPTED / PARTIALLY_FILLED)
   → poll for a bounded time for the fill.
   * filled → book at the fill price;
   * terminal failure → rejected, position stays tracked, no P&L;
   * still pending after the bound → do NOT book anything, mark the
     symbol as having a pending close (guards against double-sell) and
     let the next position sync reconcile from broker fill history.
4. A close that is already pending for a symbol (previous attempt not
   confirmed) is never re-submitted — no duplicate sell orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.execution.alpaca_broker import is_order_alive
from src.execution.broker import Order, OrderSide, OrderType
from src.execution.position_manager import ClosedTrade, PositionManager

logger = logging.getLogger(__name__)

# Order statuses that count as "the broker actually filled" for accounting.
_FILLED_STATUSES = frozenset({"filled", "done_for_day"})


@dataclass
class CloseOutcome:
    """Result of a verified close attempt.

    ``status`` is one of:

    ``"filled"``
        The broker confirmed a fill; realised P&L was booked at
        :attr:`fill_price` (the broker's average fill price).
    ``"pending"``
        The close order is live but unfilled; NO P&L was booked and the
        position is still tracked.  Reconcile at the next position sync.
    ``"rejected"``
        The order died (rejected/canceled/expired) or submission failed;
        NO P&L was booked and the position is still tracked.
    ``"no_position"``
        Nothing was tracked for the symbol.
    """

    status: str
    fill_price: float | None = None
    order_id: str = ""
    message: str = ""
    trade: ClosedTrade | None = None

    @property
    def confirmed(self) -> bool:
        """True when the broker verified a fill and P&L was booked."""
        return self.status == "filled"


async def close_position_verified(
    pm: PositionManager,
    broker,
    symbol: str,
    exit_reason: str = "signal",
    wait_timeout: float = 8.0,
    poll_interval: float = 0.25,
    client_id: str | None = None,
) -> CloseOutcome:
    """Close *symbol* and only book realised P&L after the broker confirms a fill.

    Parameters
    ----------
    pm : PositionManager
        Position tracker — read the open quantity from here and book the
        realised P&L here (at the fill price, never the mark).
    broker : Broker
        Broker implementation used to place the order and poll for the fill.
    symbol : str
        Ticker to close (long → SELL, short → BUY-to-cover).
    exit_reason : str
        Reason recorded on the closed trade (``"signal"``, ``"stop_loss"``,
        ``"post_close_cleanup"``, ``"eod"``, ...).
    wait_timeout : float
        How long to poll a pending order for its fill before giving up and
        leaving the close as ``"pending"`` (bounded retry — never endless).
    poll_interval : float
        Poll cadence while waiting for the fill.
    client_id : str | None
        Optional idempotency key passed through to the broker.
    """
    sym = symbol.upper()
    if pm.has_pending_close(sym):
        # A previous close order for this symbol is still unconfirmed.
        # Resubmitting would double-sell (Alpaca rejects it with
        # held_for_orders) or, worse, oversell.  Do nothing.
        return CloseOutcome(
            "pending",
            message="close already pending — not resubmitting",
        )
    pos = pm.get_positions().get(sym)
    if pos is None:
        return CloseOutcome("no_position", message=f"no tracked position for {sym}")

    is_short = pos.quantity < 0
    abs_qty = abs(pos.quantity)
    side = OrderSide.BUY if is_short else OrderSide.SELL
    action = "COVER" if is_short else "SELL"

    order = Order(
        symbol=sym,
        side=side,
        quantity=abs_qty,
        order_type=OrderType.MARKET,
        client_id=client_id,
    )
    try:
        result = await broker.place_order(order)
    except Exception as exc:
        logger.error("Close %s %s failed to submit: %s", action, sym, exc)
        return CloseOutcome("rejected", message=str(exc))

    order_id = getattr(result, "order_id", "") or ""
    status = _status_of(result)

    # ── Already filled on submission ────────────────────────────────
    if status in _FILLED_STATUSES:
        return _book_confirmed_fill(pm, sym, result, exit_reason)

    if not is_order_alive(status):
        err = getattr(result, "error_message", None) or ""
        detail = f"{status}: {err}" if err else status
        logger.warning(
            "Close %s %s rejected (%s); keeping position tracked",
            action, sym, status,
        )
        return CloseOutcome("rejected", order_id=order_id, message=detail)

    # ── Pending (NEW/ACCEPTED/PARTIALLY_FILLED): bounded wait ──────
    wait = getattr(broker, "wait_for_order_fill", None)
    if wait is not None and order_id:
        try:
            filled = await wait(order_id, timeout=wait_timeout, poll_interval=poll_interval)
        except Exception as exc:
            logger.warning("Fill poll for %s %s errored: %s", action, sym, exc)
            filled = None
        if filled is False:
            logger.warning(
                "Close %s %s reached a terminal status before filling; keeping position tracked",
                action, sym,
            )
            return CloseOutcome("rejected", order_id=order_id, message="terminal before fill")
        if filled is not None:
            return _book_confirmed_fill(pm, sym, filled, exit_reason)
    else:
        logger.debug(
            "Broker %s has no wait_for_order_fill — treating unfilled submission as pending",
            type(broker).__name__,
        )

    # ── Still pending after the bounded wait ────────────────────────
    pm.mark_pending_close(sym)
    logger.info(
        "⏳ %s %s placed (id=%s) but fill NOT yet confirmed — position kept "
        "tracked, no P&L booked; will reconcile at next position sync",
        action, sym, order_id,
    )
    return CloseOutcome("pending", order_id=order_id, message="fill not confirmed")


def _book_confirmed_fill(pm, sym, order_like, exit_reason) -> CloseOutcome:
    """Book realised P&L from a broker-confirmed fill (at the fill price).

    If the broker says FILLED but returns no average fill price, we refuse
    to fabricate a mark-based P&L: the symbol is marked pending so the next
    position sync can resolve the true fill from broker fill history.
    """
    price = _fill_price_of(order_like)
    oid = str(getattr(order_like, "id", "") or "") or str(getattr(order_like, "order_id", "") or "")
    if price is None:
        logger.warning(
            "Close %s reported filled but broker returned no fill price — "
            "P&L deferred to next sync",
            sym,
        )
        pm.mark_pending_close(sym)
        return CloseOutcome("pending", order_id=oid, message="filled without price")
    trade = pm.close_position(sym, exit_price=price, exit_reason=exit_reason)
    logger.info(
        "Closed %s at broker fill $%.2f (%s)", sym, price, exit_reason,
    )
    return CloseOutcome("filled", fill_price=price, order_id=oid, trade=trade)


def _status_of(result) -> str:
    raw = getattr(result, "status", "") or ""
    return str(raw).lower().removeprefix("orderstatus.")


def _fill_price_of(order_like) -> float | None:
    """Best-effort average fill price from an order submission/poll result."""
    avg = getattr(order_like, "filled_avg_price", None)
    if avg is None:
        avg = getattr(order_like, "avg_fill_price", None)
    if avg is None:
        return None
    try:
        price = float(avg)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None