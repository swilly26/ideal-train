"""Exit-structure planning for live single-name positions (TP + protective stop).

WHY THIS MODULE EXISTS (live defect, 2026-09-22)
------------------------------------------------
Every ScalpSet position is protected by a broker-side GTC stop.  Alpaca
reserves (``held_for_orders``) the quantity of a resting order, and the two
exits were being submitted as two independent orders for the same shares, so
the second one was always rejected:

    logs/trades_20260921.log 19:52:13Z (META short 21, GTC BUY stop resting)
    {"available":"0","code":40310000,"existing_qty":"21",
     "held_for_orders":"21","message":"insufficient qty available for order
     (requested: 21, available: 0)","symbol":"META"}

    same session 19:52Z TSLA long 42.92623498, GTC SELL stop placed for the
    WHOLE-SHARE quantity 42  ->  the fractional TP (42.926) had 0.926
    available  ->  40310000 as well.

So the take-profit never rested, and worse, the rejection path nulled the
signal's TP level out of the position state, leaving the position with NO
upside exit at all — every live session was structurally downside-only
(broker stop or nothing).

BROKER FACTS MEASURED AGAINST THE PAPER API (2026-09-22 03:0xZ, market closed)
-----------------------------------------------------------------------------
Run with raw REST against paper-api.alpaca.markets; every accepted order was
cancelled immediately:

* Whole-share OCO exit (``order_class=oco``, limit TP + stop leg) IS accepted
  for an existing position when the quantity is a whole number — one order,
  both legs held once:  HTTP 200 id=9f462257 class=oco legs=[buy stop 388 held].
* FRACTIONAL OCO/BRACKET is refused outright:
  ``qty=1.5 gtc`` -> 42210000 "fractional orders must be DAY orders"
  ``qty=1.5 day`` -> 42210000 "fractional orders must be simple orders"
  ``bracket BUY 0.5`` -> 42210000 "fractional orders must be simple orders"
  Live scalp LONGS are fractional (TSLA 42.92623498, COIN 79.570566562,
  NVDA 71.015806777), so a complex order cannot cover them at all.
* A fractional STANDALONE stop is allowed only as a DAY order; the GTC form
  the teardown-proof design depends on is refused:
  ``qty=0.5 gtc stop`` -> 42210000 "stop/stop_limit fractional GTC orders are
  not enabled"   (``qty=0.5 day stop`` -> HTTP 200).
* Both a stop and a TP *can* rest side by side as long as the SUM of their
  quantities stays within the position (a 0.5 stop + 0.5 limit on a 48-share
  short both returned HTTP 200) — which is precisely why an exit covering the
  WHOLE position cannot be expressed as two orders.

CONCLUSION / CHOSEN STRUCTURE
-----------------------------
Because the mandatory requirement is "whole-share AND fractional positions
both work", the broker-side complex-order route is out.  The structure is:

    SL  — resting whole-share GTC protective stop at the broker (unchanged,
          survives process death; whole-share because Alpaca refuses
          fractional GTC stops).
    TP  — one of two homes, decided by ``plan_exit_structure``:
          * resting broker DAY-limit order, used only when the stop does not
            already reserve the shares (stop placement failed / nothing
            resting);
          * trader-side MONITORED exit for the full position — the level is
            kept in the position state and the per-tick risk pass submits a
            confirmed cancel-then-close when price touches it.

The monitor is deliberately not a second resting order: it re-uses the
verified-close machinery (cancel orders -> close -> re-protect on failure)
that already exists for EOD/liquidation paths, so a partial fill cannot leave
a naked residual (see ``_reprotect_residual``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Alpaca refuses fractional GTC stop orders (42210000 "stop/stop_limit
# fractional GTC orders are not enabled"), so the resting stop can only ever
# reserve a WHOLE number of shares.
STOP_QTY_WHOLE_SHARES = True


def stop_capacity(qty: float) -> int:
    """Whole-share quantity a RESTING GTC stop can reserve for *qty* shares."""
    try:
        q = abs(float(qty))
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(q):
        return 0
    return int(math.floor(q))


@dataclass(frozen=True)
class ExitPlan:
    """Where each leg of the exit lives for one position.

    ``broker_tp_qty`` — quantity the trailing DAY-limit TP order may cover
    (0.0 when that order is doomed: the resting stop already reserves the
    whole position).
    ``monitored_tp`` — the trader-side monitor owns the upside exit for the
    whole position.
    ``reason`` — human-readable justification, logged at entry so the live
    record says which structure a position is running under.
    """

    qty: float
    stop_qty: int
    broker_tp_qty: float
    monitored_tp: bool
    reason: str


def plan_exit_structure(
    qty: float,
    *,
    stop_placed: bool = True,
    tp: float | None = None,
) -> ExitPlan:
    """Decide the exit structure for a position of *qty* shares (abs value).

    Rules (measured against the paper API, see module docstring):

    * No TP level -> nothing to monitor (stop only).
    * The resting stop reserves ``stop_capacity(qty)`` whole shares.  When
      that reservation consumes the whole position, a second order for the
      same shares is ALWAYS rejected (40310000) — so the broker TP is not
      attempted at all and the trader-side monitor takes the upside exit.
      (Attempting it just burns an API call and logs a scary rejection on
      every entry, which is what the live logs were full of.)
    * When nothing is reserved yet (stop placement failed), a full-quantity
      DAY-limit TP is attempted first — a resting exit is strictly better
      than a monitored one — with the monitor remaining the fallback if the
      order is rejected for any other reason.
    * A PARTIAL broker TP (only the sub-share remainder left over by the
      whole-share stop) is never placed: it would split one position across
      two different exit paths for a fraction of a share, and it cannot
      cover the position it is supposed to pay for.  Either the resting TP
      covers the whole position or the monitor owns the whole exit.
    """
    try:
        size = abs(float(qty))
    except (TypeError, ValueError):
        size = 0.0
    if not math.isfinite(size) or size <= 0:
        return ExitPlan(0.0, 0, 0.0, False, "no position")
    if tp is None:
        return ExitPlan(size, stop_capacity(size) if stop_placed else 0,
                        0.0, False, "signal carries no TP level (stop only)")

    held = stop_capacity(size) if stop_placed else 0
    if held > 0:
        return ExitPlan(
            size, held, 0.0, True,
            f"resting whole-share GTC stop reserves {held:.0f} of "
            f"{size:.6f} shares — Alpaca 40310000 for any second order on "
            "those shares, so the upside exit is trader-side monitored",
        )
    return ExitPlan(
        size, 0, size, True,
        "no resting stop to reserve the shares — trying a resting "
        "DAY-limit TP for the full quantity, trader-side monitor stays the "
        "fallback",
    )


def format_exit_plan(plan: ExitPlan) -> str:
    """One-line structure summary for the entry log."""
    if not plan.monitored_tp and plan.broker_tp_qty > 0:
        home = "broker DAY limit"
    elif plan.broker_tp_qty > 0:
        home = "broker DAY limit (monitor fallback)"
    elif plan.monitored_tp:
        home = "trader-side monitor"
    else:
        home = "none"
    return (f"SL=broker GTC stop x{plan.stop_qty} | TP={home} "
            f"({plan.reason})")
