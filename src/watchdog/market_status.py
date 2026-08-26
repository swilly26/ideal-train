"""Market-hours helpers for the watchdog's stale-log exemption.

Determines whether the US equity market is open using a plain
regular-trading-hours schedule in America/New_York time.  Pure
stdlib, no broker SDK, no network: the supervisor must never depend
on the same external API that an outage could take down.

This deliberately mirrors the traders' local-time gate
(``_seconds_until_open`` in live_trader.py / turbo_trader.py):
Mon-Fri 09:30-16:00 ET is "open", everything else (pre-open,
after close, weekends) is "closed".
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
RTH_OPEN = (9, 30)
RTH_CLOSE = (16, 0)


def now_et() -> datetime:
    """Current time in America/New_York."""
    return datetime.now(NY_TZ)


def in_rth_schedule(now: datetime | None = None) -> bool:
    """True if ``now`` (default: current ET time) falls inside regular
    trading hours: Monday-Friday 09:30-16:00 America/New_York.
    """
    dt = now if now is not None else now_et()
    if dt.weekday() >= 5:  # Saturday / Sunday
        return False
    time_of_day = (dt.hour, dt.minute)
    return RTH_OPEN <= time_of_day < RTH_CLOSE


def market_closed_by_schedule(now: datetime | None = None) -> bool:
    """Inverse of :func:`in_rth_schedule` — True when the market is
    closed per the RTH schedule."""
    return not in_rth_schedule(now)