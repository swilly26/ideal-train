"""Tests for the pre-open wait logic in live_trader.py / turbo_trader.py.

Covers the DST bug: market open is 9:30 AM America/New_York — 13:30 UTC in
summer (EDT) but 14:30 UTC in winter (EST).  The old hardcoded 13:30 UTC
target was wrong for half the year.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import live_trader
import turbo_trader

# Trader class name differs per module (LiveTrader vs TurboTrader).
_TRADER_CLASS = {
    live_trader: "LiveTrader",
    turbo_trader: "TurboTrader",
}


def _freeze_utc(module, monkeypatch, utc_iso: str) -> datetime:
    """Monkeypatch ``module.datetime.now`` to return a fixed UTC instant."""
    fixed = datetime.fromisoformat(utc_iso)  # tz-aware (UTC)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed
            return fixed.astimezone(tz)

    monkeypatch.setattr(module, "datetime", FakeDT)
    return fixed


@pytest.mark.parametrize("module", [live_trader, turbo_trader])
class TestSecondsUntilOpenDst:
    def _seconds_until_open(self, module, monkeypatch, utc_iso: str) -> float:
        _freeze_utc(module, monkeypatch, utc_iso)
        trader_cls = getattr(module, _TRADER_CLASS[module])
        return trader_cls._seconds_until_open()

    def test_winter_13_utc_is_not_about_to_open(self, module, monkeypatch):
        """13:00 UTC in January == 08:00 EST — still 90 min to open.

        The old hardcoded 13:30 UTC target would have returned 30 min here,
        causing pre-open checks to resume far too early (or mis-sleep).
        """
        assert self._seconds_until_open(module, monkeypatch, "2026-01-12T13:00:00+00:00") == pytest.approx(5400.0)

    def test_winter_14_30_utc_is_exactly_open_time(self, module, monkeypatch):
        """14:30 UTC in January == 09:30 EST — 0 s to open."""
        assert self._seconds_until_open(module, monkeypatch, "2026-01-12T14:30:00+00:00") == pytest.approx(0.0)

    def test_summer_13_utc_is_exactly_open_time(self, module, monkeypatch):
        """13:00 UTC in July == 09:00 EDT — 30 min to open (EDT = UTC-4)."""
        assert self._seconds_until_open(module, monkeypatch, "2026-07-13T13:00:00+00:00") == pytest.approx(1800.0)

    def test_after_close_targets_next_weekday(self, module, monkeypatch):
        """Monday 15:00 ET → target is Tuesday 9:30 ET (~18.5 h later)."""
        assert self._seconds_until_open(module, monkeypatch, "2026-01-12T20:00:00+00:00") == pytest.approx(18.5 * 3600.0)

    def test_weekend_targets_monday_9_30_et(self, module, monkeypatch):
        """Saturday 20:00 UTC → next open is Monday 9:30 AM ET."""
        fixed = datetime.fromisoformat("2026-01-10T20:00:00+00:00")  # Saturday
        secs = self._seconds_until_open(module, monkeypatch, "2026-01-10T20:00:00+00:00")
        ny = ZoneInfo("America/New_York")
        arrival = (fixed + timedelta(seconds=secs)).astimezone(ny)
        assert arrival.weekday() == 0  # Monday
        assert (arrival.hour, arrival.minute) == (9, 30)

    def test_friday_evening_targets_monday_9_30_et(self, module, monkeypatch):
        """Friday 22:00 UTC (17:00 ET) → next open is Monday 9:30 AM ET."""
        fixed = datetime.fromisoformat("2026-01-09T22:00:00+00:00")  # Friday
        secs = self._seconds_until_open(module, monkeypatch, "2026-01-09T22:00:00+00:00")
        ny = ZoneInfo("America/New_York")
        arrival = (fixed + timedelta(seconds=secs)).astimezone(ny)
        assert arrival.weekday() == 0  # Monday
        assert (arrival.hour, arrival.minute) == (9, 30)
