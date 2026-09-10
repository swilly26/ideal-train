"""Tests for the market-status CLI used by the session-independent supervisor.

``python -m src.watchdog.market_status`` prints OPEN when the US regular
trading session is running (DST-aware, pure stdlib — never depends on a
network clock) and CLOSED otherwise.  The cron supervisor
(``scripts/supervise_traders.sh``) gates stack restarts on this so a
pre-open boot can never trigger the post-startup cleanup liquidation.
"""
import subprocess
import sys
from datetime import datetime

from src.watchdog import market_status
from src.watchdog.market_status import NY_TZ, in_rth_schedule


def test_in_rth_schedule_weekday_preopen():
    dt = datetime(2026, 9, 10, 8, 0, tzinfo=NY_TZ)  # Wed 8:00 ET — pre-open
    assert in_rth_schedule(dt) is False


def test_in_rth_schedule_weekday_open():
    dt = datetime(2026, 9, 10, 12, 0, tzinfo=NY_TZ)  # Wed noon ET — open
    assert in_rth_schedule(dt) is True


def test_in_rth_schedule_close_boundary():
    dt = datetime(2026, 9, 10, 16, 0, tzinfo=NY_TZ)  # Wed 16:00 ET — closed
    assert in_rth_schedule(dt) is False


def test_in_rth_schedule_weekend():
    dt = datetime(2026, 9, 12, 12, 0, tzinfo=NY_TZ)  # Sat noon ET
    assert in_rth_schedule(dt) is False


def test_cli_prints_open_or_closed():
    """The supervisor parses this output; it must be exactly OPEN or CLOSED."""
    out = subprocess.run(
        [sys.executable, "-m", "src.watchdog.market_status"],
        capture_output=True, text=True, cwd="/home/team/shared/engine",
    )
    assert out.returncode == 0
    assert out.stdout.strip() in ("OPEN", "CLOSED")
    expected = "OPEN" if market_status.in_rth_schedule() else "CLOSED"
    assert out.stdout.strip() == expected