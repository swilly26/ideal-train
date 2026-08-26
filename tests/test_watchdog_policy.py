"""Tests for the watchdog restart policy (src/watchdog/*).

Covers the pure decision logic, the RTH schedule helpers, config
resolution, the integrated ``check`` and the CLI contract that
``watchdog.sh`` consumes.
"""
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.watchdog.market_status import (
    in_rth_schedule,
    market_closed_by_schedule,
)
from src.watchdog.policy import (
    RESTART_TOKEN,
    OK_TOKEN,
    check,
    decide_restart,
    main,
    read_config,
)

NY = ZoneInfo("America/New_York")


def dt(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=NY)


WED = dt(2026, 8, 12)  # a Wednesday
WED_MORNING = dt(2026, 8, 12, 7, 0)  # pre-open
WED_MIDDAY = dt(2026, 8, 12, 11, 0)  # RTH open
SAT = dt(2026, 8, 15, 10, 0)  # Saturday


# ---------------------------------------------------------------------------
# decide_restart — the core decision
# ---------------------------------------------------------------------------
def test_restart_when_process_not_running_regardless_of_market():
    # Market closed, exemption on — still must restart a dead process.
    restart, reason = decide_restart(
        process_running=False, has_log=True, log_age=0,
        stale_seconds=300, market_closed=True, skip_stale_when_closed=True,
    )
    assert restart is True
    assert "not running" in reason
    # Market open — still restart.
    restart, _ = decide_restart(
        process_running=False, has_log=True, log_age=999,
        stale_seconds=300, market_closed=False, skip_stale_when_closed=True,
    )
    assert restart is True


def test_restart_when_running_but_no_log():
    restart, _ = decide_restart(
        process_running=True, has_log=False, log_age=0,
        stale_seconds=300, market_closed=True, skip_stale_when_closed=True,
    )
    assert restart is True


def test_no_restart_when_log_fresh():
    restart, reason = decide_restart(
        process_running=True, has_log=True, log_age=50,
        stale_seconds=300, market_closed=False, skip_stale_when_closed=True,
    )
    assert restart is False
    assert "fresh" in reason


def test_restart_when_stale_and_market_open():
    restart, _ = decide_restart(
        process_running=True, has_log=True, log_age=400,
        stale_seconds=300, market_closed=False, skip_stale_when_closed=True,
    )
    assert restart is True


def test_no_restart_when_stale_but_market_closed_and_skip_enabled():
    # THE core regression: healthy trader, stale log, market closed -> exempt.
    restart, reason = decide_restart(
        process_running=True, has_log=True, log_age=400,
        stale_seconds=300, market_closed=True, skip_stale_when_closed=True,
    )
    assert restart is False
    assert "exempt" in reason


def test_restart_when_stale_and_market_closed_but_skip_disabled():
    # Legacy behaviour preserved when the exemption is explicitly disabled.
    restart, _ = decide_restart(
        process_running=True, has_log=True, log_age=400,
        stale_seconds=300, market_closed=True, skip_stale_when_closed=False,
    )
    assert restart is True


def test_no_restart_when_age_equal_to_threshold():
    # Strictly greater-than, not at-or-above.
    restart, _ = decide_restart(
        process_running=True, has_log=True, log_age=300,
        stale_seconds=300, market_closed=False, skip_stale_when_closed=True,
    )
    assert restart is False


# ---------------------------------------------------------------------------
# RTH schedule helpers
# ---------------------------------------------------------------------------
def test_rth_midday_is_open():
    assert in_rth_schedule(WED_MIDDAY) is True


def test_rth_1559_is_open():
    assert in_rth_schedule(dt(2026, 8, 12, 15, 59)) is True


def test_rth_0900_preopen_is_closed():
    assert in_rth_schedule(dt(2026, 8, 12, 9, 0)) is False


def test_rth_1600_close_is_closed():
    assert in_rth_schedule(dt(2026, 8, 12, 16, 0)) is False


def test_weekend_is_closed():
    assert in_rth_schedule(SAT) is False


def test_market_closed_is_inverse_of_open():
    assert market_closed_by_schedule(WED_MIDDAY) is False
    assert market_closed_by_schedule(WED_MORNING) is True
    assert market_closed_by_schedule(SAT) is True


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def test_read_config_defaults():
    stale, skip = read_config({})
    assert stale == 300
    assert skip is True


def test_read_config_disables_exemption():
    stale, skip = read_config(
        {"WATCHDOG_SKIP_STALE_WHEN_CLOSED": "0",
         "WATCHDOG_STALE_SECONDS": "300"}
    )
    assert stale == 300
    assert skip is False


def test_read_config_overrides_stale_seconds():
    stale, skip = read_config({"WATCHDOG_STALE_SECONDS": "600"})
    assert stale == 600
    assert skip is True


def test_read_config_normalizes_false_words():
    for bad in ("false", "no", "off", "FALSE", "OFF"):
        _, skip = read_config({"WATCHDOG_SKIP_STALE_WHEN_CLOSED": bad})
        assert skip is False


def test_read_config_handles_invalid_stale_seconds():
    stale, _ = read_config({"WATCHDOG_STALE_SECONDS": "abc"})
    assert stale == 300


# ---------------------------------------------------------------------------
# Integrated check() (config + schedule + decision)
# ---------------------------------------------------------------------------
def test_check_exempts_stale_when_closed_on_schedule():
    # 07:00 Wednesday ET = pre-open = closed.
    restart, reason = check(True, True, 400.0, now=WED_MORNING, env={})
    assert restart is False
    assert "exempt" in reason


def test_check_restarts_stale_during_rth_on_schedule():
    # 11:00 Wednesday ET = open.
    restart, _ = check(True, True, 400.0, now=WED_MIDDAY, env={})
    assert restart is True


def test_check_restarts_dead_process_even_when_closed():
    restart, _ = check(False, True, 400.0, now=WED_MORNING, env={})
    assert restart is True


def test_check_restarts_when_exemption_disabled_even_when_closed():
    restart, _ = check(True, True, 400.0, now=WED_MORNING,
                       env={"WATCHDOG_SKIP_STALE_WHEN_CLOSED": "0"})
    assert restart is True


# ---------------------------------------------------------------------------
# CLI contract (what watchdog.sh actually parses)
# ---------------------------------------------------------------------------
def _run_cli(*args, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-m", "src.watchdog.policy", *args],
        capture_output=True, text=True, env=merged,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_cli_exempts_stale_when_closed():
    out = _run_cli(
        "--process-running=yes", "--has-log=yes", "--age=400",
        "--now=2026-08-12T07:00:00",
    )
    assert out.returncode == 0
    assert out.stdout.startswith(f"{OK_TOKEN}|")
    assert "exempt" in out.stdout


def test_cli_restarts_stale_when_open():
    out = _run_cli(
        "--process-running=yes", "--has-log=yes", "--age=400",
        "--now=2026-08-12T11:00:00",
    )
    assert out.returncode == 0
    assert out.stdout.startswith(f"{RESTART_TOKEN}|")


def test_cli_restarts_when_process_gone_even_when_closed():
    out = _run_cli(
        "--process-running=no", "--has-log=yes", "--age=400",
        "--now=2026-08-12T07:00:00",
    )
    assert out.stdout.startswith(f"{RESTART_TOKEN}|")


def test_cli_restarts_when_exemption_disabled_via_env():
    out = _run_cli(
        "--process-running=yes", "--has-log=yes", "--age=400",
        "--now=2026-08-12T07:00:00",
        env={"WATCHDOG_SKIP_STALE_WHEN_CLOSED": "0"},
    )
    assert out.stdout.startswith(f"{RESTART_TOKEN}|")


def test_cli_uses_env_stale_seconds_override():
    # age 400 > 300 default but <= 600 override: fresh w.r.t. override.
    out = _run_cli(
        "--process-running=yes", "--has-log=yes", "--age=400",
        "--now=2026-08-12T11:00:00",
        env={"WATCHDOG_STALE_SECONDS": "600"},
    )
    assert out.stdout.startswith(f"{OK_TOKEN}|")


def test_main_returns_zero_without_args():
    assert main([]) == 0
