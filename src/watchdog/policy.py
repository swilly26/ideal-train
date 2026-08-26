"""Watchdog restart policy — the decision behind ``watchdog.sh``.

The bash supervisor gathers the per-trader facts (is the process
alive, does it have a log file, how old is the newest log) and asks
this module for a verdict.  Centralising the decision in Python keeps
the critical logic unit-testable and explicit.

Why a market-hours exemption at all?
  Before the open (and after close / on weekends) the traders legally
  produce no log output for hours — they sleep inside
  ``wait_for_market_open``.  The watchdog's staleness check would kill
  and restart a perfectly healthy trader, and each restart triggers the
  session-start/cleanup path, which has realized real losses (e.g. a
  -$539 liquidation of a leftover TZA position).  So the stale-log kill
  is exempted while the market is closed.  A process that is NOT running
  is still restarted regardless of market hours, and a genuinely hung
  trader is still caught during market hours.

Environment knobs (all optional, safe defaults — existing deployments
keep their current ``WATCHDOG_STALE_SECONDS=300`` behaviour):

  WATCHDOG_STALE_SECONDS           default 300   (unchanged)
  WATCHDOG_SKIP_STALE_WHEN_CLOSED  default 1/on  (new exemption)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Mapping, Optional, Tuple

from src.watchdog.market_status import NY_TZ, market_closed_by_schedule

# Verdict tokens printed by the CLI (parsed by watchdog.sh).
RESTART_TOKEN = "RESTART"
OK_TOKEN = "OK"


def read_config(
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[int, bool]:
    """Resolve configuration from the environment with safe defaults.

    Returns ``(stale_seconds, skip_stale_when_closed)``.
    """
    env = os.environ if env is None else env
    raw_stale = (env.get("WATCHDOG_STALE_SECONDS") or "300").strip()
    try:
        stale_seconds = int(raw_stale)
    except ValueError:
        stale_seconds = 300
    raw_skip = (env.get("WATCHDOG_SKIP_STALE_WHEN_CLOSED") or "1").strip()
    skip = raw_skip.lower() not in ("", "0", "false", "no", "off")
    return stale_seconds, skip


def decide_restart(
    *,
    process_running: bool,
    has_log: bool,
    log_age: float,
    stale_seconds: int,
    market_closed: bool,
    skip_stale_when_closed: bool,
) -> Tuple[bool, str]:
    """Return ``(should_restart, reason)`` for one trader check.

    Rules (in priority order):
      1. Process not running            -> restart, always.
      2. Process running but no log     -> restart, always (a running
         trader must be writing logs; absence means a bad start).
      3. Log stale beyond threshold     -> restart, UNLESS the market is
         confirmed closed and the closed-market exemption is enabled.
      4. Log fresh                      -> no restart.
    """
    if not process_running:
        return True, "process is not running"
    if not has_log:
        return True, "no log file found"
    if log_age > stale_seconds:
        if skip_stale_when_closed and market_closed:
            return (
                False,
                f"log stale ({log_age:.0f}s) but market closed — "
                f"exempting stale-kill",
            )
        return True, f"log stale ({log_age:.0f}s)"
    return False, "log fresh"


def check(
    process_running: bool,
    has_log: bool,
    log_age: float,
    *,
    now: Optional[datetime] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, str]:
    """Full policy check: resolve config + market state, then decide.

    ``now`` is injectable for tests (defaults to the real clock);
    ``env`` is injectable for tests (defaults to ``os.environ``).
    """
    stale_seconds, skip = read_config(env)
    closed = market_closed_by_schedule(now)
    return decide_restart(
        process_running=process_running,
        has_log=has_log,
        log_age=log_age,
        stale_seconds=stale_seconds,
        market_closed=closed,
        skip_stale_when_closed=skip,
    )


def _parse_now(value: str) -> datetime:
    """Parse an ISO ``YYYY-MM-DDTHH:MM[:SS]`` timestamp as ET."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    return dt


def main(argv: Optional[list] = None) -> int:
    """CLI entry point used by ``watchdog.sh``.

    Arguments (``--key=value``):
      --process-running   yes|no   (default: yes)
      --has-log           yes|no   (default: yes)
      --age               seconds  (float; default: 0)
      --now               ISO timestamp in ET (optional; for tests/ops)

    Prints ``RESTART|<reason>`` or ``OK|<reason>`` on stdout and
    always exits 0 (the verdict token is the contract with the shell;
    a non-zero exit from the helper is treated as "restart" by the
    watchdog, preserving legacy fail-safe behaviour).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    args: dict = {}
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            key, _, value = arg[2:].partition("=")
            args[key] = value

    def flag(name: str, default: str) -> bool:
        raw = args.get(name, default)
        return raw.strip().lower() not in ("", "0", "false", "no", "off")

    try:
        log_age = float(args.get("age", "0"))
    except ValueError:
        log_age = 0.0

    now = _parse_now(args["now"]) if args.get("now") else None
    restart, reason = check(
        process_running=flag("process-running", "yes"),
        has_log=flag("has-log", "yes"),
        log_age=log_age,
        now=now,
    )
    print(f"{RESTART_TOKEN if restart else OK_TOKEN}|{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())