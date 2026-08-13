"""Persist day-level trading state so session P&L is stable across restarts.

The session baseline (``start_equity``) is captured once at market open.
Writing it to a small JSON file means a watchdog-restarted process reloads the
same day baseline instead of re-capturing mid-day equity as a fresh "start" —
which would make the day's reported P&L meaningless.

The file is date-stamped with the trading day (America/New_York), so a stale
entry from a previous day is simply ignored and overwritten at the next open.
The file lives under ``logs/`` which is gitignored.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SESSION_STATE_FILENAME = "session_state.json"

# Engine root: src/execution/../..  →  <engine>/logs/session_state.json
_DEFAULT_STATE_FILE = (
    Path(__file__).resolve().parent.parent.parent / "logs" / SESSION_STATE_FILENAME
)


def ny_today() -> str:
    """Current date in America/New_York as ISO ``YYYY-MM-DD`` (the trading day)."""
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def load_start_equity(
    state_file: str | Path | None = None,
    trading_date: str | None = None,
) -> float | None:
    """Return the persisted start equity for *trading_date* (default: today).

    Returns ``None`` when the file is missing, unreadable, malformed, or holds
    a baseline for a different trading day — never a fabricated number.
    """
    state_file = Path(state_file) if state_file else _DEFAULT_STATE_FILE
    trading_date = trading_date or ny_today()
    try:
        raw = state_file.read_text()
    except OSError:
        return None
    try:
        state = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if state.get("trading_date") != trading_date:
        return None
    equity = state.get("start_equity")
    if equity is None:
        return None
    try:
        return float(equity)
    except (TypeError, ValueError):
        return None


def save_start_equity(
    equity: float,
    state_file: str | Path | None = None,
) -> bool:
    """Persist *equity* as today's session baseline. Returns ``True`` on success."""
    state_file = Path(state_file) if state_file else _DEFAULT_STATE_FILE
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "trading_date": ny_today(),
            "start_equity": float(equity),
            "captured_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        }
        state_file.write_text(json.dumps(state, indent=2))
        return True
    except OSError:
        return False
