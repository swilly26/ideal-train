"""Shared pytest fixtures and configuration."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv():
    """Return a small OHLCV DataFrame for use across test modules."""
    idx = pd.date_range("2026-01-01 09:30", periods=20, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(20)],
            "high": [101.0 + i * 0.1 for i in range(20)],
            "low": [99.0 + i * 0.1 for i in range(20)],
            "close": [100.5 + i * 0.1 for i in range(20)],
            "volume": [1000] * 20,
        },
        index=idx,
    )


@pytest.fixture
def uptrend_ohlcv():
    """Steadily rising prices — good for momentum and upside breakout."""
    periods = 40
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(42)
    base = 100.0
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        base += 0.5  # steady uptrend
        noise = rng.normal(0, 0.2)
        close = base + noise
        high = close + abs(rng.normal(0.15, 0.05))
        low = close - abs(rng.normal(0.15, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def downtrend_ohlcv():
    """Steadily falling prices — should trigger momentum SELL."""
    periods = 40
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(99)
    base = 100.0
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        base -= 0.5  # steady downtrend
        noise = rng.normal(0, 0.2)
        close = base + noise
        high = close + abs(rng.normal(0.15, 0.05))
        low = close - abs(rng.normal(0.15, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def oscillating_ohlcv():
    """Prices oscillating around a fixed mean — ideal for mean reversion."""
    periods = 60
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(7)
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        # Sine wave around 100 with amplitude 5, plus small noise
        close = 100.0 + 5.0 * np.sin(2 * np.pi * i / 20) + rng.normal(0, 0.2)
        high = close + abs(rng.normal(0.15, 0.05))
        low = close - abs(rng.normal(0.15, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def breakout_ohlcv():
    """Range-bound for most of the period, then a sharp upside breakout."""
    periods = 40
    idx = pd.date_range("2026-01-01 09:30", periods=periods, freq="1min")
    rng = np.random.default_rng(13)
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(periods):
        if i < 30:
            # Tight range between 99 and 101
            close = 100.0 + rng.normal(0, 0.3)
            high = close + abs(rng.normal(0.2, 0.05))
            low = close - abs(rng.normal(0.2, 0.05))
        else:
            # Sharp breakout upward
            close = 105.0 + (i - 30) * 1.0 + rng.normal(0, 0.2)
            high = close + abs(rng.normal(0.3, 0.05))
            low = close - abs(rng.normal(0.3, 0.05))
        open_p = low + rng.random() * (high - low)
        closes.append(round(close, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        opens.append(round(open_p, 4))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * periods},
        index=idx,
    )


@pytest.fixture
def make_frame():
    """Build a CandleFrame from plain lists (hand-made mock candles).

    ``ts_list`` is a list of "YYYY-MM-DD HH:MM" local (exchange-naive) times.
    """
    import numpy as np
    import pandas as pd
    from src.strategies.scalp.types import CandleFrame

    def _make(symbol, timeframe, ts_list, o, h, l, c, v):
        ts = np.asarray(pd.DatetimeIndex(ts_list).to_numpy(), dtype="datetime64[ns]")
        return CandleFrame(
            symbol=symbol,
            timeframe=timeframe,
            ts=ts,
            open=np.asarray(o, dtype=float),
            high=np.asarray(h, dtype=float),
            low=np.asarray(l, dtype=float),
            close=np.asarray(c, dtype=float),
            volume=np.asarray(v, dtype=float),
        )

    return _make


# ── Trading-stack containment (2026-09-23) ──────────────────────────────────
# A full-suite run must be INCAPABLE of touching the live stack.  See
# tests/containment.py for the mechanism and the incident behind it.  The
# guard is installed before collection and torn down after the last test:
# every spawn of a live-stack script is inspected BEFORE it exists (fail
# closed), every child carries the ALGOFLOW_TEST_SESSION marker that makes the
# shell scripts refuse a production engine root, whatever the tests are
# allowed to spawn is reaped by process group, and the session FAILS if a
# live-stack process appeared that was not there when the session started.
import tempfile as _tempfile
from pathlib import Path as _Path

from tests import containment as _containment

_GUARD = None


@pytest.fixture(scope="session")
def containment_guard():
    """The session's containment guard (for the containment tests)."""
    if _GUARD is None:  # pragma: no cover - hooks always install it first
        raise RuntimeError("containment guard was not installed")
    return _GUARD


def pytest_sessionstart(session):
    global _GUARD
    if _GUARD is None and not getattr(session.config.option, "collectonly", False):
        _GUARD = _containment.ContainmentGuard(
            _tempfile.mkdtemp(prefix="pytest-containment-")
        )
        _GUARD.install()


def pytest_sessionfinish(session, exitstatus):
    global _GUARD
    if _GUARD is None:
        return
    escaped = _GUARD.escaped_pids()
    if escaped:
        _containment.report(
            "LIVE-STACK PROCESSES APPEARED DURING THIS TEST SESSION -- the suite "
            "touched the live stack.  Killing them by process group and failing "
            "the run:"
        )
        for pid, cmd in sorted(escaped.items()):
            _containment.report(f"  pid {pid}: {cmd}")
        for pid in escaped:
            _containment.kill_pid_group(pid, 15)
        import time as _time
        _time.sleep(0.4)
        for pid in _GUARD.escaped_pids():
            _containment.kill_pid_group(pid, 9)
        session.exitstatus = 1
    reaped = _GUARD.reap()
    if reaped:
        _containment.report(
            f"reaped {len(reaped)} stray process(es) started by this session: "
            f"{sorted(reaped)}"
        )
    if _GUARD.refusals:
        _containment.report(
            f"{len(_GUARD.refusals)} spawn(s) refused before a process was "
            "created (an offending test failed loudly instead of touching the "
            "live stack):"
        )
        for line in _GUARD.refusals:
            _containment.report(f"  {line}")
    remaining = _GUARD.resolved_pids()
    _containment.report(
        "containment summary: "
        f"{len(_GUARD.refusals)} refusal(s), {len(reaped)} stray(s) reaped, "
        f"{len(remaining)} live-stack process(es) on the box at session end"
    )
    _GUARD.uninstall()
