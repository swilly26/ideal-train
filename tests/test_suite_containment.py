"""Regression tests for test-harness containment (tests/containment.py).

These pin the containment that the 2026-09-23 storm makes mandatory: a
full-suite run must be INCAPABLE of touching the live trading stack.

The incident: a full-suite run executed the production copy of
``watchdog.sh`` (``/home/team/shared/engine/watchdog.sh``) instead of a
throwaway copy.  That copy hardcoded its engine root, so the test's
``ALGOFLOW_ENGINE_DIR`` redirect, its stubbed process probe and its
``WATCHDOG_MAX_ITERATIONS`` bound were all ignored -- it restarted the real
``live_trader.py`` / ``turbo_trader.py`` in a tight loop
(``WATCHDOG_CHECK_INTERVAL=0``) and left 61 orphans when the suite was killed.

Every test here is hermetic: it either exercises the pure decision function,
or spawns a *throwaway copy* in ``tmp_path`` whose engine root is a synthetic
directory.  Nothing in this file can start, kill or write to the live stack;
that is the property under test.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import containment

TREE = Path(__file__).resolve().parents[1]
PROD = containment.PRODUCTION_ENGINE_ROOT
SESSION = containment.SESSION_ENV

#: The pre-fix call pattern, reduced to its essential line: an engine root that
#: is pinned to the production path whatever the caller asks for.
PRE_FIX_WATCHDOG_SH = """#!/usr/bin/env bash
# Pre-fix watchdog.sh (2026-09-23): the engine root is hardcoded, so the
# ALGOFLOW_ENGINE_DIR redirect a test passes is silently ignored.
set -u
ENGINE_DIR="/home/team/shared/engine"
LOG_DIR="$ENGINE_DIR/logs"
while true; do
    sleep "${WATCHDOG_CHECK_INTERVAL:-60}"
done
"""


def _redirect_env(tmp_path, **extra):
    """A child env that redirects the engine root into tmp_path."""
    engine = tmp_path / "throwaway-engine"
    (engine / "logs").mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "ALGOFLOW_ENGINE_DIR": str(engine),
        "WATCHDOG_MAX_ITERATIONS": "1",
        SESSION: str(PROD),
    }
    env.update(extra)
    return engine, env


# ── the guard is really installed ───────────────────────────────────────────

def test_session_marker_is_installed_and_points_at_the_production_root():
    assert os.environ.get(SESSION) == containment.canonical(PROD), (
        "the session marker must name the production engine root: it is what "
        "makes the live-stack shell scripts refuse to run under test"
    )


def test_popen_is_guarded():
    assert isinstance(
        getattr(subprocess.Popen, "__self__", None), containment.ContainmentGuard
    ), "subprocess.Popen must be the containment-guarded one during a test run"


def test_ordinary_spawns_are_not_blocked():
    """Containment must not get in the way of everything else."""
    out = subprocess.run(
        [sys.executable, "-c", "print('hello')"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0 and "hello" in out.stdout


# ── the decision function ───────────────────────────────────────────────────

def test_the_pre_fix_spawn_pattern_is_refused(tmp_path):
    """THE 2026-09-23 REGRESSION.

    This is the exact call the storm made: the production watchdog path, an
    engine-root redirect that the script ignores, and a check interval of 0.
    Containment must flag it -- and refuse it *before* a process exists, so
    there is nothing to orphan.  (The stand-in script below is harmless if it
    ever does get executed, which is what makes this test safe to run.)
    """
    pre_fix = tmp_path / "watchdog.sh"
    pre_fix.write_text(PRE_FIX_WATCHDOG_SH)
    _, env = _redirect_env(tmp_path, WATCHDOG_CHECK_INTERVAL="0")

    violations = containment.spawn_violations(
        ["bash", str(pre_fix)], env=env, cwd="/"
    )
    assert violations, (
        "a script that pins its engine root to the production root must never "
        "be spawned by the suite, however it is redirected"
    )
    assert any("pins the engine root" in v for v in violations)

    with pytest.raises(containment.ContainmentViolation):
        subprocess.run(["bash", str(pre_fix)], env=env, capture_output=True, timeout=30)
    assert containment.live_stack_pids(exclude_pytest_tmp=False) == {}, (
        "the refusal must happen before the process exists"
    )
    assert containment.production_processes() == {}


def test_live_watchdog_is_refused_without_an_engine_root_redirect(tmp_path):
    _, env = _redirect_env(tmp_path)
    del env["ALGOFLOW_ENGINE_DIR"]
    violations = containment.spawn_violations(
        ["bash", str(TREE / "watchdog.sh")], env=env, cwd="/"
    )
    assert any("no engine-root redirect" in v for v in violations), violations


def test_live_watchdog_is_refused_when_the_redirect_points_at_production(tmp_path):
    _, env = _redirect_env(tmp_path, ALGOFLOW_ENGINE_DIR=str(PROD))
    violations = containment.spawn_violations(
        ["bash", str(TREE / "watchdog.sh")], env=env, cwd="/"
    )
    assert any("production engine root" in v for v in violations), violations


def test_watchdog_spawn_without_a_bound_is_refused(tmp_path):
    _, env = _redirect_env(tmp_path)
    del env["WATCHDOG_MAX_ITERATIONS"]
    violations = containment.spawn_violations(
        ["bash", str(TREE / "watchdog.sh")], env=env, cwd="/"
    )
    assert any("no bound" in v for v in violations), (
        "a spawned watchdog must carry WATCHDOG_MAX_ITERATIONS (or "
        "WATCHDOG_MAX_SECONDS) so it cannot spin forever: " + repr(violations)
    )


def test_production_trader_module_is_never_spawned(tmp_path):
    _, env = _redirect_env(tmp_path)
    violations = containment.spawn_violations(
        [sys.executable, str(Path(PROD) / "live_trader.py")], env=env, cwd="/"
    )
    assert any("production trader module" in v for v in violations), violations


def test_supervise_spawn_requires_its_own_redirect(tmp_path):
    _, env = _redirect_env(tmp_path)
    del env["ALGOFLOW_ENGINE_DIR"]
    violations = containment.spawn_violations(
        ["bash", str(TREE / "scripts" / "supervise_traders.sh")], env=env, cwd="/"
    )
    assert violations, "the cron supervisor must never be spawned unredirected"


def test_paths_outside_the_live_stack_are_ignored():
    assert containment.spawn_violations(
        [sys.executable, "-m", "src.watchdog.policy", "--age=10"], env={}
    ) == []


# ── the tree's own scripts must be redirect-safe ─────────────────────────────

@pytest.mark.parametrize(
    "relative,override",
    [
        ("watchdog.sh", "ALGOFLOW_ENGINE_DIR"),
        ("start_trader.sh", "ALGOFLOW_ENGINE_DIR"),
        ("start_turbo.sh", "ALGOFLOW_ENGINE_DIR"),
        ("scripts/supervise_traders.sh", "SUPERVISE_ENGINE_DIR"),
    ],
)
def test_every_live_stack_script_honours_a_redirect(tmp_path, relative, override):
    """A spawn of the tree's own copy, redirected into tmp, must be allowed.

    This is the property the storm violated: a redirect the script ignores is
    worse than no redirect, because it looks safe.  The checker reads the
    script's source, so a future edit that hardcodes an engine root (or drops
    the ``ALGOFLOW_TEST_SESSION`` refusal) turns this red.
    """
    _, env = _redirect_env(tmp_path)
    env[override] = str(tmp_path / "throwaway-engine")
    script = TREE / relative
    assert script.is_file(), script
    assert SESSION in script.read_text(), (
        f"{relative} must carry the test-session refusal guard"
    )
    assert containment.spawn_violations(
        ["bash", str(script)], env=env, cwd="/"
    ) == [], (
        f"{relative} would resolve the production engine root for a test that "
        f"redirected it -- the redirect is not honoured"
    )


@pytest.mark.parametrize(
    "relative",
    ["watchdog.sh", "start_trader.sh", "start_turbo.sh"],
)
def test_scripts_refuse_to_run_against_a_test_session_engine_root(tmp_path, relative):
    """The shell-side guard, exercised for real (against a synthetic root).

    The script is copied into tmp and told both that its engine root IS the
    test session's root and that it is running under a test session: it must
    refuse (exit 98) instead of starting a trader or writing state.  This is
    the layer that still works when a caller's test-side guard is missing or
    stale.
    """
    engine = tmp_path / "engine"
    (engine / "logs").mkdir(parents=True)
    script = engine / Path(relative).name
    script.write_text((TREE / relative).read_text())
    script.chmod(0o755)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "ALGOFLOW_ENGINE_DIR": str(engine),
        SESSION: str(engine),
        "WATCHDOG_MAX_ITERATIONS": "1",
    }
    out = subprocess.run(
        ["bash", str(script)], env=env, cwd="/",
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 98, (
        f"{relative} ran against a production engine root under a test session "
        f"instead of refusing (rc={out.returncode}, stderr={out.stderr!r})"
    )
    assert "REFUSING" in out.stderr


def test_watchdog_loop_is_bounded_by_max_iterations(tmp_path):
    """A spawned watchdog stops by itself -- no orphan to clean up."""
    engine = tmp_path / "engine"
    (engine / "logs").mkdir(parents=True)
    script = engine / "watchdog.sh"
    script.write_text((TREE / "watchdog.sh").read_text())
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "ALGOFLOW_ENGINE_DIR": str(engine),
        SESSION: str(PROD),  # not the throwaway root: the guard must not fire
        "WATCHDOG_MAX_ITERATIONS": "2",
        "WATCHDOG_CHECK_INTERVAL": "0",
    }
    out = subprocess.run(
        ["bash", str(script)], env=env, cwd="/",
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    log = (engine / "logs" / "watchdog.log").read_text()
    assert "leaving the loop after 2 iteration" in log, log
    assert "raised to 1s" in out.stderr, (
        "a check interval of 0 must be floored, never busy-looped"
    )


# ── the live stack itself ───────────────────────────────────────────────────

def test_no_live_stack_process_is_running():
    """The owner has paused trading: nothing may be up while the suite runs."""
    assert containment.production_processes() == {}, (
        "a live-stack process is running while the suite runs -- the run "
        "cannot be trusted, and containment would have to fight it"
    )
