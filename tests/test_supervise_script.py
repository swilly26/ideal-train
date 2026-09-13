"""Regression tests for scripts/supervise_traders.sh (the cron supervisor).

Both behaviours below were exposed by the 2026-09-10 unattended kill-drill
(killed watchdog + both traders; cron was supposed to relaunch the stack
within ~3 min and never did):

1. CWD-INDEPENDENT MARKET GATE — the gate runs
   ``python -c 'from src.watchdog.market_status import ...'``.  cron starts
   this job in the crontab owner's $HOME, NOT in the repo, so the import
   failed with ModuleNotFoundError and the supervisor logged
   "watchdog down but market closed (status=unknown) — deferring until open"
   on every tick while the market was in fact OPEN.  The script must cd to
   the engine dir before the gate so the import resolves from any cwd.

2. LOCK NOT PINNED BY CHILDREN — the launched watchdog must not inherit the
   supervisor's flock fd (9).  If it does, descendants of a killed stack can
   hold the lock for up to a minute (observed at the first post-kill cron
   tick), delaying recovery.  The script closes fd 9 inside the launch
   subshell before setsid forks.

The script under test can be pointed at any copy via SUPERVISE_SCRIPT (used
to prove these tests fail against the pre-fix version of the script).
"""
import os
import subprocess
import time
from pathlib import Path

ENGINE = "/home/team/shared/engine"
SCRIPT = os.environ.get("SUPERVISE_SCRIPT", f"{ENGINE}/scripts/supervise_traders.sh")

PATH_ENV = os.environ.get("PATH", "/usr/bin:/bin")


def _run(env_extra, cwd="/", timeout=30):
    """Run the supervise script with a scratch lockfile + log in tmp_path."""
    log = env_extra.pop("_LOG_PATH")
    lock = env_extra.pop("_LOCK_PATH")
    env = {
        "PATH": PATH_ENV,
        "SUPERVISE_ENGINE_DIR": str(env_extra.pop("_ENGINE_DIR")),
        "SUPERVISE_LOCKFILE": str(lock),
        "SUPERVISE_LOG": str(log),
    }
    env.update(env_extra)
    subprocess.run(["bash", SCRIPT], cwd=cwd, env=env,
                   capture_output=True, text=True, timeout=timeout,
                   check=False)
    return log.read_text()


def _env_base(tmp_path):
    return {
        "_LOCK_PATH": tmp_path / "supervise.lock",
        "_LOG_PATH": tmp_path / "supervise.log",
    }


def _make_fake_engine(tmp_path, gate_output="OPEN", watchdog_sleep="30"):
    """A minimal fake engine dir: a python that prints the gate verdict and
    a watchdog.sh that just marks itself up and sleeps."""
    engine_dir = tmp_path / "fake-engine"
    bin_dir = engine_dir / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    py = bin_dir / "python"
    py.write_text(f"#!/bin/sh\necho {gate_output}\n")
    py.chmod(0o755)
    marker = tmp_path / "watchdog_up"
    wd = engine_dir / "watchdog.sh"
    wd.write_text(f"#!/usr/bin/env bash\nprintf '%s' \"$$\" > {marker}\nsleep {watchdog_sleep}\n")
    wd.chmod(0o755)
    (engine_dir / "logs").mkdir(exist_ok=True)
    return engine_dir, marker


def _flock_state(lockfile):
    r = subprocess.run(
        ["bash", "-c",
         f'exec 9>"{lockfile}"; if flock -n 9; then echo FREE; else echo HELD; fi'],
        capture_output=True, text=True)
    return r.stdout.strip()


def _watchdog_count(engine_dir):
    pat = f"bash {engine_dir}/watchdog[.]sh"
    r = subprocess.run(["pgrep", "-cf", pat], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


# ── 1. CWD-independence of the market gate ─────────────────────────

def test_gate_resolves_from_foreign_cwd(tmp_path):
    """Regression: cron's cwd is not the repo; the gate must still resolve
    the src import (never degrade to status=unknown).  Market open/closed
    state is irrelevant — only that the check actually ran."""
    log = _run({**_env_base(tmp_path),
                "_ENGINE_DIR": Path(ENGINE),  # the REAL engine (real .venv/src)
                "SUPERVISE_DRYRUN": "1"}, cwd="/")
    assert "status=unknown" not in log, \
        "gate could not import src.watchdog.market_status from foreign cwd"
    assert "dry-run: gate passed" in log or "deferring until open" in log, \
        "gate produced neither a pass nor a clean defer"


def test_gate_defers_on_non_open_status(tmp_path):
    """A gate that resolves to anything non-OPEN must defer (safe default)."""
    engine_dir, _ = _make_fake_engine(tmp_path, gate_output="BROKEN")
    log = _run({**_env_base(tmp_path),
                "_ENGINE_DIR": engine_dir,
                "SUPERVISE_DRYRUN": "1"}, cwd="/")
    assert "deferring until open" in log
    assert "status=BROKEN" in log
    assert "dry-run: gate passed" not in log


# ── 2. The launched watchdog must not pin the supervisor lock ──────

def test_child_does_not_inherit_flock_fd(tmp_path):
    """While the (fake) watchdog is alive the supervisor lock must be FREE:
    the launch subshell must close fd 9 before setsid forks.  Pre-fix this
    fails (child inherits fd 9 -> lock HELD)."""
    engine_dir, marker = _make_fake_engine(tmp_path, gate_output="OPEN")
    env = {**_env_base(tmp_path),
           "_ENGINE_DIR": engine_dir}
    log = _run(env, cwd="/")
    assert "launched watchdog" in log
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "supervisor did not launch the fake watchdog"
    try:
        assert _flock_state(env["_LOCK_PATH"]) == "FREE", \
            "fake watchdog inherited fd 9 and pinned the supervisor lock"
    finally:
        _cleanup_fake_watchdog(engine_dir, marker)


def test_noop_guard_holds_while_watchdog_alive(tmp_path):
    """Idempotence: a second supervise run while the (fake) watchdog lives
    must be a no-op (scoped pgrep guard), even with the real production
    watchdog running elsewhere on the same host."""
    engine_dir, marker = _make_fake_engine(tmp_path, gate_output="OPEN")
    env = {**_env_base(tmp_path),
           "_ENGINE_DIR": engine_dir}
    log1 = _run(env, cwd="/")
    assert "launched watchdog" in log1
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "supervisor did not launch the fake watchdog"
    try:
        log2 = _run(env, cwd="/", timeout=30)
        assert log2.strip() == "", "second run must be a silent no-op"
        assert _watchdog_count(engine_dir) == 1, "supervisor spawned a second watchdog"
        assert log1.count("launched watchdog") == 1
    finally:
        _cleanup_fake_watchdog(engine_dir, marker)


def _cleanup_fake_watchdog(engine_dir, marker):
    pid = None
    try:
        pid = int(marker.read_text().strip())
    except Exception:
        pass
    if pid:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    subprocess.run(["pkill", "-f", f"bash {engine_dir}/watchdog[.]sh"],
                   capture_output=True)