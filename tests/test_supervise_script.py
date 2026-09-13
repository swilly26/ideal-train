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

HERMETICITY CONTRACT — these tests must NEVER touch the production
environment, even if the script under test regresses:

* Every test overrides ALL four script env hooks to tmp_path:
  SUPERVISE_ENGINE_DIR, SUPERVISE_LOG_DIR, SUPERVISE_LOG and
  SUPERVISE_LOCKFILE (defaults would otherwise resolve to the real
  /home/team/shared/engine, its logs/, and /tmp/supervise_traders.lock).
  ``_run`` hard-asserts the engine dir is NOT the real engine, so a future
  edit that accidentally points a test at production fails loudly instead
  of spawning the real stack.
* The launch path is exercised ONLY against a tmp_path stub watchdog
  (``_make_fake_engine``), never the real watchdog.sh — the script builds
  its launch command as ``$ENGINE_DIR/watchdog.sh``, so a stub engine dir
  is sufficient (verified; there is no hardcoded engine path in the launch
  block).
* Fake watchdogs started via the real launch block are killed by process
  GROUP (setsid makes the watchdog its own PG leader) plus a pgrep pattern
  scoped to that test's fake engine dir, so NO background process survives
  the test and the scoped pattern can never match the real stack.

The script under test can be pointed at any copy via SUPERVISE_SCRIPT (used
to prove these tests fail against the pre-fix version of the script).
"""
import os
import signal
import subprocess
import time
from pathlib import Path

ENGINE = Path("/home/team/shared/engine")
SCRIPT = os.environ.get("SUPERVISE_SCRIPT", f"{ENGINE}/scripts/supervise_traders.sh")

PATH_ENV = os.environ.get("PATH", "/usr/bin:/bin")


def _run(env_extra, cwd="/", timeout=30):
    """Run the supervise script fully redirected into tmp_path.

    ``env_extra`` is COPIED at the top so repeated calls with the same dict
    cannot KeyError.  Every one of the script's env hooks (engine dir, log
    dir, log file, lockfile) is taken from the test's tmp_path; the engine
    dir is hard-asserted to NOT be the real engine so this suite can never
    touch the production stack, logs or lockfile.
    """
    env_extra = dict(env_extra)
    log = env_extra.pop("_LOG_PATH")
    lock = env_extra.pop("_LOCK_PATH")
    log_dir = env_extra.pop("_LOG_DIR")
    engine_dir = Path(env_extra.pop("_ENGINE_DIR"))
    assert engine_dir != ENGINE, \
        "supervise tests must never run against the real engine dir"
    env = {
        "PATH": PATH_ENV,
        "SUPERVISE_ENGINE_DIR": str(engine_dir),
        "SUPERVISE_LOG_DIR": str(log_dir),
        "SUPERVISE_LOCKFILE": str(lock),
        "SUPERVISE_LOG": str(log),
    }
    env.update(env_extra)
    subprocess.run(["bash", SCRIPT], cwd=cwd, env=env,
                   capture_output=True, text=True, timeout=timeout,
                   check=False)
    return log.read_text()


def _env_base(tmp_path):
    """The per-test tmp_path overrides for every script env hook."""
    return {
        "_LOCK_PATH": tmp_path / "supervise.lock",
        "_LOG_PATH": tmp_path / "supervise.log",
        "_LOG_DIR": tmp_path / "logs",
    }


def _make_fake_engine(tmp_path, gate_output="OPEN", watchdog_sleep="60",
                      cwd_check=False):
    """A minimal fake engine dir, entirely inside tmp_path.

    * ``.venv/bin/python`` — a stub that prints the market-gate verdict.
      With ``cwd_check=True`` it only prints OPEN when its cwd IS the fake
      engine dir, so a script that stops cd-ing before the gate (the cron
      cwd regression) degrades to an UNKNOWN/ERRORED verdict instead of
      silently importing: the regression is detected hermetically, without
      ever invoking the real python/src.
    * ``watchdog.sh`` — a harmless stub that marks itself up (writes its
      pid to the marker file) and sleeps, instead of the real watchdog.
      The supervise launch block uses ``$ENGINE_DIR/watchdog.sh``, so this
      stub IS the entire launch target; nothing outside tmp_path can be
      launched by the tests.
    """
    engine_dir = tmp_path / "fake-engine"
    bin_dir = engine_dir / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    py = bin_dir / "python"
    if cwd_check:
        py.write_text(
            "#!/bin/sh\n"
            'if [ "$(pwd)" = "{engine_dir}" ]; then echo OPEN; '
            'else echo GATE-IMPORTFAIL; fi\n'.format(engine_dir=engine_dir)
        )
    else:
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
    state is irrelevant — only that the check actually ran.

    Hermetic: the fake engine's python prints OPEN only when the script
    cd'd to the fake engine dir before invoking it, so a pre-fix script
    (no cd) yields an UNKNOWN/ERRORED verdict and this test fails — with
    zero contact with the real engine or its .venv.
    """
    engine_dir, _ = _make_fake_engine(tmp_path, cwd_check=True)
    log = _run({**_env_base(tmp_path),
                "_ENGINE_DIR": engine_dir,
                "SUPERVISE_DRYRUN": "1"}, cwd="/")
    assert "UNKNOWN/ERRORED" not in log, \
        "gate could not resolve src.watchdog.market_status from foreign cwd " \
        f"(gate output: {log!r})"
    assert "dry-run: gate passed" in log, \
        "gate produced neither a pass nor a clean defer"


def test_gate_defers_on_confirmed_closed(tmp_path):
    """A CONFIRMED closed market must defer (the only defer reason)."""
    engine_dir, _ = _make_fake_engine(tmp_path, gate_output="CLOSED")
    log = _run({**_env_base(tmp_path),
                "_ENGINE_DIR": engine_dir,
                "SUPERVISE_DRYRUN": "1"}, cwd="/")
    assert "deferring until open" in log
    assert "dry-run: gate passed" not in log


def test_gate_fails_open_on_unknown_status(tmp_path):
    """Gate status unknown/errored must log LOUDLY and FAIL-OPEN.

    Regression for drill #2's failure mode: when the gate cannot resolve
    (cron cwd broke the import -> status=unknown -> deferred), the stack
    stayed down with held positions unmanaged.  Any non-OPEN/non-CLOSED
    verdict (garbage, traceback, empty) must log a WARN and still launch.
    """
    engine_dir, _ = _make_fake_engine(tmp_path, gate_output="BROKEN")
    log = _run({**_env_base(tmp_path),
                "_ENGINE_DIR": engine_dir,
                "SUPERVISE_DRYRUN": "1"}, cwd="/")
    assert "WARN: market-hours gate UNKNOWN/ERRORED" in log, \
        "unknown gate status must be logged loudly"
    assert "FAIL-OPEN" in log
    assert "BROKEN" in log, "gate output should be echoed for diagnosis"
    assert "dry-run: gate passed" in log, \
        "unknown gate status must NOT defer — the stack must start"


def test_run_helper_survives_repeated_calls_with_same_dict(tmp_path):
    """Regression: _run must copy env_extra instead of popping the caller's
    dict.  Two runs with the SAME dict must both work (pre-fix the second
    call KeyError'd on the already-popped _LOG_PATH)."""
    engine_dir, _ = _make_fake_engine(tmp_path, gate_output="OPEN")
    env = {**_env_base(tmp_path),
           "_ENGINE_DIR": engine_dir,
           "SUPERVISE_DRYRUN": "1"}
    log1 = _run(env, cwd="/")
    assert "dry-run: gate passed" in log1
    # Same dict again — must not KeyError (dict copy added in _run).
    log2 = _run(env, cwd="/")
    assert "dry-run: gate passed" in log2


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
    watchdog running elsewhere on the same host.

    The second run is verified by "no NEW log lines" (``log2 == log1``),
    NOT by ``log2.strip() == ""``: the log file legitimately retains run
    1's launch lines, so a whole-file-empty assertion can never pass even
    when the second run is a perfect no-op (this exact broken assertion
    made the suite red: 2026-09-13)."""
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
        # Settle: the setsid'd watchdog's /proc cmdline can lag a few ms
        # behind its marker write, so a back-to-back run could miss it via
        # pgrep (production ticks are 60s apart — this is purely a test
        # cadence artifact).  Give the process table a moment to settle.
        time.sleep(0.5)
        log2 = _run(env, cwd="/", timeout=30)
        assert log2 == log1, \
            "second run must be a silent no-op (no new log lines) — " \
            "if it launched a watchdog, the no-op guard failed"
        assert _watchdog_count(engine_dir) == 1, \
            "supervisor spawned a second watchdog"
        assert log1.count("launched watchdog") == 1
    finally:
        _cleanup_fake_watchdog(engine_dir, marker)


def _cleanup_fake_watchdog(engine_dir, marker):
    """Kill the fake watchdog AND its process group.

    The fake watchdog is setsid'd by the real launch block, so it is its
    own session/process-group leader (PGID == pid); killpg() reaps it and
    its ``sleep`` child together so nothing outlives the test.  A pgrep
    pattern scoped to THIS test's fake engine dir is the fallback — it can
    never match the real stack (different path).
    """
    pid = None
    try:
        pid = int(marker.read_text().strip())
    except Exception:
        pass
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    subprocess.run(["pkill", "-f", f"bash {engine_dir}/watchdog[.]sh"],
                   capture_output=True)