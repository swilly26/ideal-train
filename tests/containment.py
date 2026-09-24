"""HARD CONTAINMENT: a test run must be incapable of touching the live stack.

WHY THIS EXISTS (2026-09-23 storm)
----------------------------------
A full-suite run spawned the REAL ``watchdog.sh`` (the copy at the production
engine root ``/home/team/shared/engine``) instead of a throwaway copy.  That
watchdog's engine root was hardcoded, so the test's redirect
(``ALGOFLOW_ENGINE_DIR=<tmp>``), its stubbed process probe
(``WATCHDOG_PIDS_CMD``) and its loop bound (``WATCHDOG_MAX_ITERATIONS``) were
all silently ignored: it resolved the production root, found no traders
running (the owner had paused trading) and restarted the real
``live_trader.py`` / ``turbo_trader.py`` in a tight loop
(``WATCHDOG_CHECK_INTERVAL=0`` came straight from the test env) -- 34 + 23 + 4
processes and a load average of ~50 on a 2-core box.  Killing the suite with
``-9`` orphaned them.

The offending call looked like this (it is what ``tests/test_watchdog_restart.py``
did on branch ``feature/protection-never-naked``, and a compiled copy of that
file was sitting in the shared tree's ``tests/`` directory during the storm)::

    SCRIPT = os.environ.get("WATCHDOG_SCRIPT", f"/home/team/shared/engine/watchdog.sh")
    subprocess.run(["bash", str(SCRIPT)], env={... "ALGOFLOW_ENGINE_DIR": tmp ...})

The redirect was there.  The script executed was the *pre-fix* one at the
production root, which ignored it.  Nothing in the suite noticed.

THE CONTRACT ENFORCED HERE
--------------------------
Every process the suite spawns is inspected *before* it exists:

1. **The engine root must be redirected.**  A spawn of a live-stack script
   (``watchdog.sh`` / ``start_trader.sh`` / ``start_turbo.sh`` /
   ``supervise_traders.sh``) must carry an engine-root override
   (``ALGOFLOW_ENGINE_DIR`` / ``SUPERVISE_ENGINE_DIR``) that is not the
   production root.
2. **The executed script must actually honour that override.**  The check
   reads the script's source and refuses the spawn when a non-comment line
   pins the engine root to the production root outside a ``${VAR:-default}``
   override that this child environment sets.  That is the version-skew case
   above: a redirect nobody reads is worse than no redirect, because it looks
   safe.
3. **The session marker must reach the child.**  ``ALGOFLOW_TEST_SESSION``
   (value: the production engine root) is injected into every spawned child;
   the live-stack shell scripts refuse to run when they resolve their engine
   root to that value, so containment does not depend on the test-side guard
   being present.
4. **A watchdog loop must be bounded.**  A spawn of a real watchdog loop must
   set ``WATCHDOG_MAX_ITERATIONS`` or ``WATCHDOG_MAX_SECONDS``, so a spawned
   watchdog can never spin forever.
5. **A production trader module is never spawned.**  ``live_trader.py`` /
   ``turbo_trader.py`` may only be executed when the file is outside the
   production engine root (a stub).  Importing them is fine.

The refusal is fail-closed and happens *before* ``fork``: a violating spawn
raises :class:`ContainmentViolation` and no process is created, so there is
nothing to orphan.  Whatever the tests *are* allowed to spawn is reaped by
process group at session teardown, and the session FAILS if a live-stack
process appeared that was not there at session start.

Production never sets ``ALGOFLOW_TEST_SESSION``, so none of this changes
production behaviour; the shell scripts' guard is inert outside a test run.
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

#: The live engine root.  Overridable so the checker itself can be unit-tested
#: against a synthetic "production" root (never against the real one).
PRODUCTION_ENGINE_ROOT = os.environ.get(
    "ALGOFLOW_PRODUCTION_ENGINE_ROOT", "/home/team/shared/engine"
)

#: Session marker injected into every child.  Its VALUE is the production
#: engine root: a live-stack script that resolves its engine root to that value
#: while this marker is set refuses to run (exit 98, see watchdog.sh).
SESSION_ENV = "ALGOFLOW_TEST_SESSION"

#: Env hooks that redirect a live-stack script's engine root.
ENGINE_OVERRIDE_VARS = ("ALGOFLOW_ENGINE_DIR", "SUPERVISE_ENGINE_DIR")

#: Names that identify the live stack.  Everything here is checked on spawn.
LIVE_SCRIPT_NAMES = (
    "watchdog.sh",
    "start_trader.sh",
    "start_turbo.sh",
    "supervise_traders.sh",
)
TRADER_MODULE_NAMES = ("live_trader.py", "turbo_trader.py")
LIVE_STACK_NAMES = LIVE_SCRIPT_NAMES + TRADER_MODULE_NAMES

#: A real watchdog loop: spawns of these must carry an iteration/time bound.
WATCHDOG_LOOP_MARKERS = ("check_trader", "WATCHDOG_CHECK_INTERVAL")


class ContainmentViolation(RuntimeError):
    """A spawned child would have been able to reach the live stack."""


def canonical(path) -> str:
    """Canonical absolute path (symlinks resolved); need not exist."""
    return os.path.realpath(os.fspath(path))


def read_script(path) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def engine_root_pins(text: str, env, production_root=None) -> list[tuple[int, str]]:
    """Non-comment lines that pin the engine root to the production root.

    A line is *not* a pin when the production path sits in the default of a
    ``${VAR:-/home/team/shared/engine}`` override that ``env`` actually sets --
    that is the redirect working as intended.  Everything else (a bare
    ``ENGINE_DIR="/home/team/shared/engine"``, a ``cd /home/team/shared/engine``
    in a launcher) is a pin: the script uses the production root whatever the
    caller asked for.
    """
    prod = canonical(production_root or PRODUCTION_ENGINE_ROOT)
    pins: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        start = 0
        while True:
            idx = line.find(prod, start)
            if idx == -1:
                break
            start = idx + len(prod)
            prefix = line[max(0, idx - 120):idx]
            honoured = False
            overrides = re.findall(r"\$\{(\w+):-", prefix)
            if overrides:
                var = overrides[-1]
                val = (env or {}).get(var)
                if var in ENGINE_OVERRIDE_VARS and val and canonical(val) != prod:
                    honoured = True
            if not honoured:
                pins.append((lineno, stripped))
    return pins


def is_bounded(env) -> bool:
    """True when the child cannot loop forever."""
    for var in ("WATCHDOG_MAX_ITERATIONS", "WATCHDOG_MAX_SECONDS"):
        raw = str((env or {}).get(var, "")).strip()
        if raw.isdigit() and int(raw) > 0:
            return True
    return False


def is_watchdog_loop(text: str) -> bool:
    return any(marker in text for marker in WATCHDOG_LOOP_MARKERS)


def live_hits(argv) -> list[str]:
    """The argv entries that name a live-stack script or trader module."""
    return [os.fspath(a) for a in argv if Path(os.fspath(a)).name in LIVE_STACK_NAMES]


def _locate(argv_hits, cwd) -> Path | None:
    for raw in argv_hits:
        cand = Path(raw)
        if cand.is_file():
            return cand
        if not cand.is_absolute() and cwd is not None:
            alt = Path(cwd) / cand
            if alt.is_file():
                return alt
    return None


def spawn_violations(argv, env=None, cwd=None, production_root=None) -> list[str]:
    """Every reason this spawn could reach the live stack (empty = safe).

    Pure decision function: it never executes anything, so the checker itself
    can be tested against a synthetic production root.
    """
    prod = canonical(production_root or PRODUCTION_ENGINE_ROOT)
    env = dict(os.environ) if env is None else dict(env)
    args = [os.fspath(a) for a in argv]
    hits = live_hits(args)
    if not hits:
        return []

    violations: list[str] = []
    target = _locate(hits, cwd)
    children = " ".join(args)

    if not env.get(SESSION_ENV):
        violations.append(
            f"spawned without {SESSION_ENV}: the live-stack scripts' own "
            f"refusal guard cannot fire for `{children}`"
        )

    declared = {var: env[var] for var in ENGINE_OVERRIDE_VARS if env.get(var)}
    if not declared:
        violations.append(
            f"no engine-root redirect ({' / '.join(ENGINE_OVERRIDE_VARS)}) in "
            f"the child env: `{children}` would resolve its engine root to "
            f"whatever the script defaults to -- the production root {prod}"
        )
    for var, val in declared.items():
        if canonical(val) == prod:
            violations.append(
                f"{var}={val} points at the production engine root {prod}"
            )

    if target is None:
        if any(Path(h).name in TRADER_MODULE_NAMES for h in hits):
            violations.append(
                f"`{children}` starts a trader process and the file could not "
                f"be located -- a test must never start a real trader"
            )
        return violations

    text = read_script(target)
    if (Path(target).name in TRADER_MODULE_NAMES
            and canonical(target).startswith(prod + os.sep)):
        violations.append(
            f"`{target}` is the production trader module -- a test must never "
            f"start it (import it instead)"
        )
    for lineno, line in engine_root_pins(text, env, prod):
        violations.append(
            f"{target}:{lineno} pins the engine root to the production root "
            f"ignoring the redirect, so this child would operate the LIVE "
            f"stack: {line}"
        )
    if is_watchdog_loop(text) and not is_bounded(env):
        violations.append(
            f"{target} is a watchdog loop with no bound: set "
            f"WATCHDOG_MAX_ITERATIONS (or WATCHDOG_MAX_SECONDS) so a spawned "
            f"watchdog cannot spin forever"
        )
    return violations


# ── process bookkeeping ─────────────────────────────────────────────────────

LIVE_STACK_PGREP = r"watchdog[.]sh|live_trader[.]py|turbo_trader[.]py|supervise_traders[.]sh"

#: pytest's own scratch root -- where the suite's throwaway stubs live.
PYTEST_TMP_MARKER = "/tmp/pytest-"


def live_stack_pids(exclude_pytest_tmp: bool = True) -> dict[int, str]:
    """pid -> cmdline for every live-stack-shaped process on the box.

    ``exclude_pytest_tmp`` (default) drops processes whose command line names a
    pytest scratch directory: the suite legitimately runs *stub* scripts called
    ``watchdog.sh`` inside ``tmp_path``, and those are not the live stack.  A
    process that would actually operate the live stack names a real path, so it
    is always kept.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-af", LIVE_STACK_PGREP],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found: dict[int, str] = {}
    for line in out.splitlines():
        pid_str, _, cmd = line.partition(" ")
        if pid_str.isdigit() and int(pid_str) == os.getpid():
            continue
        if exclude_pytest_tmp and PYTEST_TMP_MARKER in cmd:
            continue
        found[int(pid_str)] = cmd
    return found


def production_processes() -> dict[int, str]:
    """Live-stack processes that name the PRODUCTION engine root.

    This is the `pgrep` check the safe-invocation note asks for, minus the
    suite's own throwaway stubs: anything here is the live stack.
    """
    prod = canonical(PRODUCTION_ENGINE_ROOT)
    return {
        pid: cmd
        for pid, cmd in live_stack_pids(exclude_pytest_tmp=False).items()
        if prod in cmd and PYTEST_TMP_MARKER not in cmd
    }


def pids_referencing(fragment: str) -> list[int]:
    """pids whose /proc cmdline mentions a fragment (our scratch root)."""
    pids = []
    need = os.fsencode(fragment)
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                if need in fh.read():
                    pids.append(pid)
        except OSError:
            continue
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def kill_pid_group(pid: int, sig: int) -> None:
    """Signal a pid and its process group, ignoring anything already gone."""
    for target in (pid, _pgid_of(pid)):
        if not target:
            continue
        try:
            os.killpg(target, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(target, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass


def _pgid_of(pid: int) -> int | None:
    try:
        pgid = os.getpgid(pid)
        return pgid if pgid != os.getpgid(0) else None
    except OSError:
        return None


class ContainmentGuard:
    """Installed for a whole pytest session; fail-closed before every fork."""

    def __init__(self, scratch_root, production_root=None):
        self.scratch_root = Path(scratch_root)
        self.production_root = Path(canonical(production_root or PRODUCTION_ENGINE_ROOT))
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.children: list[int] = []
        self.refusals: list[str] = []
        self.baseline_pids: dict[int, str] = {}
        self._saved: dict = {}
        self.installed = False

    # ── install / teardown ─────────────────────────────────────────────
    def install(self) -> None:
        os.environ[SESSION_ENV] = str(self.production_root)
        os.environ.setdefault("ALGOFLOW_PRODUCTION_ENGINE_ROOT", str(self.production_root))
        self.baseline_pids = live_stack_pids()
        self._saved = {"Popen": subprocess.Popen, "system": os.system}
        subprocess.Popen = self._popen  # type: ignore[assignment]
        os.system = self._system  # type: ignore[assignment]
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        subprocess.Popen = self._saved["Popen"]  # type: ignore[assignment]
        os.system = self._saved["system"]  # type: ignore[assignment]
        self.installed = False
        os.environ.pop(SESSION_ENV, None)

    # ── the guard ──────────────────────────────────────────────────────
    def _child_env(self, env):
        """Guarantee the child carries the session marker.

        The marker is only *injected* when the caller has not set it: a test
        that sets it deliberately (to exercise the shell-side refusal against a
        synthetic root) must keep the value it chose.  The production root is
        always exported under its own name, which is what the shell guards
        compare against and what the spawn checker reads.
        """
        if env is None:
            os.environ.setdefault(SESSION_ENV, str(self.production_root))
            os.environ.setdefault(
                "ALGOFLOW_PRODUCTION_ENGINE_ROOT", str(self.production_root)
            )
            return None
        env = dict(env)
        env.setdefault(SESSION_ENV, str(self.production_root))
        env.setdefault("ALGOFLOW_PRODUCTION_ENGINE_ROOT", str(self.production_root))
        return env

    def check(self, argv, env, cwd) -> None:
        violations = spawn_violations(
            argv, env=env, cwd=cwd, production_root=self.production_root
        )
        if not violations:
            return
        self.refusals.append(" | ".join(violations) + f" [argv={list(argv)!r}]")
        raise ContainmentViolation(
            "test-harness containment refused a spawn that could reach the live "
            "trading stack (tests/containment.py):\n  - "
            + "\n  - ".join(violations)
            + f"\n  argv: {list(argv)!r}"
            + (f"\n  cwd: {cwd}" if cwd else "")
        )

    @staticmethod
    def _argv_of(args, kwargs):
        return args[0] if args else kwargs.get("args")

    def _popen(self, *args, **kwargs):
        raw = self._argv_of(args, kwargs)
        self._last_spawn_was_live_stack = False
        if raw is not None:
            if isinstance(raw, (str, bytes)):
                argv = shlex.split(raw.decode() if isinstance(raw, bytes) else raw)
            else:
                argv = [os.fspath(a) for a in raw]
            if live_hits(argv):
                self._last_spawn_was_live_stack = True
                env = self._child_env(kwargs.get("env"))
                if env is not None:
                    kwargs["env"] = env
                self.check(argv, env if env is not None else os.environ, kwargs.get("cwd"))
        proc = self._saved["Popen"](*args, **kwargs)
        if self._last_spawn_was_live_stack:
            try:
                self.children.append(int(proc.pid))
            except (AttributeError, TypeError, ValueError):
                pass
        return proc

    def _system(self, command):
        argv = shlex.split(command) if isinstance(command, str) else [os.fspath(a) for a in command]
        if live_hits(argv):
            self.check(argv, os.environ, None)
        return self._saved["system"](command)

    # ── teardown ───────────────────────────────────────────────────────
    def stray_pids(self) -> list[int]:
        """Processes this session started: recorded live-stack children plus
        anything whose cmdline still mentions a scratch root this session used
        (its own, or pytest's ``tmp_path`` tree).  A process with no scratch
        path in its command line is somebody else's -- never ours to kill."""
        victims = {pid for pid in self.children if pid}
        for fragment in (str(self.scratch_root), PYTEST_TMP_MARKER):
            victims.update(pids_referencing(fragment))
        return sorted(victims)

    def reap(self) -> list[int]:
        """Kill this session's strays by process group; return what we killed."""
        victims = self.stray_pids()
        for pid in victims:
            kill_pid_group(pid, signal.SIGTERM)
        if victims:
            time.sleep(0.4)
        for pid in self.stray_pids():
            kill_pid_group(pid, signal.SIGKILL)
        return victims

    def escaped_pids(self) -> dict[int, str]:
        """Live-stack processes that appeared during this session.

        Throwaway stubs under pytest's scratch tree are not escapes; a process
        that names the production engine root is.
        """
        now = live_stack_pids()
        return {pid: cmd for pid, cmd in now.items() if pid not in self.baseline_pids}

    def resolved_pids(self) -> dict[int, str]:
        """Live-stack processes at the end of the session."""
        return live_stack_pids()


def report(message: str) -> None:
    sys.stderr.write(f"\n[containment] {message}\n")
    sys.stderr.flush()
