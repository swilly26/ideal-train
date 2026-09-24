# Running the test suite safely on a box where the live stack exists

**Read this before running `pytest` in this repo.** The suite spawns shell
scripts. One of those scripts is a supervisor that starts real traders, so a
test bug can turn a test run into a live-trading incident.

## The incident this document exists for (2026-09-23)

A full-suite run executed the **production** copy of `watchdog.sh`
(`/home/team/shared/engine/watchdog.sh`) instead of a throwaway copy:

```python
SCRIPT = os.environ.get("WATCHDOG_SCRIPT", f"/home/team/shared/engine/watchdog.sh")
subprocess.run(["bash", str(SCRIPT)], env={"ALGOFLOW_ENGINE_DIR": str(tmp), ...})
```

That copy **hardcoded** its engine root, so the test's redirect, its stubbed
process probe and its `WATCHDOG_MAX_ITERATIONS` bound were all silently
ignored. It resolved the production root, found no traders running, and
restarted the real `live_trader.py` / `turbo_trader.py` in a tight loop
(`WATCHDOG_CHECK_INTERVAL=0` came from the test env). Result: 34
`live_trader.py` + 23 `turbo_trader.py` + 4 `watchdog.sh` processes, load
average ~50 on a 2-core box, and 61 orphans when the suite was killed.

**A redirect that the script ignores is worse than no redirect, because it
looks safe.**

## The rule

1. **The suite never executes a live-stack script from the live engine root.**
   It executes the copy in *the tree under test* (`watchdog.sh`,
   `start_*.sh`, `scripts/supervise_traders.sh`) with its engine root
   redirected into a throwaway directory.
2. **The live stack stays down while the suite runs.** The owner has paused
   trading: 0 traders, 0 watchdogs, and the crontab supervisor line stays
   commented out. Never start, restart or "helpfully" re-launch any of it.
3. **Anything the suite spawns is bounded and reaped.** A spawned watchdog
   carries `WATCHDOG_MAX_ITERATIONS` (or `WATCHDOG_MAX_SECONDS`), and teardown
   kills this session's process groups — including after a `kill -9` of the
   suite, because these scripts are only ever spawned through the guard.

## Safe invocation (copy-paste)

```bash
cd /tmp/wt-my-branch                 # the tree under test (worktree recommended)
export WATCHDOG_SCRIPT="$PWD/watchdog.sh"
export SUPERVISE_SCRIPT="$PWD/scripts/supervise_traders.sh"
PY=/home/team/shared/engine/.venv/bin/python   # a working interpreter, any tree

pgrep -af 'watchdog.sh|live_trader.py|turbo_trader.py'   # BEFORE: expect no output

nice -n 19 "$PY" -m pytest tests -q; echo "pytest exit=$?"

pgrep -af 'watchdog.sh|live_trader.py|turbo_trader.py'   # AFTER: MUST be empty
crontab -l | grep supervise_traders                      # MUST still be commented out
```

Notes:

* `WATCHDOG_SCRIPT` / `SUPERVISE_SCRIPT` are read by `tests/test_watchdog_restart.py`
  and `tests/test_supervise_script.py`. **On any branch that predates the
  containment merge they are mandatory**; after the merge the suite refuses a
  spawn aimed at the live root even if you forget them.
* Run **one suite at a time**, `nice -n 19`: this is a 2-core box, and a loaded
  box has already produced a load-sensitive failure (see *Flaky under load*).
* Run the suite **from the tree under test**, never from a copy whose `tests/`
  directory did not come with it.
* Never `pkill -f watchdog` / `pkill -f live_trader` — a pattern kill is what
  turns a small mistake into a 61-process storm. Kill by pid.

## If `pgrep` shows something after a run

1. `ps -o pid,ppid,pgid,lstart,cmd -p <pid>` — is it an orphan from this run
   (start time within the run window, cmdline naming a scratch or live path)?
2. `kill <pid>` by pid (SIGTERM first, `kill -9` only if needed). Kill the
   whole group with `kill -- -<pgid>` if it has children.
3. Re-check `pgrep`, check `crontab -l`, check
   `/home/team/shared/engine/logs/watchdog.log` and the broker for
   unexpected order activity, then **report it as a containment failure** —
   do not retry the run until it passes.

## What is enforced automatically (tests/containment.py)

Installed for the whole pytest session by `tests/conftest.py`:

| # | Rule | Fails when |
|---|------|-----------|
| 1 | Every live-stack spawn carries an engine-root redirect (`ALGOFLOW_ENGINE_DIR` / `SUPERVISE_ENGINE_DIR`) that is not the production root | a test spawns a live script with no redirect |
| 2 | The executed script's source is inspected: a line that pins the engine root to the production root outside an honoured `${VAR:-default}` override refuses the spawn | the Sep-23 configuration (redirect a script ignores) |
| 3 | `ALGOFLOW_TEST_SESSION=<production engine root>` is injected into every child; the shell scripts *refuse to run* (exit 98) when they resolve their engine root to that value | a live script would operate the production root, even from a caller with no guard |
| 4 | A real watchdog loop must carry `WATCHDOG_MAX_ITERATIONS` / `WATCHDOG_MAX_SECONDS` | a spawned watchdog could spin forever |
| 5 | `live_trader.py` / `turbo_trader.py` are never spawned from the production root (import them instead) | a test starts a real trader |
| 6 | At session end: escaped live-stack processes are killed by process group and the session **fails**; this session's strays are reaped | the suite touched the live stack, or left an orphan |

The refusal happens **before the fork**, so a violating test fails without
creating a process. `tests/test_suite_containment.py` pins all of it — the
pre-fix spawn pattern, the unhonoured-redirect case, the shell-side refusal,
and the loop bound.
