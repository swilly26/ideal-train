#!/usr/bin/env bash
# supervise_traders.sh — session-independent liveness supervisor for the
# ApexTrade trading stack (watchdog.sh + live_trader.py + turbo_trader.py).
#
# WHY THIS EXISTS
# --------------
# watchdog.sh, live_trader.py and turbo_trader.py all used to run inside the
# interactive session's process group, so an external teardown of that group
# killed all three at once — and watchdog.sh (also dead) could not restart
# anyone.  Recovery was manual (`nohup bash watchdog.sh`) and the unmanaged
# windows were 20 minutes to 5 hours (2026-08-26, 2026-09-09, 2026-09-10).
# This script fixes the recovery side.  It is invoked every minute by cron —
# session-independent by construction — and, only when the watchdog is
# actually dead AND the US market is open, starts the whole stack fully
# detached with `setsid`, so the new watchdog lives in its OWN session and
# survives the next teardown of any interactive session.
#
# INSTALLATION (run once as root — already done in this deployment)
# ----------------------------------------------------------------
#   1. apt-get install -y --no-install-recommends cron
#   2. setsid /usr/sbin/cron   # this container has no systemd; start the
#                              # daemon DETACHED so it also survives teardown
#   3. crontab -l 2>/dev/null | { cat; echo "* * * * * /home/team/shared/engine/scripts/supervise_traders.sh >> /home/team/shared/engine/logs/supervisor_cron.log 2>&1"; } | crontab -
#   Verify: crontab -l | grep supervise_traders && pgrep -x cron
#   After a container restart, repeat step 2 (and check step 3 persisted —
#   crontab survives on the persistent volume).
#
# SAFETY PROPERTIES
# -----------------
#   * Idempotent — no-op whenever THIS engine's watchdog.sh is already
#     running (pgrep guard scoped to $ENGINE_DIR, so a supervisor for one
#     deployment can never see another deployment's watchdog as "ours") even
#     if traders are missing (the watchdog restarts them itself).
#   * Locked — flock on $LOCKFILE serialises concurrent invocations, so
#     multiple supervisors can NEVER spawn a second watchdog (cron fires
#     every minute; a slow start just waits).  The lock is held ONLY for the
#     brief lifetime of each supervisor run: the launched child closes fd 9
#     (`exec 9>&-` inside the launch subshell) BEFORE setsid forks, so no
#     descendant of the watchdog can ever pin the lock after a kill — the
#     next cron tick after a teardown finds the lock free and relaunches.
#   * Gated — the stack is only started during regular trading hours
#     (DST-aware, pure-stdlib check via `python -c 'from
#     src.watchdog.market_status ...'`, run from $ENGINE_DIR so the import
#     resolves regardless of cron's cwd — the gate MUST NOT depend on the
#     caller's working directory, because cron starts jobs in the crontab
#     owner's $HOME, not in the repo).  SUPERVISE_FORCE=1 overrides the gate
#     for drills/emergency.
#   * Detached — `setsid nohup ... < /dev/null` puts the watchdog in a new
#     session with no controlling terminal; the traders it spawns detach the
#     same way via start_trader.sh / start_turbo.sh.
#
# TEST HOOKS (env overrides, all default to production values)
# -----------------------------------------------------------
#   SUPERVISE_ENGINE_DIR  — engine root (default /home/team/shared/engine)
#   SUPERVISE_LOG_DIR     — log directory (default $ENGINE_DIR/logs)
#   SUPERVISE_LOG         — supervisor log file (default $LOG_DIR/supervise.log)
#   SUPERVISE_LOCKFILE    — flock file (default /tmp/supervise_traders.lock)
#   SUPERVISE_DRYRUN=1    — log the gate result and exit WITHOUT launching
#                           (used by tests; also skips the pgrep guard so the
#                           gate path is exercised deterministically)
set -u
ENGINE_DIR="${SUPERVISE_ENGINE_DIR:-/home/team/shared/engine}"
# Exported so the watchdog it launches (and the launchers below that)
# resolve the SAME engine root instead of falling back to the production
# default -- redirecting only this script used to leave the launched
# watchdog pointed at the live tree.
export ALGOFLOW_ENGINE_DIR="$ENGINE_DIR"
LOG_DIR="${SUPERVISE_LOG_DIR:-$ENGINE_DIR/logs}"
SUPERVISE_LOG="${SUPERVISE_LOG:-$LOG_DIR/supervise.log}"
LOCKFILE="${SUPERVISE_LOCKFILE:-/tmp/supervise_traders.lock}"
PYTHON_BIN="$ENGINE_DIR/.venv/bin/python"

# ── Test-session guard (see tests/containment.py, docs/SAFE_TEST_RUN.md) ──
# The test suite exports ALGOFLOW_TEST_SESSION=<production engine root> into
# every process it spawns.  If this script resolves its engine root to that
# value it is about to operate the LIVE stack (start/kill real traders, write
# real logs) -- refuse, loudly.  This is exactly the 2026-09-23 storm: a test
# ran the production copy of watchdog.sh while asking it to use a throwaway
# root; the copy ignored the request and restarted the real traders in a tight
# loop.  Production never sets the marker, so this branch is inert there.
if [[ -n "${ALGOFLOW_TEST_SESSION:-}" ]] \
   && [[ "$(realpath -m -- "$ENGINE_DIR")" == "$(realpath -m -- "$ALGOFLOW_TEST_SESSION")" ]]; then
    echo "REFUSING to run: this is a test session (ALGOFLOW_TEST_SESSION set) and ENGINE_DIR resolves to the production engine root:" >&2
    echo "  ENGINE_DIR=$ENGINE_DIR" >&2
    echo "A test must run a throwaway copy with a redirected engine root; see docs/SAFE_TEST_RUN.md." >&2
    exit 98
fi
mkdir -p "$LOG_DIR"
log() {
    printf '%s [supervise] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$SUPERVISE_LOG"
}
# ── Work from the engine dir for the whole run ────────────────────
# cron starts this job in the crontab owner's $HOME, so EVERY python -c
# import below would fail with "No module named 'src'" unless we cd first
# (this exact bug kept the supervisor deferring during the 2026-09-10 kill
# drill and left the stack down ~7 min until manual restore).
cd "$ENGINE_DIR"
# ── Serialise concurrent invocations ───────────────────────────────
exec 9>"$LOCKFILE"
flock -n 9 || { log "another supervisor run holds the lock — exiting"; exit 0; }
# ── Already supervised? no-op (scoped to THIS engine's watchdog) ──
# Primary check: pgrep on the engine-scoped watchdog cmdline.  Secondary:
# the last pid THIS supervisor launched (`.supervise_last_pid`) if it is
# still alive — covers the instant after launch when the setsid'd child's
# /proc cmdline is not yet visible to a back-to-back pgrep (a cron-adjacent
# race that otherwise double-launches the stack), and covers a watchdog
# started with a relative path that the absolute-path pattern cannot see.
# After a kill, the stale pid is dead (kill -0 fails) and the next tick
# relaunches.
WATCHDOG_RE="bash ${ENGINE_DIR}/watchdog[.]sh"
LAST_PID_FILE="$LOG_DIR/.supervise_last_pid"
last_pid="$(cat "$LAST_PID_FILE" 2>/dev/null || true)"
if [[ "${SUPERVISE_DRYRUN:-0}" != "1" ]] && {
    pgrep -f "$WATCHDOG_RE" >/dev/null \
    || { [[ -n "$last_pid" ]] && kill -0 "$last_pid" 2>/dev/null; }
}; then
    # Hourly healthy tick: while the stack is healthy this script is a
    # silent no-op every minute, so an empty supervise.log is ambiguous —
    # it cannot distinguish "supervisor not running" from "everything
    # fine" (2026-09-14: supervise.log had zero lines Monday because cron
    # was dead, and nobody could tell).  One line per hour proves this
    # supervisor is alive — silence is never ambiguous.
    HEARTBEAT_FILE="$LOG_DIR/.supervise_last_healthy"
    last_h="$(cat "$HEARTBEAT_FILE" 2>/dev/null || true)"
    now_h="$(date +%s)"
    if [[ -z "$last_h" ]] || (( now_h - last_h >= 3600 )); then
        log "healthy tick: watchdog alive — stack under watchdog control"
        echo "$now_h" > "$HEARTBEAT_FILE"
    fi
    exit 0
fi
# ── Market-hours gate (DST-aware, pure stdlib, no network) ─────────
# Uses `python -c` (NOT `-m`) so the gate works on ANY checked-out tree
# state — the supervisor must keep functioning even before this branch is
# merged, and must never depend on a file that a later checkout could
# remove.  Runs from $ENGINE_DIR (cd'd above), so it is cwd-independent.
# FAIL-OPEN on unknown/error: the ONLY reason to defer is a CONFIRMED
# closed market.  If the gate cannot run (broken tree, import error,
# garbage output) the stack must still start — a rare pre-open boot is the
# lesser risk compared to leaving held positions unmanaged (their GTC
# broker stops cover the pre-open window; the traders re-place them on
# boot).  This is the exact mode that broke drill #2: a gate that could
# not import from cron's cwd looked "closed" and the stack stayed down.
if [[ "${SUPERVISE_FORCE:-0}" != "1" ]]; then
    gate_out="$("$PYTHON_BIN" -c 'from src.watchdog.market_status import in_rth_schedule; print("OPEN" if in_rth_schedule() else "CLOSED")' 2>&1)"
    status="$(printf '%s\n' "$gate_out" | tail -n 1)"
    if [[ "$status" == "CLOSED" ]]; then
        log "watchdog down but market closed (status=CLOSED) — deferring until open"
        exit 0
    fi
    if [[ "$status" != "OPEN" ]]; then
        log "WARN: market-hours gate UNKNOWN/ERRORED (status=${status:-empty}) — FAIL-OPEN: starting stack; positions must stay managed"
        log "gate output: ${gate_out}"
    fi
fi
# ── Test hook: dry-run reports the gate verdict without launching ──
if [[ "${SUPERVISE_DRYRUN:-0}" == "1" ]]; then
    log "dry-run: gate passed, would start stack detached (setsid)"
    exit 0
fi
# ── Start the stack fully detached ─────────────────────────────────
# The subshell closes fd 9 (the supervisor's flock fd) BEFORE setsid forks,
# so neither the watchdog nor any of its descendants can hold the lock after
# this run exits.  Without this, a teardown that kills the stack can leave
# trailing children pinning the lock for up to ~60 s, delaying the next
# cron-tick relaunch (observed in the 2026-09-10 drill).
log "watchdog not running — starting stack detached (setsid)"
(
    exec 9>&-
    setsid nohup bash "$ENGINE_DIR/watchdog.sh" < /dev/null >> "$LOG_DIR/watchdog_supervisor.log" 2>&1 &
    echo $! > "$LOG_DIR/.supervise_last_pid"
)
log "launched watchdog (pid $(cat "$LOG_DIR/.supervise_last_pid" 2>/dev/null)) — traders will be (re)started by watchdog as needed"
exit 0