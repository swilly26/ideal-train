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
#   * Idempotent — no-op whenever watchdog.sh is already running (pgrep
#     guard) even if traders are missing (watchdog restarts them itself).
#   * Locked — flock on /tmp/supervise_traders.lock serialises concurrent
#     invocations, so multiple supervisors can NEVER spawn a second
#     watchdog (cron fires every minute; a slow start just waits).
#   * Gated — the stack is only started during regular trading hours
#     (DST-aware, pure-stdlib check via `python -m src.watchdog.market_status`),
#     so a pre-open boot can never trigger the post-startup cleanup
#     liquidation that has historically realised losses on leftover
#     positions.  SUPERVISE_FORCE=1 overrides the gate for drills/emergency.
#   * Detached — `setsid nohup ... < /dev/null` puts the watchdog in a new
#     session with no controlling terminal; the traders it spawns detach the
#     same way via start_trader.sh / start_turbo.sh.
set -u

ENGINE_DIR="/home/team/shared/engine"
LOG_DIR="$ENGINE_DIR/logs"
SUPERVISE_LOG="$LOG_DIR/supervise.log"
LOCKFILE="/tmp/supervise_traders.lock"
PYTHON_BIN="$ENGINE_DIR/.venv/bin/python"

mkdir -p "$LOG_DIR"
log() {
    printf '%s [supervise] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$SUPERVISE_LOG"
}

# ── Serialise concurrent invocations ───────────────────────────────
exec 9>"$LOCKFILE"
flock -n 9 || { log "another supervisor run holds the lock — exiting"; exit 0; }

# ── Already supervised? no-op ──────────────────────────────────────
if pgrep -f '[w]atchdog\.sh' >/dev/null; then
    exit 0
fi

# ── Market-hours gate (DST-aware, pure stdlib, no network) ─────────
# Uses `python -c` (NOT `-m`) so the gate works on ANY checked-out tree
# state — the supervisor must keep functioning even before this branch is
# merged, and must never depend on a file that a later checkout could
# remove.
if [[ "${SUPERVISE_FORCE:-0}" != "1" ]]; then
    status="$("$PYTHON_BIN" -c 'from src.watchdog.market_status import in_rth_schedule; print("OPEN" if in_rth_schedule() else "CLOSED")' 2>/dev/null)"
    if [[ "$status" != "OPEN" ]]; then
        log "watchdog down but market closed (status=${status:-unknown}) — deferring until open"
        exit 0
    fi
fi

# ── Start the stack fully detached ─────────────────────────────────
log "watchdog not running — starting stack detached (setsid)"
cd "$ENGINE_DIR"
setsid nohup bash "$ENGINE_DIR/watchdog.sh" < /dev/null >> "$LOG_DIR/watchdog_supervisor.log" 2>&1 &
log "launched watchdog (pid $!) — traders will be (re)started by watchdog as needed"

exit 0