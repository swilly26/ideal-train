#!/usr/bin/env bash
# Keep the AlgoFlow traders alive and recover them from stalled data/network calls.
set -u
# Engine root.  ALGOFLOW_ENGINE_DIR redirects EVERY path this script
# touches (logs, pid files, state, the start_*.sh launchers) -- it is how
# the test suite runs this script hermetically against a throwaway root.
# Production never sets it, so the default keeps the live deployment
# identical.  Exported so the launchers it invokes resolve the same root.
ENGINE_DIR="${ALGOFLOW_ENGINE_DIR:-/home/team/shared/engine}"
export ALGOFLOW_ENGINE_DIR="$ENGINE_DIR"
LOG_DIR="$ENGINE_DIR/logs"

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
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
# Configurable via environment variables with safe defaults, so existing
# deployments keep their current behaviour unless they opt in/out.
#   WATCHDOG_STALE_SECONDS           how old a log must be before it is
#                                     considered stale (default 300)
#   WATCHDOG_SKIP_STALE_WHEN_CLOSED  skip the stale-log kill while the market
#                                     is closed (pre-open / after close /
#                                     weekends) — default 1 (on). Safe: a
#                                     trader that is NOT running is still
#                                     restarted regardless of market hours.
: "${WATCHDOG_STALE_SECONDS:=300}"
: "${WATCHDOG_SKIP_STALE_WHEN_CLOSED:=1}"
export WATCHDOG_STALE_SECONDS WATCHDOG_SKIP_STALE_WHEN_CLOSED
STALE_SECONDS="$WATCHDOG_STALE_SECONDS"
CHECK_INTERVAL="${WATCHDOG_CHECK_INTERVAL:-60}"
# A spawn that carries CHECK_INTERVAL=0 (the test env that produced the
# 2026-09-23 storm) must never busy-loop: floor it at 1s and say so.
if (( CHECK_INTERVAL < 1 )); then
    echo "[watchdog] WATCHDOG_CHECK_INTERVAL=${CHECK_INTERVAL} raised to 1s (never busy-loop)" >&2
    CHECK_INTERVAL=1
fi
# Hard bounds for a spawned watchdog, so it cannot run forever even if
# nobody reaps it: WATCHDOG_MAX_ITERATIONS (checks) / WATCHDOG_MAX_SECONDS.
MAX_ITERATIONS="${WATCHDOG_MAX_ITERATIONS:-0}"
MAX_SECONDS="${WATCHDOG_MAX_SECONDS:-0}"
ITERATIONS=0
WATCHDOG_STARTED_AT="$(date +%s)"
# The restart decision lives in a testable Python policy module.
# watchdog.sh passes the per-trader facts and lets it decide; the policy
# applies the market-hours exemption (see src/watchdog/policy.py).
PYTHON_BIN="$ENGINE_DIR/.venv/bin/python"
WATCHDOG_POLICY_CMD=( "$PYTHON_BIN" -m src.watchdog.policy )
mkdir -p "$LOG_DIR"
cd "$ENGINE_DIR" || exit 1
log_action() {
    printf '%s [watchdog] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$WATCHDOG_LOG"
}
# Return the newest output/trade log for a trader. Launchers create a new
# timestamped runner log on each start, while the trader also writes a daily
# trade log, so consider both names.
latest_log() {
    local kind="$1"
    if [[ "$kind" == "live" ]]; then
        ls -1t "$LOG_DIR"/runner_*.out "$LOG_DIR"/trades_*.log 2>/dev/null | head -n 1
    else
        ls -1t "$LOG_DIR"/turbo_runner_*.out "$LOG_DIR"/turbo_*.log 2>/dev/null | head -n 1
    fi
}
restart_trader() {
    local kind="$1" reason="$2" pid state_file prev_pid out new_pid
    if [[ "$kind" == "live" ]]; then
        state_file="$LOG_DIR/.watchdog_last_pid_live"
    else
        state_file="$LOG_DIR/.watchdog_last_pid_turbo"
    fi
    prev_pid="$(cat "$state_file" 2>/dev/null || true)"
    # Durable restart attribution: every restart records WHY it happened and
    # the previous pid (or FIRST-BOOT) so future incidents are attributable
    # from the log alone (2026-09-14: two unattributed restarts).
    if [[ -n "$prev_pid" ]]; then
        log_action "restart decision: $kind trader (reason=$reason, previous_pid=$prev_pid)"
    else
        log_action "restart decision: $kind trader (reason=$reason, previous_pid=FIRST-BOOT)"
    fi
    if [[ "$kind" == "live" ]]; then
        pgrep -f '[l]ive_trader.py' | while read -r pid; do
            kill "$pid" 2>/dev/null || true
            log_action "killed stalled live trader pid=$pid"
        done
        log_action "starting live trader"
        out="$(bash "$ENGINE_DIR/start_trader.sh" 2>&1)"
        printf '%s\n' "$out" >> "$WATCHDOG_LOG"
        new_pid="$(printf '%s\n' "$out" | sed -n 's/.*Started! PID: \([0-9][0-9]*\).*/\1/p' | head -n 1)"
    else
        pgrep -f '[t]urbo_trader.py' | while read -r pid; do
            kill "$pid" 2>/dev/null || true
            log_action "killed stalled turbo trader pid=$pid"
        done
        log_action "starting turbo trader"
        out="$(bash "$ENGINE_DIR/start_turbo.sh" 2>&1)"
        printf '%s\n' "$out" >> "$WATCHDOG_LOG"
        new_pid="$(printf '%s\n' "$out" | sed -n 's/.*Started! PID: \([0-9][0-9]*\).*/\1/p' | head -n 1)"
    fi
    if [[ -n "$new_pid" ]]; then
        printf '%s\n' "$new_pid" > "$state_file"
    fi
}
check_trader() {
    local kind="$1" log_file age
    if [[ "$kind" == "live" ]]; then
        if ! pgrep -f '[l]ive_trader.py' >/dev/null; then
            log_action "live trader is not running"
            restart_trader live "process-not-running"
            return
        fi
    elif ! pgrep -f '[t]urbo_trader.py' >/dev/null; then
        log_action "turbo trader is not running"
        restart_trader turbo "process-not-running"
        return
    fi
    log_file="$(latest_log "$kind")"
    if [[ -z "$log_file" ]]; then
        log_action "$kind trader has no log file; restarting"
        restart_trader "$kind" "no-log-file"
        return
    fi
    age=$(( $(date +%s) - $(stat -c %Y "$log_file") ))
    if (( age > STALE_SECONDS )); then
        # Ask the Python policy for the verdict. It applies the market-hours
        # exemption: a stale log is NOT a reason to restart while the market
        # is closed (the traders legitimately log nothing pre-open). A dead
        # process is handled above and always restarted.
        output="$(
            "${WATCHDOG_POLICY_CMD[@]}" \
                --process-running=yes --has-log=yes --age="$age"
        )"
        if [[ -z "$output" ]]; then
            # Policy helper failed — fall back to the legacy stale-kill.
            log_action "watchdog policy helper failed for $kind; restoring legacy stale-kill"
            restart_trader "$kind" "policy-helper-failed"
            return
        fi
        action="${output%%|*}"
        reason="${output#*|}"
        if [[ "$action" == "RESTART" ]]; then
            log_action "$kind trader log is stale (${age}s): $log_file; restarting ($reason)"
            restart_trader "$kind" "log-stale"
        else
            log_action "$kind trader log is stale (${age}s) but exempted: $reason"
        fi
    fi
}
log_action "watchdog started (interval=${CHECK_INTERVAL}s, stale=${STALE_SECONDS}s, skip-stale-when-closed=${WATCHDOG_SKIP_STALE_WHEN_CLOSED})"
# Hourly healthy tick: the watchdog logs nothing while everything is fine,
# so an hour-long silence is indistinguishable from a dead watchdog
# (2026-09-14).  One line per hour proves liveness — silence is never
# ambiguous while the process is up.
LAST_HEALTHY_LOG="$(date +%s)"
while true; do
    ITERATIONS=$(( ITERATIONS + 1 ))
    check_trader live
    check_trader turbo
    if (( MAX_ITERATIONS > 0 )) && (( ITERATIONS >= MAX_ITERATIONS )); then
        log_action "watchdog leaving the loop after ${ITERATIONS} iteration(s) (WATCHDOG_MAX_ITERATIONS=$MAX_ITERATIONS)"
        break
    fi
    if (( MAX_SECONDS > 0 )) && (( $(date +%s) - WATCHDOG_STARTED_AT >= MAX_SECONDS )); then
        log_action "watchdog leaving the loop after ${MAX_SECONDS}s (WATCHDOG_MAX_SECONDS)"
        break
    fi
    now_ts="$(date +%s)"
    if (( now_ts - LAST_HEALTHY_LOG >= 3600 )); then
        live_state="yes"; pgrep -f '[l]ive_trader.py' >/dev/null || live_state="no"
        turbo_state="yes"; pgrep -f '[t]urbo_trader.py' >/dev/null || turbo_state="no"
        log_action "watchdog healthy tick (live=${live_state}, turbo=${turbo_state})"
        LAST_HEALTHY_LOG="$now_ts"
    fi
    sleep "$CHECK_INTERVAL"
done
