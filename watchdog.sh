#!/usr/bin/env bash
# Keep the AlgoFlow traders alive and recover them from stalled data/network calls.
set -u
# Test/drill overridable (same convention as scripts/supervise_traders.sh)
# so the supervisor can be exercised hermetically; production never sets it.
ENGINE_DIR="${ALGOFLOW_ENGINE_DIR:-/home/team/shared/engine}"
LOG_DIR="$ENGINE_DIR/logs"
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
# The restart decision lives in a testable Python policy module.
# watchdog.sh passes the per-trader facts and lets it decide; the policy
# applies the market-hours exemption (see src/watchdog/policy.py).
PYTHON_BIN="${WATCHDOG_PYTHON:-$ENGINE_DIR/.venv/bin/python}"
WATCHDOG_POLICY_CMD=( "$PYTHON_BIN" -m src.watchdog.policy )
# Test/drill hooks (never set in production):
#   WATCHDOG_NOW             ISO ET timestamp used instead of the real clock
#   WATCHDOG_PIDS_CMD        prints a trader's live pids (one per line)
#   WATCHDOG_MAX_ITERATIONS  leave the loop after N checks (0 = forever)
#   ALGOFLOW_INCIDENT_LOG    one-line incident file (default: repo root)
NOW_ARG=()
if [[ -n "${WATCHDOG_NOW:-}" ]]; then
    NOW_ARG=( "--now=$WATCHDOG_NOW" )
fi
INCIDENT_LOG="${ALGOFLOW_INCIDENT_LOG:-$ENGINE_DIR/INCIDENTS.md}"
MAX_ITERATIONS="${WATCHDOG_MAX_ITERATIONS:-0}"
MISSED_SESSION_TOKEN="MISSED_SESSION"
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
# Live pids for a trader pattern.  A ZOMBIE IS NOT A LIVE TRADER: a killed
# process the host has not reaped still shows up in a pgrep scan, which is
# how a dead MAIN trader kept its closed-market stale-log exemption instead of
# being restarted (2026-09-23).  Reaping is the host's job; detecting the
# defunct process is ours.
trader_pids() {
    local pattern="$1" pid stat
    if [[ -n "${WATCHDOG_PIDS_CMD:-}" ]]; then
        bash "$WATCHDOG_PIDS_CMD" "$pattern" 2>/dev/null || true
        return 0
    fi
    for pid in $(pgrep -f "$pattern" 2>/dev/null); do
        stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')"
        case "$stat" in
            ""|*Z*) continue ;;   # gone, or a zombie awaiting reaping
        esac
        printf '%s\n' "$pid"
    done
}
trader_process_running() {
    [[ -n "$(trader_pids "$1")" ]]
}
# An unmistakable incident line: "MISSED SESSION" in the watchdog log AND a
# one-line note in the repo, so a whole missed session can never again be
# filed as a routine exemption.
report_missed_session() {
    local kind="$1" age="$2" log_file="$3" detail="$4" line
    line="MISSED SESSION: $kind trader produced no log line for ${age}s (~$(( age / 3600 ))h) spanning a regular session ($detail); newest log $log_file"
    log_action "$line"
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$line" >> "$INCIDENT_LOG"
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
        # trader_pids() (not raw pgrep) so the kill skips zombies and honours
        # the WATCHDOG_PIDS_CMD test hook — a drill must never be able to kill
        # the real stack.
        trader_pids '[l]ive_trader.py' | while read -r pid; do
            kill "$pid" 2>/dev/null || true
            log_action "killed stalled live trader pid=$pid"
        done
        log_action "starting live trader"
        out="$(bash "$ENGINE_DIR/start_trader.sh" 2>&1)"
        printf '%s\n' "$out" >> "$WATCHDOG_LOG"
        new_pid="$(printf '%s\n' "$out" | sed -n 's/.*Started! PID: \([0-9][0-9]*\).*/\1/p' | head -n 1)"
    else
        trader_pids '[t]urbo_trader.py' | while read -r pid; do
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
    local pattern kind="$1" log_file age gap output action reason
    if [[ "$kind" == "live" ]]; then
        pattern='[l]ive_trader.py'
    else
        pattern='[t]urbo_trader.py'
    fi
    # 1. PROCESS LIVENESS COMES FIRST AND IS NEVER EXEMPTED.  The
    #    closed-market exemption below applies to a stale LOG only: a trader
    #    that is not running (dead, or a zombie the host never reaped) is
    #    restarted whatever the clock says.
    if ! trader_process_running "$pattern"; then
        log_action "$kind trader is not running (market hours are irrelevant: a dead trader is always restarted)"
        restart_trader "$kind" "process-not-running"
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
        # 2. GAP SCAN FIRST: a gap long enough to have swallowed a regular
        #    session is an INCIDENT.  This runs before (and independently of)
        #    the stale-log verdict, so a closed-market exemption can never
        #    again bury a missed session in a quiet one-liner.
        gap="$( "${WATCHDOG_POLICY_CMD[@]}" --gap-age="$age" "${NOW_ARG[@]}" 2>/dev/null )"
        if [[ "$gap" == "$MISSED_SESSION_TOKEN"* ]]; then
            report_missed_session "$kind" "$age" "$log_file" "${gap#*|}"
        fi
        # 3. Ask the Python policy for the stale-log verdict. It applies the
        #    market-hours exemption: a stale log is NOT a reason to restart
        #    while the market is closed (the traders legitimately log nothing
        #    pre-open). A dead process is handled above and always restarted.
        output="$(
            "${WATCHDOG_POLICY_CMD[@]}" \
                --process-running=yes --has-log=yes --age="$age" "${NOW_ARG[@]}"
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
            log_action "$kind trader FROZEN/alive-but-silent: no output for ${age}s during market hours (${log_file}); restarting ($reason)"
            restart_trader "$kind" "log-stale"
        else
            # 4. An exemption must never hide a process that died while the
            #    verdict was being computed: re-verify liveness before
            #    accepting it.
            if ! trader_process_running "$pattern"; then
                log_action "$kind trader died while the stale-log verdict was being evaluated — restarting (market hours are irrelevant)"
                restart_trader "$kind" "process-not-running-after-exemption"
            else
                log_action "$kind trader log is stale (${age}s) but exempted: $reason"
            fi
        fi
    fi
}
log_action "watchdog started (interval=${CHECK_INTERVAL}s, stale=${STALE_SECONDS}s, skip-stale-when-closed=${WATCHDOG_SKIP_STALE_WHEN_CLOSED})"
# Hourly healthy tick: the watchdog logs nothing while everything is fine,
# so an hour-long silence is indistinguishable from a dead watchdog
# (2026-09-14).  One line per hour proves liveness — silence is never
# ambiguous while the process is up.
LAST_HEALTHY_LOG="$(date +%s)"
ITERATIONS=0
while true; do
    check_trader live
    check_trader turbo
    now_ts="$(date +%s)"
    if (( now_ts - LAST_HEALTHY_LOG >= 3600 )); then
        live_state="yes"; trader_process_running '[l]ive_trader.py' || live_state="no"
        turbo_state="yes"; trader_process_running '[t]urbo_trader.py' || turbo_state="no"
        log_action "watchdog healthy tick (live=${live_state}, turbo=${turbo_state})"
        if [[ "$live_state" == "no" || "$turbo_state" == "no" ]]; then
            log_action "WARNING: healthy tick reports a trader that is NOT running (live=${live_state}, turbo=${turbo_state}) — check_trader restarts it in this same iteration"
        fi
        LAST_HEALTHY_LOG="$now_ts"
    fi
    ITERATIONS=$(( ITERATIONS + 1 ))
    if (( MAX_ITERATIONS > 0 )) && (( ITERATIONS >= MAX_ITERATIONS )); then
        log_action "watchdog leaving the loop after ${ITERATIONS} iteration(s) (WATCHDOG_MAX_ITERATIONS=$MAX_ITERATIONS)"
        break
    fi
    sleep "$CHECK_INTERVAL"
done
