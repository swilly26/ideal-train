#!/usr/bin/env bash
# Keep the AlgoFlow traders alive and recover them from stalled data/network calls.
set -u

ENGINE_DIR="/home/team/shared/engine"
LOG_DIR="$ENGINE_DIR/logs"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
STALE_SECONDS=300
CHECK_INTERVAL=60

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
    local kind="$1" pid
    if [[ "$kind" == "live" ]]; then
        pgrep -f '[l]ive_trader.py' | while read -r pid; do
            kill "$pid" 2>/dev/null || true
            log_action "killed stalled live trader pid=$pid"
        done
        log_action "starting live trader"
        bash "$ENGINE_DIR/start_trader.sh" >> "$WATCHDOG_LOG" 2>&1
    else
        pgrep -f '[t]urbo_trader.py' | while read -r pid; do
            kill "$pid" 2>/dev/null || true
            log_action "killed stalled turbo trader pid=$pid"
        done
        log_action "starting turbo trader"
        bash "$ENGINE_DIR/start_turbo.sh" >> "$WATCHDOG_LOG" 2>&1
    fi
}

check_trader() {
    local kind="$1" log_file age
    if [[ "$kind" == "live" ]]; then
        if ! pgrep -f '[l]ive_trader.py' >/dev/null; then
            log_action "live trader is not running"
            restart_trader live
            return
        fi
    elif ! pgrep -f '[t]urbo_trader.py' >/dev/null; then
        log_action "turbo trader is not running"
        restart_trader turbo
        return
    fi

    log_file="$(latest_log "$kind")"
    if [[ -z "$log_file" ]]; then
        log_action "$kind trader has no log file; restarting"
        restart_trader "$kind"
        return
    fi
    age=$(( $(date +%s) - $(stat -c %Y "$log_file") ))
    if (( age > STALE_SECONDS )); then
        log_action "$kind trader log is stale (${age}s): $log_file; restarting"
        restart_trader "$kind"
    fi
}

log_action "watchdog started (interval=${CHECK_INTERVAL}s, stale=${STALE_SECONDS}s)"
while true; do
    check_trader live
    check_trader turbo
    sleep "$CHECK_INTERVAL"
done
