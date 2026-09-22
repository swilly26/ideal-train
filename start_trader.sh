#!/bin/bash
# AlgoFlow Live Trader Launcher
# Start with: bash start_trader.sh

cd /home/team/shared/engine
source .venv/bin/activate
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "Starting AlgoFlow Live Trader..."
echo "Symbols: NVDA, META, QQQ, TSLA, COIN, AVGO"
echo "Strategy: Mean Reversion (z-score)"
echo "Log: logs/trades_$(date +%Y%m%d).log"
echo ""

# setsid: trader runs in its OWN session so an external teardown of the
# invoking session (interactive terminal or watchdog.sh) cannot take it down.
# nohup: SIGHUP immune.  stdin from /dev/null: no terminal dependency.
export PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED: stdout is a FILE here, so Python block-buffers it and the
# runner .out stays silent for hours while the trader is alive and working
# (live 2026-09-21/22: last line 20:16Z, process still submitting orders at
# 03:00Z).  Unbuffered stdout keeps runner logs usable for monitoring.
setsid nohup python3 -u live_trader.py < /dev/null > logs/runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &
PID=$!
echo "Started! PID: $PID"
echo "Monitor: tail -f /home/team/shared/engine/logs/trades_$(date +%Y%m%d).log"
echo "Stop: kill $PID"
