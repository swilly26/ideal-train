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
setsid nohup python3 live_trader.py < /dev/null > logs/runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &
PID=$!
echo "Started! PID: $PID"
echo "Monitor: tail -f /home/team/shared/engine/logs/trades_$(date +%Y%m%d).log"
echo "Stop: kill $PID"
