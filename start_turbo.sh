#!/bin/bash
# AlgoFlow TURBO Live Trader Launcher
# Start with: bash start_turbo.sh
#
# Runs the aggressive leveraged-ETF trading mode on the same Alpaca paper
# account as live_trader.py.  Both can run simultaneously — separate log
# files, separate PIDs, no shared state.

cd /home/team/shared/engine
source .venv/bin/activate
[ -f .env.turbo ] && export $(grep -v '^#' .env.turbo | xargs)

echo "🚀 AlgoFlow TURBO Live Trader"
echo "Symbols: SOXL, TQQQ, FNGU, SPXL (3x Leveraged ETFs)"
echo "Strategy: Mean Reversion + Momentum (dual)"
echo "Stop-Loss: 6% | Take-Profit: 8%"
echo "⚠️  Mandatory EOD liquidation 5 min before close"
echo "Log: logs/turbo_$(date +%Y%m%d).log"
echo ""

nohup python3 turbo_trader.py > logs/turbo_runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &
PID=$!
echo "Started! PID: $PID"
echo "Monitor: tail -f /home/team/shared/engine/logs/turbo_$(date +%Y%m%d).log"
echo "Stop:    kill $PID"
