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
echo "Symbols: SOXL, TQQQ, FNGU, SPXL (3x Leveraged ETFs) + VIOLENCE tier (TNA, TZA, LABU, LABD, UVXY, NVDL, TSLR)"
echo "Strategy: Mean Reversion + Momentum (dual) | Shorts: $(grep -c 'ENABLE_SHORT_SELLING = True' turbo_trader.py 2>/dev/null || echo 1)"
echo "Stop-Loss: 6% base / 9% violence | Take-Profit: 8% base / 13% violence | Size: 50% base / 40% violence"
echo "⚠️  Mandatory EOD liquidation 5 min before close"
echo "Log: logs/turbo_$(date +%Y%m%d).log"
echo ""

nohup python3 turbo_trader.py > logs/turbo_runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &
PID=$!
echo "Started! PID: $PID"
echo "Monitor: tail -f /home/team/shared/engine/logs/turbo_$(date +%Y%m%d).log"
echo "Stop:    kill $PID"
