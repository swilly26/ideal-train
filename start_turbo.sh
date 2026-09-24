#!/bin/bash
# AlgoFlow TURBO Live Trader Launcher
# Start with: bash start_turbo.sh
#
# Runs the aggressive leveraged-ETF trading mode on the same Alpaca paper
# account as live_trader.py.  Both can run simultaneously — separate log
# files, separate PIDs, no shared state.

# Engine root.  ALGOFLOW_ENGINE_DIR redirects the launch (cwd, logs, .env,
# the trader process itself) -- the test suite uses it to run a throwaway
# copy instead of the live stack.  Production never sets it.
ENGINE_DIR="${ALGOFLOW_ENGINE_DIR:-/home/team/shared/engine}"
export ALGOFLOW_ENGINE_DIR="$ENGINE_DIR"

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
cd "$ENGINE_DIR" || exit 1
source .venv/bin/activate
[ -f .env.turbo ] && export $(grep -v '^#' .env.turbo | xargs)

echo "🚀 AlgoFlow TURBO Live Trader"
echo "Symbols: SOXL, TQQQ, FNGU, SPXL (3x Leveraged ETFs) + VIOLENCE tier (TNA, TZA, LABU, LABD, UVXY, NVDL, TSLR)"
echo "Strategy: Mean Reversion + Momentum (dual) | Shorts: $(grep -c 'ENABLE_SHORT_SELLING = True' turbo_trader.py 2>/dev/null || echo 1)"
echo "Stop-Loss: 6% base / 9% violence | Take-Profit: 8% base / 13% violence | Size: 50% base / 40% violence"
echo "⚠️  Mandatory EOD liquidation 5 min before close"
echo "Log: logs/turbo_$(date +%Y%m%d).log"
echo ""

# setsid: trader runs in its OWN session so an external teardown of the
# invoking session (interactive terminal or watchdog.sh) cannot take it down.
# nohup: SIGHUP immune.  stdin from /dev/null: no terminal dependency.
export PYTHONUNBUFFERED=1
# same block-buffering fix as start_trader.sh (see the comment there)
setsid nohup python3 -u turbo_trader.py < /dev/null > logs/turbo_runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &
PID=$!
echo "Started! PID: $PID"
echo "Monitor: tail -f $ENGINE_DIR/logs/turbo_$(date +%Y%m%d).log"
echo "Stop:    kill $PID"
