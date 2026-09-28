#!/bin/bash
# Round 1 of the turbo classic edge search: ONE change at a time, both cost models.
cd /tmp/wt-turbo-replay || exit 1
PY=/home/team/shared/engine/.venv/bin/python
OUT=/tmp/scratch/r1out
LOG=/tmp/scratch/r1
mkdir -p "$OUT" "$LOG"
run() {  # run <tag> <variant> <--set ...>
  local tag="$1"; shift
  local variant="$1"; shift
  if [ -f "$OUT/turbo_stats_${tag}.json" ]; then echo "skip $tag (exists)"; return; fi
  echo "=== $tag variant=$variant sets=$* ==="
  nice -n 19 "$PY" scripts/run_turbo_backtest.py --logic classic --variant "$variant" \
      --start 2025-09-01 --end 2026-09-01 --out-dir "$OUT" --tag "$tag" "$@" \
      > "$LOG/$(echo "$tag" | tr '/' '_').log" 2>&1
  echo "rc=$? $tag"
}
run classic_baseline            baseline
run classic_zero_cost           zero_cost
run classic_baseline__A_latest  baseline  --set window_tie=latest
run classic_zero_cost__A_latest zero_cost --set window_tie=latest
run classic_baseline__B_linear  baseline  --set mr_conf_mode=linear
run classic_zero_cost__B_linear zero_cost --set mr_conf_mode=linear
run classic_baseline__C_age5    baseline  --set max_signal_age_bars=5
run classic_zero_cost__C_age5   zero_cost --set max_signal_age_bars=5
run classic_baseline__D_cap2    baseline  --set max_entries_per_session=2
run classic_zero_cost__D_cap2   zero_cost --set max_entries_per_session=2
run classic_baseline__D_cap5    baseline  --set max_entries_per_session=5
run classic_zero_cost__D_cap5   zero_cost --set max_entries_per_session=5
run classic_baseline__E1_tp2    baseline  --set take_profit_pct=0.02
run classic_zero_cost__E1_tp2   zero_cost --set take_profit_pct=0.02
run classic_baseline__E2_hold10 baseline  --set max_hold_minutes=10
run classic_zero_cost__E2_hold10 zero_cost --set max_hold_minutes=10
echo "ROUND1 DONE"
