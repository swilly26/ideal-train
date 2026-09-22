# ScalpSet 12-month backtest — run status (written 2026-09-20 ~20:55 UTC)

**The full-year portfolio runs were still executing when the build session's budget ran out.**
This file records exactly what is running, what has been verified, and how to finish the report.

## What is running right now (background, shared machine)

| Process | Command | Started | Log |
|---|---|---|---|
| baseline | `scripts/run_scalpset_backtest.py --variant baseline --start 2025-09-01 --end 2026-09-01 --out-dir data/backtest_out` | 20:44 UTC | `engine/logs/bt_baseline.log` |
| pessimistic | same with `--variant pessimistic` | 20:44 UTC | `engine/logs/bt_pessimistic.log` |
| chain | waits for the two above, then runs `--variant overnight` and `--variant zero_cost` | 20:46 UTC | `engine/logs/bt_chain.log`, marker `engine/logs/bt_chain.done` |
| watcher | after the chain: standalone January replay (`--tag indep_2026-01`, slice-equivalence check) then builds the report | 20:46 UTC | `engine/logs/watch_and_report.log`, marker `engine/logs/report_built.done` |

The watcher writes the finished owner report to
**`/home/team/shared/SCALPSET_BACKTEST_REPORT.md`** and prints `report exit=0` in
`engine/logs/watch_and_report.log`. Nothing needs to be re-run if it succeeds.

Each variant is one **portfolio** replay (all six symbols through one engine, so 15 %
sizing, the 6-position cap and best-R:R arbitration behave as live). Wall-clock is
~10–40 min per variant on this 2-core machine; two run at a time.

## Verified before this session's end

* The engine itself is the committed replay (`src/backtesting/scalpset_engine.py`,
  PR-branch `feature/scalpset-historical-backtest`, 695 tests green) — no engine change
  was needed.
* `scripts/run_scalpset_backtest.py` was validated end-to-end on a short window
  (`--start 2026-08-01 --end 2026-08-05`, 6 symbols, pessimistic): 780 bars, 2 sessions,
  19 trades, all artifacts (trades/equity/folds/per-symbol/module/stats/summary) written.
* `scripts/build_scalpset_report.py` turns those artifacts into the owner report
  (no optional `tabulate` dependency — that module is not installed in the engine venv,
  which is why the report renderer is hand-rolled).

## To finish by hand (if the watcher did not)

```bash
cd /home/team/shared/engine
env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY .venv/bin/python \
    scripts/run_scalpset_backtest.py --variant overnight --start 2025-09-01 --end 2026-09-01 \
    --out-dir data/backtest_out
env -u ALPACA_API_KEY -u ALPACA_SECRET_KEY .venv/bin/python \
    scripts/build_scalpset_report.py --out /home/team/shared/SCALPSET_BACKTEST_REPORT.md
```

Raw artifacts (untracked, `data/` is gitignored): `engine/data/backtest_out/`
— `trades_<variant>.csv`, `equity_<variant>.csv`, `folds_monthly_<variant>.csv`,
`folds_quarterly_<variant>.csv`, `per_symbol_<variant>.csv`, `module_<variant>.csv`,
`symbol_quarter_<variant>.csv`, `stats_<variant>.json`, `summary_<variant>.md`.
