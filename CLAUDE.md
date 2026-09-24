# Working notes for agents in this repo (AlgoFlow engine)

## The live stack is a loaded gun, and it lives in this tree

`live_trader.py`, `turbo_trader.py`, `watchdog.sh`, `start_trader.sh`,
`start_turbo.sh` and `scripts/supervise_traders.sh` run from
`/home/team/shared/engine` in production. Read
[`docs/SAFE_TEST_RUN.md`](docs/SAFE_TEST_RUN.md) **before** running the test
suite, and before touching any of those files:

* Run the suite from the tree under test, with
  `WATCHDOG_SCRIPT="$PWD/watchdog.sh"` and
  `SUPERVISE_SCRIPT="$PWD/scripts/supervise_traders.sh"` exported, and
  `pgrep -af 'watchdog.sh|live_trader.py|turbo_trader.py'` before and after —
  the "after" must be empty.
* Never start, restart or re-launch a trader, watchdog or the cron supervisor
  without the lead's explicit go-ahead. The trading stack is deliberately
  paused.
* Never `pkill` by pattern. Kill by pid.
* A test run must be *incapable* of touching the live stack:
  `tests/containment.py` + `tests/conftest.py` enforce that (see the doc).
  If you add a test that spawns a process, it inherits the guard; if you add a
  live-stack script, add the `ALGOFLOW_TEST_SESSION` refusal it needs —
  `tests/test_suite_containment.py` checks every script for it.

## Conventions

* Python 3.12, pandas/numpy; tests with `pytest` (`tests/`, one file per
  behaviour, docstrings that name the incident a test pins).
* The repo has no `pytest.ini`; run `.venv/bin/python -m pytest tests -q`
  from the repo root (the root is what makes `import live_trader` and
  `from src... import ...` resolve).
* `tests/` is a package (`__init__.py`): test modules import helpers as
  `from tests.<module> import ...`.
* Backtest artifacts live in `data/backtest_out/`; owner-facing reports go to
  `/home/team/shared/`.
* Workflow (branches, PRs, review) is in `/home/team/shared/WORKFLOW.md`.
