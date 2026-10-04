# Round-2 four must-fix defects: the pre-fix engine, committed as an artefact

Captured against **commit `90f0c53`** (branch `feature/strategy-search`, the tree
before the four fixes) in a scratch worktree with the new test file
(`tests/test_strategy_search_round2_fourfixes.py`) copied in and nothing else
changed.  Raw pytest output: `docs/ROUND2_FOURFIXES_PREFIX_PYTEST.txt`
(**16 failed** — every test in the file, each for the reason its docstring
names).  Reproduce with:

```
git worktree add /home/team/shared/wt-prefix 90f0c53
cp tests/test_strategy_search_round2_fourfixes.py /home/team/shared/wt-prefix/tests/
cd /home/team/shared/wt-prefix
export WATCHDOG_SCRIPT="$PWD/watchdog.sh"
export SUPERVISE_SCRIPT="$PWD/scripts/supervise_traders.sh"
nice -n 19 /home/team/shared/engine/.venv/bin/python -m pytest \
    tests/test_strategy_search_round2_fourfixes.py -q
```

| tree | file | result |
|---|---|---|
| `90f0c53` (before) | `tests/test_strategy_search_round2_fourfixes.py` | **16 failed** (log: `docs/ROUND2_FOURFIXES_PREFIX_PYTEST.txt`) |
| fixed tree (after) | same file | **16 passed** |

Containment line from both runs:
`0 refusal(s), 0 stray(s) reaped, 0 live-stack process(es) on the box at session end`.

## What each failure shows (short test summary, verbatim)

**A1 — exit slippage was charged on the exit bar's *open*, whatever price the
fill was made at** (`engine.py` `_close` / `_exit_fills`)

```
FAILED ...::test_an_eod_flatten_prices_exit_slippage_at_the_close_it_filled_at
  - AssertionError: exit slippage must be charged on the price the exit fill was
    made at (the 120.0 close), not on the exit bar's 110.0 open
    assert np.float64(20.995800839832036) == 21.995600879999998 ± 2.2e-05
FAILED ...::test_a_time_exit_prices_exit_slippage_at_the_close_it_filled_at
  - AssertionError: the time exit fills at the 118.0 close, not the 108.0 open
FAILED ...::test_the_zero_cost_reconstruction_matches_a_real_replay_on_a_non_flat_tape
  - AssertionError: ['reconstruction 9997.000599880026 != zero-cost net 10000.0
    (difference 2.9994001199738705)']
```

**A2 — the zero-cost column added the commissions a second time** (`engine.py`
`_stats`: `pnl_zero_cost_same_fills = Σ(pnl_gross + cost_drag)`)

```
FAILED ...::test_a_fee_charging_flat_pair_books_no_fabricated_zero_cost_profit[pessimistic]
  - AssertionError: a flat tape's zero-cost P&L is exactly zero; the pre-fix
    column added the $5.00 of commissions a second time
FAILED ...[pessimistic_x1_5]  - ... the $7.50 of commissions a second time
FAILED ...[pessimistic_x2]    - ... the $10.00 of commissions a second time
FAILED ...::test_the_cost_columns_split_slippage_from_fees[pessimistic
      | pessimistic_x1_5 | pessimistic_x2]
  - AssertionError: the trade record must carry the split columns, not one merged
    drag; Extra items in the left set: 'slip_drag', 'pnl_gross_precost'
```

**A3 — the recorded-run-set assertion was W1/baseline/W1-only, so a W1 survivor
raised after its replays and before any verdict was written**
(`run_strategy_search.py` `assert_grid_recorded`)

```
FAILED ...::test_a_family_with_a_w1_survivor_reaches_a_written_verdict
  - RuntimeError: family A screen: recorded run set does not equal the declared
    set (missing [], unexpected ['A|a_trend_trail50_noreg|W1|zero_cost|fixed_notional',
    'A|a_trend_trail50_noreg|W2|baseline|fixed_notional',
    'A|a_trend_trail50_noreg|W2|zero_cost|fixed_notional'])
FAILED ...::test_the_declared_run_set_equals_what_the_screen_records
  - AttributeError: module 'run_strategy_search' has no attribute
    'declared_screen_run_set'
```

**A4 — the hash was taken before `dataclasses.replace(cfg, sizing=sizing)`, and
the cross-run guard was keyed *with* the window so W1 and W2 were never compared**

```
FAILED ...::test_the_two_sizing_modes_of_one_cell_hash_differently
  - AssertionError: one cell's two sizing modes are two different configs
    assert '1df015d31733b924' != '1df015d31733b924'
FAILED ...::test_a_w1_w2_config_mismatch_is_refused_loudly
  - Failed: DID NOT RAISE RuntimeError
FAILED ...::test_the_same_cell_resolves_alike_on_window_1_and_window_2
  - KeyError: 'identity_key'
```

**A5 — run-record integrity: `engine_sha()` swallowed every exception and
returned the string `"unknown"`, records were replaced by an unconditional
`write_text`, and the filename carried neither hash**

```
FAILED ...::test_the_engine_sha_is_a_real_commit_or_a_loud_failure
  - Failed: DID NOT RAISE RuntimeError
FAILED ...::test_a_run_record_carries_both_hashes_and_is_never_silently_replaced
  - Failed: DID NOT RAISE RuntimeError
```

**Honest reading.**  Unlike the round-2 gate's prefix log (35 failures, most of
them `AttributeError`/`KeyError` for APIs that did not exist), these 16 failures
are all *behavioural*: the assertions named a number or a raise and the pre-fix
engine returned the wrong one or stayed silent.  That is the stronger form of
failing-first evidence, and it is why this file is the proof-of-fix for the four
must-fix defects rather than a restatement of their prose.
