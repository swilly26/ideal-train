# Round-2 cap re-pin (R3-11.1): the pre-fix tree, committed as an artefact

Captured against **commit `4a6347d`** (branch `feature/strategy-search`, the tree
before the re-pin) in a scratch worktree with the new test file
(`tests/test_strategy_search_round2_caps.py`) copied in and nothing else changed.
Raw pytest output: `docs/ROUND2_CAPREPIN_PREFIX_PYTEST.txt` (**3 failed** — every
test in the file). Reproduce with:

```
git worktree add --detach /home/team/shared/wt-caprefix 4a6347d
cp tests/test_strategy_search_round2_caps.py /home/team/shared/wt-caprefix/tests/
cd /home/team/shared/wt-caprefix
export WATCHDOG_SCRIPT="$PWD/watchdog.sh"
export SUPERVISE_SCRIPT="$PWD/scripts/supervise_traders.sh"
nice -n 19 /home/team/shared/engine/.venv/bin/python -m pytest \
    tests/test_strategy_search_round2_caps.py -q
```

| tree | file | result |
|---|---|---|
| `4a6347d` (before) | `tests/test_strategy_search_round2_caps.py` | **3 failed** (log: `docs/ROUND2_CAPREPIN_PREFIX_PYTEST.txt`) |
| re-pinned tree (after) | same file | **3 passed** |

Containment line from both runs:
`0 refusal(s), 0 stray(s) reaped, 0 live-stack process(es) on the box at session end`.

## What the change is

`families.R2_CAPS` — `max_positions` 2 → **4**, `notional_usd` 50_000 →
**25_000**. Nothing else: no signal definition, no exit rule, no window, no cost
parameter. Gross exposure is unchanged at 100 % of the 100k equity
(4 × 25_000), and the per-fill cost is unchanged (the cost model is per share,
not per notional).

The defect: `R2_UNIVERSE` is also the instrument insertion order, and the cap is
applied in that order. `gap` is session-constant, so in family E every symbol
signals on the **same bar**; at `max_positions = 2` SOXL and TQQQ always took
both slots and SPXL/SPY could never trade — family E was a deterministic
two-symbol, two-correlated-3×-leveraged-ETF strategy by construction, and any
neighbour/robustness evidence would have been about a grid that was never the
declared grid (audit §6.1, ranked finding 1, BLOCKING).

## What each failure shows (short test summary, verbatim)

```
FAILED tests/test_strategy_search_round2_caps.py::test_all_four_universe_symbols_open_at_the_pinned_caps
  - AssertionError: four symbols signal on the same bar, so the pinned cap must
    open all four, in the declared universe order — got ['SOXL', 'TQQQ']
  assert ['SOXL', 'TQQQ'] == ['SOXL', 'TQQQ', 'SPXL', 'SPY']
    Right contains 2 more items, first extra item: 'SPXL'
FAILED ...::test_the_pinned_caps_are_asserted_by_value - assert 2 == 4
FAILED ...::test_the_four_symbol_book_never_exceeds_the_equity_ceiling - assert 2 == 4
  +  where 2 = len(res.trades)  # index 0 = SOXL, index 1 = TQQQ; SPXL/SPY absent
```

The first failure is the one that matters: on a fixture where all four symbols
signal on the same bar, the pre-fix tree opens **exactly two positions, SOXL and
TQQQ, and SPXL and SPY are absent** — the original defect, reproduced.

## One finding to hand to the lead (not fixed here, deliberately)

At 100 % deployment the engine funds longs from cash, not from equity: the
``_try_open`` gate is ``outlay > self.cash`` with ``outlay = qty·fill + fee``. At
the pinned caps four longs are exactly the 100k of cash, so with **zero**
commission (``CostModel.baseline()``) all four open — floating-point aside — but
with any per-share commission the **fourth long is refused** and booked under
``skipped["cash"]``. Measured on the same all-four-signal fixture:

| cost model | positions opened | skip |
|---|---|---|
| `baseline` (fees 0) | 4 | — |
| `pessimistic` ($0.005/share) | 3 | `cash: 1` |

This is **not introduced by the re-pin**: at the old caps (2 × 50k) exactly the
same boundary existed and was worse (with fees the *second* long is refused). It
matters for the round-2 protocol only in that a fee-charging **cost-stress** run
of an all-long session would hold three symbols instead of four — i.e. the stress
level would be measuring a slightly different book than the base level. Two
options the lead may want to rule on before the screen: (a) accept it and state
it wherever the pessimistic numbers are quoted, or (b) change the engine's entry
funding gate from cash to `equity · MAX_GROSS_LEVERAGE` (an engine change, out of
this task's scope, and a change to a gate the four must-fix fixes did not
touch). Nothing in the re-pin itself needed a second change to be coherent at
cap = 4, so per the brief's instruction this was reported rather than decided.
