# The builder audit's missing assertions: evidence

Tree: **`3bcc475`** (round-2 caps already re-pinned, stage 1 pushed). Files:
`tests/test_strategy_search_round2_audit_assertions.py`; raw run
`docs/ROUND2_AUDIT_ASSERTIONS_PREFIX_PYTEST.txt`; mutation battery
`docs/ROUND2_AUDIT_ASSERTIONS_MUTATIONS.txt` (driver:
`/home/team/shared/mutate_battery.py`, run in a scratch worktree
`/home/team/shared/wt-mut` at `3bcc475`, one mutation at a time, each reverted).

## Honest reading first

**All six tests pass on the tree they are added to — there is no pre-fix
failure.** These are coverage additions for behaviour the builders already
implement, so "fails against the old tree" is not available evidence for them.
What is available, and what the audit actually demanded (§2.4: *"the test that
claims to test it is vacuous"* — i.e. it **cannot fail**), is a mutation battery:
each test is run against the single mutation it exists to catch.

| # | mutation | test | result |
|---|---|---|---|
| M1 | OR band `transform("max")` → `cummax()` (the expanding form) | OR ready gate | **FAILS — bites** |
| M2 | `G_OR_MINUTES = 60` | G gate/window invariant | **FAILS — bites** |
| M3 | `H_TIME_STOP_MINUTES = 600` | time stops by value | **FAILS — bites** |
| M4 | `H_DEV_EXIT_ATR = 0.50` | time stops by value | **FAILS — bites** |
| M5 | `E_EXIT_LEVELS` `t60` → `t10` | time stops by value | **FAILS — bites** |
| M6 | `G_STOP_LEVELS` `{15,…}` → `{10,30,60}` | time stops by value | **FAILS — bites** |
| M7 | `_r2_params` sizing → `equity_fraction` | sizing literals | **FAILS — bites** |
| M8 | `position_size_pct = 0.40` | sizing literals | **PASSES — mutation missed its target** (below) |
| M9 | H's entry window opens 09:31 | `rvol30` warm-up | **FAILS — bites** |
| M10 | G's both-band tie-break short → long | tie-break | **FAILS — bites** |

Representative failure texts (verbatim, from the raw log):

```
M2  E  AssertionError: G's OR must close exactly when its entry window opens
M3  E  assert 600 == 60
M4  E  assert 0.5 == 0.25   +  where 0.5 = families.H_DEV_EXIT_ATR
M5  E  At index 1 diff: ('t10', 10) != ('t60', 60)
M6  E  At index 0 diff: 10 != 15
M7  E  AssertionError: e1_cont_g0.5_eod   assert 'equity_fraction' == 'fixed_notional'
M9  E  AssertionError: rvol30's rolling window is 30 one-minute returns
M10 E  a both-band breach is a short, by rule
```

**M8 is a defect in my battery, not in the test.** The patch string
`position_size_pct=0.50, bp_usage_pct=0.95,` occurs first in `_base_params`
(the round-1 A–D base), and the driver replaced the first occurrence only — so
the round-2 cells kept `0.50` and the assertion correctly passed. The round-2
literal is genuinely pinned: M7, which does target `_r2_params`, fails the same
test. No re-run was possible in this session's budget; a follow-up should
re-point M8 at `_r2_params` and re-run.

## The OR ready gate (R3-11.3) — what the test can and cannot catch

`test_g_entry_dir_is_zero_on_every_bar_before_the_or_window_closes` uses a
fixture that **places its own spikes**: real breaches at 09:45 (high 100.10) and
09:59 (100.20), each clearing the hardest declared threshold (1.0 × ATR = 0.04)
against the band as it would be known at that bar — the exact thing the old
`or_day(15, True)` never did. The same day **does** trade, at 10:00, counter to
the breach, so the zeros are not an absence of signal.

M1 (the expanding OR band) fails the test — it bites on a change of the declared
whole-window form, which is precisely the decision R3-11.3 made
(`test_strategy_search_engine.py:94–100` pins that form).

**Residual finding, stated plainly:** on the shipped design the `ready` gate is
**over-determined** — it cannot be isolated by any single-line mutation. With
`or30_hi` a whole-window maximum, a bar inside the window can never breach its
own band (its high is ≤ the window max), so `ready` is a no-op *given* the form;
and G's entry window already opens at 10:00, the same bar the OR closes, so
`ready` and the window are redundant with each other (audit §2.4). The invariant
that carries weight is therefore `G_OR_MINUTES == window_open − RTH_open`, which
M2 does catch, and it is asserted by
`test_g_or_minutes_is_exactly_the_entry_window_open`. The new OR test documents
and pins the decision; it does not police `ready`.

## What was landed

* `test_g_entry_dir_is_zero_on_every_bar_before_the_or_window_closes` +
  `test_g_or_minutes_is_exactly_the_entry_window_open` (R3-11.3).
* `test_the_family_time_stop_literals_are_asserted_by_value` — `E_EXIT_LEVELS`,
  `G_STOP_LEVELS`, `H_TIME_STOP_MINUTES = 60`, `H_DEV_EXIT_ATR = 0.25`, the four
  level tuples, per-cell time stops, and the cell *name* vs its own value.
* `test_every_round2_cell_pins_the_sizing_literals` — `sizing`,
  `position_size_pct = 0.50`, `bp_usage_pct = 0.95`, per cell and through
  `build()`.
* `test_h_regime_gate_is_decidable_exactly_when_its_window_opens` — `rvol30` is
  NaN for the first 30 bars and first finite exactly at H's 10:00 window open.
* `test_g_both_band_breach_takes_the_short_side` — the tie-break, plus a
  lower-band-only breach still entering long.

Whole-file run on the shipped tree: **6 passed**, containment
`0 refusal(s), 0 stray(s) reaped, 0 live-stack process(es)`.

## Not landed (and why)

* Nothing from either list is missing. M8's re-run is the one open item.
* The audit's remaining items are other people's board items and were not
  touched: §6.2 (the degenerate-axis / clone check in the census) and §B4
  ("every cell on both windows").
