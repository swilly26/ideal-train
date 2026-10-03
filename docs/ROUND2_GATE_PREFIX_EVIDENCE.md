# Round-2 gate: the pre-fix engine, committed as an artefact (not prose)

Captured against **commit `69d5d8e`** (the engine before the round-2 correctness
gate) in a scratch worktree at `/home/team/shared/wt-prefix`, with the gate's own
test file copied in and nothing else changed.  Reproduce with:

```
.venv/bin/python scripts/round2_gate_prefix_evidence.py /home/team/shared/wt-prefix
.venv/bin/python scripts/round2_gate_prefix_evidence.py <gated tree>
```

Raw pytest output of the pre-fix run: `docs/ROUND2_GATE_PREFIX_PYTEST.txt`
(**35 failed, 1 passed** — the one pass is the pre-existing 150-trip power floor).

## E1 — the same-symbol flat pair, pre-fix vs gated (the blocking finding)

Two legs on one flat tape, one round trip, `CostModel.baseline()`
(2 bps / 1¢ adverse per fill).  `gross+drag` is the strategy's own P&L
reconstructed without costs: on identical price paths it must be **exactly
zero**.  It was **+20.004** on the pre-fix engine — a fabricated profit equal to
the cancelled toll, and that is the zero-cost number the screen's fold-breadth
rule runs on.

```
=== PRE-GATE TREE (69d5d8e, the old engine) ===
  two symbols (control)      pnl_gross=  -20.0000 cost_drag=  +20.0000 gross+drag=   +0.0000 net=  -20.0000
  one symbol, equal weights  pnl_gross=   +0.0000 cost_drag=  +20.0040 gross+drag=  +20.0040 net=   +0.0000
  one symbol, unequal weights pnl_gross=   -6.6680 cost_drag=  +20.0040 gross+drag=  +13.3360 net=   -6.6680

=== GATED TREE (this branch) ===
  two symbols (control)      pnl_gross=  -20.0000 cost_drag=  +20.0000 gross+drag=   +0.0000 net=  -20.0000
  one symbol, equal weights  pnl_gross=  -20.0000 cost_drag=  +20.0000 gross+drag=   +0.0000 net=  -20.0000
  one symbol, unequal weights pnl_gross=  -20.0013 cost_drag=  +20.0013 gross+drag=   +0.0000 net=  -20.0013
```

The pre-fix committed symmetry fixture used **two distinct symbols** and passed
while the bug was live; the gate parameterises it over one symbol on both legs.

## Failures of the gate's own tests on the pre-fix engine (from the raw output)

```
FAILED ...::test_a_flat_pair_can_never_book_a_profit[one_symbol_equal] - KeyError: 'gross_notional'
FAILED ...::test_a_flat_pair_can_never_book_a_profit[one_symbol_unequal] - KeyError: 'gross_notional'
FAILED ...::test_a_flat_pair_can_never_book_a_profit[two_symbols_equal] - KeyError: 'gross_notional'
FAILED ...::test_a_flat_pair_can_never_book_a_profit[two_symbols_unequal] - KeyError: 'gross_notional'
FAILED ...::test_the_flat_pair_toll_is_reported_honestly[one_symbol_equal] - KeyError: 'gross_notional'
FAILED ...::test_the_flat_pair_toll_is_reported_honestly[one_symbol_unequal] - KeyError: 'gross_notional'
FAILED ...::test_the_flat_pair_toll_is_reported_honestly[two_symbols_equal] - KeyError: 'gross_notional'
FAILED ...::test_the_flat_pair_toll_is_reported_honestly[two_symbols_unequal] - KeyError: 'gross_notional'
FAILED ...::test_a_same_symbol_pair_mirrors_like_a_distinct_one[1] - toll 20.004 vs 20.000 (comparison failed)
FAILED ...::test_a_same_symbol_pair_mirrors_like_a_distinct_one[-1] - toll 19.996 vs 20.000 (comparison failed)
FAILED ...::test_every_trip_carries_the_accounting_identity - KeyError: 'identity_residual'
FAILED ...::test_the_zero_cost_run_reconciles_with_the_reconstruction - AttributeError: no attribute 'reconcile_zero_cost'
FAILED ...::test_the_builder_refuses_a_universe_too_small_for_its_legs - Failed: DID NOT RAISE ValueError
FAILED ...::test_a_spec_must_declare_every_behaviour_changing_param - KeyError: 'eod_flat_min'
FAILED ...::test_the_builder_refuses_an_unknown_param_key - AttributeError: no attribute 'SpecError'
FAILED ...::test_the_builder_refuses_an_unknown_extras_key - AttributeError: no attribute 'SpecError'
FAILED ...::test_the_resolved_spec_is_returned_and_names_the_config - ValueError: not enough values to unpack (expected 3, got 2)
FAILED ...::test_the_resolved_hash_is_window_independent - ValueError: not enough values to unpack (expected 3, got 2)
FAILED ...::test_the_recorded_run_set_must_equal_the_declared_set - AttributeError: no attribute 'run_key'
FAILED ...::test_a_pending_entry_is_discarded_not_filled_later - KeyError: 'stale_entry'
FAILED ...::test_a_missing_bar_at_the_flatten_time_is_fatal[single] - Failed: DID NOT RAISE RuntimeError
FAILED ...::test_a_missing_bar_at_the_flatten_time_is_fatal[pair] - Failed: DID NOT RAISE RuntimeError
FAILED ...::test_every_drop_is_counted_and_the_accounting_holds - KeyError: 'allow_short'
FAILED ...::test_an_unknown_feature_name_raises_instead_of_reading_as_nan - Failed: DID NOT RAISE ValueError
FAILED ...::test_an_atr_skip_is_counted_as_no_atr_not_min_qty - KeyError: 'no_atr'
FAILED ...::test_family_c_is_blind_to_the_future - ValueError: not enough values to unpack (expected 3, got 2)
FAILED ...::test_family_a_is_blind_to_the_future - ValueError: not enough values to unpack (expected 3, got 2)
FAILED ...::test_a_multi_leg_config_may_not_declare_a_stop - AttributeError: no attribute 'SpecError'
FAILED ...::test_a_time_exit_is_measured_in_minutes_not_bars - AssertionError: the first bar at or after 5 *minutes*, not the fifth bar
FAILED ...::test_eod_flat_min_must_be_a_declared_value - AttributeError: no attribute 'SpecError'
FAILED ...::test_a_zero_trip_month_fails_the_stability_rule - AttributeError: no attribute 'fold_stability'
FAILED ...::test_the_ranking_is_net_then_drawdown_then_trips_then_costs_then_stability - AttributeError: no attribute 'rank_row'
FAILED ...::test_the_neighbour_rule_applies_per_axis - AttributeError: no attribute 'neighbour_verdict'
FAILED ...::test_declared_axes_come_from_the_resolved_grid - AttributeError: no attribute 'declared_axes'
FAILED ...::test_neighbour_cells_exclude_the_baseline_and_dedupe_by_hash - AttributeError: no attribute 'neighbour_cells'
35 failed, 1 passed in 2.69s
```

**Honest reading of this artefact.**  A `KeyError`/`AttributeError` failure says
*the behaviour did not exist*, which is the expected shape of a failing-first
test for an API the gate introduces; the two failures that carry the *evidence*
rather than the absence of an API are the flat-pair float assertions
(`test_a_same_symbol_pair_mirrors_like_a_distinct_one`, toll 20.004 vs 20.000)
and `test_a_time_exit_is_measured_in_minutes_not_bars`.  The direct
pre-fix/post-fix profit is the E1 table above.
