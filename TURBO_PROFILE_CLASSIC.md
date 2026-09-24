# `TURBO_PROFILE=classic` — frozen pre-2026-08-11 turbo profile

Purpose: keep the **pre-2026-08-11** turbo configuration runnable, explicitly named, and
switchable with one environment variable, so it can be compared against the current
profile on paper. It is **not** a recommendation and **not** deployed.

```bash
# current profile (default, unchanged behaviour)
python turbo_trader.py
# frozen classic profile
TURBO_PROFILE=classic python turbo_trader.py
```

Do not start any trader on the strength of this file alone: the trading stack is down by
standing instruction and restarts are the lead's call, outside market hours, with
broker-side stop verification before and after.

## What `classic` changes (and what it does not)

| Knob | current | classic | source |
|---|---|---|---|
| symbol pool | base 4 + 7 VIOLENCE (`ENABLE_VIOLENCE_TIER`) | **SOXL, TQQQ, FNGU, SPXL** | `turbo_trader.py:70` (`_effective_symbols()` :84) |
| `ENABLE_REGIME_GATE` | True | **False** | `:135` |
| `ENABLE_SHORT_SELLING` | True | **False** (long-only) | `:141` |
| `ENABLE_MEAN_REVERSION_SHORT` | True | **False** | `:147` |
| momentum SELL confidence | `0.3 + 0.7*(0.5*dist + 0.3*rsi) + cross_bonus` | **legacy `dist_pct*10`** | `_generate_momentum_signals`, `LEGACY_SELL_CONFIDENCE` |
| base stop / target | 6% / 8% | 6% / 8% (same) | `STRATEGY_CONFIG` |
| MAX_POSITIONS / size | 2 / 50% | 2 / 50% (same) | `:80-83` |
| MAX_HOLD / EOD flatten | 30 min / 30 min | 30 min / 30 min (same) | `:83-84` |
| trend extension (30→60 min) | violence tier only | inert (no violence tier) | `_trend_extension_qualifies` |

**Never changed by the profile switch** (they are safety/accounting fixes, not strategy):
buying-power cap `BUYING_POWER_USAGE_PCT=0.95` (#29), `SHORT_WHOLE_SHARES_ONLY`,
`SHORT_REQUIRE_BROKER_SHORTABLE`, rejection backoff (#29), and verified-fill close
accounting `close_position_verified` (#30). `tests/test_turbo_classic_profile.py` asserts this.

## Honest caveats (read before interpreting any comparison)

1. **The classic profile's historical return was beta, not skill.** Broker fills for
   2026-07-30 → 2026-08-15: +$18,756, of which the pooled move of the same four ETFs over
   the same holding windows explains 99.8% (residual +$40 over 112 trades); regular-session
   trading lost $1,612 while +$24,107 of the whole window's P&L was earned in overnight gaps.
   See `/home/team/shared/TURBO_FORENSICS_20260924.md`.
2. **The two largest winners are not reproducible by configuration**: FNGU +$10,697
   (48% of the entire window) and SOXL +$7,432 were positions left open by a *dead process*
   and sold by restart-cleanup into the next opening gap. `classic` restores the entry and
   exit *policy*; it does not and should not restore that failure mode.
3. **Exits are not identical.** The legacy SELL confidence is restored, but every close
   still routes through the #30 verified-fill path, so booked P&L uses the broker's fill
   price rather than the last mark (log said +$8,634, broker paid +$10,697 on that FNGU
   trade). Expect classic-profile P&L to differ slightly from the July/August logs.
4. A comparison of `classic` vs `current` over one future window mostly measures that
   window's tape. Only a fold-validated replay (positive out-of-sample across more than one
   walk-forward fold) is evidence.
