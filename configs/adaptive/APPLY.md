# Applying adaptive parameters to the live traders (APPLY.md)

**Status: not wired live.** The optimizer *outputs* configs and evidence; it
does not change live behavior. Adoption is a separate, human-reviewed step.

## What the optimizer produces (`configs/adaptive/`)

| File | Contents |
|---|---|
| `report_<trader>.md` | IS vs OOS rankings, degradation, grades, regime breakdown |
| `results_<trader>.json` | Full numeric output (rankings, baseline, regimes) |
| `<trader>_mean_reversion.json` | Best validated config (or a grade-F note) |
| `regime/<trader>_<regime>.json` | Per-regime parameter sets (experimental) |

The JSON configs use the same schema as `src.optimization.persistence`
(`save_config` / `load_config`), which both traders can consume directly.

## Regeneration

```bash
cd /home/team/shared/engine
.venv/bin/python optimize_adaptive.py --trader all --no-fetch   # offline; uses the price snapshot
.venv/bin/python optimize_adaptive.py --trader all              # fetch fresh OHLCV first
```

Price snapshot (real 5-min yfinance OHLCV, 7/24–8/30/2026):
`/home/team/shared/data/adaptive_ohlcv/` (manifest.json documents the fetch).

## Apply path (follow-up PR — not done this session)

The *conservative* adoption route keeps everything behind flags that default
to **off**, so live behavior is byte-identical until explicitly enabled:

1. **Static adoption (simplest).** In `live_trader.py` / `turbo_trader.py`:
   ```python
   from src.optimization.persistence import load_config
   ADAPTIVE_CONFIG_PATH = ""            # default "" → unchanged behavior
   if ADAPTIVE_CONFIG_PATH:
       STRATEGY_CONFIG = load_config(ADAPTIVE_CONFIG_PATH)  # replaces fixed thresholds
   ```
   Only do this if the report shows a grade **A or better** candidate AND the
   holdout window confirms it (same sign, meaningful trade count). The
   current report grades everything **F** — do not adopt yet.

2. **Regime-adaptive adoption (matches "regime-adaptive config" goal).**
   At each tick, classify the symbol's recent bars
   (`src.optimization.regime.classify_regime`) and pick the matching
   `regime/<trader>_<regime>.json` config; stop/TP can additionally be
   ATR-scaled via `scale_stop_tp(base_stop, base_tp, atr_pct, ref_atr_pct)`.
   Per-regime sets are derived from IS+OOS only and are **experimental** —
   run them in paper (or behind a fraction of size) on a fresh window before
   trusting them. The holdout was deliberately never used for rule fitting.

## Go/no-go rule of thumb

- **Adopt** only when: winning candidate grade ∈ {A, S}; OOS profit factor
  ≥ 1.2 with ≥ 8 OOS trades; holdout PF ≥ 1.0; and the edge is *not*
  concentrated in one symbol/day.
- **Do not adopt** when the optimizer says F or the data is too thin (this
  is exactly the current state: only ~a month of real history; OOS windows
  hold 2–3 realized trades per trader).

## How much more data to collect before trusting it

~4–8 more weeks of continuous realized fills (the watchdog now keeps traders
alive between sessions, so gaps should shrink), then re-run
`optimize_adaptive.py --trader all --no-fetch` with the same split fractions.
With ≥ 3 months of daily data the OOS + holdout windows can hold tens of
trades per window, which is the minimum the S/A grades were designed for.