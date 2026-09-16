# AlgoFlow — AI-Driven Algorithmic Trading Engine

AlgoFlow combines automated trade execution with a deep AI strategy engine that continuously analyzes market conditions, backtests adjustments in real time, and dynamically optimizes each user's active strategies for maximum profitability.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   src/data   │───▶│ src/strateg- │───▶│ src/optimiz- │───▶│ src/execut-  │
│  (market     │    │  ies         │    │  ation       │    │  ion         │
│   data)      │    │  (signals)   │    │  (AI tuning) │    │  (broker)    │
└──────────────┘    └──────┬───────┘    └──────┬───────┘    └──────────────┘
                           │                  │
                           ▼                  ▼
                    ┌──────────────────────────────┐
                    │     src/backtesting          │
                    │  (event-driven simulator)    │
                    └──────────────────────────────┘
```

**Data flow:** Market data providers feed OHLCV bars into strategies. Strategies generate signals (BUY/SELL/HOLD). The optimisation layer tunes strategy parameters by repeatedly running the backtesting engine over historical data. When profitable configs are found, the execution layer sends orders to a broker.

## Setup

```bash
cd engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Module Summary

| Module | Responsibility |
|--------|----------------|
| `src/data` | Abstract market data provider interface. Implementations fetch bars and stream live data. |
| `src/strategies` | Strategy base class + registry. Each strategy consumes data and emits signals. |
| `src/optimization` | AI/ML parameter search. Tunes `StrategyConfig` fields to maximise an objective (Sharpe, Sortino, etc.). |
| `src/execution` | Broker abstraction. Sends orders, manages positions. |
| `src/backtesting` | Event-driven backtester. Replays data through strategies and computes P&L. |

## Live trader — stop anchoring & execution guards (post-go-live, 2026-09-16)

`live_trader.py` runs the ScalpSet main trader. A MARKET fill can land far from
the signal-time reference (live COIN case: signal `entry=167.96 SL=167.01`,
filled at `162.58`), and the broker validates stop levels against the *live*
market price — so a strategy SL sitting above the fill is rejected with
`42210000 "stop price must be less than current price"`, leaving the position
unprotected.

**Anchor rule** (`anchor_scalp_levels`, applied by `_finalize_open_bundle` after
the fill is confirmed):

1. `min_dist = max(fill * SCALP_STOP_MIN_DISTANCE_PCT, SCALP_STOP_MIN_DISTANCE_ABS)`.
2. A strategy level still on the correct side of the fill and at least
   `min_dist` away is kept verbatim (long SL `<= fill - min_dist`, TP
   `>= fill + min_dist`; mirrored for shorts).
3. Otherwise it is re-priced from the fill using the signal's own risk/reward
   distance (`risk = |entry_ref - sl_ref|`; long `fill - risk`, short
   `fill + risk`; TP mirrored with the reward), widened to `min_dist` when that
   distance is below the floor.
4. If that level is degenerate (`<= 0`, or still on the wrong side of the fill)
   the existing 6% `PROTECTIVE_STOP_PCT` backstop is used and logged loudly
   (`"strategy SL invalid vs fill — using backstop"`); a TP that cannot be
   derived is dropped (`None`) and BE/trail take over.
5. Levels are tick-normalised: `>= $1` keeps 2 decimals (sub-penny increments
   like `211.160004` are rejected outright by Alpaca), sub-dollar keeps 4.
6. The SAME anchored levels are stored in the position state, so the
   in-process stop loss evaluates the strategy stop against the FILL.

**Execution guards**

- An `INVALID-LEVEL` rejection (`42210000` + `stop price must be less|greater
  than current price`) is permanent for that price: it is logged once and the
  retry loop fast-fails instead of hammering the broker. Every other rejection
  class keeps the 4-attempt backoff.
- A position with **no broker stop** logs a loud warning every
  `SCALP_NO_STOP_WARN_SECONDS` (default 60s) — it never passes silently.
- Re-entry churn is capped per symbol per session; the cap resets at the next
  trading day and `<= 0` disables it.

**Env knobs**

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCALP_ANCHOR_LEVELS` | `true` | `false` = pre-fix behaviour (raw signal SL/TP) |
| `SCALP_STOP_MIN_DISTANCE_PCT` | `0.0005` | min anchored distance as a fraction of the fill (0.05%) |
| `SCALP_STOP_MIN_DISTANCE_ABS` | `0.01` | absolute floor for that distance (1 cent) |
| `SCALP_NO_STOP_WARN_SECONDS` | `60` | cadence of the "NO BROKER STOP" warning |
| `SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION` | `3` | ScalpSet entries per symbol per trading day |
| `MAIN_PROTECTIVE_STOP_PCT` | `0.06` | fallback backstop when no valid strategy SL exists |

Hermetic regression tests: `tests/test_main_stop_anchor.py` (in-memory broker,
deterministic bars — no network, no real account).

## Design Principles

- **Pluggable providers** — data sources, brokers, and strategies are all behind abstract base classes so the engine never couples to a specific vendor.
- **AI-tuneable configs** — every strategy exposes a `StrategyConfig` dataclass whose fields (entry/exit thresholds, position sizing, stop-loss levels, strategy weights) are designed to be mutated by the optimisation layer.
- **Backtest-first** — no strategy or optimiser change is considered real until it passes a backtest.
- **Lean dependencies** — pandas, numpy, scikit-learn. No heavy frameworks.
