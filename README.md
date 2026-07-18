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

## Design Principles

- **Pluggable providers** — data sources, brokers, and strategies are all behind abstract base classes so the engine never couples to a specific vendor.
- **AI-tuneable configs** — every strategy exposes a `StrategyConfig` dataclass whose fields (entry/exit thresholds, position sizing, stop-loss levels, strategy weights) are designed to be mutated by the optimisation layer.
- **Backtest-first** — no strategy or optimiser change is considered real until it passes a backtest.
- **Lean dependencies** — pandas, numpy, scikit-learn. No heavy frameworks.
