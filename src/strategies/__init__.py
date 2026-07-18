"""Strategy layer — signal generation from market data.

Provides the base class every trading strategy must implement,
plus a registry for discovering and loading strategies at runtime.
"""

from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.registry import StrategyRegistry

# Singleton registry — strategies are registered below after import.
registry = StrategyRegistry()

# Import concrete strategies …
from src.strategies.momentum import MomentumStrategy  # noqa: E402
from src.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from src.strategies.breakout import BreakoutStrategy  # noqa: E402

# … and register them so the optimizer can discover them at runtime.
registry.register("momentum", MomentumStrategy)
registry.register("mean_reversion", MeanReversionStrategy)
registry.register("breakout", BreakoutStrategy)

__all__ = [
    "Signal",
    "SignalType",
    "Strategy",
    "StrategyConfig",
    "StrategyRegistry",
    "registry",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
]
