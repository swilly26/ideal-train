"""Strategy layer — signal generation from market data.

Provides the base class every trading strategy must implement,
plus a registry for discovering and loading strategies at runtime.
"""

from src.strategies.base import Signal, SignalType, Strategy, StrategyConfig
from src.strategies.registry import StrategyRegistry

__all__ = ["Signal", "SignalType", "Strategy", "StrategyConfig", "StrategyRegistry"]
