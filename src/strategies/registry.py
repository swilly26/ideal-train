"""Strategy registry for runtime discovery and loading.

Keeps a central catalogue of available strategies so the backtester
and optimiser can iterate over them without hard-coded imports.
"""

from __future__ import annotations

from typing import Type

from src.strategies.base import Strategy


class StrategyRegistry:
    """Thread-safe registry of strategy classes.

    Usage::

        registry = StrategyRegistry()
        registry.register("momentum", MomentumStrategy)
        strategy_cls = registry.get("momentum")
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Type[Strategy]] = {}

    def register(self, name: str, strategy_cls: Type[Strategy]) -> None:
        """Register a strategy class under a unique *name*."""
        if name in self._strategies:
            raise KeyError(f"Strategy '{name}' is already registered")
        self._strategies[name] = strategy_cls

    def get(self, name: str) -> Type[Strategy]:
        """Return the strategy class registered under *name*."""
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found")
        return self._strategies[name]

    def list(self) -> list[str]:
        """Return all registered strategy names."""
        return list(self._strategies.keys())

    def clear(self) -> None:
        """Remove all registered strategies (useful in tests)."""
        self._strategies.clear()
