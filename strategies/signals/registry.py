"""Registry mapping strategy names to signal evaluation handlers."""

from __future__ import annotations

from typing import Callable, Optional

from .types import StrategySignalContext, StrategySignalResult

SignalHandler = Callable[[StrategySignalContext], StrategySignalResult]


class StrategySignalRegistry:
    _handlers: dict[str, SignalHandler] = {}

    @classmethod
    def register(cls, strategy_name: str, handler: SignalHandler) -> SignalHandler:
        if not strategy_name or not strategy_name.strip():
            raise ValueError('strategy_name must be non-empty')
        cls._handlers[strategy_name] = handler
        return handler

    @classmethod
    def get(cls, strategy_name: str) -> Optional[SignalHandler]:
        return cls._handlers.get(strategy_name)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._handlers.keys())


def register_strategy_signal(strategy_name: str):
    """Decorator to register a strategy signal handler."""

    def decorator(fn: SignalHandler) -> SignalHandler:
        return StrategySignalRegistry.register(strategy_name, fn)

    return decorator
