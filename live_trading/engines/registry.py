"""Registry mapping `StrategyDefinition.name` to a `BaseLiveTradingEngine`.

Engines self-register via the `register_live_engine` decorator. The
`live_trading.apps.LiveTradingConfig.ready` hook imports this package so the
decorators run before any task or view tries to look up an engine.
"""

from __future__ import annotations

from typing import Optional, Type

from .base import BaseLiveTradingEngine


class LiveEngineRegistry:
    """Class-level registry for live trading engines."""

    _engines: dict[str, Type[BaseLiveTradingEngine]] = {}

    @classmethod
    def register(cls, strategy_name: str, engine_class: Type[BaseLiveTradingEngine]) -> Type[BaseLiveTradingEngine]:
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError('strategy_name must be a non-empty string')
        if not isinstance(engine_class, type) or not issubclass(engine_class, BaseLiveTradingEngine):
            raise TypeError(
                f"Engine class must subclass BaseLiveTradingEngine, got {engine_class!r}",
            )
        cls._engines[strategy_name] = engine_class
        if not engine_class.name:
            engine_class.name = strategy_name
        return engine_class

    @classmethod
    def get(cls, strategy_name: str) -> Optional[Type[BaseLiveTradingEngine]]:
        return cls._engines.get(strategy_name)

    @classmethod
    def has(cls, strategy_name: str) -> bool:
        return strategy_name in cls._engines

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._engines.keys())

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry. Tests use this for isolated runs."""
        cls._engines.clear()


def register_live_engine(strategy_name: str):
    """Decorator that registers `engine_class` for `strategy_name`."""

    def decorator(engine_class: Type[BaseLiveTradingEngine]) -> Type[BaseLiveTradingEngine]:
        return LiveEngineRegistry.register(strategy_name, engine_class)

    return decorator


def get_live_engine(strategy_name: str) -> Optional[Type[BaseLiveTradingEngine]]:
    """Return the engine class registered for `strategy_name` (or None)."""
    return LiveEngineRegistry.get(strategy_name)


def get_live_engine_for_deployment(deployment) -> Optional[Type[BaseLiveTradingEngine]]:
    """Return the engine class registered for `deployment.strategy.name`."""
    return LiveEngineRegistry.get(deployment.strategy.name)
