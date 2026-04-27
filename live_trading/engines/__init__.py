"""Live trading engines package.

Each strategy registers its own concrete `BaseLiveTradingEngine` here. The
engines are auto-discovered when `live_trading.apps.LiveTradingConfig.ready`
runs so that the registry is populated by the time tasks/views look them up.
"""

from .base import BaseLiveTradingEngine, LiveSignal
from .registry import (
    LiveEngineRegistry,
    get_live_engine,
    get_live_engine_for_deployment,
    register_live_engine,
)

__all__ = [
    'BaseLiveTradingEngine',
    'LiveSignal',
    'LiveEngineRegistry',
    'get_live_engine',
    'get_live_engine_for_deployment',
    'register_live_engine',
]
