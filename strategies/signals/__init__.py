"""
Strategy signal blackbox — shared entry point for backtest and live.

Usage:
    from strategies.signals import check_strategy_signal, StrategySignalContext
"""

from __future__ import annotations

from .adapters import to_backtest_order, to_live_action
from .broker import resolve_broker_side_capabilities
from .registry import StrategySignalRegistry, register_strategy_signal
from .types import (
    GAP_STRATEGY_NAME,
    PositionState,
    SignalAction,
    StrategySignalContext,
    StrategySignalResult,
    position_state_from_backtest,
    position_state_from_live,
)

# Register handlers on import
from . import handlers  # noqa: F401


def check_strategy_signal(
    strategy_name: str,
    ctx: StrategySignalContext,
) -> StrategySignalResult:
    """Evaluate strategy signal using the registered handler for strategy_name."""
    handler = StrategySignalRegistry.get(strategy_name)
    if handler is None:
        raise ValueError(f'No signal handler registered for strategy: {strategy_name!r}')
    return handler(ctx)


__all__ = [
    'GAP_STRATEGY_NAME',
    'PositionState',
    'SignalAction',
    'StrategySignalContext',
    'StrategySignalResult',
    'StrategySignalRegistry',
    'check_strategy_signal',
    'position_state_from_backtest',
    'position_state_from_live',
    'register_strategy_signal',
    'resolve_broker_side_capabilities',
    'to_backtest_order',
    'to_live_action',
]
