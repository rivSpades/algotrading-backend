"""Map canonical StrategySignalResult to runtime-specific order types."""

from __future__ import annotations

from typing import Optional

from .types import (
    LIVE_SIGNAL_EXIT_LONG,
    LIVE_SIGNAL_EXIT_SHORT,
    LIVE_SIGNAL_HOLD,
    LIVE_SIGNAL_LONG,
    LIVE_SIGNAL_SHORT,
    SignalAction,
    StrategySignalResult,
    action_to_side,
)

# Re-exported for callers that historically imported these from here.
__all__ = [
    'LIVE_SIGNAL_LONG',
    'LIVE_SIGNAL_SHORT',
    'LIVE_SIGNAL_EXIT_LONG',
    'LIVE_SIGNAL_EXIT_SHORT',
    'LIVE_SIGNAL_HOLD',
    'to_backtest_order',
    'to_live_action',
]


def to_backtest_order(result: StrategySignalResult) -> Optional[str]:
    """Map canonical action to backtest buy/sell/None."""
    side = action_to_side(result.action)
    return side or None


def to_live_action(result: StrategySignalResult) -> str:
    """Map canonical action to LiveSignal.action string."""
    mapping = {
        SignalAction.LONG: LIVE_SIGNAL_LONG,
        SignalAction.SHORT: LIVE_SIGNAL_SHORT,
        SignalAction.EXIT_LONG: LIVE_SIGNAL_EXIT_LONG,
        SignalAction.EXIT_SHORT: LIVE_SIGNAL_EXIT_SHORT,
        SignalAction.HOLD: LIVE_SIGNAL_HOLD,
    }
    return mapping.get(result.action, LIVE_SIGNAL_HOLD)
