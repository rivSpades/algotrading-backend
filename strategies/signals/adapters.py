"""Map canonical StrategySignalResult to runtime-specific order types."""

from __future__ import annotations

from typing import Optional

from .types import SignalAction, StrategySignalResult

# Live engine signal string constants (mirror live_trading.engines.base)
LIVE_SIGNAL_LONG = 'long'
LIVE_SIGNAL_SHORT = 'short'
LIVE_SIGNAL_EXIT_LONG = 'exit_long'
LIVE_SIGNAL_EXIT_SHORT = 'exit_short'
LIVE_SIGNAL_HOLD = 'hold'


def to_backtest_order(result: StrategySignalResult) -> Optional[str]:
    """Map canonical action to backtest buy/sell/None."""
    if result.action == SignalAction.LONG:
        return 'buy'
    if result.action == SignalAction.SHORT:
        return 'sell'
    if result.action == SignalAction.EXIT_LONG:
        return 'sell'
    if result.action == SignalAction.EXIT_SHORT:
        return 'buy'
    return None


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
