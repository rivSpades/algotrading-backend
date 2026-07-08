"""Canonical types for strategy signal evaluation (backtest + live)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SignalAction(str, Enum):
    LONG = 'long'
    SHORT = 'short'
    EXIT_LONG = 'exit_long'
    EXIT_SHORT = 'exit_short'
    HOLD = 'hold'


class PositionState(str, Enum):
    FLAT = 'flat'
    LONG = 'long'
    SHORT = 'short'


GAP_STRATEGY_NAME = 'Gap-Up and Gap-Down'


def position_state_from_backtest(position: Optional[dict]) -> PositionState:
    """Map backtest position dict to canonical position state."""
    if not position:
        return PositionState.FLAT
    ptype = (position.get('type') or '').lower()
    if ptype == 'buy':
        return PositionState.LONG
    if ptype == 'sell':
        return PositionState.SHORT
    return PositionState.FLAT


def position_state_from_live(position_mode: Optional[str]) -> PositionState:
    """Map open LiveTrade.position_mode to canonical state."""
    if not position_mode:
        return PositionState.FLAT
    mode = position_mode.lower()
    if mode == 'long':
        return PositionState.LONG
    if mode == 'short':
        return PositionState.SHORT
    return PositionState.FLAT


@dataclass
class StrategySignalContext:
    """Inputs for a single signal evaluation (after runtime-specific prep)."""

    returns: Optional[float]
    std: Optional[float]
    threshold: float = 0.25
    position_mode: str = 'long'
    position_state: PositionState = PositionState.FLAT
    long_allowed: bool = True
    short_allowed: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySignalResult:
    """Outcome of check_strategy_signal."""

    action: SignalAction
    reason: str
    decision: Any = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.action in (SignalAction.LONG, SignalAction.SHORT)

    @property
    def is_exit(self) -> bool:
        return self.action in (SignalAction.EXIT_LONG, SignalAction.EXIT_SHORT)

    @property
    def is_actionable(self) -> bool:
        return self.is_entry or self.is_exit
