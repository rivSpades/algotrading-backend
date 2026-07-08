"""Pure gap-up / gap-down rule and position classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..types import PositionState, SignalAction


@dataclass
class GapDecision:
    """Outcome of evaluating the gap rule on a single bar."""

    direction: Optional[str]  # 'long' | 'short' | None
    long_signal: bool
    short_signal: bool
    long_threshold: float
    short_threshold: float
    returns: float
    std: float
    threshold: float
    context: dict = field(default_factory=dict)


def gap_up_gap_down_decision(
    *,
    returns: Optional[float],
    std: Optional[float],
    threshold: float = 0.25,
) -> GapDecision:
    """Apply the gap-up / gap-down inequality (pure math, no position state)."""

    if returns is None or std is None:
        return GapDecision(
            direction=None,
            long_signal=False,
            short_signal=False,
            long_threshold=0.0,
            short_threshold=0.0,
            returns=float(returns) if returns is not None else float('nan'),
            std=float(std) if std is not None else float('nan'),
            threshold=threshold,
            context={'reason': 'missing_inputs'},
        )

    try:
        returns_f = float(returns)
        std_f = float(std)
    except (TypeError, ValueError):
        return GapDecision(
            direction=None,
            long_signal=False,
            short_signal=False,
            long_threshold=0.0,
            short_threshold=0.0,
            returns=float('nan'),
            std=float('nan'),
            threshold=threshold,
            context={'reason': 'non_numeric_inputs'},
        )

    if std_f != std_f or std_f <= 0.0:
        return GapDecision(
            direction=None,
            long_signal=False,
            short_signal=False,
            long_threshold=0.0,
            short_threshold=0.0,
            returns=returns_f,
            std=std_f,
            threshold=threshold,
            context={'reason': 'invalid_std'},
        )

    long_threshold = threshold * std_f
    short_threshold = -threshold * std_f
    long_signal = returns_f > long_threshold
    short_signal = returns_f < short_threshold

    direction: Optional[str] = None
    if long_signal:
        direction = 'long'
    elif short_signal:
        direction = 'short'

    return GapDecision(
        direction=direction,
        long_signal=long_signal,
        short_signal=short_signal,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        returns=returns_f,
        std=std_f,
        threshold=threshold,
        context={},
    )


def classify_gap_position_action(
    *,
    decision: GapDecision,
    position_state: PositionState,
    position_mode: str,
    long_allowed: bool,
    short_allowed: bool,
) -> tuple[SignalAction, str]:
    """
    Map GapDecision + position/mode/broker caps to a canonical SignalAction.

    Shared by backtest and live — exit priority before entry; no same-bar flip.
    """
    mode = (position_mode or 'long').lower()

    if position_state == PositionState.LONG:
        if mode == 'short':
            return SignalAction.EXIT_LONG, 'mode_disallows_long'
        if mode == 'long' and decision.short_signal:
            return SignalAction.EXIT_LONG, 'opposite_signal_long'
        return SignalAction.HOLD, 'in_position_no_exit'

    if position_state == PositionState.SHORT:
        if mode == 'long':
            return SignalAction.EXIT_SHORT, 'mode_disallows_short'
        if mode == 'short' and decision.long_signal:
            return SignalAction.EXIT_SHORT, 'opposite_signal_short'
        return SignalAction.HOLD, 'in_position_no_exit'

    if decision.direction is None:
        return SignalAction.HOLD, decision.context.get('reason', 'no_signal')

    if decision.direction == 'long' and mode in ('long', 'all') and long_allowed:
        return SignalAction.LONG, 'long_entry'
    if decision.direction == 'short' and mode in ('short', 'all') and short_allowed:
        return SignalAction.SHORT, 'short_entry'

    if decision.direction == 'long' and mode == 'short':
        return SignalAction.HOLD, 'long_signal_but_short_only'
    if decision.direction == 'short' and mode == 'long':
        return SignalAction.HOLD, 'short_signal_but_long_only'
    if decision.direction == 'long' and not long_allowed:
        return SignalAction.HOLD, 'long_disabled_by_broker'
    if decision.direction == 'short' and not short_allowed:
        return SignalAction.HOLD, 'short_disabled_by_broker'

    return SignalAction.HOLD, 'no_actionable_path'
