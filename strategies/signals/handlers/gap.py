"""Gap-Up and Gap-Down strategy signal handler."""

from __future__ import annotations

from ..registry import register_strategy_signal
from ..rules.gap import classify_gap_position_action, gap_up_gap_down_decision
from ..types import GAP_STRATEGY_NAME, SignalAction, StrategySignalContext, StrategySignalResult


@register_strategy_signal(GAP_STRATEGY_NAME)
def evaluate_gap_signal(ctx: StrategySignalContext) -> StrategySignalResult:
    decision = gap_up_gap_down_decision(
        returns=ctx.returns,
        std=ctx.std,
        threshold=ctx.threshold,
    )

    if decision.context.get('reason') in ('missing_inputs', 'non_numeric_inputs', 'invalid_std'):
        return StrategySignalResult(
            action=SignalAction.HOLD,
            reason=decision.context.get('reason', 'missing_inputs'),
            decision=decision,
            context={'long_signal': False, 'short_signal': False},
        )

    action, reason = classify_gap_position_action(
        decision=decision,
        position_state=ctx.position_state,
        position_mode=ctx.position_mode,
        long_allowed=ctx.long_allowed,
        short_allowed=ctx.short_allowed,
    )

    return StrategySignalResult(
        action=action,
        reason=reason,
        decision=decision,
        context={
            'long_signal': decision.long_signal,
            'short_signal': decision.short_signal,
            'long_threshold': decision.long_threshold,
            'short_threshold': decision.short_threshold,
            'returns': decision.returns,
            'std': decision.std,
            'raw_direction': decision.direction,
        },
    )
