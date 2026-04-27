"""Pure rule for the Gap-Up and Gap-Down strategy.

`returns_t = (open_t - close_{t-1}) / close_{t-1}` is normalised by a rolling
standard deviation of returns calculated using only data up to today's open
(bias-safe). The rule is a pure function of `(returns, std, threshold)` so it
can be reused by the backtest executor and the live trading engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    """Apply the gap-up / gap-down inequality.

    Returns a `GapDecision` describing what the *raw* rule says before any
    position-mode / broker-capability filtering. Callers (backtest executor,
    live engine, etc.) layer those concerns on top.

    If `returns` or `std` is missing or non-positive, `direction` is `None`.
    """

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
