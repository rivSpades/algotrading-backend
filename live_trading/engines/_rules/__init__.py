"""Pure helper functions implementing the per-strategy signal rules.

These helpers are intentionally side-effect free so they can be reused by
both the backtest executor and the live trading engine. They do not touch
the database, models, or any orchestration concerns.
"""

from .gap import GapDecision, gap_up_gap_down_decision

__all__ = ['GapDecision', 'gap_up_gap_down_decision']
