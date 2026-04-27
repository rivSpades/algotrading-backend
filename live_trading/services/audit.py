"""Audit logging helper for the live trading subsystem.

Every action that affects a `StrategyDeployment` (creation, lifecycle change,
signal evaluation, order placement, recalc, etc.) should call `log_event`
instead of plain `logger.info(...)` so we can render a rich audit feed in the
dashboard and trace what task / user performed each step.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

LEVEL_TO_LOGGING = {
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}


def log_event(
    deployment=None,
    *,
    event_type: str,
    actor_type: str = 'system',
    actor_id: Optional[str] = None,
    deployment_symbol=None,
    level: str = 'info',
    message: str = '',
    context: Optional[dict[str, Any]] = None,
    error: Optional[BaseException | str] = None,
):
    """Persist a `DeploymentEvent` row and mirror it into the python logger.

    Args:
        deployment: `StrategyDeployment` instance (or None for system events).
        event_type: one of `DeploymentEvent.EVENT_TYPES`.
        actor_type: `user` / `task` / `system` / `broker`.
        actor_id: identifier of who/what performed the action (user id,
            celery task id, …).
        deployment_symbol: optional `DeploymentSymbol` row this event is about.
        level: `info` / `warning` / `error`.
        message: short human-readable summary (≤ 500 chars).
        context: additional JSON-serialisable details.
        error: exception or string captured when level == 'error'.
    """

    from ..models import DeploymentEvent

    error_text = ''
    if error is not None:
        error_text = str(error)
        if isinstance(error, BaseException) and level == 'info':
            level = 'error'

    safe_context = dict(context or {})

    try:
        event = DeploymentEvent.objects.create(
            deployment=deployment,
            deployment_symbol=deployment_symbol,
            event_type=event_type,
            level=level,
            actor_type=actor_type,
            actor_id=str(actor_id) if actor_id is not None else '',
            message=(message or '')[:500],
            context=safe_context,
            error=error_text,
        )
    except Exception:
        # Never let audit logging break the caller.
        logger.exception('Failed to persist DeploymentEvent (%s)', event_type)
        event = None

    log_method = LEVEL_TO_LOGGING.get(level, logging.INFO)
    deployment_id = getattr(deployment, 'id', None)
    deployment_symbol_id = getattr(deployment_symbol, 'id', None)
    logger.log(
        log_method,
        '[%s] deployment=%s symbol=%s actor=%s/%s message=%s%s',
        event_type,
        deployment_id,
        deployment_symbol_id,
        actor_type,
        actor_id or '',
        message,
        f' error={error_text}' if error_text else '',
    )
    return event
