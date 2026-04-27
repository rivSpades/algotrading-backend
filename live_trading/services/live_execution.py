"""Glue between the live trading engine registry and the deployment lifecycle.

The functions here are the *only* place the rest of the codebase (Celery
tasks, manual fire-now endpoint, future order placement) should reach into
to evaluate a deployment symbol. They take care of:

- Resolving the right `BaseLiveTradingEngine` for the deployment's strategy.
- Caching engine instances per `(deployment_id, strategy_id)` within a fire
  so multiple symbols don't pay the construction cost twice.
- Persisting `DeploymentEvent` rows for `signal_evaluated` (and `error`) so
  the audit log captures every fire, including no-ops with diagnostic info.
- Writing `last_signal_at` on the deployment / deployment symbol when an
  engine actually emitted an actionable signal.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Optional

from django.db import transaction
from django.utils import timezone

from ..engines import BaseLiveTradingEngine, get_live_engine_for_deployment
from ..engines.base import EngineEvaluation
from ..models import DeploymentSymbol, StrategyDeployment
from .audit import log_event

logger = logging.getLogger(__name__)


class EngineNotRegistered(Exception):
    """Raised when no engine is registered for the deployment's strategy."""


def get_engine_instance(
    deployment: StrategyDeployment,
    *,
    broker_adapter=None,
    clock=None,
) -> BaseLiveTradingEngine:
    """Build a fresh engine instance for a deployment.

    Caller is responsible for caching across multiple symbols when fanning
    out within a single task fire.
    """
    engine_cls = get_live_engine_for_deployment(deployment)
    if engine_cls is None:
        raise EngineNotRegistered(
            f"No live engine registered for strategy {deployment.strategy.name!r}",
        )
    return engine_cls(deployment, broker_adapter=broker_adapter, clock=clock)


def evaluate_deployment_symbol(
    deployment_symbol: DeploymentSymbol,
    *,
    fire_at: Optional[datetime] = None,
    actor_type: str = 'system',
    actor_id: str = '',
    broker_adapter=None,
    engine: Optional[BaseLiveTradingEngine] = None,
    log: bool = True,
    place_orders: bool = False,
) -> EngineEvaluation:
    """Run one engine fire for a single deployment symbol.

    `engine` may be passed in for batched fires that re-use the same instance
    across multiple symbols. When omitted, a fresh engine is constructed.
    Set `place_orders=True` to also submit the broker order via the order
    service when the engine emits an actionable signal.
    """

    fire_at = fire_at or timezone.now()
    deployment = deployment_symbol.deployment

    try:
        instance = engine or get_engine_instance(
            deployment, broker_adapter=broker_adapter,
        )
    except EngineNotRegistered as exc:
        if log:
            log_event(
                deployment,
                deployment_symbol=deployment_symbol,
                event_type='error',
                actor_type=actor_type,
                actor_id=actor_id,
                level='error',
                message=str(exc),
                error=exc,
                context={'ticker': deployment_symbol.symbol.ticker},
            )
        raise

    try:
        evaluation = instance.evaluate(deployment_symbol, fire_at)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Live engine fire failed for deployment=%s symbol=%s",
            deployment.id, deployment_symbol.symbol.ticker,
        )
        if log:
            log_event(
                deployment,
                deployment_symbol=deployment_symbol,
                event_type='error',
                actor_type=actor_type,
                actor_id=actor_id,
                level='error',
                message=f"Engine evaluate() raised: {exc}",
                error=exc,
                context={'ticker': deployment_symbol.symbol.ticker},
            )
        raise

    if log:
        _log_evaluation(deployment, deployment_symbol, evaluation, actor_type, actor_id)
        _persist_signal_marks(deployment, deployment_symbol, evaluation, fire_at)

    if place_orders and evaluation.signal and evaluation.signal.is_actionable:
        from .order_service import place_signal_order  # late import to avoid cycle
        try:
            place_signal_order(
                deployment_symbol,
                evaluation.signal,
                broker_adapter=broker_adapter,
                actor_type=actor_type,
                actor_id=actor_id,
                fire_at=fire_at,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                'place_signal_order failed for deployment=%s symbol=%s',
                deployment.id, deployment_symbol.symbol.ticker,
            )
            if log:
                log_event(
                    deployment,
                    deployment_symbol=deployment_symbol,
                    event_type='order_failed',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    level='error',
                    message=f"place_signal_order raised: {exc}",
                    error=exc,
                    context={'ticker': deployment_symbol.symbol.ticker},
                )

    return evaluation


def evaluate_deployment(
    deployment: StrategyDeployment,
    *,
    fire_at: Optional[datetime] = None,
    actor_type: str = 'system',
    actor_id: str = '',
    broker_adapter=None,
    only_symbol_ids: Optional[Iterable[int]] = None,
    include_disabled: bool = False,
    place_orders: bool = False,
) -> list[EngineEvaluation]:
    """Run the engine for every active deployment symbol.

    Returns the per-symbol evaluations in priority order. The caller (Phase 5
    task) is responsible for actually placing orders; this function only
    persists audit rows.
    """

    fire_at = fire_at or timezone.now()
    qs = deployment.deployment_symbols.select_related(
        'symbol', 'symbol__exchange',
    )
    if not include_disabled:
        qs = qs.filter(status='active')
    if only_symbol_ids is not None:
        qs = qs.filter(id__in=list(only_symbol_ids))
    qs = qs.order_by('priority', 'symbol__ticker')

    try:
        engine = get_engine_instance(deployment, broker_adapter=broker_adapter)
    except EngineNotRegistered as exc:
        log_event(
            deployment,
            event_type='error',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=str(exc),
            error=exc,
        )
        return []

    log_event(
        deployment,
        event_type='task_tick',
        actor_type=actor_type,
        actor_id=actor_id,
        message=f"Engine fire ({deployment.strategy.name})",
        context={
            'fire_at': fire_at.isoformat(),
            'symbol_count': qs.count(),
            'engine': engine.engine_id(),
        },
    )

    evaluations: list[EngineEvaluation] = []
    for ds in qs:
        try:
            evaluations.append(
                evaluate_deployment_symbol(
                    ds,
                    fire_at=fire_at,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    broker_adapter=broker_adapter,
                    engine=engine,
                    place_orders=place_orders,
                ),
            )
        except Exception:
            # The error was already logged in evaluate_deployment_symbol;
            # keep going so one bad symbol doesn't cancel the whole fire.
            continue

    return evaluations


def _log_evaluation(
    deployment: StrategyDeployment,
    deployment_symbol: DeploymentSymbol,
    evaluation: EngineEvaluation,
    actor_type: str,
    actor_id: str,
) -> None:
    signal = evaluation.signal
    if signal is None:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='signal_evaluated',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=f"{deployment_symbol.symbol.ticker}: no signal produced",
            context={
                'ticker': deployment_symbol.symbol.ticker,
                'fire_at': evaluation.fire_at.isoformat(),
                'skipped_reason': evaluation.skipped_reason,
                'ohlcv_count': evaluation.ohlcv_count,
                'indicators_count': evaluation.indicators_count,
            },
        )
        return

    level = 'info'
    if signal.error:
        level = 'warning'

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='signal_evaluated',
        actor_type=actor_type,
        actor_id=actor_id,
        level=level,
        message=(
            f"{deployment_symbol.symbol.ticker}: action={signal.action} "
            f"({signal.context.get('reason', '-')})"
        ),
        context={
            'ticker': deployment_symbol.symbol.ticker,
            'fire_at': evaluation.fire_at.isoformat(),
            'action': signal.action,
            'confidence': signal.confidence,
            'price': str(signal.price) if signal.price is not None else None,
            'bar_timestamp': (
                signal.bar_timestamp.isoformat() if signal.bar_timestamp else None
            ),
            'ohlcv_count': evaluation.ohlcv_count,
            'indicators_count': evaluation.indicators_count,
            'rule': signal.context,
        },
        error=signal.error,
    )


def _persist_signal_marks(
    deployment: StrategyDeployment,
    deployment_symbol: DeploymentSymbol,
    evaluation: EngineEvaluation,
    fire_at: datetime,
) -> None:
    if evaluation.signal is None:
        return
    update_symbol_fields = ['last_evaluated_at', 'updated_at']
    deployment_symbol.last_evaluated_at = fire_at
    if evaluation.signal.is_actionable:
        deployment_symbol.last_signal_at = fire_at
        update_symbol_fields.insert(0, 'last_signal_at')
    with transaction.atomic():
        deployment_symbol.save(update_fields=update_symbol_fields)
        if evaluation.signal.is_actionable:
            deployment.last_signal_at = fire_at
            deployment.save(update_fields=['last_signal_at', 'updated_at'])
