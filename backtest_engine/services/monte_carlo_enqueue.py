"""Enqueue order-variance Monte Carlo after a portfolio backtest completes."""

from __future__ import annotations

import logging
from typing import Optional

from backtest_engine.models import Backtest, PortfolioMonteCarloSimulation

logger = logging.getLogger(__name__)


def maybe_enqueue_monte_carlo_for_backtest(backtest: Backtest) -> Optional[str]:
    """
    If backtest is parent-linked and monte_carlo_num_paths > 0, start MC task.
    Returns Celery task id or None.
    """
    num_paths = int(getattr(backtest, 'monte_carlo_num_paths', 0) or 0)
    if num_paths <= 0 or not backtest.parameter_set_id:
        return None

    from backtest_engine.tasks import run_portfolio_monte_carlo_task

    simulation = (
        PortfolioMonteCarloSimulation.objects.filter(backtest=backtest)
        .order_by('-created_at')
        .first()
    )
    if simulation and simulation.status in ('pending', 'running'):
        task = run_portfolio_monte_carlo_task.delay(simulation.id)
        return task.id

    if not simulation:
        simulation = PortfolioMonteCarloSimulation.objects.create(
            backtest=backtest,
            num_paths=num_paths,
            reference_symbol_order=list(backtest.symbol_priority_order or []),
            status='pending',
        )
    else:
        simulation.num_paths = num_paths
        simulation.status = 'pending'
        simulation.error_message = ''
        simulation.save(update_fields=['num_paths', 'status', 'error_message', 'updated_at'])

    task = run_portfolio_monte_carlo_task.delay(simulation.id)
    logger.info(
        'Enqueued Monte Carlo simulation %s for backtest %s (task=%s)',
        simulation.id,
        backtest.id,
        task.id,
    )
    return task.id
