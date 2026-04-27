"""Snapshot recalc + DeploymentSymbol reconciliation.

Phase 6 owns the weekend job that re-runs the latest snapshot backtest for
every symbol enrolled in an active deployment and then re-classifies each
`DeploymentSymbol` based on the new stats.

State machine for `DeploymentSymbol.status`:

- was `active`/`pending_enable` and color stays GREEN -> no transition
  (we still refresh the cached stats and emit a `recalc_finished` event).
- was `active`/`pending_enable` and color drops below GREEN:
  - if the symbol still has an `open` `LiveTrade` -> `flagged_for_disable`
    (we never close a position behind the engine's back; the trader is
    notified and the symbol is taken offline once the trade resolves).
  - otherwise -> `disabled` immediately.
- was `disabled` and color recovers to GREEN -> `pending_enable`
  (the next market-open task can flip it to `active` once OHLCV is fresh).
- was `flagged_for_disable` and color recovers to GREEN -> `active`
  (we cancel the takedown).

Every transition is mirrored in the audit log via `log_event`. The recalc
phase also writes `recalc_started` / `recalc_finished` rows so the dashboard
can render a weekend-job timeline.

`reconcile_deployment_symbols(deployment)` is invoked by the weekend Celery
task `live_trading.weekend_snapshot_recalc` once all per-symbol
`backtest_engine.run_symbol_backtest_run` tasks finish. It is also exposed
manually via the API (Phase 7 will consider promotion gating).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from django.db import transaction
from django.utils import timezone

from backtest_engine.models import (
    SymbolBacktestParameterSet,
    SymbolBacktestRun,
)

from ..models import DeploymentSymbol, LiveTrade, StrategyDeployment
from ..utils.colors import GREEN
from .audit import log_event
from .symbol_selection import build_symbol_candidates

logger = logging.getLogger(__name__)


@dataclass
class SymbolRecalcOutcome:
    """Per-`DeploymentSymbol` reconciliation result."""

    deployment_symbol_id: int
    ticker: str
    previous_status: str
    new_status: str
    previous_color: str
    new_color: str
    has_open_trade: bool = False
    note: str = ''


@dataclass
class ReconciliationSummary:
    """Aggregate result of a single deployment recalc + reconcile pass."""

    deployment_id: int
    parameter_set: str
    started_at: str
    completed_at: Optional[str] = None
    candidates_total: int = 0
    candidates_green: int = 0
    transitions: list[SymbolRecalcOutcome] = field(default_factory=list)
    color_change_counts: Counter = field(default_factory=Counter)
    status_change_counts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            'deployment_id': self.deployment_id,
            'parameter_set': self.parameter_set,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'candidates_total': self.candidates_total,
            'candidates_green': self.candidates_green,
            'color_change_counts': dict(self.color_change_counts),
            'status_change_counts': dict(self.status_change_counts),
            'transitions': [
                {
                    'deployment_symbol_id': t.deployment_symbol_id,
                    'ticker': t.ticker,
                    'previous_status': t.previous_status,
                    'new_status': t.new_status,
                    'previous_color': t.previous_color,
                    'new_color': t.new_color,
                    'has_open_trade': t.has_open_trade,
                    'note': t.note,
                }
                for t in self.transitions
            ],
        }


def queue_snapshot_recalc(
    deployment: StrategyDeployment,
    *,
    actor_type: str = 'task',
    actor_id: str = '',
    only_tickers: Optional[list[str]] = None,
    enqueue: bool = True,
) -> dict:
    """Re-queue `run_symbol_backtest_run` for each symbol in the deployment.

    For every enrolled `DeploymentSymbol` we look up the most recent
    `SymbolBacktestRun` for the deployment's parameter set and re-dispatch the
    backtest task. When `enqueue=False` the function still resolves the run
    ids (useful for tests / offline reconciliation that re-uses existing
    runs without running them again).
    """

    from backtest_engine.tasks import run_symbol_backtest_run_task

    parameter_set = deployment.parameter_set
    if parameter_set is None:
        log_event(
            deployment,
            event_type='recalc_started',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=(
                f"Deployment {deployment.id} has no parameter_set; "
                "skipping snapshot recalc."
            ),
        )
        return {
            'status': 'skipped',
            'reason': 'no_parameter_set',
            'deployment_id': deployment.id,
        }

    deployment_symbols = list(deployment.deployment_symbols.select_related('symbol'))
    if only_tickers:
        deployment_symbols = [
            ds for ds in deployment_symbols if ds.symbol.ticker in only_tickers
        ]

    tickers = [ds.symbol.ticker for ds in deployment_symbols]
    runs = (
        SymbolBacktestRun.objects.filter(
            parameter_set=parameter_set,
            symbol__ticker__in=tickers,
        )
        .order_by('symbol__ticker', '-created_at')
        .select_related('symbol')
    )

    latest_runs: dict[str, SymbolBacktestRun] = {}
    for run in runs:
        ticker = run.symbol.ticker
        if ticker not in latest_runs:
            latest_runs[ticker] = run

    log_event(
        deployment,
        event_type='recalc_started',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"Snapshot recalc queued for {len(latest_runs)}/{len(tickers)} "
            f"symbols (deployment {deployment.id})."
        ),
        context={
            'parameter_set': parameter_set.signature,
            'symbol_count': len(tickers),
            'runs_found': len(latest_runs),
            'enqueue': enqueue,
        },
    )

    queued = []
    missing = [t for t in tickers if t not in latest_runs]
    for ticker, run in latest_runs.items():
        if enqueue:
            try:
                async_result = run_symbol_backtest_run_task.delay(run.id)
                queued.append({
                    'ticker': ticker, 'run_id': run.id,
                    'task_id': async_result.id,
                })
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    'Failed to enqueue snapshot recalc for run=%s', run.id,
                )
                log_event(
                    deployment,
                    event_type='error',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    level='warning',
                    message=f"Failed to enqueue recalc for {ticker}: {exc}",
                    error=exc,
                    context={'ticker': ticker, 'run_id': run.id},
                )
        else:
            queued.append({'ticker': ticker, 'run_id': run.id, 'task_id': None})

    return {
        'status': 'success',
        'deployment_id': deployment.id,
        'parameter_set': parameter_set.signature,
        'queued_count': len(queued),
        'queued': queued,
        'missing_tickers': missing,
    }


def reconcile_deployment_symbols(
    deployment: StrategyDeployment,
    *,
    actor_type: str = 'task',
    actor_id: str = '',
    parameter_set: Optional[SymbolBacktestParameterSet] = None,
) -> ReconciliationSummary:
    """Re-classify every `DeploymentSymbol` based on the latest snapshot stats.

    Builds a fresh `SelectionResult` from the deployment's parameter set,
    indexes it by ticker, then walks `deployment.deployment_symbols` applying
    the state-machine transitions described at the top of the module.
    """

    parameter_set = parameter_set or deployment.parameter_set
    started_at = timezone.now()
    summary = ReconciliationSummary(
        deployment_id=deployment.id,
        parameter_set=parameter_set.signature if parameter_set else '',
        started_at=started_at.isoformat(),
    )

    if parameter_set is None:
        log_event(
            deployment,
            event_type='recalc_finished',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message='No parameter_set; reconciliation skipped.',
        )
        summary.completed_at = timezone.now().isoformat()
        return summary

    selection = build_symbol_candidates(parameter_set, deployment.position_mode)
    summary.candidates_total = len(selection.candidates)
    summary.candidates_green = len(selection.green_candidates)
    candidates_by_ticker = {c.symbol.ticker: c for c in selection.candidates}

    for ds in deployment.deployment_symbols.select_related('symbol'):
        candidate = candidates_by_ticker.get(ds.symbol.ticker)
        outcome = _reconcile_one(
            deployment=deployment,
            deployment_symbol=ds,
            candidate=candidate,
            actor_type=actor_type,
            actor_id=actor_id,
        )
        if outcome is None:
            continue
        summary.transitions.append(outcome)
        if outcome.previous_color != outcome.new_color:
            summary.color_change_counts[
                f'{outcome.previous_color}->{outcome.new_color}'
            ] += 1
        if outcome.previous_status != outcome.new_status:
            summary.status_change_counts[
                f'{outcome.previous_status}->{outcome.new_status}'
            ] += 1

    completed_at = timezone.now()
    summary.completed_at = completed_at.isoformat()

    log_event(
        deployment,
        event_type='recalc_finished',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"Reconciled {len(summary.transitions)} symbol(s); "
            f"color changes={dict(summary.color_change_counts)}, "
            f"status changes={dict(summary.status_change_counts)}."
        ),
        context={
            'parameter_set': summary.parameter_set,
            'started_at': summary.started_at,
            'completed_at': summary.completed_at,
            'candidates_total': summary.candidates_total,
            'candidates_green': summary.candidates_green,
            'color_change_counts': dict(summary.color_change_counts),
            'status_change_counts': dict(summary.status_change_counts),
            'transitions_logged': len(summary.transitions),
        },
    )

    return summary


# ---------------------------------------------------------------------------
# Internal: per-symbol transition
# ---------------------------------------------------------------------------


def _reconcile_one(
    *,
    deployment: StrategyDeployment,
    deployment_symbol: DeploymentSymbol,
    candidate,
    actor_type: str,
    actor_id: str,
) -> Optional[SymbolRecalcOutcome]:
    previous_status = deployment_symbol.status
    previous_color = deployment_symbol.color_overall

    has_open_trade = LiveTrade.objects.filter(
        deployment_symbol=deployment_symbol, status='open',
    ).exists()

    if candidate is None:
        # Snapshot no longer covers this symbol — leave as-is but log.
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='recalc_finished',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=(
                f"{deployment_symbol.symbol.ticker}: no snapshot stats found; "
                f"keeping status='{previous_status}'."
            ),
            context={'ticker': deployment_symbol.symbol.ticker},
        )
        return SymbolRecalcOutcome(
            deployment_symbol_id=deployment_symbol.id,
            ticker=deployment_symbol.symbol.ticker,
            previous_status=previous_status,
            new_status=previous_status,
            previous_color=previous_color,
            new_color=previous_color,
            has_open_trade=has_open_trade,
            note='no_snapshot_stats',
        )

    new_color = candidate.color_overall
    new_status, transition_event, transition_message = _next_status(
        previous_status, previous_color, new_color, has_open_trade,
    )

    update_fields = ['updated_at']
    deployment_symbol.sharpe_long = candidate.sharpe_long
    deployment_symbol.sharpe_short = candidate.sharpe_short
    deployment_symbol.max_dd_long = candidate.max_dd_long
    deployment_symbol.max_dd_short = candidate.max_dd_short
    deployment_symbol.total_trades_long = candidate.total_trades_long
    deployment_symbol.total_trades_short = candidate.total_trades_short
    deployment_symbol.color_long = candidate.color_long
    deployment_symbol.color_short = candidate.color_short
    deployment_symbol.color_overall = candidate.color_overall
    deployment_symbol.tier = candidate.tier
    update_fields.extend([
        'sharpe_long', 'sharpe_short', 'max_dd_long', 'max_dd_short',
        'total_trades_long', 'total_trades_short',
        'color_long', 'color_short', 'color_overall', 'tier',
    ])

    if new_status != previous_status:
        deployment_symbol.status = new_status
        update_fields.append('status')

    with transaction.atomic():
        deployment_symbol.save(update_fields=update_fields)

    if previous_color != new_color:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='color_changed',
            actor_type=actor_type,
            actor_id=actor_id,
            message=(
                f"{deployment_symbol.symbol.ticker}: color "
                f"{previous_color} -> {new_color}"
            ),
            context={
                'ticker': deployment_symbol.symbol.ticker,
                'previous_color': previous_color,
                'new_color': new_color,
                'sharpe_long': candidate.sharpe_long,
                'sharpe_short': candidate.sharpe_short,
                'max_dd_long': candidate.max_dd_long,
                'max_dd_short': candidate.max_dd_short,
                'tier': candidate.tier,
            },
        )

    if transition_event:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type=transition_event,
            actor_type=actor_type,
            actor_id=actor_id,
            message=transition_message,
            context={
                'ticker': deployment_symbol.symbol.ticker,
                'previous_status': previous_status,
                'new_status': new_status,
                'previous_color': previous_color,
                'new_color': new_color,
                'has_open_trade': has_open_trade,
            },
        )

    return SymbolRecalcOutcome(
        deployment_symbol_id=deployment_symbol.id,
        ticker=deployment_symbol.symbol.ticker,
        previous_status=previous_status,
        new_status=new_status,
        previous_color=previous_color,
        new_color=new_color,
        has_open_trade=has_open_trade,
        note=transition_event or '',
    )


def _next_status(
    previous_status: str,
    previous_color: str,
    new_color: str,
    has_open_trade: bool,
) -> tuple[str, Optional[str], str]:
    """Return `(new_status, audit_event_or_None, audit_message)`.

    Implements the state machine described at the top of the module.
    """

    is_green_now = new_color == GREEN

    if previous_status == 'active':
        if is_green_now:
            return 'active', None, ''
        if has_open_trade:
            return (
                'flagged_for_disable',
                'symbol_flagged',
                (
                    f"Color dropped to '{new_color}' but symbol has an "
                    f"open trade; flagged for disable."
                ),
            )
        return (
            'disabled',
            'symbol_disabled',
            (
                f"Color dropped to '{new_color}' with no open trade; "
                f"disabling symbol."
            ),
        )

    if previous_status == 'flagged_for_disable':
        if is_green_now:
            return (
                'active',
                'symbol_enabled',
                'Color recovered to green; clearing flagged-for-disable.',
            )
        if not has_open_trade:
            return (
                'disabled',
                'symbol_disabled',
                'Flagged symbol no longer has an open trade; disabling.',
            )
        return previous_status, None, ''

    if previous_status == 'pending_enable':
        if is_green_now:
            return (
                'active',
                'symbol_enabled',
                'Pending-enable symbol verified green; activating.',
            )
        return (
            'disabled',
            'symbol_disabled',
            (
                f"Pending-enable symbol no longer green ({new_color}); "
                f"disabling."
            ),
        )

    if previous_status == 'disabled':
        if is_green_now:
            return (
                'pending_enable',
                'symbol_pending_enable',
                'Disabled symbol recovered to green; staging for enable.',
            )
        return previous_status, None, ''

    return previous_status, None, ''
