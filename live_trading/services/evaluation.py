"""Paper-deployment evaluation service.

A `StrategyDeployment` of `deployment_type='paper'` graduates to
real-money trading by passing a configurable evaluation check. The shape of
`StrategyDeployment.evaluation_criteria` mirrors the legacy schema so the API
surface stays compatible::

    {
        "min_trades": 10,
        "min_sharpe_ratio": 1.0,
        "min_win_rate": 0.4,           # 0..1
        "min_total_pnl": 100.0,         # absolute currency amount
        "max_drawdown": 30.0,           # absolute % (e.g. 30 -> -30%)
        "min_runtime_days": 7,          # at least N days since activation
    }

Each key is optional. The evaluation pulls live trade statistics from the
deployment's `live_trades` and computes the actual values; we then compare
them to the criteria and stash the breakdown in `evaluation_results`.

The result is::

    {
        "passed": bool,
        "evaluated_at": iso8601,
        "metrics": {
            "trades_count": int,
            "winning_trades": int,
            "win_rate": float,
            "total_pnl": float,
            "sharpe_ratio": float | None,
            "max_drawdown": float | None,
            "runtime_days": float | None,
        },
        "checks": [
            {"name": "min_trades", "passed": True, "actual": ..., "required": ...},
            ...
        ],
        "failed_checks": [...names...],
    }

`evaluate_deployment_for_promotion(...)` mutates `evaluation_results`,
optionally flips `status` to `passed`/`failed`, and returns the result so
the API layer can include it in the response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from django.db.models import Sum
from django.utils import timezone

from ..models import LiveTrade, StrategyDeployment
from .audit import log_event


@dataclass
class EvaluationCheck:
    name: str
    required: Any
    actual: Any
    passed: bool

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'required': self.required,
            'actual': self.actual,
            'passed': self.passed,
        }


@dataclass
class EvaluationResult:
    passed: bool
    evaluated_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[EvaluationCheck] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            'passed': self.passed,
            'evaluated_at': self.evaluated_at,
            'metrics': self.metrics,
            'checks': [c.to_dict() for c in self.checks],
            'failed_checks': self.failed_checks,
        }


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def compute_deployment_metrics(deployment: StrategyDeployment) -> dict[str, Any]:
    """Aggregate trade-level statistics from the deployment's `LiveTrade`s."""

    trades = list(deployment.live_trades.all())
    closed = [t for t in trades if t.status == 'closed' and t.pnl is not None]
    winners = [t for t in closed if (t.pnl or Decimal('0')) > 0]

    total_pnl = float(
        deployment.live_trades.filter(status='closed').aggregate(
            total=Sum('pnl'),
        )['total'] or 0,
    )

    win_rate = (len(winners) / len(closed)) if closed else None
    pnl_list = [float(t.pnl) for t in closed]
    sharpe = _trade_sharpe(pnl_list) if len(pnl_list) >= 2 else None
    max_dd = _max_drawdown(pnl_list) if pnl_list else None

    activated_at = deployment.activated_at or deployment.started_at
    runtime_days = None
    if activated_at:
        delta = timezone.now() - activated_at
        runtime_days = round(delta.total_seconds() / 86_400.0, 4)

    open_count = sum(1 for t in trades if t.status == 'open')

    return {
        'trades_count': len(closed),
        'open_trades_count': open_count,
        'winning_trades': len(winners),
        'losing_trades': len(closed) - len(winners),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'runtime_days': runtime_days,
    }


def evaluate_deployment_for_promotion(
    deployment: StrategyDeployment,
    *,
    actor_type: str = 'task',
    actor_id: str = '',
    save: bool = True,
    transition_status: bool = True,
) -> EvaluationResult:
    """Score `deployment` against its `evaluation_criteria`.

    When `save=True` the result is persisted on `evaluation_results` and
    (if `transition_status=True`) the deployment's status is flipped to
    `passed` or `failed`. Returns the `EvaluationResult` either way.
    """

    metrics = compute_deployment_metrics(deployment)
    criteria = deployment.evaluation_criteria or {}
    checks = _build_checks(criteria, metrics)
    passed = all(c.passed for c in checks) if checks else True
    evaluated_at = timezone.now().isoformat()

    result = EvaluationResult(
        passed=passed,
        evaluated_at=evaluated_at,
        metrics=metrics,
        checks=checks,
    )

    if save:
        deployment.evaluation_results = result.to_dict()
        update_fields = ['evaluation_results', 'evaluated_at', 'updated_at']
        deployment.evaluated_at = timezone.now()
        if transition_status and deployment.status in ('evaluating', 'active'):
            deployment.status = 'passed' if passed else 'failed'
            update_fields.append('status')
        deployment.save(update_fields=update_fields)

        log_event(
            deployment,
            event_type='evaluation_passed' if passed else 'evaluation_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='info' if passed else 'warning',
            message=(
                f"Evaluation {'passed' if passed else 'failed'} "
                f"({len(checks)} check(s), {len(result.failed_checks)} failed)."
            ),
            context={
                'metrics': metrics,
                'criteria': dict(criteria),
                'failed_checks': result.failed_checks,
            },
        )

    return result


def promote_to_real_money(
    deployment: StrategyDeployment,
    *,
    actor_type: str = 'user',
    actor_id: str = '',
    name_suffix: str = ' (Live)',
) -> StrategyDeployment:
    """Create a sibling `real_money` `StrategyDeployment`.

    Copies the parent's parameter set, broker, position mode, capital, bet
    sizing, evaluation criteria, and re-builds `DeploymentSymbol` rows from
    the parent's currently-active enrollment. The new deployment starts in
    `pending` status (the user must explicitly activate it). The link back
    to the paper deployment is stored on `parent_deployment`.
    """

    if deployment.deployment_type != 'paper':
        raise ValueError('Only paper deployments can be promoted.')
    if not deployment.can_promote_to_real_money():
        raise ValueError(
            'Deployment is not eligible for real-money promotion '
            f'(status={deployment.status}, broker_real_active='
            f'{deployment.broker.is_active_for_deployment_type("real_money")}).',
        )

    from ..models import DeploymentSymbol

    real = StrategyDeployment.objects.create(
        strategy=deployment.strategy,
        parameter_set=deployment.parameter_set,
        broker=deployment.broker,
        position_mode=deployment.position_mode,
        deployment_type='real_money',
        status='pending',
        name=(deployment.name or deployment.strategy.name) + name_suffix,
        initial_capital=deployment.initial_capital,
        bet_size_percentage=deployment.bet_size_percentage,
        strategy_parameters=dict(deployment.strategy_parameters or {}),
        evaluation_criteria=dict(deployment.evaluation_criteria or {}),
        evaluation_results={},
        parent_deployment=deployment,
        hedge_enabled=bool(deployment.hedge_enabled),
        hedge_config=dict(deployment.hedge_config or {}),
    )

    sources = deployment.deployment_symbols.filter(status='active').select_related('symbol')
    new_symbols = [
        DeploymentSymbol(
            deployment=real,
            symbol=src.symbol,
            position_mode=src.position_mode,
            sharpe_long=src.sharpe_long,
            sharpe_short=src.sharpe_short,
            max_dd_long=src.max_dd_long,
            max_dd_short=src.max_dd_short,
            total_trades_long=src.total_trades_long,
            total_trades_short=src.total_trades_short,
            color_long=src.color_long,
            color_short=src.color_short,
            color_overall=src.color_overall,
            tier=src.tier,
            priority=src.priority,
            status='active',
        )
        for src in sources
    ]
    DeploymentSymbol.objects.bulk_create(new_symbols)

    log_event(
        deployment,
        event_type='promote_to_real',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"Promoted paper deployment {deployment.id} to real-money "
            f"sibling {real.id} ({len(new_symbols)} symbols)."
        ),
        context={
            'paper_deployment_id': deployment.id,
            'real_deployment_id': real.id,
            'symbols_copied': len(new_symbols),
        },
    )
    log_event(
        real,
        event_type='deploy_created',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"Real-money deployment created from paper sibling {deployment.id}."
        ),
        context={
            'paper_deployment_id': deployment.id,
            'real_deployment_id': real.id,
            'broker': deployment.broker.code,
            'parameter_set': (
                deployment.parameter_set.signature if deployment.parameter_set else None
            ),
            'symbols_copied': len(new_symbols),
        },
    )
    return real


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_checks(criteria: dict, metrics: dict) -> list[EvaluationCheck]:
    checks: list[EvaluationCheck] = []

    if 'min_trades' in criteria:
        required = int(criteria['min_trades'])
        actual = int(metrics.get('trades_count') or 0)
        checks.append(EvaluationCheck(
            'min_trades', required, actual, actual >= required,
        ))

    if 'min_sharpe_ratio' in criteria:
        required = float(criteria['min_sharpe_ratio'])
        actual = _safe_float(metrics.get('sharpe_ratio'))
        passed = actual is not None and actual >= required
        checks.append(EvaluationCheck('min_sharpe_ratio', required, actual, passed))

    if 'min_win_rate' in criteria:
        required = float(criteria['min_win_rate'])
        actual = _safe_float(metrics.get('win_rate'))
        passed = actual is not None and actual >= required
        checks.append(EvaluationCheck('min_win_rate', required, actual, passed))

    if 'min_total_pnl' in criteria:
        required = float(criteria['min_total_pnl'])
        actual = float(metrics.get('total_pnl') or 0.0)
        checks.append(EvaluationCheck(
            'min_total_pnl', required, actual, actual >= required,
        ))

    if 'max_drawdown' in criteria:
        required = float(criteria['max_drawdown'])
        actual = _safe_float(metrics.get('max_drawdown'))
        # max_drawdown values are absolute % already; passes when within budget.
        passed = actual is None or abs(actual) <= required
        checks.append(EvaluationCheck('max_drawdown', required, actual, passed))

    if 'min_runtime_days' in criteria:
        required = float(criteria['min_runtime_days'])
        actual = _safe_float(metrics.get('runtime_days'))
        passed = actual is not None and actual >= required
        checks.append(EvaluationCheck('min_runtime_days', required, actual, passed))

    return checks


def _trade_sharpe(pnl_list: list[float]) -> Optional[float]:
    n = len(pnl_list)
    if n < 2:
        return None
    mean = sum(pnl_list) / n
    var = sum((p - mean) ** 2 for p in pnl_list) / (n - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var)


def _max_drawdown(pnl_list: list[float]) -> Optional[float]:
    """Return the worst peak-to-trough drawdown of the cumulative PnL curve.

    Reported as an absolute percentage of the running peak. Returns 0 when
    the curve never dips below its peak.
    """

    if not pnl_list:
        return None
    cum = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in pnl_list:
        cum += pnl
        if cum > peak:
            peak = cum
        if peak > 0:
            dd = (peak - cum) / peak * 100.0
            if dd > worst:
                worst = dd
    return worst
