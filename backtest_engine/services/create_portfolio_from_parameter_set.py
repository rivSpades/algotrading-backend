"""
Create a portfolio backtest locked to a SymbolBacktestParameterSet (global test parent).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from backtest_engine.models import Backtest, SymbolBacktestParameterSet, SymbolBacktestRun, PortfolioMonteCarloSimulation
from backtest_engine.position_modes import normalize_position_modes
from backtest_engine.services.hybrid_vix_hedge import resolved_hedge_config_for_backtest
from backtest_engine.services.order_permutations import cap_variant_request
from backtest_engine.tasks import run_backtest_task
from strategies.models import StrategyDefinition

logger = logging.getLogger(__name__)


def _parse_dt(value: Any):
    if value is None:
        return None
    if hasattr(value, 'isoformat'):
        return value
    if isinstance(value, str):
        return parse_datetime(value.replace('Z', '+00:00') if value.endswith('Z') else value)
    return value


def _params_from_parameter_set(ps: SymbolBacktestParameterSet) -> Dict[str, Any]:
    """Extract run config from canonical parameter_set.parameters payload."""
    p = ps.parameters if isinstance(ps.parameters, dict) else {}
    start_date = _parse_dt(p.get('start_date')) or timezone.now().replace(year=1900, month=1, day=1)
    end_date = _parse_dt(p.get('end_date')) or timezone.now()
    initial_capital = p.get('initial_capital', '10000')
    try:
        initial_capital = Decimal(str(initial_capital))
    except Exception:
        initial_capital = Decimal('10000')
    hedge_enabled = bool(p.get('hedge_enabled', False))
    hedge_config = p.get('hedge_config') or {}
    if hedge_enabled:
        hedge_config = resolved_hedge_config_for_backtest(hedge_config)
    return {
        'start_date': start_date,
        'end_date': end_date,
        'split_ratio': float(p.get('split_ratio', 0.7)),
        'initial_capital': initial_capital,
        'bet_size_percentage': float(p.get('bet_size_percentage', 100.0)),
        'strategy_parameters': dict(p.get('strategy_parameters') or {}),
        'position_modes': normalize_position_modes(p.get('position_modes')),
        'hedge_enabled': hedge_enabled,
        'run_strategy_only_baseline': bool(p.get('run_strategy_only_baseline', True)),
        'hedge_config': hedge_config if hedge_enabled else {},
        'broker': ps.broker,
    }


def _completed_symbols_for_parameter_set(
    strategy: StrategyDefinition,
    ps: SymbolBacktestParameterSet,
) -> Tuple[List, List[str], List[str]]:
    """
    Return (Symbol instances, tickers) for latest completed run per ticker under this parameter set.
    """
    runs = (
        SymbolBacktestRun.objects.filter(strategy=strategy, parameter_set=ps)
        .select_related('symbol')
        .order_by('symbol__ticker', '-created_at')
    )
    latest_by_ticker: Dict[str, SymbolBacktestRun] = {}
    pending_tickers: List[str] = []
    for run in runs:
        t = run.symbol.ticker
        if t in latest_by_ticker:
            continue
        latest_by_ticker[t] = run
        if run.status in ('pending', 'running'):
            pending_tickers.append(t)

    completed = [r for r in latest_by_ticker.values() if r.status == 'completed']
    symbols = [r.symbol for r in completed]
    tickers = sorted(r.symbol.ticker for r in completed)
    return symbols, tickers, pending_tickers


MAX_MC_PATHS = 2000


def create_portfolio_from_parameter_set(
    strategy: StrategyDefinition,
    parameter_set: SymbolBacktestParameterSet,
    *,
    name: str = '',
    num_monte_carlo_paths: int = 500,
) -> Tuple[Backtest, str]:
    """
    Create and enqueue a portfolio backtest with params locked to the parameter set.

    Returns (backtest, celery_task_id).
    Raises ValueError on validation errors.
    """
    if parameter_set.strategy_id != strategy.id:
        raise ValueError('Parameter set does not belong to this strategy')

    symbols, tickers, pending_tickers = _completed_symbols_for_parameter_set(strategy, parameter_set)
    if pending_tickers:
        raise ValueError(
            f'Some symbols still running: {", ".join(sorted(pending_tickers)[:10])}'
            + ('…' if len(pending_tickers) > 10 else '')
        )
    if len(symbols) < 2:
        raise ValueError(
            'At least 2 completed single-symbol runs are required for a portfolio backtest'
        )

    cfg = _params_from_parameter_set(parameter_set)
    symbol_priority_order = list(tickers)
    mc_paths = max(0, min(int(num_monte_carlo_paths or 0), MAX_MC_PATHS))
    if mc_paths > 0:
        mc_paths, _max_unique = cap_variant_request(len(tickers), mc_paths)

    label = (parameter_set.label or '').strip()
    default_name = name.strip() or (f'{label} — portfolio' if label else f'{strategy.name} — portfolio')

    backtest = Backtest.objects.create(
        name=default_name[:200],
        strategy=strategy,
        strategy_assignment=None,
        broker=cfg['broker'],
        parameter_set=parameter_set,
        symbol_priority_order=symbol_priority_order,
        start_date=cfg['start_date'],
        end_date=cfg['end_date'],
        split_ratio=cfg['split_ratio'],
        initial_capital=cfg['initial_capital'],
        bet_size_percentage=cfg['bet_size_percentage'],
        strategy_parameters=cfg['strategy_parameters'],
        hedge_enabled=cfg['hedge_enabled'],
        run_strategy_only_baseline=cfg['run_strategy_only_baseline'],
        hedge_config=cfg['hedge_config'],
        position_modes=cfg['position_modes'],
        monte_carlo_num_paths=mc_paths,
        status='pending',
    )
    backtest.symbols.set(symbols)

    if mc_paths > 0:
        PortfolioMonteCarloSimulation.objects.create(
            backtest=backtest,
            num_paths=mc_paths,
            reference_symbol_order=symbol_priority_order,
            status='pending',
        )

    task = run_backtest_task.delay(backtest.id)
    logger.info(
        'Portfolio backtest %s from parameter_set %s, task_id=%s',
        backtest.id,
        parameter_set.signature[:10],
        task.id,
    )
    return backtest, task.id


def get_latest_portfolio_for_parameter_set(
    strategy: StrategyDefinition,
    parameter_set: SymbolBacktestParameterSet,
) -> Optional[Backtest]:
    return (
        Backtest.objects.filter(strategy=strategy, parameter_set=parameter_set)
        .order_by('-created_at')
        .first()
    )
