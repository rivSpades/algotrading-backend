"""
Monte Carlo simulation over symbol priority order permutations for portfolio backtests.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest_engine.models import (
    Backtest,
    BacktestStatistics,
    PortfolioMonteCarloPath,
    PortfolioMonteCarloSimulation,
)
from backtest_engine.position_modes import normalize_position_modes
from backtest_engine.services.backtest_executor import BacktestExecutor
from backtest_engine.services.hybrid_vix_hedge import compute_trade_hedge_overlay
from backtest_engine.services.order_permutations import cap_variant_request, variant_symbol_orders
from backtest_engine.tasks import _preprocessed_test_window_bounds

logger = logging.getLogger(__name__)

MAX_PATHS = 2000
HISTOGRAM_BUCKETS = 20
SAMPLE_CURVE_COUNT = 20
SUBSAMPLE_POINTS = 100

PERFORMANCE_METRIC_KEYS = (
    'total_trades',
    'win_rate',
    'total_pnl',
    'profit_factor',
    'sharpe_ratio',
    'cagr',
    'max_drawdown',
    'average_pnl',
    'average_winner',
    'average_loser',
)


def _pick_performance_metrics(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw:
        return {}
    out: Dict[str, Any] = {}
    for key in PERFORMANCE_METRIC_KEYS:
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if key == 'total_trades':
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def _primary_position_mode(backtest: Backtest) -> str:
    modes = normalize_position_modes(backtest.position_modes)
    return 'long' if 'long' in modes else modes[0]


def _load_reference_performance_metrics(backtest: Backtest, position_mode: str) -> Dict[str, Any]:
    stats = BacktestStatistics.objects.filter(backtest=backtest, symbol__isnull=True).first()
    if not stats:
        return {}
    primary_mode = _primary_position_mode(backtest)
    if position_mode == primary_mode:
        raw = {
            'total_trades': stats.total_trades,
            'win_rate': stats.win_rate,
            'total_pnl': stats.total_pnl,
            'profit_factor': stats.profit_factor,
            'sharpe_ratio': stats.sharpe_ratio,
            'cagr': stats.cagr,
            'max_drawdown': stats.max_drawdown,
            'average_pnl': stats.average_pnl,
            'average_winner': stats.average_winner,
            'average_loser': stats.average_loser,
        }
    else:
        raw = (stats.additional_stats or {}).get(position_mode) or {}
    return _pick_performance_metrics(raw)


def _mean_performance_metrics(paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average performance metrics across all runs (Run 0 + order variants)."""
    metrics_list = [
        p.get('performance_metrics') or {}
        for p in paths
        if p.get('performance_metrics')
    ]
    if not metrics_list:
        return {}
    out: Dict[str, Any] = {}
    for key in PERFORMANCE_METRIC_KEYS:
        values = []
        for metrics in metrics_list:
            if key not in metrics or metrics[key] is None:
                continue
            values.append(float(metrics[key]))
        if not values:
            continue
        if key == 'total_trades':
            out[key] = round(float(np.mean(values)), 2)
        elif key in ('total_pnl', 'average_pnl', 'average_winner', 'average_loser'):
            out[key] = round(float(np.mean(values)), 2)
        else:
            out[key] = round(float(np.mean(values)), 4)
    return out


def _ts_key(ts) -> str:
    if hasattr(ts, 'isoformat'):
        return ts.isoformat()
    return str(ts)


def _subsample_curve(curve: List[Tuple], max_points: int = SUBSAMPLE_POINTS) -> List[Dict[str, Any]]:
    if not curve:
        return []
    if len(curve) <= max_points:
        return [{'timestamp': _ts_key(t), 'equity': float(e)} for t, e in curve]
    indices = np.linspace(0, len(curve) - 1, max_points, dtype=int)
    return [
        {'timestamp': _ts_key(curve[i][0]), 'equity': float(curve[i][1])}
        for i in indices
    ]


def _reference_profit_from_backtest(backtest: Backtest) -> Tuple[float, List[Dict[str, Any]]]:
    """Profit and equity curve from the saved portfolio run (Results tab source of truth)."""
    reference_curve = _load_reference_equity_curve(backtest)
    stats = BacktestStatistics.objects.filter(backtest=backtest, symbol__isnull=True).first()
    initial_capital = float(backtest.initial_capital)
    if stats and stats.total_pnl is not None:
        profit = float(stats.total_pnl)
    elif reference_curve:
        profit = float(reference_curve[-1]['equity']) - initial_capital
    else:
        profit = 0.0
    return profit, reference_curve


def _load_reference_equity_curve(backtest: Backtest) -> List[Dict[str, Any]]:
    stats = BacktestStatistics.objects.filter(backtest=backtest, symbol__isnull=True).first()
    if not stats or not stats.equity_curve:
        return []
    out = []
    for pt in stats.equity_curve:
        if isinstance(pt, dict):
            ts = pt.get('timestamp')
            eq = pt.get('equity')
            if ts is not None and eq is not None:
                out.append({'timestamp': str(ts), 'equity': float(eq)})
    return out


def _reference_timestamps(reference_curve: List[Dict[str, Any]]) -> List[str]:
    return [p['timestamp'] for p in reference_curve if p.get('timestamp')]


def _interpolate_equity_at(curve_points: List[Dict[str, Any]], target_ts: str) -> Optional[float]:
    if not curve_points:
        return None
    # Parse timestamps for comparison
    try:
        target = pd.Timestamp(target_ts)
    except Exception:
        return None
    best_eq = None
    best_dt = None
    for pt in curve_points:
        try:
            ts = pd.Timestamp(pt['timestamp'])
        except Exception:
            continue
        if ts <= target and (best_dt is None or ts > best_dt):
            best_dt = ts
            best_eq = float(pt['equity'])
    if best_eq is not None:
        return best_eq
    return float(curve_points[0]['equity']) if curve_points else None


def _build_confidence_bands(
    reference_timestamps: List[str],
    path_curves: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    bands = []
    for ts in reference_timestamps:
        values = []
        for curve in path_curves:
            eq = _interpolate_equity_at(curve, ts)
            if eq is not None:
                values.append(eq)
        if not values:
            continue
        arr = np.array(values, dtype=float)
        bands.append({
            'timestamp': ts,
            'p5': round(float(np.percentile(arr, 5)), 2),
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p50': round(float(np.percentile(arr, 50)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
        })
    return bands


def _build_preprocessed_data(backtest: Backtest, position_mode: str) -> Dict:
    """Load OHLCV + indicators once for all symbols (Monte Carlo Phase 1)."""
    executor = BacktestExecutor(backtest, position_mode=position_mode)
    preprocessed: Dict = {}
    for symbol in executor.symbols:
        df = executor._load_data_for_symbol(symbol)
        if df is None or df.empty:
            continue
        indicators = executor._compute_indicators_for_symbol(symbol, df)
        split_idx = int(len(df) * backtest.split_ratio)
        if split_idx >= len(df):
            split_idx = max(0, len(df) - 1)
        test_df = df.iloc[split_idx:]
        if test_df.empty:
            continue
        price_cache = {}
        _sub = test_df[test_df['close'].notna()]
        for ts, cl in zip(_sub['timestamp'], _sub['close']):
            price_cache[ts] = float(cl)
        sorted_timestamps = sorted(price_cache.keys())
        preprocessed[symbol] = {
            'df': df,
            'indicators': indicators,
            'test_df': test_df,
            'price_cache': price_cache,
            'sorted_timestamps': sorted_timestamps,
        }
    return preprocessed


def _build_hedge_overlay(backtest: Backtest, preprocessed_data: Dict) -> Optional[Dict]:
    if not getattr(backtest, 'hedge_enabled', False) or not preprocessed_data:
        return None
    mn_ts, mx_ts = _preprocessed_test_window_bounds(preprocessed_data)
    if mn_ts is None or mx_ts is None:
        return None
    raw = compute_trade_hedge_overlay(
        mn_ts,
        mx_ts,
        backtest.hedge_config or {},
        yahoo_only=False,
    )
    if raw and not raw.get('error') and raw.get('index_ns'):
        return raw
    return None


def _run_single_path(
    backtest: Backtest,
    preprocessed_data: Dict,
    symbol_order: List[str],
    position_mode: str,
    hedge_overlay: Optional[Dict],
) -> Tuple[float, bool, List[Dict[str, Any]], Dict[str, Any]]:
    executor = BacktestExecutor(
        backtest,
        position_mode=position_mode,
        preprocessed_data=preprocessed_data,
        hedge_overlay=hedge_overlay,
        symbol_priority_order_override=symbol_order,
    )
    executor.execute_strategy()
    final_equity = executor.get_portfolio_final_equity()
    blew_up = bool(executor.portfolio_blew_up or final_equity <= 0)
    curve = []
    if executor.equity_curves:
        raw = next(iter(executor.equity_curves.values()), [])
        curve = _subsample_curve(raw)
    all_stats = executor.calculate_statistics()
    performance_metrics = _pick_performance_metrics(all_stats.get(None))
    return final_equity, blew_up, curve, performance_metrics


def _profit_histogram(profits: List[float], buckets: int = HISTOGRAM_BUCKETS) -> List[Dict[str, Any]]:
    if not profits:
        return []
    arr = np.array(profits, dtype=float)
    counts, edges = np.histogram(arr, bins=buckets)
    return [
        {
            'bin_start': float(edges[i]),
            'bin_end': float(edges[i + 1]),
            'count': int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _path_summary(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'path_index': p['path_index'],
        'symbol_order': p['symbol_order'],
        'profit': round(p['profit'], 2),
        'final_equity': round(p['final_equity'], 2),
        'blew_up': p['blew_up'],
        'is_reference': bool(p.get('is_reference')),
    }


def run_monte_carlo_simulation(
    simulation_id: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> PortfolioMonteCarloSimulation:
    """
    Execute all paths for a PortfolioMonteCarloSimulation row.
    """
    simulation = PortfolioMonteCarloSimulation.objects.select_related('backtest').get(id=simulation_id)
    backtest = simulation.backtest
    simulation.status = 'running'
    simulation.save(update_fields=['status', 'updated_at'])

    try:
        if backtest.status != 'completed':
            raise ValueError('Backtest must be completed before Monte Carlo')
        if not backtest.parameter_set_id:
            raise ValueError('Monte Carlo requires a portfolio backtest linked to a parameter set')

        modes = normalize_position_modes(backtest.position_modes)
        position_mode = 'long' if 'long' in modes else modes[0]

        tickers = list(backtest.symbol_priority_order or [])
        if not tickers:
            tickers = sorted(s.ticker for s in backtest.symbols.all())
        if len(tickers) < 2:
            raise ValueError('At least 2 symbols required for Monte Carlo')

        initial_capital = float(backtest.initial_capital)
        reference_profit, reference_curve = _reference_profit_from_backtest(backtest)
        reference_final_equity = initial_capital + reference_profit
        simulation.reference_equity_curve = reference_curve
        simulation.reference_symbol_order = tickers
        simulation.reference_profit = Decimal(str(round(reference_profit, 2)))
        simulation.save(update_fields=[
            'reference_equity_curve', 'reference_symbol_order', 'reference_profit', 'updated_at',
        ])

        if progress_callback:
            progress_callback(5, 'Pre-processing symbols…')
        preprocessed = _build_preprocessed_data(backtest, position_mode)
        if len(preprocessed) < 2:
            raise ValueError('Insufficient pre-processed symbol data for Monte Carlo')

        hedge_overlay = _build_hedge_overlay(backtest, preprocessed)
        num_variant_paths, max_unique = cap_variant_request(len(tickers), simulation.num_paths)
        simulation.num_paths = num_variant_paths
        variant_orders, _ = variant_symbol_orders(tickers, tickers, num_variant_paths)

        reference_metrics = _load_reference_performance_metrics(backtest, position_mode)

        path_results: List[Dict[str, Any]] = [{
            'path_index': 0,
            'symbol_order': list(tickers),
            'final_equity': reference_final_equity,
            'profit': reference_profit,
            'blew_up': reference_final_equity <= 0,
            'equity_curve': reference_curve,
            'is_reference': True,
            'performance_metrics': reference_metrics,
        }]

        for i, order in enumerate(variant_orders, start=1):
            final_equity, blew_up, equity_curve, performance_metrics = _run_single_path(
                backtest, preprocessed, order, position_mode, hedge_overlay
            )
            profit = final_equity - initial_capital
            path_results.append({
                'path_index': i,
                'symbol_order': order,
                'final_equity': final_equity,
                'profit': profit,
                'blew_up': blew_up,
                'equity_curve': equity_curve,
                'is_reference': False,
                'performance_metrics': performance_metrics,
            })
            if progress_callback and i % max(1, len(variant_orders) // 20) == 0:
                pct = 10 + int((i / max(len(variant_orders), 1)) * 85)
                progress_callback(pct, f'Order variant {i}/{len(variant_orders)}')

        variant_results = [p for p in path_results if not p.get('is_reference')]
        # Aggregate profits / metrics over ALL runs (Run 0 + variants)
        all_profits = [p['profit'] for p in path_results]
        blew_count = sum(1 for p in path_results if p['blew_up'])
        profit_positive = sum(1 for p in path_results if p['profit'] > 0)
        arr = np.array(all_profits, dtype=float) if all_profits else np.array([0.0])

        ref_ts = _reference_timestamps(reference_curve)
        variant_curves = [p['equity_curve'] for p in variant_results]
        confidence_bands = _build_confidence_bands(ref_ts, variant_curves) if ref_ts and variant_curves else []

        # Chart payload: all runs (reference + every unique variant)
        sample_equity_curves = [
            {
                'path_index': p['path_index'],
                'is_reference': bool(p.get('is_reference')),
                'points': p['equity_curve'],
            }
            for p in path_results
        ]

        best = max(path_results, key=lambda p: p['profit']) if path_results else None
        worst = min(path_results, key=lambda p: p['profit']) if path_results else None
        all_n = len(path_results)

        simulation.prob_broke = blew_count / all_n if all_n else 0
        simulation.prob_profit_positive = profit_positive / all_n if all_n else 0
        simulation.mean_profit = Decimal(str(round(float(np.mean(arr)), 2)))
        simulation.median_profit = Decimal(str(round(float(np.median(arr)), 2)))
        simulation.percentile_5 = Decimal(str(round(float(np.percentile(arr, 5)), 2))) if all_n else Decimal('0')
        simulation.percentile_95 = Decimal(str(round(float(np.percentile(arr, 95)), 2))) if all_n else Decimal('0')
        simulation.profit_histogram = _profit_histogram(all_profits)
        simulation.confidence_bands = confidence_bands
        simulation.sample_equity_curves = sample_equity_curves
        simulation.best_path = _path_summary(best) if best else {}
        simulation.worst_path = _path_summary(worst) if worst else {}
        simulation.sample_paths = [_path_summary(p) for p in path_results[:15]]
        simulation.mean_performance_metrics = _mean_performance_metrics(path_results)
        simulation.status = 'completed'
        from django.utils import timezone
        simulation.completed_at = timezone.now()
        simulation.save()

        PortfolioMonteCarloPath.objects.filter(simulation=simulation).delete()
        PortfolioMonteCarloPath.objects.bulk_create([
            PortfolioMonteCarloPath(
                simulation=simulation,
                path_index=p['path_index'],
                symbol_order=p['symbol_order'],
                final_equity=Decimal(str(round(p['final_equity'], 2))),
                profit=Decimal(str(round(p['profit'], 2))),
                blew_up=p['blew_up'],
                is_reference=bool(p.get('is_reference')),
                equity_curve=p.get('equity_curve') or [],
                performance_metrics=p.get('performance_metrics') or {},
            )
            for p in path_results
        ])

        if progress_callback:
            progress_callback(100, 'Order variance simulation complete')
        return simulation

    except Exception as exc:
        logger.exception('Monte Carlo simulation %s failed: %s', simulation_id, exc)
        simulation.status = 'failed'
        simulation.error_message = str(exc)
        simulation.save(update_fields=['status', 'error_message', 'updated_at'])
        raise
