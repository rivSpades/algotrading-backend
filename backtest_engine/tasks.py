"""
Celery tasks for backtest execution
"""

from celery import shared_task
import time
from django.utils import timezone
from django.db import connections
from .models import (
    Backtest,
    Trade,
    BacktestStatistics,
    SymbolBacktestRun,
    SymbolBacktestTrade,
    SymbolBacktestStatistics,
)
from .position_modes import normalize_position_modes
from .services.backtest_executor import BacktestExecutor
from .services.sp500_benchmark import compute_sp500_buy_hold_curve
from .services.hybrid_vix_hedge import compute_trade_hedge_overlay
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import pandas as pd
import os
from datetime import timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)

# Decimal places when persisting Trade fields (must match DecimalField definitions)
_TRADE_FIELD_DECIMAL_PLACES = {
    'entry_price': 8,
    'exit_price': 8,
    'quantity': 8,
    'pnl': 8,
    'pnl_percentage': 4,
    'max_drawdown': 4,
}


def _trade_pnl_from_stored_prices(trade_type: str, entry_price, exit_price, quantity):
    """
    PnL and return % from the same rounded prices stored on Trade, so
    pnl ≈ (exit - entry) * quantity (long) / (entry - exit) * quantity (short) within quantize error.
    """
    d_ep = Decimal(str(entry_price))
    d_xp = Decimal(str(exit_price))
    d_qty = Decimal(str(quantity))
    if trade_type == 'buy':
        pnl_d = (d_xp - d_ep) * d_qty
        pct_d = ((d_xp - d_ep) / d_ep) * Decimal(100) if d_ep != 0 else Decimal(0)
    else:
        pnl_d = (d_ep - d_xp) * d_qty
        pct_d = ((d_ep - d_xp) / d_ep) * Decimal(100) if d_ep != 0 else Decimal(0)
    q8 = Decimal('0.00000001')
    q4 = Decimal('0.0001')
    return float(pnl_d.quantize(q8)), float(pct_d.quantize(q4))


def _preprocessed_test_window_bounds(preprocessed_data):
    """Min/max timestamps across all symbols' test_df (for hedge overlay range)."""
    mn_dt, mx_dt = None, None
    for _sym, pack in (preprocessed_data or {}).items():
        td = pack.get('test_df')
        if td is None or len(td) == 0:
            continue
        t0 = td.iloc[0]['timestamp']
        t1 = td.iloc[-1]['timestamp']
        if mn_dt is None or pd.Timestamp(t0) < pd.Timestamp(mn_dt):
            mn_dt = t0
        if mx_dt is None or pd.Timestamp(t1) > pd.Timestamp(mx_dt):
            mx_dt = t1
    return mn_dt, mx_dt


def _strategy_only_snapshot_from_stats(b):
    """Build additional_stats.strategy_only[mode] payload from portfolio or symbol stats dict (baseline run)."""
    if not b:
        return None
    return {
        'equity_curve': b.get('equity_curve') or [],
        'total_trades': b.get('total_trades', 0),
        'winning_trades': b.get('winning_trades', 0),
        'losing_trades': b.get('losing_trades', 0),
        'win_rate': b.get('win_rate', 0),
        'total_pnl': b.get('total_pnl', 0),
        'total_pnl_percentage': b.get('total_pnl_percentage', 0),
        'average_pnl': b.get('average_pnl', 0),
        'average_winner': b.get('average_winner', 0),
        'average_loser': b.get('average_loser', 0),
        'profit_factor': b.get('profit_factor', 0),
        'max_drawdown': b.get('max_drawdown', 0),
        'max_drawdown_duration': b.get('max_drawdown_duration', 0),
        'sharpe_ratio': b.get('sharpe_ratio', 0),
        'cagr': b.get('cagr', 0),
        'total_return': b.get('total_return', 0),
        'skipped_trades_count': b.get('skipped_trades_count', 0),
    }


def _position_modes_to_run(backtest) -> list:
    """Ordered list of modes to execute from Backtest.position_modes."""
    return normalize_position_modes(getattr(backtest, 'position_modes', None))


def _primary_secondary_modes(modes_list: list) -> tuple:
    """Primary DB row is long when long ran, else short. Secondary is the other side."""
    s = set(modes_list)
    primary = 'long' if 'long' in s else 'short'
    secondary = 'short' if primary == 'long' else 'long'
    return primary, secondary


@shared_task(bind=True, name='backtest_engine.run_backtest', time_limit=48 * 60 * 60, soft_time_limit=48 * 60 * 60)
def run_backtest_task(self, backtest_id):
    """
    Execute a backtest asynchronously
    
    Args:
        backtest_id: ID of the Backtest instance to execute
    """
    try:
        # Get backtest
        backtest = Backtest.objects.get(id=backtest_id)
        backtest.status = 'running'
        backtest.save()
        
        # Update task name for better visibility in active tasks
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': 0,
                'message': f'Starting backtest: {backtest.strategy.name}',
                'task_name': f'Backtest: {backtest.strategy.name}'
            }
        )
        
        logger.info(f"Starting backtest {backtest_id}: {backtest.strategy.name}")
        
        # Update progress: 10% - Starting
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'message': 'Initializing backtest...'}
        )
        
        # Phase 1: Pre-process all symbols in parallel (ONCE, before long + short runs)
        # Load data and compute indicators for all symbols that will be used
        logger.info("Phase 1: Pre-processing all symbols (load data -> compute indicators)")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 12, 'message': 'Pre-processing symbols...'}
        )
        
        # Get symbols from the backtest (already filtered by user selection: specific symbols, random selection, or "select all")
        from live_trading.models import SymbolBrokerAssociation
        from django.db.models import Q
        
        all_symbols = list(backtest.symbols.all())
        
        if backtest.broker:
            # Filter to only symbols from backtest.symbols that are:
            # - Active status
            # - Linked to the broker with at least one trading capability (long_active or short_active)
            # Symbol model uses 'ticker' as primary key, not 'id'
            symbol_tickers = [s.ticker for s in all_symbols]
            associations = SymbolBrokerAssociation.objects.filter(
                broker=backtest.broker,
                symbol__ticker__in=symbol_tickers,
                symbol__status='active'
            ).filter(
                Q(long_active=True) | Q(short_active=True)
            )
            symbols_to_preprocess = [assoc.symbol for assoc in associations.select_related('symbol')]
        else:
            # No broker - filter by status only from the selected symbols
            symbols_to_preprocess = [s for s in all_symbols if s.status == 'active']
        
        logger.info(f"Phase 1: Pre-processing {len(symbols_to_preprocess)} symbols in parallel")

        # Date-bounded OHLCV loading to prevent timeouts on large symbol sets.
        # We still load a small warmup window so indicators have enough lookback.
        merged_strategy_params = {}
        if getattr(backtest.strategy, "default_parameters", None):
            merged_strategy_params.update(backtest.strategy.default_parameters or {})
        merged_strategy_params.update(backtest.strategy_parameters or {})

        warmup_days = 0
        required_tool_configs = backtest.strategy.required_tool_configs or []
        for tool_config in required_tool_configs:
            parameters = (tool_config.get('parameters') or {}).copy()
            parameter_mapping = tool_config.get('parameter_mapping') or {}
            for tool_param, strategy_param in parameter_mapping.items():
                if merged_strategy_params.get(strategy_param) is not None:
                    parameters[tool_param] = merged_strategy_params[strategy_param]

            for k, v in parameters.items():
                if isinstance(v, (int, float)) and ('period' in str(k).lower() or str(k).lower() == 'period'):
                    warmup_days = max(warmup_days, int(v))

        # Add a small buffer for rolling indicators and ensure warmup_days >= 1 when warmup exists
        warmup_days = max(0, warmup_days) + (5 if warmup_days > 0 else 0)
        date_window_start = backtest.start_date - timedelta(days=warmup_days)
        date_window_end = backtest.end_date
        
        # Pre-process in parallel
        preprocessed_data = {}  # {symbol: {'df': DataFrame, 'indicators': dict, 'test_df': DataFrame, 'price_cache': dict}}
        preprocessed_lock = threading.Lock()
        
        def preprocess_symbol(symbol, bulk_ohlcv_by_symbol_local):
            """Load data and compute indicators for a single symbol (Phase 1)"""
            try:
                logger.info(f"Pre-processing symbol: {symbol.ticker}")
                
                # OPTIMIZATION: Use bulk-loaded OHLCV data instead of individual query
                if symbol not in bulk_ohlcv_by_symbol_local:
                    logger.warning(f"No OHLCV data found for symbol {symbol.ticker} in bulk load")
                    return symbol, None
                
                ohlcv_list = bulk_ohlcv_by_symbol_local[symbol]
                
                if not ohlcv_list:
                    return symbol, None
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv_list)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                if df.empty:
                    return symbol, None
                
                # Compute indicators directly (no executor needed)
                from market_data.services.indicator_service import compute_strategy_indicators_for_ohlcv
                ohlcv_list_for_indicators = df.to_dict('records')
                indicator_values = compute_strategy_indicators_for_ohlcv(
                    backtest.strategy, ohlcv_list_for_indicators, symbol, strategy_parameters=backtest.strategy_parameters
                )
                
                # Convert indicator values to pandas Series aligned with DataFrame index
                indicators = {}
                for indicator_key, indicator_data in indicator_values.items():
                    if isinstance(indicator_data, dict) and 'values' in indicator_data:
                        values = indicator_data['values']
                        if isinstance(values, list):
                            if len(values) < len(df):
                                values.extend([None] * (len(df) - len(values)))
                            elif len(values) > len(df):
                                values = values[:len(df)]
                            series = pd.Series(values, index=df.index[:len(values)])
                            indicators[indicator_key] = series
                            # Also add base name for backward compatibility
                            base_name = indicator_key.split('_')[0]
                            if base_name not in indicators:
                                indicators[base_name] = series
                    elif isinstance(indicator_data, list):
                        values = indicator_data
                        if len(values) < len(df):
                            values.extend([None] * (len(df) - len(values)))
                        elif len(values) > len(df):
                            values = values[:len(df)]
                        series = pd.Series(values, index=df.index[:len(values)])
                        indicators[indicator_key] = series
                        base_name = indicator_key.split('_')[0]
                        if base_name not in indicators:
                            indicators[base_name] = series
                    elif isinstance(indicator_data, pd.Series):
                        indicators[indicator_key] = indicator_data
                
                # Split data
                split_idx = int(len(df) * backtest.split_ratio)
                if split_idx >= len(df):
                    split_idx = max(0, len(df) - 1)
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]
                
                if test_df.empty:
                    return symbol, None
                
                # Build price cache for fast lookups
                price_cache = {}
                for idx, row in test_df.iterrows():
                    timestamp = row['timestamp']
                    if pd.notna(row['close']):
                        price_cache[timestamp] = float(row['close'])
                
                with preprocessed_lock:
                    preprocessed_data[symbol] = {
                        'df': df,
                        'indicators': indicators,
                        'test_df': test_df,
                        'price_cache': price_cache
                    }
                
                return symbol, True
            except Exception as e:
                logger.error(f"Error pre-processing {symbol.ticker}: {str(e)}")
                connections.close_all()
                return symbol, None
            finally:
                # Always close connections in thread to prevent connection leaks
                connections.close_all()
        
        # OPTIMIZATION: Increase thread pool size based on CPU cores and symbol count
        # For I/O-bound tasks (database queries, indicator computation), use 2-3x CPU cores
        cpu_count = os.cpu_count() or 4
        # Scale workers based on symbol count and available CPU cores
        # Cap at 50 to prevent excessive resource usage, but allow more than 10 for large jobs
        base_workers = max(cpu_count * 3, 20)  # At least 20 workers if 3x CPU cores is less
        max_workers = min(len(symbols_to_preprocess), base_workers, 50)
        
        logger.info(f"Using {max_workers} worker threads for preprocessing (CPU cores: {cpu_count}, symbols: {len(symbols_to_preprocess)})")
        
        # OPTIMIZATION: Batch processing for very large symbol sets to prevent memory issues
        BATCH_SIZE = 1000  # Process 1000 symbols at a time
        all_preprocessed_data = {}
        
        if symbols_to_preprocess:
            # Process in batches if we have a very large number of symbols
            if len(symbols_to_preprocess) > BATCH_SIZE:
                logger.info(f"Processing {len(symbols_to_preprocess)} symbols in batches of {BATCH_SIZE}")
                
                for batch_start in range(0, len(symbols_to_preprocess), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(symbols_to_preprocess))
                    batch_symbols = symbols_to_preprocess[batch_start:batch_end]
                    batch_num = (batch_start // BATCH_SIZE) + 1
                    total_batches = (len(symbols_to_preprocess) + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches}: symbols {batch_start} to {batch_end-1}")

                    # Load OHLCV only for this batch, bounded by the backtest window (+ warmup)
                    from market_data.models import OHLCV
                    batch_tickers = [s.ticker for s in batch_symbols]
                    ticker_to_symbol = {s.ticker: s for s in batch_symbols}
                    bulk_ohlcv_queryset = OHLCV.objects.filter(
                        symbol__ticker__in=batch_tickers,
                        timeframe='daily',
                        timestamp__gte=date_window_start,
                        timestamp__lte=date_window_end,
                    ).select_related('symbol').order_by('symbol__ticker', 'timestamp')

                    bulk_ohlcv_data = {}
                    values_qs = bulk_ohlcv_queryset.values(
                        'symbol__ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume'
                    ).iterator(chunk_size=50000)
                    for ohlcv in values_qs:
                        ticker = ohlcv['symbol__ticker']
                        if ticker not in bulk_ohlcv_data:
                            bulk_ohlcv_data[ticker] = []

                        timestamp = ohlcv['timestamp']
                        if hasattr(timestamp, 'isoformat'):
                            timestamp = timestamp.isoformat()

                        bulk_ohlcv_data[ticker].append({
                            'timestamp': timestamp,
                            'open': float(ohlcv['open']),
                            'high': float(ohlcv['high']),
                            'low': float(ohlcv['low']),
                            'close': float(ohlcv['close']),
                            'volume': float(ohlcv['volume'])
                        })

                    bulk_ohlcv_by_symbol_local = {}
                    for ticker, data in bulk_ohlcv_data.items():
                        if ticker in ticker_to_symbol:
                            bulk_ohlcv_by_symbol_local[ticker_to_symbol[ticker]] = data
                    
                    # Process batch in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(preprocess_symbol, symbol, bulk_ohlcv_by_symbol_local): symbol
                            for symbol in batch_symbols
                        }
                        for future in as_completed(futures):
                            future.result()  # Raise exception if any occurred
                    
                    # Merge batch results
                    all_preprocessed_data.update(preprocessed_data)
                    preprocessed_data = {}  # Reset for next batch
                    
                    # Update progress
                    progress = 12 + int((batch_end / len(symbols_to_preprocess)) * 6)  # 12-18% for Phase 1
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': progress, 'message': f'Pre-processed batch {batch_num}/{total_batches} ({batch_end}/{len(symbols_to_preprocess)} symbols)'}
                    )
                
                # Use accumulated preprocessed data
                preprocessed_data = all_preprocessed_data
            else:
                # Process all symbols at once for smaller sets
                from market_data.models import OHLCV
                symbol_tickers = [s.ticker for s in symbols_to_preprocess]
                ticker_to_symbol = {s.ticker: s for s in symbols_to_preprocess}

                bulk_ohlcv_queryset = OHLCV.objects.filter(
                    symbol__ticker__in=symbol_tickers,
                    timeframe='daily',
                    timestamp__gte=date_window_start,
                    timestamp__lte=date_window_end,
                ).select_related('symbol').order_by('symbol__ticker', 'timestamp')

                bulk_ohlcv_data = {}
                values_qs = bulk_ohlcv_queryset.values(
                    'symbol__ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ).iterator(chunk_size=50000)
                for ohlcv in values_qs:
                    ticker = ohlcv['symbol__ticker']
                    if ticker not in bulk_ohlcv_data:
                        bulk_ohlcv_data[ticker] = []

                    timestamp = ohlcv['timestamp']
                    if hasattr(timestamp, 'isoformat'):
                        timestamp = timestamp.isoformat()

                    bulk_ohlcv_data[ticker].append({
                        'timestamp': timestamp,
                        'open': float(ohlcv['open']),
                        'high': float(ohlcv['high']),
                        'low': float(ohlcv['low']),
                        'close': float(ohlcv['close']),
                        'volume': float(ohlcv['volume'])
                    })

                bulk_ohlcv_by_symbol_local = {}
                for ticker, data in bulk_ohlcv_data.items():
                    if ticker in ticker_to_symbol:
                        bulk_ohlcv_by_symbol_local[ticker_to_symbol[ticker]] = data

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(preprocess_symbol, symbol, bulk_ohlcv_by_symbol_local): symbol
                        for symbol in symbols_to_preprocess
                    }
                    for future in as_completed(futures):
                        future.result()  # Raise exception if any occurred
        
        logger.info(f"Phase 1 complete: Pre-processed {len(preprocessed_data)} symbols")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 18, 'message': f'Pre-processed {len(preprocessed_data)} symbols'}
        )

        hedge_overlay = None
        if getattr(backtest, 'hedge_enabled', False) and preprocessed_data:
            mn_ts, mx_ts = _preprocessed_test_window_bounds(preprocessed_data)
            if mn_ts is not None and mx_ts is not None:
                raw_overlay = compute_trade_hedge_overlay(
                    mn_ts,
                    mx_ts,
                    backtest.hedge_config or {},
                    yahoo_only=False,
                )
                if raw_overlay and not raw_overlay.get('error') and raw_overlay.get('index_ns'):
                    hedge_overlay = raw_overlay
                    logger.info(
                        'Built integrated hedge overlay: %s trading days (data_source=%s)',
                        len(hedge_overlay['index_ns']),
                        hedge_overlay.get('data_source', 'unknown'),
                    )
                else:
                    logger.warning(
                        'Backtest %s: hedge_enabled but overlay unavailable (%s); running without hedge split',
                        backtest_id,
                        raw_overlay.get('error') if raw_overlay else 'no overlay',
                    )
        
        position_modes_to_run = _position_modes_to_run(backtest)
        primary_mode, secondary_mode = _primary_secondary_modes(position_modes_to_run)
        num_modes = len(position_modes_to_run)

        # Phase 2: Execute strategy for each selected position mode (pre-processed data from Phase 1)
        mode_label = ' and '.join(m.upper() for m in position_modes_to_run)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'message': f'Executing strategy ({mode_label})...'}
        )
        
        all_trades_by_mode = {}
        all_statistics = {}
        all_baseline_stats_by_mode = {}
        
        def execute_position_mode(position_mode):
            """Execute strategy for a single position mode (thread-safe helper)
            
            When hedge_overlay is set, runs twice: baseline (no hedge split) then hedged.
            Trades persisted are from the hedged run only.
            
            Note: Cannot call self.update_state() from within threads.
            State updates must be done from the main task thread.
            """
            try:
                logger.info(f"[{position_mode.upper()}] Starting strategy execution (Phase 2)")
                
                baseline_full_stats = None
                if hedge_overlay and getattr(backtest, 'run_strategy_only_baseline', True):
                    logger.info(f"[{position_mode.upper()}] Baseline run (strategy only, no hedge split)...")
                    ex_base = BacktestExecutor(
                        backtest,
                        position_mode=position_mode,
                        preprocessed_data=preprocessed_data,
                        hedge_overlay=None,
                    )
                    ex_base.execute_strategy()
                    stats_base = ex_base.calculate_statistics()
                    baseline_full_stats = stats_base
                    bp = stats_base.get(None) or {}
                    logger.info(
                        f"[{position_mode.upper()}] Baseline: {len(ex_base.trades)} trades, "
                        f"portfolio trades={bp.get('total_trades', 0)}"
                    )
                
                mode_executor = BacktestExecutor(
                    backtest,
                    position_mode=position_mode,
                    preprocessed_data=preprocessed_data,
                    hedge_overlay=hedge_overlay,
                )
                
                logger.info(
                    f"[{position_mode.upper()}] Executing strategy logic ({'with hedge split' if hedge_overlay else 'no hedge'})..."
                )
                mode_executor.execute_strategy()
                
                buy_trades = [t for t in mode_executor.trades if t.get('trade_type') == 'buy']
                sell_trades = [t for t in mode_executor.trades if t.get('trade_type') == 'sell']
                logger.info(f"[{position_mode.upper()}] Generated {len(mode_executor.trades)} trades: {len(buy_trades)} buy, {len(sell_trades)} sell")
                
                if not mode_executor.trades:
                    logger.warning(f"[{position_mode.upper()}] No trades generated for backtest {backtest_id}")
                
                logger.info(f"[{position_mode.upper()}] Calculating statistics...")
                mode_stats = mode_executor.calculate_statistics()
                
                logger.info(f"[{position_mode.upper()}] Completed strategy execution")
                
                return position_mode, mode_executor.trades, mode_stats, baseline_full_stats
            except Exception as e:
                logger.error(f"[{position_mode.upper()}] Error executing strategy: {str(e)}")
                # Close database connection in this thread
                connections.close_all()
                raise
            finally:
                # Always close connections in thread to prevent connection leaks
                connections.close_all()
        
        logger.info(
            "Starting execution for position mode(s): %s",
            ", ".join(position_modes_to_run),
        )
        mode_stats_results = {}
        results_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=max(1, num_modes)) as executor_pool:
            future_to_mode = {
                executor_pool.submit(execute_position_mode, mode): mode 
                for mode in position_modes_to_run
            }
            
            # Update progress as each mode completes (from main thread)
            completed_count = 0
            total_modes = num_modes
            
            # Collect results as they complete
            for future in as_completed(future_to_mode):
                position_mode = future_to_mode[future]
                try:
                    mode_result, trades, stats, baseline_full = future.result()
                    with results_lock:
                        all_trades_by_mode[mode_result] = trades
                        mode_stats_results[mode_result] = stats
                        if baseline_full:
                            all_baseline_stats_by_mode[mode_result] = baseline_full
                    completed_count += 1
                    
                    # Update progress from main thread (20% to 70% for execution)
                    progress = 20 + int((completed_count / total_modes) * 50)
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': progress, 'message': f'Completed {completed_count}/{total_modes} position modes...'}
                    )
                    
                    logger.info(f"[{position_mode.upper()}] Results collected successfully ({completed_count}/{total_modes})")
                except Exception as e:
                    logger.error(f"[{position_mode.upper()}] Exception in strategy execution: {str(e)}")
                    # Close database connections in case of error
                    connections.close_all()
                    raise
        
        # Close database connections after parallel execution
        connections.close_all()
        
        # Update progress: 70% - All execution modes completed
        self.update_state(
            state='PROGRESS',
            meta={'progress': 70, 'message': 'Strategy execution completed'}
        )
        
        # Process statistics from each executed mode
        for position_mode in position_modes_to_run:
            mode_stats = mode_stats_results.get(position_mode, {})
            
            # Store portfolio-level statistics
            if None in mode_stats:
                all_statistics[None] = all_statistics.get(None, {})
                all_statistics[None][position_mode] = mode_stats[None]
            # Intentionally skip symbol-level statistics for portfolio backtests.
            # Single-symbol runs have their own storage (SymbolBacktestStatistics).
        
        # Update progress: 70% - Calculating statistics
        self.update_state(
            state='PROGRESS',
            meta={'progress': 70, 'message': 'Calculating statistics...'}
        )
        
        # Combine portfolio stats into the expected structure
        if None in all_statistics:
            portfolio_stats = all_statistics[None]
            if 'long' in portfolio_stats or 'short' in portfolio_stats:
                pass
            else:
                all_statistics[None] = {
                    'long': portfolio_stats.get('long', portfolio_stats),
                    'short': portfolio_stats.get('short', {}),
                }
        
        statistics = all_statistics
        
        # Update progress: 80% - Saving results
        self.update_state(
            state='PROGRESS',
            meta={'progress': 80, 'message': 'Saving trades and statistics...'}
        )
        
        # Save trades from each executed mode; metadata.position_mode for filtering
        for position_mode in position_modes_to_run:
            mode_trades = all_trades_by_mode.get(position_mode, [])
            logger.info(f"Saving {len(mode_trades)} trades for position_mode={position_mode}")
            
            # Count buy vs sell trades for logging
            buy_count = sum(1 for t in mode_trades if t.get('trade_type') == 'buy')
            sell_count = sum(1 for t in mode_trades if t.get('trade_type') == 'sell')
            
            # Count completed positions (entry + exit) - a long requires buy with exit_timestamp, a short requires sell with exit_timestamp
            long_count = sum(1 for t in mode_trades if t.get('trade_type') == 'buy' and t.get('exit_timestamp') is not None)
            short_count = sum(1 for t in mode_trades if t.get('trade_type') == 'sell' and t.get('exit_timestamp') is not None)
            open_long_count = sum(1 for t in mode_trades if t.get('trade_type') == 'buy' and t.get('exit_timestamp') is None)
            open_short_count = sum(1 for t in mode_trades if t.get('trade_type') == 'sell' and t.get('exit_timestamp') is None)
            
            logger.info(f"  - Buy trades: {buy_count}, Sell trades: {sell_count}")
            logger.info(f"  - Completed LONG positions: {long_count}, Completed SHORT positions: {short_count}")
            if open_long_count > 0 or open_short_count > 0:
                logger.info(f"  - Open positions: {open_long_count} long, {open_short_count} short")
            
            # Validate: Strategy should generate at least 2 trades to be considered successful
            # Less than 2 trades indicates potential issues:
            # - 0 trades: No signals generated (possible indicator/computation issue)
            # - 1 trade: Only one opportunity detected (possible data/split ratio issue)
            if len(mode_trades) < 2:
                logger.warning(
                    f"⚠️ WARNING: Only {len(mode_trades)} trade(s) generated for position_mode={position_mode}. "
                    f"This may indicate: "
                    f"{'No signals generated - check indicator computation' if len(mode_trades) == 0 else 'Only one opportunity - check data period/split ratio'}. "
                    f"Strategy execution may need investigation."
                )
            
            for trade_data in mode_trades:
                # Add position_mode to metadata
                trade_metadata = trade_data.get('metadata', {})
                if not isinstance(trade_metadata, dict):
                    trade_metadata = {}
                trade_metadata['position_mode'] = position_mode
                
                # Get symbol ticker for error logging
                symbol_ticker = 'unknown'
                if 'symbol' in trade_data and trade_data['symbol']:
                    if hasattr(trade_data['symbol'], 'ticker'):
                        symbol_ticker = trade_data['symbol'].ticker
                    elif isinstance(trade_data['symbol'], str):
                        symbol_ticker = trade_data['symbol']
                
                # Convert numpy types to Python native types to avoid Django field validation errors
                def convert_value(value, field_name=None):
                    """Convert numpy/pandas types to Python float, round per field (8dp money, 4dp %), validate limits.
                    
                    For DecimalField(max_digits=20, decimal_places=8), values must be < 10^12.
                    If value exceeds limit, raises ValueError (indicates bad data/calculation error).
                    """
                    if value is None:
                        return None
                    import numpy as np
                    float_value = None
                    
                    # Handle numpy arrays first (before checking other numpy types)
                    if isinstance(value, np.ndarray):
                        # For arrays, return None or handle appropriately
                        if value.size == 0:
                            return None
                        if value.size == 1:
                            float_value = float(value.item())
                        else:
                            # Multiple values - this shouldn't happen for a single field
                            return None
                    # Handle numpy scalars
                    elif isinstance(value, (np.integer, np.floating)):
                        float_value = float(value)
                    # Handle numpy types that have .item() method
                    elif hasattr(value, 'item') and not isinstance(value, (str, list, dict)):
                        try:
                            float_value = float(value.item())
                        except (ValueError, AttributeError):
                            pass
                    # Handle pandas types
                    elif hasattr(value, 'iloc'):  # pandas Series/DataFrame
                        try:
                            float_value = float(value)
                        except (ValueError, TypeError):
                            pass
                    # If it's already a Python native type, convert to float
                    elif isinstance(value, (int, float, str)):
                        try:
                            float_value = float(value)
                        except (ValueError, TypeError):
                            pass
                    
                    # Try to convert to float as last resort
                    if float_value is None:
                        try:
                            float_value = float(value)
                        except (ValueError, TypeError):
                            # If all else fails, return None (safer than passing invalid type)
                            logger.warning(f"Could not convert value {type(value)} to Python native type, using None")
                            return None
                    
                    places = _TRADE_FIELD_DECIMAL_PLACES.get(field_name, 2)
                    rounded_value = round(float_value, places)
                    
                    # Validate database limits for DecimalField(max_digits=20, decimal_places=8)
                    # The database requires values to be strictly less than 10^12 (absolute value)
                    # If value exceeds limit, this indicates bad data or calculation error - raise error
                    if rounded_value >= 10**12:
                        error_msg = (
                            f"Value {rounded_value} for field '{field_name}' exceeds database limit (10^12). "
                            f"This likely indicates bad data or a calculation error. "
                            f"Symbol: {symbol_ticker}, "
                            f"entry_price={trade_data.get('entry_price')}, exit_price={trade_data.get('exit_price')}, "
                            f"quantity={trade_data.get('quantity')}, pnl={trade_data.get('pnl')}, "
                            f"bet_amount={trade_metadata.get('bet_amount')}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    elif rounded_value <= -(10**12):
                        error_msg = (
                            f"Value {rounded_value} for field '{field_name}' exceeds database limit (-10^12). "
                            f"This likely indicates bad data or a calculation error. "
                            f"Symbol: {symbol_ticker}, "
                            f"entry_price={trade_data.get('entry_price')}, exit_price={trade_data.get('exit_price')}, "
                            f"quantity={trade_data.get('quantity')}, pnl={trade_data.get('pnl')}, "
                            f"bet_amount={trade_metadata.get('bet_amount')}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    return rounded_value
                
                entry_price = convert_value(trade_data['entry_price'], 'entry_price')
                exit_price = convert_value(trade_data.get('exit_price'), 'exit_price')
                quantity = convert_value(trade_data['quantity'], 'quantity')
                trade_type = trade_data['trade_type']
                # Integrated hedge: executor sets strategy_pnl + hedge_pnl on exit metadata; keep stored pnl consistent
                if (
                    exit_price is not None
                    and entry_price is not None
                    and quantity is not None
                    and trade_metadata.get('strategy_pnl') is None
                ):
                    pnl, pnl_percentage = _trade_pnl_from_stored_prices(trade_type, entry_price, exit_price, quantity)
                    is_winner = pnl > 0
                else:
                    pnl = convert_value(trade_data.get('pnl'), 'pnl')
                    pnl_percentage = convert_value(trade_data.get('pnl_percentage'), 'pnl_percentage')
                    is_winner = (pnl > 0) if pnl is not None else trade_data.get('is_winner')
                
                Trade.objects.create(
                    backtest=backtest,
                    symbol=trade_data['symbol'],
                    trade_type=trade_type,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_timestamp=trade_data['entry_timestamp'],
                    exit_timestamp=trade_data.get('exit_timestamp'),
                    quantity=quantity,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    is_winner=is_winner,
                    max_drawdown=convert_value(trade_data.get('max_drawdown'), 'max_drawdown'),
                    metadata=trade_metadata
                )
        
        # Helper function to round and validate statistics values before saving
        def round_and_validate_stat(value, field_name=None, decimal_places=2, symbol=None):
            """Round value to specified decimal places and validate database limits
            
            If value exceeds database limit, raises ValueError (indicates bad data/calculation error)
            
            Fields with max_digits=10, decimal_places=4 (percentage fields) have limit of 10^6
            Fields with max_digits=20, decimal_places=8 (PnL fields) have limit of 10^12
            """
            if value is None:
                return None
            
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {field_name} value {value} to float, using None")
                return None
            
            # Round to specified decimal places
            rounded_value = round(float_value, decimal_places)
            
            # Determine field type and validation limit based on field name
            # Percentage fields (max_digits=10, decimal_places=4): limit is 10^6
            percentage_fields = ['win_rate', 'total_pnl_percentage', 'profit_factor', 'max_drawdown',
                               'avg_intra_trade_drawdown', 'worst_intra_trade_drawdown',
                               'sharpe_ratio', 'cagr', 'total_return']
            
            if field_name in percentage_fields:
                # Validate database limits for DecimalField(max_digits=10, decimal_places=4)
                # The database requires values to be strictly less than 10^6 (absolute value)
                max_value = 10**6
            else:
                # Validate database limits for DecimalField(max_digits=20, decimal_places=8)
                # The database requires values to be strictly less than 10^12 (absolute value)
                max_value = 10**12
            
            # If value exceeds limit, this indicates bad data or calculation error - raise error
            if abs(rounded_value) >= max_value:
                symbol_info = f"Symbol: {symbol.ticker if symbol and hasattr(symbol, 'ticker') else 'portfolio-level'}"
                error_msg = (
                    f"Value {rounded_value} for field '{field_name}' exceeds database limit ({max_value}). "
                    f"This likely indicates bad data or a calculation error. "
                    f"{symbol_info}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return rounded_value
        
        # Save statistics (primary mode on main row; secondary mode nested in additional_stats)
        for symbol, stats in statistics.items():
            if symbol is None:
                portfolio_primary = stats.get(primary_mode, {}) or {}
                if not portfolio_primary or not portfolio_primary.get('total_trades', 0):
                    portfolio_primary = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0,
                        'total_pnl': 0,
                        'total_pnl_percentage': 0,
                        'average_pnl': 0,
                        'average_winner': 0,
                        'average_loser': 0,
                        'profit_factor': 0,
                        'max_drawdown': 0,
                        'max_drawdown_duration': 0,
                        'avg_intra_trade_drawdown': 0,
                        'worst_intra_trade_drawdown': 0,
                        'sharpe_ratio': 0,
                        'cagr': 0,
                        'total_return': 0,
                        'equity_curve': [],
                    }
                
                # Prepare statistics with rounded and validated values
                stats_dict = {k: v for k, v in portfolio_primary.items() if k not in ['additional_stats', 'equity_curve', 'skipped_trades_count', 'independent_bet_amounts']}
                
                # Round and validate numeric fields
                numeric_fields = ['total_pnl', 'total_pnl_percentage', 'average_pnl', 'average_winner', 
                                'average_loser', 'profit_factor', 'max_drawdown', 'avg_intra_trade_drawdown',
                                'worst_intra_trade_drawdown', 'sharpe_ratio', 
                                'cagr', 'total_return', 'win_rate']
                
                for field in numeric_fields:
                    if field in stats_dict:
                        stats_dict[field] = round_and_validate_stat(stats_dict[field], field, decimal_places=2, symbol=None)
                
                skipped_trades_count = portfolio_primary.get('skipped_trades_count', 0)
                
                benchmark_block = {}
                eq_curve = portfolio_primary.get('equity_curve') or []
                if eq_curve and isinstance(eq_curve, list):
                    try:
                        first_pt = eq_curve[0]
                        last_pt = eq_curve[-1]
                        first_ts = first_pt.get('timestamp')
                        last_ts = last_pt.get('timestamp')
                        start_cap = float(
                            first_pt.get('equity', backtest.initial_capital) or backtest.initial_capital
                        )
                        b_curve, b_meta = compute_sp500_buy_hold_curve(first_ts, last_ts, start_cap)
                        benchmark_block = {
                            'ticker': b_meta.get('ticker', '^GSPC'),
                            'equity_curve': b_curve,
                            'source': b_meta.get('source', 'none'),
                        }
                        if b_meta.get('error'):
                            benchmark_block['error'] = b_meta['error']
                    except Exception as bench_ex:
                        logger.warning(
                            'S&P 500 benchmark curve failed for backtest %s: %s',
                            backtest_id,
                            bench_ex,
                            exc_info=True,
                        )
                        benchmark_block = {
                            'ticker': '^GSPC',
                            'equity_curve': [],
                            'error': str(bench_ex),
                            'source': 'none',
                        }
                
                portfolio_secondary = stats.get(secondary_mode, {}) or {}
                extra_stats = {
                    primary_mode: {
                        'skipped_trades_count': skipped_trades_count,
                    },
                    secondary_mode: portfolio_secondary if portfolio_secondary else {},
                }
                if benchmark_block:
                    extra_stats['benchmark'] = benchmark_block

                strategy_only_payload = {}
                for pm in ('long', 'short'):
                    bfull = all_baseline_stats_by_mode.get(pm) or {}
                    b = bfull.get(None) or {}
                    if not b:
                        continue
                    snap = _strategy_only_snapshot_from_stats(b)
                    if snap:
                        strategy_only_payload[pm] = snap
                if strategy_only_payload:
                    extra_stats['strategy_only'] = strategy_only_payload
                
                BacktestStatistics.objects.create(
                    backtest=backtest,
                    symbol=None,
                    **stats_dict,
                    equity_curve=portfolio_primary.get('equity_curve', []),
                    additional_stats=extra_stats,
                )
            else:
                # Skip symbol-level BacktestStatistics for portfolio backtests.
                continue
        
        # Mark as completed
        backtest.status = 'completed'
        backtest.completed_at = timezone.now()
        backtest.save()

        logger.info(f"Backtest {backtest_id} completed successfully")
        
        # Calculate total trades count from all modes
        total_trades_count = sum(len(trades) for trades in all_trades_by_mode.values())
        
        # Update progress: 100% - Completed
        self.update_state(
            state='SUCCESS',
            meta={'progress': 100, 'message': 'Backtest completed successfully'}
        )
        
        return {
            'status': 'completed',
            'backtest_id': backtest_id,
            'trades_count': total_trades_count,
            'statistics_count': len(statistics)
        }
        
    except Exception as e:
        logger.error(f"Error executing backtest {backtest_id}: {str(e)}", exc_info=True)
        
        # Mark as failed in database
        try:
            backtest = Backtest.objects.get(id=backtest_id)
            backtest.status = 'failed'
            backtest.error_message = str(e)
            backtest.save()

        except Exception as db_error:
            logger.error(f"Error updating backtest status: {str(db_error)}")
        
        # Don't update state to FAILURE before raising - let Celery handle the exception
        # Just re-raise the exception so Celery can properly serialize it
        raise


### Legacy bulk_symbol_snapshots_task removed: single-symbol runs now use SymbolBacktestRun + bulk_symbol_runs_queue_task.


@shared_task(bind=True, name='backtest_engine.run_symbol_backtest_run', time_limit=60 * 60, soft_time_limit=60 * 60)
def run_symbol_backtest_run_task(self, run_id: int):
    """
    Execute a SymbolBacktestRun and persist its trades + statistics.

    This is intentionally separate from portfolio Backtest to avoid polluting Backtest table.
    """
    run = SymbolBacktestRun.objects.select_related('strategy', 'symbol', 'broker').get(id=run_id)

    logger.info("Starting symbol backtest run %s: %s - %s", run_id, run.strategy.name, run.symbol.ticker)
    self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Initializing symbol run...'})

    run.status = 'running'
    run.error_message = ''
    run.save(update_fields=['status', 'error_message', 'updated_at'])

    # Ensure clean artifacts
    SymbolBacktestTrade.objects.filter(run=run).delete()
    SymbolBacktestStatistics.objects.filter(run=run).delete()

    try:
        hedge_overlay = None
        if getattr(run, 'hedge_enabled', False):
            raw_overlay = compute_trade_hedge_overlay(
                run.start_date,
                run.end_date,
                run.hedge_config or {},
                yahoo_only=False,
            )
            if raw_overlay and not raw_overlay.get('error') and raw_overlay.get('index_ns'):
                hedge_overlay = raw_overlay
            else:
                logger.warning(
                    "Symbol run %s: hedge_enabled but overlay unavailable (%s); running without hedge split",
                    run_id,
                    raw_overlay.get('error') if raw_overlay else 'no overlay',
                )

        position_modes_to_run = _position_modes_to_run(run)
        mode_label = ' and '.join(m.upper() for m in position_modes_to_run)
        self.update_state(state='PROGRESS', meta={'progress': 20, 'message': f'Executing strategy ({mode_label})...'})

        all_trades_by_mode = {}
        mode_stats_results = {}
        all_baseline_stats_by_mode = {}

        for idx, position_mode in enumerate(position_modes_to_run):
            baseline_full_stats = None
            if hedge_overlay and getattr(run, 'run_strategy_only_baseline', True):
                ex_base = BacktestExecutor(
                    run,
                    position_mode=position_mode,
                    preprocessed_data=None,
                    hedge_overlay=None,
                )
                ex_base.execute_strategy()
                baseline_full_stats = ex_base.calculate_statistics()

            mode_executor = BacktestExecutor(
                run,
                position_mode=position_mode,
                preprocessed_data=None,
                hedge_overlay=hedge_overlay,
            )
            mode_executor.execute_strategy()
            mode_stats = mode_executor.calculate_statistics()

            all_trades_by_mode[position_mode] = mode_executor.trades
            mode_stats_results[position_mode] = mode_stats
            if baseline_full_stats:
                all_baseline_stats_by_mode[position_mode] = baseline_full_stats

            progress = 20 + int(((idx + 1) / max(len(position_modes_to_run), 1)) * 40)
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'message': f'Completed {idx + 1}/{len(position_modes_to_run)} position modes...',
                },
            )

        self.update_state(state='PROGRESS', meta={'progress': 70, 'message': 'Saving trades...'})
        for position_mode, trades in all_trades_by_mode.items():
            for trade_data in trades:
                trade_metadata = trade_data.get('metadata', {}) or {}

                def convert_value(value, field_name):
                    if value is None:
                        return None
                    try:
                        float_value = float(value)
                    except (ValueError, TypeError):
                        logger.warning("Could not convert trade %s value %r to float, using None", field_name, value)
                        return None
                    places = _TRADE_FIELD_DECIMAL_PLACES.get(field_name, 2)
                    return round(float_value, places)

                entry_price = convert_value(trade_data['entry_price'], 'entry_price')
                exit_price = convert_value(trade_data.get('exit_price'), 'exit_price')
                quantity = convert_value(trade_data['quantity'], 'quantity')
                trade_type = trade_data['trade_type']
                if (
                    exit_price is not None
                    and entry_price is not None
                    and quantity is not None
                    and trade_metadata.get('strategy_pnl') is None
                ):
                    pnl, pnl_percentage = _trade_pnl_from_stored_prices(trade_type, entry_price, exit_price, quantity)
                    is_winner = pnl > 0
                else:
                    pnl = convert_value(trade_data.get('pnl'), 'pnl')
                    pnl_percentage = convert_value(trade_data.get('pnl_percentage'), 'pnl_percentage')
                    is_winner = (pnl > 0) if pnl is not None else trade_data.get('is_winner')

                SymbolBacktestTrade.objects.create(
                    run=run,
                    symbol=trade_data['symbol'],
                    trade_type=trade_type,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_timestamp=trade_data['entry_timestamp'],
                    exit_timestamp=trade_data.get('exit_timestamp'),
                    quantity=quantity,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    is_winner=is_winner,
                    max_drawdown=convert_value(trade_data.get('max_drawdown'), 'max_drawdown'),
                    metadata={**trade_metadata, 'position_mode': position_mode},
                )

        self.update_state(state='PROGRESS', meta={'progress': 85, 'message': 'Saving statistics...'})
        primary_mode, secondary_mode = _primary_secondary_modes(position_modes_to_run)

        # Run-level (portfolio-like) stats (key None)
        primary_port = (mode_stats_results.get(primary_mode) or {}).get(None) or {}
        secondary_port = (mode_stats_results.get(secondary_mode) or {}).get(None) or {}
        SymbolBacktestStatistics.objects.create(
            run=run,
            symbol=None,
            total_trades=primary_port.get('total_trades', 0) or 0,
            winning_trades=primary_port.get('winning_trades', 0) or 0,
            losing_trades=primary_port.get('losing_trades', 0) or 0,
            win_rate=primary_port.get('win_rate'),
            total_pnl=primary_port.get('total_pnl', 0) or 0,
            total_pnl_percentage=primary_port.get('total_pnl_percentage'),
            average_pnl=primary_port.get('average_pnl'),
            average_winner=primary_port.get('average_winner'),
            average_loser=primary_port.get('average_loser'),
            profit_factor=primary_port.get('profit_factor'),
            max_drawdown=primary_port.get('max_drawdown'),
            max_drawdown_duration=primary_port.get('max_drawdown_duration'),
            avg_intra_trade_drawdown=primary_port.get('avg_intra_trade_drawdown'),
            worst_intra_trade_drawdown=primary_port.get('worst_intra_trade_drawdown'),
            sharpe_ratio=primary_port.get('sharpe_ratio'),
            cagr=primary_port.get('cagr'),
            total_return=primary_port.get('total_return'),
            equity_curve=primary_port.get('equity_curve') or [],
            additional_stats={'by_mode': {primary_mode: primary_port, secondary_mode: secondary_port}},
        )

        # Symbol-level stats
        symbol = run.symbol
        sym_primary = (mode_stats_results.get(primary_mode) or {}).get(symbol) or {}
        sym_secondary = (mode_stats_results.get(secondary_mode) or {}).get(symbol) or {}
        # Keep the secondary mode equity curve so the single-symbol UI can render it.
        # The primary mode curve is stored on the model field `equity_curve` already.
        sym_additional = {
            primary_mode: {k: v for k, v in sym_primary.items() if k != 'equity_curve'},
            secondary_mode: dict(sym_secondary),
        }
        strategy_only_sym = {}
        for pm in ('long', 'short'):
            bfull = all_baseline_stats_by_mode.get(pm) or {}
            sym_b = bfull.get(symbol)
            if sym_b:
                snap = _strategy_only_snapshot_from_stats(sym_b)
                if snap:
                    strategy_only_sym[pm] = snap
        if strategy_only_sym:
            sym_additional['strategy_only'] = strategy_only_sym

        SymbolBacktestStatistics.objects.create(
            run=run,
            symbol=symbol,
            total_trades=sym_primary.get('total_trades', 0) or 0,
            winning_trades=sym_primary.get('winning_trades', 0) or 0,
            losing_trades=sym_primary.get('losing_trades', 0) or 0,
            win_rate=sym_primary.get('win_rate'),
            total_pnl=sym_primary.get('total_pnl', 0) or 0,
            total_pnl_percentage=sym_primary.get('total_pnl_percentage'),
            average_pnl=sym_primary.get('average_pnl'),
            average_winner=sym_primary.get('average_winner'),
            average_loser=sym_primary.get('average_loser'),
            profit_factor=sym_primary.get('profit_factor'),
            max_drawdown=sym_primary.get('max_drawdown'),
            max_drawdown_duration=sym_primary.get('max_drawdown_duration'),
            avg_intra_trade_drawdown=sym_primary.get('avg_intra_trade_drawdown'),
            worst_intra_trade_drawdown=sym_primary.get('worst_intra_trade_drawdown'),
            sharpe_ratio=sym_primary.get('sharpe_ratio'),
            cagr=sym_primary.get('cagr'),
            total_return=sym_primary.get('total_return'),
            equity_curve=sym_primary.get('equity_curve') or [],
            additional_stats=sym_additional,
        )

        run.status = 'completed'
        run.completed_at = timezone.now()
        run.save(update_fields=['status', 'completed_at', 'updated_at'])

        self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Symbol run completed successfully'})
        return {'status': 'completed', 'run_id': run_id}
    except Exception as e:
        logger.error("Error executing symbol run %s: %s", run_id, str(e), exc_info=True)
        run.status = 'failed'
        run.error_message = str(e)
        run.save(update_fields=['status', 'error_message', 'updated_at'])
        raise


@shared_task(bind=True, name='backtest_engine.bulk_symbol_runs_queue', time_limit=60 * 60, soft_time_limit=60 * 60)
def bulk_symbol_runs_queue_task(self, strategy_id: int, tickers: list, run_body: dict):
    """
    Queue many SymbolBacktestRun jobs (one per ticker) and report progress.

    This task creates SymbolBacktestRun rows and enqueues `run_symbol_backtest_run_task` for each.
    """
    from strategies.models import StrategyDefinition, StrategyAssignment
    from market_data.models import Symbol
    from live_trading.models import Broker
    from backtest_engine.models import SymbolBacktestParameterSet
    from backtest_engine.parameter_sets import build_symbol_run_parameter_payload, signature_for_payload

    tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    total = len(tickers)
    self.update_state(state='PROGRESS', meta={'progress': 0, 'message': f'Queuing {total} symbol run(s)...'})

    strategy = StrategyDefinition.objects.get(id=int(strategy_id))
    broker = None
    broker_id = (run_body or {}).get('broker_id')
    if broker_id:
        try:
            broker = Broker.objects.get(id=int(broker_id))
        except Exception:
            broker = None

    runs = []
    errors = []

    base_body = dict(run_body or {})
    # Remove keys that are not SymbolBacktestRun fields
    base_body.pop('symbol_tickers', None)
    base_body.pop('strategy_id', None)

    # Create/get the shared parameter_set for this bulk request (shared across symbols).
    # Signature excludes symbol and name by design.
    start_dt = base_body.get('start_date') or timezone.now().replace(year=1900, month=1, day=1)
    end_dt = base_body.get('end_date') or timezone.now()
    # Accept ISO strings (API payload) as well as datetimes.
    try:
        from django.utils.dateparse import parse_datetime
        if isinstance(start_dt, str):
            start_dt = parse_datetime(start_dt) or start_dt
        if isinstance(end_dt, str):
            end_dt = parse_datetime(end_dt) or end_dt
    except Exception:
        pass
    position_modes = normalize_position_modes(base_body.get('position_modes'))
    # IMPORTANT: parameter_set is based on the user-chosen base config (not per-symbol assignment overrides),
    # so many symbols can be grouped under the same global identifier for cross-symbol analysis.
    payload = build_symbol_run_parameter_payload(
        strategy_id=strategy.id,
        broker_id=broker.id if broker else None,
        start_date=start_dt,
        end_date=end_dt,
        split_ratio=base_body.get('split_ratio', 0.7),
        initial_capital=base_body.get('initial_capital', 10000.0),
        bet_size_percentage=base_body.get('bet_size_percentage', 100.0),
        strategy_parameters=base_body.get('strategy_parameters') or {},
        position_modes=position_modes,
        hedge_enabled=bool(base_body.get('hedge_enabled', False)),
        run_strategy_only_baseline=bool(base_body.get('run_strategy_only_baseline', True)),
        hedge_config=base_body.get('hedge_config') or {},
    )
    sig = signature_for_payload(payload)
    bulk_label = (base_body.get('name') or '').strip()
    ps, _created = SymbolBacktestParameterSet.objects.get_or_create(
        signature=sig,
        defaults={
            'strategy': strategy,
            'broker': broker,
            'parameters': payload,
            'label': bulk_label[:200] if bulk_label else '',
        },
    )
    if bulk_label and not ps.label:
        ps.label = bulk_label[:200]
        ps.save(update_fields=['label'])

    for idx, ticker in enumerate(tickers):
        progress = int(((idx) / max(total, 1)) * 100)
        self.update_state(
            state='PROGRESS',
            meta={'progress': progress, 'message': f'Queuing {ticker} ({idx + 1}/{total})'},
        )
        try:
            sym = Symbol.objects.get(ticker=ticker, status='active')

            # Merge strategy parameters the same way portfolio backtest creation does
            merged_params = (strategy.default_parameters or {}).copy()
            assignment = StrategyAssignment.objects.filter(strategy=strategy, symbol=sym).first()
            if assignment:
                merged_params.update(assignment.parameters or {})
            else:
                global_assignment = StrategyAssignment.objects.filter(strategy=strategy, symbol__isnull=True).first()
                if global_assignment:
                    merged_params.update(global_assignment.parameters or {})
            merged_params.update(base_body.get('strategy_parameters') or {})

            run = SymbolBacktestRun.objects.create(
                name=base_body.get('name', '') or f'{strategy.name} — {ticker}',
                strategy=strategy,
                symbol=sym,
                broker=broker,
                parameter_set=ps,
                start_date=base_body.get('start_date') or timezone.now().replace(year=1900, month=1, day=1),
                end_date=base_body.get('end_date') or timezone.now(),
                split_ratio=base_body.get('split_ratio', 0.7),
                initial_capital=base_body.get('initial_capital', 10000.0),
                bet_size_percentage=base_body.get('bet_size_percentage', 100.0),
                strategy_parameters=merged_params,
                hedge_enabled=bool(base_body.get('hedge_enabled', False)),
                run_strategy_only_baseline=bool(base_body.get('run_strategy_only_baseline', True)),
                hedge_config=base_body.get('hedge_config') or {},
                position_modes=normalize_position_modes(base_body.get('position_modes')),
                status='pending',
            )
            task = run_symbol_backtest_run_task.delay(run.id)
            runs.append({'ticker': ticker, 'run_id': run.id, 'task_id': task.id})
        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})

    # Wait for all queued runs to finish so the UI "100%" means truly done.
    run_ids = [r.get('run_id') for r in runs if r.get('run_id') is not None]
    terminal = {'completed', 'failed'}
    total_runs = len(run_ids)
    if total_runs == 0:
        self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'No runs were queued.'})
        return {
            'status': 'completed',
            'progress': 100,
            'message': 'No runs were queued.',
            'queued': runs,
            'errors': errors,
            'total_requested': total,
            'total_queued': len(runs),
            'total_completed': 0,
            'total_failed': 0,
            'parameter_set': ps.signature,
        }

    # Poll DB; avoids needing to track all subtask ids.
    while True:
        statuses = (
            SymbolBacktestRun.objects.filter(id__in=run_ids)
            .values_list('status', flat=True)
        )
        done = sum(1 for s in statuses if s in terminal)
        failed = sum(1 for s in statuses if s == 'failed')
        progress = int(done / max(total_runs, 1) * 100)
        self.update_state(
            state='PROGRESS' if done < total_runs else 'SUCCESS',
            meta={
                'progress': progress,
                'message': f'Running symbol backtests… {done}/{total_runs} finished',
                'queued': runs,
                'errors': errors,
                'total_requested': total,
                'total_queued': len(runs),
                'total_completed': done - failed,
                'total_failed': failed,
                'parameter_set': ps.signature,
            },
        )
        if done >= total_runs:
            break
        time.sleep(3)

    return {
        'status': 'completed',
        'progress': 100,
        'message': f'Completed {done}/{total_runs} symbol run(s).',
        'queued': runs,
        'errors': errors,
        'total_requested': total,
        'total_queued': len(runs),
        'total_completed': done - failed,
        'total_failed': failed,
        'parameter_set': ps.signature,
    }

