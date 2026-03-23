"""
Celery tasks for backtest execution
"""

from celery import shared_task
from django.utils import timezone
from django.db import connections
from .models import Backtest, Trade, BacktestStatistics
from .services.backtest_executor import BacktestExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import pandas as pd
import os
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
        
        # OPTIMIZATION: Bulk load all OHLCV data in a single query instead of N queries
        # This dramatically reduces database round-trips (10-50x faster for large symbol sets)
        logger.info(f"Bulk loading OHLCV data for {len(symbols_to_preprocess)} symbols...")
        from market_data.models import OHLCV
        
        symbol_tickers = [s.ticker for s in symbols_to_preprocess]
        ticker_to_symbol = {s.ticker: s for s in symbols_to_preprocess}
        
        # Single bulk query for all OHLCV data
        bulk_ohlcv_queryset = OHLCV.objects.filter(
            symbol__ticker__in=symbol_tickers,
            timeframe='daily'
        ).select_related('symbol').order_by('symbol__ticker', 'timestamp')
        
        # Group OHLCV data by symbol ticker in memory (much faster than N queries)
        bulk_ohlcv_data = {}
        for ohlcv in bulk_ohlcv_queryset.values('symbol__ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume'):
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
        
        # Create symbol->data mapping
        bulk_ohlcv_by_symbol = {}
        for ticker, data in bulk_ohlcv_data.items():
            if ticker in ticker_to_symbol:
                bulk_ohlcv_by_symbol[ticker_to_symbol[ticker]] = data
        
        logger.info(f"Bulk loaded OHLCV data for {len(bulk_ohlcv_by_symbol)} symbols")
        
        # Pre-process in parallel
        preprocessed_data = {}  # {symbol: {'df': DataFrame, 'indicators': dict, 'test_df': DataFrame, 'price_cache': dict}}
        preprocessed_lock = threading.Lock()
        
        def preprocess_symbol(symbol):
            """Load data and compute indicators for a single symbol (Phase 1)"""
            try:
                logger.info(f"Pre-processing symbol: {symbol.ticker}")
                
                # OPTIMIZATION: Use bulk-loaded OHLCV data instead of individual query
                if symbol not in bulk_ohlcv_by_symbol:
                    logger.warning(f"No OHLCV data found for symbol {symbol.ticker} in bulk load")
                    return symbol, None
                
                ohlcv_list = bulk_ohlcv_by_symbol[symbol]
                
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
                    
                    # Process batch in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(preprocess_symbol, symbol): symbol for symbol in batch_symbols}
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
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(preprocess_symbol, symbol): symbol for symbol in symbols_to_preprocess}
                    for future in as_completed(futures):
                        future.result()  # Raise exception if any occurred
        
        logger.info(f"Phase 1 complete: Pre-processed {len(preprocessed_data)} symbols")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 18, 'message': f'Pre-processed {len(preprocessed_data)} symbols'}
        )
        
        # Phase 2: Execute strategy twice: LONG and SHORT in parallel
        # Each mode uses the pre-processed data from Phase 1
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'message': 'Executing strategy for long and short position modes...'}
        )
        
        all_trades_by_mode = {}
        all_statistics = {}
        
        def execute_position_mode(position_mode):
            """Execute strategy for a single position mode (thread-safe helper)
            
            Note: Cannot call self.update_state() from within threads.
            State updates must be done from the main task thread.
            """
            try:
                logger.info(f"[{position_mode.upper()}] Starting strategy execution (Phase 2)")
                
                # Create a new executor instance for this mode with pre-processed data
                mode_executor = BacktestExecutor(backtest, position_mode=position_mode, preprocessed_data=preprocessed_data)
                
                # Execute strategy for this mode (data/indicators loaded on-demand)
                logger.info(f"[{position_mode.upper()}] Executing strategy logic...")
                mode_executor.execute_strategy()
                
                # Log trade counts for debugging
                buy_trades = [t for t in mode_executor.trades if t.get('trade_type') == 'buy']
                sell_trades = [t for t in mode_executor.trades if t.get('trade_type') == 'sell']
                logger.info(f"[{position_mode.upper()}] Generated {len(mode_executor.trades)} trades: {len(buy_trades)} buy, {len(sell_trades)} sell")
                
                if not mode_executor.trades:
                    logger.warning(f"[{position_mode.upper()}] No trades generated for backtest {backtest_id}")
                
                # Calculate statistics for this mode
                logger.info(f"[{position_mode.upper()}] Calculating statistics...")
                mode_stats = mode_executor.calculate_statistics()
                
                logger.info(f"[{position_mode.upper()}] Completed strategy execution")
                
                return position_mode, mode_executor.trades, mode_stats
            except Exception as e:
                logger.error(f"[{position_mode.upper()}] Error executing strategy: {str(e)}")
                # Close database connection in this thread
                connections.close_all()
                raise
            finally:
                # Always close connections in thread to prevent connection leaks
                connections.close_all()
        
        # Execute long and short position modes in parallel
        logger.info("Starting parallel execution of strategy for position modes (long, short)")
        mode_stats_results = {}
        results_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=2) as executor_pool:
            future_to_mode = {
                executor_pool.submit(execute_position_mode, mode): mode 
                for mode in ['long', 'short']
            }
            
            # Update progress as each mode completes (from main thread)
            completed_count = 0
            total_modes = 2
            
            # Collect results as they complete
            for future in as_completed(future_to_mode):
                position_mode = future_to_mode[future]
                try:
                    mode_result, trades, stats = future.result()
                    with results_lock:
                        all_trades_by_mode[mode_result] = trades
                        mode_stats_results[mode_result] = stats
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
            meta={'progress': 70, 'message': 'Strategy execution completed for long and short modes'}
        )
        
        # Process statistics from both modes
        for position_mode in ['long', 'short']:
            mode_stats = mode_stats_results.get(position_mode, {})
            
            # Store portfolio-level statistics
            if None in mode_stats:
                all_statistics[None] = all_statistics.get(None, {})
                all_statistics[None][position_mode] = mode_stats[None]
            
            # Store symbol-level statistics for each position mode (long, short)
            # This allows frontend to display stats for each mode without calculations
            for symbol, symbol_stats in mode_stats.items():
                if symbol is not None:  # Skip portfolio-level stats (None key)
                    if symbol not in all_statistics:
                        all_statistics[symbol] = {}
                    all_statistics[symbol][position_mode] = symbol_stats
                    logger.info(f"Stored symbol-level statistics for {symbol.ticker} ({position_mode}): {len(symbol_stats)} fields")
        
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
        
        # Save trades from each mode (LONG, SHORT)
        # Store the mode in metadata so we can filter by mode later
        for position_mode in ['long', 'short']:
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
                # Derive PnL from stored prices × qty so DB rows match (exit - entry) * qty (long) / (entry - exit) * qty (short)
                if exit_price is not None and entry_price is not None and quantity is not None:
                    pnl, pnl_percentage = _trade_pnl_from_stored_prices(trade_type, entry_price, exit_price, quantity)
                    is_winner = pnl > 0
                else:
                    pnl = convert_value(trade_data.get('pnl'), 'pnl')
                    pnl_percentage = convert_value(trade_data.get('pnl_percentage'), 'pnl_percentage')
                    is_winner = trade_data.get('is_winner')
                
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
        
        # Save statistics
        for symbol, stats in statistics.items():
            if symbol is None:
                # Portfolio-level stats: main row = long mode; short nested in additional_stats
                portfolio_long = stats.get('long', {})
                if not portfolio_long or not portfolio_long.get('total_trades', 0):
                    portfolio_long = {
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
                stats_dict = {k: v for k, v in portfolio_long.items() if k not in ['additional_stats', 'equity_curve', 'skipped_trades_count', 'independent_bet_amounts']}
                
                # Round and validate numeric fields
                numeric_fields = ['total_pnl', 'total_pnl_percentage', 'average_pnl', 'average_winner', 
                                'average_loser', 'profit_factor', 'max_drawdown', 'avg_intra_trade_drawdown',
                                'worst_intra_trade_drawdown', 'sharpe_ratio', 
                                'cagr', 'total_return', 'win_rate']
                
                for field in numeric_fields:
                    if field in stats_dict:
                        stats_dict[field] = round_and_validate_stat(stats_dict[field], field, decimal_places=2, symbol=None)
                
                skipped_trades_count = portfolio_long.get('skipped_trades_count', 0)
                
                BacktestStatistics.objects.create(
                    backtest=backtest,
                    symbol=None,
                    **stats_dict,
                    equity_curve=portfolio_long.get('equity_curve', []),
                    additional_stats={
                        'long': {
                            'skipped_trades_count': skipped_trades_count,
                        },
                        'short': stats.get('short', {}),
                    }
                )
            else:
                # Symbol-specific stats: main row = long mode; short in additional_stats
                symbol_long = stats.get('long', {})
                stats_to_save = {k: v for k, v in symbol_long.items() if k not in ['equity_curve', 'independent_bet_amounts', 'skipped_trades_count', 'additional_stats']} if symbol_long else {}
                equity_curve_long_main = symbol_long.get('equity_curve', []) if symbol_long else []
                independent_bet_amounts_long = stats.get('long', {}).get('independent_bet_amounts', {})
                equity_curve_short = stats.get('short', {}).get('equity_curve', [])
                independent_bet_amounts_short = stats.get('short', {}).get('independent_bet_amounts', {})
                
                if not symbol_long:
                    stats_to_save = {
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
                    }
                
                # Round and validate numeric fields in stats_to_save
                numeric_fields = ['total_pnl', 'total_pnl_percentage', 'average_pnl', 'average_winner', 
                                'average_loser', 'profit_factor', 'max_drawdown', 'avg_intra_trade_drawdown',
                                'worst_intra_trade_drawdown', 'sharpe_ratio', 
                                'cagr', 'total_return', 'win_rate']
                
                for field in numeric_fields:
                    if field in stats_to_save:
                        stats_to_save[field] = round_and_validate_stat(stats_to_save[field], field, decimal_places=2, symbol=symbol)
                
                skipped_trades_count = symbol_long.get('skipped_trades_count', 0) if symbol_long else 0
                
                BacktestStatistics.objects.create(
                    backtest=backtest,
                    symbol=symbol,
                    equity_curve=equity_curve_long_main,
                    additional_stats={
                        'long': {
                            'independent_bet_amounts': independent_bet_amounts_long,
                            'skipped_trades_count': skipped_trades_count,
                        },
                        'short': {
                            **{k: v for k, v in stats.get('short', {}).items() if k not in ['equity_curve', 'independent_bet_amounts', 'skipped_trades_count']},
                            'equity_curve': equity_curve_short,
                            'independent_bet_amounts': independent_bet_amounts_short,
                        },
                    },
                    **stats_to_save
                )
        
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

