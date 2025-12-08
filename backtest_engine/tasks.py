"""
Celery tasks for backtest execution
"""

from celery import shared_task
from django.utils import timezone
from .models import Backtest, Trade, BacktestStatistics
from .services.backtest_executor import BacktestExecutor
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='backtest_engine.run_backtest')
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
        
        # Initialize executor
        executor = BacktestExecutor(backtest)
        
        # Update progress: 20% - Loading data
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'message': 'Loading OHLCV data...'}
        )
        
        # Load data
        executor.load_data()
        
        if not executor.ohlcv_data:
            raise ValueError("No OHLCV data available for backtest symbols")
        
        # Update progress: 40% - Computing indicators
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'message': 'Computing indicators...'}
        )
        
        # Compute indicators
        executor.compute_indicators()
        
        # Update progress: 60% - Executing strategy
        # Execute strategy three times: ALL, LONG, and SHORT
        # This creates three separate strategy runs as if they were different strategies
        all_trades_by_mode = {}
        all_statistics = {}
        
        for position_mode in ['all', 'long', 'short']:
            logger.info(f"Executing strategy with position_mode={position_mode}")
            
            self.update_state(
                state='PROGRESS',
                meta={'progress': 60 + (position_mode == 'long' and 5 or position_mode == 'short' and 10 or 0), 
                      'message': f'Executing strategy ({position_mode.upper()} positions)...'}
            )
            
            # Create a new executor instance for each mode
            mode_executor = BacktestExecutor(backtest, position_mode=position_mode)
            mode_executor.ohlcv_data = executor.ohlcv_data  # Reuse loaded data
            mode_executor.indicators = executor.indicators  # Reuse computed indicators
            
            # Execute strategy for this mode
            mode_executor.execute_strategy()
            
            # Store trades for this mode (will be saved later)
            all_trades_by_mode[position_mode] = mode_executor.trades
            
            # Log trade counts for debugging
            buy_trades = [t for t in mode_executor.trades if t.get('trade_type') == 'buy']
            sell_trades = [t for t in mode_executor.trades if t.get('trade_type') == 'sell']
            logger.info(f"Generated {len(mode_executor.trades)} trades for position_mode={position_mode}: {len(buy_trades)} buy, {len(sell_trades)} sell")
            
            if not mode_executor.trades:
                logger.warning(f"No trades generated for backtest {backtest_id} with position_mode={position_mode}")
            
            # Calculate statistics for this mode
            mode_stats = mode_executor.calculate_statistics()
            
            # Store portfolio-level statistics
            if None in mode_stats:
                all_statistics[None] = all_statistics.get(None, {})
                all_statistics[None][position_mode] = mode_stats[None]
            
            # Store symbol-level statistics for each position mode (all, long, short)
            # This allows frontend to display stats for each mode without calculations
            for symbol, symbol_stats in mode_stats.items():
                if symbol is not None:  # Skip portfolio-level stats (None key)
                    if symbol not in all_statistics:
                        all_statistics[symbol] = {}
                    all_statistics[symbol][position_mode] = symbol_stats
                    logger.info(f"Stored symbol-level statistics for {symbol.ticker} ({position_mode}): {len(symbol_stats)} fields")
        
        # Update progress: 80% - Calculating statistics
        self.update_state(
            state='PROGRESS',
            meta={'progress': 80, 'message': 'Calculating statistics...'}
        )
        
        # Combine portfolio stats into the expected structure
        # The executor returns stats with mode keys ('all', 'long', 'short') already
        # But we need to ensure the structure is correct
        if None in all_statistics:
            portfolio_stats = all_statistics[None]
            # If portfolio_stats is already a dict with 'all', 'long', 'short' keys, use it
            # Otherwise, wrap it in the expected structure
            if 'all' in portfolio_stats or 'long' in portfolio_stats or 'short' in portfolio_stats:
                # Already in correct format
                pass
            else:
                # Old format - wrap it
                all_statistics[None] = {
                    'all': portfolio_stats.get('all', portfolio_stats),  # Use portfolio_stats itself if no 'all' key
                    'long': portfolio_stats.get('long', {}),
                    'short': portfolio_stats.get('short', {}),
                }
        
        statistics = all_statistics
        
        # Update progress: 90% - Saving results
        self.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'message': 'Saving trades and statistics...'}
        )
        
        # Save trades from all modes (ALL, LONG, SHORT)
        # Store the mode in metadata so we can filter by mode later
        for position_mode in ['all', 'long', 'short']:
            mode_trades = all_trades_by_mode.get(position_mode, [])
            logger.info(f"Saving {len(mode_trades)} trades for position_mode={position_mode}")
            
            # Count buy vs sell trades for logging
            buy_count = sum(1 for t in mode_trades if t.get('trade_type') == 'buy')
            sell_count = sum(1 for t in mode_trades if t.get('trade_type') == 'sell')
            logger.info(f"  - Buy trades: {buy_count}, Sell trades: {sell_count}")
            
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
                
                # Convert numpy types to Python native types to avoid Django field validation errors
                def convert_value(value):
                    """Convert numpy/pandas types to Python native types"""
                    if value is None:
                        return None
                    import numpy as np
                    # Handle numpy arrays first (before checking other numpy types)
                    if isinstance(value, np.ndarray):
                        # For arrays, return None or handle appropriately
                        if value.size == 0:
                            return None
                        if value.size == 1:
                            return float(value.item())
                        # Multiple values - this shouldn't happen for a single field
                        return None
                    # Handle numpy scalars
                    if isinstance(value, (np.integer, np.floating)):
                        return float(value)
                    # Handle numpy types that have .item() method
                    if hasattr(value, 'item') and not isinstance(value, (str, list, dict)):
                        try:
                            return float(value.item())
                        except (ValueError, AttributeError):
                            pass
                    # Handle pandas types
                    if hasattr(value, 'iloc'):  # pandas Series/DataFrame
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            pass
                    # If it's already a Python native type, return as is
                    if isinstance(value, (int, float, str, bool)):
                        return value
                    # Try to convert to float as last resort
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        # If all else fails, return None (safer than passing invalid type)
                        logger.warning(f"Could not convert value {type(value)} to Python native type, using None")
                        return None
                
                Trade.objects.create(
                    backtest=backtest,
                    symbol=trade_data['symbol'],
                    trade_type=trade_data['trade_type'],
                    entry_price=convert_value(trade_data['entry_price']),
                    exit_price=convert_value(trade_data.get('exit_price')),
                    entry_timestamp=trade_data['entry_timestamp'],
                    exit_timestamp=trade_data.get('exit_timestamp'),
                    quantity=convert_value(trade_data['quantity']),
                    pnl=convert_value(trade_data.get('pnl')),
                    pnl_percentage=convert_value(trade_data.get('pnl_percentage')),
                    is_winner=trade_data.get('is_winner'),
                    max_drawdown=convert_value(trade_data.get('max_drawdown')),
                    metadata=trade_metadata
                )
        
        # Save statistics
        for symbol, stats in statistics.items():
            if symbol is None:
                # Portfolio-level stats: store all/long/short breakdown in additional_stats
                # Save the 'all' stats as the main record, with long/short in additional_stats
                portfolio_all = stats.get('all', {})
                # Always create portfolio stats, even if 'all' mode is empty
                # This ensures portfolio stats exist for the frontend
                if not portfolio_all or not portfolio_all.get('total_trades', 0):
                    # Create empty portfolio stats if 'all' mode is empty or has no trades
                    portfolio_all = {
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
                        'sharpe_ratio': 0,
                        'cagr': 0,
                        'total_return': 0,
                        'equity_curve': [],
                    }
                
                BacktestStatistics.objects.create(
                    backtest=backtest,
                    symbol=None,
                    **{k: v for k, v in portfolio_all.items() if k != 'additional_stats' and k != 'equity_curve'},
                    equity_curve=portfolio_all.get('equity_curve', []),
                    additional_stats={
                        'long': stats.get('long', {}),
                        'short': stats.get('short', {}),
                    }
                )
            else:
                # Symbol-specific stats: store all/long/short breakdown in additional_stats
                # Similar to portfolio stats structure
                symbol_all = stats.get('all', {})
                # Always save statistics even if empty (no trades) so frontend knows symbol was processed
                # Exclude equity_curve and additional_stats from main fields
                stats_to_save = {k: v for k, v in symbol_all.items() if k != 'equity_curve' and k != 'additional_stats'} if symbol_all else {}
                equity_curve_all = symbol_all.get('equity_curve', []) if symbol_all else []
                
                # Get equity curves for long and short modes
                equity_curve_long = stats.get('long', {}).get('equity_curve', [])
                equity_curve_short = stats.get('short', {}).get('equity_curve', [])
                
                # If no stats were calculated (empty dict), create default empty statistics
                if not symbol_all:
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
                        'sharpe_ratio': 0,
                        'cagr': 0,
                        'total_return': 0,
                    }
                
                BacktestStatistics.objects.create(
                    backtest=backtest,
                    symbol=symbol,
                    equity_curve=equity_curve_all,  # Store 'all' mode equity curve in main field
                    additional_stats={
                        'long': {
                            **stats.get('long', {}),
                            'equity_curve': equity_curve_long,  # Include equity curve in additional_stats
                        },
                        'short': {
                            **stats.get('short', {}),
                            'equity_curve': equity_curve_short,  # Include equity curve in additional_stats
                        },
                    },
                    **stats_to_save
                )
        
        # Mark as completed
        backtest.status = 'completed'
        backtest.completed_at = timezone.now()
        backtest.save()
        
        logger.info(f"Backtest {backtest_id} completed successfully")
        
        # Update progress: 100% - Completed
        self.update_state(
            state='SUCCESS',
            meta={'progress': 100, 'message': 'Backtest completed successfully'}
        )
        
        return {
            'status': 'completed',
            'backtest_id': backtest_id,
            'trades_count': len(executor.trades),
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

