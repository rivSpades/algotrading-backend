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
        if None in all_statistics:
            portfolio_stats = all_statistics[None]
            all_statistics[None] = {
                'all': portfolio_stats.get('all', {}),
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
            
            for trade_data in mode_trades:
                # Add position_mode to metadata
                trade_metadata = trade_data.get('metadata', {})
                if not isinstance(trade_metadata, dict):
                    trade_metadata = {}
                trade_metadata['position_mode'] = position_mode
                
                Trade.objects.create(
                    backtest=backtest,
                    symbol=trade_data['symbol'],
                    trade_type=trade_data['trade_type'],
                    entry_price=trade_data['entry_price'],
                    exit_price=trade_data.get('exit_price'),
                    entry_timestamp=trade_data['entry_timestamp'],
                    exit_timestamp=trade_data.get('exit_timestamp'),
                    quantity=trade_data['quantity'],
                    pnl=trade_data.get('pnl'),
                    pnl_percentage=trade_data.get('pnl_percentage'),
                    is_winner=trade_data.get('is_winner'),
                    max_drawdown=trade_data.get('max_drawdown'),
                    metadata=trade_metadata
                )
        
        # Save statistics
        for symbol, stats in statistics.items():
            if symbol is None:
                # Portfolio-level stats: store all/long/short breakdown in additional_stats
                # Save the 'all' stats as the main record, with long/short in additional_stats
                portfolio_all = stats.get('all', {})
                if portfolio_all:
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
                if symbol_all:  # Only save if 'all' stats exist
                    # Exclude equity_curve and additional_stats from main fields
                    stats_to_save = {k: v for k, v in symbol_all.items() if k != 'equity_curve' and k != 'additional_stats'}
                    equity_curve_all = symbol_all.get('equity_curve', [])
                    
                    # Get equity curves for long and short modes
                    equity_curve_long = stats.get('long', {}).get('equity_curve', [])
                    equity_curve_short = stats.get('short', {}).get('equity_curve', [])
                    
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

