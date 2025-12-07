"""
Backtest Executor Service
Executes trading strategies on historical data and generates trades and statistics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from decimal import Decimal
from django.utils import timezone
from django.db import models
import pytz
from market_data.models import Symbol, OHLCV
from strategies.models import StrategyDefinition
from analytical_tools.indicators import compute_indicator
from market_data.services.indicator_service import compute_indicators_for_ohlcv
import logging

logger = logging.getLogger(__name__)


class BacktestExecutor:
    """Executes backtests for trading strategies"""
    
    def __init__(self, backtest, position_mode='all'):
        """
        Initialize backtest executor
        
        Args:
            backtest: Backtest model instance
            position_mode: 'all' (both long and short), 'long' (only long positions), 'short' (only short positions)
        """
        self.backtest = backtest
        self.strategy = backtest.strategy
        self.parameters = backtest.strategy_parameters
        self.symbols = list(backtest.symbols.all())
        self.start_date = backtest.start_date
        self.end_date = backtest.end_date
        self.split_ratio = backtest.split_ratio
        self.position_mode = position_mode  # 'all', 'long', or 'short'
        
        # Initialize data storage
        self.ohlcv_data = {}  # {symbol: DataFrame}
        self.indicators = {}  # {symbol: {indicator_name: Series}}
        self.trades = []  # List of trade dicts
        self.equity_curves = {}  # {symbol: [(timestamp, equity), ...]}
        
    def load_data(self):
        """Load OHLCV data for all symbols"""
        logger.info(f"Loading OHLCV data for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            # Fetch OHLCV data - if dates are provided, filter by them, otherwise get all data
            ohlcv_filter = {
                'symbol': symbol,
                'timeframe': 'daily',
            }
            
            # Add date filters if provided
            from django.utils import timezone
            
            if self.start_date:
                ohlcv_filter['timestamp__gte'] = self.start_date
            if self.end_date:
                ohlcv_filter['timestamp__lte'] = self.end_date
            
            ohlcv_queryset = OHLCV.objects.filter(**ohlcv_filter).order_by('timestamp')
            
            # Convert to list of dicts
            ohlcv_list = []
            for ohlcv in ohlcv_queryset.values('timestamp', 'open', 'high', 'low', 'close', 'volume'):
                timestamp = ohlcv['timestamp']
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()
                ohlcv_list.append({
                    'timestamp': timestamp,
                    'open': float(ohlcv['open']),
                    'high': float(ohlcv['high']),
                    'low': float(ohlcv['low']),
                    'close': float(ohlcv['close']),
                    'volume': float(ohlcv['volume'])
                })
            
            if not ohlcv_list:
                logger.warning(f"No OHLCV data found for {symbol.ticker}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Store the actual data range for this symbol
            symbol_start = df['timestamp'].min().to_pydatetime()
            symbol_end = df['timestamp'].max().to_pydatetime()
            
            # Convert to timezone-aware if needed
            if timezone.is_naive(symbol_start):
                symbol_start = timezone.make_aware(symbol_start)
            if timezone.is_naive(symbol_end):
                symbol_end = timezone.make_aware(symbol_end)
            
            # Update backtest date range to actual data range (use earliest start, latest end)
            # Only update if we have valid dates (not the default "all data" placeholder)
            from datetime import datetime
            min_valid_date = timezone.make_aware(datetime(2000, 1, 1))  # Only update if date is after 2000
            
            # Ensure both dates are timezone-aware for comparison
            if self.start_date and timezone.is_naive(self.start_date):
                self.start_date = timezone.make_aware(self.start_date)
            if self.end_date and timezone.is_naive(self.end_date):
                self.end_date = timezone.make_aware(self.end_date)
            
            # Compare and update start_date
            if not self.start_date:
                self.start_date = symbol_start
                self.backtest.start_date = self.start_date
            else:
                # Only update if current date is invalid (before 2000) or if symbol_start is earlier
                try:
                    if self.start_date < min_valid_date:
                        self.start_date = symbol_start
                        self.backtest.start_date = self.start_date
                    elif symbol_start < self.start_date:
                        self.start_date = symbol_start
                        self.backtest.start_date = self.start_date
                except TypeError:
                    # If comparison fails, just use symbol_start
                    self.start_date = symbol_start
                    self.backtest.start_date = self.start_date
            
            # Compare and update end_date
            if not self.end_date:
                self.end_date = symbol_end
                self.backtest.end_date = self.end_date
            else:
                # Only update if current date is invalid (before 2000) or if symbol_end is later
                try:
                    if self.end_date < min_valid_date:
                        self.end_date = symbol_end
                        self.backtest.end_date = self.end_date
                    elif symbol_end > self.end_date:
                        self.end_date = symbol_end
                        self.backtest.end_date = self.end_date
                except TypeError:
                    # If comparison fails, just use symbol_end
                    self.end_date = symbol_end
                    self.backtest.end_date = self.end_date
            
            # Save updated dates
            self.backtest.save(update_fields=['start_date', 'end_date'])
            
            self.ohlcv_data[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol.ticker}")
    
    def compute_indicators(self):
        """Compute required indicators for all symbols"""
        logger.info("Computing indicators for all symbols")
        
        # Get required indicators from strategy
        required_indicators = self.strategy.analytic_tools_used
        
        for symbol, df in self.ohlcv_data.items():
            symbol_indicators = {}
            
            # For SMA Crossover strategy, compute SMAs directly from strategy parameters
            if 'SMA' in required_indicators:
                from analytical_tools.indicators import compute_indicator
                
                # Get strategy parameters
                fast_period = self.parameters.get('fast_period', 20)
                slow_period = self.parameters.get('slow_period', 50)
                
                # Compute fast SMA
                fast_sma_df = compute_indicator('SMA', df.copy(), {'period': fast_period})
                if 'value' in fast_sma_df.columns:
                    fast_sma_key = f'SMA_{fast_period}'
                    # Align with original DataFrame index
                    fast_values = fast_sma_df['value'].values
                    if len(fast_values) == len(df):
                        symbol_indicators[fast_sma_key] = pd.Series(fast_values, index=df.index)
                        logger.info(f"Computed {fast_sma_key} for {symbol.ticker}: {len(fast_values)} values, non-null: {pd.Series(fast_values).notna().sum()}")
                    else:
                        logger.error(f"Fast SMA length mismatch: {len(fast_values)} != {len(df)}")
                
                # Compute slow SMA
                slow_sma_df = compute_indicator('SMA', df.copy(), {'period': slow_period})
                if 'value' in slow_sma_df.columns:
                    slow_sma_key = f'SMA_{slow_period}'
                    # Align with original DataFrame index
                    slow_values = slow_sma_df['value'].values
                    if len(slow_values) == len(df):
                        symbol_indicators[slow_sma_key] = pd.Series(slow_values, index=df.index)
                        logger.info(f"Computed {slow_sma_key} for {symbol.ticker}: {len(slow_values)} values, non-null: {pd.Series(slow_values).notna().sum()}")
                    else:
                        logger.error(f"Slow SMA length mismatch: {len(slow_values)} != {len(df)}")
            else:
                # For other strategies, use the indicator service
                ohlcv_list = df.to_dict('records')
                indicator_values = compute_indicators_for_ohlcv(symbol, ohlcv_list)
                
                for indicator_key, values in indicator_values.items():
                    base_name = indicator_key.split('_')[0]
                    if base_name in required_indicators:
                        series = pd.Series(values, index=df.index)
                        symbol_indicators[base_name] = series
                        symbol_indicators[indicator_key] = series
            
            self.indicators[symbol] = symbol_indicators
            logger.info(f"Computed {len(symbol_indicators)} indicators for {symbol.ticker}: {list(symbol_indicators.keys())}")
    
    def execute_strategy(self):
        """Execute strategy logic and generate trades
        
        For multi-symbol backtests, processes all symbols chronologically using shared portfolio capital.
        For single-symbol backtests, uses symbol-specific capital.
        """
        logger.info(f"Executing strategy: {self.strategy.name}")
        
        # Check if this is a multi-symbol backtest
        is_multi_symbol = len(self.symbols) > 1
        
        if is_multi_symbol:
            # For multi-symbol: process all symbols chronologically with shared capital
            self._execute_strategy_multi_symbol()
        else:
            # For single symbol: process with symbol-specific capital (original logic)
            self._execute_strategy_single_symbol()
    
    def _execute_strategy_single_symbol(self):
        """Execute strategy for a single symbol (original implementation)"""
        symbol = self.symbols[0]
        
        if symbol not in self.ohlcv_data:
            return
            
        df = self.ohlcv_data[symbol]
        indicators = self.indicators.get(symbol, {})
        
        # Validate we have data
        if df.empty:
            logger.warning(f"No data available for symbol {symbol.ticker}, skipping")
            return
        
        # Split data into training and testing
        split_idx = int(len(df) * self.split_ratio)
        if split_idx >= len(df):
            split_idx = max(0, len(df) - 1)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        if test_df.empty:
            logger.warning(f"No test data available for symbol {symbol.ticker} after split")
            return
        
        # Initialize position tracking
        position = None
        initial_capital = float(self.backtest.initial_capital)
        bet_size_pct = float(self.backtest.bet_size_percentage) / 100.0
        capital = initial_capital
        
        # Initialize equity curve
        if not train_df.empty:
            first_timestamp = train_df.iloc[0]['timestamp']
        elif not test_df.empty:
            first_timestamp = test_df.iloc[0]['timestamp']
        else:
            first_timestamp = df.iloc[0]['timestamp']
        
        equity_curve = [(first_timestamp, initial_capital)]
        
        # Execute strategy on test data
        prev_indicator_values = None
        
        logger.info(f"Executing strategy on {len(test_df)} test rows for {symbol.ticker}")
        
        for i, (idx, row) in enumerate(test_df.iterrows()):
            timestamp = row['timestamp']
            price = float(row['close']) if pd.notna(row['close']) else None
            if price is None:
                continue
            
            # Get indicator values
            indicator_values = {}
            for name, series in indicators.items():
                if idx in series.index:
                    try:
                        value = series.loc[idx]
                        if pd.notna(value):
                            indicator_values[name] = float(value)
                    except (KeyError, IndexError, TypeError):
                        pass
            
            # For SMA crossover, we need both SMAs
            fast_period = self.parameters.get('fast_period', 20)
            slow_period = self.parameters.get('slow_period', 50)
            fast_sma_key = f'SMA_{fast_period}'
            slow_sma_key = f'SMA_{slow_period}'
            
            fast_sma = indicator_values.get(fast_sma_key)
            slow_sma = indicator_values.get(slow_sma_key)
            
            if fast_sma is None or slow_sma is None:
                if prev_indicator_values is None and indicator_values:
                    prev_indicator_values = indicator_values.copy()
                continue
            
            # Generate signal
            signal = self._generate_signal(row, indicator_values, position, prev_indicator_values)
            
            if signal:
                logger.info(f"[{self.position_mode.upper()}] Signal at {timestamp}: {signal}")
            
            if indicator_values:
                prev_indicator_values = indicator_values.copy()
            
            # Execute trades based on signal (reuse existing logic)
            capital, position, equity_curve = self._process_trade_signal(
                symbol, timestamp, price, signal, position, capital, bet_size_pct, equity_curve, indicators, idx
            )
        
        # Close any open position at the end
        if position is not None:
            final_price = float(test_df.iloc[-1]['close'])
            final_timestamp = test_df.iloc[-1]['timestamp']
            capital = self._close_position_at_end(symbol, position, final_price, final_timestamp, capital)
            equity_curve.append((final_timestamp, float(capital)))
        
        # Store equity curve
        self.equity_curves[symbol] = equity_curve
    
    def _execute_strategy_multi_symbol(self):
        """Execute strategy for multiple symbols with shared portfolio capital
        
        Optimized version using price cache for fast lookups instead of DataFrame filtering.
        """
        logger.info(f"Executing multi-symbol strategy with shared capital for {len(self.symbols)} symbols")
        
        # Initialize shared portfolio capital
        initial_capital = float(self.backtest.initial_capital)
        bet_size_pct = float(self.backtest.bet_size_percentage) / 100.0
        portfolio_capital = initial_capital
        
        # Track positions per symbol: {symbol: position_dict}
        positions = {}  # {symbol: {'type': 'buy'|'sell', 'entry_price': float, ...}}
        
        # Track previous indicator values per symbol
        prev_indicators = {}  # {symbol: {indicator_name: value}}
        
        # Prepare data for all symbols
        symbol_data = {}  # {symbol: {'df': DataFrame, 'test_df': DataFrame, 'indicators': dict}}
        
        # Build timestamp-to-price lookup dictionaries for fast access (O(1) instead of O(n) DataFrame filtering)
        symbol_price_cache = {}  # {symbol: {timestamp: price}}
        
        for symbol in self.symbols:
            if symbol not in self.ohlcv_data:
                continue
            
            df = self.ohlcv_data[symbol]
            indicators = self.indicators.get(symbol, {})
            
            if df.empty:
                continue
            
            # Split data
            split_idx = int(len(df) * self.split_ratio)
            if split_idx >= len(df):
                split_idx = max(0, len(df) - 1)
            
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            if test_df.empty:
                continue
            
            # Build price cache - O(n) once instead of O(n*m) repeatedly
            # This avoids expensive DataFrame filtering in the main loop
            price_cache = {}
            for idx, row in test_df.iterrows():
                timestamp = row['timestamp']
                if pd.notna(row['close']):
                    price_cache[timestamp] = float(row['close'])
            symbol_price_cache[symbol] = price_cache
            
            symbol_data[symbol] = {
                'df': df,
                'test_df': test_df,
                'indicators': indicators
            }
            positions[symbol] = None
            prev_indicators[symbol] = None
        
        if not symbol_data:
            logger.warning("No valid symbol data for multi-symbol execution")
            return
        
        # Create a merged timeline of all events from all symbols
        # Collect all timestamps with their corresponding symbol and row data
        all_events = []  # List of (timestamp, symbol, row_index, row_data)
        
        for symbol, data in symbol_data.items():
            test_df = data['test_df']
            for idx, row in test_df.iterrows():
                timestamp = row['timestamp']
                if pd.notna(row['close']):
                    all_events.append((timestamp, symbol, idx, row))
        
        # Sort all events chronologically
        all_events.sort(key=lambda x: x[0])
        
        logger.info(f"Processing {len(all_events)} events across {len(symbol_data)} symbols chronologically")
        
        # Helper function to get current price for a symbol at a timestamp (uses cache)
        def get_price_at_timestamp(symbol, timestamp):
            """Get price for symbol at timestamp using cached lookup (O(1) dict lookup)"""
            price_cache = symbol_price_cache.get(symbol, {})
            # Find the closest timestamp <= current timestamp
            # Since events are processed chronologically, we can use the last known price
            valid_timestamps = [ts for ts in price_cache.keys() if ts <= timestamp]
            if valid_timestamps:
                # Get the most recent price (closest to timestamp)
                closest_ts = max(valid_timestamps)
                return price_cache[closest_ts]
            return None
        
        # Initialize portfolio equity curve
        portfolio_equity_curve = []
        first_event_timestamp = all_events[0][0] if all_events else None
        if first_event_timestamp:
            portfolio_equity_curve.append((first_event_timestamp, initial_capital))
        
        # Process events chronologically with shared capital
        for timestamp, symbol, idx, row in all_events:
            price = float(row['close'])
            position = positions[symbol]
            indicators = symbol_data[symbol]['indicators']
            prev_indicator_values = prev_indicators[symbol]
            
            # Get indicator values for current row
            indicator_values = {}
            for name, series in indicators.items():
                if idx in series.index:
                    try:
                        value = series.loc[idx]
                        if pd.notna(value):
                            indicator_values[name] = float(value)
                    except (KeyError, IndexError, TypeError):
                        pass
            
            # Check if we have required indicators
            fast_period = self.parameters.get('fast_period', 20)
            slow_period = self.parameters.get('slow_period', 50)
            fast_sma_key = f'SMA_{fast_period}'
            slow_sma_key = f'SMA_{slow_period}'
            
            fast_sma = indicator_values.get(fast_sma_key)
            slow_sma = indicator_values.get(slow_sma_key)
            
            if fast_sma is None or slow_sma is None:
                if prev_indicator_values is None and indicator_values:
                    prev_indicators[symbol] = indicator_values.copy()
                continue
            
            # Generate signal for this symbol
            signal = self._generate_signal(row, indicator_values, position, prev_indicator_values)
            
            if indicator_values:
                prev_indicators[symbol] = indicator_values.copy()
            
            # Process trade with shared portfolio capital
            portfolio_capital, new_position, _ = self._process_trade_signal(
                symbol, timestamp, price, signal, position, portfolio_capital, bet_size_pct,
                [], indicators, idx  # Pass empty list since we'll track portfolio equity separately
            )
            
            # Update position for this symbol BEFORE calculating equity
            positions[symbol] = new_position
            
            # Calculate portfolio equity ONCE after trade (much faster with cache)
            # This replaces the expensive DataFrame filtering with O(1) dictionary lookups
            portfolio_equity = portfolio_capital
            for sym, pos in positions.items():
                if pos is not None:
                    # Use cached price lookup instead of DataFrame filtering
                    sym_current_price = get_price_at_timestamp(sym, timestamp)
                    if sym_current_price is None:
                        # Fallback: use entry price if no cache entry found
                        sym_current_price = pos['entry_price']
                    
                    if pos['type'] == 'buy':
                        portfolio_equity += sym_current_price * pos['quantity']
                    else:  # short
                        portfolio_equity -= sym_current_price * pos['quantity']
            
            # Also include the new position if any (use current event price)
            if new_position is not None:
                if new_position['type'] == 'buy':
                    portfolio_equity += price * new_position['quantity']
                else:
                    portfolio_equity -= price * new_position['quantity']
            
            # Update portfolio equity curve
            portfolio_equity_curve.append((timestamp, float(portfolio_equity)))
        
        # Close any open positions at the end
        final_timestamp = all_events[-1][0] if all_events else None
        for symbol, position in positions.items():
            if position is not None:
                test_df = symbol_data[symbol]['test_df']
                final_price = float(test_df.iloc[-1]['close'])
                final_timestamp = test_df.iloc[-1]['timestamp']
                portfolio_capital = self._close_position_at_end(symbol, position, final_price, final_timestamp, portfolio_capital)
        
        # Update final portfolio equity curve point
        if final_timestamp:
            portfolio_equity_curve.append((final_timestamp, float(portfolio_capital)))
        
        # For multi-symbol, use portfolio-level equity curve for all symbols
        # (since capital is shared, all symbols share the same portfolio equity)
        # Ensure chronological order
        portfolio_equity_curve.sort(key=lambda x: x[0])
        for symbol in self.symbols:
            if symbol in symbol_data:
                self.equity_curves[symbol] = portfolio_equity_curve
    
    def _process_trade_signal(self, symbol, timestamp, price, signal, position, capital, bet_size_pct, equity_curve, indicators, row_idx):
        """Process a trade signal and return updated capital, position, and equity curve"""
        
        # Execute trades based on signal
        if signal == 'buy' and position is None:
            # Open long position
            bet_amount = float(capital * bet_size_pct)
            quantity = float(bet_amount / price)
            position = {
                'type': 'buy',
                'entry_price': float(price),
                'entry_timestamp': timestamp,
                'quantity': quantity,
                'bet_amount': bet_amount
            }
            # Reduce capital by bet amount (cash spent to buy shares)
            capital = float(capital - bet_amount)
            logger.debug(f"{symbol.ticker} LONG ENTRY @ {price} on {timestamp}, quantity: {quantity:.4f}, bet_amount: ${bet_amount:.2f}, capital after: ${capital:.2f}")
        
        elif signal == 'sell' and position is None:
            # Open short position
            bet_amount = float(capital * bet_size_pct)
            quantity = float(bet_amount / price)
            position = {
                'type': 'sell',
                'entry_price': float(price),
                'entry_timestamp': timestamp,
                'quantity': quantity,
                'bet_amount': bet_amount
            }
            # Increase capital by bet amount (cash received from selling borrowed shares)
            capital = float(capital + bet_amount)
            logger.debug(f"{symbol.ticker} SHORT ENTRY @ {price} on {timestamp}, quantity: {quantity:.4f}, bet_amount: ${bet_amount:.2f}, capital after: ${capital:.2f}")
        
        elif signal == 'sell' and position is not None and position['type'] == 'buy':
            # Close long position
            exit_price = float(price)
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            pnl = float((exit_price - position['entry_price']) * position['quantity'])
            pnl_percentage = float(((exit_price - position['entry_price']) / position['entry_price']) * 100)
            
            # Calculate maximum drawdown for this trade
            max_drawdown = self._calculate_trade_drawdown(
                symbol, 
                position['entry_timestamp'], 
                timestamp, 
                position['entry_price'], 
                is_long=True
            )
            
            # Record trade
            trade = {
                'symbol': symbol,
                'trade_type': 'buy',
                'entry_price': float(position['entry_price']),
                'exit_price': exit_price,
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': timestamp,
                'quantity': float(position['quantity']),
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'is_winner': pnl > 0,
                'max_drawdown': float(max_drawdown) if max_drawdown is not None else None,
                'metadata': {
                    'bet_amount': float(bet_amount)
                }
            }
            self.trades.append(trade)
            
            # Update capital: add back the bet amount + profit/loss
            capital = float(capital + bet_amount + pnl)
            position = None
            logger.debug(f"{symbol.ticker} LONG EXIT @ {exit_price} on {timestamp}, PnL: {pnl:.2f}, bet_amount returned: ${bet_amount:.2f}, capital after: ${capital:.2f}")
            
            # In ALL mode, immediately open opposite position (short) since crossover already happened
            if self.position_mode == 'all':
                bet_amount = float(capital * bet_size_pct)
                quantity = float(bet_amount / price)
                position = {
                    'type': 'sell',
                    'entry_price': float(price),
                    'entry_timestamp': timestamp,
                    'quantity': quantity,
                    'bet_amount': bet_amount
                }
                # Increase capital by bet amount (cash received from selling borrowed shares)
                capital = float(capital + bet_amount)
                logger.debug(f"{symbol.ticker} SHORT ENTRY (after long exit) @ {price} on {timestamp}, quantity: {quantity:.4f}, bet_amount: ${bet_amount:.2f}, capital after: ${capital:.2f}")
        
        elif signal == 'buy' and position is not None and position['type'] == 'sell':
            # Close short position
            exit_price = float(price)
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            pnl = float((position['entry_price'] - exit_price) * position['quantity'])
            pnl_percentage = float(((position['entry_price'] - exit_price) / position['entry_price']) * 100)
            
            # Calculate maximum drawdown for this trade
            max_drawdown = self._calculate_trade_drawdown(
                symbol, 
                position['entry_timestamp'], 
                timestamp, 
                position['entry_price'], 
                is_long=False
            )
            
            # Record trade
            trade = {
                'symbol': symbol,
                'trade_type': 'sell',
                'entry_price': float(position['entry_price']),
                'exit_price': exit_price,
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': timestamp,
                'quantity': float(position['quantity']),
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'is_winner': pnl > 0,
                'max_drawdown': float(max_drawdown) if max_drawdown is not None else None,
                'metadata': {
                    'bet_amount': float(bet_amount)
                }
            }
            self.trades.append(trade)
            
            # Update capital: return the bet amount we received earlier, then add/subtract profit/loss
            capital = float(capital - bet_amount + pnl)
            position = None
            logger.debug(f"{symbol.ticker} SHORT EXIT @ {exit_price} on {timestamp}, PnL: {pnl:.2f}, bet_amount returned: ${bet_amount:.2f}, capital after: ${capital:.2f}")
            
            # In ALL mode, immediately open opposite position (long) since crossover already happened
            if self.position_mode == 'all':
                bet_amount = float(capital * bet_size_pct)
                quantity = float(bet_amount / price)
                position = {
                    'type': 'buy',
                    'entry_price': float(price),
                    'entry_timestamp': timestamp,
                    'quantity': quantity,
                    'bet_amount': bet_amount
                }
                # Reduce capital by bet amount (cash spent to buy shares)
                capital = float(capital - bet_amount)
                logger.debug(f"{symbol.ticker} LONG ENTRY (after short exit) @ {price} on {timestamp}, quantity: {quantity:.4f}, bet_amount: ${bet_amount:.2f}, capital after: ${capital:.2f}")
        
        # Update equity curve
        current_equity = capital
        if position is not None:
            bet_amount_pos = position.get('bet_amount', position['entry_price'] * position['quantity'])
            if position['type'] == 'buy':
                position_value = price * position['quantity']
                current_equity = capital + position_value
            else:
                position_liability = price * position['quantity']
                current_equity = capital - position_liability
        equity_curve.append((timestamp, current_equity))
        
        return capital, position, equity_curve
    
    def _close_position_at_end(self, symbol, position, final_price, final_timestamp, capital):
        """Close a position at the end of backtest and return updated capital"""
        
        if position['type'] == 'buy':
            # Close long position
            exit_price = final_price
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            pnl = float((exit_price - position['entry_price']) * position['quantity'])
            
            # Record trade
            trade = {
                'symbol': symbol,
                'trade_type': 'buy',
                'entry_price': float(position['entry_price']),
                'exit_price': exit_price,
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': final_timestamp,
                'quantity': float(position['quantity']),
                'pnl': pnl,
                'pnl_percentage': float(((exit_price - position['entry_price']) / position['entry_price']) * 100),
                'is_winner': pnl > 0,
                'max_drawdown': None,
                'metadata': {
                    'closed_at_end': True,
                    'bet_amount': float(bet_amount)
                }
            }
            self.trades.append(trade)
            
            capital = float(capital + bet_amount + pnl)
            logger.debug(f"{symbol.ticker} LONG POSITION CLOSED AT END @ {exit_price}, PnL: {pnl:.2f}, capital: ${capital:.2f}")
        else:
            # Close short position
            exit_price = final_price
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            pnl = float((position['entry_price'] - exit_price) * position['quantity'])
            
            # Record trade
            trade = {
                'symbol': symbol,
                'trade_type': 'sell',
                'entry_price': float(position['entry_price']),
                'exit_price': exit_price,
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': final_timestamp,
                'quantity': float(position['quantity']),
                'pnl': pnl,
                'pnl_percentage': float(((position['entry_price'] - exit_price) / position['entry_price']) * 100),
                'is_winner': pnl > 0,
                'max_drawdown': None,
                'metadata': {
                    'closed_at_end': True,
                    'bet_amount': float(bet_amount)
                }
            }
            self.trades.append(trade)
            
            capital = float(capital - bet_amount + pnl)
            logger.debug(f"{symbol.ticker} SHORT POSITION CLOSED AT END @ {exit_price}, PnL: {pnl:.2f}, capital: ${capital:.2f}")
        
        return capital
    
    def _generate_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None) -> Optional[str]:
        """
        Generate trading signal based on strategy logic
        
        Args:
            row: Current OHLCV row
            indicators: Dictionary of indicator values
            position: Current position (if any)
            prev_indicators: Previous indicator values (for crossover detection)
        
        Returns:
            'buy', 'sell', or None
        """
        strategy_name = self.strategy.name
        
        if strategy_name == 'Simple Moving Average Crossover':
            return self._sma_crossover_signal(row, indicators, position, prev_indicators)
        elif strategy_name == 'RSI Mean Reversion':
            return self._rsi_mean_reversion_signal(row, indicators)
        elif strategy_name == 'Bollinger Bands Breakout':
            return self._bollinger_breakout_signal(row, indicators)
        elif strategy_name == 'MACD Crossover':
            return self._macd_crossover_signal(row, indicators)
        else:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return None
    
    def _sma_crossover_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None) -> Optional[str]:
        """
        SMA Crossover strategy signal - supports both long and short positions
        - Long entry: fast SMA crosses above slow SMA
        - Long exit: fast SMA crosses below slow SMA
        - Short entry: fast SMA crosses below slow SMA
        - Short exit: fast SMA crosses above slow SMA
        
        Respects position_mode: 'all' (both), 'long' (only longs), 'short' (only shorts)
        """
        fast_period = self.parameters.get('fast_period', 20)
        slow_period = self.parameters.get('slow_period', 50)
        
        fast_sma_key = f'SMA_{fast_period}'
        slow_sma_key = f'SMA_{slow_period}'
        
        fast_sma = indicators.get(fast_sma_key)
        slow_sma = indicators.get(slow_sma_key)
        
        if fast_sma is None or slow_sma is None:
            logger.debug(f"SMA crossover: Missing indicators. fast_sma={fast_sma}, slow_sma={slow_sma}, keys={list(indicators.keys())}")
            return None
        
        # Need previous values to detect crossover
        if prev_indicators is None:
            logger.debug(f"SMA crossover: No previous indicators available")
            return None
        
        prev_fast_sma = prev_indicators.get(fast_sma_key)
        prev_slow_sma = prev_indicators.get(slow_sma_key)
        
        if prev_fast_sma is None or prev_slow_sma is None:
            logger.debug(f"SMA crossover: Missing previous indicators. prev_fast={prev_fast_sma}, prev_slow={prev_slow_sma}, prev_keys={list(prev_indicators.keys())}")
            return None
        
        # Check if we have an open position
        has_position = position is not None
        position_type = position['type'] if has_position else None
        
        # Detect crossover: fast crosses above slow (golden cross)
        if prev_fast_sma <= prev_slow_sma and fast_sma > slow_sma:
            if not has_position:
                # No position: Enter LONG (only if position_mode allows longs)
                if self.position_mode in ('all', 'long'):
                    logger.info(f"[{self.position_mode.upper()}] LONG ENTRY signal: Fast SMA ({fast_sma:.2f}) crossed above Slow SMA ({slow_sma:.2f})")
                    return 'buy'
                else:
                    logger.debug(f"[{self.position_mode.upper()}] LONG ENTRY signal ignored (position_mode={self.position_mode})")
            elif position_type == 'sell':
                # Short position open: Exit SHORT (close short = buy) - always allowed
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: Fast SMA ({fast_sma:.2f}) crossed above Slow SMA ({slow_sma:.2f})")
                return 'buy'
        
        # Detect crossover: fast crosses below slow (death cross)
        elif prev_fast_sma >= prev_slow_sma and fast_sma < slow_sma:
            if not has_position:
                # No position: Enter SHORT (only if position_mode allows shorts)
                if self.position_mode in ('all', 'short'):
                    logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: Fast SMA ({fast_sma:.2f}) crossed below Slow SMA ({slow_sma:.2f})")
                    return 'sell'
                else:
                    logger.debug(f"[{self.position_mode.upper()}] SHORT ENTRY signal ignored (position_mode={self.position_mode})")
            elif position_type == 'buy':
                # Long position open: Exit LONG (close long = sell) - always allowed
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: Fast SMA ({fast_sma:.2f}) crossed below Slow SMA ({slow_sma:.2f})")
                return 'sell'
        
        return None
    
    def _rsi_mean_reversion_signal(self, row: pd.Series, indicators: Dict) -> Optional[str]:
        """RSI Mean Reversion strategy signal"""
        rsi_period = self.parameters.get('rsi_period', 14)
        oversold = self.parameters.get('oversold_threshold', 30)
        overbought = self.parameters.get('overbought_threshold', 70)
        
        rsi_key = f'RSI_{rsi_period}'
        rsi = indicators.get(rsi_key) or indicators.get('RSI')
        
        if rsi is None:
            return None
        
        if rsi < oversold:
            return 'buy'
        elif rsi > overbought:
            return 'sell'
        return None
    
    def _bollinger_breakout_signal(self, row: pd.Series, indicators: Dict) -> Optional[str]:
        """Bollinger Bands Breakout strategy signal"""
        price = row['close']
        
        # Bollinger Bands are stored with keys like 'BollingerBands_upper', etc.
        upper = indicators.get('BollingerBands_upper') or indicators.get('upper')
        lower = indicators.get('BollingerBands_lower') or indicators.get('lower')
        
        if upper is None or lower is None:
            return None
        
        if price > upper:
            return 'buy'
        elif price < lower:
            return 'sell'
        return None
    
    def _macd_crossover_signal(self, row: pd.Series, indicators: Dict) -> Optional[str]:
        """MACD Crossover strategy signal"""
        macd_line = indicators.get('MACD') or indicators.get('macd')
        signal_line = indicators.get('MACD_signal') or indicators.get('signal')
        
        if macd_line is None or signal_line is None:
            return None
        
        if macd_line > signal_line:
            return 'buy'
        elif macd_line < signal_line:
            return 'sell'
        return None
    
    def calculate_statistics(self) -> Dict:
        """Calculate backtest statistics"""
        logger.info("Calculating backtest statistics")
        
        stats = {}
        
        # Per-symbol statistics
        for symbol in self.symbols:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            
            if not symbol_trades:
                continue
            
            symbol_stats = self._calculate_symbol_statistics(symbol, symbol_trades)
            stats[symbol] = symbol_stats
        
        # Portfolio-level statistics are calculated separately for each position_mode
        # This is handled in the task which runs the executor multiple times
        portfolio_stats = self._calculate_portfolio_statistics(trade_type_filter=None)
        if portfolio_stats: # Only add if not empty
            stats[None] = portfolio_stats # Use None as key for portfolio-level stats
        
        return stats
    
    def _calculate_symbol_statistics(self, symbol: Symbol, trades: List[Dict]) -> Dict:
        """Calculate statistics for a single symbol"""
        if not trades:
            return {}
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('is_winner', False)]
        losing_trades = [t for t in trades if not t.get('is_winner', False)]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        # Calculate total_pnl_percentage from total_pnl and initial capital, not sum of percentages
        initial_capital = float(self.backtest.initial_capital)
        total_pnl_percentage = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        avg_winner = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loser = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # Equity curve - always calculate from trades for clean, consistent chart
        # This ensures:
        # 1. Equity curve matches total_pnl (calculated from trades only)
        # 2. Clean chart with points only at trade exits (not every data row)
        # 3. Consistency between single-symbol and multi-symbol backtests
        equity_curve_data = self._calculate_symbol_equity_curve_from_trades(trades, initial_capital)
        # Convert to list of tuples for compatibility with drawdown calculation
        equity_curve = [
            (pd.to_datetime(point['timestamp']), point['equity'])
            for point in equity_curve_data
        ]
        
        # Calculate max drawdown - use average of individual trade drawdowns from closed trades only
        # Only include trades that have an exit (closed trades) for max drawdown calculation
        closed_trades = [t for t in trades if t.get('exit_timestamp') is not None]
        trade_drawdowns = [t.get('max_drawdown') for t in closed_trades if t.get('max_drawdown') is not None]
        if trade_drawdowns:
            max_drawdown = sum(trade_drawdowns) / len(trade_drawdowns)
        else:
            max_drawdown = 0
        
        # Calculate drawdown duration from equity curve
        _, max_drawdown_duration = self._calculate_drawdown(equity_curve)
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['pnl_percentage'] / 100 for t in trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Calculate CAGR and total_return - use same calculation as portfolio stats (with timezone handling)
        # For symbol-level stats, total_return should equal total_pnl_percentage since:
        # - equity curve starts at initial_capital and ends at initial_capital + total_pnl
        # - total_return = ((final_equity - initial_equity) / initial_equity) * 100
        # - = ((initial_capital + total_pnl - initial_capital) / initial_capital) * 100
        # - = (total_pnl / initial_capital) * 100 = total_pnl_percentage
        if equity_curve:
            initial_equity = equity_curve[0][1]
            final_equity = equity_curve[-1][1]
            start_ts = equity_curve[0][0]
            end_ts = equity_curve[-1][0]
            
            # Convert to timezone-naive for timedelta calculation (matching portfolio stats)
            if timezone.is_aware(start_ts):
                start_ts = start_ts.astimezone(pytz.utc).replace(tzinfo=None)
            if timezone.is_aware(end_ts):
                end_ts = end_ts.astimezone(pytz.utc).replace(tzinfo=None)
            
            days = (end_ts - start_ts).days if len(equity_curve) > 1 else 0
            
            if days > 0 and initial_equity > 0:
                # Calculate total_return from equity curve
                # For symbol-level stats, this should match total_pnl_percentage (within rounding)
                total_return = ((final_equity - initial_equity) / initial_equity) * 100
                
                # Ensure total_return matches total_pnl_percentage for symbol-level stats
                # (they should be mathematically equivalent, but use equity curve value for consistency)
                # Round to match the precision of total_pnl_percentage
                total_return = round(total_return, 4)
                
                # Calculate CAGR - handle negative equity ratio to avoid complex numbers
                equity_ratio = final_equity / initial_equity
                if equity_ratio > 0:
                    cagr_value = ((equity_ratio ** (365.25 / days)) - 1) * 100
                    # Ensure cagr is a real number (not complex)
                    if isinstance(cagr_value, complex):
                        cagr = 0.0
                    else:
                        cagr = float(cagr_value)
                else:
                    # If final equity is negative or zero, CAGR is not meaningful
                    cagr = 0.0
            else:
                total_return = 0
                cagr = 0
        else:
            total_return = 0
            cagr = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 8),
            'total_pnl_percentage': round(total_pnl_percentage, 4),
            'average_pnl': round(average_pnl, 8),
            'average_winner': round(avg_winner, 8),
            'average_loser': round(avg_loser, 8),
            'profit_factor': round(profit_factor, 4),
            'max_drawdown': round(max_drawdown, 4),
            'max_drawdown_duration': max_drawdown_duration,
            'sharpe_ratio': round(sharpe_ratio, 4),
            'cagr': round(cagr, 4),
            'total_return': round(total_return, 4),
            'equity_curve': equity_curve_data,
        }
    
    def _calculate_symbol_equity_curve_from_trades(self, trades: List[Dict], initial_capital: float) -> List[Dict]:
        """Calculate symbol-specific equity curve from trades only (for multi-symbol backtests)
        
        This ensures the equity curve matches total_pnl which is calculated from trades only.
        """
        equity_curve_data = []
        
        if not trades:
            return equity_curve_data
        
        # Helper to normalize timestamps
        def normalize_timestamp(ts):
            if ts is None:
                return None
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if timezone.is_naive(ts):
                ts = timezone.make_aware(ts)
            return ts
        
        # Get all trades with exits, sort by exit timestamp (when PnL is realized)
        trades_with_exits = [t for t in trades if t.get('exit_timestamp') and t.get('pnl') is not None]
        
        if trades_with_exits:
            # Sort trades by exit timestamp to process chronologically
            sorted_trades = sorted(trades_with_exits, key=lambda t: normalize_timestamp(t['exit_timestamp']) or datetime.min)
            
            # Find the earliest timestamp (entry or exit) to start the curve
            all_timestamps = []
            for trade in trades:
                if trade.get('entry_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
                if trade.get('exit_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['exit_timestamp']))
            
            earliest_ts = min(all_timestamps) if all_timestamps else None
            
            # Start with initial capital at earliest timestamp
            if earliest_ts:
                equity_curve_data.append({
                    'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts),
                    'equity': initial_capital
                })
            
            # Track cumulative PnL as trades exit (when PnL is realized)
            cumulative_pnl = 0.0
            for trade in sorted_trades:
                cumulative_pnl += float(trade['pnl'])
                exit_ts = normalize_timestamp(trade['exit_timestamp'])
                
                current_equity = initial_capital + cumulative_pnl
                equity_curve_data.append({
                    'timestamp': exit_ts.isoformat() if hasattr(exit_ts, 'isoformat') else str(exit_ts),
                    'equity': round(current_equity, 2)
                })
            
            # Sort by timestamp to ensure chronological order BEFORE final point adjustment
            equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
            
            # Remove duplicates (same timestamp, keep the one with correct equity)
            seen_timestamps = {}
            for point in equity_curve_data:
                ts_key = point['timestamp']
                # Keep the last equity value for each timestamp (most up-to-date)
                seen_timestamps[ts_key] = point
            equity_curve_data = list(seen_timestamps.values())
            equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
            
            # Ensure final point matches total_pnl exactly
            if equity_curve_data:
                total_pnl = sum(t['pnl'] for t in trades)
                final_equity = initial_capital + total_pnl
                # Get the latest timestamp from trades
                latest_ts = max((normalize_timestamp(t.get('exit_timestamp') or t.get('entry_timestamp')) for t in trades if t.get('exit_timestamp') or t.get('entry_timestamp')), default=earliest_ts)
                
                # Normalize the last point's timestamp for comparison
                last_point_ts = pd.to_datetime(equity_curve_data[-1]['timestamp'])
                if timezone.is_naive(last_point_ts):
                    last_point_ts = timezone.make_aware(last_point_ts)
                
                # Compare timestamps properly
                if latest_ts and latest_ts != last_point_ts:
                    # Add new point if latest trade timestamp is different
                    equity_curve_data.append({
                        'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts),
                        'equity': round(final_equity, 2)
                    })
                    # Re-sort after adding
                    equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
                else:
                    # Just update equity value
                    equity_curve_data[-1]['equity'] = round(final_equity, 2)
            
            # Ensure final point matches total_pnl
            if equity_curve_data:
                total_pnl = sum(t['pnl'] for t in trades)
                final_equity = initial_capital + total_pnl
                equity_curve_data[-1]['equity'] = round(final_equity, 2)
        else:
            # No trades with exits - create simple curve
            all_timestamps = []
            for trade in trades:
                if trade.get('entry_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
            
            if all_timestamps:
                earliest_ts = min(all_timestamps)
                latest_ts = max(all_timestamps)
                total_pnl = sum(t['pnl'] for t in trades)
                equity_curve_data = [
                    {'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts), 'equity': initial_capital},
                    {'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts), 'equity': round(initial_capital + total_pnl, 2)}
                ]
            else:
                now = timezone.now()
                total_pnl = sum(t['pnl'] for t in trades)
                equity_curve_data = [
                    {'timestamp': now.isoformat(), 'equity': initial_capital},
                    {'timestamp': now.isoformat(), 'equity': round(initial_capital + total_pnl, 2)}
                ]
        
        # Ensure we have at least initial and final points
        if len(equity_curve_data) == 0:
            now = timezone.now()
            total_pnl = sum(t['pnl'] for t in trades)
            equity_curve_data = [
                {'timestamp': now.isoformat(), 'equity': initial_capital},
                {'timestamp': now.isoformat(), 'equity': round(initial_capital + total_pnl, 2)}
            ]
        
        return equity_curve_data
    
    def _calculate_portfolio_statistics(self, trade_type_filter: Optional[str] = None) -> Dict:
        """
        Calculate portfolio-level statistics
        
        Args:
            trade_type_filter: 'buy' for long only, 'sell' for short only, None for all trades
        """
        # Filter trades by type if specified
        if trade_type_filter:
            all_trades = [t for t in self.trades if t.get('trade_type') == trade_type_filter]
        else:
            all_trades = self.trades
        
        if not all_trades:
            return {}
        
        # Similar calculation as symbol statistics but aggregated
        total_trades = len(all_trades)
        winning_trades = [t for t in all_trades if t.get('is_winner', False)]
        losing_trades = [t for t in all_trades if not t.get('is_winner', False)]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in all_trades)
        # Calculate total_pnl_percentage from total_pnl and initial capital, not sum of percentages
        initial_capital = float(self.backtest.initial_capital)
        total_pnl_percentage = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        avg_winner = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loser = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # Calculate max drawdown as average of all individual trade drawdowns from closed trades only
        # Only include trades that have an exit (closed trades) for max drawdown calculation
        closed_trades = [t for t in all_trades if t.get('exit_timestamp') is not None]
        trade_drawdowns = [t.get('max_drawdown') for t in closed_trades if t.get('max_drawdown') is not None]
        if trade_drawdowns:
            max_drawdown = sum(trade_drawdowns) / len(trade_drawdowns)
        else:
            max_drawdown = 0
        
        # Equity curve and CAGR - use the equity curve from this executor instance
        # Each executor instance (ALL, LONG, SHORT) has its own equity curve
        equity_curve_data = []
        max_drawdown_duration = 0
        sharpe_ratio = 0
        cagr = 0
        total_return = 0
        
        # Calculate portfolio equity curve from all trades chronologically
        # This ensures the equity curve matches total_pnl calculation
        initial_capital = float(self.backtest.initial_capital)
        equity_curve_data = []
        
        if all_trades:
            # Helper to normalize timestamps
            def normalize_timestamp(ts):
                if ts is None:
                    return None
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                if timezone.is_naive(ts):
                    ts = timezone.make_aware(ts)
                return ts
            
            # Get all trades with exits, sort by exit timestamp (when PnL is realized)
            trades_with_exits = [t for t in all_trades if t.get('exit_timestamp') and t.get('pnl') is not None]
            
            if trades_with_exits:
                # Sort trades by exit timestamp to process chronologically
                sorted_trades = sorted(trades_with_exits, key=lambda t: normalize_timestamp(t['exit_timestamp']) or datetime.min)
                
                # Find the earliest timestamp (entry or exit) to start the curve
                all_timestamps = []
                for trade in all_trades:
                    if trade.get('entry_timestamp'):
                        all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
                    if trade.get('exit_timestamp'):
                        all_timestamps.append(normalize_timestamp(trade['exit_timestamp']))
                
                earliest_ts = min(all_timestamps) if all_timestamps else None
                
                # Start with initial capital at earliest timestamp
                if earliest_ts:
                    equity_curve_data.append({
                        'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts),
                        'equity': initial_capital
                    })
                
                # Track cumulative PnL as trades exit (when PnL is realized)
                cumulative_pnl = 0.0
                for trade in sorted_trades:
                    cumulative_pnl += float(trade['pnl'])
                    exit_ts = normalize_timestamp(trade['exit_timestamp'])
                    
                    current_equity = initial_capital + cumulative_pnl
                    equity_curve_data.append({
                        'timestamp': exit_ts.isoformat() if hasattr(exit_ts, 'isoformat') else str(exit_ts),
                        'equity': round(current_equity, 2)
                    })
                
                # Sort by timestamp to ensure chronological order BEFORE final point adjustment
                equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
                
                # Remove duplicates (same timestamp, keep the one with correct equity)
                seen_timestamps = {}
                for point in equity_curve_data:
                    ts_key = point['timestamp']
                    # Keep the last equity value for each timestamp (most up-to-date)
                    seen_timestamps[ts_key] = point
                equity_curve_data = list(seen_timestamps.values())
                equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
                
                # Ensure final point matches total_pnl exactly
                if equity_curve_data:
                    final_equity = initial_capital + total_pnl
                    # Get the latest timestamp from trades
                    latest_ts = max((normalize_timestamp(t.get('exit_timestamp') or t.get('entry_timestamp')) for t in all_trades if t.get('exit_timestamp') or t.get('entry_timestamp')), default=earliest_ts)
                    
                    # Normalize the last point's timestamp for comparison
                    last_point_ts = pd.to_datetime(equity_curve_data[-1]['timestamp'])
                    if timezone.is_naive(last_point_ts):
                        last_point_ts = timezone.make_aware(last_point_ts)
                    
                    # Compare timestamps properly
                    if latest_ts and latest_ts != last_point_ts:
                        # Add new point if latest trade timestamp is different
                        equity_curve_data.append({
                            'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts),
                            'equity': round(final_equity, 2)
                        })
                        # Re-sort after adding
                        equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
                    else:
                        # Just update equity value
                        equity_curve_data[-1]['equity'] = round(final_equity, 2)
                
                # Ensure final point matches total_pnl
                if equity_curve_data:
                    final_equity = initial_capital + total_pnl
                    equity_curve_data[-1]['equity'] = round(final_equity, 2)
            else:
                # No trades with exits - create simple curve
                all_timestamps = []
                for trade in all_trades:
                    if trade.get('entry_timestamp'):
                        all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
                
                if all_timestamps:
                    earliest_ts = min(all_timestamps)
                    latest_ts = max(all_timestamps)
                    equity_curve_data = [
                        {'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts), 'equity': initial_capital},
                        {'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts), 'equity': round(initial_capital + total_pnl, 2)}
                    ]
                else:
                    now = timezone.now()
                    equity_curve_data = [
                        {'timestamp': now.isoformat(), 'equity': initial_capital},
                        {'timestamp': now.isoformat(), 'equity': round(initial_capital + total_pnl, 2)}
                    ]
        
        # Ensure we have at least initial and final points
        if len(equity_curve_data) == 0:
            now = timezone.now()
            equity_curve_data = [
                {'timestamp': now.isoformat(), 'equity': initial_capital},
                {'timestamp': now.isoformat(), 'equity': round(initial_capital + total_pnl, 2)}
            ]
        elif len(equity_curve_data) == 1:
            # Add final point if only one point exists
            final_equity = initial_capital + total_pnl
            equity_curve_data.append({
                'timestamp': equity_curve_data[0]['timestamp'],
                'equity': round(final_equity, 2)
            })
        else:
            # Ensure final point matches total_pnl
            final_equity = initial_capital + total_pnl
            equity_curve_data[-1]['equity'] = round(final_equity, 2)
        
        # Convert to list of tuples for drawdown calculation
        equity_curve_tuples = []
        if equity_curve_data:
            for point in equity_curve_data:
                ts_str = point['timestamp']
                try:
                    if isinstance(ts_str, str):
                        ts = pd.to_datetime(ts_str)
                        if timezone.is_naive(ts):
                            ts = timezone.make_aware(ts)
                    else:
                        ts = ts_str
                    equity_curve_tuples.append((ts, point['equity']))
                except:
                    continue
        
        # Calculate drawdown duration from equity curve (but max_drawdown value is from average of trade drawdowns above)
        if equity_curve_tuples:
            _, max_drawdown_duration = self._calculate_drawdown(equity_curve_tuples)
        else:
            max_drawdown_duration = 0
        
        # Calculate Sharpe ratio
        returns = [t['pnl_percentage'] / 100 for t in all_trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Calculate CAGR
        if equity_curve_tuples and len(equity_curve_tuples) > 1:
            initial_equity = equity_curve_tuples[0][1]
            final_equity = equity_curve_tuples[-1][1]
            start_ts = equity_curve_tuples[0][0]
            end_ts = equity_curve_tuples[-1][0]
            
            # Convert to timezone-naive for timedelta calculation if necessary
            if timezone.is_aware(start_ts):
                start_ts = start_ts.astimezone(pytz.utc).replace(tzinfo=None)
            if timezone.is_aware(end_ts):
                end_ts = end_ts.astimezone(pytz.utc).replace(tzinfo=None)

            days = (end_ts - start_ts).days if len(equity_curve_tuples) > 1 else 0

            if days > 0 and initial_equity > 0:
                total_return = ((final_equity - initial_equity) / initial_equity) * 100
                
                # Calculate CAGR - handle negative equity ratio to avoid complex numbers
                equity_ratio = final_equity / initial_equity
                if equity_ratio > 0:
                    cagr_value = ((equity_ratio ** (365.25 / days)) - 1) * 100
                    # Ensure cagr is a real number (not complex)
                    if isinstance(cagr_value, complex):
                        cagr = 0.0
                    else:
                        cagr = float(cagr_value)
                else:
                    # If final equity is negative or zero, CAGR is not meaningful
                    cagr = 0.0
            else:
                total_return = 0
                cagr = 0
        else:
            total_return = 0
            cagr = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 8),
            'total_pnl_percentage': round(total_pnl_percentage, 4),
            'average_pnl': round(average_pnl, 8),
            'average_winner': round(avg_winner, 8),
            'average_loser': round(avg_loser, 8),
            'profit_factor': round(profit_factor, 4),
            'max_drawdown': round(max_drawdown, 4),
            'max_drawdown_duration': max_drawdown_duration,
            'sharpe_ratio': round(sharpe_ratio, 4),
            'cagr': round(cagr, 4),
            'total_return': round(total_return, 4),
            'equity_curve': equity_curve_data,
        }
    
    def _calculate_drawdown(self, equity_curve: List[Tuple]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0
        
        equities = [eq for _, eq in equity_curve]
        peak = equities[0]
        max_drawdown = 0.0
        drawdown_start = None
        max_drawdown_duration = 0
        
        for i, equity in enumerate(equities):
            if equity > peak:
                peak = equity
                drawdown_start = None
            else:
                drawdown = ((peak - equity) / peak) * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    if drawdown_start is None:
                        drawdown_start = i
                if drawdown_start is not None:
                    duration = i - drawdown_start
                    if duration > max_drawdown_duration:
                        max_drawdown_duration = duration
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_trade_drawdown(self, symbol: Symbol, entry_timestamp, exit_timestamp, entry_price: float, is_long: bool) -> Optional[float]:
        """
        Calculate maximum drawdown for a single trade using OHLCV data
        
        Args:
            symbol: Symbol instance
            entry_timestamp: Entry datetime
            exit_timestamp: Exit datetime
            entry_price: Entry price
            is_long: True for long position, False for short
        
        Returns:
            Maximum drawdown percentage, or None if calculation fails
        """
        try:
            # Get OHLCV data for the trade period
            ohlcv_queryset = OHLCV.objects.filter(
                symbol=symbol,
                timeframe='daily',
                timestamp__gte=entry_timestamp,
                timestamp__lte=exit_timestamp
            ).order_by('timestamp')
            
            if not ohlcv_queryset.exists():
                logger.warning(f"No OHLCV data found for {symbol.ticker} between {entry_timestamp} and {exit_timestamp}")
                return None
            
            if is_long:
                # For long positions: find the lowest low price
                lowest_low = ohlcv_queryset.aggregate(models.Min('low'))['low__min']
                if lowest_low is None:
                    return None
                lowest_low = float(lowest_low)
                drawdown = ((entry_price - lowest_low) / entry_price) * 100
                return round(drawdown, 4)
            else:
                # For short positions: find the highest high price
                highest_high = ohlcv_queryset.aggregate(models.Max('high'))['high__max']
                if highest_high is None:
                    return None
                highest_high = float(highest_high)
                drawdown = ((highest_high - entry_price) / entry_price) * 100
                return round(drawdown, 4)
                
        except Exception as e:
            logger.error(f"Error calculating trade drawdown for {symbol.ticker}: {str(e)}")
            return None
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        return sharpe

