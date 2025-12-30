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
        self.broker = backtest.broker
        self.start_date = backtest.start_date
        self.end_date = backtest.end_date
        self.split_ratio = backtest.split_ratio
        self.position_mode = position_mode  # 'all', 'long', or 'short'
        
        # Filter symbols based on broker associations if broker is set
        all_symbols = list(backtest.symbols.all())
        self.symbols = self._filter_symbols_by_broker(all_symbols, position_mode)
        
        # Initialize data storage
        self.ohlcv_data = {}  # {symbol: DataFrame}
        self.indicators = {}  # {symbol: {indicator_name: Series}}
        self.trades = []  # List of trade dicts
        self.equity_curves = {}  # {symbol: [(timestamp, equity), ...]}
    
    def _filter_symbols_by_broker(self, symbols, position_mode):
        """
        Filter symbols based on broker associations and position mode
        
        If a broker is set on the backtest, only include symbols that:
        - Are associated with the broker
        - Support the requested position mode (based on long_active/short_active flags)
        
        Filtering rules:
        - 'all' mode: symbols with both long_active=True AND short_active=True
        - 'long' mode: symbols with long_active=True
        - 'short' mode: symbols with short_active=True
        
        Args:
            symbols: List of Symbol instances
            position_mode: 'all', 'long', or 'short'
        
        Returns:
            Filtered list of Symbol instances
        """
        if not self.broker:
            # No broker set, return all symbols
            logger.debug(f"No broker set on backtest, using all {len(symbols)} symbols")
            return symbols
        
        # Import here to avoid circular imports
        from live_trading.models import SymbolBrokerAssociation
        
        logger.info(f"Filtering {len(symbols)} symbols for broker {self.broker.name} with position_mode={position_mode}")
        
        filtered_symbols = []
        for symbol in symbols:
            try:
                association = SymbolBrokerAssociation.objects.get(
                    symbol=symbol,
                    broker=self.broker
                )
                
                # Check if symbol supports the requested position mode
                if position_mode == 'all':
                    if association.long_active and association.short_active:
                        filtered_symbols.append(symbol)
                elif position_mode == 'long':
                    if association.long_active:
                        filtered_symbols.append(symbol)
                elif position_mode == 'short':
                    if association.short_active:
                        filtered_symbols.append(symbol)
            except SymbolBrokerAssociation.DoesNotExist:
                # Symbol not associated with broker, skip it
                logger.debug(f"Symbol {symbol.ticker} not associated with broker {self.broker.name}, skipping")
                continue
        
        logger.info(f"Filtered to {len(filtered_symbols)} symbols for position_mode={position_mode} (from {len(symbols)} total)")
        
        if not filtered_symbols:
            logger.warning(
                f"No symbols available for broker {self.broker.name} with position_mode={position_mode}. "
                f"This may indicate missing broker associations or incompatible position mode."
            )
        
        return filtered_symbols
        
    def load_data(self):
        """Load OHLCV data for all symbols - uses ALL available data for each symbol"""
        logger.info(f"Loading OHLCV data for {len(self.symbols)} symbols (using all available data)")
        
        for symbol in self.symbols:
            # Fetch ALL OHLCV data for this symbol - no date filtering
            # Each symbol will use its own full date range
            ohlcv_filter = {
                'symbol': symbol,
                'timeframe': 'daily',
            }
            
            # DO NOT filter by start_date or end_date - use all available data
            # This allows each symbol to use its own full historical data range
            
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
            
            # Store the actual data range for this symbol (for logging only)
            symbol_start = df['timestamp'].min().to_pydatetime()
            symbol_end = df['timestamp'].max().to_pydatetime()
            
            # Log the actual data range for this symbol
            logger.info(f"Loaded {len(df)} rows for {symbol.ticker} (date range: {symbol_start.date()} to {symbol_end.date()})")
            
            # Store the data - each symbol uses its own full date range
            self.ohlcv_data[symbol] = df
    
    def compute_indicators(self):
        """Compute required indicators for all symbols using strategy's required_tool_configs"""
        logger.info("Computing indicators for all symbols")
        
        # Use strategy's required_tool_configs to compute indicators
        # This handles different parameter names (fast_period/slow_period vs short_period/long_period)
        from market_data.services.indicator_service import compute_strategy_indicators_for_ohlcv
        
        for symbol, df in self.ohlcv_data.items():
            symbol_indicators = {}
            
            # Convert DataFrame to list of dicts for indicator service
            ohlcv_list = df.to_dict('records')
            
            # Compute indicators using strategy's required_tool_configs
            # This properly handles parameter mapping (e.g., short_period -> period)
            # Use backtest's strategy_parameters (may override default_parameters)
            indicator_values = compute_strategy_indicators_for_ohlcv(
                self.strategy, ohlcv_list, symbol, strategy_parameters=self.parameters
            )
            
            # Convert indicator values to pandas Series aligned with DataFrame index
            # compute_strategy_indicators_for_ohlcv returns dict with structure:
            # {indicator_key: {'values': [...], 'display_name': '...', ...}}
            for indicator_key, indicator_data in indicator_values.items():
                if isinstance(indicator_data, dict) and 'values' in indicator_data:
                    # Extract values list from dict structure
                    values = indicator_data['values']
                    if isinstance(values, list):
                        # Create Series aligned with DataFrame index
                        # Pad with None if values list is shorter than df
                        if len(values) < len(df):
                            values.extend([None] * (len(df) - len(values)))
                        elif len(values) > len(df):
                            values = values[:len(df)]
                        series = pd.Series(values, index=df.index[:len(values)])
                        symbol_indicators[indicator_key] = series
                        
                        # Also add base name for backward compatibility
                        base_name = indicator_key.split('_')[0]
                        if base_name not in symbol_indicators:
                            symbol_indicators[base_name] = series
                elif isinstance(indicator_data, list):
                    # Direct list (legacy format)
                    values = indicator_data
                    if len(values) < len(df):
                        values.extend([None] * (len(df) - len(values)))
                    elif len(values) > len(df):
                        values = values[:len(df)]
                    series = pd.Series(values, index=df.index[:len(values)])
                    symbol_indicators[indicator_key] = series
                    
                    base_name = indicator_key.split('_')[0]
                    if base_name not in symbol_indicators:
                        symbol_indicators[base_name] = series
                elif isinstance(indicator_data, pd.Series):
                    # Already a Series, just store it
                    symbol_indicators[indicator_key] = indicator_data
            
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
            
            # Generate signal - let the strategy-specific signal function handle indicator validation
            # Different strategies may use different parameter names (fast_period/slow_period vs short_period/long_period)
            signal = self._generate_signal(row, indicator_values, position, prev_indicator_values)
            
            if signal:
                logger.info(f"[{self.position_mode.upper()}] Signal at {timestamp}: {signal}")
            
            if indicator_values:
                prev_indicator_values = indicator_values.copy()
            
            # Calculate current equity BEFORE processing trade signal
            # This includes cash + mark-to-market value of open position (if any)
            # We need this to calculate bet_amount correctly (bet_size_pct of current equity, not just cash)
            current_equity = capital
            if position is not None:
                # Add mark-to-market value of open position
                if position['type'] == 'buy':
                    current_equity += price * position['quantity']
                else:  # short
                    current_equity -= price * position['quantity']
            
            # Execute trades based on signal (reuse existing logic)
            # Pass capital (cash) and current_equity (total equity) separately
            # This ensures bet_amount is calculated from total equity while cash tracking remains correct
            capital, position, equity_curve = self._process_trade_signal(
                symbol, timestamp, price, signal, position, capital, bet_size_pct, equity_curve, indicators, idx, current_equity=current_equity
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
            
            # Generate signal for this symbol - let the strategy-specific signal function handle indicator validation
            # Different strategies may use different parameter names (fast_period/slow_period vs short_period/long_period)
            signal = self._generate_signal(row, indicator_values, position, prev_indicator_values)
            
            if indicator_values:
                prev_indicators[symbol] = indicator_values.copy()
            
            # Calculate CURRENT portfolio equity BEFORE processing trade
            # This includes cash + mark-to-market value of all open positions
            # We need this to calculate bet_amount correctly (bet_size_pct of current equity, not just cash)
            current_portfolio_equity = portfolio_capital
            for sym, pos in positions.items():
                if pos is not None:
                    # Use cached price lookup instead of DataFrame filtering
                    sym_current_price = get_price_at_timestamp(sym, timestamp)
                    if sym_current_price is None:
                        # Fallback: use entry price if no cache entry found
                        sym_current_price = pos['entry_price']
                    
                    if pos['type'] == 'buy':
                        current_portfolio_equity += sym_current_price * pos['quantity']
                    else:  # short
                        current_portfolio_equity -= sym_current_price * pos['quantity']
            
            # Process trade: calculate bet_amount from total equity, but pass cash capital
            # This ensures bet_amount scales with portfolio performance while maintaining correct cash tracking
            portfolio_capital, new_position, _ = self._process_trade_signal(
                symbol, timestamp, price, signal, position, portfolio_capital, bet_size_pct,
                [], indicators, idx, current_equity=current_portfolio_equity  # Pass equity for bet_amount calculation
            )
            
            # Update position for this symbol BEFORE calculating equity
            positions[symbol] = new_position
            
            # Calculate portfolio equity AFTER trade (much faster with cache)
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
    
    def _process_trade_signal(self, symbol, timestamp, price, signal, position, capital, bet_size_pct, equity_curve, indicators, row_idx, current_equity=None):
        """Process a trade signal and return updated capital, position, and equity curve
        
        Args:
            capital: Current cash capital (not total equity)
            current_equity: Current total equity (cash + mark-to-market positions). If provided, bet_amount is calculated from this instead of capital.
        """
        
        # Use current_equity for bet_amount calculation if provided (for multi-symbol shared capital)
        # Otherwise use capital (for single-symbol or when equity not provided)
        equity_for_bet = current_equity if current_equity is not None else capital
        
        # Execute trades based on signal
        if signal == 'buy' and position is None:
            # Open long position
            bet_amount = float(equity_for_bet * bet_size_pct)
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
            bet_amount = float(equity_for_bet * bet_size_pct)
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
                # After closing, the new equity is the updated capital (for single-symbol)
                # For multi-symbol, the caller should recalculate portfolio equity
                # Use updated capital as equity (works for single-symbol; multi-symbol handles separately)
                equity_after_close = capital  # Capital now includes the closed position's PnL
                bet_amount = float(equity_after_close * bet_size_pct)
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
        elif strategy_name == 'Moving Average Crossover':
            return self._moving_average_crossover_signal(row, indicators, position, prev_indicators)
        elif strategy_name == 'Gap-Up and Gap-Down':
            return self._gap_up_gap_down_signal(row, indicators, position)
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
    
    def _moving_average_crossover_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None) -> Optional[str]:
        """
        Moving Average Crossover strategy signal - supports LONG and SHORT modes only (ALL disabled)
        - Long entry: Short SMA crosses above Long SMA
        - Long exit: Short SMA crosses below Long SMA
        - Short entry: Short SMA crosses below Long SMA
        - Short exit: Short SMA crosses above Long SMA
        
        Only supports position_mode: 'long' (only longs), 'short' (only shorts)
        ALL mode is intentionally disabled for this strategy
        """
        short_period = self.parameters.get('short_period', 20)
        long_period = self.parameters.get('long_period', 50)
        
        short_key = f'SMA_{short_period}'
        long_key = f'SMA_{long_period}'
        
        short_sma = indicators.get(short_key)
        long_sma = indicators.get(long_key)
        
        if short_sma is None or long_sma is None:
            logger.debug(f"Moving Average Crossover: Missing indicators. short_sma={short_sma}, long_sma={long_sma}, keys={list(indicators.keys())}")
            return None
        
        # Need previous values to detect crossover
        if prev_indicators is None:
            logger.debug(f"Moving Average Crossover: No previous indicators available")
            return None
        
        prev_short_sma = prev_indicators.get(short_key)
        prev_long_sma = prev_indicators.get(long_key)
        
        if prev_short_sma is None or prev_long_sma is None:
            logger.debug(f"Moving Average Crossover: Missing previous indicators. prev_short={prev_short_sma}, prev_long={prev_long_sma}")
            return None
        
        # Check if we have an open position
        has_position = position is not None
        position_type = position['type'] if has_position else None
        
        # Bullish crossover: Short SMA crosses above Long SMA
        if prev_short_sma <= prev_long_sma and short_sma > long_sma:
            if not has_position:
                # No position: Enter LONG (only if position_mode is 'long')
                if self.position_mode == 'long':
                    logger.info(f"[{self.position_mode.upper()}] LONG ENTRY signal: Short SMA ({short_sma:.2f}) crossed above Long SMA ({long_sma:.2f})")
                    return 'buy'
                else:
                    logger.debug(f"[{self.position_mode.upper()}] LONG ENTRY signal ignored (position_mode={self.position_mode}, ALL mode disabled)")
            elif position_type == 'sell':
                # Short position open: Exit SHORT (close short = buy) - always allowed
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: Short SMA ({short_sma:.2f}) crossed above Long SMA ({long_sma:.2f})")
                return 'buy'
        
        # Bearish crossover: Short SMA crosses below Long SMA
        elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
            if not has_position:
                # No position: Enter SHORT (only if position_mode is 'short')
                if self.position_mode == 'short':
                    logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: Short SMA ({short_sma:.2f}) crossed below Long SMA ({long_sma:.2f})")
                    return 'sell'
                else:
                    logger.debug(f"[{self.position_mode.upper()}] SHORT ENTRY signal ignored (position_mode={self.position_mode}, ALL mode disabled)")
            elif position_type == 'buy':
                # Long position open: Exit LONG (close long = sell) - always allowed
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: Short SMA ({short_sma:.2f}) crossed below Long SMA ({long_sma:.2f})")
                return 'sell'
        
        return None
    
    def _gap_up_gap_down_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict]) -> Optional[str]:
        """
        Gap-Up and Gap-Down strategy signal
        - Long entry: returns > threshold × std (gap-up)
        - Short entry: returns < -(threshold × std) (gap-down)
        - Exit: Opposite signal or position not allowed in mode
        
        Supports position_mode: 'long', 'short', 'all' (with flipping in ALL mode)
        
        IMPORTANT: Returns and STD are calculated using only data up to today's open
        to avoid lookahead bias.
        """
        threshold = self.parameters.get('threshold', 0.25)
        std_period = self.parameters.get('std_period', 90)
        
        # Get indicators - Returns and RollingSTD
        # Indicator keys are formatted as: {tool_name}_{period} if period exists, else {tool_name}
        returns_key = 'Returns'
        std_key = f'RollingSTD_{std_period}'
        
        # Try multiple key formats for compatibility
        returns = indicators.get(returns_key) or indicators.get('returns')
        std = indicators.get(std_key) or indicators.get(f'STD_{std_period}') or indicators.get('RollingSTD')
        
        if returns is None or std is None:
            logger.debug(f"Gap-Up and Gap-Down: Missing indicators. returns={returns}, std={std}, keys={list(indicators.keys())}")
            return None
        
        # Check if we have an open position
        has_position = position is not None
        position_type = position['type'] if has_position else None  # "buy" or "sell"
        
        # Calculate signal thresholds
        long_threshold = threshold * std
        short_threshold = -threshold * std
        
        long_signal = returns > long_threshold
        short_signal = returns < short_threshold
        
        # LONG ENTRY/FLIP
        if long_signal:
            if self.position_mode in ('long', 'all'):
                if not has_position:
                    # No position: Enter LONG
                    logger.info(f"[{self.position_mode.upper()}] LONG ENTRY signal: returns ({returns:.4f}) > threshold×std ({long_threshold:.4f})")
                    return 'buy'
                elif self.position_mode == 'all' and position_type == 'sell':
                    # Short position open in ALL mode: Flip to LONG
                    logger.info(f"[{self.position_mode.upper()}] FLIP SHORT→LONG: returns ({returns:.4f}) > threshold×std ({long_threshold:.4f})")
                    return 'buy'
        
        # SHORT ENTRY/FLIP
        if short_signal:
            if self.position_mode in ('short', 'all'):
                if not has_position:
                    # No position: Enter SHORT
                    logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: returns ({returns:.4f}) < -threshold×std ({short_threshold:.4f})")
                    return 'sell'
                elif self.position_mode == 'all' and position_type == 'buy':
                    # Long position open in ALL mode: Flip to SHORT
                    logger.info(f"[{self.position_mode.upper()}] FLIP LONG→SHORT: returns ({returns:.4f}) < -threshold×std ({short_threshold:.4f})")
                    return 'sell'
        
        # EXIT CONDITIONS
        # Exit LONG if opposite signal (gap-down) or mode doesn't allow long
        if has_position and position_type == 'buy':
            if short_signal or self.position_mode == 'short':
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: returns ({returns:.4f})")
                return 'sell'
        
        # Exit SHORT if opposite signal (gap-up) or mode doesn't allow short
        if has_position and position_type == 'sell':
            if long_signal or self.position_mode == 'long':
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: returns ({returns:.4f})")
                return 'buy'
        
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
                # Also handle overflow for very large ratios or very small time periods
                equity_ratio = final_equity / initial_equity
                if equity_ratio > 0:
                    try:
                        # Use logarithmic calculation to avoid overflow: CAGR = (exp(ln(ratio) * (365.25/days)) - 1) * 100
                        # This is mathematically equivalent but more numerically stable
                        safe_days = max(days, 1)
                        years = safe_days / 365.25
                        
                        # Use logarithms to avoid overflow in exponentiation
                        # CAGR = (exp(ln(equity_ratio) / years) - 1) * 100
                        # But we need to cap the exponent to prevent exp() overflow
                        if equity_ratio > 0 and years > 0:
                            # Cap equity ratio before taking log to prevent extreme values
                            max_ratio = 1000.0
                            min_ratio = 0.01
                            capped_ratio = max(min_ratio, min(equity_ratio, max_ratio))
                            
                            log_ratio = np.log(capped_ratio)
                            
                            # Cap the exponent to prevent exp() overflow
                            # exp(700) is near the limit for float64, so cap at 700
                            max_exp_arg = 700.0
                            exp_arg = log_ratio / years
                            
                            if exp_arg > max_exp_arg:
                                # Extreme case: cap CAGR at maximum
                                cagr = 10000.0
                            elif exp_arg < -max_exp_arg:
                                # Extreme loss case: cap at -100%
                                cagr = -100.0
                            else:
                                try:
                                    cagr_value = (np.exp(exp_arg) - 1) * 100
                                    
                                    # Ensure cagr is a real number (not complex or inf/nan)
                                    if isinstance(cagr_value, complex) or not np.isfinite(cagr_value):
                                        # Fallback: cap at reasonable maximum
                                        if equity_ratio > 1000.0:
                                            cagr = 10000.0  # Cap at 10000% for extreme returns
                                        elif equity_ratio < 0.01:
                                            cagr = -100.0  # Cap losses at -100%
                                        else:
                                            cagr = 0.0
                                    else:
                                        cagr = float(cagr_value)
                                        
                                        # Cap CAGR at reasonable maximum (e.g., 10000% annual return)
                                        max_cagr = 10000.0
                                        if cagr > max_cagr:
                                            cagr = max_cagr
                                        elif cagr < -100.0:  # Cap losses at -100% (total loss)
                                            cagr = -100.0
                                except (OverflowError, FloatingPointError):
                                    # If exp still overflows, use fallback
                                    if equity_ratio > 1000.0:
                                        cagr = 10000.0
                                    else:
                                        cagr = 0.0
                        else:
                            cagr = 0.0
                                
                    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError) as e:
                        logger.warning(f"CAGR calculation overflow/error for symbol {symbol.ticker if symbol else 'portfolio'}: {e}. Using 0.0")
                        cagr = 0.0
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
                # Also handle overflow for very large ratios or very small time periods
                equity_ratio = final_equity / initial_equity
                if equity_ratio > 0:
                    try:
                        # Use logarithmic calculation to avoid overflow: CAGR = (exp(ln(ratio) * (365.25/days)) - 1) * 100
                        # This is mathematically equivalent but more numerically stable
                        safe_days = max(days, 1)
                        years = safe_days / 365.25
                        
                        # Use logarithms to avoid overflow in exponentiation
                        # CAGR = (exp(ln(equity_ratio) / years) - 1) * 100
                        # But we need to cap the exponent to prevent exp() overflow
                        if equity_ratio > 0 and years > 0:
                            # Cap equity ratio before taking log to prevent extreme values
                            max_ratio = 1000.0
                            min_ratio = 0.01
                            capped_ratio = max(min_ratio, min(equity_ratio, max_ratio))
                            
                            log_ratio = np.log(capped_ratio)
                            
                            # Cap the exponent to prevent exp() overflow
                            # exp(700) is near the limit for float64, so cap at 700
                            max_exp_arg = 700.0
                            exp_arg = log_ratio / years
                            
                            if exp_arg > max_exp_arg:
                                # Extreme case: cap CAGR at maximum
                                cagr = 10000.0
                            elif exp_arg < -max_exp_arg:
                                # Extreme loss case: cap at -100%
                                cagr = -100.0
                            else:
                                try:
                                    cagr_value = (np.exp(exp_arg) - 1) * 100
                                    
                                    # Ensure cagr is a real number (not complex or inf/nan)
                                    if isinstance(cagr_value, complex) or not np.isfinite(cagr_value):
                                        # Fallback: cap at reasonable maximum
                                        if equity_ratio > 1000.0:
                                            cagr = 10000.0  # Cap at 10000% for extreme returns
                                        elif equity_ratio < 0.01:
                                            cagr = -100.0  # Cap losses at -100%
                                        else:
                                            cagr = 0.0
                                    else:
                                        cagr = float(cagr_value)
                                        
                                        # Cap CAGR at reasonable maximum (e.g., 10000% annual return)
                                        max_cagr = 10000.0
                                        if cagr > max_cagr:
                                            cagr = max_cagr
                                        elif cagr < -100.0:  # Cap losses at -100% (total loss)
                                            cagr = -100.0
                                except (OverflowError, FloatingPointError):
                                    # If exp still overflows, use fallback
                                    if equity_ratio > 1000.0:
                                        cagr = 10000.0
                                    else:
                                        cagr = 0.0
                        else:
                            cagr = 0.0
                                
                    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError) as e:
                        logger.warning(f"CAGR calculation overflow/error for portfolio: {e}. Using 0.0")
                        cagr = 0.0
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

