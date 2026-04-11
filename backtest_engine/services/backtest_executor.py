"""
Backtest Executor Service
Executes trading strategies on historical data and generates trades and statistics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from django.utils import timezone
from django.db import models
from django.db import connections
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from market_data.models import Symbol, OHLCV
from strategies.models import StrategyDefinition
from analytical_tools.indicators import compute_indicator
from market_data.services.indicator_service import compute_indicators_for_ohlcv
import logging
import threading
import bisect
import heapq
from itertools import chain, groupby

logger = logging.getLogger(__name__)

# Float tolerance for "no free cash" (avoid edge-case comparisons)
_CASH_EPS = 1e-9


class BacktestExecutor:
    """Executes backtests for trading strategies"""
    
    def __init__(self, backtest, position_mode='long', preprocessed_data=None, hedge_overlay=None):
        """
        Initialize backtest executor
        
        Args:
            backtest: Backtest model instance
            position_mode: 'long' (only long positions) or 'short' (only short positions)
            preprocessed_data: Optional dict of pre-processed symbol data from Phase 1 {symbol: {'df': DataFrame, 'indicators': dict, 'test_df': DataFrame, 'price_cache': dict}}
            hedge_overlay: Optional dict from compute_trade_hedge_overlay — splits each entry bet between strategy and VIX sleeve
        """
        self.backtest = backtest
        self.strategy = backtest.strategy
        self.parameters = backtest.strategy_parameters
        self.broker = backtest.broker
        self.start_date = backtest.start_date
        self.end_date = backtest.end_date
        self.split_ratio = backtest.split_ratio
        self.position_mode = position_mode  # 'long' or 'short'
        self.preprocessed_data = preprocessed_data  # Pre-processed data from Phase 1
        
        # Filter symbols based on broker associations if broker is set (long vs short capability)
        all_symbols = list(backtest.symbols.all())
        self.symbols = self._filter_symbols_by_broker(all_symbols, position_mode)
        
        # If preprocessed_data is provided, filter symbols to only those that were pre-processed
        if self.preprocessed_data:
            # Only use symbols that exist in preprocessed_data and are in our filtered list
            preprocessed_symbols = set(self.preprocessed_data.keys())
            self.symbols = [s for s in self.symbols if s in preprocessed_symbols]
            logger.info(f"[{self.position_mode.upper()}] Using {len(self.symbols)} pre-processed symbols (out of {len(preprocessed_symbols)} available)")
        
        # Initialize data storage
        self.ohlcv_data = {}  # {symbol: DataFrame}
        self.indicators = {}  # {symbol: {indicator_name: Series}}
        self.trades = []  # List of trade dicts
        self.equity_curves = {}  # {symbol: [(timestamp, equity), ...]}
        self.skipped_trades_count = 0  # Count of trades skipped due to insufficient cash
        self.hedge_overlay = hedge_overlay

    def _hedge_overlay_active(self) -> bool:
        h = self.hedge_overlay
        return bool(
            h
            and not h.get("error")
            and h.get("index_ns")
            and len(h["index_ns"]) > 0
            and getattr(self.backtest, "hedge_enabled", False)
        )

    def _event_ns_utc_midnight(self, ts) -> int:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        t = t.normalize()
        return int(t.value)

    def _hedge_weights_at(self, timestamp) -> Tuple[float, float]:
        if not self._hedge_overlay_active():
            return 1.0, 0.0
        arr = self.hedge_overlay["index_ns"]
        ws = self.hedge_overlay["w_strategy"]
        wh = self.hedge_overlay["w_hedge"]
        ns = self._event_ns_utc_midnight(timestamp)
        i = bisect.bisect_right(arr, ns) - 1
        i = max(0, min(i, len(ws) - 1))
        return float(ws[i]), float(wh[i])

    def _hedge_overlay_compound_factor(self, entry_timestamp, exit_timestamp) -> float:
        if not self._hedge_overlay_active():
            return 1.0
        arr = self.hedge_overlay["index_ns"]
        rh = self.hedge_overlay["r_hedge"]
        ns_e = self._event_ns_utc_midnight(entry_timestamp)
        ns_x = self._event_ns_utc_midnight(exit_timestamp)
        ie = bisect.bisect_right(arr, ns_e) - 1
        ix = bisect.bisect_right(arr, ns_x) - 1
        ie = max(0, min(ie, len(arr) - 1))
        ix = max(0, min(ix, len(arr) - 1))
        if ix <= ie:
            return 1.0
        f = 1.0
        for j in range(ie + 1, ix + 1):
            if j < len(rh):
                f *= 1.0 + float(rh[j])
        return f

    def _filter_symbols_by_broker(self, symbols, position_mode):
        """
        Filter symbols based on broker associations and position mode
        
        If a broker is set on the backtest, only include symbols that:
        - Are associated with the broker
        - Support the requested position mode (based on long_active/short_active flags)
        
        Filtering rules:
        - 'long' mode: symbols with long_active=True
        - 'short' mode: symbols with short_active=True
        
        Args:
            symbols: List of Symbol instances
            position_mode: 'long' or 'short'
        
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
            # First check if symbol is active
            if symbol.status != 'active':
                logger.debug(f"Symbol {symbol.ticker} is not active (status={symbol.status}), skipping")
                continue
            
            try:
                association = SymbolBrokerAssociation.objects.get(
                    symbol=symbol,
                    broker=self.broker
                )
                
                # Check if symbol supports the requested position mode
                if position_mode == 'long':
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
    
    def _load_data_for_symbol(self, symbol):
        """Load OHLCV data for a single symbol (on-demand/lazy loading)"""
        try:
            # Check if already loaded
            if symbol in self.ohlcv_data:
                return self.ohlcv_data[symbol]
            
            logger.info(f"[{self.position_mode.upper()}] Loading OHLCV data for symbol: {symbol.ticker}")
            
            # Fetch ALL OHLCV data for this symbol - no date filtering
            ohlcv_filter = {
                'symbol': symbol,
                'timeframe': 'daily',
            }
            
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
                logger.warning(f"[{self.position_mode.upper()}] No OHLCV data found for {symbol.ticker}")
                self.ohlcv_data[symbol] = pd.DataFrame()  # Store empty DataFrame to avoid reloading
                return self.ohlcv_data[symbol]
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Store the actual data range for this symbol (for logging only)
            symbol_start = df['timestamp'].min().to_pydatetime()
            symbol_end = df['timestamp'].max().to_pydatetime()
            
            # Log the actual data range for this symbol
            logger.info(f"[{self.position_mode.upper()}] Loaded {len(df)} rows for {symbol.ticker} (date range: {symbol_start.date()} to {symbol_end.date()})")
            
            # Cache the loaded data
            self.ohlcv_data[symbol] = df
            return df
        except Exception as e:
            logger.error(f"[{self.position_mode.upper()}] Error loading data for {symbol.ticker}: {str(e)}")
            # Store empty DataFrame to avoid reloading
            self.ohlcv_data[symbol] = pd.DataFrame()
            raise
    
    def load_data(self):
        """Load OHLCV data for all symbols in parallel - uses ALL available data for each symbol"""
        logger.info(f"Loading OHLCV data for {len(self.symbols)} symbols in parallel (using all available data)")
        
        # Use ThreadPoolExecutor to load data in parallel
        # Each symbol's data loading is independent, so parallel execution is safe
        max_workers = min(len(self.symbols), 10)  # Limit to 10 threads to avoid overwhelming the system
        
        # Thread-safe dictionary to store results
        data_lock = threading.Lock()
        results = {}
        
        def process_symbol(symbol):
            """Wrapper function to load data for a single symbol"""
            try:
                symbol_result, df = self._load_data_for_symbol(symbol)
                with data_lock:
                    if df is not None:
                        results[symbol_result] = df
                return symbol_result, df
            except Exception as e:
                logger.error(f"Failed to load data for {symbol.ticker}: {str(e)}")
                # Close database connection in this thread
                connections.close_all()
                raise
        
        # Execute data loading in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_symbol, symbol): symbol 
                for symbol in self.symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    # Result is already stored in results dict by process_symbol
                    future.result()  # This will raise if there was an exception
                except Exception as e:
                    logger.error(f"Exception in data loading for {symbol.ticker}: {str(e)}")
                    # Close database connections in case of error
                    connections.close_all()
                    raise
        
        # Assign all results to self.ohlcv_data (thread-safe since all threads have completed)
        self.ohlcv_data = results
        
        # Close all database connections after parallel execution
        connections.close_all()
        
        logger.info(f"Completed loading OHLCV data for {len(self.ohlcv_data)} symbols")
    
    def _compute_indicators_for_symbol(self, symbol, df):
        """Compute indicators for a single symbol (on-demand/lazy loading)"""
        from market_data.services.indicator_service import compute_strategy_indicators_for_ohlcv
        
        try:
            # Check if already computed
            if symbol in self.indicators:
                return self.indicators[symbol]
            
            logger.info(f"[{self.position_mode.upper()}] Computing indicators for symbol: {symbol.ticker}")
            
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
            
            # Cache the computed indicators
            self.indicators[symbol] = symbol_indicators
            logger.info(f"[{self.position_mode.upper()}] Computed {len(symbol_indicators)} indicators for {symbol.ticker}")
            
            return symbol_indicators
        except Exception as e:
            logger.error(f"[{self.position_mode.upper()}] Error computing indicators for {symbol.ticker}: {str(e)}")
            # Store empty dict to avoid recomputing
            self.indicators[symbol] = {}
            raise
    
    def compute_indicators(self):
        """Compute required indicators for all symbols using strategy's required_tool_configs (parallel execution)"""
        logger.info(f"Computing indicators for {len(self.ohlcv_data)} symbols in parallel")
        
        # Use ThreadPoolExecutor to compute indicators in parallel
        # Each symbol's indicator computation is independent, so parallel execution is safe
        max_workers = min(len(self.ohlcv_data), 10)  # Limit to 10 threads to avoid overwhelming the system
        
        # Thread-safe dictionary to store results
        indicators_lock = threading.Lock()
        results = {}
        
        def process_symbol(symbol, df):
            """Wrapper function to compute indicators for a single symbol"""
            try:
                symbol_result, symbol_indicators = self._compute_indicators_for_symbol(symbol, df)
                with indicators_lock:
                    results[symbol_result] = symbol_indicators
                logger.info(f"Computed {len(symbol_indicators)} indicators for {symbol_result.ticker}: {list(symbol_indicators.keys())[:5]}")
                return symbol_result, symbol_indicators
            except Exception as e:
                logger.error(f"Failed to compute indicators for {symbol.ticker}: {str(e)}")
                # Close database connection in this thread
                connections.close_all()
                raise
        
        # Execute indicator computation in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_symbol, symbol, df): symbol 
                for symbol, df in self.ohlcv_data.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    # Result is already stored in results dict by process_symbol
                    future.result()  # This will raise if there was an exception
                except Exception as e:
                    logger.error(f"Exception in indicator computation for {symbol.ticker}: {str(e)}")
                    # Close database connections in case of error
                    connections.close_all()
                    raise
        
        # Assign all results to self.indicators (thread-safe since all threads have completed)
        self.indicators = results
        
        # Close all database connections after parallel execution
        connections.close_all()
        
        logger.info(f"Completed computing indicators for {len(self.indicators)} symbols")
    
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
        
        # Load data on-demand
        df = self._load_data_for_symbol(symbol)
        
        # Validate we have data
        if df.empty:
            logger.warning(f"[{self.position_mode.upper()}] No data available for symbol {symbol.ticker}, skipping")
            return
        
        # Compute indicators on-demand
        indicators = self._compute_indicators_for_symbol(symbol, df)
        
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
        cash_available = initial_capital
        cash_invested = 0.0
        
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
            signal = self._generate_signal(row, indicator_values, position, prev_indicator_values, symbol)
            
            if signal:
                logger.info(f"[{self.position_mode.upper()}] Signal at {timestamp}: {signal}")
            
            if indicator_values:
                prev_indicator_values = indicator_values.copy()
            
            # Execute trades based on signal
            cash_available, cash_invested, position, equity_curve = self._process_trade_signal(
                symbol, timestamp, price, signal, position, cash_available, cash_invested, bet_size_pct, equity_curve, indicators, idx
            )
        
        # Close any open position at the end
        if position is not None:
            final_price = float(test_df.iloc[-1]['close'])
            final_timestamp = test_df.iloc[-1]['timestamp']
            cash_available, cash_invested = self._close_position_at_end(symbol, position, final_price, final_timestamp, cash_available, cash_invested)
            final_equity = cash_available + cash_invested
            equity_curve.append((final_timestamp, float(final_equity)))
        
        # Store equity curve
        self.equity_curves[symbol] = equity_curve
    
    @staticmethod
    def _same_timestamp_portfolio_priority(signal, position) -> int:
        """Within one calendar bar across symbols, process exits before entries.
        
        heapq.merge only orders by time; ties use arbitrary symbol order, so an entry
        could run before another symbol's exit at the same timestamp and see stale
        cash_available (still 0). Lower priority value runs first.
        """
        if signal is None:
            return 2
        if position is None:
            return 1 if signal in ('buy', 'sell') else 2
        if position['type'] == 'buy' and signal == 'sell':
            return 0
        if position['type'] == 'sell' and signal == 'buy':
            return 0
        return 2
    
    def _execute_strategy_multi_symbol(self):
        """Execute strategy for multiple symbols with shared portfolio capital
        
        Phase 2: Uses pre-processed data from Phase 1 (if available) or loads on-demand.
        Collects events and processes them chronologically with shared capital.
        """
        logger.info(f"Executing multi-symbol strategy with shared capital for {len(self.symbols)} symbols")
        
        # Initialize shared portfolio cash management
        initial_capital = float(self.backtest.initial_capital)
        bet_size_pct = float(self.backtest.bet_size_percentage) / 100.0
        portfolio_cash_available = initial_capital
        portfolio_cash_invested = 0.0
        
        # Track positions per symbol: {symbol: position_dict}
        positions = {}  # {symbol: {'type': 'buy'|'sell', 'entry_price': float, ...}}
        
        # Track previous indicator values per symbol
        prev_indicators = {}  # {symbol: {indicator_name: value}}
        
        # Store processed symbol data for later chronological processing
        symbol_data = {}  # {symbol: {'df': DataFrame, 'test_df': DataFrame, 'indicators': dict, 'price_cache': dict, 'sorted_timestamps': list}}
        
        # Collect events per symbol as (timestamp, symbol, idx) — no row/Series in heap
        # (avoids millions of Series copies and a giant merged list in memory)
        symbol_events = {}  # {symbol: list of (timestamp, symbol, idx)}
        
        # Single loop: for each symbol, use pre-processed data or load on-demand, then collect events
        for symbol in self.symbols:
            # Check if we have pre-processed data for this symbol
            if self.preprocessed_data and symbol in self.preprocessed_data:
                # Use pre-processed data from Phase 1
                preprocessed = self.preprocessed_data[symbol]
                df = preprocessed['df']
                indicators = preprocessed['indicators']
                test_df = preprocessed['test_df']
                price_cache = preprocessed['price_cache']
                logger.debug(f"[{self.position_mode.upper()}] Using pre-processed data for {symbol.ticker}")
            else:
                # Load data on-demand (fallback if preprocessed_data not available)
                logger.info(f"[{self.position_mode.upper()}] Loading data on-demand for {symbol.ticker}")
                df = self._load_data_for_symbol(symbol)
                
                if df.empty:
                    logger.warning(f"[{self.position_mode.upper()}] No data available for symbol {symbol.ticker}, skipping")
                    continue
                
                # Compute indicators on-demand
                indicators = self._compute_indicators_for_symbol(symbol, df)
                
                # Split data
                split_idx = int(len(df) * self.split_ratio)
                if split_idx >= len(df):
                    split_idx = max(0, len(df) - 1)
                
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]
                
                if test_df.empty:
                    logger.warning(f"[{self.position_mode.upper()}] No test data available for symbol {symbol.ticker} after split, skipping")
                    continue
                
                # Build price cache (vectorized; avoid iterrows)
                price_cache = {}
                _sub = test_df[test_df['close'].notna()]
                for ts, cl in zip(_sub['timestamp'], _sub['close']):
                    price_cache[ts] = float(cl)
            
            # OPTIMIZATION: Create sorted timestamp list for binary search price lookups
            # This enables O(log n) lookups instead of O(n) list comprehension
            sorted_timestamps = sorted(price_cache.keys())
            
            # Store symbol data for chronological processing
            symbol_data[symbol] = {
                'df': df,
                'test_df': test_df,
                'indicators': indicators,
                'price_cache': price_cache,
                'sorted_timestamps': sorted_timestamps  # For binary search
            }
            positions[symbol] = None
            prev_indicators[symbol] = None
            
            # Events per symbol: same order as test_df (sorted by time) for heapq.merge
            _valid = test_df['close'].notna()
            _sub = test_df.loc[_valid]
            symbol_event_list = [
                (ts, symbol, idx) for idx, ts in zip(_sub.index, _sub['timestamp'])
            ]
            if symbol_event_list:
                symbol_events[symbol] = symbol_event_list
                logger.debug(
                    f"[{self.position_mode.upper()}] Collected {len(symbol_event_list)} events for {symbol.ticker}"
                )
        
        if not symbol_data:
            logger.warning("No valid symbol data for multi-symbol execution")
            return
        
        total_events = sum(len(v) for v in symbol_events.values())
        if symbol_events:
            logger.info(
                f"[{self.position_mode.upper()}] Merged stream: {total_events} events from "
                f"{len(symbol_events)} symbols (lazy heapq.merge, no full materialized list)"
            )
            merged_event_stream = heapq.merge(*symbol_events.values(), key=lambda x: x[0])
        else:
            merged_event_stream = iter(())
            logger.warning(f"[{self.position_mode.upper()}] No events collected from any symbol")
        
        logger.info(
            f"[{self.position_mode.upper()}] Starting strategy execution: processing {total_events} events "
            f"chronologically across {len(symbol_data)} symbols with shared capital"
        )
        
        # OPTIMIZATION: Binary search for price lookup (O(log n) instead of O(n))
        def get_price_at_timestamp(symbol, timestamp):
            """Get price for symbol at timestamp using binary search (O(log n))"""
            price_cache = symbol_data[symbol]['price_cache']
            sorted_timestamps = symbol_data[symbol]['sorted_timestamps']
            
            # Binary search for the closest timestamp <= current timestamp
            # bisect_right returns the position where timestamp would be inserted
            pos = bisect.bisect_right(sorted_timestamps, timestamp)
            
            if pos > 0:
                # Get the most recent price (closest to timestamp)
                closest_ts = sorted_timestamps[pos - 1]
                return price_cache[closest_ts]
            return None
        
        # Stream merge output (peek first row for initial equity point — no materialized all_events list)
        _merged_iter = iter(merged_event_stream)
        _first_ev = next(_merged_iter, None)
        portfolio_equity_curve = []
        if _first_ev is not None:
            portfolio_equity_curve.append((_first_ev[0], initial_capital))
            event_stream = chain([_first_ev], _merged_iter)
        else:
            event_stream = iter(())
        
        last_stream_timestamp = _first_ev[0] if _first_ev is not None else None
        
        # Track logging state for progress updates
        last_logged_symbol = None
        processed_symbols = set()
        event_count = 0
        log_interval = 1000  # Log progress every 1000 events
        
        # OPTIMIZATION: Cache last known price per symbol as we process chronologically
        # Since events are processed in chronological order, we can track the last price
        # for each symbol and use it instead of doing expensive lookups
        last_known_price = {}  # {symbol: last_price} - tracks price as we process chronologically
        
        # Process events chronologically with shared capital (strategy execution happens here).
        # Same-timestamp batching: exits before entries so freed cash is visible before new entries
        # (heapq.merge does not order ties; arbitrary symbol order could process an entry first).
        execution_stopped = False
        for timestamp, group_iter in groupby(event_stream, key=lambda x: x[0]):
            if execution_stopped:
                break
            last_stream_timestamp = timestamp
            batch = list(group_iter)
            phase1 = []
            for _ts, symbol, idx in batch:
                event_count += 1
                test_df = symbol_data[symbol]['test_df']
                row = test_df.loc[idx]
                if symbol != last_logged_symbol:
                    if symbol not in processed_symbols:
                        logger.info(f"[{self.position_mode.upper()}] Processing events for symbol: {symbol.ticker}")
                        processed_symbols.add(symbol)
                    last_logged_symbol = symbol
                if event_count % log_interval == 0:
                    logger.info(f"[{self.position_mode.upper()}] Processed {event_count}/{total_events} events (currently: {symbol.ticker} @ {_ts.date()})")
                price = float(row['close'])
                last_known_price[symbol] = price
                position = positions[symbol]
                indicators = symbol_data[symbol]['indicators']
                prev_indicator_values = prev_indicators[symbol]
                indicator_values = {}
                for name, series in indicators.items():
                    if idx in series.index:
                        try:
                            value = series.loc[idx]
                            if pd.notna(value):
                                indicator_values[name] = float(value)
                        except (KeyError, IndexError, TypeError):
                            pass
                signal = self._generate_signal(row, indicator_values, position, prev_indicator_values, symbol)
                priority = self._same_timestamp_portfolio_priority(signal, position)
                phase1.append((priority, symbol.ticker, _ts, symbol, idx, price, signal, indicator_values))
            phase1.sort(key=lambda x: (x[0], x[1]))
            for _, _, _ts, symbol, idx, price, signal, indicator_values in phase1:
                position = positions[symbol]
                indicators = symbol_data[symbol]['indicators']
                if indicator_values:
                    prev_indicators[symbol] = indicator_values.copy()
                current_portfolio_equity = portfolio_cash_available + portfolio_cash_invested
                if current_portfolio_equity <= 0:
                    logger.info(f"[{self.position_mode.upper()}] Portfolio equity ({current_portfolio_equity:.2f}) <= 0. Stopping execution (account blown up). Individual symbol stats will still be calculated independently.")
                    for sym, pos in positions.items():
                        if pos is not None:
                            if sym == symbol:
                                close_price = price
                            elif sym in last_known_price:
                                close_price = last_known_price[sym]
                            else:
                                close_price = pos['entry_price']
                            portfolio_cash_available, portfolio_cash_invested = self._close_position_at_end(
                                sym, pos, close_price, timestamp, portfolio_cash_available, portfolio_cash_invested
                            )
                    final_equity = portfolio_cash_available + portfolio_cash_invested
                    portfolio_equity_curve.append((timestamp, float(final_equity)))
                    execution_stopped = True
                    break
                portfolio_cash_available, portfolio_cash_invested, new_position, _ = self._process_trade_signal(
                    symbol, timestamp, price, signal, position, portfolio_cash_available, portfolio_cash_invested, bet_size_pct,
                    [], indicators, idx
                )
                positions[symbol] = new_position
                portfolio_equity = portfolio_cash_available + portfolio_cash_invested
                portfolio_equity_curve.append((timestamp, float(portfolio_equity)))
            if execution_stopped:
                break
        
        logger.info(f"[{self.position_mode.upper()}] Completed chronological event processing: {total_events} events processed")
        
        # Close any open positions at the end
        final_timestamp = last_stream_timestamp
        for symbol, position in positions.items():
            if position is not None:
                test_df = symbol_data[symbol]['test_df']
                final_price = float(test_df.iloc[-1]['close'])
                final_timestamp = test_df.iloc[-1]['timestamp']
                portfolio_cash_available, portfolio_cash_invested = self._close_position_at_end(
                    symbol, position, final_price, final_timestamp, portfolio_cash_available, portfolio_cash_invested
                )
        
        # Update final portfolio equity curve point
        if final_timestamp:
            final_equity = portfolio_cash_available + portfolio_cash_invested
            portfolio_equity_curve.append((final_timestamp, float(final_equity)))
        
        # For multi-symbol, use portfolio-level equity curve for all symbols
        # (since capital is shared, all symbols share the same portfolio equity)
        # Ensure chronological order
        portfolio_equity_curve.sort(key=lambda x: x[0])
        for symbol in self.symbols:
            if symbol in symbol_data:
                self.equity_curves[symbol] = portfolio_equity_curve
        
        logger.info(f"[{self.position_mode.upper()}] Multi-symbol execution complete: {len([t for t in self.trades])} trades generated")
    
    def _process_trade_signal(self, symbol, timestamp, price, signal, position, cash_available, cash_invested, bet_size_pct, equity_curve, indicators, row_idx):
        """Process a trade signal and return updated cash_available, cash_invested, position, and equity curve
        
        Args:
            cash_available: Current liquid cash available for new trades
            cash_invested: Total amount currently locked in open positions
            bet_size_pct: Bet size as percentage of equity
            equity_curve: List of (timestamp, equity) tuples
            indicators: Dictionary of indicator values
            row_idx: Current row index in dataframe
        
        Returns:
            (cash_available, cash_invested, position, equity_curve)
        """
        
        # Calculate equity = cash_available + cash_invested
        equity = cash_available + cash_invested
        
        # If equity is negative or zero, account is blown up - don't open new positions
        if equity <= 0:
            logger.debug(f"{symbol.ticker} Skipping position opening: equity ({equity:.2f}) <= 0 (account blown up)")
            return cash_available, cash_invested, position, equity_curve
        
        # Execute trades based on signal
        if signal == 'buy' and position is None:
            # Open long position
            bet_amount = float(equity * bet_size_pct)
            w_s, w_h = self._hedge_weights_at(timestamp)
            bet_strategy = float(bet_amount * w_s)
            bet_hedge = float(bet_amount * w_h)
            quantity = float(bet_strategy / price) if price > 0 else 0.0
            
            # Ensure bet_amount and quantity are positive (safety check)
            if bet_amount <= 0 or quantity <= 0:
                logger.warning(f"{symbol.ticker} Skipping long entry: bet_amount={bet_amount:.2f}, quantity={quantity:.4f} (must be > 0)")
                return cash_available, cash_invested, position, equity_curve
            
            # No deployable cash (all capital already in positions)
            effective_cash = max(0.0, float(cash_available))
            if effective_cash <= _CASH_EPS:
                self.skipped_trades_count += 1
                logger.debug(f"{symbol.ticker} Skipping long entry: no free cash (cash_available={cash_available:.2f})")
                return cash_available, cash_invested, position, equity_curve
            
            # Check if we have enough cash available (use max(0,.) so tiny float negatives don't bypass)
            if bet_amount > effective_cash:
                self.skipped_trades_count += 1
                logger.debug(f"{symbol.ticker} Skipping long entry: bet_amount ${bet_amount:.2f} > cash_available ${cash_available:.2f} (insufficient cash)")
                return cash_available, cash_invested, position, equity_curve
            
            # Open position: subtract from cash_available, add to cash_invested
            cash_available = float(cash_available - bet_amount)
            cash_invested = float(cash_invested + bet_amount)
            
            position = {
                'type': 'buy',
                'entry_price': float(price),
                'entry_timestamp': timestamp,
                'quantity': quantity,
                'bet_amount': bet_amount,
                'bet_strategy': bet_strategy,
                'bet_hedge': bet_hedge,
                'cash_invested': bet_amount  # Track cash invested in this position
            }
            logger.debug(
                f"{symbol.ticker} LONG ENTRY @ {price} on {timestamp}, quantity: {quantity:.4f}, "
                f"bet_total: ${bet_amount:.2f} (strategy ${bet_strategy:.2f}, hedge ${bet_hedge:.2f}), "
                f"cash_available: ${cash_available:.2f}, cash_invested: ${cash_invested:.2f}"
            )
        
        elif signal == 'sell' and position is None:
            # Open short position
            bet_amount = float(equity * bet_size_pct)
            w_s, w_h = self._hedge_weights_at(timestamp)
            bet_strategy = float(bet_amount * w_s)
            bet_hedge = float(bet_amount * w_h)
            quantity = float(bet_strategy / price) if price > 0 else 0.0
            
            # Ensure bet_amount and quantity are positive (safety check)
            if bet_amount <= 0 or quantity <= 0:
                logger.warning(f"{symbol.ticker} Skipping short entry: bet_amount={bet_amount:.2f}, quantity={quantity:.4f} (must be > 0)")
                return cash_available, cash_invested, position, equity_curve
            
            effective_cash = max(0.0, float(cash_available))
            if effective_cash <= _CASH_EPS:
                self.skipped_trades_count += 1
                logger.debug(f"{symbol.ticker} Skipping short entry: no free cash (cash_available={cash_available:.2f})")
                return cash_available, cash_invested, position, equity_curve
            
            # For shorts, we need cash to cover potential losses, so we still check cash_available
            if bet_amount > effective_cash:
                self.skipped_trades_count += 1
                logger.debug(f"{symbol.ticker} Skipping short entry: bet_amount ${bet_amount:.2f} > cash_available ${cash_available:.2f} (insufficient cash)")
                return cash_available, cash_invested, position, equity_curve
            
            # Open short position: subtract from cash_available, add to cash_invested
            # Note: For shorts, we lock up cash as margin/collateral
            cash_available = float(cash_available - bet_amount)
            cash_invested = float(cash_invested + bet_amount)
            
            position = {
                'type': 'sell',
                'entry_price': float(price),
                'entry_timestamp': timestamp,
                'quantity': quantity,
                'bet_amount': bet_amount,
                'bet_strategy': bet_strategy,
                'bet_hedge': bet_hedge,
                'cash_invested': bet_amount  # Track cash invested in this position
            }
            logger.debug(
                f"{symbol.ticker} SHORT ENTRY @ {price} on {timestamp}, quantity: {quantity:.4f}, "
                f"bet_total: ${bet_amount:.2f} (strategy ${bet_strategy:.2f}, hedge ${bet_hedge:.2f}), "
                f"cash_available: ${cash_available:.2f}, cash_invested: ${cash_invested:.2f}"
            )
        
        elif signal == 'sell' and position is not None and position['type'] == 'buy':
            # Close long position (EXIT)
            exit_price = float(price)
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            cash_invested_in_position = float(position.get('cash_invested', bet_amount))
            bet_strategy = float(position.get('bet_strategy', bet_amount))
            bet_hedge = float(position.get('bet_hedge', 0.0))
            strategy_pnl = float((exit_price - position['entry_price']) * position['quantity'])
            h_factor = self._hedge_overlay_compound_factor(position['entry_timestamp'], timestamp)
            hedge_proceeds = float(bet_hedge * h_factor)
            hedge_pnl = float(hedge_proceeds - bet_hedge)
            pnl = float(strategy_pnl + hedge_pnl)
            pnl_percentage = float((pnl / bet_amount * 100) if bet_amount > 1e-12 else 0.0)
            
            # Calculate maximum drawdown for this trade
            max_drawdown = self._calculate_trade_drawdown(
                symbol, 
                position['entry_timestamp'], 
                timestamp, 
                position['entry_price'], 
                is_long=True
            )
            
            # Record trade (completed long position: entry + exit)
            trade = {
                'symbol': symbol,
                'trade_type': 'buy',  # 'buy' indicates long position
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
                    'bet_amount': float(bet_amount),
                    'bet_strategy': bet_strategy,
                    'bet_hedge': bet_hedge,
                    'strategy_pnl': strategy_pnl,
                    'hedge_pnl': hedge_pnl,
                    'hedge_proceeds': hedge_proceeds,
                    'action_type': 'long_exit',  # This trade represents closing a long position
                    'position_type': 'long',  # The position type that was closed
                    'entry_action': 'long_entry',  # When this position was opened (entry)
                    'exit_action': 'long_exit'  # When this position was closed (exit)
                }
            }
            self.trades.append(trade)
            
            # Return strategy leg + hedge sleeve (hedge compounding applied through exit day)
            return_amount = float(bet_strategy + strategy_pnl + hedge_proceeds)
            if return_amount < 0:
                return_amount = 0.0
            
            # Update cash: add return to cash_available, subtract cash_invested_in_position from cash_invested
            cash_available = float(cash_available + return_amount)
            cash_invested = float(cash_invested - cash_invested_in_position)
            
            position = None
            logger.debug(
                f"{symbol.ticker} LONG EXIT @ {exit_price} on {timestamp}, total PnL: {pnl:.2f} "
                f"(strat {strategy_pnl:.2f}, hedge {hedge_pnl:.2f}), return: ${return_amount:.2f}, "
                f"cash_available: ${cash_available:.2f}, cash_invested: ${cash_invested:.2f}"
            )
        
        elif signal == 'buy' and position is not None and position['type'] == 'sell':
            # Close short position (EXIT)
            exit_price = float(price)
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            cash_invested_in_position = float(position.get('cash_invested', bet_amount))
            bet_strategy = float(position.get('bet_strategy', bet_amount))
            bet_hedge = float(position.get('bet_hedge', 0.0))
            strategy_pnl = float((position['entry_price'] - exit_price) * position['quantity'])
            h_factor = self._hedge_overlay_compound_factor(position['entry_timestamp'], timestamp)
            hedge_proceeds = float(bet_hedge * h_factor)
            hedge_pnl = float(hedge_proceeds - bet_hedge)
            pnl = float(strategy_pnl + hedge_pnl)
            pnl_percentage = float((pnl / bet_amount * 100) if bet_amount > 1e-12 else 0.0)
            
            # Calculate maximum drawdown for this trade
            max_drawdown = self._calculate_trade_drawdown(
                symbol, 
                position['entry_timestamp'], 
                timestamp, 
                position['entry_price'], 
                is_long=False
            )
            
            # Record trade (completed short position: entry + exit)
            trade = {
                'symbol': symbol,
                'trade_type': 'sell',  # 'sell' indicates short position
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
                    'bet_amount': float(bet_amount),
                    'bet_strategy': bet_strategy,
                    'bet_hedge': bet_hedge,
                    'strategy_pnl': strategy_pnl,
                    'hedge_pnl': hedge_pnl,
                    'hedge_proceeds': hedge_proceeds,
                    'action_type': 'short_exit',  # This trade represents closing a short position
                    'position_type': 'short',  # The position type that was closed
                    'entry_action': 'short_entry',  # When this position was opened (entry)
                    'exit_action': 'short_exit'  # When this position was closed (exit)
                }
            }
            self.trades.append(trade)
            
            return_amount = float(bet_strategy + strategy_pnl + hedge_proceeds)
            if return_amount < 0:
                return_amount = 0.0
            
            # Update cash: add return to cash_available, subtract cash_invested_in_position from cash_invested
            cash_available = float(cash_available + return_amount)
            cash_invested = float(cash_invested - cash_invested_in_position)
            
            position = None
            logger.debug(
                f"{symbol.ticker} SHORT EXIT @ {exit_price} on {timestamp}, total PnL: {pnl:.2f} "
                f"(strat {strategy_pnl:.2f}, hedge {hedge_pnl:.2f}), return: ${return_amount:.2f}, "
                f"cash_available: ${cash_available:.2f}, cash_invested: ${cash_invested:.2f}"
            )
        
        # Update equity curve: equity = cash_available + cash_invested
        # Note: cash_invested already includes all open positions, so we don't need to mark-to-market
        current_equity = cash_available + cash_invested
        equity_curve.append((timestamp, current_equity))
        
        return cash_available, cash_invested, position, equity_curve
    
    def _close_position_at_end(self, symbol, position, final_price, final_timestamp, cash_available, cash_invested):
        """Close a position at the end of backtest and return updated cash_available and cash_invested"""
        
        if position['type'] == 'buy':
            # Close long position
            exit_price = final_price
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            cash_invested_in_position = float(position.get('cash_invested', bet_amount))
            bet_strategy = float(position.get('bet_strategy', bet_amount))
            bet_hedge = float(position.get('bet_hedge', 0.0))
            strategy_pnl = float((exit_price - position['entry_price']) * position['quantity'])
            h_factor = self._hedge_overlay_compound_factor(position['entry_timestamp'], final_timestamp)
            hedge_proceeds = float(bet_hedge * h_factor)
            hedge_pnl = float(hedge_proceeds - bet_hedge)
            pnl = float(strategy_pnl + hedge_pnl)
            pnl_pct = float((pnl / bet_amount * 100) if bet_amount > 1e-12 else 0.0)
            
            # Record trade (completed long position closed at end of backtest)
            trade = {
                'symbol': symbol,
                'trade_type': 'buy',  # 'buy' indicates long position
                'entry_price': float(position['entry_price']),
                'exit_price': exit_price,
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': final_timestamp,
                'quantity': float(position['quantity']),
                'pnl': pnl,
                'pnl_percentage': pnl_pct,
                'is_winner': pnl > 0,
                'max_drawdown': None,
                'metadata': {
                    'closed_at_end': True,
                    'bet_amount': float(bet_amount),
                    'bet_strategy': bet_strategy,
                    'bet_hedge': bet_hedge,
                    'strategy_pnl': strategy_pnl,
                    'hedge_pnl': hedge_pnl,
                    'hedge_proceeds': hedge_proceeds,
                    'action_type': 'long_exit',  # Clear: this is exiting a long position
                    'position_type': 'long'  # The position type that was closed
                }
            }
            self.trades.append(trade)
            
            return_amount = float(bet_strategy + strategy_pnl + hedge_proceeds)
            if return_amount < 0:
                return_amount = 0.0
            
            # Update cash: add return to cash_available, subtract cash_invested_in_position from cash_invested
            cash_available = float(cash_available + return_amount)
            cash_invested = float(cash_invested - cash_invested_in_position)
            
            logger.debug(
                f"{symbol.ticker} LONG POSITION CLOSED AT END @ {exit_price}, total PnL: {pnl:.2f}, "
                f"return: ${return_amount:.2f}, cash_available: ${cash_available:.2f}, cash_invested: ${cash_invested:.2f}"
            )
        else:
            # Close short position (EXIT)
            exit_price = final_price
            bet_amount = float(position.get('bet_amount', position['entry_price'] * position['quantity']))
            cash_invested_in_position = float(position.get('cash_invested', bet_amount))
            bet_strategy = float(position.get('bet_strategy', bet_amount))
            bet_hedge = float(position.get('bet_hedge', 0.0))
            strategy_pnl = float((position['entry_price'] - exit_price) * position['quantity'])
            h_factor = self._hedge_overlay_compound_factor(position['entry_timestamp'], final_timestamp)
            hedge_proceeds = float(bet_hedge * h_factor)
            hedge_pnl = float(hedge_proceeds - bet_hedge)
            pnl = float(strategy_pnl + hedge_pnl)
            pnl_pct = float((pnl / bet_amount * 100) if bet_amount > 1e-12 else 0.0)
            
            # Record trade (completed short position closed at end of backtest)
            trade = {
                'symbol': symbol,
                'trade_type': 'sell',  # 'sell' indicates short position
                'entry_price': float(position['entry_price']),
                'exit_price': exit_price,
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': final_timestamp,
                'quantity': float(position['quantity']),
                'pnl': pnl,
                'pnl_percentage': pnl_pct,
                'is_winner': pnl > 0,
                'max_drawdown': None,
                'metadata': {
                    'closed_at_end': True,
                    'bet_amount': float(bet_amount),
                    'bet_strategy': bet_strategy,
                    'bet_hedge': bet_hedge,
                    'strategy_pnl': strategy_pnl,
                    'hedge_pnl': hedge_pnl,
                    'hedge_proceeds': hedge_proceeds,
                    'action_type': 'short_exit',  # Clear: this is exiting a short position
                    'position_type': 'short'  # The position type that was closed
                }
            }
            self.trades.append(trade)
            
            return_amount = float(bet_strategy + strategy_pnl + hedge_proceeds)
            if return_amount < 0:
                return_amount = 0.0
            
            # Update cash: add return to cash_available, subtract cash_invested_in_position from cash_invested
            cash_available = float(cash_available + return_amount)
            cash_invested = float(cash_invested - cash_invested_in_position)
            
            logger.debug(
                f"{symbol.ticker} SHORT POSITION CLOSED AT END @ {exit_price}, total PnL: {pnl:.2f}, "
                f"return: ${return_amount:.2f}, cash_available: ${cash_available:.2f}, cash_invested: ${cash_invested:.2f}"
            )
        
        return cash_available, cash_invested
    
    def _generate_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None, symbol: Symbol = None) -> Optional[str]:
        """
        Generate trading signal based on strategy logic
        
        Args:
            row: Current OHLCV row
            indicators: Dictionary of indicator values
            position: Current position (if any)
            prev_indicators: Previous indicator values (for crossover detection)
            symbol: Symbol instance (for broker capability checks when applicable)
        
        Returns:
            'buy', 'sell', or None
        """
        strategy_name = self.strategy.name
        
        if strategy_name == 'Simple Moving Average Crossover':
            return self._sma_crossover_signal(row, indicators, position, prev_indicators, symbol)
        elif strategy_name == 'Moving Average Crossover':
            return self._moving_average_crossover_signal(row, indicators, position, prev_indicators, symbol)
        elif strategy_name == 'Gap-Up and Gap-Down':
            return self._gap_up_gap_down_signal(row, indicators, position, symbol)
        elif strategy_name == 'RSI Mean Reversion':
            return self._rsi_mean_reversion_signal(row, indicators)
        elif strategy_name == 'Bollinger Bands Breakout':
            return self._bollinger_breakout_signal(row, indicators)
        elif strategy_name == 'MACD Crossover':
            return self._macd_crossover_signal(row, indicators)
        else:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return None
    
    def _sma_crossover_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None, symbol: Symbol = None) -> Optional[str]:
        """
        SMA Crossover strategy signal - supports both long and short positions
        - Long entry: fast SMA crosses above slow SMA
        - Long exit: fast SMA crosses below slow SMA
        - Short entry: fast SMA crosses below slow SMA
        - Short exit: fast SMA crosses above slow SMA
        
        Respects position_mode: 'long' (only longs) or 'short' (only shorts)
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
        
        # Broker flags when a broker is configured
        long_allowed = True
        short_allowed = True
        if self.broker and symbol:
            from live_trading.models import SymbolBrokerAssociation
            try:
                association = SymbolBrokerAssociation.objects.get(symbol=symbol, broker=self.broker)
                long_allowed = association.long_active
                short_allowed = association.short_active
            except SymbolBrokerAssociation.DoesNotExist:
                long_allowed = False
                short_allowed = False
        
        # Check if we have an open position
        has_position = position is not None
        position_type = position['type'] if has_position else None
        
        # Detect crossover: fast crosses above slow (golden cross)
        if prev_fast_sma <= prev_slow_sma and fast_sma > slow_sma:
            if not has_position:
                # No position: Enter LONG (only in long mode AND broker allows)
                if self.position_mode == 'long' and long_allowed:
                    logger.info(f"[{self.position_mode.upper()}] LONG ENTRY signal: Fast SMA ({fast_sma:.2f}) crossed above Slow SMA ({slow_sma:.2f})")
                    return 'buy'
                else:
                    logger.debug(f"[{self.position_mode.upper()}] LONG ENTRY signal ignored (position_mode={self.position_mode}, long_allowed={long_allowed})")
            elif position_type == 'sell':
                # Short position open: Exit SHORT (close short = buy) - always allowed
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: Fast SMA ({fast_sma:.2f}) crossed above Slow SMA ({slow_sma:.2f})")
                return 'buy'
        
        # Detect crossover: fast crosses below slow (death cross)
        elif prev_fast_sma >= prev_slow_sma and fast_sma < slow_sma:
            if not has_position:
                # No position: Enter SHORT (only in short mode AND broker allows)
                if self.position_mode == 'short' and short_allowed:
                    logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: Fast SMA ({fast_sma:.2f}) crossed below Slow SMA ({slow_sma:.2f})")
                    return 'sell'
                else:
                    logger.debug(f"[{self.position_mode.upper()}] SHORT ENTRY signal ignored (position_mode={self.position_mode}, short_allowed={short_allowed})")
            elif position_type == 'buy':
                # Long position open: Exit LONG (close long = sell) - always allowed
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: Fast SMA ({fast_sma:.2f}) crossed below Slow SMA ({slow_sma:.2f})")
                return 'sell'
        
        return None
    
    def _moving_average_crossover_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None, symbol: Symbol = None) -> Optional[str]:
        """
        Moving Average Crossover strategy signal — long-only or short-only runs
        - Long entry: Short SMA crosses above Long SMA
        - Long exit: Short SMA crosses below Long SMA
        - Short entry: Short SMA crosses below Long SMA
        - Short exit: Short SMA crosses above Long SMA
        
        Only position_mode: 'long' or 'short'
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
                    logger.debug(f"[{self.position_mode.upper()}] LONG ENTRY signal ignored (position_mode={self.position_mode})")
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
                    logger.debug(f"[{self.position_mode.upper()}] SHORT ENTRY signal ignored (position_mode={self.position_mode})")
            elif position_type == 'buy':
                # Long position open: Exit LONG (close long = sell) - always allowed
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: Short SMA ({short_sma:.2f}) crossed below Long SMA ({long_sma:.2f})")
                return 'sell'
        
        return None
    
    def _gap_up_gap_down_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], symbol: Symbol = None) -> Optional[str]:
        """
        Gap-Up and Gap-Down strategy signal
        - Long entry: returns > threshold × std (gap-up)
        - Short entry: returns < -(threshold × std) (gap-down)
        - Exit: Opposite signal or position not allowed in mode
        
        Supports position_mode: 'long' or 'short' only (no same-bar flip between directions).
        
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
        
        long_allowed = True
        short_allowed = True
        if self.broker and symbol:
            from live_trading.models import SymbolBrokerAssociation
            try:
                association = SymbolBrokerAssociation.objects.get(symbol=symbol, broker=self.broker)
                long_allowed = association.long_active
                short_allowed = association.short_active
            except SymbolBrokerAssociation.DoesNotExist:
                long_allowed = False
                short_allowed = False
        
        # Check if we have an open position
        has_position = position is not None
        position_type = position['type'] if has_position else None  # "buy" or "sell"
        
        # Calculate signal thresholds
        long_threshold = threshold * std
        short_threshold = -threshold * std
        
        long_signal = returns > long_threshold
        short_signal = returns < short_threshold
        
        # EXIT CONDITIONS (check exits first, before entries)
        if has_position and position_type == 'buy':
            if self.position_mode == 'short':
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: mode doesn't allow long")
                return 'sell'
            if self.position_mode == 'long' and short_signal:
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: returns ({returns:.4f})")
                return 'sell'
        
        if has_position and position_type == 'sell':
            if self.position_mode == 'long':
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: mode doesn't allow short")
                return 'buy'
            if self.position_mode == 'short' and long_signal:
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: returns ({returns:.4f})")
                return 'buy'
        
        # ENTRY (no flip: opposite direction closes first on a later bar)
        if long_signal and long_allowed and self.position_mode == 'long' and not has_position:
            logger.info(f"[{self.position_mode.upper()}] LONG ENTRY signal: returns ({returns:.4f}) > threshold×std ({long_threshold:.4f})")
            return 'buy'
        
        if short_signal and short_allowed and self.position_mode == 'short' and not has_position:
            logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: returns ({returns:.4f}) < -threshold×std ({short_threshold:.4f})")
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
        # IMPORTANT: Always include stats for all symbols in self.symbols, even if they have 0 trades
        # This ensures frontend always has a complete stats structure for each mode
        for symbol in self.symbols:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            
            if not symbol_trades:
                # Return empty statistics structure with all fields set to 0/None
                # This ensures consistency across all modes
                stats[symbol] = {
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
                    'independent_bet_amounts': {},
                }
            else:
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
        
        # Equity curve + max drawdown: cumulative realized PnL at exits (same construction as portfolio).
        # Independent bet sizing for UI metadata still comes from the single-symbol simulation below.
        equity_curve_data = self._build_cumulative_equity_curve_from_trades(trades, initial_capital, total_pnl)
        _, independent_bet_amounts = self._calculate_symbol_equity_curve_from_trades(trades, initial_capital)
        equity_curve = [
            (pd.to_datetime(point['timestamp']), float(point['equity']))
            for point in equity_curve_data
        ]
        
        # Intra-trade drawdown (per Trade.max_drawdown): average and worst adverse excursion while position open
        closed_trades = [t for t in trades if t.get('exit_timestamp') is not None]
        trade_drawdowns = [t.get('max_drawdown') for t in closed_trades if t.get('max_drawdown') is not None]
        if trade_drawdowns:
            avg_intra_trade_drawdown = sum(trade_drawdowns) / len(trade_drawdowns)
            worst_intra_trade_drawdown = max(trade_drawdowns)
        else:
            avg_intra_trade_drawdown = 0
            worst_intra_trade_drawdown = 0
        
        # Max drawdown: peak-to-trough on the equity curve (standard definition)
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_curve)
        
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
            'avg_intra_trade_drawdown': round(avg_intra_trade_drawdown, 4),
            'worst_intra_trade_drawdown': round(worst_intra_trade_drawdown, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'cagr': round(cagr, 4),
            'total_return': round(total_return, 4),
            'equity_curve': equity_curve_data,
            'independent_bet_amounts': independent_bet_amounts,  # Map entry_timestamp -> bet_amount for individual symbol view
            'skipped_trades_count': getattr(self, 'skipped_trades_count', 0),  # Count of trades skipped due to insufficient cash
        }
    
    def _calculate_symbol_equity_curve_from_trades(self, trades: List[Dict], initial_capital: float) -> tuple[List[Dict], Dict]:
        """Calculate symbol-specific equity curve by simulating independent trading
        
        This simulates what would have happened if this symbol was traded independently
        with the full initial_capital. It processes trades chronologically, tracking cash
        and positions to build an equity curve that matches what bet_amounts would have
        been if trading independently.
        
        The equity curve shows equity at each point, which should match what bet_amount
        would be calculated from (equity * bet_size_pct).
        
        Returns:
            tuple: (equity_curve_data, independent_bet_amounts)
                - equity_curve_data: List of {timestamp, equity} dicts
                - independent_bet_amounts: Dict mapping entry_timestamp (ISO string) to bet_amount (float)
        """
        equity_curve_data = []
        independent_bet_amounts = {}  # Map entry_timestamp -> bet_amount
        
        if not trades:
            return equity_curve_data, independent_bet_amounts
        
        # Helper to normalize timestamps
        def normalize_timestamp(ts):
            if ts is None:
                return None
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if timezone.is_naive(ts):
                ts = timezone.make_aware(ts)
            return ts
        
        # Get bet_size_pct from backtest
        bet_size_pct = float(self.backtest.bet_size_percentage) / 100.0
        
        # Deduplicate trades - keep only unique trades by (entry_timestamp, exit_timestamp, entry_price, exit_price)
        # This handles cases where the same trade appears multiple times in the database
        seen_trades = {}
        unique_trades = []
        for trade in trades:
            entry_ts = normalize_timestamp(trade.get('entry_timestamp'))
            exit_ts = normalize_timestamp(trade.get('exit_timestamp'))
            entry_price = trade.get('entry_price')
            exit_price = trade.get('exit_price')
            
            # Create a unique key for this trade
            trade_key = (entry_ts, exit_ts, entry_price, exit_price)
            if trade_key not in seen_trades:
                seen_trades[trade_key] = trade
                unique_trades.append(trade)
        
        # Build list of all trade events (entry and exit) from unique trades only
        events = []
        for trade in unique_trades:
            if trade.get('entry_timestamp'):
                events.append({
                    'type': 'entry',
                    'timestamp': normalize_timestamp(trade['entry_timestamp']),
                    'trade': trade
                })
            if trade.get('exit_timestamp') and trade.get('pnl') is not None:
                events.append({
                    'type': 'exit',
                    'timestamp': normalize_timestamp(trade['exit_timestamp']),
                    'trade': trade
                })
        
        if not events:
            return equity_curve_data, independent_bet_amounts
        
        # Sort events chronologically
        events.sort(key=lambda e: (e['timestamp'] or datetime.min, 0 if e['type'] == 'exit' else 1))  # Process exits before entries at same timestamp
        
        # Simulate independent trading: track cash
        cash = initial_capital
        position = None  # {type: 'buy'|'sell', entry_price: float, quantity: float}
        
        # Start with initial capital
        first_timestamp = events[0]['timestamp']
        if first_timestamp:
            equity_curve_data.append({
                'timestamp': first_timestamp.isoformat() if hasattr(first_timestamp, 'isoformat') else str(first_timestamp),
                'equity': initial_capital
            })
        
        # Process events chronologically
        for event in events:
            trade = event['trade']
            timestamp = event['timestamp']
            
            if event['type'] == 'entry':
                # If there's already a position open, close it first (can't have overlapping positions for individual symbol)
                if position is not None:
                    # Close existing position at current price (use entry price as approximation)
                    close_price = float(trade['entry_price'])
                    if position['type'] == 'buy':
                        # Sell the shares
                        close_value = close_price * position['quantity']
                        cash += close_value
                    else:  # short
                        # Buy back the shares
                        close_value = close_price * position['quantity']
                        cash -= close_value
                    position = None
                
                # Calculate current equity (cash only, no open position)
                current_equity = cash
                
                # Calculate bet_amount from current equity (10% of equity)
                bet_amount = float(current_equity * bet_size_pct)
                if bet_amount <= 0 or current_equity <= 0:
                    continue  # Skip if can't bet
                
                # Store independent bet_amount for this trade entry
                entry_timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                independent_bet_amounts[entry_timestamp_str] = round(bet_amount, 2)
                
                # Open position
                entry_price = float(trade['entry_price'])
                quantity = float(bet_amount / entry_price)
                
                if trade['trade_type'] == 'buy':
                    position = {
                        'type': 'buy',
                        'entry_price': entry_price,
                        'quantity': quantity
                    }
                    cash -= bet_amount  # Spend cash to buy shares
                else:  # sell (short)
                    position = {
                        'type': 'sell',
                        'entry_price': entry_price,
                        'quantity': quantity
                    }
                    cash += bet_amount  # Receive cash from short sale
                
                # Calculate equity after opening position (cash + position value)
                current_equity = cash
                if position is not None:
                    if position['type'] == 'buy':
                        current_equity += entry_price * position['quantity']  # Cash + shares value
                    else:  # short
                        current_equity -= entry_price * position['quantity']  # Cash - liability
                
                equity_curve_data.append({
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'equity': round(current_equity, 2)
                })
                
            elif event['type'] == 'exit':
                if position is None:
                    continue  # No position to close
                
                # Close position - calculate PnL based on our independent quantity
                exit_price = float(trade['exit_price'])
                
                if position['type'] == 'buy':
                    # Calculate PnL: (exit_price - entry_price) * quantity
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    # Sell the shares - receive exit_price * quantity
                    cash += exit_price * position['quantity']
                else:  # short
                    # Calculate PnL: (entry_price - exit_price) * quantity (reversed for short)
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                    # Buy back the shares - pay exit_price * quantity
                    cash -= exit_price * position['quantity']
                
                position = None
                
                # Equity after closing position is just cash (no open position)
                current_equity = cash
                
                equity_curve_data.append({
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'equity': round(current_equity, 2)
                })
        
        # After processing all events, sort and clean up
        equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
        
        # Remove duplicates: key by (timestamp, equity) so entry+exit at same time keep two points
        # (timestamp-only dedup broke symbol max drawdown — _calculate_drawdown needs ≥2 distinct values)
        seen_curve_keys = {}
        for point in equity_curve_data:
            k = (point['timestamp'], round(float(point.get('equity', 0.0)), 8))
            seen_curve_keys[k] = point
        equity_curve_data = list(seen_curve_keys.values())
        equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
        
        # CRITICAL: Ensure the first point always starts with initial_capital
        if equity_curve_data and abs(float(equity_curve_data[0].get('equity', 0)) - initial_capital) > 0.01:
            equity_curve_data[0]['equity'] = initial_capital
        
        # If there are open positions at the end, mark-to-market them
        if position is not None and equity_curve_data:
            # Use the last trade's exit price to mark-to-market
            last_trade_with_exit = None
            for trade in trades:
                if trade.get('exit_timestamp'):
                    if last_trade_with_exit is None or normalize_timestamp(trade['exit_timestamp']) > normalize_timestamp(last_trade_with_exit.get('exit_timestamp')):
                        last_trade_with_exit = trade
            
            if last_trade_with_exit:
                final_price = float(last_trade_with_exit['exit_price'])
                final_equity = cash
                if position['type'] == 'buy':
                    final_equity += final_price * position['quantity']
                else:  # short
                    final_equity -= final_price * position['quantity']
                
                # Update or add final point
                latest_ts = max((normalize_timestamp(t.get('exit_timestamp') or t.get('entry_timestamp')) 
                               for t in trades if t.get('exit_timestamp') or t.get('entry_timestamp')), 
                              default=None)
                if latest_ts:
                    equity_curve_data.append({
                        'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts),
                        'equity': round(final_equity, 2)
                    })
                    equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
        
        # Ensure we have at least initial and final points if no events processed
        if len(equity_curve_data) == 0:
            all_timestamps = []
            for trade in trades:
                if trade.get('entry_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
            
            if all_timestamps:
                earliest_ts = min(all_timestamps)
                latest_ts = max(all_timestamps)
                total_pnl = sum(t.get('pnl', 0) for t in trades if t.get('pnl') is not None)
                equity_curve_data = [
                    {'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts), 'equity': initial_capital},
                    {'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts), 'equity': round(initial_capital + total_pnl, 2)}
                ]
            else:
                now = timezone.now()
                total_pnl = sum(t.get('pnl', 0) for t in trades if t.get('pnl') is not None)
                equity_curve_data = [
                    {'timestamp': now.isoformat(), 'equity': initial_capital},
                    {'timestamp': (now + timedelta(days=1)).isoformat(), 'equity': round(initial_capital + total_pnl, 2)}
                ]
        
        # CRITICAL: Ensure the first point always starts with initial_capital (independent equity curve)
        if equity_curve_data:
            equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
            first_point = equity_curve_data[0]
            if abs(float(first_point.get('equity', 0)) - initial_capital) > 0.01:
                first_point['equity'] = initial_capital
        
        return equity_curve_data, independent_bet_amounts
    
    def _build_cumulative_equity_curve_from_trades(
        self,
        trades: List[Dict],
        initial_capital: float,
        total_pnl: float,
    ) -> List[Dict]:
        """
        Equity curve from realized PnL only: initial_capital + cumulative sum(pnl) at each exit time.
        Same construction as portfolio statistics — used for portfolio, symbol charts, and symbol max drawdown.
        """
        initial_capital = float(initial_capital)
        equity_curve_data: List[Dict] = []
        
        def normalize_timestamp(ts):
            if ts is None:
                return None
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if timezone.is_naive(ts):
                ts = timezone.make_aware(ts)
            return ts
        
        if not trades:
            return equity_curve_data
        
        trades_with_exits = [t for t in trades if t.get('exit_timestamp') and t.get('pnl') is not None]
        
        if trades_with_exits:
            sorted_trades = sorted(
                trades_with_exits,
                key=lambda t: normalize_timestamp(t['exit_timestamp']) or datetime.min,
            )
            all_timestamps = []
            for trade in trades:
                if trade.get('entry_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
                if trade.get('exit_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['exit_timestamp']))
            earliest_ts = min(all_timestamps) if all_timestamps else None
            
            if earliest_ts:
                equity_curve_data.append({
                    'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts),
                    'equity': initial_capital,
                })
            
            cumulative_pnl = 0.0
            for trade in sorted_trades:
                cumulative_pnl += float(trade['pnl'])
                exit_ts = normalize_timestamp(trade['exit_timestamp'])
                current_equity = initial_capital + cumulative_pnl
                equity_curve_data.append({
                    'timestamp': exit_ts.isoformat() if hasattr(exit_ts, 'isoformat') else str(exit_ts),
                    'equity': round(current_equity, 2),
                })
            
            equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
            seen_timestamps = {}
            for point in equity_curve_data:
                ts_key = point['timestamp']
                seen_timestamps[ts_key] = point
            equity_curve_data = list(seen_timestamps.values())
            equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
            
            if equity_curve_data:
                final_equity = initial_capital + total_pnl
                latest_ts = max(
                    (
                        normalize_timestamp(t.get('exit_timestamp') or t.get('entry_timestamp'))
                        for t in trades
                        if t.get('exit_timestamp') or t.get('entry_timestamp')
                    ),
                    default=earliest_ts,
                )
                last_point_ts = pd.to_datetime(equity_curve_data[-1]['timestamp'])
                if timezone.is_naive(last_point_ts):
                    last_point_ts = timezone.make_aware(last_point_ts)
                if latest_ts and latest_ts != last_point_ts:
                    equity_curve_data.append({
                        'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts),
                        'equity': round(final_equity, 2),
                    })
                    equity_curve_data.sort(key=lambda x: pd.to_datetime(x['timestamp']))
                else:
                    equity_curve_data[-1]['equity'] = round(final_equity, 2)
            if equity_curve_data:
                final_equity = initial_capital + total_pnl
                equity_curve_data[-1]['equity'] = round(final_equity, 2)
        else:
            all_timestamps = []
            for trade in trades:
                if trade.get('entry_timestamp'):
                    all_timestamps.append(normalize_timestamp(trade['entry_timestamp']))
            if all_timestamps:
                earliest_ts = min(all_timestamps)
                latest_ts = max(all_timestamps)
                equity_curve_data = [
                    {
                        'timestamp': earliest_ts.isoformat() if hasattr(earliest_ts, 'isoformat') else str(earliest_ts),
                        'equity': initial_capital,
                    },
                    {
                        'timestamp': latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts),
                        'equity': round(initial_capital + total_pnl, 2),
                    },
                ]
            else:
                now = timezone.now()
                equity_curve_data = [
                    {'timestamp': now.isoformat(), 'equity': initial_capital},
                    {'timestamp': (now + timedelta(days=1)).isoformat(), 'equity': round(initial_capital + total_pnl, 2)},
                ]
        
        if len(equity_curve_data) == 0:
            now = timezone.now()
            equity_curve_data = [
                {'timestamp': now.isoformat(), 'equity': initial_capital},
                {'timestamp': (now + timedelta(days=1)).isoformat(), 'equity': round(initial_capital + total_pnl, 2)},
            ]
        elif len(equity_curve_data) == 1:
            final_equity = initial_capital + total_pnl
            first_ts = pd.to_datetime(equity_curve_data[0]['timestamp'])
            if timezone.is_naive(first_ts):
                first_ts = timezone.make_aware(first_ts)
            final_ts = first_ts + timedelta(days=1)
            equity_curve_data.append({
                'timestamp': final_ts.isoformat(),
                'equity': round(final_equity, 2),
            })
        else:
            final_equity = initial_capital + total_pnl
            equity_curve_data[-1]['equity'] = round(final_equity, 2)
        
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
        
        # Intra-trade metrics (per-trade OHLCV adverse excursion while position open)
        closed_trades = [t for t in all_trades if t.get('exit_timestamp') is not None]
        trade_drawdowns = [t.get('max_drawdown') for t in closed_trades if t.get('max_drawdown') is not None]
        if trade_drawdowns:
            avg_intra_trade_drawdown = sum(trade_drawdowns) / len(trade_drawdowns)
            worst_intra_trade_drawdown = max(trade_drawdowns)
        else:
            avg_intra_trade_drawdown = 0
            worst_intra_trade_drawdown = 0
        
        # Equity curve: realized cumulative PnL (shared with symbol stats — same max drawdown definition)
        equity_curve_data = self._build_cumulative_equity_curve_from_trades(all_trades, initial_capital, total_pnl)
        max_drawdown_duration = 0
        sharpe_ratio = 0
        cagr = 0
        total_return = 0
        
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
        
        # Peak-to-trough max drawdown and duration from the same equity curve
        if equity_curve_tuples:
            max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_curve_tuples)
        else:
            max_drawdown = 0
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
            'avg_intra_trade_drawdown': round(avg_intra_trade_drawdown, 4),
            'worst_intra_trade_drawdown': round(worst_intra_trade_drawdown, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'cagr': round(cagr, 4),
            'total_return': round(total_return, 4),
            'equity_curve': equity_curve_data,
            'skipped_trades_count': getattr(self, 'skipped_trades_count', 0),  # Count of trades skipped due to insufficient cash
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

