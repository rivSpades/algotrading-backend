"""
Alpaca Provider
Handles fetching OHLCV data from Alpaca Markets API
Free tier limitations:
- Up to 200 symbols per API call
- Maximum 1,000 data points per call (total across all symbols)
- 200 API calls per minute rate limit
- Historical data since 2016 (except latest 15 minutes)

Note: Uses adjustment='all' parameter to get split and dividend adjusted prices,
matching the behavior of Yahoo Finance (which provides adjusted data).
"""

import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from decimal import Decimal


class AlpacaProvider:
    """Provider for Alpaca Markets data"""
    
    # Class variables for credentials (set by initialize method)
    _api_key = None
    _api_secret = None
    _base_url = None
    _initialized = False
    
    # Free tier limits
    MAX_SYMBOLS_PER_CALL = 200
    MAX_DATA_POINTS_PER_CALL = 1000
    RATE_LIMIT_CALLS_PER_MINUTE = 200
    RATE_LIMIT_DELAY_SECONDS = 60.0 / RATE_LIMIT_CALLS_PER_MINUTE  # ~0.3 seconds between calls
    
    @classmethod
    def initialize(cls, api_key: str, api_secret: str, base_url: str = None):
        """
        Initialize Alpaca provider with credentials
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret key
            base_url: Base URL (defaults to data API if not provided)
        """
        cls._api_key = api_key
        cls._api_secret = api_secret
        
        # For OHLCV data, Alpaca uses the DATA API, not the trading API
        # Data API base URL: https://data.alpaca.markets/
        # Trading API base URL: https://paper-api.alpaca.markets/ (or https://api.alpaca.markets/)
        # These are separate services with different endpoints
        if not base_url:
            # Use data API for market data (not trading API)
            cls._base_url = 'https://data.alpaca.markets'
        else:
            # If base_url is provided, it should be the data API URL
            # Remove trailing slash and /v2 if present
            cls._base_url = base_url.rstrip('/').rstrip('/v2')
        
        cls._initialized = True
    
    @classmethod
    def _check_initialized(cls):
        """Check if provider is initialized"""
        if not cls._initialized or not cls._api_key or not cls._api_secret:
            raise ValueError("Alpaca provider not initialized. Call initialize() first.")
    
    @classmethod
    def _get_headers(cls) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        cls._check_initialized()
        return {
            'APCA-API-KEY-ID': cls._api_key,
            'APCA-API-SECRET-KEY': cls._api_secret,
        }
    
    @classmethod
    def _calculate_batch_size(cls, num_symbols: int, num_days: int) -> int:
        """
        Calculate optimal batch size based on data points limit
        
        Args:
            num_symbols: Number of symbols to fetch
            num_days: Number of days of data per symbol
        
        Returns:
            Optimal number of symbols per batch
        """
        # Total data points = num_symbols * num_days
        # We need: batch_size * num_days <= MAX_DATA_POINTS_PER_CALL
        # So: batch_size <= MAX_DATA_POINTS_PER_CALL / num_days
        
        if num_days <= 0:
            return min(num_symbols, cls.MAX_SYMBOLS_PER_CALL)
        
        max_batch_by_data_points = cls.MAX_DATA_POINTS_PER_CALL // num_days
        max_batch_by_symbols = cls.MAX_SYMBOLS_PER_CALL
        
        # Use the smaller of the two limits
        optimal_batch = min(max_batch_by_data_points, max_batch_by_symbols, num_symbols)
        
        # Ensure at least 1 symbol per batch
        return max(1, optimal_batch)
    
    @classmethod
    def _rate_limit_delay(cls):
        """Add delay to respect rate limits"""
        time.sleep(cls.RATE_LIMIT_DELAY_SECONDS)
    
    @classmethod
    def get_historical_data(
        cls,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d'
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a single symbol
        
        Args:
            ticker: Symbol ticker (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data (datetime object)
            end_date: End date for data (datetime object)
            period: Period string (e.g., '1mo', '1y', 'max'). If provided, overrides start_date/end_date
            interval: Data interval ('1d' for daily, '1h' for hourly, '1m' for minute)
        
        Returns:
            List of dictionaries with OHLCV data
        """
        # Use bulk method for single symbol
        result = cls.get_multiple_symbols_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval
        )
        return result.get(ticker, [])
    
    @classmethod
    def get_multiple_symbols_data(
        cls,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d'
    ) -> Dict[str, List[Dict]]:
        """
        Get historical data for multiple symbols (with bulk fetching support and date chunking)
        
        Args:
            tickers: List of symbol tickers
            start_date: Start date for data
            end_date: End date for data
            period: Period string (overrides start_date/end_date)
            interval: Data interval
        
        Returns:
            Dictionary mapping tickers to their OHLCV data
        """
        cls._check_initialized()
        
        if not tickers:
            return {}
        
        # Convert period to date range if needed
        if period:
            end_date = timezone.now()
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '1wk':
                start_date = end_date - timedelta(weeks=1)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '5y':
                start_date = end_date - timedelta(days=1825)
            elif period == 'max':
                start_date = timezone.make_aware(datetime(2016, 1, 1))  # Alpaca data starts from 2016
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Ensure dates are timezone-aware for proper comparisons
        if start_date and timezone.is_naive(start_date):
            start_date = timezone.make_aware(start_date)
        if end_date and timezone.is_naive(end_date):
            end_date = timezone.make_aware(end_date)
        
        # Calculate number of days for batch size optimization
        if start_date and end_date:
            num_days = (end_date.date() - start_date.date()).days
        else:
            num_days = 365  # Default estimate
        
        # Map interval to Alpaca timeframe format
        timeframe_map = {
            '1d': '1Day',
            '1day': '1Day',
            'daily': '1Day',
            '1h': '1Hour',
            '1hour': '1Hour',
            'hourly': '1Hour',
            '1m': '1Min',
            '1min': '1Min',
            'minute': '1Min',
        }
        alpaca_timeframe = timeframe_map.get(interval.lower(), '1Day')
        
        # Initialize result dictionary
        result = {ticker: [] for ticker in tickers}
        
        # Smart batching strategy with date chunking:
        # Alpaca has a 1,000 data points limit per page (total across all symbols)
        # Pagination is unreliable for very large date ranges, so we use date chunking.
        # Strategy:
        # - For large date ranges (>1 year): Chunk dates into 1-year segments
        # - For each date chunk: Use reasonable batch sizes to stay under 1000 points per page
        # - This ensures complete data retrieval without relying on problematic pagination
        # - For small date ranges (<=1 year): Use larger batches since pagination should work
        
        # Determine if we need date chunking
        trading_days_per_year = 252
        estimated_trading_days = int(num_days * (trading_days_per_year / 365))
        max_days_per_chunk = 365  # 1 year chunks for large date ranges
        
        # Calculate batch size based on date range and data point limit
        # Alpaca has a 1000 data points limit per API call (total across all symbols)
        # For 1 year of daily data: ~252 trading days per symbol
        # So max batch size = floor(1000 / 252) = 3 symbols to stay safely under limit
        # We use 3 symbols per batch for 1-year chunks to ensure all symbols are returned
        if num_days > 365:  # > 1 year - will use date chunking (1 year chunks)
            # For 1-year chunks: ~252 trading days, so max 3 symbols per batch (3 * 252 = 756 < 1000)
            # This ensures all symbols are returned in the response
            batch_size = min(len(tickers), max(1, min(3, cls.MAX_SYMBOLS_PER_CALL)))
        else:  # <= 1 year
            # For small date ranges, calculate based on estimated trading days
            estimated_trading_days = int(num_days * (252 / 365))
            if estimated_trading_days > 0:
                # Max symbols that would stay under 1000 points: floor(1000 / estimated_trading_days)
                max_symbols = max(1, min(3, 1000 // estimated_trading_days))
            else:
                max_symbols = 3
            batch_size = min(len(tickers), max(1, min(max_symbols, cls.MAX_SYMBOLS_PER_CALL)))
        
        # Create date chunks for large date ranges
        date_chunks = []
        if start_date and end_date and num_days > max_days_per_chunk:
            # Split into 1-year chunks
            current_start = start_date
            while current_start < end_date:
                current_end = min(
                    current_start + timedelta(days=max_days_per_chunk),
                    end_date
                )
                date_chunks.append((current_start, current_end))
                current_start = current_end + timedelta(days=1)
        else:
            # Single date range
            date_chunks = [(start_date, end_date)] if start_date and end_date else [(None, None)]
        
        # Process symbols in batches
        i = 0
        while i < len(tickers):
            batch_tickers = tickers[i:i + batch_size]
            
            # Process each date chunk for this batch
            for chunk_start, chunk_end in date_chunks:
                chunk_processed = False
                chunk_attempts = 0
                max_attempts = 3
                
                while not chunk_processed and chunk_attempts < max_attempts:
                    chunk_attempts += 1
                    try:
                        # Make API request for batch and date chunk
                        params = {
                            'symbols': ','.join(batch_tickers),
                            'timeframe': alpaca_timeframe,
                            'adjustment': 'all',  # Apply all adjustments (split, dividend, spin-off) to get adjusted prices
                            # Note: Don't specify feed parameter - let Alpaca API choose (works for 2016-2025 on free tier with pagination)
                        }
                        
                        if chunk_start:
                            params['start'] = chunk_start.strftime('%Y-%m-%dT%H:%M:%S-05:00')  # Alpaca expects ISO format
                        if chunk_end:
                            params['end'] = chunk_end.strftime('%Y-%m-%dT%H:%M:%S-05:00')
                    
                        # Rate limiting (skip for first batch and first chunk)
                        if i > 0 or (chunk_start != date_chunks[0][0] if date_chunks else False):
                            cls._rate_limit_delay()
                        
                        # Alpaca bulk bars endpoint: /v2/stocks/bars (without symbol in path)
                        # Symbols are passed as comma-separated query parameter
                        full_url = f'{cls._base_url}/v2/stocks/bars'
                        
                        # Debug logging (only for first batch and first chunk)
                        if i == 0 and chunk_start == date_chunks[0][0] if date_chunks else i == 0:
                            print(f"Alpaca API Request: {full_url}")
                            print(f"  Headers: APCA-API-KEY-ID={cls._api_key[:10]}..., APCA-API-SECRET-KEY=***")
                            print(f"  Params: symbols={len(batch_tickers)} symbols, start={chunk_start}, end={chunk_end}")
                            print(f"  Batch size: {batch_size}, Total symbols: {len(tickers)}, Date chunks: {len(date_chunks)}")
                            print(f"  Note: Using bulk API with date chunking for large date ranges")
                        
                        response = requests.get(
                            full_url,
                            headers=cls._get_headers(),
                            params=params,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Handle pagination for this date chunk - Alpaca uses next_page_token
                            # For date chunks (1 year), pagination should work correctly
                            page_count = 0
                            while True:
                                page_count += 1
                                bars = data.get('bars', {})
                                next_page_token = data.get('next_page_token')
                                
                                # Debug: Count bars per symbol on first page (only for first batch and first chunk)
                                if page_count == 1 and i == 0 and (chunk_start == date_chunks[0][0] if date_chunks else True):
                                    for ticker in batch_tickers:
                                        symbol_bars = bars.get(ticker, [])
                                        print(f"    Page 1: {ticker} has {len(symbol_bars)} bars")
                                
                                # Process bars for each symbol in batch
                                # Debug: Check which symbols are in the response (only for first batch and first chunk)
                                if page_count <= 2 and i == 0 and (chunk_start == date_chunks[0][0] if date_chunks else True):
                                    response_symbols = list(bars.keys())
                                    print(f"    Page {page_count}: API returned bars for {len(response_symbols)} symbols: {response_symbols[:10]}{'...' if len(response_symbols) > 10 else ''}")
                                
                                for ticker in batch_tickers:
                                    symbol_bars = bars.get(ticker, [])
                                    ohlcv_list = []
                                
                                    for bar in symbol_bars:
                                        # Convert Alpaca bar format to our format
                                        try:
                                            timestamp_str = bar.get('t')  # Alpaca uses 't' for timestamp
                                            if timestamp_str:
                                                # Parse ISO format timestamp
                                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                                if timezone.is_naive(timestamp):
                                                    timestamp = timezone.make_aware(timestamp)
                                                
                                                ohlcv_list.append({
                                                    'timestamp': timestamp,
                                                    'open': float(bar.get('o', 0)),
                                                    'high': float(bar.get('h', 0)),
                                                    'low': float(bar.get('l', 0)),
                                                    'close': float(bar.get('c', 0)),
                                                    'volume': int(bar.get('v', 0))
                                                })
                                        except Exception as e:
                                            print(f"Error parsing bar data for {ticker}: {e}")
                                            continue
                                    
                                    # Append to existing data for this ticker
                                    if ticker in result:
                                        result[ticker].extend(ohlcv_list)
                                    else:
                                        result[ticker] = ohlcv_list
                                
                                # If there's a next page token, fetch the next page
                                if next_page_token:
                                    # For pagination, only include essential params (page_token overrides start/end)
                                    # The page_token contains the continuation point, so we should NOT include start/end
                                    # to avoid any potential conflicts or limitations
                                    next_params = {
                                        'symbols': params['symbols'],
                                        'timeframe': params['timeframe'],
                                        'adjustment': params.get('adjustment', 'all'),  # Include adjustment parameter
                                        'page_token': next_page_token,
                                    }
                                    # Don't include feed parameter in pagination requests - let API choose
                                    # Don't include start/end in pagination - page_token handles continuation
                                    
                                    # Small delay before next page request
                                    time.sleep(0.1)  # 100ms delay between pages
                                    
                                    # Fetch next page
                                    next_response = requests.get(
                                        full_url,
                                        headers=cls._get_headers(),
                                        params=next_params,
                                        timeout=30
                                    )
                                    
                                    if next_response.status_code == 200:
                                        next_data = next_response.json()
                                        
                                        # Update data and continue pagination
                                        # Continue as long as there's a next_page_token, regardless of whether we got bars
                                        # This ensures we don't miss data that starts later in the date range
                                        data = next_data
                                        
                                        # Check if there's a next page token - continue pagination if token exists
                                        if next_data.get('next_page_token'):
                                            continue  # Continue to next page
                                        else:
                                            # No more pages, break out of pagination loop
                                            break
                                    else:
                                        print(f"Error fetching next page for batch {batch_tickers}: {next_response.status_code} - {next_response.text[:200]}")
                                        break  # Stop pagination on error
                                else:
                                    # No more pages, break out of pagination loop
                                    break
                            
                            chunk_processed = True
                            
                        elif response.status_code == 429:
                            # Rate limit exceeded, wait longer and retry
                            print(f"Rate limit exceeded, waiting 60 seconds before retry...")
                            time.sleep(60)
                            # Continue in the retry loop to retry this chunk
                            continue
                            
                        else:
                            # Enhanced error logging
                            print(f"Error fetching Alpaca data: {response.status_code} - {response.text[:200]}")
                            print(f"  URL: {full_url}")
                            print(f"  Params: symbols={len(batch_tickers)} symbols, date chunk: {chunk_start} to {chunk_end}")
                            print(f"  Symbols: {batch_tickers[:5]}{'...' if len(batch_tickers) > 5 else ''}")
                            # Mark as processed to move to next chunk (don't retry on other errors)
                            chunk_processed = True
                            
                    except Exception as e:
                        print(f"Error fetching Alpaca data for batch {batch_tickers}, date chunk {chunk_start} to {chunk_end}: {e}")
                        # Mark as processed to move to next chunk
                        chunk_processed = True
            
            # Move to next batch
            i += batch_size
        
        # Note: We use adjustment='all' in API params, so Alpaca returns adjusted data directly
        # No need for manual split adjustment
        
        return result
