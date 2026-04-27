"""
Yahoo Finance Provider
Handles fetching OHLCV data from Yahoo Finance using yfinance library
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from django.utils import timezone

# yfinance can block indefinitely on network issues; cap wall time so Celery workers don't hang.
YAHOO_HISTORY_TIMEOUT_SEC = 90


class YahooFinanceProvider:
    """Provider for Yahoo Finance data"""
    
    @staticmethod
    def get_historical_data(
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d'
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a symbol
        
        Args:
            ticker: Symbol ticker (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data (datetime object)
            end_date: End date for data (datetime object)
            period: Period string (e.g., '1mo', '1y', 'max'). If provided, overrides start_date/end_date
            interval: Data interval ('1d' for daily, '1h' for hourly, '1m' for minute)
        
        Returns:
            List of dictionaries with OHLCV data:
            [
                {
                    'timestamp': datetime,
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': int
                },
                ...
            ]
        """
        try:
            def _load_history():
                stock = yf.Ticker(ticker)
                if period:
                    return stock.history(period=period, interval=interval)
                if start_date and end_date:
                    end_date_inclusive = end_date + timedelta(days=1)
                    return stock.history(
                        start=start_date, end=end_date_inclusive, interval=interval,
                    )
                if start_date:
                    return stock.history(start=start_date, interval=interval)
                return stock.history(period='1y', interval=interval)

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_load_history)
                try:
                    hist = future.result(timeout=YAHOO_HISTORY_TIMEOUT_SEC)
                except FuturesTimeout as exc:
                    raise TimeoutError(
                        f"Yahoo Finance history timed out after {YAHOO_HISTORY_TIMEOUT_SEC}s "
                        f"for {ticker!r}"
                    ) from exc

            if hist.empty:
                return []
            
            # Convert to list of dictionaries
            data = []
            for timestamp, row in hist.iterrows():
                # Convert pandas Timestamp to datetime
                if hasattr(timestamp, 'to_pydatetime'):
                    dt = timestamp.to_pydatetime()
                else:
                    dt = timestamp
                
                # Ensure timezone awareness
                if timezone.is_naive(dt):
                    dt = timezone.make_aware(dt)
                
                # With auto_adjust=True, Yahoo Finance returns adjusted prices
                # The Open, High, Low, Close are all adjusted for splits and dividends
                data.append({
                    'timestamp': dt,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),  # Already adjusted when auto_adjust=True
                    'volume': int(row['Volume']) if 'Volume' in row else 0
                })
            
            return data
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data for {ticker}: {str(e)}")
            raise
    
    @staticmethod
    def get_multiple_symbols_data(
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d'
    ) -> Dict[str, List[Dict]]:
        """
        Get historical data for multiple symbols
        
        Args:
            tickers: List of symbol tickers
            start_date: Start date for data
            end_date: End date for data
            period: Period string (overrides start_date/end_date)
            interval: Data interval
        
        Returns:
            Dictionary mapping tickers to their OHLCV data
        """
        result = {}
        for ticker in tickers:
            try:
                data = YahooFinanceProvider.get_historical_data(
                    ticker, start_date, end_date, period, interval
                )
                result[ticker] = data
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                result[ticker] = []
        
        return result














