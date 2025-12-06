"""
Statistics calculation functions for market data
Provides statistical metrics that are always computed automatically
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_mean_price(ohlcv_data: List[Dict]) -> Optional[float]:
    """
    Calculate mean closing price
    
    Args:
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
    
    Returns:
        Mean closing price as float, or None if insufficient data
    """
    if not ohlcv_data or len(ohlcv_data) < 1:
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Calculate mean of closing prices
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        mean_price = df['close'].mean()
        
        # Return as float, rounded to 2 decimal places
        return round(float(mean_price), 2) if not pd.isna(mean_price) else None
        
    except Exception as e:
        print(f"Error calculating mean price: {str(e)}")
        return None


def calculate_volatility(ohlcv_data: List[Dict]) -> Optional[float]:
    """
    Calculate volatility as standard deviation of returns in percentage
    
    Args:
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
    
    Returns:
        Volatility as standard deviation of returns in percentage (e.g., 2.5 for 2.5%)
        Returns None if insufficient data
    """
    if not ohlcv_data or len(ohlcv_data) < 2:
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Ensure timestamp is datetime - handle different formats
        if 'timestamp' in df.columns:
            # Try to parse timestamp, handling various formats
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
            except:
                # Fallback: try without format specification
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Sort by timestamp (ascending)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns (percentage change in closing prices)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['returns'] = df['close'].pct_change() * 100  # Convert to percentage
        
        # Remove NaN values (first row will be NaN)
        returns = df['returns'].dropna()
        
        if len(returns) < 2:
            return None
        
        # Calculate standard deviation of returns (volatility)
        volatility = returns.std()
        
        # Return as float, rounded to 2 decimal places
        return round(float(volatility), 2) if not pd.isna(volatility) else None
        
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None


def calculate_bollinger_phase(ohlcv_data: List[Dict], period: int = 20, num_std: float = 2.0) -> Optional[str]:
    """
    Calculate current Bollinger Band phase based on specific conditions
    
    Phases:
    - Squeeze: All bands flat, parallel, bandwidth constant
    - Expansion: Upper trends higher, lower trends lower, bandwidth increases
    - Continuation: All bands move parallel in same direction
    - Contraction: Upper trends lower, lower trends higher, sharp bandwidth decline
    
    Args:
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
        period: Bollinger Band period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
    
    Returns:
        Phase name as string, or None if insufficient data
    """
    # Need at least period + 10 data points for reliable phase calculation
    # But try with less data if available (minimum 20 points)
    min_required = max(period + 5, 20)
    if not ohlcv_data or len(ohlcv_data) < min_required:
        print(f"Bollinger phase calculation: insufficient data. Need {min_required}, have {len(ohlcv_data) if ohlcv_data else 0}")
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Ensure timestamp is datetime - handle different formats
        if 'timestamp' in df.columns:
            # Try to parse timestamp, handling various formats
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
            except:
                # Fallback: try without format specification
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Sort by timestamp (ascending)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate Bollinger Bands
        from analytical_tools.indicators import calculate_bollinger_bands
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        bands = calculate_bollinger_bands(df['close'], period=period, num_std=num_std)
        
        # Get last 10 values for trend analysis (need enough data to detect trends)
        last_10_upper = bands['upper'].tail(10).dropna()
        last_10_lower = bands['lower'].tail(10).dropna()
        last_10_middle = bands['middle'].tail(10).dropna()
        last_10_bandwidth = bands['bandwidth'].tail(10).dropna()
        last_10_price = df['close'].tail(10).dropna()
        
        if len(last_10_upper) < 5 or len(last_10_lower) < 5 or len(last_10_middle) < 5 or len(last_10_bandwidth) < 5:
            return None
        
        # Calculate trends (slope) using linear regression on last 5-10 points
        def calculate_slope(series):
            """Calculate slope of series (trend direction)"""
            if len(series) < 3:
                return 0
            x = np.arange(len(series))
            y = series.values
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # Calculate slopes for each band
        upper_slope = calculate_slope(last_10_upper)
        lower_slope = calculate_slope(last_10_lower)
        middle_slope = calculate_slope(last_10_middle)
        bandwidth_slope = calculate_slope(last_10_bandwidth)
        price_slope = calculate_slope(last_10_price)
        
        # Calculate price range for relative thresholds
        price_range = last_10_price.max() - last_10_price.min()
        avg_price = last_10_price.mean()
        
        # Threshold for "flat" - very small movement (0.1% of price range per day)
        flat_threshold = 0.001 * price_range if price_range > 0 else 0.001 * avg_price
        # Threshold for "trending" - significant movement (0.5% of price range per day)
        trend_threshold = 0.005 * price_range if price_range > 0 else 0.005 * avg_price
        # Threshold for "sharp" bandwidth change - 10% of average bandwidth
        sharp_threshold = 0.10 * last_10_bandwidth.mean() if last_10_bandwidth.mean() > 0 else 0.1
        
        # Check if bands are parallel (similar slopes)
        def are_parallel(slope1, slope2, threshold=flat_threshold):
            return abs(slope1 - slope2) < threshold
        
        # Check if band is flat
        def is_flat(slope, threshold=flat_threshold):
            return abs(slope) < threshold
        
        # Check if bandwidth is constant
        bandwidth_variance = last_10_bandwidth.tail(5).std()
        bandwidth_mean = last_10_bandwidth.tail(5).mean()
        is_bandwidth_constant = bandwidth_variance < (0.02 * bandwidth_mean)  # Less than 2% variance
        
        # Check for sharp bandwidth decline
        bandwidth_change = last_10_bandwidth.iloc[-1] - last_10_bandwidth.iloc[-5]
        sharp_bandwidth_decline = bandwidth_change < -abs(sharp_threshold)
        
        # Phase detection based on conditions
        
        # 1. SQUEEZE PHASE
        # All bands flat, parallel, bandwidth constant
        if (is_flat(upper_slope, flat_threshold) and 
            is_flat(lower_slope, flat_threshold) and 
            is_flat(middle_slope, flat_threshold) and
            are_parallel(upper_slope, lower_slope, flat_threshold) and
            are_parallel(upper_slope, middle_slope, flat_threshold) and
            is_bandwidth_constant):
            return 'Squeeze'
        
        # 2. CONTRACTION PHASE
        # Upper trends lower, lower trends higher, sharp bandwidth decline
        if (upper_slope < -abs(trend_threshold) and 
            lower_slope > abs(trend_threshold) and 
            sharp_bandwidth_decline):
            return 'Contraction'
        
        # 3. EXPANSION PHASE
        # Upper trends higher, lower trends lower, bandwidth increases
        if (upper_slope > abs(trend_threshold) and 
            lower_slope < -abs(trend_threshold) and 
            bandwidth_slope > abs(trend_threshold)):
            return 'Expansion'
        
        # 4. CONTINUATION PHASE
        # All bands move parallel in same direction
        if (are_parallel(upper_slope, middle_slope, trend_threshold) and
            are_parallel(middle_slope, lower_slope, trend_threshold) and
            abs(upper_slope) > abs(trend_threshold)):  # Bands are trending
            # Check if bands trend in same direction as price
            if (price_slope > 0 and upper_slope > 0) or (price_slope < 0 and upper_slope < 0):
                return 'Continuation'
        
        # Default: if none match, return None (strict rules - no phase detected)
        # If bandwidth is increasing, likely expansion
        if bandwidth_slope > abs(trend_threshold):
            return 'Expansion'
        # If bandwidth is decreasing sharply, likely contraction
        if sharp_bandwidth_decline:
            return 'Contraction'
        # If bands are parallel and trending, likely continuation
        if are_parallel(upper_slope, lower_slope, trend_threshold) and abs(upper_slope) > abs(trend_threshold):
            return 'Continuation'
        
        return None
        
    except Exception as e:
        print(f"Error calculating Bollinger phase: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_benchmark_ticker(exchange_code: str) -> Optional[str]:
    """
    Get benchmark ticker for an exchange
    
    Args:
        exchange_code: Exchange code (e.g., 'US', 'NASDAQ', 'NYSE')
    
    Returns:
        Benchmark ticker (e.g., '^GSPC' for US exchanges) or None
    """
    # Mapping of exchange codes to benchmark tickers
    benchmark_map = {
        'US': '^GSPC',  # S&P 500 for US stocks
        'NASDAQ': '^GSPC',
        'NYSE': '^GSPC',
        # Add more exchanges as needed
    }
    
    # Check exact match first
    if exchange_code in benchmark_map:
        return benchmark_map[exchange_code]
    
    # Check if exchange code contains 'US' (for variations like 'US-NASDAQ')
    if 'US' in exchange_code.upper():
        return '^GSPC'
    
    return None


def calculate_beta(stock_ohlcv_data: List[Dict], benchmark_ohlcv_data: List[Dict]) -> Optional[float]:
    """
    Calculate beta (slope) of stock returns relative to benchmark returns
    
    Beta measures the stock's volatility relative to the market:
    - Beta > 1: Stock is more volatile than the market
    - Beta = 1: Stock moves with the market
    - Beta < 1: Stock is less volatile than the market
    - Beta < 0: Stock moves inversely to the market (rare)
    
    Args:
        stock_ohlcv_data: List of OHLCV dicts for the stock
        benchmark_ohlcv_data: List of OHLCV dicts for the benchmark
    
    Returns:
        Beta value as float, or None if insufficient data
    """
    if not stock_ohlcv_data or not benchmark_ohlcv_data or len(stock_ohlcv_data) < 2 or len(benchmark_ohlcv_data) < 2:
        return None
    
    try:
        # Convert to DataFrames
        stock_df = pd.DataFrame(stock_ohlcv_data)
        benchmark_df = pd.DataFrame(benchmark_ohlcv_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in stock_df.columns:
            stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'], errors='coerce', format='mixed')
        if 'timestamp' in benchmark_df.columns:
            benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'], errors='coerce', format='mixed')
        
        # Sort by timestamp
        stock_df = stock_df.sort_values('timestamp').reset_index(drop=True)
        benchmark_df = benchmark_df.sort_values('timestamp').reset_index(drop=True)
        
        # Merge on timestamp to align data
        merged_df = pd.merge(
            stock_df[['timestamp', 'close']],
            benchmark_df[['timestamp', 'close']],
            on='timestamp',
            how='inner',
            suffixes=('_stock', '_benchmark')
        )
        
        if len(merged_df) < 2:
            return None
        
        # Calculate returns (percentage change)
        merged_df['close_stock'] = pd.to_numeric(merged_df['close_stock'], errors='coerce')
        merged_df['close_benchmark'] = pd.to_numeric(merged_df['close_benchmark'], errors='coerce')
        
        merged_df['returns_stock'] = merged_df['close_stock'].pct_change()
        merged_df['returns_benchmark'] = merged_df['close_benchmark'].pct_change()
        
        # Remove NaN values (first row will be NaN)
        returns_df = merged_df[['returns_stock', 'returns_benchmark']].dropna()
        
        if len(returns_df) < 2:
            return None
        
        # Calculate beta using covariance/variance formula
        # Beta = Covariance(stock_returns, benchmark_returns) / Variance(benchmark_returns)
        covariance = returns_df['returns_stock'].cov(returns_df['returns_benchmark'])
        benchmark_variance = returns_df['returns_benchmark'].var()
        
        if benchmark_variance == 0 or pd.isna(benchmark_variance):
            return None
        
        beta = covariance / benchmark_variance
        
        # Return as float, rounded to 3 decimal places
        return round(float(beta), 3) if not pd.isna(beta) else None
        
    except Exception as e:
        print(f"Error calculating beta: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_statistics(ohlcv_data: List[Dict], symbol=None, benchmark_ohlcv_data: Optional[List[Dict]] = None) -> Dict:
    """
    Calculate all statistics for a symbol's OHLCV data
    
    Args:
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
        symbol: Symbol instance (optional, for future use)
        benchmark_ohlcv_data: Benchmark OHLCV data (optional, for beta calculation)
    
    Returns:
        Dictionary with statistics:
        {
            'volatility': float,  # Standard deviation of returns in %
            'mean_price': float,  # Mean closing price
            'beta': float,  # Beta relative to benchmark
            'bollinger_phase': str,  # Current Bollinger Band phase
            ...
        }
    """
    stats = {}
    
    # Calculate mean price
    mean_price = calculate_mean_price(ohlcv_data)
    if mean_price is not None:
        stats['mean_price'] = mean_price
    
    # Calculate volatility
    volatility = calculate_volatility(ohlcv_data)
    if volatility is not None:
        stats['volatility'] = volatility
    
    # Calculate beta if benchmark data is provided
    if benchmark_ohlcv_data:
        beta = calculate_beta(ohlcv_data, benchmark_ohlcv_data)
        if beta is not None:
            stats['beta'] = beta
    
    # Calculate Bollinger Band phase (if we have enough data)
    # Always try to calculate phase - it will return None if insufficient data
    try:
        bollinger_phase = calculate_bollinger_phase(ohlcv_data)
        if bollinger_phase is not None:
            stats['bollinger_phase'] = bollinger_phase
    except Exception as e:
        # Silently fail - phase calculation is optional
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error calculating Bollinger phase: {str(e)}")
    
    return stats

