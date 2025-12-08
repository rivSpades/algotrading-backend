"""
Indicator Calculation Functions
Provides reusable analytical and mathematical functions for market data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from decimal import Decimal


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default: 14)
    
    Returns:
        Series of ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
    
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' series
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }


def calculate_variance(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Variance (statistical measure)
    
    Args:
        prices: Series of closing prices
        period: Rolling window period (default: 20)
    
    Returns:
        Series of variance values
    """
    return prices.rolling(window=period).var()


def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        prices: Series of prices
        period: SMA period (default: 20)
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        prices: Series of prices
        period: EMA period (default: 20)
    
    Returns:
        Series of EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Series of closing prices
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower', and 'bandwidth' series
    """
    # Middle band is SMA
    middle = prices.rolling(window=period).mean()
    
    # Standard deviation
    std = prices.rolling(window=period).std()
    
    # Upper and lower bands
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # Bandwidth = (upper - lower) / middle * 100
    bandwidth = ((upper - lower) / middle) * 100
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': bandwidth
    }


def calculate_returns(open_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
    """
    Calculate gap returns: (Open_t - Close_{t-1}) / Close_{t-1}
    
    This calculates the return from previous day's close to today's open.
    Used for gap-up/gap-down analysis.
    
    Args:
        open_prices: Series of opening prices
        close_prices: Series of closing prices
    
    Returns:
        Series of gap returns (as decimal, e.g., 0.02 for 2%)
    """
    # Shift close prices by 1 to get previous day's close
    prev_close = close_prices.shift(1)
    
    # Calculate gap return: (open - prev_close) / prev_close
    returns = (open_prices - prev_close) / prev_close
    
    return returns


def calculate_rolling_std(returns: pd.Series, period: int = 90) -> pd.Series:
    """
    Calculate rolling standard deviation of returns
    
    IMPORTANT: This function calculates STD using only historical data up to the current point.
    For bias prevention, ensure returns are calculated from data available before the current bar.
    
    Args:
        returns: Series of returns (e.g., from calculate_returns)
        period: Rolling window period (default: 90)
    
    Returns:
        Series of rolling standard deviation values
    """
    return returns.rolling(window=period).std()


# Indicator registry mapping tool names to calculation functions
INDICATOR_FUNCTIONS = {
    'RSI': calculate_rsi,
    'ATR': calculate_atr,
    'MACD': calculate_macd,
    'Variance': calculate_variance,
    'SMA': calculate_sma,
    'EMA': calculate_ema,
    'BollingerBands': calculate_bollinger_bands,
    'Returns': calculate_returns,
    'RollingSTD': calculate_rolling_std,
}


def compute_indicator(
    tool_name: str,
    ohlcv_data: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Compute indicator values directly on OHLCV DataFrame using pandas
    Much faster - adds column to existing dataframe
    
    Args:
        tool_name: Name of the indicator tool
        ohlcv_data: DataFrame with columns: timestamp, open, high, low, close, volume
        parameters: Dictionary of indicator parameters
    
    Returns:
        DataFrame with timestamp and computed values aligned with OHLCV timestamps
    """
    if tool_name not in INDICATOR_FUNCTIONS:
        raise ValueError(f"Unknown indicator: {tool_name}")
    
    func = INDICATOR_FUNCTIONS[tool_name]
    
    # Work with copy to avoid modifying original
    df = ohlcv_data.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Compute indicator directly on dataframe
    if tool_name == 'RSI':
        result = func(df['close'], **parameters)
        df['indicator_value'] = result.values
    
    elif tool_name == 'ATR':
        result = func(df['high'], df['low'], df['close'], **parameters)
        df['indicator_value'] = result.values
    
    elif tool_name == 'MACD':
        result_dict = func(df['close'], **parameters)
        df['indicator_value'] = result_dict['macd'].values
        df['metadata'] = [
            {
                'macd': float(macd) if pd.notna(macd) else None,
                'signal': float(sig) if pd.notna(sig) else None,
                'histogram': float(hist) if pd.notna(hist) else None
            }
            for macd, sig, hist in zip(
                result_dict['macd'].values,
                result_dict['signal'].values,
                result_dict['histogram'].values
            )
        ]
    
    elif tool_name == 'BollingerBands':
        result_dict = func(df['close'], **parameters)
        # Store all bands in metadata
        df['indicator_value'] = result_dict['middle'].values  # Default to middle band
        df['metadata'] = [
            {
                'upper': float(upper) if pd.notna(upper) else None,
                'middle': float(middle) if pd.notna(middle) else None,
                'lower': float(lower) if pd.notna(lower) else None,
                'bandwidth': float(bandwidth) if pd.notna(bandwidth) else None
            }
            for upper, middle, lower, bandwidth in zip(
                result_dict['upper'].values,
                result_dict['middle'].values,
                result_dict['lower'].values,
                result_dict['bandwidth'].values
            )
        ]
    
    elif tool_name in ['Variance', 'SMA', 'EMA']:
        result = func(df['close'], **parameters)
        df['indicator_value'] = result.values
    
    elif tool_name == 'Returns':
        # Returns: (open - prev_close) / prev_close
        result = func(df['open'], df['close'], **parameters)
        df['indicator_value'] = result.values
    
    elif tool_name == 'RollingSTD':
        # RollingSTD: rolling standard deviation of returns
        # First calculate returns, then calculate rolling STD
        # This ensures bias prevention - STD is calculated from historical returns only
        returns = calculate_returns(df['open'], df['close'])
        result = func(returns, **parameters)
        df['indicator_value'] = result.values
    
    else:
        raise ValueError(f"Indicator computation not implemented for: {tool_name}")
    
    # Return only timestamp and value columns, aligned with OHLCV data
    result_df = df[['timestamp', 'indicator_value']].copy()
    result_df = result_df.rename(columns={'indicator_value': 'value'})
    
    # Add metadata if it exists
    if 'metadata' in df.columns:
        result_df['metadata'] = df['metadata']
    
    # Keep all rows (including NaN) for proper alignment - convert NaN to None later
    # Don't filter out NaN - we need to maintain alignment with OHLCV data
    result_df = result_df.reset_index(drop=True)
    
    return result_df

