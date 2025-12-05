"""
Hardcoded Indicator Definitions
All available analytical tools are defined here
"""

INDICATOR_DEFINITIONS = [
    {
        'name': 'RSI',
        'description': 'Relative Strength Index - Momentum oscillator that measures speed and magnitude of price changes',
        'category': 'indicator',
        'default_parameters': {'period': 14}
    },
    {
        'name': 'ATR',
        'description': 'Average True Range - Measures market volatility',
        'category': 'indicator',
        'default_parameters': {'period': 14}
    },
    {
        'name': 'MACD',
        'description': 'Moving Average Convergence Divergence - Trend-following momentum indicator',
        'category': 'indicator',
        'default_parameters': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    },
    {
        'name': 'Variance',
        'description': 'Statistical variance - Measures the spread of price data',
        'category': 'statistical',
        'default_parameters': {'period': 20}
    },
    {
        'name': 'SMA',
        'description': 'Simple Moving Average - Average of prices over a period',
        'category': 'indicator',
        'default_parameters': {'period': 20}
    },
    {
        'name': 'EMA',
        'description': 'Exponential Moving Average - Weighted average giving more importance to recent prices',
        'category': 'indicator',
        'default_parameters': {'period': 20}
    },
]

def get_indicator_definition(name):
    """Get indicator definition by name"""
    for indicator in INDICATOR_DEFINITIONS:
        if indicator['name'] == name:
            return indicator
    return None

def get_all_indicator_definitions():
    """Get all indicator definitions"""
    return INDICATOR_DEFINITIONS

