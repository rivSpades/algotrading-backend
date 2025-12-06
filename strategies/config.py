"""
Strategy Definitions Configuration
Hardcoded strategy definitions that are automatically created/updated on app startup
"""

STRATEGY_DEFINITIONS = [
    {
        'name': 'Simple Moving Average Crossover',
        'description_short': 'Long and short strategy: Long when fast SMA crosses above slow SMA, Short when fast SMA crosses below slow SMA',
        'description_long': '''
        A momentum strategy that uses two Simple Moving Averages (SMA) and supports both long and short positions:
        - Fast SMA: Shorter period (default: 20 days)
        - Slow SMA: Longer period (default: 50 days)
        
        Long Position:
        - Entry Signal: When fast SMA crosses above slow SMA (golden cross)
        - Exit Signal: When fast SMA crosses below slow SMA (death cross)
        
        Short Position:
        - Entry Signal: When fast SMA crosses below slow SMA (death cross)
        - Exit Signal: When fast SMA crosses above slow SMA (golden cross)
        
        The strategy can be run in three modes:
        - ALL: Executes both long and short positions
        - LONG: Only executes long positions (ignores short entry signals)
        - SHORT: Only executes short positions (ignores long entry signals)
        
        Required Analytical Tools:
        - SMA with period = fast_period (default: 20)
        - SMA with period = slow_period (default: 50)
        ''',
        'default_parameters': {
            'fast_period': 20,
            'slow_period': 50,
        },
        'analytic_tools_used': ['SMA'],  # Strategy uses SMA tool with different periods
        'required_tool_configs': [
            {
                'tool_name': 'SMA',
                'parameters': {'period': 20},  # fast_period from strategy params
                'parameter_mapping': {'period': 'fast_period'},  # Maps strategy param to tool param
                'display_name': 'Fast SMA',
                'locked': True,  # Cannot be disabled/removed when strategy is active
            },
            {
                'tool_name': 'SMA',
                'parameters': {'period': 50},  # slow_period from strategy params
                'parameter_mapping': {'period': 'slow_period'},  # Maps strategy param to tool param
                'display_name': 'Slow SMA',
                'locked': True,  # Cannot be disabled/removed when strategy is active
            },
        ],
        'example_code': '''
        # Pseudocode
        if fast_sma[t] > slow_sma[t] and fast_sma[t-1] <= slow_sma[t-1]:
            signal = BUY
        elif fast_sma[t] < slow_sma[t] and fast_sma[t-1] >= slow_sma[t-1]:
            signal = SELL
        ''',
        'globally_enabled': False,
    },
    {
        'name': 'RSI Mean Reversion',
        'description_short': 'Buy when RSI is oversold, sell when RSI is overbought',
        'description_long': '''
        A mean reversion strategy using the Relative Strength Index (RSI):
        - RSI below 30: Oversold condition (buy signal)
        - RSI above 70: Overbought condition (sell signal)
        
        Entry Signal: RSI crosses below 30 (oversold)
        Exit Signal: RSI crosses above 70 (overbought) or reaches target profit
        
        Optional parameters:
        - rsi_period: RSI calculation period (default: 14)
        - oversold_threshold: RSI level for oversold (default: 30)
        - overbought_threshold: RSI level for overbought (default: 70)
        - stop_loss: Stop loss percentage (default: 0.02)
        - take_profit: Take profit percentage (default: 0.05)
        ''',
        'default_parameters': {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'stop_loss': 0.02,
            'take_profit': 0.05,
        },
        'analytic_tools_used': ['RSI'],
        'example_code': '''
        # Pseudocode
        if rsi[t] < oversold_threshold and rsi[t-1] >= oversold_threshold:
            signal = BUY
        elif rsi[t] > overbought_threshold and rsi[t-1] <= overbought_threshold:
            signal = SELL
        ''',
        'globally_enabled': False,
    },
    {
        'name': 'Bollinger Bands Breakout',
        'description_short': 'Buy when price breaks above upper band, sell when it breaks below lower band',
        'description_long': '''
        A breakout strategy using Bollinger Bands:
        - Upper Band: Price + (num_std * standard deviation)
        - Lower Band: Price - (num_std * standard deviation)
        - Middle Band: Simple Moving Average
        
        Entry Signal: Price breaks above upper band (bullish breakout)
        Exit Signal: Price breaks below lower band (bearish breakout) or reaches target
        
        Optional parameters:
        - period: Bollinger Bands period (default: 20)
        - num_std: Number of standard deviations (default: 2.0)
        - stop_loss: Stop loss percentage (default: 0.02)
        - take_profit: Take profit percentage (default: 0.05)
        ''',
        'default_parameters': {
            'period': 20,
            'num_std': 2.0,
            'stop_loss': 0.02,
            'take_profit': 0.05,
        },
        'analytic_tools_used': ['BollingerBands'],
        'example_code': '''
        # Pseudocode
        if price[t] > upper_band[t] and price[t-1] <= upper_band[t-1]:
            signal = BUY
        elif price[t] < lower_band[t] and price[t-1] >= lower_band[t-1]:
            signal = SELL
        ''',
        'globally_enabled': False,
    },
    {
        'name': 'MACD Crossover',
        'description_short': 'Buy when MACD line crosses above signal line, sell when it crosses below',
        'description_long': '''
        A momentum strategy using MACD (Moving Average Convergence Divergence):
        - MACD Line: Fast EMA - Slow EMA
        - Signal Line: EMA of MACD Line
        - Histogram: MACD Line - Signal Line
        
        Entry Signal: MACD line crosses above signal line (bullish crossover)
        Exit Signal: MACD line crosses below signal line (bearish crossover)
        
        Optional parameters:
        - fast_period: Fast EMA period (default: 12)
        - slow_period: Slow EMA period (default: 26)
        - signal_period: Signal line EMA period (default: 9)
        - stop_loss: Stop loss percentage (default: 0.02)
        - take_profit: Take profit percentage (default: 0.05)
        ''',
        'default_parameters': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'stop_loss': 0.02,
            'take_profit': 0.05,
        },
        'analytic_tools_used': ['MACD'],
        'example_code': '''
        # Pseudocode
        if macd_line[t] > signal_line[t] and macd_line[t-1] <= signal_line[t-1]:
            signal = BUY
        elif macd_line[t] < signal_line[t] and macd_line[t-1] >= signal_line[t-1]:
            signal = SELL
        ''',
        'globally_enabled': False,
    },
]

