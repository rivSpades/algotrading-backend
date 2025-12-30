"""
Strategy Definitions Configuration
Hardcoded strategy definitions that are automatically created/updated on app startup
"""

STRATEGY_DEFINITIONS = [
    {
        'name': 'Gap-Up and Gap-Down',
        'description_short': 'Trades gap-up/down patterns with volatility-adjusted thresholds.',
        'description_long': '''This strategy measures today's opening gap relative to the previous close and normalizes it with a rolling standard deviation of returns. To avoid lookahead bias, the rolling standard deviation is calculated strictly using data up to today's open, not today's close.

Steps:
1. Calculate returns: returns_t = (Open_t - Close_{t-1}) / Close_{t-1}
2. Compute rolling standard deviation using a rolling window of N days (default: 90), excluding today's close
3. Evaluate momentum:
   - Gap-Up → bullish if returns > (threshold × std)
   - Gap-Down → bearish if returns < -(threshold × std)
4. Execute trades based on mode (LONG / SHORT / ALL)''',
        'default_parameters': {
            'threshold': 0.25,
            'std_period': 90,
        },
        'analytic_tools_used': ['RollingSTD', 'Returns'],
        'required_tool_configs': [
            {
                'tool_name': 'Returns',
                'parameters': {},
                'parameter_mapping': {},
                'display_name': 'Gap Returns',
                'locked': True,
            },
            {
                'tool_name': 'RollingSTD',
                'parameters': {'period': 90},
                'parameter_mapping': {'period': 'std_period'},
                'display_name': 'Standard Deviation (Bias-Safe)',
                'locked': False,
            },
        ],
        'globally_enabled': False,
    },
]
