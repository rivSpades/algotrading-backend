# Strategy Product Requirements Document (PRD)

## 1. Strategy Overview

### Strategy Name
**[Strategy Name Here]**

### Short Description
**[Brief one-line description of the strategy (max 200 characters)]**

### Long Description
**[Detailed description explaining the strategy's philosophy, approach, and methodology]**

---

## 2. Position Mode Configuration

### Supported Position Modes
Choose one or more:

- [ ] **ALL** - Strategy can execute both long and short positions
- [ ] **LONG** - Strategy only executes long positions
- [ ] **SHORT** - Strategy only executes short positions

**Selected Modes:** `[ALL / LONG / SHORT / ALL+LONG / ALL+SHORT / LONG+SHORT]`

### Mode-Specific Behavior
**[Explain how the strategy behaves differently in each supported mode, if applicable]**

---

## 3. Entry and Exit Points

### Long Position Rules

#### Entry Conditions
**[Describe the exact conditions that trigger a long entry]**
- Condition 1: [Description]
- Condition 2: [Description]
- Additional filters: [Any additional requirements]

#### Exit Conditions
**[Describe the exact conditions that trigger a long exit]**
- Exit Condition 1: [Description]
- Exit Condition 2: [Description]
- Stop Loss: [If applicable]
- Take Profit: [If applicable]

### Short Position Rules

#### Entry Conditions
**[Describe the exact conditions that trigger a short entry]**
- Condition 1: [Description]
- Condition 2: [Description]
- Additional filters: [Any additional requirements]

#### Exit Conditions
**[Describe the exact conditions that trigger a short exit]**
- Exit Condition 1: [Description]
- Exit Condition 2: [Description]
- Stop Loss: [If applicable]
- Take Profit: [If applicable]

### Position Management
- **Position Sizing:** [Fixed / Percentage-based / Other]
- **Maximum Position Size:** [If applicable]
- **Position Overlap:** [Can multiple positions exist simultaneously?]

---

## 4. Analytical Tools and Indicators

### Required Tools
List all analytical tools/indicators required by this strategy:

1. **Tool Name:** `[e.g., SMA]`
   - **Purpose:** [What it's used for]
   - **Parameters:** [Required parameters]
   - **Parameter Mapping:** [How strategy parameters map to tool parameters]
   - **Display Name:** [User-friendly name]
   - **Locked:** [Yes/No - can users modify parameters?]

2. **Tool Name:** `[e.g., RSI]`
   - **Purpose:** [What it's used for]
   - **Parameters:** [Required parameters]
   - **Parameter Mapping:** [How strategy parameters map to tool parameters]
   - **Display Name:** [User-friendly name]
   - **Locked:** [Yes/No]

**[Add more tools as needed]**

### Tool Configuration JSON Format
```json
{
  "analytic_tools_used": ["SMA", "RSI", "MACD"],
  "required_tool_configs": [
    {
      "tool_name": "SMA",
      "parameters": {
        "period": 20
      },
      "parameter_mapping": {
        "period": "fast_period"
      },
      "display_name": "Fast SMA",
      "locked": true
    },
    {
      "tool_name": "SMA",
      "parameters": {
        "period": 50
      },
      "parameter_mapping": {
        "period": "slow_period"
      },
      "display_name": "Slow SMA",
      "locked": true
    }
  ]
}
```

---

## 5. Strategy Parameters

### Default Parameters
**[List all configurable parameters with their default values and descriptions]**

```json
{
  "parameter_name_1": {
    "default": 20,
    "description": "Period for fast moving average",
    "min": 1,
    "max": 200,
    "type": "integer"
  },
  "parameter_name_2": {
    "default": 50,
    "description": "Period for slow moving average",
    "min": 1,
    "max": 200,
    "type": "integer"
  }
}
```

### Parameter Descriptions
- **parameter_name_1:** [Detailed description]
- **parameter_name_2:** [Detailed description]

---

## 6. Implementation Details

### Signal Generation Logic

#### Signal Return Values
- `'buy'` - Enter long position or exit short position
- `'sell'` - Enter short position or exit long position
- `None` - No action

#### Signal Generation Flow
```
1. Check if indicators are available
2. Check current position status
3. Evaluate entry conditions (if no position)
4. Evaluate exit conditions (if position exists)
5. Return appropriate signal or None
```

### Special Considerations
- **[Any edge cases, special handling, or considerations]**
- **[Crossover detection requirements]**
- **[Previous indicator value requirements]**
- **[Timing considerations]**

---

## 7. Sample Python Implementation

### Strategy Signal Function
```python
def _strategy_name_signal(self, row: pd.Series, indicators: Dict, position: Optional[Dict], prev_indicators: Optional[Dict] = None) -> Optional[str]:
    """
    [Strategy Name] signal generation logic
    
    Args:
        row: Current OHLCV row with price data
        indicators: Dictionary of current indicator values (e.g., {'SMA_20': 100.5, 'RSI_14': 65.2})
        position: Current position dict if open (e.g., {'type': 'buy', 'entry_price': 100.0, ...})
        prev_indicators: Previous indicator values (required for crossover detection)
    
    Returns:
        'buy', 'sell', or None
    """
    # Get parameters
    param1 = self.parameters.get('parameter_name_1', default_value)
    param2 = self.parameters.get('parameter_name_2', default_value)
    
    # Get indicator values
    indicator1 = indicators.get(f'IndicatorName_{param1}')
    indicator2 = indicators.get(f'IndicatorName_{param2}')
    
    # Validate indicators are available
    if indicator1 is None or indicator2 is None:
        return None
    
    # Check if we have a position
    has_position = position is not None
    position_type = position['type'] if has_position else None
    
    # LONG ENTRY LOGIC
    if not has_position:
        # Check entry conditions for long
        if self.position_mode in ('all', 'long'):
            if [ENTRY_CONDITION_1] and [ENTRY_CONDITION_2]:
                logger.info(f"[{self.position_mode.upper()}] LONG ENTRY signal: [reason]")
                return 'buy'
        
        # Check entry conditions for short
        if self.position_mode in ('all', 'short'):
            if [ENTRY_CONDITION_1] and [ENTRY_CONDITION_2]:
                logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: [reason]")
                return 'sell'
    
    # EXIT LOGIC
    elif has_position:
        # Exit long position
        if position_type == 'buy':
            if [EXIT_CONDITION_1] or [EXIT_CONDITION_2]:
                logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: [reason]")
                return 'sell'
        
        # Exit short position
        elif position_type == 'sell':
            if [EXIT_CONDITION_1] or [EXIT_CONDITION_2]:
                logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: [reason]")
                return 'buy'
    
    return None
```

### Integration Point
**[Specify where this function needs to be added in the codebase]**

File: `algo_trading_backend/backtest_engine/services/backtest_executor.py`

1. Add the signal function to the `BacktestExecutor` class
2. Register it in `_generate_signal()` method:
```python
elif strategy_name == '[Strategy Name]':
    return self._strategy_name_signal(row, indicators, position, prev_indicators)
```

---

## 8. Testing Requirements

### Test Scenarios
1. **[Scenario 1: Description]**
2. **[Scenario 2: Description]**
3. **[Edge case scenario]**

### Expected Behaviors
- **[Behavior 1]**
- **[Behavior 2]**

---

## 9. Performance Considerations

### Computational Requirements
- **[Any performance considerations, optimization needs]**

### Data Requirements
- **[Minimum data points required, lookback period]**

---

## 10. Database Configuration

### Strategy Definition JSON
```json
{
  "name": "[Strategy Name]",
  "description_short": "[Short description - max 200 chars]",
  "description_long": "[Long description with all details]",
  "default_parameters": {
    "parameter_name_1": default_value_1,
    "parameter_name_2": default_value_2
  },
  "analytic_tools_used": ["Tool1", "Tool2"],
  "required_tool_configs": [
    {
      "tool_name": "Tool1",
      "parameters": {"param": "value"},
      "parameter_mapping": {"param": "strategy_param"},
      "display_name": "Display Name",
      "locked": true
    }
  ],
  "globally_enabled": false
}
```

---

## 11. Documentation Notes

### Additional Notes
**[Any additional information, references, papers, or documentation]**

### Related Strategies
**[Links to similar strategies or variations]**

---

## 12. Implementation Checklist

- [ ] Strategy logic implemented in `backtest_executor.py`
- [ ] Signal function registered in `_generate_signal()` method
- [ ] Position mode handling verified (ALL/LONG/SHORT)
- [ ] Indicator requirements confirmed
- [ ] Tool configurations defined
- [ ] Default parameters set
- [ ] Edge cases handled
- [ ] Logging added for debugging
- [ ] Tests performed
- [ ] Documentation updated

---

**Created:** [Date]  
**Author:** [Name]  
**Version:** [Version Number]  
**Status:** [Draft / Review / Approved / Implemented]




