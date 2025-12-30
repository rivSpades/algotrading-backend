# Strategy PRD Example: Simple Moving Average Crossover

This is a filled-out example of the PRD template to help you understand how to fill it out.

---

## 1. Strategy Overview

### Strategy Name
**Simple Moving Average Crossover**

### Short Description
**Long and short strategy: Long when fast SMA crosses above slow SMA, Short when fast SMA crosses below slow SMA**

### Long Description
**A momentum strategy that uses two Simple Moving Averages (SMA) and supports both long and short positions:
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
- SHORT: Only executes short positions (ignores long entry signals)**

---

## 2. Position Mode Configuration

### Supported Position Modes
- [x] **ALL** - Strategy can execute both long and short positions
- [x] **LONG** - Strategy only executes long positions
- [x] **SHORT** - Strategy only executes short positions

**Selected Modes:** `ALL, LONG, SHORT (all three modes)`

### Mode-Specific Behavior
**- ALL mode: Executes both long and short trades as signals occur**
**- LONG mode: Only enters/exits long positions, ignores short entry signals**
**- SHORT mode: Only enters/exits short positions, ignores long entry signals**

---

## 3. Entry and Exit Points

### Long Position Rules

#### Entry Conditions
**- Fast SMA crosses above Slow SMA (golden cross)**
**- No existing position**

#### Exit Conditions
**- Fast SMA crosses below Slow SMA (death cross)**
**- This also serves as the entry signal for SHORT mode**

### Short Position Rules

#### Entry Conditions
**- Fast SMA crosses below Slow SMA (death cross)**
**- No existing position**

#### Exit Conditions
**- Fast SMA crosses above Slow SMA (golden cross)**
**- This also serves as the entry signal for LONG mode**

---

## 4. Analytical Tools and Indicators

### Required Tools

1. **Tool Name:** `SMA`
   - **Purpose:** Fast moving average for trend detection
   - **Parameters:** `period: 20`
   - **Parameter Mapping:** `period -> fast_period`
   - **Display Name:** `Fast SMA`
   - **Locked:** `Yes`

2. **Tool Name:** `SMA`
   - **Purpose:** Slow moving average for trend detection
   - **Parameters:** `period: 50`
   - **Parameter Mapping:** `period -> slow_period`
   - **Display Name:** `Slow SMA`
   - **Locked:** `Yes`

### Tool Configuration JSON
```json
{
  "analytic_tools_used": ["SMA"],
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
```json
{
  "fast_period": {
    "default": 20,
    "description": "Period for fast moving average",
    "min": 1,
    "max": 200,
    "type": "integer"
  },
  "slow_period": {
    "default": 50,
    "description": "Period for slow moving average",
    "min": 1,
    "max": 200,
    "type": "integer"
  }
}
```

---

## 6. Implementation Details

### Signal Generation Logic

#### Signal Return Values
- `'buy'` - Enter long position or exit short position
- `'sell'` - Enter short position or exit long position
- `None` - No action

#### Special Considerations
- **Requires previous indicator values for crossover detection**
- **Crossover is detected by comparing current and previous indicator relationships**
- **Must handle edge case when previous indicators are not available**

---

## 7. Sample Python Implementation

```python
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
        return None
    
    # Need previous values to detect crossover
    if prev_indicators is None:
        return None
    
    prev_fast_sma = prev_indicators.get(fast_sma_key)
    prev_slow_sma = prev_indicators.get(slow_sma_key)
    
    if prev_fast_sma is None or prev_slow_sma is None:
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
        elif position_type == 'sell':
            # Short position open: Exit SHORT (close short = buy)
            logger.info(f"[{self.position_mode.upper()}] SHORT EXIT signal: Fast SMA ({fast_sma:.2f}) crossed above Slow SMA ({slow_sma:.2f})")
            return 'buy'
    
    # Detect crossover: fast crosses below slow (death cross)
    elif prev_fast_sma >= prev_slow_sma and fast_sma < slow_sma:
        if not has_position:
            # No position: Enter SHORT (only if position_mode allows shorts)
            if self.position_mode in ('all', 'short'):
                logger.info(f"[{self.position_mode.upper()}] SHORT ENTRY signal: Fast SMA ({fast_sma:.2f}) crossed below Slow SMA ({slow_sma:.2f})")
                return 'sell'
        elif position_type == 'buy':
            # Long position open: Exit LONG (close long = sell)
            logger.info(f"[{self.position_mode.upper()}] LONG EXIT signal: Fast SMA ({fast_sma:.2f}) crossed below Slow SMA ({slow_sma:.2f})")
            return 'sell'
    
    return None
```

---

## 8. Database Configuration

### Strategy Definition JSON
```json
{
  "name": "Simple Moving Average Crossover",
  "description_short": "Long and short strategy: Long when fast SMA crosses above slow SMA, Short when fast SMA crosses below slow SMA",
  "description_long": "A momentum strategy that uses two Simple Moving Averages (SMA) and supports both long and short positions:\n- Fast SMA: Shorter period (default: 20 days)\n- Slow SMA: Longer period (default: 50 days)\n\nLong Position:\n- Entry Signal: When fast SMA crosses above slow SMA (golden cross)\n- Exit Signal: When fast SMA crosses below slow SMA (death cross)\n\nShort Position:\n- Entry Signal: When fast SMA crosses below slow SMA (death cross)\n- Exit Signal: When fast SMA crosses above slow SMA (golden cross)\n\nThe strategy can be run in three modes:\n- ALL: Executes both long and short positions\n- LONG: Only executes long positions (ignores short entry signals)\n- SHORT: Only executes short positions (ignores long entry signals)",
  "default_parameters": {
    "fast_period": 20,
    "slow_period": 50
  },
  "analytic_tools_used": ["SMA"],
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
  ],
  "globally_enabled": false
}
```

---

This example shows how to fill out each section of the PRD template. Use this as a reference when creating new strategy PRDs.









