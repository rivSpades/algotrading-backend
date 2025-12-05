"""
Indicator computation service
Computes indicators on-the-fly for OHLCV data using pandas
"""

import pandas as pd
from analytical_tools.models import ToolAssignment
from analytical_tools.indicators import compute_indicator


def compute_indicators_for_ohlcv(symbol, ohlcv_data):
    """
    Compute all enabled indicators for a symbol's OHLCV data
    
    Args:
        symbol: Symbol instance
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
    
    Returns:
        Dictionary mapping indicator names to their values aligned with ohlcv_data
        Format: {
            'SMA_15': [value1, value2, ...],  # Same length as ohlcv_data
            'RSI_14': [value1, value2, ...],
            ...
        }
    """
    # Get enabled tool assignments for this symbol
    assignments = ToolAssignment.objects.filter(
        symbol=symbol,
        enabled=True
    ).select_related('tool')
    
    if not assignments.exists() or not ohlcv_data:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp (ascending for proper indicator calculation)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Store original order indices for alignment
    original_indices = list(range(len(df)))
    
    # Compute each indicator
    indicator_values = {}
    
    for assignment in assignments:
        tool_name = assignment.tool.name
        parameters = assignment.tool.default_parameters.copy()
        parameters.update(assignment.parameters)
        
        try:
            # Compute indicator - this returns a DataFrame with same length as input
            result_df = compute_indicator(
                tool_name=tool_name,
                ohlcv_data=df,
                parameters=parameters
            )
            
            # Create indicator key (e.g., 'SMA_15' for SMA with period 15)
            period = parameters.get('period', '')
            indicator_key = f"{tool_name}_{period}" if period else tool_name
            
            # Debug: Check if result_df has correct length
            if len(result_df) != len(df):
                print(f"WARNING: result_df length ({len(result_df)}) != df length ({len(df)}) for {tool_name}")
            
            # Extract values directly by index - result_df should be aligned with df
            # Since compute_indicator works on the same DataFrame, indices match
            values = []
            non_null_count = 0
            for idx in original_indices:
                if idx < len(result_df):
                    value = result_df.iloc[idx]['value']
                    # Convert NaN to None for JSON serialization
                    if pd.notna(value):
                        values.append(float(value))
                        non_null_count += 1
                    else:
                        values.append(None)
                else:
                    values.append(None)
            
            # Debug: Log computation results
            print(f"Computed {indicator_key} for {symbol.ticker}: {non_null_count}/{len(values)} non-null values")
            
            indicator_values[indicator_key] = {
                'values': values,
                'color': assignment.style.get('color', '#3B82F6') if assignment.style else '#3B82F6',
                'line_width': assignment.style.get('line_width', 2) if assignment.style else 2,
                'subchart': assignment.subchart,  # Include subchart flag
            }
        except Exception as e:
            # Skip indicator if computation fails
            import traceback
            print(f"Error computing {tool_name} for {symbol.ticker}: {str(e)}")
            print(traceback.format_exc())
            continue
    
    return indicator_values

