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
            # Create indicator key and display name first (needed for both regular and Bollinger Bands)
            period = parameters.get('period', '')
            indicator_key = f"{tool_name}_{period}" if period else tool_name
            
            # Create display name (e.g., 'SMA15' for SMA with period 15) - used for legend
            if tool_name == 'MACD':
                # MACD has multiple parameters
                fast = parameters.get('fast_period', 12)
                slow = parameters.get('slow_period', 26)
                signal = parameters.get('signal_period', 9)
                display_name = f"MACD({fast},{slow},{signal})"
            elif tool_name == 'BollingerBands':
                # Bollinger Bands display name
                num_std = parameters.get('num_std', 2.0)
                if period:
                    display_name = f"BB{period}"
                else:
                    display_name = "BB"
            elif period:
                display_name = f"{tool_name}{period}"
            else:
                display_name = tool_name
            
            # Compute indicator - this returns a DataFrame with same length as input
            result_df = compute_indicator(
                tool_name=tool_name,
                ohlcv_data=df,
                parameters=parameters
            )
            
            # Debug: Check if result_df has correct length
            if len(result_df) != len(df):
                print(f"WARNING: result_df length ({len(result_df)}) != df length ({len(df)}) for {tool_name}")
            
            # Special handling for Bollinger Bands (multiple series)
            if tool_name == 'BollingerBands':
                # Extract metadata for Bollinger Bands
                metadata_list = []
                for idx in original_indices:
                    if idx < len(result_df) and 'metadata' in result_df.columns:
                        meta = result_df.iloc[idx]['metadata']
                        if isinstance(meta, dict):
                            metadata_list.append(meta)
                        else:
                            metadata_list.append({})
                    else:
                        metadata_list.append({})
                
                # Extract upper, middle, lower, and bandwidth values
                upper_values = [m.get('upper') if m else None for m in metadata_list]
                middle_values = [m.get('middle') if m else None for m in metadata_list]
                lower_values = [m.get('lower') if m else None for m in metadata_list]
                bandwidth_values = [m.get('bandwidth') if m else None for m in metadata_list]
                
                # Create separate keys for each band
                base_key = indicator_key
                upper_key = f"{base_key}_upper"
                middle_key = f"{base_key}_middle"
                lower_key = f"{base_key}_lower"
                bandwidth_key = f"{base_key}_bandwidth"
                
                # Get colors for each band (support separate colors)
                style = assignment.style or {}
                upper_color = style.get('upper_color') or style.get('color', '#3B82F6')
                middle_color = style.get('middle_color') or style.get('color', '#3B82F6')
                lower_color = style.get('lower_color') or style.get('color', '#3B82F6')
                bandwidth_color = style.get('bandwidth_color') or style.get('color', '#3B82F6')
                line_width = style.get('line_width', 2)
                
                # Store bands (upper, middle, lower) - these go on main chart
                indicator_values[upper_key] = {
                    'values': upper_values,
                    'display_name': f"{display_name} Upper",
                    'color': upper_color,
                    'line_width': line_width,
                    'subchart': False,  # Bands go on main chart
                }
                indicator_values[middle_key] = {
                    'values': middle_values,
                    'display_name': f"{display_name} Middle",
                    'color': middle_color,
                    'line_width': line_width,
                    'subchart': False,  # Bands go on main chart
                }
                indicator_values[lower_key] = {
                    'values': lower_values,
                    'display_name': f"{display_name} Lower",
                    'color': lower_color,
                    'line_width': line_width,
                    'subchart': False,  # Bands go on main chart
                }
                
                # Store bandwidth - this goes in subchart
                indicator_values[bandwidth_key] = {
                    'values': bandwidth_values,
                    'display_name': f"{display_name} Bandwidth",
                    'color': bandwidth_color,
                    'line_width': line_width,
                    'subchart': True,  # Bandwidth goes in subchart
                }
                
                # Debug: Log computation results
                print(f"Computed Bollinger Bands {indicator_key} ({display_name}) for {symbol.ticker}: upper/middle/lower/bandwidth")
            else:
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
                print(f"Computed {indicator_key} ({display_name}) for {symbol.ticker}: {non_null_count}/{len(values)} non-null values")
                
                indicator_values[indicator_key] = {
                    'values': values,
                    'display_name': display_name,  # User-friendly name for legend
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

