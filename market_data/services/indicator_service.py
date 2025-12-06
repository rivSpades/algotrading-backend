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
    # Get enabled tool assignments: both global (symbol=None) and symbol-specific
    from django.db.models import Q
    assignments = ToolAssignment.objects.filter(
        Q(symbol=symbol) | Q(symbol__isnull=True),
        enabled=True
    ).select_related('tool').order_by('symbol')  # Symbol-specific first, then global
    
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


def compute_strategy_indicators_for_ohlcv(strategy, ohlcv_data, symbol):
    """
    Compute strategy's required indicators for OHLCV data
    
    Args:
        strategy: StrategyDefinition instance
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
        symbol: Symbol instance (for logging)
    
    Returns:
        Dictionary mapping indicator names to their values aligned with ohlcv_data
    """
    if not strategy.required_tool_configs or not ohlcv_data:
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
    
    # Get strategy parameters
    strategy_params = strategy.default_parameters or {}
    
    # Compute each required indicator
    indicator_values = {}
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(strategy.required_tool_configs)} tool configs for strategy {strategy.name}")
    
    for idx, tool_config in enumerate(strategy.required_tool_configs, 1):
        logger.info(f"Processing tool config {idx}/{len(strategy.required_tool_configs)}: {tool_config.get('tool_name')} with params {tool_config.get('parameters')}")
        tool_name = tool_config.get('tool_name')
        if not tool_name:
            continue
        
        # Resolve parameters from strategy params
        parameters = tool_config.get('parameters', {}).copy()
        parameter_mapping = tool_config.get('parameter_mapping', {})
        
        # Map strategy parameters to tool parameters
        for tool_param, strategy_param in parameter_mapping.items():
            if strategy_params.get(strategy_param) is not None:
                parameters[tool_param] = strategy_params[strategy_param]
        
        try:
            # Create indicator key
            period = parameters.get('period', '')
            indicator_key = f"{tool_name}_{period}" if period else tool_name
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Computing strategy indicator {indicator_key} for {symbol.ticker} with parameters: {parameters}")
            
            # Compute indicator - use a copy of df to avoid modifying the original
            result_df = compute_indicator(
                tool_name=tool_name,
                ohlcv_data=df.copy(),  # Use copy to ensure each computation is independent
                parameters=parameters
            )
            
            logger.info(f"Indicator computation result for {indicator_key}: {len(result_df)} rows, columns: {list(result_df.columns)}")
            
            # Extract values
            values = []
            for idx in original_indices:
                if idx < len(result_df):
                    value = result_df.iloc[idx]['value']
                    if pd.notna(value):
                        values.append(float(value))
                    else:
                        values.append(None)
                else:
                    values.append(None)
            
            non_null_count = sum(1 for v in values if v is not None)
            logger.info(f"Extracted {len(values)} values for {indicator_key}, {non_null_count} non-null")
            
            # Get display name from tool config
            display_name = tool_config.get('display_name', tool_name)
            
            # Get style from tool config
            style = tool_config.get('style', {})
            
            indicator_values[indicator_key] = {
                'values': values,
                'display_name': display_name,
                'color': style.get('color', '#3B82F6'),
                'line_width': style.get('line_width', 2),
                'subchart': tool_config.get('subchart', False),
            }
            
            # Also create alternative key without underscores for compatibility (e.g., SMA_50 -> SMA50)
            # Replace ALL underscores, not just the first one
            alt_key = indicator_key.replace('_', '')
            if alt_key != indicator_key:
                indicator_values[alt_key] = {
                    'values': values,  # Same values array
                    'display_name': display_name,
                    'color': style.get('color', '#3B82F6'),
                    'line_width': style.get('line_width', 2),
                    'subchart': tool_config.get('subchart', False),
                }
            
            logger.info(f"Successfully computed strategy indicator {indicator_key} (alt: {alt_key}) for {symbol.ticker}: {len(values)} values, {non_null_count} non-null")
                
        except Exception as e:
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error computing strategy indicator {tool_name} for {symbol.ticker} with parameters {parameters}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Completed processing strategy indicators. Computed {len(indicator_values)} indicators: {list(indicator_values.keys())}")
    return indicator_values

