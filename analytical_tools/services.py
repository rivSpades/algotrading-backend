"""
Synchronous indicator computation service
Computes indicators directly with pandas and saves efficiently
"""

from .models import ToolAssignment, IndicatorValue
from market_data.models import OHLCV
from .indicators import compute_indicator
import pandas as pd
from decimal import Decimal
import json
from django.db import transaction


def compute_indicator_sync(assignment):
    """
    Compute indicator values synchronously using pandas directly on OHLCV data
    
    Args:
        assignment: ToolAssignment instance
    
    Returns:
        Dictionary with computation results
    """
    try:
        if not assignment.enabled:
            return {
                'status': 'skipped',
                'message': f'Assignment {assignment.id} is disabled'
            }
        
        # Get OHLCV data for the symbol - use values() for efficiency
        ohlcv_queryset = OHLCV.objects.filter(
            symbol=assignment.symbol
        ).order_by('timestamp').values('timestamp', 'open', 'high', 'low', 'close', 'volume')
        
        if not ohlcv_queryset.exists():
            return {
                'status': 'error',
                'message': f'No OHLCV data found for symbol {assignment.symbol.ticker}'
            }
        
        # Convert directly to DataFrame (much faster)
        df = pd.DataFrame(list(ohlcv_queryset))
        
        # Get parameters (merge with defaults)
        parameters = assignment.tool.default_parameters.copy()
        parameters.update(assignment.parameters)
        
        # Compute indicator directly on dataframe
        result_df = compute_indicator(
            tool_name=assignment.tool.name,
            ohlcv_data=df,
            parameters=parameters
        )
        
        # Bulk delete and create for efficiency
        with transaction.atomic():
            # Delete existing values
            IndicatorValue.objects.filter(assignment=assignment).delete()
            
            # Prepare bulk create data
            indicator_values = []
            for _, row in result_df.iterrows():
                if pd.notna(row['value']):
                    metadata = {}
                    if 'metadata' in row and pd.notna(row['metadata']):
                        if isinstance(row['metadata'], dict):
                            metadata = row['metadata']
                        elif isinstance(row['metadata'], str):
                            try:
                                metadata = json.loads(row['metadata'])
                            except:
                                metadata = {}
                    
                    indicator_values.append(
                        IndicatorValue(
                            assignment=assignment,
                            timestamp=row['timestamp'],
                            value=Decimal(str(row['value'])),
                            metadata=metadata
                        )
                    )
            
            # Bulk create (much faster than individual creates)
            if indicator_values:
                IndicatorValue.objects.bulk_create(indicator_values, batch_size=1000)
        
        return {
            'status': 'completed',
            'assignment_id': assignment.id,
            'symbol': assignment.symbol.ticker,
            'tool': assignment.tool.name,
            'values_created': len(indicator_values),
            'message': f'Successfully computed {assignment.tool.name} for {assignment.symbol.ticker}'
        }
    
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': f'Error computing indicator: {str(e)}',
            'traceback': traceback.format_exc()
        }

