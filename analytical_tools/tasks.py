"""
Celery Tasks for Analytical Tools
"""

from celery import shared_task
from django.utils import timezone
from .models import ToolAssignment, IndicatorValue
from market_data.models import Symbol, OHLCV
from .indicators import compute_indicator
import pandas as pd
from decimal import Decimal
import json


@shared_task(bind=True, name='analytical_tools.compute_indicator')
def compute_indicator_task(self, assignment_id):
    """
    Compute indicator values for a tool assignment
    
    Args:
        assignment_id: ID of the ToolAssignment
    
    Returns:
        Dictionary with computation results
    """
    try:
        assignment = ToolAssignment.objects.get(id=assignment_id)
        
        if not assignment.enabled:
            return {
                'status': 'skipped',
                'message': f'Assignment {assignment_id} is disabled'
            }
        
        # Get OHLCV data for the symbol
        ohlcv_queryset = OHLCV.objects.filter(
            symbol=assignment.symbol
        ).order_by('timestamp')
        
        if not ohlcv_queryset.exists():
            return {
                'status': 'error',
                'message': f'No OHLCV data found for symbol {assignment.symbol.ticker}'
            }
        
        # Convert to DataFrame
        ohlcv_data = []
        for ohlcv in ohlcv_queryset:
            ohlcv_data.append({
                'timestamp': ohlcv.timestamp,
                'open': float(ohlcv.open),
                'high': float(ohlcv.high),
                'low': float(ohlcv.low),
                'close': float(ohlcv.close),
                'volume': float(ohlcv.volume)
            })
        
        df = pd.DataFrame(ohlcv_data)
        
        # Get parameters (merge with defaults)
        parameters = assignment.tool.default_parameters.copy()
        parameters.update(assignment.parameters)
        
        # Compute indicator
        result_df = compute_indicator(
            tool_name=assignment.tool.name,
            ohlcv_data=df,
            parameters=parameters
        )
        
        # Delete existing values for this assignment
        IndicatorValue.objects.filter(assignment=assignment).delete()
        
        # Create new indicator values
        created_count = 0
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
                
                IndicatorValue.objects.create(
                    assignment=assignment,
                    timestamp=row['timestamp'],
                    value=Decimal(str(row['value'])),
                    metadata=metadata
                )
                created_count += 1
        
        return {
            'status': 'completed',
            'assignment_id': assignment_id,
            'symbol': assignment.symbol.ticker,
            'tool': assignment.tool.name,
            'values_created': created_count,
            'message': f'Successfully computed {assignment.tool.name} for {assignment.symbol.ticker}'
        }
    
    except ToolAssignment.DoesNotExist:
        return {
            'status': 'error',
            'message': f'ToolAssignment {assignment_id} not found'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error computing indicator: {str(e)}'
        }



