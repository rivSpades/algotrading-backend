"""
OHLCV Service
Handles saving and managing OHLCV data
"""

from typing import List, Dict, Optional
from datetime import datetime
from django.utils import timezone
from django.db import transaction
from ..models import Symbol, OHLCV, Provider


class OHLCVService:
    """Service for managing OHLCV data"""
    
    @staticmethod
    def get_or_create_yahoo_provider() -> Provider:
        """Get or create Yahoo Finance provider"""
        provider, created = Provider.objects.get_or_create(
            code='YAHOO',
            defaults={
                'name': 'Yahoo Finance',
                'is_active': True
            }
        )
        return provider
    
    @staticmethod
    def save_ohlcv_data(
        symbol: Symbol,
        ohlcv_data: List[Dict],
        timeframe: str = 'daily',
        provider: Optional[Provider] = None,
        replace_existing: bool = False
    ) -> Dict:
        """
        Save OHLCV data for a symbol
        
        Args:
            symbol: Symbol instance
            ohlcv_data: List of OHLCV data dictionaries with keys:
                - timestamp: datetime
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: int
            timeframe: Timeframe ('daily', 'hourly', 'minute')
            provider: Provider instance (optional)
            replace_existing: If True, delete existing data for the date range before saving
        
        Returns:
            Dictionary with save statistics:
            {
                'created': int,
                'updated': int,
                'errors': int,
                'total': int
            }
        """
        if not ohlcv_data:
            return {'created': 0, 'updated': 0, 'errors': 0, 'total': 0}
        
        # Get or set provider
        if provider is None:
            provider = OHLCVService.get_or_create_yahoo_provider()
        
        # Update symbol provider if not set
        if symbol.provider is None:
            symbol.provider = provider
            symbol.save(update_fields=['provider'])
        
        created_count = 0
        updated_count = 0
        error_count = 0
        
        # Get date range for replacement if needed
        if replace_existing and ohlcv_data:
            timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
            if timestamps:
                min_timestamp = min(timestamps)
                max_timestamp = max(timestamps)
                # Delete existing data in this range
                OHLCV.objects.filter(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp__gte=min_timestamp,
                    timestamp__lte=max_timestamp
                ).delete()
        
        # Save data in batches
        with transaction.atomic():
            for data in ohlcv_data:
                try:
                    timestamp = data.get('timestamp')
                    if not timestamp:
                        error_count += 1
                        continue
                    
                    # Ensure timestamp is timezone-aware
                    if timezone.is_naive(timestamp):
                        timestamp = timezone.make_aware(timestamp)
                    
                    ohlcv, created = OHLCV.objects.update_or_create(
                        symbol=symbol,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        defaults={
                            'open': data.get('open', 0),
                            'high': data.get('high', 0),
                            'low': data.get('low', 0),
                            'close': data.get('close', 0),
                            'volume': data.get('volume', 0)
                        }
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1
                        
                except Exception as e:
                    print(f"Error saving OHLCV data for {symbol.ticker} at {data.get('timestamp')}: {str(e)}")
                    error_count += 1
        
        return {
            'created': created_count,
            'updated': updated_count,
            'errors': error_count,
            'total': len(ohlcv_data)
        }




