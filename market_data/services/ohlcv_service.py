"""
OHLCV Service
Handles saving and managing OHLCV data
"""

from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, date, timedelta
from django.utils import timezone
from django.db import transaction
from ..models import Symbol, OHLCV, Provider
from .data_validation import validate_ohlcv_data


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
    def get_provider(provider_code: str = 'YAHOO') -> Provider:
        """
        Get provider by code (defaults to YAHOO for backward compatibility)
        
        Args:
            provider_code: Provider code (e.g., 'YAHOO', 'POLYGON')
        
        Returns:
            Provider instance
        """
        try:
            provider = Provider.objects.get(code=provider_code.upper(), is_active=True)
            return provider
        except Provider.DoesNotExist:
            # Fallback to Yahoo if provider not found
            return OHLCVService.get_or_create_yahoo_provider()
    
    @staticmethod
    def get_existing_timestamps(
        symbol: Symbol,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        timeframe: str = 'daily'
    ) -> set:
        """
        Get set of existing timestamps for a symbol in the given date range
        
        Args:
            symbol: Symbol instance
            start_date: Start date (datetime object, optional)
            end_date: End date (datetime object, optional)
            timeframe: Timeframe ('daily', 'hourly', 'minute')
        
        Returns:
            Set of timestamps (as date objects for daily, or datetime for other timeframes)
        """
        query = OHLCV.objects.filter(
            symbol=symbol,
            timeframe=timeframe
        )
        
        if start_date:
            query = query.filter(timestamp__gte=start_date)
        if end_date:
            query = query.filter(timestamp__lte=end_date)
        
        if timeframe == 'daily':
            # For daily data, normalize to date (ignore time) for comparison
            existing_dates = set()
            for ohlcv_ts in query.values_list('timestamp', flat=True):
                if isinstance(ohlcv_ts, datetime):
                    existing_dates.add(ohlcv_ts.date())
                elif isinstance(ohlcv_ts, date):
                    existing_dates.add(ohlcv_ts)
            return existing_dates
        else:
            # For hourly/minute, use full timestamps
            return set(query.values_list('timestamp', flat=True))
    
    @staticmethod
    def get_missing_timestamps(
        symbol: Symbol,
        requested_timestamps: List[datetime],
        timeframe: str = 'daily'
    ) -> List[datetime]:
        """
        Filter requested timestamps to only include those that don't exist in the database
        
        Args:
            symbol: Symbol instance
            requested_timestamps: List of timestamps to check
            timeframe: Timeframe ('daily', 'hourly', 'minute')
        
        Returns:
            List of timestamps that are missing from the database
        """
        if not requested_timestamps:
            return []
        
        # Get date range from requested timestamps
        min_timestamp = min(requested_timestamps)
        max_timestamp = max(requested_timestamps)
        
        # Get existing timestamps in the range
        existing = OHLCVService.get_existing_timestamps(
            symbol=symbol,
            start_date=min_timestamp,
            end_date=max_timestamp,
            timeframe=timeframe
        )
        
        # Filter to only missing timestamps
        missing = []
        for ts in requested_timestamps:
            if timeframe == 'daily':
                # For daily, compare dates
                ts_date = ts.date() if isinstance(ts, datetime) else ts
                if ts_date not in existing:
                    missing.append(ts)
            else:
                # For other timeframes, compare full timestamps
                if ts not in existing:
                    missing.append(ts)
        
        return missing
    
    @staticmethod
    def get_latest_timestamp(
        symbol: Symbol,
        timeframe: str = 'daily'
    ) -> Optional[datetime]:
        """
        Get the latest timestamp for a symbol
        
        Args:
            symbol: Symbol instance
            timeframe: Timeframe ('daily', 'hourly', 'minute')
        
        Returns:
            Latest timestamp as datetime, or None if no data exists
        """
        latest_ohlcv = OHLCV.objects.filter(
            symbol=symbol,
            timeframe=timeframe
        ).order_by('-timestamp').first()
        
        if latest_ohlcv:
            return latest_ohlcv.timestamp
        return None
    
    @staticmethod
    def check_date_range_coverage(
        symbol: Symbol,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        timeframe: str = 'daily'
    ) -> Tuple[bool, Set]:
        """
        Check if a date range is already fully covered by existing data
        
        Args:
            symbol: Symbol instance
            start_date: Start date (datetime object, optional)
            end_date: End date (datetime object, optional)
            timeframe: Timeframe ('daily', 'hourly', 'minute')
        
        Returns:
            Tuple of (is_fully_covered, existing_timestamps_set)
            If start_date/end_date not provided, returns (False, empty_set)
        """
        if not start_date or not end_date or timeframe != 'daily':
            # Can't check coverage without dates or for non-daily timeframes
            return False, set()
        
        # Normalize to dates for daily data
        start_d = start_date.date() if isinstance(start_date, datetime) else start_date
        end_d = end_date.date() if isinstance(end_date, datetime) else end_date
        
        # Get existing dates in the range
        existing_dates = OHLCVService.get_existing_timestamps(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        # For daily data, we can't know exactly which trading days exist
        # So we check if we have reasonable coverage: data at start, end, and some in between
        # If we have data spanning the full range (start to end), consider it covered
        if existing_dates:
            existing_dates_sorted = sorted(existing_dates)
            earliest_existing = existing_dates_sorted[0]
            latest_existing = existing_dates_sorted[-1]
            
            # Check if the range is covered (with some tolerance for missing days)
            # We consider it covered if we have data at or before start_date and at or after end_date
            is_covered = earliest_existing <= start_d and latest_existing >= end_d
            
            return is_covered, existing_dates
        
        return False, set()
    
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
        # Validate data before saving
        is_valid, validation_reason = validate_ohlcv_data(ohlcv_data)
        
        if not ohlcv_data:
            # Mark as invalid if no data
            symbol.validation_status = 'invalid'
            symbol.validation_reason = 'No data provided'
            symbol.status = 'disabled'
            symbol.save(update_fields=['validation_status', 'validation_reason', 'status'])
            return {
                'created': 0, 
                'updated': 0, 
                'errors': 0, 
                'total': 0,
                'validation_passed': False,
                'validation_reason': 'No data provided'
            }
        
        # Update symbol validation status
        symbol.validation_status = 'valid' if is_valid else 'invalid'
        symbol.validation_reason = validation_reason if not is_valid else ''
        
        # Set symbol status based on validation
        if is_valid:
            symbol.status = 'active'
        else:
            symbol.status = 'disabled'
        
        symbol.save(update_fields=['validation_status', 'validation_reason', 'status'])
        
        # Get or set provider
        if provider is None:
            provider = OHLCVService.get_or_create_yahoo_provider()
        
        # Check if symbol has a different provider
        if symbol.provider is not None and symbol.provider.id != provider.id:
            # Only allow overwrite if symbol is in disabled state
            if symbol.status != 'disabled':
                raise ValueError(
                    f"Cannot update OHLCV data for symbol {symbol.ticker} with provider {provider.code}. "
                    f"Symbol already has data from provider {symbol.provider.code} and is in '{symbol.status}' status. "
                    f"Only disabled symbols can have their provider changed. Please disable the symbol first."
                )
            
            # Symbol is disabled - allow overwrite: delete all existing OHLCV data and update provider
            deleted_count, _ = OHLCV.objects.filter(symbol=symbol).delete()
            
            # Update provider field
            symbol.provider = provider
            symbol.save(update_fields=['provider'])
            
            # Reset replace_existing flag since we've already deleted all data
            replace_existing = False
        
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
        
        # Save data in batches - only create new records (don't update existing ones)
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
                    
                    # Only create if it doesn't exist (don't update existing records)
                    ohlcv, created = OHLCV.objects.get_or_create(
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
                    # If not created, it already exists - skip it (don't update)
                    # updated_count remains 0 as we're not updating existing records
                        
                except Exception as e:
                    print(f"Error saving OHLCV data for {symbol.ticker} at {data.get('timestamp')}: {str(e)}")
                    error_count += 1
        
        return {
            'created': created_count,
            'updated': updated_count,
            'errors': error_count,
            'total': len(ohlcv_data),
            'validation_passed': is_valid,
            'validation_reason': validation_reason
        }




