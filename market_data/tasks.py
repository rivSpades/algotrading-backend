"""
Celery tasks for market data operations
"""

from celery import shared_task
from django.utils import timezone
from django.db import connections, transaction
from typing import List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .models import Symbol, OHLCV, Exchange, Provider
from .services.symbol_service import SymbolService
from .services.ohlcv_service import OHLCVService
from .providers.eod_api import EODAPIProvider
from .providers.yahoo_finance import YahooFinanceProvider
from .providers.factory import ProviderFactory


@shared_task(bind=True)
def update_symbol_data_task(self, ticker):
    """
    Background task to update symbol data
    This is a placeholder - actual implementation will depend on provider
    """
    try:
        symbol = Symbol.objects.get(ticker=ticker)
        
        # Update task state
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Starting update for {ticker}'
            }
        )

        # TODO: Implement actual data fetching from provider
        # This is a placeholder that simulates the update process
        
        # Simulate progress updates
        for i in range(1, 101):
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': i,
                    'message': f'Processing data... {i}%'
                }
            )
            # Simulate work
            import time
            time.sleep(0.1)

        # Update symbol timestamp
        symbol.last_updated = timezone.now()
        symbol.save(update_fields=['last_updated'])

        return {
            'progress': 100,
            'message': f'Successfully updated {ticker}',
            'status': 'completed'
        }

    except Symbol.DoesNotExist:
        return {
            'progress': 0,
            'message': f'Symbol {ticker} not found',
            'status': 'failed'
        }
    except Exception as e:
        return {
            'progress': 0,
            'message': f'Error updating {ticker}: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def fetch_symbols_from_exchange_task(self, exchange_code: str):
    """
    Background task to fetch and import symbols from an exchange
    Updates progress during processing
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Starting symbol fetch for {exchange_code}'
            }
        )
        
        # Fetch symbols data
        symbols_data = EODAPIProvider.get_exchange_symbols(exchange_code)
        total_symbols = len(symbols_data)
        
        if total_symbols == 0:
            return {
                'progress': 100,
                'message': f'No symbols found for {exchange_code}',
                'status': 'completed',
                'result': {'exchange': exchange_code, 'total': 0, 'created': 0, 'updated': 0, 'errors': 0}
            }
        
        # Get all exchanges list to map symbol exchanges properly
        exchanges_list = EODAPIProvider.get_exchanges_list()
        exchanges_map = {ex.get('Code'): ex for ex in exchanges_list if ex.get('Code')}
        
        created_count = 0
        updated_count = 0
        error_count = 0
        exchange_stats = {}
        
        # Process symbols in batches and update progress
        for idx, symbol_data in enumerate(symbols_data):
            # Update progress
            progress = int((idx / total_symbols) * 90)  # Reserve 10% for final processing
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processing symbol {idx + 1}/{total_symbols} from {exchange_code}'
                }
            )
            
            # Each symbol has its own Exchange field (NYSE, NASDAQ, etc.)
            symbol_exchange_code = symbol_data.get('Exchange', exchange_code)
            
            # Get exchange data from map or create basic data
            if symbol_exchange_code in exchanges_map:
                exchange_data = exchanges_map[symbol_exchange_code]
            else:
                exchange_data = {
                    'Code': symbol_exchange_code,
                    'Name': symbol_exchange_code,
                    'Country': '',
                    'Timezone': 'UTC'
                }
            
            # Get or create exchange with proper data
            exchange = SymbolService.get_or_create_exchange(exchange_data)
            
            # Provider should not be set for symbol creation - only for OHLCV data
            symbol = SymbolService.create_symbol_from_eod_data(symbol_data, exchange, None)
            if symbol:
                if symbol.created_at == symbol.updated_at:
                    created_count += 1
                else:
                    updated_count += 1
                
                # Track stats per exchange
                if symbol_exchange_code not in exchange_stats:
                    exchange_stats[symbol_exchange_code] = {'created': 0, 'updated': 0}
                if symbol.created_at == symbol.updated_at:
                    exchange_stats[symbol_exchange_code]['created'] += 1
                else:
                    exchange_stats[symbol_exchange_code]['updated'] += 1
            else:
                error_count += 1
        
        result = {
            'exchange': exchange_code,
            'total': total_symbols,
            'created': created_count,
            'updated': updated_count,
            'errors': error_count,
            'exchange_stats': exchange_stats
        }
        
        # Final update
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 100,
                'message': f'Completed: {created_count} created, {updated_count} updated'
            }
        )
        
        return {
            'progress': 100,
            'message': f'Successfully imported {created_count} new and {updated_count} updated symbols from {exchange_code}',
            'status': 'completed',
            'result': result
        }
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching symbols from {exchange_code}: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching symbols from {exchange_code}: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def fetch_symbols_from_multiple_exchanges_task(self, exchange_codes: List[str]):
    """
    Background task to fetch and import symbols from multiple exchanges
    Updates progress during processing
    """
    try:
        total_exchanges = len(exchange_codes)
        results = {}
        
        for idx, exchange_code in enumerate(exchange_codes):
            # Calculate progress (reserve 5% for final processing)
            progress = int((idx / total_exchanges) * 95)
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processing {exchange_code} ({idx + 1}/{total_exchanges})'
                }
            )
            
            # Call the single exchange task to get progress updates
            # We'll process it inline to maintain progress tracking
            try:
                symbols_data = EODAPIProvider.get_exchange_symbols(exchange_code)
                exchanges_list = EODAPIProvider.get_exchanges_list()
                exchanges_map = {ex.get('Code'): ex for ex in exchanges_list if ex.get('Code')}
                
                created_count = 0
                updated_count = 0
                error_count = 0
                exchange_stats = {}
                
                for symbol_data in symbols_data:
                    symbol_exchange_code = symbol_data.get('Exchange', exchange_code)
                    if symbol_exchange_code in exchanges_map:
                        exchange_data = exchanges_map[symbol_exchange_code]
                    else:
                        exchange_data = {
                            'Code': symbol_exchange_code,
                            'Name': symbol_exchange_code,
                            'Country': '',
                            'Timezone': 'UTC'
                        }
                    
                    exchange = SymbolService.get_or_create_exchange(exchange_data)
                    symbol = SymbolService.create_symbol_from_eod_data(symbol_data, exchange, None)
                    if symbol:
                        if symbol.created_at == symbol.updated_at:
                            created_count += 1
                        else:
                            updated_count += 1
                        if symbol_exchange_code not in exchange_stats:
                            exchange_stats[symbol_exchange_code] = {'created': 0, 'updated': 0}
                        if symbol.created_at == symbol.updated_at:
                            exchange_stats[symbol_exchange_code]['created'] += 1
                        else:
                            exchange_stats[symbol_exchange_code]['updated'] += 1
                    else:
                        error_count += 1
                
                results[exchange_code] = {
                    'exchange': exchange_code,
                    'total': len(symbols_data),
                    'created': created_count,
                    'updated': updated_count,
                    'errors': error_count,
                    'exchange_stats': exchange_stats
                }
            except Exception as e:
                results[exchange_code] = {
                    'exchange': exchange_code,
                    'total': 0,
                    'created': 0,
                    'updated': 0,
                    'errors': 1,
                    'error': str(e)
                }
        
        total_created = sum(r.get('created', 0) for r in results.values())
        total_updated = sum(r.get('updated', 0) for r in results.values())
        
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 100,
                'message': f'Completed: {total_created} created, {total_updated} updated'
            }
        )
        
        return {
            'progress': 100,
            'message': f'Successfully imported {total_created} new and {total_updated} updated symbols from {total_exchanges} exchanges',
            'status': 'completed',
            'result': results
        }
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching symbols: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching symbols: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def fetch_symbols_from_all_exchanges_task(self):
    """
    Background task to fetch and import symbols from all available exchanges
    Updates progress during processing
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': 'Fetching list of available exchanges...'
            }
        )
        
        exchanges = EODAPIProvider.get_exchanges_list()
        exchange_codes = [ex.get('Code', '') for ex in exchanges if ex.get('Code')]
        total_exchanges = len(exchange_codes)
        
        if total_exchanges == 0:
            return {
                'progress': 100,
                'message': 'No exchanges found',
                'status': 'completed',
                'result': {}
            }
        
        # Process exchanges inline to maintain progress updates
        results = {}
        for idx, exchange_code in enumerate(exchange_codes):
            # Calculate progress (reserve 5% for final processing)
            progress = int((idx / total_exchanges) * 95)
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processing {exchange_code} ({idx + 1}/{total_exchanges})'
                }
            )
            
            # Process this exchange inline (similar to multiple exchanges task)
            try:
                symbols_data = EODAPIProvider.get_exchange_symbols(exchange_code)
                exchanges_list = EODAPIProvider.get_exchanges_list()
                exchanges_map = {ex.get('Code'): ex for ex in exchanges_list if ex.get('Code')}
                
                created_count = 0
                updated_count = 0
                error_count = 0
                exchange_stats = {}
                
                for symbol_data in symbols_data:
                    symbol_exchange_code = symbol_data.get('Exchange', exchange_code)
                    if symbol_exchange_code in exchanges_map:
                        exchange_data = exchanges_map[symbol_exchange_code]
                    else:
                        exchange_data = {
                            'Code': symbol_exchange_code,
                            'Name': symbol_exchange_code,
                            'Country': '',
                            'Timezone': 'UTC'
                        }
                    
                    exchange = SymbolService.get_or_create_exchange(exchange_data)
                    symbol = SymbolService.create_symbol_from_eod_data(symbol_data, exchange, None)
                    if symbol:
                        if symbol.created_at == symbol.updated_at:
                            created_count += 1
                        else:
                            updated_count += 1
                        if symbol_exchange_code not in exchange_stats:
                            exchange_stats[symbol_exchange_code] = {'created': 0, 'updated': 0}
                        if symbol.created_at == symbol.updated_at:
                            exchange_stats[symbol_exchange_code]['created'] += 1
                        else:
                            exchange_stats[symbol_exchange_code]['updated'] += 1
                    else:
                        error_count += 1
                
                results[exchange_code] = {
                    'exchange': exchange_code,
                    'total': len(symbols_data),
                    'created': created_count,
                    'updated': updated_count,
                    'errors': error_count,
                    'exchange_stats': exchange_stats
                }
            except Exception as e:
                results[exchange_code] = {
                    'exchange': exchange_code,
                    'total': 0,
                    'created': 0,
                    'updated': 0,
                    'errors': 1,
                    'error': str(e)
                }
        
        total_created = sum(r.get('created', 0) for r in results.values())
        total_updated = sum(r.get('updated', 0) for r in results.values())
        
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 100,
                'message': f'Completed: {total_created} created, {total_updated} updated'
            }
        )
        
        return {
            'progress': 100,
            'message': f'Successfully imported {total_created} new and {total_updated} updated symbols from {total_exchanges} exchanges',
            'status': 'completed',
            'result': results
        }
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching symbols from all exchanges: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching symbols from all exchanges: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def fetch_ohlcv_data_task(
    self,
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    replace_existing: bool = False,
    broker_id: Optional[int] = None,
    provider_code: str = 'YAHOO'
):
    """
    Background task to fetch and save OHLCV data for a single symbol
    
    Args:
        ticker: Symbol ticker
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        period: Period string (e.g., '1mo', '1y', 'max')
        replace_existing: If True, replace existing data in the date range
        broker_id: Optional broker ID to filter symbols by broker linkage
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Starting OHLCV data fetch for {ticker}'
            }
        )
        
        # Get symbol
        try:
            symbol = Symbol.objects.get(ticker=ticker)
        except Symbol.DoesNotExist:
            return {
                'progress': 0,
                'message': f'Symbol {ticker} not found',
                'status': 'failed'
            }
        
        # If broker_id is provided, verify the symbol is linked to the broker
        if broker_id:
            from live_trading.models import SymbolBrokerAssociation
            try:
                association = SymbolBrokerAssociation.objects.get(
                    broker_id=broker_id,
                    symbol=symbol
                )
                # Check if at least one trading flag is active
                if not (association.long_active or association.short_active):
                    return {
                        'progress': 0,
                        'message': f'Symbol {ticker} is linked to broker {broker_id} but has no active trading flags',
                        'status': 'failed'
                    }
            except SymbolBrokerAssociation.DoesNotExist:
                return {
                    'progress': 0,
                    'message': f'Symbol {ticker} is not linked to broker {broker_id}',
                    'status': 'failed'
                }
        
        # Parse dates if provided and make timezone-aware
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if timezone.is_naive(start_dt):
                start_dt = timezone.make_aware(start_dt)
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if timezone.is_naive(end_dt):
                end_dt = timezone.make_aware(end_dt)
        
        # Check if date range is already fully covered (only for daily timeframe with date range)
        if not replace_existing and start_dt and end_dt and not period:
            is_covered, existing_dates = OHLCVService.check_date_range_coverage(
                symbol=symbol,
                start_date=start_dt,
                end_date=end_dt,
                timeframe='daily'
            )
            if is_covered:
                # Date range is already covered, skip API call
                connections.close_all()
                return {
                    'progress': 100,
                    'message': f'Date range already fully covered for {ticker}, skipping fetch',
                    'status': 'completed',
                    'result': {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'skipped': True}
                }
        
        # Get provider instance
        try:
            data_provider = ProviderFactory.get_provider(provider_code)
            provider_model = OHLCVService.get_provider(provider_code)
        except Exception as e:
            connections.close_all()
            return {
                'progress': 0,
                'message': f'Error getting provider {provider_code}: {str(e)}',
                'status': 'failed'
            }
        
        # Fetch data from provider
        provider_name = provider_model.name
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 20,
                'message': f'Fetching data from {provider_name} for {ticker}...'
            }
        )
        
        ohlcv_data = data_provider.get_historical_data(
            ticker=ticker,
            start_date=start_dt,
            end_date=end_dt,
            period=period,
            interval='1d'
        )
        
        if not ohlcv_data:
            connections.close_all()
            return {
                'progress': 100,
                'message': f'No data found for {ticker}',
                'status': 'completed',
                'result': {'created': 0, 'updated': 0, 'errors': 0, 'total': 0}
            }
        
        # Filter to only missing timestamps before saving
        # For period-based fetches, we still filter to only save missing data
        if not replace_existing:
            # Get timestamps from fetched data
            fetched_timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
            # Get missing timestamps
            missing_timestamps = OHLCVService.get_missing_timestamps(
                symbol=symbol,
                requested_timestamps=fetched_timestamps,
                timeframe='daily'
            )
            # Create set for quick lookup
            missing_timestamps_set = set()
            for ts in missing_timestamps:
                if isinstance(ts, datetime):
                    missing_timestamps_set.add(ts.date())
                else:
                    missing_timestamps_set.add(ts)
            
            # Filter ohlcv_data to only include missing timestamps
            filtered_ohlcv_data = []
            for data in ohlcv_data:
                ts = data.get('timestamp')
                if ts:
                    ts_date = ts.date() if isinstance(ts, datetime) else ts
                    if ts_date in missing_timestamps_set:
                        filtered_ohlcv_data.append(data)
            
            ohlcv_data = filtered_ohlcv_data
            
            # If all fetched data already exists, skip saving
            if not ohlcv_data:
                connections.close_all()
                return {
                    'progress': 100,
                    'message': f'All data already exists for {ticker}, skipping save',
                    'status': 'completed',
                    'result': {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'skipped': True}
                }
        
        # Save data
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 60,
                'message': f'Saving {len(ohlcv_data)} records for {ticker}...'
            }
        )
        
        result = OHLCVService.save_ohlcv_data(
            symbol=symbol,
            ohlcv_data=ohlcv_data,
            timeframe='daily',
            provider=provider_model,
            replace_existing=replace_existing
        )
        
        # Update symbol last_updated
        symbol.last_updated = timezone.now()
        symbol.save(update_fields=['last_updated'])
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully saved {result["created"]} new records for {ticker}',
            'status': 'completed',
            'result': result
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching OHLCV data for {ticker}: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching OHLCV data for {ticker}: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def fetch_ohlcv_data_multiple_symbols_task(
    self,
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    replace_existing: bool = False,
    broker_id: Optional[int] = None,
    provider_code: str = 'YAHOO'
):
    """
    Background task to fetch and save OHLCV data for multiple symbols
    
    Args:
        tickers: List of symbol tickers
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        period: Period string
        replace_existing: If True, replace existing data
    """
    try:
        total_symbols = len(tickers)
        results = {}
        
        # Process symbols in parallel using ThreadPoolExecutor
        # This significantly speeds up fetching when dealing with multiple symbols
        # Use 10 threads for optimal parallelism with Yahoo Finance API
        MAX_WORKERS = 10
        BATCH_SIZE = MAX_WORKERS  # Align batch size with thread count for maximum efficiency
        max_workers = min(MAX_WORKERS, len(tickers))  # Use up to 10 threads, or number of tickers if less
        processed_count = 0
        lock = threading.Lock()
        
        # Get provider instance once (thread-safe, but we'll get it per thread to be safe)
        # Actually, let's get it once and pass it, as providers are typically stateless
        try:
            data_provider = ProviderFactory.get_provider(provider_code)
            provider_model = OHLCVService.get_provider(provider_code)
        except Exception as e:
            return {
                'progress': 0,
                'message': f'Error getting provider {provider_code}: {str(e)}',
                'status': 'failed',
                'result': {}
            }
        
        # Optimize for Alpaca: use bulk fetching when available (up to 200 symbols per call)
        if provider_code.upper() == 'ALPACA' and len(tickers) > 1:
            start_dt = None
            end_dt = None
            if start_date:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if end_date:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            try:
                # Fetch all symbols with bulk API call (or multiple if > 200)
                bulk_data = data_provider.get_multiple_symbols_data(
                    tickers=tickers,
                    start_date=start_dt,
                    end_date=end_dt,
                    period=period,
                    interval='1d'
                )
                
                # Process results for each symbol
                for ticker in tickers:
                    try:
                        symbol = Symbol.objects.get(ticker=ticker)
                        
                        # If broker_id is provided, verify the symbol is linked to the broker
                        if broker_id:
                            from live_trading.models import SymbolBrokerAssociation
                            try:
                                association = SymbolBrokerAssociation.objects.get(
                                    broker_id=broker_id,
                                    symbol=symbol
                                )
                                if not (association.long_active or association.short_active):
                                    results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol is linked but has no active trading flags'}
                                    processed_count += 1
                                    continue
                            except SymbolBrokerAssociation.DoesNotExist:
                                results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol is not linked to broker'}
                                processed_count += 1
                                continue
                        
                        ohlcv_data = bulk_data.get(ticker, [])
                        
                        if not ohlcv_data:
                            results[ticker] = {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'}
                            processed_count += 1
                            continue
                        
                        # Filter to only missing timestamps before saving (if date range was provided)
                        if not replace_existing and start_dt and end_dt and not period:
                            fetched_timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
                            missing_timestamps = OHLCVService.get_missing_timestamps(
                                symbol=symbol,
                                requested_timestamps=fetched_timestamps,
                                timeframe='daily'
                            )
                            missing_timestamps_set = set()
                            for ts in missing_timestamps:
                                if isinstance(ts, datetime):
                                    missing_timestamps_set.add(ts.date())
                                else:
                                    missing_timestamps_set.add(ts)
                            
                            filtered_ohlcv_data = []
                            for data in ohlcv_data:
                                ts = data.get('timestamp')
                                if ts:
                                    ts_date = ts.date() if isinstance(ts, datetime) else ts
                                    if ts_date in missing_timestamps_set:
                                        filtered_ohlcv_data.append(data)
                            
                            ohlcv_data = filtered_ohlcv_data
                            
                            if not ohlcv_data:
                                results[ticker] = {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'All data already exists, skipped', 'skipped': True}
                                processed_count += 1
                                continue
                        
                        result = OHLCVService.save_ohlcv_data(
                            symbol=symbol,
                            ohlcv_data=ohlcv_data,
                            timeframe='daily',
                            provider=provider_model,
                            replace_existing=replace_existing
                        )
                        symbol.last_updated = timezone.now()
                        symbol.save(update_fields=['last_updated'])
                        
                        results[ticker] = result
                        processed_count += 1
                        
                        # Update progress
                        progress = int((processed_count / total_symbols) * 95)
                        self.update_state(
                            state='RUNNING',
                            meta={
                                'progress': progress,
                                'message': f'Processed {processed_count}/{total_symbols} symbols (bulk mode)'
                            }
                        )
                        
                    except Symbol.DoesNotExist:
                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'}
                        processed_count += 1
                    except Exception as e:
                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': str(e)[:200]}
                        processed_count += 1
                
                # Final update
                self.update_state(
                    state='RUNNING',
                    meta={
                        'progress': 100,
                        'message': f'Completed: {processed_count}/{total_symbols} symbols processed'
                    }
                )
                
                total_created = sum(r.get('created', 0) for r in results.values())
                total_updated = sum(r.get('updated', 0) for r in results.values())
                total_errors = sum(r.get('errors', 0) for r in results.values())
                
                connections.close_all()
                
                return {
                    'progress': 100,
                    'message': f'Successfully processed {total_symbols} symbols using bulk fetching',
                    'status': 'completed',
                    'result': results
                }
                
            except Exception as e:
                # If bulk fetch fails, fall back to individual processing
                print(f"Alpaca bulk fetch failed, falling back to individual processing: {e}")
                # Continue to individual processing below
        
        def process_single_symbol(ticker):
            """Process a single symbol and return the result"""
            nonlocal processed_count
            try:
                symbol = Symbol.objects.get(ticker=ticker)
                
                # If broker_id is provided, verify the symbol is linked to the broker
                if broker_id:
                    from live_trading.models import SymbolBrokerAssociation
                    try:
                        association = SymbolBrokerAssociation.objects.get(
                            broker_id=broker_id,
                            symbol=symbol
                        )
                        # Check if at least one trading flag is active
                        if not (association.long_active or association.short_active):
                            with lock:
                                processed_count += 1
                            return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol is linked but has no active trading flags'})
                    except SymbolBrokerAssociation.DoesNotExist:
                        with lock:
                            processed_count += 1
                        return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol is not linked to broker'})
                
                start_dt = None
                end_dt = None
                if start_date:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                # Check if date range is already fully covered (only for daily timeframe with date range)
                if not replace_existing and start_dt and end_dt and not period:
                    is_covered, existing_dates = OHLCVService.check_date_range_coverage(
                        symbol=symbol,
                        start_date=start_dt,
                        end_date=end_dt,
                        timeframe='daily'
                    )
                    if is_covered:
                        # Date range is already covered, skip API call
                        with lock:
                            processed_count += 1
                        return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'Date range already fully covered, skipped', 'skipped': True})
                
                # Fetch OHLCV data
                ohlcv_data = data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=start_dt,
                    end_date=end_dt,
                    period=period,
                    interval='1d'
                )
                
                if not ohlcv_data:
                    with lock:
                        processed_count += 1
                    return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'})
                
                # Filter to only missing timestamps before saving (if date range was provided)
                if not replace_existing and start_dt and end_dt and not period:
                    # Get timestamps from fetched data
                    fetched_timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
                    # Get missing timestamps
                    missing_timestamps = OHLCVService.get_missing_timestamps(
                        symbol=symbol,
                        requested_timestamps=fetched_timestamps,
                        timeframe='daily'
                    )
                    # Create set for quick lookup
                    missing_timestamps_set = set()
                    for ts in missing_timestamps:
                        if isinstance(ts, datetime):
                            missing_timestamps_set.add(ts.date())
                        else:
                            missing_timestamps_set.add(ts)
                    
                    # Filter ohlcv_data to only include missing timestamps
                    filtered_ohlcv_data = []
                    for data in ohlcv_data:
                        ts = data.get('timestamp')
                        if ts:
                            ts_date = ts.date() if isinstance(ts, datetime) else ts
                            if ts_date in missing_timestamps_set:
                                filtered_ohlcv_data.append(data)
                    
                    ohlcv_data = filtered_ohlcv_data
                    
                    # If all fetched data already exists, skip saving
                    if not ohlcv_data:
                        with lock:
                            processed_count += 1
                        return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'All data already exists, skipped', 'skipped': True})
                
                # Save OHLCV data
                result = OHLCVService.save_ohlcv_data(
                    symbol=symbol,
                    ohlcv_data=ohlcv_data,
                    timeframe='daily',
                    provider=provider_model,
                    replace_existing=replace_existing
                )
                symbol.last_updated = timezone.now()
                symbol.save(update_fields=['last_updated'])
                
                # Update progress
                with lock:
                    processed_count += 1
                    progress = int((processed_count / total_symbols) * 95)
                    self.update_state(
                        state='RUNNING',
                        meta={
                            'progress': progress,
                            'message': f'Processed {processed_count}/{total_symbols} symbols (parallel)'
                        }
                    )
                
                return (ticker, result)
                    
            except Symbol.DoesNotExist:
                with lock:
                    processed_count += 1
                return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'})
            except Exception as e:
                with lock:
                    processed_count += 1
                return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': str(e)})
        
        # Execute in parallel
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {executor.submit(process_single_symbol, ticker): ticker for ticker in tickers}
                
                # Collect results as they complete
                for future in as_completed(future_to_ticker):
                    try:
                        ticker, result = future.result()
                        results[ticker] = result
                    except Exception as e:
                        # Handle exceptions from worker threads gracefully
                        ticker = future_to_ticker[future]
                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'Thread error: {str(e)}'}
        finally:
            # Close database connections after all threads complete
            connections.close_all()
        
        total_created = sum(r.get('created', 0) for r in results.values())
        total_updated = sum(r.get('updated', 0) for r in results.values())
        
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 100,
                'message': f'Completed: {total_created} created, {total_updated} updated'
            }
        )
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully processed {total_symbols} symbols',
            'status': 'completed',
            'result': results
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching OHLCV data: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching OHLCV data: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True, time_limit=24 * 60 * 60, soft_time_limit=24 * 60 * 60)
def fetch_ohlcv_data_by_broker_task(
    self,
    broker_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    replace_existing: bool = False,
    provider_code: str = 'YAHOO'
):
    """
    Background task to fetch and save OHLCV data for all symbols linked to a broker
    
    Args:
        broker_id: Broker ID
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        period: Period string
        replace_existing: If True, replace existing data
    """
    import time
    from celery.exceptions import SoftTimeLimitExceeded
    
    try:
        from live_trading.models import SymbolBrokerAssociation
        from django.db.models import Q
        
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Getting symbols for broker {broker_id}...'
            }
        )
        
        # Get symbols linked to the broker with at least one active flag
        associations = SymbolBrokerAssociation.objects.filter(
            broker_id=broker_id
        ).filter(
            Q(long_active=True) | Q(short_active=True)
        ).select_related('symbol')
        
        symbols = [assoc.symbol for assoc in associations]
        total_symbols = len(symbols)
        
        if total_symbols == 0:
            return {
                'progress': 100,
                'message': f'No symbols found for broker {broker_id}',
                'status': 'completed',
                'result': {}
            }
        
        tickers = [symbol.ticker for symbol in symbols]
        
        # Determine batch size based on provider
        # For Alpaca: use smaller batches (50 symbols) for better reliability with date chunking
        #   - Smaller batches process faster and show progress more frequently
        #   - Reduces impact if one batch fails or times out
        #   - Still leverages bulk API efficiently (Alpaca supports up to 200 symbols per API call)
        # For other providers: use smaller batches with parallel processing (10 symbols = 10 threads)
        if provider_code.upper() == 'ALPACA':
            MAX_WORKERS = 10  # Still use some parallelism for post-processing
            # Alpaca has 1000 data points limit per API call
            # With date chunking (1 year = ~252 trading days), max 3 symbols per batch (3 * 252 = 756 < 1000)
            # This ensures all symbols are returned in the API response
            BATCH_SIZE = 3
        else:
            MAX_WORKERS = 10  # Thread pool size
            BATCH_SIZE = MAX_WORKERS  # Align batch size with thread count (10 symbols per batch = 10 threads)
        
        processed_count = 0
        results = {}
        
        for batch_start in range(0, total_symbols, BATCH_SIZE):
            # Wrap entire batch processing in try-except to ensure we continue even if batch fails
            try:
                batch_end = min(batch_start + BATCH_SIZE, total_symbols)
                batch_tickers = tickers[batch_start:batch_end]
                
                progress = int((batch_start / total_symbols) * 95)
                # Show batch info with provider-specific messaging
                batch_info = f'batch {batch_start + 1}-{batch_end}'
                if provider_code.upper() == 'ALPACA':
                    batch_info += f' (bulk mode: up to {BATCH_SIZE} symbols per API call)'
                self.update_state(
                    state='RUNNING',
                    meta={
                        'progress': progress,
                        'message': f'Processing {batch_info} of {total_symbols} symbols for broker {broker_id} (processed: {processed_count})'
                    }
                )
                
                # Get provider instance once per batch (more efficient)
                try:
                    data_provider = ProviderFactory.get_provider(provider_code)
                    provider_model = OHLCVService.get_provider(provider_code)
                except Exception as e:
                    # If provider fails, mark all tickers in batch as failed
                    print(f"Error getting provider for batch {batch_start}-{batch_end}: {e}")
                    for ticker in batch_tickers:
                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'Error getting provider {provider_code}: {str(e)}'}
                    processed_count += len(batch_tickers)
                    connections.close_all()
                    continue
                
                # Optimize for Alpaca: use bulk fetching when available (up to 200 symbols per call)
                if provider_code.upper() == 'ALPACA' and len(batch_tickers) > 1:
                    # Alpaca supports bulk fetching - use get_multiple_symbols_data for the entire batch
                    start_dt = None
                    end_dt = None
                    if start_date:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if end_date:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    
                    try:
                        # Fetch all symbols in batch with a single API call (or multiple if batch > 200)
                        bulk_data = data_provider.get_multiple_symbols_data(
                            tickers=batch_tickers,
                            start_date=start_dt,
                            end_date=end_dt,
                            period=period,
                            interval='1d'
                        )
                        
                        # Process results for each symbol in batch
                        for ticker in batch_tickers:
                            try:
                                symbol = Symbol.objects.get(ticker=ticker)
                                ohlcv_data = bulk_data.get(ticker, [])
                                
                                if not ohlcv_data:
                                    results[ticker] = {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'}
                                    processed_count += 1
                                    continue
                                
                                # Filter to only missing timestamps before saving (if date range was provided)
                                if not replace_existing and start_dt and end_dt and not period:
                                    fetched_timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
                                    missing_timestamps = OHLCVService.get_missing_timestamps(
                                        symbol=symbol,
                                        requested_timestamps=fetched_timestamps,
                                        timeframe='daily'
                                    )
                                    missing_timestamps_set = set()
                                    for ts in missing_timestamps:
                                        if isinstance(ts, datetime):
                                            missing_timestamps_set.add(ts.date())
                                        else:
                                            missing_timestamps_set.add(ts)
                                    
                                    filtered_ohlcv_data = []
                                    for data in ohlcv_data:
                                        ts = data.get('timestamp')
                                        if ts:
                                            ts_date = ts.date() if isinstance(ts, datetime) else ts
                                            if ts_date in missing_timestamps_set:
                                                filtered_ohlcv_data.append(data)
                                    
                                    ohlcv_data = filtered_ohlcv_data
                                    
                                    if not ohlcv_data:
                                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'All data already exists, skipped', 'skipped': True}
                                        processed_count += 1
                                        continue
                                
                                result = OHLCVService.save_ohlcv_data(
                                    symbol=symbol,
                                    ohlcv_data=ohlcv_data,
                                    timeframe='daily',
                                    provider=provider_model,
                                    replace_existing=replace_existing
                                )
                                symbol.last_updated = timezone.now()
                                symbol.save(update_fields=['last_updated'])
                                
                                results[ticker] = result
                                processed_count += 1
                                
                            except Symbol.DoesNotExist:
                                results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'}
                                processed_count += 1
                            except Exception as e:
                                results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': str(e)[:200]}
                                processed_count += 1
                        
                        connections.close_all()
                        continue
                        
                    except Exception as e:
                        # If bulk fetch fails, fall back to individual processing
                        print(f"Alpaca bulk fetch failed for batch, falling back to individual processing: {e}")
                        # Continue to individual processing below
                
                # Process batch in parallel using ThreadPoolExecutor (for non-Alpaca or fallback)
                # This aligns with the batch size (10 symbols) and uses 10 threads for optimal efficiency
                batch_max_workers = min(MAX_WORKERS, len(batch_tickers))
                batch_processed = 0
                batch_lock = threading.Lock()
                
                def process_batch_symbol(ticker):
                    """Process a single symbol in the batch"""
                    nonlocal batch_processed
                    try:
                        symbol = Symbol.objects.get(ticker=ticker)
                        
                        start_dt = None
                        end_dt = None
                        if start_date:
                            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        if end_date:
                            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        
                        # Check if date range is already fully covered (only for daily timeframe with date range)
                        if not replace_existing and start_dt and end_dt and not period:
                            is_covered, existing_dates = OHLCVService.check_date_range_coverage(
                                symbol=symbol,
                                start_date=start_dt,
                                end_date=end_dt,
                                timeframe='daily'
                            )
                            if is_covered:
                                # Date range is already covered, skip API call
                                with batch_lock:
                                    batch_processed += 1
                                return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'Date range already fully covered, skipped', 'skipped': True})
                        
                        # Add timeout handling for individual requests
                        try:
                            ohlcv_data = data_provider.get_historical_data(
                                ticker=ticker,
                                start_date=start_dt,
                                end_date=end_dt,
                                period=period,
                                interval='1d'
                            )
                        except Exception as api_error:
                            # Handle network/API errors gracefully
                            error_msg = str(api_error)
                            with batch_lock:
                                batch_processed += 1
                            if 'curl' in error_msg.lower() or 'timeout' in error_msg.lower():
                                return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'Network error: {error_msg[:100]}'})
                            else:
                                return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'API error: {error_msg[:100]}'})
                        
                        if not ohlcv_data:
                            with batch_lock:
                                batch_processed += 1
                            return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'})
                        
                        # Filter to only missing timestamps before saving (if date range was provided)
                        if not replace_existing and start_dt and end_dt and not period:
                            # Get timestamps from fetched data
                            fetched_timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
                            # Get missing timestamps
                            missing_timestamps = OHLCVService.get_missing_timestamps(
                                symbol=symbol,
                                requested_timestamps=fetched_timestamps,
                                timeframe='daily'
                            )
                            # Create set for quick lookup
                            missing_timestamps_set = set()
                            for ts in missing_timestamps:
                                if isinstance(ts, datetime):
                                    missing_timestamps_set.add(ts.date())
                                else:
                                    missing_timestamps_set.add(ts)
                            
                            # Filter ohlcv_data to only include missing timestamps
                            filtered_ohlcv_data = []
                            for data in ohlcv_data:
                                ts = data.get('timestamp')
                                if ts:
                                    ts_date = ts.date() if isinstance(ts, datetime) else ts
                                    if ts_date in missing_timestamps_set:
                                        filtered_ohlcv_data.append(data)
                            
                            ohlcv_data = filtered_ohlcv_data
                            
                            # If all fetched data already exists, skip saving
                            if not ohlcv_data:
                                with batch_lock:
                                    batch_processed += 1
                                return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'All data already exists, skipped', 'skipped': True})
                        
                        result = OHLCVService.save_ohlcv_data(
                            symbol=symbol,
                            ohlcv_data=ohlcv_data,
                            timeframe='daily',
                            provider=provider_model,
                            replace_existing=replace_existing
                        )
                        symbol.last_updated = timezone.now()
                        symbol.save(update_fields=['last_updated'])
                        
                        with batch_lock:
                            batch_processed += 1
                            processed_count += 1
                        
                        return (ticker, result)
                        
                    except Symbol.DoesNotExist:
                        with batch_lock:
                            batch_processed += 1
                            processed_count += 1
                        return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'})
                    except SoftTimeLimitExceeded:
                        # Re-raise to be handled at batch level
                        raise
                    except Exception as e:
                        error_msg = str(e)
                        with batch_lock:
                            batch_processed += 1
                            processed_count += 1
                        return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': error_msg[:200]})
                    finally:
                        connections.close_all()
                
                # Execute batch in parallel
                try:
                    with ThreadPoolExecutor(max_workers=batch_max_workers) as batch_executor:
                        batch_futures = {batch_executor.submit(process_batch_symbol, ticker): ticker for ticker in batch_tickers}
                        
                        # Collect batch results
                        for batch_future in as_completed(batch_futures):
                            try:
                                ticker, batch_result = batch_future.result()
                                results[ticker] = batch_result
                            except Exception as e:
                                # Handle exceptions from worker threads gracefully
                                ticker = batch_futures[batch_future]
                                results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'Thread error: {str(e)}'}
                except SoftTimeLimitExceeded:
                    # Gracefully handle soft time limit at batch level
                    connections.close_all()
                    total_created = sum(r.get('created', 0) for r in results.values())
                    total_updated = sum(r.get('updated', 0) for r in results.values())
                    total_errors = sum(r.get('errors', 0) for r in results.values())
                    return {
                        'progress': progress,
                        'message': f'Time limit reached. Processed {processed_count}/{total_symbols} symbols for broker {broker_id}. Created: {total_created}, Updated: {total_updated}, Errors: {total_errors}',
                        'status': 'partial',
                        'result': results,
                        'processed': processed_count,
                        'total': total_symbols
                    }
                
                # Close connections after each batch
                connections.close_all()
                
            except SoftTimeLimitExceeded:
                # Gracefully handle soft time limit at batch level
                connections.close_all()
                total_created = sum(r.get('created', 0) for r in results.values())
                total_updated = sum(r.get('updated', 0) for r in results.values())
                total_errors = sum(r.get('errors', 0) for r in results.values())
                return {
                    'progress': progress,
                    'message': f'Time limit reached. Processed {processed_count}/{total_symbols} symbols for broker {broker_id}. Created: {total_created}, Updated: {total_updated}, Errors: {total_errors}',
                    'status': 'partial',
                    'result': results,
                    'processed': processed_count,
                    'total': total_symbols
                }
            except Exception as batch_error:
                # Catch any other exceptions in batch processing and continue to next batch
                import traceback
                print(f"Error processing batch {batch_start}-{batch_end} for broker {broker_id}: {batch_error}")
                traceback.print_exc()
                # Mark any unprocessed tickers in this batch as failed
                for ticker in batch_tickers:
                    if ticker not in results:
                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'Batch processing error: {str(batch_error)[:200]}'}
                        processed_count += 1
                connections.close_all()
                continue  # Continue to next batch
        
        total_created = sum(r.get('created', 0) for r in results.values())
        total_updated = sum(r.get('updated', 0) for r in results.values())
        total_errors = sum(r.get('errors', 0) for r in results.values())
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully processed {total_symbols} symbols for broker {broker_id}. Created: {total_created}, Updated: {total_updated}, Errors: {total_errors}',
            'status': 'completed',
            'result': results
        }
        
    except SoftTimeLimitExceeded:
        connections.close_all()
        total_created = sum(r.get('created', 0) for r in results.values()) if 'results' in locals() else 0
        total_updated = sum(r.get('updated', 0) for r in results.values()) if 'results' in locals() else 0
        total_errors = sum(r.get('errors', 0) for r in results.values()) if 'results' in locals() else 0
        processed_count = processed_count if 'processed_count' in locals() else 0
        total_symbols = total_symbols if 'total_symbols' in locals() else 0
        return {
            'progress': 95,
            'message': f'Time limit exceeded. Processed {processed_count}/{total_symbols} symbols for broker {broker_id}. Created: {total_created}, Updated: {total_updated}, Errors: {total_errors}',
            'status': 'partial',
            'result': results if 'results' in locals() else {},
            'processed': processed_count,
            'total': total_symbols
        }
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching OHLCV data for broker {broker_id}: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching OHLCV data for broker {broker_id}: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def fetch_ohlcv_data_by_exchange_task(
    self,
    exchange_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    replace_existing: bool = False,
    broker_id: Optional[int] = None,
    provider_code: str = 'YAHOO'
):
    """
    Background task to fetch and save OHLCV data for all symbols in an exchange
    
    Args:
        exchange_code: Exchange code (e.g., 'NYSE', 'NASDAQ')
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        period: Period string
        replace_existing: If True, replace existing data
        broker_id: Optional broker ID to filter symbols by broker linkage
        provider_code: Provider code (e.g., 'YAHOO', 'POLYGON')
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Getting symbols for {exchange_code}...'
            }
        )
        
        # Get exchange and symbols
        try:
            exchange = Exchange.objects.get(code=exchange_code)
        except Exchange.DoesNotExist:
            return {
                'progress': 0,
                'message': f'Exchange {exchange_code} not found',
                'status': 'failed'
            }
        
        symbols = Symbol.objects.filter(exchange=exchange)
        
        # If broker_id is provided, filter symbols by broker linkage
        if broker_id:
            from live_trading.models import SymbolBrokerAssociation
            from django.db.models import Q
            # Get symbols linked to the broker with at least one active flag
            broker_symbols = SymbolBrokerAssociation.objects.filter(
                broker_id=broker_id,
                symbol__exchange=exchange
            ).filter(
                Q(long_active=True) | Q(short_active=True)
            ).values_list('symbol__ticker', flat=True)
            symbols = symbols.filter(ticker__in=broker_symbols)
        
        total_symbols = symbols.count()
        
        if total_symbols == 0:
            broker_msg = f' (filtered by broker {broker_id})' if broker_id else ''
            return {
                'progress': 100,
                'message': f'No symbols found for {exchange_code}{broker_msg}',
                'status': 'completed',
                'result': {}
            }
        
        tickers = list(symbols.values_list('ticker', flat=True))
        
        # Determine batch size based on provider
        # For Alpaca: use smaller batches (50 symbols) for better reliability with date chunking
        #   - Smaller batches process faster and show progress more frequently
        #   - Reduces impact if one batch fails or times out
        #   - Still leverages bulk API efficiently (Alpaca supports up to 200 symbols per API call)
        # For other providers: use smaller batches with parallel processing (10 symbols = 10 threads)
        if provider_code.upper() == 'ALPACA':
            MAX_WORKERS = 10  # Still use some parallelism for post-processing
            # Alpaca has 1000 data points limit per API call
            # With date chunking (1 year = ~252 trading days), max 3 symbols per batch (3 * 252 = 756 < 1000)
            # This ensures all symbols are returned in the API response
            BATCH_SIZE = 3
        else:
            MAX_WORKERS = 10  # Thread pool size
            BATCH_SIZE = MAX_WORKERS  # Align batch size with thread count (10 symbols per batch = 10 threads)
        
        results = {}
        
        for batch_start in range(0, total_symbols, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_symbols)
            batch_tickers = tickers[batch_start:batch_end]
            
            progress = int((batch_start / total_symbols) * 95)
            # Show batch info with provider-specific messaging
            batch_info = f'batch {batch_start + 1}-{batch_end}'
            if provider_code.upper() == 'ALPACA':
                batch_info += f' (bulk mode: up to {BATCH_SIZE} symbols per API call)'
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processing {batch_info} of {total_symbols} symbols for {exchange_code}'
                }
            )
            
            # Get provider instance once per batch (more efficient)
            try:
                data_provider = ProviderFactory.get_provider(provider_code)
                provider_model = OHLCVService.get_provider(provider_code)
            except Exception as e:
                # If provider fails, mark all tickers in batch as failed
                for ticker in batch_tickers:
                    results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': f'Error getting provider {provider_code}: {str(e)}'}
                connections.close_all()
                continue
            
            # Process batch in parallel using ThreadPoolExecutor
            # This aligns with the batch size (10 symbols) and uses 10 threads for optimal efficiency
            batch_max_workers = min(MAX_WORKERS, len(batch_tickers))
            
            def process_batch_symbol(ticker):
                """Process a single symbol in the batch"""
                try:
                    symbol = Symbol.objects.get(ticker=ticker)
                    
                    start_dt = None
                    end_dt = None
                    if start_date:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if end_date:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    
                    # Check if date range is already fully covered (only for daily timeframe with date range)
                    if not replace_existing and start_dt and end_dt and not period:
                        is_covered, existing_dates = OHLCVService.check_date_range_coverage(
                            symbol=symbol,
                            start_date=start_dt,
                            end_date=end_dt,
                            timeframe='daily'
                        )
                        if is_covered:
                            # Date range is already covered, skip API call
                            return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'Date range already fully covered, skipped', 'skipped': True})
                    
                    ohlcv_data = data_provider.get_historical_data(
                        ticker=ticker,
                        start_date=start_dt,
                        end_date=end_dt,
                        period=period,
                        interval='1d'
                    )
                    
                    if not ohlcv_data:
                        return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'})
                    
                    # Filter to only missing timestamps before saving (if date range was provided)
                    if not replace_existing and start_dt and end_dt and not period:
                        # Get timestamps from fetched data
                        fetched_timestamps = [d['timestamp'] for d in ohlcv_data if 'timestamp' in d]
                        # Get missing timestamps
                        missing_timestamps = OHLCVService.get_missing_timestamps(
                            symbol=symbol,
                            requested_timestamps=fetched_timestamps,
                            timeframe='daily'
                        )
                        # Create set for quick lookup
                        missing_timestamps_set = set()
                        for ts in missing_timestamps:
                            if isinstance(ts, datetime):
                                missing_timestamps_set.add(ts.date())
                            else:
                                missing_timestamps_set.add(ts)
                        
                        # Filter ohlcv_data to only include missing timestamps
                        filtered_ohlcv_data = []
                        for data in ohlcv_data:
                            ts = data.get('timestamp')
                            if ts:
                                ts_date = ts.date() if isinstance(ts, datetime) else ts
                                if ts_date in missing_timestamps_set:
                                    filtered_ohlcv_data.append(data)
                        
                        ohlcv_data = filtered_ohlcv_data
                        
                        # If all fetched data already exists, skip saving
                        if not ohlcv_data:
                            return (ticker, {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'All data already exists, skipped', 'skipped': True})
                    
                    result = OHLCVService.save_ohlcv_data(
                        symbol=symbol,
                        ohlcv_data=ohlcv_data,
                        timeframe='daily',
                        provider=provider_model,
                        replace_existing=replace_existing
                    )
                    symbol.last_updated = timezone.now()
                    symbol.save(update_fields=['last_updated'])
                    
                    return (ticker, result)
                except Symbol.DoesNotExist:
                    return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'})
                except Exception as e:
                    return (ticker, {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': str(e)})
                finally:
                    connections.close_all()
            
            # Execute batch in parallel
            with ThreadPoolExecutor(max_workers=batch_max_workers) as batch_executor:
                batch_futures = {batch_executor.submit(process_batch_symbol, ticker): ticker for ticker in batch_tickers}
                
                # Collect batch results
                for batch_future in as_completed(batch_futures):
                    ticker, batch_result = batch_future.result()
                    results[ticker] = batch_result
        
        total_created = sum(r.get('created', 0) for r in results.values())
        total_updated = sum(r.get('updated', 0) for r in results.values())
        
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 100,
                'message': f'Completed: {total_created} created, {total_updated} updated'
            }
        )
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully processed {total_symbols} symbols from {exchange_code}',
            'status': 'completed',
            'result': results
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error fetching OHLCV data for {exchange_code}: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error fetching OHLCV data for {exchange_code}: {str(e)}',
            'status': 'failed'
        }


@shared_task(bind=True)
def delete_ohlcv_data_task(self, ticker: str):
    """
    Delete all OHLCV data for a single symbol
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Starting deletion of OHLCV data for {ticker}'
            }
        )
        
        try:
            symbol = Symbol.objects.get(ticker=ticker)
        except Symbol.DoesNotExist:
            return {
                'progress': 100,
                'message': f'Symbol {ticker} not found',
                'status': 'failed',
                'deleted_count': 0
            }
        
        # Count before deletion
        count_before = OHLCV.objects.filter(symbol=symbol).count()
        
        # Delete all OHLCV data for this symbol
        deleted_count, _ = OHLCV.objects.filter(symbol=symbol).delete()
        
        # Clear provider and disable the symbol after OHLCV data deletion
        symbol.provider = None
        symbol.status = 'disabled'
        symbol.validation_status = 'invalid'
        symbol.validation_reason = 'OHLCV data has been deleted'
        symbol.save(update_fields=['provider', 'status', 'validation_status', 'validation_reason'])
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully deleted {deleted_count} OHLCV records for {ticker} and disabled the symbol',
            'status': 'completed',
            'deleted_count': deleted_count
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error deleting OHLCV data for {ticker}: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error deleting OHLCV data for {ticker}: {str(e)}',
            'status': 'failed',
            'deleted_count': 0
        }


@shared_task(bind=True)
def delete_ohlcv_data_multiple_symbols_task(self, tickers: List[str]):
    """
    Delete OHLCV data for multiple symbols
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Starting deletion of OHLCV data for {len(tickers)} symbols'
            }
        )
        
        total_tickers = len(tickers)
        total_deleted = 0
        results = {}
        
        for idx, ticker in enumerate(tickers):
            try:
                symbol = Symbol.objects.get(ticker=ticker)
                deleted_count, _ = OHLCV.objects.filter(symbol=symbol).delete()
                total_deleted += deleted_count
                
                # Clear provider and disable the symbol after OHLCV data deletion
                symbol.provider = None
                symbol.status = 'disabled'
                symbol.validation_status = 'invalid'
                symbol.validation_reason = 'OHLCV data has been deleted'
                symbol.save(update_fields=['provider', 'status', 'validation_status', 'validation_reason'])
                
                results[ticker] = {'deleted': deleted_count, 'status': 'success'}
            except Symbol.DoesNotExist:
                results[ticker] = {'deleted': 0, 'status': 'not_found'}
            except Exception as e:
                results[ticker] = {'deleted': 0, 'status': 'error', 'message': str(e)}
            
            progress = int(((idx + 1) / total_tickers) * 100)
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processed {idx + 1}/{total_tickers} symbols'
                }
            )
            
            # Close connections after each symbol to prevent locking
            if (idx + 1) % 10 == 0:
                connections.close_all()
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully deleted OHLCV data for {total_tickers} symbols. Total records deleted: {total_deleted}',
            'status': 'completed',
            'deleted_count': total_deleted,
            'results': results
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error deleting OHLCV data: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error deleting OHLCV data: {str(e)}',
            'status': 'failed',
            'deleted_count': 0
        }


@shared_task(bind=True)
def delete_ohlcv_data_by_exchange_task(self, exchange_code: str):
    """
    Delete OHLCV data for all symbols in an exchange
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': f'Starting deletion of OHLCV data for exchange {exchange_code}'
            }
        )
        
        try:
            exchange = Exchange.objects.get(code=exchange_code)
        except Exchange.DoesNotExist:
            return {
                'progress': 100,
                'message': f'Exchange {exchange_code} not found',
                'status': 'failed',
                'deleted_count': 0
            }
        
        symbols = Symbol.objects.filter(exchange=exchange)
        total_symbols = symbols.count()
        
        if total_symbols == 0:
            return {
                'progress': 100,
                'message': f'No symbols found for exchange {exchange_code}',
                'status': 'completed',
                'deleted_count': 0
            }
        
        # Delete OHLCV data for all symbols in this exchange
        # Use a subquery to get all OHLCV records for symbols in this exchange
        deleted_count, _ = OHLCV.objects.filter(symbol__exchange=exchange).delete()
        
        # Clear provider and disable all symbols in this exchange after OHLCV data deletion
        symbols = Symbol.objects.filter(exchange=exchange)
        symbols.update(
            provider=None,
            status='disabled',
            validation_status='invalid',
            validation_reason='OHLCV data has been deleted'
        )
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully deleted {deleted_count} OHLCV records for {total_symbols} symbols in exchange {exchange_code} and disabled all symbols',
            'status': 'completed',
            'deleted_count': deleted_count,
            'symbols_count': total_symbols
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error deleting OHLCV data for exchange {exchange_code}: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error deleting OHLCV data for exchange {exchange_code}: {str(e)}',
            'status': 'failed',
            'deleted_count': 0
        }


@shared_task(bind=True)
def delete_all_ohlcv_data_task(self):
    """
    Delete all OHLCV data for all symbols
    """
    try:
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 0,
                'message': 'Starting deletion of all OHLCV data'
            }
        )
        
        # Count before deletion
        count_before = OHLCV.objects.count()
        symbol_count = Symbol.objects.count()
        
        # Delete all OHLCV data
        deleted_count, _ = OHLCV.objects.all().delete()
        
        # Clear provider and disable all symbols after OHLCV data deletion
        Symbol.objects.all().update(
            provider=None,
            status='disabled',
            validation_status='invalid',
            validation_reason='OHLCV data has been deleted'
        )
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully deleted {deleted_count} OHLCV records for {symbol_count} symbols and disabled all symbols',
            'status': 'completed',
            'deleted_count': deleted_count,
            'symbols_count': symbol_count
        }
        
    except Exception as e:
        connections.close_all()
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'Error deleting all OHLCV data: {str(e)}'
            }
        )
        return {
            'progress': 0,
            'message': f'Error deleting all OHLCV data: {str(e)}',
            'status': 'failed',
            'deleted_count': 0
        }

