"""
Celery tasks for market data operations
"""

from celery import shared_task
from django.utils import timezone
from django.db import connections, transaction
from typing import List, Optional
from datetime import datetime
from .models import Symbol, OHLCV, Exchange
from .services.symbol_service import SymbolService
from .services.ohlcv_service import OHLCVService
from .providers.eod_api import EODAPIProvider
from .providers.yahoo_finance import YahooFinanceProvider


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
    replace_existing: bool = False
):
    """
    Background task to fetch and save OHLCV data for a single symbol
    
    Args:
        ticker: Symbol ticker
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        period: Period string (e.g., '1mo', '1y', 'max')
        replace_existing: If True, replace existing data in the date range
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
        
        # Parse dates if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Fetch data from Yahoo Finance
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 20,
                'message': f'Fetching data from Yahoo Finance for {ticker}...'
            }
        )
        
        ohlcv_data = YahooFinanceProvider.get_historical_data(
            ticker=ticker,
            start_date=start_dt,
            end_date=end_dt,
            period=period,
            interval='1d'
        )
        
        if not ohlcv_data:
            return {
                'progress': 100,
                'message': f'No data found for {ticker}',
                'status': 'completed',
                'result': {'created': 0, 'updated': 0, 'errors': 0, 'total': 0}
            }
        
        # Save data
        self.update_state(
            state='RUNNING',
            meta={
                'progress': 60,
                'message': f'Saving {len(ohlcv_data)} records for {ticker}...'
            }
        )
        
        provider = OHLCVService.get_or_create_yahoo_provider()
        result = OHLCVService.save_ohlcv_data(
            symbol=symbol,
            ohlcv_data=ohlcv_data,
            timeframe='daily',
            provider=provider,
            replace_existing=replace_existing
        )
        
        # Update symbol last_updated
        symbol.last_updated = timezone.now()
        symbol.save(update_fields=['last_updated'])
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully saved {result["created"]} new and {result["updated"]} updated records for {ticker}',
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
    replace_existing: bool = False
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
        
        for idx, ticker in enumerate(tickers):
            progress = int((idx / total_symbols) * 95)
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processing {ticker} ({idx + 1}/{total_symbols})'
                }
            )
            
            # Call single symbol task logic inline
            try:
                symbol = Symbol.objects.get(ticker=ticker)
                
                start_dt = None
                end_dt = None
                if start_date:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                ohlcv_data = YahooFinanceProvider.get_historical_data(
                    ticker=ticker,
                    start_date=start_dt,
                    end_date=end_dt,
                    period=period,
                    interval='1d'
                )
                
                if ohlcv_data:
                    provider = OHLCVService.get_or_create_yahoo_provider()
                    result = OHLCVService.save_ohlcv_data(
                        symbol=symbol,
                        ohlcv_data=ohlcv_data,
                        timeframe='daily',
                        provider=provider,
                        replace_existing=replace_existing
                    )
                    symbol.last_updated = timezone.now()
                    symbol.save(update_fields=['last_updated'])
                    results[ticker] = result
                else:
                    results[ticker] = {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'}
                    
            except Symbol.DoesNotExist:
                results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'}
            except Exception as e:
                results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': str(e)}
            
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


@shared_task(bind=True)
def fetch_ohlcv_data_by_exchange_task(
    self,
    exchange_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    replace_existing: bool = False
):
    """
    Background task to fetch and save OHLCV data for all symbols in an exchange
    
    Args:
        exchange_code: Exchange code (e.g., 'NYSE', 'NASDAQ')
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        period: Period string
        replace_existing: If True, replace existing data
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
        total_symbols = symbols.count()
        
        if total_symbols == 0:
            return {
                'progress': 100,
                'message': f'No symbols found for {exchange_code}',
                'status': 'completed',
                'result': {}
            }
        
        tickers = list(symbols.values_list('ticker', flat=True))
        
        # Process symbols in batches
        results = {}
        BATCH_SIZE = 10  # Process 10 symbols at a time to avoid overwhelming the API
        
        for batch_start in range(0, total_symbols, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_symbols)
            batch_tickers = tickers[batch_start:batch_end]
            
            progress = int((batch_start / total_symbols) * 95)
            self.update_state(
                state='RUNNING',
                meta={
                    'progress': progress,
                    'message': f'Processing batch {batch_start + 1}-{batch_end}/{total_symbols} for {exchange_code}'
                }
            )
            
            # Process batch
            for ticker in batch_tickers:
                try:
                    symbol = Symbol.objects.get(ticker=ticker)
                    
                    start_dt = None
                    end_dt = None
                    if start_date:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if end_date:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    
                    ohlcv_data = YahooFinanceProvider.get_historical_data(
                        ticker=ticker,
                        start_date=start_dt,
                        end_date=end_dt,
                        period=period,
                        interval='1d'
                    )
                    
                    if ohlcv_data:
                        provider = OHLCVService.get_or_create_yahoo_provider()
                        result = OHLCVService.save_ohlcv_data(
                            symbol=symbol,
                            ohlcv_data=ohlcv_data,
                            timeframe='daily',
                            provider=provider,
                            replace_existing=replace_existing
                        )
                        symbol.last_updated = timezone.now()
                        symbol.save(update_fields=['last_updated'])
                        results[ticker] = result
                    else:
                        results[ticker] = {'created': 0, 'updated': 0, 'errors': 0, 'total': 0, 'message': 'No data found'}
                        
                except Symbol.DoesNotExist:
                    results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': 'Symbol not found'}
                except Exception as e:
                    results[ticker] = {'created': 0, 'updated': 0, 'errors': 1, 'total': 0, 'message': str(e)}
            
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
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully deleted {deleted_count} OHLCV records for {ticker}',
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
        
        connections.close_all()
        
        return {
            'progress': 100,
            'message': f'Successfully deleted {deleted_count} OHLCV records for {total_symbols} symbols in exchange {exchange_code}',
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

