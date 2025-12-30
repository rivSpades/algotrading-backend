"""
Celery tasks for live trading execution
"""

from celery import shared_task
from django.utils import timezone
from django.shortcuts import get_object_or_404
from .models import LiveTradingDeployment, Broker, SymbolBrokerAssociation
from market_data.models import Symbol, Exchange
from .services.live_trading_executor import LiveTradingExecutor
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='live_trading.start_deployment')
def start_deployment_task(self, deployment_id):
    """
    Start a live trading deployment
    
    Args:
        deployment_id: ID of the LiveTradingDeployment instance
    """
    try:
        deployment = LiveTradingDeployment.objects.get(id=deployment_id)
        
        logger.info(f"Starting deployment {deployment_id}")
        
        executor = LiveTradingExecutor(deployment)
        success = executor.start()
        
        if success:
            logger.info(f"Deployment {deployment_id} started successfully")
            return {'status': 'success', 'deployment_id': deployment_id}
        else:
            logger.error(f"Failed to start deployment {deployment_id}")
            return {'status': 'failed', 'deployment_id': deployment_id}
            
    except Exception as e:
        logger.error(f"Error starting deployment {deployment_id}: {e}", exc_info=True)
        try:
            deployment = LiveTradingDeployment.objects.get(id=deployment_id)
            deployment.status = 'failed'
            deployment.error_message = str(e)
            deployment.save()
        except:
            pass
        return {'status': 'error', 'error': str(e), 'deployment_id': deployment_id}


@shared_task(bind=True, name='live_trading.process_market_data')
def process_market_data_task(self, deployment_id, symbol_ticker, market_data):
    """
    Process market data update for a deployment
    
    Args:
        deployment_id: ID of the LiveTradingDeployment instance
        symbol_ticker: Symbol ticker
        market_data: Dict with OHLCV data
    """
    try:
        from market_data.models import Symbol
        
        deployment = LiveTradingDeployment.objects.get(id=deployment_id)
        symbol = Symbol.objects.get(ticker=symbol_ticker)
        
        # Skip if deployment is not active
        if deployment.status not in ['active', 'evaluating']:
            logger.debug(f"Deployment {deployment_id} is not active (status: {deployment.status}), skipping")
            return {'status': 'skipped', 'reason': f'Deployment status is {deployment.status}'}
        
        executor = LiveTradingExecutor(deployment)
        result = executor.process_market_update(symbol, market_data)
        
        if result:
            logger.info(f"Processed market data for {symbol_ticker} in deployment {deployment_id}, trade executed")
            return {'status': 'success', 'trade_executed': True, 'trade': result}
        else:
            return {'status': 'success', 'trade_executed': False}
            
    except Exception as e:
        logger.error(f"Error processing market data for {symbol_ticker} in deployment {deployment_id}: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


@shared_task(bind=True, name='live_trading.stop_deployment')
def stop_deployment_task(self, deployment_id):
    """
    Stop a live trading deployment
    
    Args:
        deployment_id: ID of the LiveTradingDeployment instance
    """
    try:
        deployment = LiveTradingDeployment.objects.get(id=deployment_id)
        
        logger.info(f"Stopping deployment {deployment_id}")
        
        executor = LiveTradingExecutor(deployment)
        success = executor.stop()
        
        if success:
            logger.info(f"Deployment {deployment_id} stopped successfully")
            return {'status': 'success', 'deployment_id': deployment_id}
        else:
            logger.error(f"Failed to stop deployment {deployment_id}")
            return {'status': 'failed', 'deployment_id': deployment_id}
            
    except Exception as e:
        logger.error(f"Error stopping deployment {deployment_id}: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e), 'deployment_id': deployment_id}


@shared_task(bind=True, name='live_trading.update_positions')
def update_positions_task(self, deployment_id):
    """
    Update positions for a deployment
    
    Args:
        deployment_id: ID of the LiveTradingDeployment instance
    """
    try:
        deployment = LiveTradingDeployment.objects.get(id=deployment_id)
        
        executor = LiveTradingExecutor(deployment)
        positions = executor.update_positions()
        
        return {'status': 'success', 'positions': positions}
        
    except Exception as e:
        logger.error(f"Error updating positions for deployment {deployment_id}: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


@shared_task(bind=True, name='live_trading.periodic_evaluation_check')
def periodic_evaluation_check_task(self):
    """
    Periodic task to check evaluations for all paper trading deployments
    
    This should be scheduled to run periodically (e.g., every hour)
    """
    try:
        from .services.evaluation_service import EvaluationService
        
        deployments = LiveTradingDeployment.objects.filter(
            deployment_type='paper',
            status__in=['evaluating', 'active']
        )
        
        results = []
        for deployment in deployments:
            try:
                EvaluationService.check_and_update_evaluation(deployment)
                results.append({
                    'deployment_id': deployment.id,
                    'status': deployment.status,
                    'evaluation_passed': deployment.has_evaluation_passed()
                })
            except Exception as e:
                logger.error(f"Error checking evaluation for deployment {deployment.id}: {e}")
                results.append({
                    'deployment_id': deployment.id,
                    'status': 'error',
                    'error': str(e)
                })
        
        return {'status': 'success', 'checked': len(results), 'results': results}
        
    except Exception as e:
        logger.error(f"Error in periodic evaluation check: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


@shared_task(bind=True, name='live_trading.link_broker_symbols')
def link_broker_symbols_task(self, broker_id, symbol_tickers=None, exchange_code=None, link_all_available=False, verify_capabilities=True):
    """
    Link symbols to a broker asynchronously
    
    Args:
        broker_id: ID of the Broker instance
        symbol_tickers: List of symbol tickers to link (optional)
        exchange_code: Exchange code to link all symbols from (optional)
        link_all_available: If True, link all available broker symbols that exist in DB and have no broker association
        verify_capabilities: Whether to verify broker capabilities (long/short support) via API
    
    Returns:
        Dictionary with results: created, skipped, total, status
    """
    try:
        broker = Broker.objects.get(id=broker_id)
        
        symbols_to_link = []
        
        # Handle "link all available" option
        if link_all_available:
            from .adapters.factory import get_broker_adapter
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': 10,
                    'message': f'Getting tradable symbols from broker {broker.name}...'
                }
            )
            
            # Try paper trading first, then real money
            adapter = get_broker_adapter(broker, paper_trading=True)
            if not adapter and broker.has_real_money_credentials():
                adapter = get_broker_adapter(broker, paper_trading=False)
            
            if not adapter:
                return {
                    'status': 'error',
                    'error': 'Broker must have at least paper trading or real money credentials configured and active'
                }
            
            try:
                # Get all tradable symbols from broker WITH their capabilities in one API call
                # This is much faster than calling get_symbol_capabilities() for each symbol
                broker_symbols_data = adapter.get_all_symbols_with_capabilities()
                broker_symbols = list(broker_symbols_data.keys())
                
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': 30,
                        'message': f'Found {len(broker_symbols)} tradable symbols, filtering...'
                    }
                )
                
                # Get all symbols that exist in our database
                db_symbols = Symbol.objects.filter(ticker__in=broker_symbols)
                
                # Get all symbols that already have ANY broker association
                symbols_with_broker = SymbolBrokerAssociation.objects.values_list('symbol_id', flat=True).distinct()
                
                # Filter to only symbols that exist in DB and have NO broker association
                symbols_to_link = [s for s in db_symbols if s.ticker not in symbols_with_broker]
                
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': 50,
                        'message': f'Found {len(symbols_to_link)} symbols to link'
                    }
                )
                
                # Store broker_symbols_data for later use (we already have capabilities)
                
            except Exception as e:
                logger.error(f"Error getting tradable symbols from broker: {e}")
                return {
                    'status': 'error',
                    'error': f'Failed to get tradable symbols from broker: {str(e)}'
                }
        else:
            # For non-bulk linking modes (individual, list, exchange)
            broker_symbols_data = None
            symbols_to_link = []
            
            # Get symbols by tickers
            if symbol_tickers:
                symbols = Symbol.objects.filter(ticker__in=symbol_tickers)
                symbols_to_link.extend(symbols)
            
            # Get symbols by exchange
            if exchange_code:
                exchange = get_object_or_404(Exchange, code=exchange_code)
                exchange_symbols = Symbol.objects.filter(exchange=exchange)
                symbols_to_link.extend(exchange_symbols)
            
            # Remove duplicates
            symbols_to_link = list(set(symbols_to_link))
            
            if not symbols_to_link:
                return {
                    'status': 'error',
                    'error': 'No symbols found to link'
                }
            
            # Filter out symbols already linked to this broker
            existing_associations = SymbolBrokerAssociation.objects.filter(
                broker=broker,
                symbol__in=symbols_to_link
            ).values_list('symbol_id', flat=True)
            
            existing_tickers = set(existing_associations)
            symbols_to_link = [s for s in symbols_to_link if s.ticker not in existing_tickers]
        
        if not symbols_to_link:
            return {
                'status': 'success',
                'message': 'No symbols to link (all already linked)',
                'created': 0,
                'skipped': 0,
                'total': 0
            }
        
        total_symbols = len(symbols_to_link)
        created_count = 0
        failed_count = 0
        
        # Get adapter and capabilities data for verification if needed
        # If we're in "link_all_available" mode, broker_symbols_data is already set above
        # Otherwise, try to get bulk capabilities if adapter supports it
        if verify_capabilities and not link_all_available:
            from .adapters.factory import get_broker_adapter
            adapter = get_broker_adapter(broker, paper_trading=True)
            if not adapter and broker.has_real_money_credentials():
                adapter = get_broker_adapter(broker, paper_trading=False)
            
            # Try to get bulk capabilities if adapter supports it (much faster)
            if adapter and hasattr(adapter, 'get_all_symbols_with_capabilities'):
                try:
                    broker_symbols_data = adapter.get_all_symbols_with_capabilities()
                except Exception as e:
                    logger.warning(f"Could not get bulk capabilities: {e}")
                    broker_symbols_data = None
        elif not verify_capabilities:
            adapter = None
            broker_symbols_data = None
        
        # Process symbols with bulk database operations
        associations_to_create = []
        
        for index, symbol in enumerate(symbols_to_link):
            try:
                # Determine capabilities from cached data if available (FAST - no API call)
                long_active = False
                short_active = False
                
                if verify_capabilities:
                    if broker_symbols_data and symbol.ticker in broker_symbols_data:
                        # Use cached capabilities data (FAST - no API call)
                        capabilities = broker_symbols_data[symbol.ticker]
                        long_active = capabilities.get('long_supported', False)
                        short_active = capabilities.get('short_supported', False)
                    elif adapter:
                        # Fallback: individual API call (SLOW - only if bulk method not available)
                        try:
                            capabilities = adapter.get_symbol_capabilities(symbol.ticker)
                            long_active = capabilities.get('long_supported', False)
                            short_active = capabilities.get('short_supported', False)
                        except Exception as e:
                            logger.error(f"Error verifying capabilities for {symbol.ticker}: {e}")
                
                # Prepare association for bulk create
                associations_to_create.append(
                    SymbolBrokerAssociation(
                        symbol=symbol,
                        broker=broker,
                        long_active=long_active,
                        short_active=short_active,
                        verified_at=timezone.now() if verify_capabilities and (long_active or short_active) else None,
                    )
                )
                
                # Update progress every 100 symbols or at the end
                if (index + 1) % 100 == 0 or (index + 1) == total_symbols:
                    progress = 50 + int((index + 1) / total_symbols * 50)
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'progress': progress,
                            'message': f'Prepared {index + 1}/{total_symbols} symbols for linking...'
                        }
                    )
            except Exception as e:
                logger.error(f"Error preparing symbol {symbol.ticker}: {e}")
                failed_count += 1
        
        # Bulk create all associations (MUCH faster than individual creates)
        if associations_to_create:
            try:
                # Use bulk_create for better performance
                SymbolBrokerAssociation.objects.bulk_create(
                    associations_to_create,
                    batch_size=500,  # Process in batches of 500
                    ignore_conflicts=False
                )
                created_count = len(associations_to_create)
                logger.info(f"Bulk created {created_count} symbol-broker associations")
            except Exception as e:
                logger.error(f"Error bulk creating associations: {e}")
                # Fallback to individual creates if bulk fails
                for assoc in associations_to_create:
                    try:
                        assoc.save()
                        created_count += 1
                    except Exception as e2:
                        logger.error(f"Error creating association for {assoc.symbol.ticker}: {e2}")
                        failed_count += 1
        
        return {
            'status': 'success',
            'message': f'Processed {total_symbols} symbols',
            'created': created_count,
            'failed': failed_count,
            'total': total_symbols,
            'broker_id': broker_id,
            'broker_name': broker.name
        }
        
    except Broker.DoesNotExist:
        return {
            'status': 'error',
            'error': f'Broker with id {broker_id} not found'
        }
    except Exception as e:
        logger.error(f"Error in link_broker_symbols_task: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


