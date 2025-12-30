"""
API Views for Market Data
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from django.utils import timezone
from .models import Symbol, Exchange, Provider, OHLCV
from .serializers import (
    SymbolSerializer, SymbolListSerializer, ExchangeSerializer,
    ProviderSerializer, OHLCVSerializer, PeriodicTaskSerializer,
    CrontabScheduleSerializer, IntervalScheduleSerializer
)
from django_celery_beat.models import PeriodicTask, CrontabSchedule, IntervalSchedule
import json
from .tasks import (
    update_symbol_data_task,
    fetch_symbols_from_exchange_task,
    fetch_symbols_from_multiple_exchanges_task,
    fetch_symbols_from_all_exchanges_task
)
from .providers.eod_api import EODAPIProvider
from algo_trading_backend.celery import app as celery_app
from celery.result import AsyncResult
from celery import states
from datetime import datetime, timedelta
import redis
from django.conf import settings


class ExchangeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Exchange model"""
    queryset = Exchange.objects.all()
    serializer_class = ExchangeSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'code']
    ordering_fields = ['name', 'code']
    ordering = ['name']


class ProviderViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Provider model"""
    queryset = Provider.objects.filter(is_active=True)
    serializer_class = ProviderSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'code']
    ordering_fields = ['name', 'code']
    ordering = ['name']


class SymbolViewSet(viewsets.ModelViewSet):
    """ViewSet for Symbol model"""
    queryset = Symbol.objects.select_related('exchange', 'provider').all()
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['ticker', 'name']
    ordering_fields = ['ticker', 'last_updated', 'created_at']
    ordering = ['ticker']

    def get_queryset(self):
        """Filter by exchange and status if provided"""
        queryset = super().get_queryset()
        
        # Filter by exchange code
        exchange_code = self.request.query_params.get('exchange', None)
        if exchange_code:
            queryset = queryset.filter(exchange__code=exchange_code)
        
        # Filter by status
        status = self.request.query_params.get('status', None)
        if status:
            queryset = queryset.filter(status=status)
        
        return queryset

    def get_serializer_class(self):
        if self.action == 'list':
            return SymbolListSerializer
        return SymbolSerializer

    @action(detail=False, methods=['get'], url_path='random', url_name='random')
    def random(self, request):
        """Get a random selection of symbols
        
        Query parameters:
        - count: Number of symbols to return (required)
        - status: Filter by status (default: 'active')
        - exchange: Filter by exchange code (optional)
        - broker_id: Filter by broker ID - only return symbols linked to this broker (optional)
        """
        count = request.query_params.get('count')
        if not count:
            return Response({
                'error': 'count parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            count = int(count)
            if count <= 0:
                return Response({
                    'error': 'count must be a positive integer'
                }, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            return Response({
                'error': 'count must be a valid integer'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Build queryset with filters
        queryset = Symbol.objects.select_related('exchange', 'provider').all()
        
        # Filter by status (default to 'active')
        status_filter = request.query_params.get('status', 'active')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by exchange code if provided
        exchange_code = request.query_params.get('exchange', None)
        if exchange_code:
            queryset = queryset.filter(exchange__code=exchange_code)
        
        # Filter by broker if provided (only symbols linked to the broker with at least one active flag)
        broker_id = request.query_params.get('broker_id', None)
        if broker_id:
            try:
                from live_trading.models import SymbolBrokerAssociation
                from django.db.models import Q
                
                # Get symbols linked to this broker with at least one trading capability (long_active or short_active)
                # Filter by status first on the association's symbol relationship
                associations = SymbolBrokerAssociation.objects.filter(
                    broker_id=broker_id,
                    symbol__status=status_filter if status_filter else 'active'
                ).filter(
                    Q(long_active=True) | Q(short_active=True)
                ).select_related('symbol')
                
                # Get symbol tickers from associations
                symbol_tickers = list(associations.values_list('symbol__ticker', flat=True).distinct())
                
                # Filter queryset to only include symbols from associations by ticker
                if symbol_tickers:
                    queryset = queryset.filter(ticker__in=symbol_tickers)
                else:
                    # No symbols match, return empty queryset
                    queryset = queryset.none()
            except Exception as e:
                return Response({
                    'error': f'Invalid broker_id: {str(e)}'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get total count for validation
        total_count = queryset.count()
        
        if total_count == 0:
            return Response({
                'results': [],
                'count': 0,
                'total_available': 0
            })
        
        # If requested count exceeds available symbols, return all
        if count > total_count:
            count = total_count
        
        # Get random symbols using order_by('?') - Django's random ordering
        random_symbols = list(queryset.order_by('?')[:count])
        
        # Serialize results
        serializer = SymbolListSerializer(random_symbols, many=True)
        
        return Response({
            'results': serializer.data,
            'count': len(random_symbols),
            'total_available': total_count
        })

    @action(detail=True, methods=['get'])
    def ohlcv(self, request, pk=None):
        """Get OHLCV data for a symbol with pagination and on-the-fly indicator computation"""
        symbol = self.get_object()
        timeframe = request.query_params.get('timeframe', 'daily')
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 50))

        # Get all OHLCV data for indicator computation (need full dataset for accurate indicators)
        full_queryset = OHLCV.objects.filter(symbol=symbol, timeframe=timeframe)
        
        if start_date:
            full_queryset = full_queryset.filter(timestamp__gte=start_date)
        if end_date:
            full_queryset = full_queryset.filter(timestamp__lte=end_date)

        # Get paginated queryset for response
        total_count = full_queryset.count()
        
        # Check if this is a chart view (backtest_id or strategy_id provided)
        # For chart views, return ALL data ordered by timestamp ascending (oldest to newest), no pagination
        # For table views, use pagination with newest first
        strategy_id = request.query_params.get('strategy_id')
        backtest_id = request.query_params.get('backtest_id')
        is_chart_view = bool(backtest_id or strategy_id)
        
        if is_chart_view:
            # Chart view: return ALL data, ordered by timestamp ascending (oldest to newest)
            # No pagination - return all OHLCV data for the chart
            paginated_queryset = full_queryset.order_by('timestamp')
            serializer = OHLCVSerializer(paginated_queryset, many=True)
            results = serializer.data
        else:
            # Table view: use pagination with newest first
            paginated_queryset = full_queryset.order_by('-timestamp')
            # Pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_queryset = paginated_queryset[start_idx:end_idx]
            serializer = OHLCVSerializer(paginated_queryset, many=True)
            results = serializer.data

        # Compute indicators on-the-fly using full dataset
        # Convert full queryset to list for pandas (need full dataset for accurate indicators)
        full_queryset_ordered = full_queryset.order_by('timestamp')
        full_ohlcv_list = []
        for ohlcv in full_queryset_ordered.values('timestamp', 'open', 'high', 'low', 'close', 'volume'):
            # Convert timestamp to ISO format string if it's a datetime object
            timestamp = ohlcv['timestamp']
            if hasattr(timestamp, 'isoformat'):
                timestamp = timestamp.isoformat()
            elif isinstance(timestamp, str):
                timestamp = timestamp
            else:
                timestamp = str(timestamp)
            
            full_ohlcv_list.append({
                'timestamp': timestamp,
                'open': float(ohlcv['open']),
                'high': float(ohlcv['high']),
                'low': float(ohlcv['low']),
                'close': float(ohlcv['close']),
                'volume': float(ohlcv['volume'])
            })
        
        # If strategy_id or backtest_id is provided, ONLY compute strategy's required indicators
        # Otherwise, compute all enabled indicators
        # (strategy_id and backtest_id already retrieved above for is_chart_view)
        
        if strategy_id or backtest_id:
            # For strategy/backtest views: ONLY compute strategy indicators (not all enabled indicators)
            full_indicator_values = {}
        else:
            # For regular views: compute all enabled indicators
            from .services.indicator_service import compute_indicators_for_ohlcv
            full_indicator_values = compute_indicators_for_ohlcv(symbol, full_ohlcv_list)
        
        if strategy_id or backtest_id:
            try:
                from strategies.models import StrategyDefinition
                from backtest_engine.models import Backtest
                strategy = None
                
                # Prefer backtest_id over strategy_id (backtest has the actual strategy instance)
                if backtest_id:
                    try:
                        backtest_id_int = int(backtest_id)
                        backtest = Backtest.objects.get(id=backtest_id_int)
                        strategy = backtest.strategy
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info(f"Found strategy from backtest {backtest_id_int}: {strategy.name} (id={strategy.id})")
                    except (Backtest.DoesNotExist, ValueError, TypeError) as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Could not find backtest {backtest_id}: {str(e)}")
                        pass
                
                # If no strategy from backtest, try strategy_id
                if not strategy and strategy_id:
                    try:
                        strategy_id_int = int(strategy_id)
                        strategy = StrategyDefinition.objects.get(id=strategy_id_int)
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info(f"Found strategy from strategy_id {strategy_id_int}: {strategy.name}")
                    except (StrategyDefinition.DoesNotExist, ValueError, TypeError) as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Could not find strategy {strategy_id}: {str(e)}")
                        pass
                
                if strategy and strategy.required_tool_configs:
                    # Compute strategy's required indicators
                    from .services.indicator_service import compute_strategy_indicators_for_ohlcv
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Computing strategy indicators for strategy {strategy.name} (id={strategy.id}) with {len(strategy.required_tool_configs)} tool configs")
                    
                    # Get strategy parameters from backtest if available, otherwise use strategy defaults
                    strategy_parameters = None
                    if backtest_id:
                        try:
                            backtest_id_int = int(backtest_id)
                            backtest = Backtest.objects.get(id=backtest_id_int)
                            strategy_parameters = backtest.strategy_parameters
                            logger.info(f"Using backtest strategy_parameters: {strategy_parameters}")
                        except (Backtest.DoesNotExist, ValueError, TypeError):
                            pass
                    
                    strategy_indicator_values = compute_strategy_indicators_for_ohlcv(
                        strategy, full_ohlcv_list, symbol, strategy_parameters=strategy_parameters
                    )
                    # Log computed strategy indicators
                    logger.info(f"Computed {len(strategy_indicator_values)} strategy indicators: {list(strategy_indicator_values.keys())}")
                    # Merge strategy indicators with existing indicators (strategy indicators take precedence)
                    full_indicator_values.update(strategy_indicator_values)
                    logger.info(f"Total indicators after merge: {len(full_indicator_values)} keys: {list(full_indicator_values.keys())[:10]}")
            except Exception as e:
                # Log error but don't break the OHLCV endpoint
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error computing strategy indicators: {str(e)}")
                # Continue without strategy indicators - regular indicators will still work
        
        # Create timestamp to index mapping for paginated results
        # Build a mapping from serialized timestamp strings to indices in full_ohlcv_list
        # full_ohlcv_list is ordered by timestamp (ascending), same as indicator computation
        from datetime import datetime as dt
        timestamp_to_index = {}
        for idx, ohlcv_item in enumerate(full_ohlcv_list):
            ts = ohlcv_item['timestamp']
            # Normalize timestamp to ISO string format for matching
            # DRF serializes DateTimeField as ISO string, so we need to match that format
            if isinstance(ts, str):
                # Already a string, use as-is
                timestamp_to_index[ts] = idx
            else:
                # Datetime object, convert to ISO string (DRF format)
                iso_str = ts.isoformat()
                timestamp_to_index[iso_str] = idx
                # Also store with 'Z' suffix (common format)
                timestamp_to_index[iso_str.replace('+00:00', 'Z')] = idx
                # Store original string representation
                timestamp_to_index[str(ts)] = idx
        
        # Add indicator values to paginated results
        indicators_metadata = {}
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Adding {len(full_indicator_values)} indicators to {len(results)} paginated results. Indicator keys: {list(full_indicator_values.keys())[:20]}")
        
        for indicator_key, indicator_data in full_indicator_values.items():
            full_values = indicator_data['values']
            
            # Skip if no values
            if not full_values:
                logger.warning(f"Skipping indicator {indicator_key} - no values")
                continue
            
            logger.debug(f"Processing indicator {indicator_key} with {len(full_values)} values")
            
            # Extract values for paginated results by matching timestamps
            for result in results:
                result_timestamp = result['timestamp']
                # Try to find matching index - result_timestamp is already serialized by DRF
                full_index = timestamp_to_index.get(result_timestamp)
                
                # If not found, try normalizing the timestamp string
                if full_index is None and isinstance(result_timestamp, str):
                    # Try with 'Z' replaced by '+00:00'
                    normalized = result_timestamp.replace('Z', '+00:00')
                    full_index = timestamp_to_index.get(normalized)
                    # Try parsing and converting back to ISO
                    if full_index is None:
                        try:
                            parsed_ts = dt.fromisoformat(normalized)
                            full_index = timestamp_to_index.get(parsed_ts.isoformat())
                        except:
                            pass
                
                # Assign indicator value if index found
                if full_index is not None and full_index < len(full_values):
                    result[indicator_key] = full_values[full_index]
                else:
                    result[indicator_key] = None
            
            # Store metadata
            indicators_metadata[indicator_key] = {
                'display_name': indicator_data.get('display_name', indicator_key),
                'color': indicator_data['color'],
                'line_width': indicator_data['line_width'],
                'subchart': indicator_data.get('subchart', False)
            }
            
            # Log that we added this indicator
            if 'SMA' in indicator_key:
                logger.info(f"Added SMA indicator {indicator_key} to results. Sample values: {[v for v in full_values[:5] if v is not None]}")

        # Calculate statistics (volatility, etc.) from full dataset
        from analytical_tools.statistics import calculate_statistics, get_benchmark_ticker
        
        # Get benchmark data for beta calculation
        benchmark_ohlcv_data = None
        benchmark_ticker = get_benchmark_ticker(symbol.exchange.code)
        if benchmark_ticker:
            try:
                # Get benchmark symbol
                benchmark_symbol = Symbol.objects.filter(ticker=benchmark_ticker).first()
                if benchmark_symbol:
                    # Get overlapping date range between stock and benchmark
                    # This ensures we only calculate beta for dates where both have data
                    stock_dates = full_queryset.values_list('timestamp', flat=True).order_by('timestamp')
                    benchmark_dates = OHLCV.objects.filter(
                        symbol=benchmark_symbol,
                        timeframe=timeframe
                    ).values_list('timestamp', flat=True).order_by('timestamp')
                    
                    if stock_dates.exists() and benchmark_dates.exists():
                        # Find overlapping date range
                        stock_min = stock_dates.first()
                        stock_max = stock_dates.last()
                        benchmark_min = benchmark_dates.first()
                        benchmark_max = benchmark_dates.last()
                        
                        overlap_start = max(stock_min, benchmark_min)
                        overlap_end = min(stock_max, benchmark_max)
                        
                        # Get benchmark OHLCV data for the overlapping date range
                        benchmark_queryset = OHLCV.objects.filter(
                            symbol=benchmark_symbol,
                            timeframe=timeframe,
                            timestamp__gte=overlap_start,
                            timestamp__lte=overlap_end
                        )
                        
                        benchmark_queryset_ordered = benchmark_queryset.order_by('timestamp')
                        benchmark_ohlcv_list = []
                        for ohlcv in benchmark_queryset_ordered.values('timestamp', 'open', 'high', 'low', 'close', 'volume'):
                            timestamp = ohlcv['timestamp']
                            if hasattr(timestamp, 'isoformat'):
                                timestamp = timestamp.isoformat()
                            benchmark_ohlcv_list.append({
                                'timestamp': timestamp,
                                'open': float(ohlcv['open']),
                                'high': float(ohlcv['high']),
                                'low': float(ohlcv['low']),
                                'close': float(ohlcv['close']),
                                'volume': float(ohlcv['volume'])
                            })
                        if benchmark_ohlcv_list:
                            benchmark_ohlcv_data = benchmark_ohlcv_list
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error fetching benchmark data for beta calculation: {str(e)}")
        
        statistics = calculate_statistics(full_ohlcv_list, symbol=symbol, benchmark_ohlcv_data=benchmark_ohlcv_data)
        
        # Debug: Log statistics to see what's being calculated
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Statistics calculated for {symbol.ticker}: {statistics}")
        
        # Build response with appropriate pagination metadata
        if is_chart_view:
            # Chart view: no pagination (all data returned)
            return Response({
                'results': results,
                'count': total_count,
                'page': 1,
                'page_size': total_count,  # Return total count as page_size to indicate all data
                'next': None,
                'previous': None,
                'indicators': indicators_metadata,  # Metadata about enabled indicators
                'statistics': statistics if statistics else {}  # Statistics (volatility, etc.) - ensure it's always a dict
            })
        else:
            # Table view: include pagination metadata
            end_idx = start_idx + page_size
            return Response({
                'results': results,
                'count': total_count,
                'page': page,
                'page_size': page_size,
                'next': f'?page={page + 1}' if end_idx < total_count else None,
                'previous': f'?page={page - 1}' if page > 1 else None,
                'indicators': indicators_metadata,  # Metadata about enabled indicators
                'statistics': statistics if statistics else {}  # Statistics (volatility, etc.) - ensure it's always a dict
            })

    @action(detail=True, methods=['post'], url_path='update-data', url_name='update-data')
    def update_data(self, request, pk=None):
        """Trigger background task to update symbol OHLCV data (incremental update from latest timestamp)"""
        from .tasks import fetch_ohlcv_data_task
        from .services.ohlcv_service import OHLCVService
        from datetime import timedelta
        
        symbol = self.get_object()
        
        # Get the symbol's provider - required for update
        if not symbol.provider:
            return Response({
                'error': f'Symbol {symbol.ticker} does not have a data provider configured. Please fetch data first.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        provider_code = symbol.provider.code
        
        # Get the latest timestamp in the database
        latest_timestamp = OHLCVService.get_latest_timestamp(symbol, timeframe='daily')
        
        if latest_timestamp is None:
            return Response({
                'error': f'No OHLCV data found for {symbol.ticker}. Please use "Fetch Data" to fetch initial data first.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Normalize latest_timestamp to date if it's a datetime
        if isinstance(latest_timestamp, timezone.datetime):
            latest_date = latest_timestamp.date()
        else:
            latest_date = latest_timestamp
        
        # Get current time and today's date
        now = timezone.now()
        today = now.date()
        
        # Calculate days since latest data
        days_diff = (today - latest_date).days
        
        # If data is already up to date (latest is today or yesterday), check if update is needed
        # End date logic: after 10pm UTC use today, before 10pm UTC use yesterday
        current_hour_utc = now.hour
        if current_hour_utc >= 22:  # 10pm UTC (22:00)
            end_date = today
        else:
            end_date = today - timedelta(days=1)
        
        # If latest_date is already >= end_date, no update needed
        if latest_date >= end_date:
            return Response({
                'message': f'Data is already up to date. Latest data point: {latest_date}, End date: {end_date}',
                'symbol': symbol.ticker,
                'latest_timestamp': latest_date.isoformat(),
                'end_date': end_date.isoformat()
            }, status=status.HTTP_200_OK)
        
        # For incremental updates with Alpaca provider, fetch from 1 year before latest_date
        # This ensures we have a large enough historical range that works properly with Alpaca's API
        # The filtering logic in fetch_ohlcv_data_task will automatically skip dates we already have
        # Using 1 year back ensures proper chunking and avoids 403 errors with small recent date ranges
        start_date = latest_date - timedelta(days=365)  # 1 year back from latest_date
        
        # Ensure start_date is not before 2016-01-01 (Alpaca data starts from 2016)
        min_start_date = timezone.datetime(2016, 1, 1).date()
        if start_date < min_start_date:
            start_date = min_start_date
        
        # Convert to ISO format strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Trigger Celery task with symbol's provider and incremental update using date range
        task = fetch_ohlcv_data_task.delay(
            ticker=symbol.ticker,
            start_date=start_date_str,
            end_date=end_date_str,
            period=None,  # Use date range, not period
            replace_existing=False,  # Don't replace, only add new data
            broker_id=None,
            provider_code=provider_code
        )

        return Response({
            'message': f'OHLCV data update task started (from {start_date_str} to {end_date_str}) using provider {provider_code}',
            'task_id': task.id,
            'symbol': symbol.ticker,
            'provider': provider_code,
            'start_date': start_date_str,
            'end_date': end_date_str
        }, status=status.HTTP_202_ACCEPTED)
    
    @action(detail=True, methods=['post'], url_path='refetch-data', url_name='refetch-data')
    def refetch_data(self, request, pk=None):
        """Refetch all OHLCV data for a symbol (deletes existing and fetches all again)"""
        from .tasks import fetch_ohlcv_data_task, delete_ohlcv_data_task
        from django.db import transaction
        
        symbol = self.get_object()
        
        # Get provider code from request (optional - defaults to symbol's current provider)
        provider_code = request.data.get('provider_code')
        
        if provider_code:
            # Validate that the provider exists
            from .models import Provider
            try:
                provider = Provider.objects.get(code=provider_code, is_active=True)
            except Provider.DoesNotExist:
                return Response({
                    'error': f'Provider with code {provider_code} not found or is not active.'
                }, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Use symbol's current provider if no provider_code provided
            if not symbol.provider:
                return Response({
                    'error': f'Symbol {symbol.ticker} does not have a data provider configured. Please specify a provider_code or fetch data first.'
                }, status=status.HTTP_400_BAD_REQUEST)
            provider_code = symbol.provider.code
        
        # Get date parameters from request (optional - defaults to max period)
        start_date = request.data.get('start_date')
        end_date = request.data.get('end_date')
        period = request.data.get('period')
        
        # If dates are provided, don't use period (period should only be used if dates are not provided)
        # If neither dates nor period are provided, default to 'max'
        if not start_date and not end_date and not period:
            period = 'max'

        # Delete all existing OHLCV data for this symbol first
        # We do this synchronously to ensure it completes before refetch
        from .models import OHLCV, Provider
        deleted_count, _ = OHLCV.objects.filter(symbol=symbol).delete()
        
        # Update provider if it's different from current provider
        update_fields = ['status', 'validation_status', 'validation_reason']
        current_provider_code = symbol.provider.code if symbol.provider else None
        if provider_code != current_provider_code:
            symbol.provider = Provider.objects.get(code=provider_code, is_active=True)
            update_fields.append('provider')
        
        # Disable the symbol (will be re-enabled after successful refetch and validation)
        symbol.status = 'disabled'
        symbol.validation_status = 'invalid'
        symbol.validation_reason = 'OHLCV data deleted for refetch'
        symbol.save(update_fields=update_fields)

        # Trigger Celery task to fetch all data again with the symbol's provider
        task = fetch_ohlcv_data_task.delay(
            ticker=symbol.ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
            replace_existing=False,  # Not needed since we already deleted
            broker_id=None,
            provider_code=provider_code
        )

        return Response({
            'message': f'OHLCV data refetch task started (deleted {deleted_count} records, will fetch all data again) using provider {provider_code}',
            'task_id': task.id,
            'symbol': symbol.ticker,
            'provider': provider_code,
            'deleted_count': deleted_count
        }, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=['get'])
    def available_exchanges(self, request):
        """Get list of available exchanges from EOD API"""
        try:
            exchanges = EODAPIProvider.get_exchanges_list()
            if not exchanges:
                return Response({
                    'error': 'Failed to fetch exchanges from EOD API. Please check API connection.',
                    'results': [],
                    'count': 0
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Log for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Returning {len(exchanges)} exchanges from EOD API")
            
            return Response({
                'results': exchanges,
                'count': len(exchanges)
            })
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error fetching exchanges: {str(e)}", exc_info=True)
            return Response({
                'error': f'Error fetching exchanges: {str(e)}',
                'results': [],
                'count': 0
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['post'])
    def fetch_symbols(self, request):
        """Trigger background task to fetch symbols from exchanges"""
        exchange_codes = request.data.get('exchange_codes', [])
        fetch_all = request.data.get('fetch_all', False)
        
        if fetch_all:
            task = fetch_symbols_from_all_exchanges_task.delay()
        elif len(exchange_codes) == 1:
            task = fetch_symbols_from_exchange_task.delay(exchange_codes[0])
        elif len(exchange_codes) > 1:
            task = fetch_symbols_from_multiple_exchanges_task.delay(exchange_codes)
        else:
            return Response({
                'error': 'Please provide exchange_codes or set fetch_all to true'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        return Response({
            'message': 'Symbol fetch task started',
            'task_id': task.id,
            'exchange_codes': exchange_codes if not fetch_all else 'all'
        }, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        """Delete all symbols and all related OHLCV data"""
        try:
            # Get count before deletion
            symbol_count = Symbol.objects.count()
            ohlcv_count = OHLCV.objects.count()
            
            # Delete all symbols (OHLCV will be deleted automatically due to CASCADE)
            deleted_symbols = Symbol.objects.all().delete()
            
            return Response({
                'message': f'Successfully deleted all symbols and related data',
                'deleted_symbols': deleted_symbols[0],
                'symbol_count': symbol_count,
                'ohlcv_count': ohlcv_count
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': f'Failed to delete all symbols: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'], url_path='fetch-ohlcv', url_name='fetch-ohlcv')
    def fetch_ohlcv(self, request):
        """Fetch OHLCV data for one or more symbols, or by exchange, or by broker"""
        from .tasks import (
            fetch_ohlcv_data_task,
            fetch_ohlcv_data_multiple_symbols_task,
            fetch_ohlcv_data_by_exchange_task,
            fetch_ohlcv_data_by_broker_task
        )
        
        ticker = request.data.get('ticker')
        tickers = request.data.get('tickers', [])
        exchange_code = request.data.get('exchange_code')
        broker_id = request.data.get('broker_id')
        start_date = request.data.get('start_date')
        end_date = request.data.get('end_date')
        period = request.data.get('period')
        replace_existing = request.data.get('replace_existing', False)
        provider_code = request.data.get('provider_code', 'YAHOO')  # Default to YAHOO for backward compatibility
        
        # If only broker_id is provided (without exchange_code, ticker, or tickers), fetch for all broker symbols
        if broker_id and not exchange_code and not ticker and (not tickers or len(tickers) == 0):
            task = fetch_ohlcv_data_by_broker_task.delay(
                broker_id=broker_id,
                start_date=start_date,
                end_date=end_date,
                period=period,
                replace_existing=replace_existing,
                provider_code=provider_code
            )
            return Response({
                'message': f'OHLCV data fetch task started for broker {broker_id}',
                'task_id': task.id,
                'broker_id': broker_id
            }, status=status.HTTP_202_ACCEPTED)
        
        if exchange_code:
            # Fetch by exchange
            task = fetch_ohlcv_data_by_exchange_task.delay(
                exchange_code=exchange_code,
                broker_id=broker_id,
                start_date=start_date,
                end_date=end_date,
                period=period,
                replace_existing=replace_existing,
                provider_code=provider_code
            )
            return Response({
                'message': f'OHLCV data fetch task started for exchange {exchange_code}' + (f' (broker {broker_id})' if broker_id else ''),
                'task_id': task.id,
                'exchange_code': exchange_code,
                'broker_id': broker_id
            }, status=status.HTTP_202_ACCEPTED)
        elif tickers and len(tickers) > 0:
            # Fetch for multiple symbols
            task = fetch_ohlcv_data_multiple_symbols_task.delay(
                tickers=tickers,
                broker_id=broker_id,
                start_date=start_date,
                end_date=end_date,
                period=period,
                replace_existing=replace_existing,
                provider_code=provider_code
            )
            return Response({
                'message': f'OHLCV data fetch task started for {len(tickers)} symbols' + (f' (broker {broker_id})' if broker_id else ''),
                'task_id': task.id,
                'tickers': tickers,
                'broker_id': broker_id
            }, status=status.HTTP_202_ACCEPTED)
        elif ticker:
            # Fetch for single symbol
            task = fetch_ohlcv_data_task.delay(
                ticker=ticker,
                broker_id=broker_id,
                start_date=start_date,
                end_date=end_date,
                period=period,
                replace_existing=replace_existing,
                provider_code=provider_code
            )
            return Response({
                'message': f'OHLCV data fetch task started for {ticker}' + (f' (broker {broker_id})' if broker_id else ''),
                'task_id': task.id,
                'ticker': ticker,
                'broker_id': broker_id
            }, status=status.HTTP_202_ACCEPTED)
        else:
            return Response({
                'error': 'Must provide either ticker, tickers (list), or exchange_code'
            }, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'], url_path='delete-ohlcv', url_name='delete-ohlcv')
    def delete_ohlcv(self, request):
        """Delete OHLCV data for one or more symbols, by exchange, or all"""
        from .tasks import (
            delete_ohlcv_data_task,
            delete_ohlcv_data_multiple_symbols_task,
            delete_ohlcv_data_by_exchange_task,
            delete_all_ohlcv_data_task
        )
        
        delete_all = request.data.get('delete_all', False)
        ticker = request.data.get('ticker')
        tickers = request.data.get('tickers', [])
        exchange_code = request.data.get('exchange_code')
        
        if delete_all:
            # Delete all OHLCV data
            task = delete_all_ohlcv_data_task.delay()
            return Response({
                'message': 'OHLCV data deletion task started for all symbols',
                'task_id': task.id,
                'delete_all': True
            }, status=status.HTTP_202_ACCEPTED)
        elif exchange_code:
            # Delete by exchange
            task = delete_ohlcv_data_by_exchange_task.delay(exchange_code=exchange_code)
            return Response({
                'message': f'OHLCV data deletion task started for exchange {exchange_code}',
                'task_id': task.id,
                'exchange_code': exchange_code
            }, status=status.HTTP_202_ACCEPTED)
        elif tickers and len(tickers) > 0:
            # Delete for multiple symbols
            task = delete_ohlcv_data_multiple_symbols_task.delay(tickers=tickers)
            return Response({
                'message': f'OHLCV data deletion task started for {len(tickers)} symbols',
                'task_id': task.id,
                'tickers': tickers
            }, status=status.HTTP_202_ACCEPTED)
        elif ticker:
            # Delete for single symbol
            task = delete_ohlcv_data_task.delay(ticker=ticker)
            return Response({
                'message': f'OHLCV data deletion task started for {ticker}',
                'task_id': task.id,
                'ticker': ticker
            }, status=status.HTTP_202_ACCEPTED)
        else:
            return Response({
                'error': 'Must provide either delete_all=true, ticker, tickers (list), or exchange_code'
            }, status=status.HTTP_400_BAD_REQUEST)


class CrontabScheduleViewSet(viewsets.ModelViewSet):
    """ViewSet for CrontabSchedule"""
    queryset = CrontabSchedule.objects.all()
    serializer_class = CrontabScheduleSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['timezone']
    ordering = ['timezone']


class IntervalScheduleViewSet(viewsets.ModelViewSet):
    """ViewSet for IntervalSchedule"""
    queryset = IntervalSchedule.objects.all()
    serializer_class = IntervalScheduleSerializer
    filter_backends = [OrderingFilter]
    ordering_fields = ['period', 'every']
    ordering = ['period', 'every']


class PeriodicTaskViewSet(viewsets.ModelViewSet):
    """ViewSet for PeriodicTask"""
    queryset = PeriodicTask.objects.all()
    serializer_class = PeriodicTaskSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'task', 'description']
    ordering_fields = ['name', 'enabled', 'last_run_at', 'date_changed']
    ordering = ['-date_changed']

    @action(detail=True, methods=['post'])
    def enable(self, request, pk=None):
        """Enable a periodic task"""
        task = self.get_object()
        task.enabled = True
        task.save()
        return Response({'message': 'Task enabled', 'enabled': True})

    @action(detail=True, methods=['post'])
    def disable(self, request, pk=None):
        """Disable a periodic task"""
        task = self.get_object()
        task.enabled = False
        task.save()
        return Response({'message': 'Task disabled', 'enabled': False})

    @action(detail=False, methods=['post'], url_path='create-fetch-symbols-task', url_name='create-fetch-symbols-task')
    def create_fetch_symbols_task(self, request):
        """Create a scheduled task for fetching symbols"""
        task_name = request.data.get('name', 'Fetch Symbols')
        exchange_codes = request.data.get('exchange_codes', [])
        fetch_all = request.data.get('fetch_all', False)
        schedule_type = request.data.get('schedule_type', 'interval')  # 'interval' or 'crontab'
        schedule_data = request.data.get('schedule', {})
        
        # Determine task and kwargs
        if fetch_all:
            task_path = 'market_data.tasks.fetch_symbols_from_all_exchanges_task'
            kwargs = {}
        elif len(exchange_codes) == 1:
            task_path = 'market_data.tasks.fetch_symbols_from_exchange_task'
            kwargs = {'exchange_code': exchange_codes[0]}
        else:
            task_path = 'market_data.tasks.fetch_symbols_from_multiple_exchanges_task'
            kwargs = {'exchange_codes': exchange_codes}
        
        # Create schedule
        if schedule_type == 'interval':
            interval, created = IntervalSchedule.objects.get_or_create(
                every=schedule_data.get('every', 1),
                period=schedule_data.get('period', 'days')
            )
            periodic_task = PeriodicTask.objects.create(
                name=task_name,
                task=task_path,
                interval=interval,
                enabled=True,
                kwargs=json.dumps(kwargs)
            )
        else:  # crontab
            crontab, created = CrontabSchedule.objects.get_or_create(
                minute=schedule_data.get('minute', '*'),
                hour=schedule_data.get('hour', '*'),
                day_of_week=schedule_data.get('day_of_week', '*'),
                day_of_month=schedule_data.get('day_of_month', '*'),
                month_of_year=schedule_data.get('month_of_year', '*'),
                timezone=schedule_data.get('timezone', 'UTC')
            )
            periodic_task = PeriodicTask.objects.create(
                name=task_name,
                task=task_path,
                crontab=crontab,
                enabled=True,
                kwargs=json.dumps(kwargs)
            )
        
        serializer = PeriodicTaskSerializer(periodic_task)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class TaskExecutionViewSet(viewsets.ViewSet):
    """ViewSet for managing active tasks and task history"""
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get all currently active/running tasks"""
        try:
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return Response({
                    'results': [],
                    'count': 0
                })
            
            # Flatten the dictionary of worker -> tasks
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task_id = task.get('id')
                    task_result = AsyncResult(task_id, app=celery_app)
                    
                    # Get task info from state metadata
                    # Celery stores progress in backend metadata when update_state is called
                    progress = 0
                    message = 'Running...'
                    
                    try:
                        # First try to get from backend metadata (most reliable)
                        backend = task_result.backend
                        if backend:
                            meta = backend.get_task_meta(task_result.id)
                            if meta:
                                # Metadata is stored in meta['result'] when update_state is called
                                result = meta.get('result')
                                if isinstance(result, dict):
                                    progress = result.get('progress', 0)
                                    message = result.get('message', 'Running...')
                                # Also check meta['meta'] for some backends
                                elif meta.get('meta') and isinstance(meta['meta'], dict):
                                    progress = meta['meta'].get('progress', 0)
                                    message = meta['meta'].get('message', 'Running...')
                        
                        # Fallback to info property
                        if progress == 0 and message == 'Running...':
                            info = task_result.info
                            if isinstance(info, dict):
                                progress = info.get('progress', 0)
                                message = info.get('message', 'Running...')
                            elif info:
                                message = str(info)
                    except Exception:
                        # If all else fails, use defaults
                        pass
                    
                    # Convert time_start from Unix timestamp to ISO format
                    time_start = task.get('time_start')
                    if time_start:
                        try:
                            from datetime import datetime
                            # time_start is Unix timestamp (seconds since epoch)
                            time_start_dt = datetime.fromtimestamp(time_start)
                            time_start = time_start_dt.isoformat()
                        except (ValueError, TypeError, OSError):
                            # If conversion fails, keep original value
                            pass
                    
                    all_tasks.append({
                        'task_id': task_id,
                        'name': task.get('name', 'Unknown'),
                        'worker': worker,
                        'args': task.get('args', []),
                        'kwargs': task.get('kwargs', {}),
                        'time_start': time_start,
                        'progress': progress,
                        'message': message,
                        'status': 'RUNNING'
                    })
            
            return Response({
                'results': all_tasks,
                'count': len(all_tasks)
            })
        except Exception as e:
            return Response({
                'error': f'Failed to get active tasks: {str(e)}',
                'results': [],
                'count': 0
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def stop(self, request, pk=None):
        """Stop/revoke a running task"""
        task_id = pk
        try:
            # Revoke the task
            celery_app.control.revoke(task_id, terminate=True)
            
            return Response({
                'message': f'Task {task_id} has been stopped',
                'task_id': task_id
            })
        except Exception as e:
            return Response({
                'error': f'Failed to stop task: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        """Get task execution history (completed and failed tasks)"""
        try:
            # Get limit from query params
            limit = int(request.query_params.get('limit', 50))
            
            # Parse Redis URL from settings
            result_backend = settings.CELERY_RESULT_BACKEND
            if result_backend.startswith('redis://'):
                # Parse redis://localhost:6379/0
                parts = result_backend.replace('redis://', '').split('/')
                host_port = parts[0].split(':')
                host = host_port[0] if len(host_port) > 0 else 'localhost'
                port = int(host_port[1]) if len(host_port) > 1 else 6379
                db = int(parts[1]) if len(parts) > 1 else 0
            else:
                host = 'localhost'
                port = 6379
                db = 0
            
            # Connect to Redis to get task results
            try:
                redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                redis_client.ping()  # Test connection
            except Exception as e:
                return Response({
                    'error': f'Failed to connect to Redis: {str(e)}',
                    'results': [],
                    'count': 0
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Get all task keys from Redis
            try:
                task_keys = redis_client.keys('celery-task-meta-*')
            except Exception as e:
                return Response({
                    'error': f'Failed to query Redis: {str(e)}',
                    'results': [],
                    'count': 0
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            history = []
            for key in task_keys[:limit * 2]:  # Get more to filter
                task_id = key.replace('celery-task-meta-', '')
                try:
                    task_result = AsyncResult(task_id, app=celery_app)
                    
                    if task_result.ready():
                        # Get task result data
                        result_data = task_result.result
                        if isinstance(result_data, dict):
                            progress = result_data.get('progress', 100 if task_result.successful() else 0)
                            message = result_data.get('message', '')
                            status = result_data.get('status', 'completed' if task_result.successful() else 'failed')
                        else:
                            progress = 100 if task_result.successful() else 0
                            message = str(result_data) if result_data else ('Completed' if task_result.successful() else 'Failed')
                            status = 'completed' if task_result.successful() else 'failed'
                        
                        # Get task name from result backend or infer from task_id
                        task_name = 'Unknown Task'
                        if hasattr(task_result, 'name') and task_result.name:
                            task_name = task_result.name
                        
                        # Get timestamp from Redis TTL (approximate)
                        ttl = redis_client.ttl(key)
                        timestamp = None
                        if ttl > 0:
                            # Approximate timestamp (Redis doesn't store creation time, so we estimate)
                            # Tasks are typically stored for 24 hours (86400 seconds)
                            # Estimate: current_time - (max_ttl - current_ttl)
                            max_ttl = 86400  # 24 hours default
                            estimated_age = max_ttl - ttl
                            timestamp = timezone.now() - timedelta(seconds=estimated_age)
                        else:
                            # If TTL is -1 (no expiry) or -2 (key doesn't exist), use current time
                            timestamp = timezone.now()
                        
                        history.append({
                            'task_id': task_id,
                            'name': task_name,
                            'status': status,
                            'progress': progress,
                            'message': message,
                            'timestamp': timestamp.isoformat() if timestamp else None,
                            'success': task_result.successful()
                        })
                except Exception as e:
                    # Skip tasks that can't be loaded
                    continue
                
                if len(history) >= limit:
                    break
            
            # Sort by timestamp descending (most recent first)
            history.sort(key=lambda x: x['timestamp'] or '', reverse=True)
            
            return Response({
                'results': history[:limit],
                'count': len(history[:limit])
            })
        except Exception as e:
            return Response({
                'error': f'Failed to get task history: {str(e)}',
                'results': [],
                'count': 0
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """Get status of a specific task"""
        task_id = pk
        try:
            task_result = AsyncResult(task_id, app=celery_app)
            
            if task_result.ready():
                if task_result.successful():
                    result = task_result.result
                    if isinstance(result, dict):
                        # Normalize status: 'success' -> 'completed'
                        task_status = result.get('status', 'completed')
                        if task_status == 'success':
                            task_status = 'completed'
                        return Response({
                            'task_id': task_id,
                            'status': task_status,
                            'progress': result.get('progress', 100),
                            'message': result.get('message', 'Task completed'),
                            'ready': True,
                            'success': True,
                            'result': result  # Include full result for onComplete callback
                        })
                    else:
                        return Response({
                            'task_id': task_id,
                            'status': 'completed',
                            'progress': 100,
                            'message': 'Task completed successfully',
                            'ready': True,
                            'success': True
                        })
                else:
                    error_info = task_result.info
                    return Response({
                        'task_id': task_id,
                        'status': 'failed',
                        'progress': 0,
                        'message': str(error_info) if error_info else 'Task failed',
                        'ready': True,
                        'success': False
                    })
            else:
                # Task still running
                info = task_result.info
                if isinstance(info, dict):
                    return Response({
                        'task_id': task_id,
                        'status': 'running',
                        'progress': info.get('progress', 0),
                        'message': info.get('message', 'Running...'),
                        'ready': False,
                        'success': None
                    })
                else:
                    return Response({
                        'task_id': task_id,
                        'status': 'pending',
                        'progress': 0,
                        'message': 'Task is pending',
                        'ready': False,
                        'success': None
                    })
        except Exception as e:
            return Response({
                'error': f'Failed to get task status: {str(e)}',
                'task_id': task_id
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
