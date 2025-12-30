"""
API Views for Backtest Engine
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.pagination import PageNumberPagination
from django.shortcuts import get_object_or_404
from django.db.models import Q
from .models import Backtest, Trade, BacktestStatistics
from .serializers import (
    BacktestSerializer, BacktestListSerializer, BacktestDetailSerializer, BacktestCreateSerializer,
    TradeSerializer, BacktestStatisticsSerializer
)
from strategies.models import StrategyDefinition, StrategyAssignment
from market_data.models import Symbol
from .tasks import run_backtest_task
import logging

logger = logging.getLogger(__name__)


class TradePagination(PageNumberPagination):
    """Pagination for trades"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BacktestPagination(PageNumberPagination):
    """Pagination for backtests"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BacktestViewSet(viewsets.ModelViewSet):
    """ViewSet for Backtest"""
    queryset = Backtest.objects.all()
    serializer_class = BacktestSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'strategy__name']
    ordering_fields = ['created_at', 'updated_at', 'completed_at']
    ordering = ['-created_at']
    pagination_class = BacktestPagination
    
    def get_queryset(self):
        """Optimize queryset based on action - list vs detail"""
        queryset = Backtest.objects.all()
        
        if self.action == 'list':
            # For list views, only prefetch strategy (needed for strategy.name)
            # Don't prefetch symbols, trades, statistics to improve performance
            queryset = queryset.select_related('strategy', 'strategy_assignment')
        elif self.action == 'retrieve':
            # For detail views (retrieve), only prefetch strategy - trades and statistics are fetched separately
            # This dramatically improves performance for backtests with thousands of symbols
            queryset = queryset.select_related('strategy', 'strategy_assignment')
        else:
            # For other actions (update, partial_update, etc.), may need more data
            # But still avoid prefetching trades/statistics unless absolutely necessary
            queryset = queryset.select_related('strategy', 'strategy_assignment')
        
        # Filter backtests by strategy if strategy parameter is provided
        strategy_id = self.request.query_params.get('strategy', None)
        if strategy_id:
            queryset = queryset.filter(strategy_id=strategy_id)
        return queryset
    
    def get_serializer_class(self):
        if self.action == 'create':
            return BacktestCreateSerializer
        elif self.action == 'list':
            return BacktestListSerializer
        elif self.action == 'retrieve':
            # Use lightweight serializer for detail views - excludes trades and statistics
            # Trades: fetched via /backtests/{id}/trades/ endpoint
            # Statistics: fetched via /backtests/{id}/statistics/optimized/ endpoint
            # Symbols: fetched via /backtests/{id}/symbols/ endpoint
            return BacktestDetailSerializer
        return BacktestSerializer
    
    def create(self, request, *args, **kwargs):
        """Create and start a new backtest"""
        serializer = BacktestCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        
        # Get strategy
        try:
            strategy = StrategyDefinition.objects.get(id=data['strategy_id'])
        except StrategyDefinition.DoesNotExist:
            return Response(
                {'error': f'Strategy with id {data["strategy_id"]} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get symbols - either from provided tickers or broker-based filtering
        symbols = []
        symbol_tickers = data.get('symbol_tickers', [])
        broker_id = data.get('broker_id')
        exchange_code = data.get('exchange_code', '')
        
        if broker_id:
            # Broker-based filtering - get all symbols with at least one active flag
            # The backtest executor will filter by position mode automatically
            from live_trading.models import Broker, SymbolBrokerAssociation
            from market_data.models import Exchange
            
            try:
                broker = Broker.objects.get(id=broker_id)
            except Broker.DoesNotExist:
                return Response(
                    {'error': f'Broker with id {broker_id} not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get broker associations - only symbols with at least one active flag
            # The executor will automatically filter symbols by position mode eligibility
            associations = SymbolBrokerAssociation.objects.filter(
                broker=broker
            ).filter(
                Q(long_active=True) | Q(short_active=True)
            )
            
            # Filter by exchange if provided
            if exchange_code:
                try:
                    exchange = Exchange.objects.get(code=exchange_code)
                    associations = associations.filter(symbol__exchange=exchange)
                except Exchange.DoesNotExist:
                    return Response(
                        {'error': f'Exchange with code {exchange_code} not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Get symbols from associations
            symbols = [assoc.symbol for assoc in associations.select_related('symbol')]
            
            # If specific symbol_tickers provided, further filter to only those
            if symbol_tickers and len(symbol_tickers) > 0:
                symbols = [s for s in symbols if s.ticker in symbol_tickers]
            
            if not symbols:
                exchange_text = f" on exchange {exchange_code}" if exchange_code else ""
                return Response(
                    {'error': f'No symbols found for broker {broker.name}{exchange_text}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            # Normal symbol selection from tickers
            if not symbol_tickers or len(symbol_tickers) == 0:
                return Response(
                    {'error': 'symbol_tickers is required when broker_id is not provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            for ticker in symbol_tickers:
                try:
                    symbol = Symbol.objects.get(ticker=ticker)
                    symbols.append(symbol)
                except Symbol.DoesNotExist:
                    return Response(
                        {'error': f'Symbol {ticker} not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
        
        # Get or create strategy assignment (for parameter merging)
        strategy_assignment = None
        if len(symbols) == 1:
            # Try to find symbol-specific assignment
            strategy_assignment = StrategyAssignment.objects.filter(
                strategy=strategy,
                symbol=symbols[0]
            ).first()
        
        if not strategy_assignment:
            # Try global assignment
            strategy_assignment = StrategyAssignment.objects.filter(
                strategy=strategy,
                symbol__isnull=True
            ).first()
        
        # Merge parameters
        strategy_parameters = strategy.default_parameters.copy()
        if strategy_assignment:
            strategy_parameters.update(strategy_assignment.parameters)
        strategy_parameters.update(data.get('strategy_parameters', {}))
        
        # Handle date range - backtests now use ALL available data for each symbol
        # start_date and end_date are kept for backward compatibility but are not used for filtering
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # If dates not provided, set to None
        # The executor will use ALL available data for each symbol, not filter by dates
        if not start_date:
            start_date = None
        if not end_date:
            from django.utils import timezone
            end_date = timezone.now()  # Set to current date for display purposes only
        
        # Broker is already loaded if broker_id was provided (for symbol filtering)
        # Just get it again here for the backtest model
        broker = None
        if broker_id:
            from live_trading.models import Broker
            try:
                broker = Broker.objects.get(id=broker_id)
            except Broker.DoesNotExist:
                pass  # Already handled in symbol filtering section above
        
        # Create backtest
        backtest = Backtest.objects.create(
            name=data.get('name', ''),
            strategy=strategy,
            strategy_assignment=strategy_assignment,
            broker=broker,
            start_date=start_date,
            end_date=end_date,
            split_ratio=data.get('split_ratio', 0.7),
            initial_capital=data.get('initial_capital', 10000.0),
            bet_size_percentage=data.get('bet_size_percentage', 100.0),
            strategy_parameters=strategy_parameters,
            status='pending'
        )
        backtest.symbols.set(symbols)
        
        # Start backtest task asynchronously
        try:
            task = run_backtest_task.delay(backtest.id)
            logger.info(f"Started backtest task for backtest {backtest.id}, task_id: {task.id}")
            
            # Return created backtest with task_id
            serializer = BacktestSerializer(backtest)
            response_data = serializer.data
            response_data['task_id'] = task.id  # Include task_id in response
            return Response(response_data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error starting backtest task: {str(e)}")
            backtest.status = 'failed'
            backtest.error_message = f"Error starting backtest task: {str(e)}"
            backtest.save()
            return Response(
                {'error': f'Error starting backtest: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def trades(self, request, pk=None):
        """Get trades for a backtest with pagination and filtering
        
        Query parameters:
        - symbol: Filter by symbol ticker
        - mode: Filter by position mode ('all', 'long', 'short')
          - 'long' filters for trade_type='buy'
          - 'short' filters for trade_type='sell'
          - 'all' returns all trades
        - no_pagination: Set to 'true' to disable pagination (not recommended for large datasets)
        """
        backtest = self.get_object()
        trades = backtest.trades.select_related('symbol').all()
        
        # Filter by symbol ticker if provided
        symbol_ticker = request.query_params.get('symbol', None)
        if symbol_ticker:
            trades = trades.filter(symbol__ticker=symbol_ticker)
        
        # Filter by mode (position mode) - use metadata.position_mode to ensure each mode has independent bankroll
        # Each mode (ALL, LONG, SHORT) is executed separately with its own capital, so we must filter by position_mode
        # Only filter if mode is explicitly provided in query params
        mode = request.query_params.get('mode', None)
        if mode:
            mode = mode.lower()
            if mode in ['long', 'short', 'all']:
                # Filter by metadata.position_mode using JSON field lookup
                # This ensures we only get trades from the specific mode execution (which has its own bankroll)
                trades = trades.filter(metadata__position_mode=mode)
        # If mode is not provided, return trades from all modes (useful for getAllBacktestTrades)
        
        # Order by entry timestamp
        trades = trades.order_by('entry_timestamp')
        
        # Check if pagination should be disabled
        no_pagination = request.query_params.get('no_pagination', 'false').lower() == 'true'
        
        if no_pagination:
            # Return all trades without pagination (not recommended for large datasets)
            serializer = TradeSerializer(trades, many=True)
            return Response(serializer.data)
        
        # Apply pagination (default behavior)
        paginator = TradePagination()
        page = paginator.paginate_queryset(trades, request)
        if page is not None:
            serializer = TradeSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        
        # Fallback to non-paginated response if pagination not requested
        serializer = TradeSerializer(trades, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get all statistics for a backtest"""
        backtest = self.get_object()
        stats = backtest.statistics.all()
        serializer = BacktestStatisticsSerializer(stats, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'], url_path='statistics/optimized')
    def statistics_optimized(self, request, pk=None):
        """Get optimized statistics for a backtest - organized by mode (ALL/LONG/SHORT)"""
        backtest = self.get_object()
        
        # Get portfolio stats (symbol=None)
        portfolio_stats = backtest.statistics.filter(symbol__isnull=True).first()
        
        result = {
            'portfolio': None,
            'symbols': []
        }
        
        if portfolio_stats:
            serializer = BacktestStatisticsSerializer(portfolio_stats)
            result['portfolio'] = {
                'stats_by_mode': serializer.data.get('stats_by_mode', {}),
                'equity_curve_x': serializer.data.get('equity_curve_x', []),
                'equity_curve_y': serializer.data.get('equity_curve_y', []),
            }
        
        # Get symbol-level stats
        symbol_stats = backtest.statistics.filter(symbol__isnull=False).select_related('symbol')
        for stat in symbol_stats:
            serializer = BacktestStatisticsSerializer(stat)
            result['symbols'].append({
                'symbol_ticker': serializer.data.get('symbol_ticker'),
                'stats_by_mode': serializer.data.get('stats_by_mode', {}),
                'equity_curve_x': serializer.data.get('equity_curve_x', []),
                'equity_curve_y': serializer.data.get('equity_curve_y', []),
            })
        
        return Response(result)
    
    @action(detail=True, methods=['get'], url_path='symbol-list')
    def symbols(self, request, pk=None):
        """Get paginated list of symbols associated with this backtest (with search support)"""
        from market_data.serializers import SymbolListSerializer
        
        # Get backtest - ensure we have the object
        # Use pk from kwargs if available, otherwise use pk parameter
        lookup_pk = self.kwargs.get('pk', pk)
        try:
            backtest = self.get_object()
        except Exception:
            # Try to get backtest directly if get_object() fails
            from .models import Backtest
            backtest = Backtest.objects.get(pk=lookup_pk)
        
        # Get unique symbols from the backtest (ManyToMany relationship)
        symbols_queryset = backtest.symbols.all().order_by('ticker')
        
        # Apply search filter if provided
        search = request.query_params.get('search', None)
        if search:
            # Trim whitespace and ensure search is not empty
            search = search.strip()
            if search:
                symbols_queryset = symbols_queryset.filter(ticker__icontains=search)
            else:
                # If search is only whitespace, treat as no search
                search = None
        
        # Apply pagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        paginator.page_size_query_param = 'page_size'
        paginator.max_page_size = 100
        
        page = paginator.paginate_queryset(symbols_queryset, request)
        if page is not None:
            # Serialize symbols to return full symbol objects
            serializer = SymbolListSerializer(page, many=True)
            # Get pagination metadata
            paginated_response = paginator.get_paginated_response(serializer.data)
            return paginated_response
        
        # Fallback if pagination not requested - return all as array
        serializer = SymbolListSerializer(symbols_queryset, many=True)
        return Response({'results': serializer.data, 'count': len(serializer.data), 'next': None, 'previous': None})
    
    @action(detail=True, methods=['get'], url_path=r'symbol/(?P<ticker>[^/]+)/$')
    def by_symbol(self, request, pk=None, ticker=None):
        """Get backtest statistics for a specific symbol"""
        backtest = self.get_object()
        try:
            symbol = Symbol.objects.get(ticker=ticker)
            stats = backtest.statistics.filter(symbol=symbol).first()
            if stats:
                serializer = BacktestStatisticsSerializer(stats)
                return Response(serializer.data)
            return Response(
                {'error': f'No statistics found for symbol {ticker} in this backtest'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Symbol.DoesNotExist:
            return Response(
                {'error': f'Symbol {ticker} not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class TradeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Trade (read-only)"""
    queryset = Trade.objects.all().select_related('backtest', 'symbol')
    serializer_class = TradeSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['entry_timestamp', 'pnl']
    ordering = ['entry_timestamp']


class BacktestStatisticsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for BacktestStatistics (read-only)"""
    queryset = BacktestStatistics.objects.all().select_related('backtest', 'symbol')
    serializer_class = BacktestStatisticsSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['created_at', 'total_pnl', 'win_rate']
    ordering = ['-created_at']
