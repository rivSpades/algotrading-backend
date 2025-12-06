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
    BacktestSerializer, BacktestCreateSerializer,
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
    queryset = Backtest.objects.all().prefetch_related('symbols', 'strategy', 'trades', 'statistics')
    serializer_class = BacktestSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'strategy__name']
    ordering_fields = ['created_at', 'updated_at', 'completed_at']
    ordering = ['-created_at']
    pagination_class = BacktestPagination
    
    def get_queryset(self):
        """Filter backtests by strategy if strategy parameter is provided"""
        queryset = super().get_queryset()
        strategy_id = self.request.query_params.get('strategy', None)
        if strategy_id:
            queryset = queryset.filter(strategy_id=strategy_id)
        return queryset
    
    def get_serializer_class(self):
        if self.action == 'create':
            return BacktestCreateSerializer
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
        
        # Get symbols
        symbol_tickers = data['symbol_tickers']
        symbols = []
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
        
        # Handle date range - if not provided, will be determined from actual data
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # If dates not provided, set to None - will use actual data range
        # The executor will update these with actual data dates
        if not start_date:
            start_date = None
        if not end_date:
            from django.utils import timezone
            end_date = timezone.now()  # Current date for end_date, will be updated by executor
        
        # Create backtest
        backtest = Backtest.objects.create(
            name=data.get('name', ''),
            strategy=strategy,
            strategy_assignment=strategy_assignment,
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
        """Get all trades for a backtest (with pagination)"""
        backtest = self.get_object()
        trades = backtest.trades.all().order_by('entry_timestamp')
        
        # Check if pagination should be disabled
        no_pagination = request.query_params.get('no_pagination', 'false').lower() == 'true'
        
        if no_pagination:
            # Return all trades without pagination
            serializer = TradeSerializer(trades, many=True)
            return Response(serializer.data)
        
        # Apply pagination
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
    
    @action(detail=True, methods=['get'], url_path='symbol/(?P<ticker>[^/.]+)')
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
